# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-cell sample count and spread for the end-to-end tables (PR #7774 review).

The 2026-10-06 review asked the end-to-end tables for the sample count and the
per-cell min-max, "as in the decode tables".  No new measurement is needed:
``run_e2e_offline.py`` already kept every measured sample in
``<out>/raw/<size>_<config>.json``.

One record is one measured request, and ``wall_s`` is the wall time of the wave
the request belongs to divided by the wave size, so the members of one wave
share a single value.  A run of consecutive equal values is therefore one
sample: one request at batch 1, one wave above it -- the mean of exactly these
samples is the published "E2E mean ms" column.

    python benchmarks/diffusion/mammoth_moda2_vae_e2e/spread.py \
        --out ~/pro6000-vae-e2e --out ~/pro6000-vae-e2e-b4

Prints, per ``--out``, one markdown table: the sample count, the end-to-end
latency as mean (min-max), and the stage-generation means over the measured
records (the convention ``collect.py`` and the published tables use).  Cells
whose records mix batch sizes are flagged: they are not whole waves and their
mean is not per-request (the superseded five-prompt ``slicing-b4`` rows).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        action="append",
        required=True,
        metavar="DIR",
        help="a run_e2e_offline.py output directory (repeatable)",
    )
    return parser.parse_args()


def measured_records(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("phase") == "measured"]


def samples_ms(records: list[dict]) -> list[float]:
    """One end-to-end latency per measured request or wave, in issue order."""
    samples: list[float] = []
    previous: float | None = None
    for record in records:
        wall_s = record["wall_s"]
        if previous is None or wall_s != previous:
            samples.append(wall_s * 1000.0)
        previous = wall_s
    return samples


def stage_mean_ms(records: list[dict], stage: int) -> float | None:
    key = f"stage_{stage}_gen_ms"
    values = [record["stage_durations"][key] for record in records if key in record.get("stage_durations", {})]
    return statistics.fmean(values) if values else None


def fmt_ms(value: float | None) -> str:
    return f"{value:,.1f}" if value is not None else "-"


def main() -> None:
    args = parse_args()
    for out in (Path(raw).expanduser() for raw in args.out):
        print(f"# {out}")
        raw_dir = out / "raw"
        if not raw_dir.is_dir():
            print(f"no {raw_dir} -- nothing to read (wrong --out?)\n")
            continue

        rows = []
        for path in sorted(raw_dir.glob("*.json")):
            data = json.loads(path.read_text())
            if "records" not in data:
                print(f"note: {path.name} has no 'records' key -- not a run_e2e_offline.py tree, skipped")
                continue
            records = measured_records(data["records"])
            if not records:
                continue
            # First-seen order, so a mixed cell prints as it was issued (4+1).
            batches = list(dict.fromkeys(record["batch"] for record in records))
            rows.append((data["size"], data["concurrency"], data["config"], batches, samples_ms(records), records))
        rows.sort(key=lambda row: (row[0], row[1], row[2]))

        header = ["Cell", "Conc", "Batch", "n", "E2E ms/image mean (min-max)", "Stage 0 gen ms", "Stage 1 gen ms"]
        lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
        mixed = []
        for size, concurrency, config, batches, samples, records in rows:
            batch_label = "+".join(str(batch) for batch in batches)
            spread = f"{statistics.fmean(samples):,.1f} ({min(samples):,.1f}-{max(samples):,.1f})"
            lines.append(
                f"| {size}_{config} | {concurrency} | {batch_label} | {len(samples)} | {spread} "
                f"| {fmt_ms(stage_mean_ms(records, 0))} | {fmt_ms(stage_mean_ms(records, 1))} |"
            )
            if len(batches) > 1:
                mixed.append(f"{size}_{config} (batches {batch_label})")
        print("\n".join(lines))
        if mixed:
            print()
            print("Mixed-batch cells -- not whole waves, the mean is not per-request (superseded; ignore):")
            for cell in mixed:
                print(f"- {cell}")
        print()

    print("Paste the tables back: the min-max goes into the recipe's end-to-end tables as e.g. 217.8 (217.7-217.9).")


if __name__ == "__main__":
    main()
