# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn a `run_e2e.sh` output directory into results.json and results.md.

Parses each ``bench_<size>_<config>.log`` (the ``vllm-omni bench serve
--print-stage`` output), the 0.5 s device samples in ``mem_*.txt`` and, when the
images and Pillow/numpy are available, the PSNR of every batch-1 config against
that size's baseline image.

    python collect.py --out ~/pro6000-vae-e2e
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

SECTION_RE = re.compile(r"^=+\s*Stage\s+(\d+)\s*\(([^)]*)\)\s*=+\s*$")
METRIC_RE = re.compile(r"^(Mean|Median|P\d+)\s+(.+?)(?:\s+\(ms\))?:\s+([\d.]+)\s*$")
CONFIGS = {  # name -> (slicing, tiling, concurrency)
    "baseline": (False, False, 1),
    "slicing": (True, False, 1),
    "tiling": (False, True, 1),
    "both": (True, True, 1),
    "slicing-b4": (True, False, 4),
    "both-b4": (True, True, 4),
    "baseline-b4": (False, False, 4),
    "tiling-b4": (False, True, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="The directory run_e2e.sh wrote to")
    return parser.parse_args()


def parse_bench_log(path: Path) -> tuple[dict[str, float], dict[int, dict[str, float]], str]:
    """Return ``(global metrics, {stage_id: metrics}, stage names)``.

    Stage sections come from ``--print-stage``; global lines are the ones printed
    outside any section (E2E latency, throughput, ...).  Keys are the printed
    labels with the statistic folded in and lowercased, e.g.
    ``mean stage_gen_time`` -- the printers are not consistent about case
    (``stage_gen_time`` for text stages, ``IMAGE_GENERATION`` for image ones).
    """
    globals_: dict[str, float] = {}
    stages: dict[int, dict[str, float]] = {}
    names: dict[int, str] = {}
    current: int | None = None
    for line in path.read_text(errors="replace").splitlines():
        section = SECTION_RE.match(line)
        if section:
            current = int(section.group(1))
            stages.setdefault(current, {})
            names[current] = section.group(2).strip()
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        key = f"{match.group(1).lower()} {match.group(2).strip().lower()}"
        value = float(match.group(3))
        if current is None:
            globals_[key] = value
        else:
            stages[current][key] = value
    return globals_, stages, names


def pick(metrics: dict[str, float], *needles: str) -> float | None:
    """Mean of the first metric matching a needle, median as the fallback.

    Keys look like ``"mean image_generation"``; needles are matched as
    substrings, in the order given.
    """
    lowered = [needle.lower() for needle in needles]
    for prefix in ("mean ", "median "):
        for needle in lowered:
            for key, value in metrics.items():
                if key.startswith(prefix) and needle in key:
                    return value
    for needle in lowered:
        for key, value in metrics.items():
            if needle in key:
                return value
    return None


def peak_mib(path: Path) -> int | None:
    values = [int(line) for line in path.read_text().split() if line.strip().isdigit()]
    return max(values) if values else None


def psnr_db(a: Path, b: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    first = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64) / 255.0
    second = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64) / 255.0
    if first.shape != second.shape:
        return None
    mse = float(np.mean((first - second) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    runs: dict[str, dict] = {}
    for log in sorted(out.glob("bench_*.log")):
        size_str, config = log.stem[len("bench_") :].split("_", 1)
        size = int(size_str)
        tag = f"{size}_{config}"
        globals_, stages, names = parse_bench_log(log)
        slicing, tiling, concurrency = CONFIGS.get(config, (None, None, None))
        runs[tag] = {
            "size": size,
            "config": config,
            "slicing": slicing,
            "tiling": tiling,
            "concurrency": concurrency,
            "peak_mib": peak_mib(out / f"mem_{tag}.txt"),
            "e2e_mean_ms": pick(globals_, "e2e latency", "image_generation", "stage_gen_time"),
            "stage_names": {str(k): v for k, v in names.items()},
            "stage_gen_mean_ms": {
                str(k): pick(m, "stage_gen_time", "image_generation", "video_generation") for k, m in stages.items()
            },
            "stage_metrics": {str(k): v for k, v in stages.items()},
            "global_metrics": globals_,
            "psnr_db": None,
        }
        if config != "baseline" and concurrency == 1:
            baseline = out / f"img_{size}_baseline.png"
            image = out / f"img_{tag}.png"
            if baseline.is_file() and image.is_file():
                runs[tag]["psnr_db"] = psnr_db(baseline, image)

    (out / "results.json").write_text(json.dumps(runs, indent=2, sort_keys=True))

    stage_ids = sorted({int(k) for run in runs.values() for k in run["stage_gen_mean_ms"]})
    header = ["Size", "Config", "Slice", "Tile", "Conc", "Peak MiB", "E2E mean ms"]
    header += [f"Stage {i} gen ms" for i in stage_ids]
    header += ["PSNR dB"]
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    for run in sorted(runs.values(), key=lambda r: (r["size"], r["concurrency"] or 0, r["config"])):
        row = [
            f"{run['size']}x{run['size']}",
            run["config"],
            "yes" if run["slicing"] else "no",
            "yes" if run["tiling"] else "no",
            str(run["concurrency"]),
            f"{run['peak_mib']:.0f}" if run["peak_mib"] else "-",
            f"{run['e2e_mean_ms']:.1f}" if run["e2e_mean_ms"] else "-",
        ]
        for stage in stage_ids:
            value = run["stage_gen_mean_ms"].get(str(stage))
            row.append(f"{value:.1f}" if value else "-")
        psnr = run["psnr_db"]
        row.append("identical" if psnr == math.inf else (f"{psnr:.2f}" if psnr is not None else "-"))
        lines.append(f"| {' | '.join(row)} |")

    table = "\n".join(lines)
    (out / "results.md").write_text(table + "\n")
    print(table)
    print(f"\nwrote {out / 'results.json'} and {out / 'results.md'}")


if __name__ == "__main__":
    main()
