# SPDX-License-Identifier: Apache-2.0
"""Collect the combination-matrix sweep artifacts into one markdown record.

Reads /workspace/boogu-matrix/runs/<arm>.{peak_mib,load_peak_mib,reqN.ms,reqN.png,serve.log}
and prints the record table used for the #6665 reply. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

RUNS = Path("/workspace/boogu-matrix/runs")

ARMS = [
    ("bf16_noffload", "bf16 (34.6 GiB)", "none", "expected rejection"),
    ("bf16_model", "bf16 (34.6 GiB)", "module-level", "A only"),
    ("bf16_layerwise", "bf16 (34.6 GiB)", "local layerwise", "A only"),
    ("dlo_n0", "bf16 (34.6 GiB)", "distributed layerwise N=0", "A only"),
    ("dlo_n4", "bf16 (34.6 GiB)", "distributed layerwise N=4", "A only"),
    ("dlo_n4_retry1", "bf16 (34.6 GiB)", "distributed layerwise N=4 (retry 1)", "A only"),
    ("dlo_n4_retry2", "bf16 (34.6 GiB)", "distributed layerwise N=4 (retry 2)", "A only"),
    ("fp8_noffload", "Base-fp8", "none", "B only"),
    ("fp8_model", "Base-fp8", "module-level", "A+B"),
    ("fp8_layerwise", "Base-fp8", "local layerwise", "A+B"),
    ("fp8_bothflags", "Base-fp8", "module+layerwise both flags", "mode-boundary check"),
]


def sha16(path: Path) -> str:
    if not path.exists():
        return "-"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_int(path: Path) -> str:
    if not path.exists():
        return "-"
    value = path.read_text().strip()
    return value or "-"


def request_ms(arm: str) -> list[int]:
    values = []
    for index in (1, 2, 3):
        path = RUNS / f"{arm}.req{index}.ms"
        if path.exists():
            try:
                values.append(int(path.read_text().strip()))
            except ValueError:
                pass
    return values


def main() -> None:
    lines = [
        "| Arm | Checkpoint | Offload mode | Role | Peak MiB (steady) | Peak MiB (load) | E2E s min-max (n=3) | PNG sha256 | Status |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    details: list[str] = []
    for arm, checkpoint, mode, role in ARMS:
        oom = (RUNS / f"{arm}.OOM").exists()
        segv = (RUNS / f"{arm}.SEGV").exists()
        log = RUNS / f"{arm}.serve.log"
        if log.exists() and "Segfault encountered" in log.read_text(errors="replace"):
            segv = True
        ms = request_ms(arm)
        if oom:
            status = "OOM (expected rejection)"
        elif segv:
            status = "failed (startup segfault)"
        elif ms:
            status = "ok"
        else:
            status = "no measured requests"
        latency = f"{min(ms) / 1000:.2f}-{max(ms) / 1000:.2f}" if ms else "-"
        lines.append(
            f"| {arm} | {checkpoint} | {mode} | {role} | {read_int(RUNS / f'{arm}.peak_mib')} "
            f"| {read_int(RUNS / f'{arm}.load_peak_mib')} | {latency} | {sha16(RUNS / f'{arm}.req1.png')} | {status} |"
        )

        body: dict[str, object] = {"arm": arm}
        ms_json = RUNS / f"{arm}.req1.json"
        if ms_json.exists():
            try:
                payload = json.loads(ms_json.read_text())
                body["response_keys"] = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
                data = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    body["data0_keys"] = sorted(data[0].keys())
            except json.JSONDecodeError:
                body["response_keys"] = "not json"
        if ms:
            body["req_ms"] = ms
            body["req_median_ms"] = statistics.median(ms)
        details.append(f"### {arm}\n\n```json\n{json.dumps(body, indent=2)}\n```")
        for index in (0, 1, 2, 3):
            ms_path = RUNS / f"{arm}.req{index}.ms"
            if ms_path.exists():
                details.append(f"- req{index}: {ms_path.read_text().strip()} ms, png {sha16(RUNS / f'{arm}.req{index}.png')}")
        activation = RUNS / f"{arm}.activation.txt"
        if activation.exists() and activation.read_text().strip():
            details.append("\nActivation evidence:\n\n```text\n" + activation.read_text().strip() + "\n```")
        details.append("")

    print("\n".join(lines))
    print()
    print("\n".join(details))


if __name__ == "__main__":
    main()
