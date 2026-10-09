# SPDX-License-Identifier: Apache-2.0
"""Pairwise output-consistency analysis for the combo-matrix sweep.

For each arm: md5 of req1.png, and vs its group baseline: PSNR / max abs diff /
byte-equality. Also re-greps cleaned activation evidence from the raw serve logs
(the in-sweep capture includes unrelated 'Route:' lines). Stdlib + PIL + numpy.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image

RUNS = Path("/workspace/boogu-matrix/runs")

# arm -> (group, baseline arm or None)
ARMS: dict[str, tuple[str, str | None]] = {
    "fp8_noffload": ("fp8", None),
    "fp8_model": ("fp8", "fp8_noffload"),
    "fp8_layerwise": ("fp8", "fp8_noffload"),
    "fp8_bothflags": ("fp8", "fp8_noffload"),
    "bf16_model": ("bf16", None),
    "bf16_layerwise": ("bf16", "bf16_model"),
    "dlo_n0": ("bf16", "bf16_model"),
    "dlo_n4": ("bf16", "bf16_model"),
    "dlo_n4_retry1": ("bf16", "bf16_model"),
    "dlo_n4_retry2": ("bf16", "bf16_model"),
}

ACTIVATION = re.compile(
    r"Enabling offloader backend|offloading enabled|offloading disabled|layerwise offload timing"
    r"|resident|Unpacked .*Float8Tensor|Online quantization|Stream-offloaded|offloaded model back to CPU"
    r"|DLO|resident layers",
    re.IGNORECASE,
)
NOISE = re.compile(r"launcher\.py:80|Route:")


def load_png(arm: str) -> np.ndarray | None:
    path = RUNS / f"{arm}.req1.png"
    if not path.exists() or path.stat().st_size == 0:
        return None
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else "-"


def psnr(a: np.ndarray, b: np.ndarray) -> tuple[str, str, bool]:
    if a.shape != b.shape:
        return "-", "-", False
    mse = float(((a - b) ** 2).mean())
    value = "inf (identical)" if mse == 0 else f"{10 * np.log10(1.0 / mse):.2f}"
    return value, f"{float(np.abs(a - b).max()):.4f}", bool(np.array_equal(a, b))


def main() -> None:
    images = {arm: load_png(arm) for arm in ARMS}

    print("## Output consistency (req1, seed 42)\n")
    print("| Arm | Group | md5 | vs baseline | PSNR dB | max abs diff | byte-identical |")
    print("| --- | --- | --- | --- | ---: | ---: | --- |")
    for arm, (group, baseline) in ARMS.items():
        image = images[arm]
        if image is None:
            print(f"| {arm} | {group} | - | - | - | - | no output |")
            continue
        png = RUNS / f"{arm}.req1.png"
        if baseline and images.get(baseline) is not None:
            value, maxabs, same = psnr(image, images[baseline])
            print(f"| {arm} | {group} | {md5(png)} | {baseline} | {value} | {maxabs} | {'yes' if same else 'no'} |")
        else:
            print(f"| {arm} | {group} | {md5(png)} | (baseline) | - | - | - |")

    print("\n## Activation evidence (cleaned from serve logs)\n")
    for arm in ARMS:
        log = RUNS / f"{arm}.serve.log"
        if not log.exists():
            print(f"### {arm}\n\n(no serve log)\n")
            continue
        lines = [
            line.strip()
            for line in log.read_text(errors="replace").splitlines()
            if ACTIVATION.search(line) and not NOISE.search(line)
        ]
        print(f"### {arm}\n")
        print("```text")
        print("\n".join(lines) if lines else "(no matching lines)")
        print("```\n")


if __name__ == "__main__":
    main()
