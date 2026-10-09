# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Collapse gate for diffusion edits: detect outputs the model re-rendered instead of editing.

Reusing the seed that generated the source image as the edit seed deterministically
collapses Qwen-Image-2.1 edits at low resolution: the sampler re-renders the source and
drops the instruction, producing dense, over-saturated high-frequency texture (see
README.md). The signal sits in the background, so this gate measures the mean Laplacian
standard deviation over the four corner patches: normal edits leave them smooth, a
collapsed output turns them into noise.

Calibration sample (Qwen-Image-2.1, 1024x1024, corner patch 160, see README.md):
  healthy 12.8-22.7 (13 images) | collapsed 26.7-47.0 (6 images)

The gate is a detector, not a quality certificate: it flags a failed cell, it cannot
prove that an edit followed the instruction.

Usage:
  python benchmarks/qwen_image_21/quality_gate.py outputs/*.png
  python benchmarks/qwen_image_21/quality_gate.py outputs/*.png --json gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_PATCH = 160
DEFAULT_THRESHOLD = 25.0


def _laplacian_std(gray: np.ndarray) -> float:
    """Standard deviation of the 4-neighbour Laplacian of a 2D uint8 array."""
    g = gray.astype(np.float32)
    lap = (-4.0 * g + np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0) + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1))[
        1:-1, 1:-1
    ]
    return float(lap.std())


def corner_laplacian(path: str | Path, patch: int = DEFAULT_PATCH) -> float:
    """Mean high-frequency energy over the four corner patches of an image."""
    gray = np.asarray(Image.open(path).convert("L"))
    height, width = gray.shape
    size = min(patch, height // 2, width // 2)
    corners = [
        gray[:size, :size],
        gray[:size, width - size :],
        gray[height - size :, :size],
        gray[height - size :, width - size :],
    ]
    return float(np.mean([_laplacian_std(c) for c in corners]))


def evaluate(
    paths: list[str | Path],
    patch: int = DEFAULT_PATCH,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Gate each image; `collapsed` marks values at or above the threshold."""
    results = []
    for path in paths:
        value = corner_laplacian(path, patch)
        results.append(
            {
                "path": str(path),
                "corner_laplacian": round(value, 2),
                "threshold": threshold,
                "collapsed": value >= threshold,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="Edited images to check.")
    parser.add_argument("--patch", type=int, default=DEFAULT_PATCH, help="Corner patch size in pixels.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Collapse threshold.")
    parser.add_argument("--json", type=Path, default=None, help="Write results as JSON to this path.")
    args = parser.parse_args()

    results = evaluate(args.images, args.patch, args.threshold)
    width = max(len(r["path"]) for r in results)
    for r in results:
        verdict = "COLLAPSED" if r["collapsed"] else "ok"
        print(f"{r['path']:<{width}}  corner_lap {r['corner_laplacian']:7.2f}  {verdict}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")

    collapsed = sum(r["collapsed"] for r in results)
    print(f"\n{len(results) - collapsed}/{len(results)} ok (threshold {args.threshold}, patch {args.patch})")
    return 1 if collapsed else 0


if __name__ == "__main__":
    sys.exit(main())
