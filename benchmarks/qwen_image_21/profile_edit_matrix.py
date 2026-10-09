# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-GPU profile of Qwen-Image-2.1 text-to-image and edit (0/1/4 reference images).

Runs the offline examples as subprocesses -- one fresh process per cell and wave, so wall
time includes engine startup -- samples device memory in the background, parses the
generation time out of each example log, and gates every output through quality_gate.py.

Seed policy: self-generated sources are written with ``--source-seed`` (T2I) and edits run
with ``--edit-seed``. Never reuse the source generation seed as the edit seed: doing so
locks the sampler into re-rendering the source and drops the instruction, which the gate
then reports as a collapsed cell (see README.md).

Usage:
  CUDA_VISIBLE_DEVICES=0 python benchmarks/qwen_image_21/profile_edit_matrix.py \
      --model /path/to/Qwen-Image-2.1 --out-dir /tmp/qwen21_profile --waves 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from .quality_gate import DEFAULT_THRESHOLD, corner_laplacian
except ImportError:  # running as a plain script
    from quality_gate import DEFAULT_THRESHOLD, corner_laplacian

SOURCE_PROMPTS = {
    "ref_teapot": "A ceramic teapot on a wooden table",
    "ref_apple": "A single red apple on a white ceramic plate",
    "ref_mug": "A blue ceramic mug on a wooden desk",
    "ref_forest": "A misty pine forest at dawn",
}

CELLS = {
    "0ref": ("t2i", "A ceramic teapot on a wooden table", []),
    "1ref": ("edit", "Change the background to a moonlit beach with floating stars", ["ref_teapot"]),
    "4ref": (
        "edit",
        "Combine these four images into a single cozy still-life scene on a wooden table",
        ["ref_teapot", "ref_apple", "ref_mug", "ref_forest"],
    ),
}

T2I_EXAMPLE = "examples/offline_inference/text_to_image/text_to_image.py"
EDIT_EXAMPLE = "examples/offline_inference/image_to_image/image_edit.py"
GENERATION_TIME_RE = re.compile(r"Total generation time: ([0-9.]+)")


class MemorySampler:
    """Poll ``nvidia-smi`` for the used memory of one physical GPU until stopped."""

    def __init__(self, device_index: int, interval_s: float = 2.0):
        self.device_index = device_index
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(self.device_index),
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
                self.peak_mib = max(self.peak_mib, int(out))
            except (subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


def resolve_device_index(explicit: int | None) -> int:
    """Physical GPU index to sample: explicit value, else CUDA_VISIBLE_DEVICES, else 0."""
    if explicit is not None:
        return explicit
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return 0


def build_command(
    repo: Path,
    python: str,
    model: str,
    kind: str,
    output: Path,
    prompt: str,
    images: list[Path],
    seed: int,
    resolution: int,
    steps: int,
    cfg_scale: float,
) -> list[str]:
    example = T2I_EXAMPLE if kind == "t2i" else EDIT_EXAMPLE
    cmd = [
        python,
        str(repo / example),
        "--model",
        model,
        "--prompt",
        prompt,
        "--output",
        str(output),
        "--num-inference-steps",
        str(steps),
        "--cfg-scale",
        str(cfg_scale),
        "--seed",
        str(seed),
        "--width",
        str(resolution),
        "--height",
        str(resolution),
    ]
    if kind == "edit":
        # --image takes nargs="+": one flag followed by every path, not one flag per path.
        cmd += ["--color-format", "RGBA", "--image", *[str(image) for image in images]]
    return cmd


def run_cell(cmd: list[str], log_path: Path, device_index: int, repo: Path) -> dict:
    """Run one cell in a fresh process; return raw measurements."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    with MemorySampler(device_index) as sampler:
        start = time.perf_counter()
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        wall_s = time.perf_counter() - start
    text = log_path.read_text(errors="replace")
    match = GENERATION_TIME_RE.search(text)
    return {
        "rc": proc.returncode,
        "wall_s": round(wall_s, 2),
        "gen_s": float(match.group(1)) if match else None,
        "peak_mib": sampler.peak_mib,
    }


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_environment(repo: Path, python: str, args: argparse.Namespace, path: Path) -> None:
    """Record everything needed to interpret the numbers."""
    probe = (
        "import platform, torch, vllm;"
        "print('python', platform.python_version());"
        "print('torch', torch.__version__);"
        "print('vllm', vllm.__version__);"
        "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none');"
        "print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'none')"
    )
    versions = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=600)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=60
    ).stdout.strip()
    lines = [
        f"vllm-omni revision: {revision}",
        f"repo: {repo}",
        f"python: {python}",
        versions.stdout.strip() or versions.stderr.strip(),
        f"model: {args.model}",
        f"cells: {args.cells} | waves: {args.waves}",
        f"resolution: {args.resolution} | steps: {args.steps} | cfg-scale: {args.cfg_scale}",
        f"source seed (T2I): {args.source_seed} | edit seed: {args.edit_seed}",
        "flags: color-format RGBA (edits); no cache backend; no --enforce-eager; "
        "no CPU/layerwise offload; no VAE slicing/tiling",
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} | "
        f"sampled device index: {resolve_device_index(args.device_index)}",
        f"started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local path or name of the Qwen-Image-2.1 checkpoint.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Where logs, images and summaries go.")
    parser.add_argument("--repo", type=Path, default=repo_default, help="vllm-omni checkout holding the examples.")
    parser.add_argument("--python", default=sys.executable, help="Interpreter used for the examples.")
    parser.add_argument("--cells", default="0ref,1ref,4ref", help=f"Comma-separated subset of {sorted(CELLS)}.")
    parser.add_argument("--waves", type=int, default=3, help="Repeats per cell (same seeds every wave).")
    parser.add_argument("--resolution", type=int, default=1024, help="Square output resolution.")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps per cell.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="True-CFG scale (1.0 disables it).")
    parser.add_argument("--source-seed", type=int, default=42, help="T2I seed for the generated source images.")
    parser.add_argument("--edit-seed", type=int, default=7, help="Seed for edit cells; must differ from --source-seed.")
    parser.add_argument("--device-index", type=int, default=None, help="Physical GPU index for memory sampling.")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Collapse-gate threshold for corner Laplacian."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the commands and exit.")
    args = parser.parse_args()

    if args.edit_seed == args.source_seed:
        parser.error(
            f"--edit-seed ({args.edit_seed}) must differ from --source-seed ({args.source_seed}): reusing the "
            "source generation seed locks the sampler into re-rendering the source (see README.md)."
        )
    cells = [cell.strip() for cell in args.cells.split(",") if cell.strip()]
    unknown = [cell for cell in cells if cell not in CELLS]
    if unknown:
        parser.error(f"unknown cells: {unknown}; choose from {sorted(CELLS)}")

    repo = args.repo.resolve()
    out_dir: Path = args.out_dir
    sources_dir = out_dir / "sources"
    device_index = resolve_device_index(args.device_index)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)

    if args.dry_run:
        for cell in cells:
            kind, prompt, images = CELLS[cell]
            seed = args.source_seed if kind == "t2i" else args.edit_seed
            cmd = build_command(
                repo,
                args.python,
                args.model,
                kind,
                out_dir / f"{cell}_w1.png",
                prompt,
                [sources_dir / f"{name}.png" for name in images],
                seed,
                args.resolution,
                args.steps,
                args.cfg_scale,
            )
            print(" ".join(cmd))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    write_environment(repo, args.python, args, out_dir / "environment.txt")

    for name, prompt in SOURCE_PROMPTS.items():
        target = sources_dir / f"{name}.png"
        if target.exists():
            continue
        cmd = build_command(
            repo,
            args.python,
            args.model,
            "t2i",
            target,
            prompt,
            [],
            args.source_seed,
            args.resolution,
            args.steps,
            args.cfg_scale,
        )
        print(f"[sources] generating {name} (T2I seed {args.source_seed})", flush=True)
        with (sources_dir / f"{name}.log").open("w") as source_log:
            proc = subprocess.run(cmd, cwd=repo, env=env, stdout=source_log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"[sources] FAILED: {name} (see {sources_dir / f'{name}.log'})")
            return proc.returncode

    rows = []
    for cell in cells:
        kind, prompt, image_names = CELLS[cell]
        seed = args.source_seed if kind == "t2i" else args.edit_seed
        for wave in range(1, args.waves + 1):
            output = out_dir / f"{cell}_w{wave}.png"
            cmd = build_command(
                repo,
                args.python,
                args.model,
                kind,
                output,
                prompt,
                [sources_dir / f"{name}.png" for name in image_names],
                seed,
                args.resolution,
                args.steps,
                args.cfg_scale,
            )
            print(f"[{cell} w{wave}/{args.waves}] {kind} seed={seed} -> {output.name}", flush=True)
            result = run_cell(cmd, out_dir / f"{cell}_w{wave}.log", device_index, repo)
            value = corner_laplacian(output) if output.exists() else None
            row = {
                "cell": cell,
                "wave": wave,
                "kind": kind,
                "seed": seed,
                **result,
                "corner_laplacian": round(value, 2) if value is not None else None,
                "collapsed": value >= args.threshold if value is not None else None,
                "md5": md5(output) if output.exists() else None,
                "output": str(output),
            }
            rows.append(row)
            print(
                f"    rc={row['rc']} wall={row['wall_s']}s gen={row['gen_s']}s "
                f"peak={row['peak_mib']}MiB corner_lap={row['corner_laplacian']} "
                f"{'COLLAPSED' if row['collapsed'] else 'ok'}",
                flush=True,
            )

    fields = [
        "cell",
        "wave",
        "kind",
        "seed",
        "rc",
        "wall_s",
        "gen_s",
        "peak_mib",
        "corner_laplacian",
        "collapsed",
        "md5",
        "output",
    ]
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    print("\ncell  wave  rc  wall_s  gen_s  peak_mib  corner_lap  gate")
    for row in rows:
        print(
            f"{row['cell']:<5s} {row['wave']:>4d} {row['rc']:>3d} {row['wall_s']:>7.2f} "
            f"{str(row['gen_s']):>6s} {row['peak_mib']:>9d} {str(row['corner_laplacian']):>10s} "
            f"{'COLLAPSED' if row['collapsed'] else 'ok'}"
        )
    for cell in cells:
        digests = {row["md5"] for row in rows if row["cell"] == cell and row["md5"]}
        if len(digests) == 1:
            print(f"{cell}: bit-identical across waves")

    print(f"\nwrote {out_dir / 'summary.csv'} and {out_dir / 'environment.txt'}")
    failed = [row for row in rows if row["rc"] != 0 or row["collapsed"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
