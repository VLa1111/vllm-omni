# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-GPU profile of Qwen-Image-2.1 text-to-image and edit (0/1/4 reference images).

Runs the offline examples as subprocesses -- one fresh process per cell, resolution and
wave, so wall time includes engine startup -- samples device memory in the background,
parses the generation time out of each example log, and gates every output through
quality_gate.py.

Seed policy: self-generated sources are written with ``--source-seed`` (T2I) and edits run
with ``--edit-seed``. Never reuse the source generation seed as the edit seed: doing so
locks the sampler into re-rendering the source and drops the instruction, which the gate
then reports as a collapsed cell (see README.md).

Usage:
  CUDA_VISIBLE_DEVICES=0 python benchmarks/qwen_image_21/profile_edit_matrix.py \
      --model /path/to/Qwen-Image-2.1 --out-dir /tmp/qwen21_profile --waves 3

  # native resolution and the batching cell
  CUDA_VISIBLE_DEVICES=0 python benchmarks/qwen_image_21/profile_edit_matrix.py \
      --model /path/to/Qwen-Image-2.1 --out-dir /tmp/qwen21_profile_2048 \
      --resolution 2048 --waves 3
  CUDA_VISIBLE_DEVICES=0 python benchmarks/qwen_image_21/profile_edit_matrix.py \
      --model /path/to/Qwen-Image-2.1 --out-dir /tmp/qwen21_profile_batch \
      --cells 0ref,1ref --num-outputs-per-prompt 4 --waves 3
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
    num_outputs_per_prompt: int = 1,
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
        if num_outputs_per_prompt > 1:
            cmd += ["--num-outputs-per-prompt", str(num_outputs_per_prompt)]
    elif num_outputs_per_prompt > 1:
        cmd += ["--num-images-per-prompt", str(num_outputs_per_prompt)]
    return cmd


def collect_images(output: Path, num_outputs_per_prompt: int) -> list[Path]:
    """Images produced by one run: the single output, or ``stem_idx`` siblings when batched."""
    if num_outputs_per_prompt <= 1:
        return [output] if output.exists() else []
    return sorted(output.parent.glob(f"{output.stem}_*{output.suffix}"))


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


def ensure_sources(
    repo: Path,
    args: argparse.Namespace,
    sources_dir: Path,
    resolution: int,
    device_index: int,
) -> bool:
    """Generate the T2I source images for one resolution if they are not there yet."""
    del device_index  # sources are generated outside the memory sampler
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
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
            resolution,
            args.steps,
            args.cfg_scale,
        )
        print(f"[sources r{resolution}] generating {name} (T2I seed {args.source_seed})", flush=True)
        with (sources_dir / f"{name}.log").open("w") as source_log:
            proc = subprocess.run(cmd, cwd=repo, env=env, stdout=source_log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"[sources r{resolution}] FAILED: {name} (see {sources_dir / f'{name}.log'})")
            return False
    return True


def write_environment(repo: Path, python: str, args: argparse.Namespace, resolutions: list[int], path: Path) -> None:
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
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=60)
    lines = [
        f"vllm-omni revision: {revision.stdout.strip()}",
        f"repo: {repo}",
        f"python: {python}",
        versions.stdout.strip() or versions.stderr.strip(),
        f"model: {args.model}",
        f"cells: {args.cells} | waves: {args.waves} | resolutions: {resolutions}",
        f"steps: {args.steps} | cfg-scale: {args.cfg_scale} | num_outputs_per_prompt: {args.num_outputs_per_prompt}",
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
    parser.add_argument("--resolution", default="1024", help="Comma-separated square resolutions, e.g. 1024,2048.")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps per cell.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="True-CFG scale (1.0 disables it).")
    parser.add_argument("--source-seed", type=int, default=42, help="T2I seed for the generated source images.")
    parser.add_argument("--edit-seed", type=int, default=7, help="Seed for edit cells; must differ from --source-seed.")
    parser.add_argument(
        "--num-outputs-per-prompt",
        type=int,
        default=1,
        help="Images per request (batching cell); mapped to the flag each example expects.",
    )
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
    resolutions = [int(value) for value in str(args.resolution).split(",") if value.strip()]

    repo = args.repo.resolve()
    out_dir: Path = args.out_dir
    device_index = resolve_device_index(args.device_index)

    if args.dry_run:
        for resolution in resolutions:
            for cell in cells:
                kind, prompt, images = CELLS[cell]
                seed = args.source_seed if kind == "t2i" else args.edit_seed
                cmd = build_command(
                    repo,
                    args.python,
                    args.model,
                    kind,
                    out_dir / f"{cell}_r{resolution}_w1.png",
                    prompt,
                    [out_dir / "sources" / f"r{resolution}" / f"{name}.png" for name in images],
                    seed,
                    resolution,
                    args.steps,
                    args.cfg_scale,
                    args.num_outputs_per_prompt,
                )
                print(" ".join(cmd))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_environment(repo, args.python, args, resolutions, out_dir / "environment.txt")

    rows = []
    for resolution in resolutions:
        sources_dir = out_dir / "sources" / f"r{resolution}"
        sources_dir.mkdir(parents=True, exist_ok=True)
        if not ensure_sources(repo, args, sources_dir, resolution, device_index):
            return 1

        for cell in cells:
            kind, prompt, image_names = CELLS[cell]
            seed = args.source_seed if kind == "t2i" else args.edit_seed
            for wave in range(1, args.waves + 1):
                output = out_dir / f"{cell}_r{resolution}_w{wave}.png"
                cmd = build_command(
                    repo,
                    args.python,
                    args.model,
                    kind,
                    output,
                    prompt,
                    [sources_dir / f"{name}.png" for name in image_names],
                    seed,
                    resolution,
                    args.steps,
                    args.cfg_scale,
                    args.num_outputs_per_prompt,
                )
                print(f"[{cell} r{resolution} w{wave}/{args.waves}] {kind} seed={seed}", flush=True)
                result = run_cell(cmd, out_dir / f"{cell}_r{resolution}_w{wave}.log", device_index, repo)
                images = collect_images(output, args.num_outputs_per_prompt)
                gates = [corner_laplacian(image) for image in images]
                digests = [md5(image) for image in images]
                row = {
                    "cell": cell,
                    "resolution": resolution,
                    "wave": wave,
                    "kind": kind,
                    "seed": seed,
                    "num_outputs_per_prompt": args.num_outputs_per_prompt,
                    **result,
                    "n_images": len(images),
                    "n_distinct_md5": len(set(digests)),
                    "corner_lap_min": round(min(gates), 2) if gates else None,
                    "corner_lap_max": round(max(gates), 2) if gates else None,
                    "collapsed": any(value >= args.threshold for value in gates) if gates else None,
                    "md5_first": digests[0] if digests else None,
                    "output": str(images[0]) if images else str(output),
                }
                rows.append(row)
                print(
                    f"    rc={row['rc']} wall={row['wall_s']}s gen={row['gen_s']}s peak={row['peak_mib']}MiB "
                    f"images={row['n_images']} distinct={row['n_distinct_md5']} "
                    f"corner_lap={row['corner_lap_min']}-{row['corner_lap_max']} "
                    f"{'COLLAPSED' if row['collapsed'] else 'ok'}",
                    flush=True,
                )

    fields = [
        "cell",
        "resolution",
        "wave",
        "kind",
        "seed",
        "num_outputs_per_prompt",
        "rc",
        "wall_s",
        "gen_s",
        "peak_mib",
        "n_images",
        "n_distinct_md5",
        "corner_lap_min",
        "corner_lap_max",
        "collapsed",
        "md5_first",
        "output",
    ]
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    print("\ncell  res   wave  rc  wall_s  gen_s  peak_mib  images  corner_lap  gate")
    for row in rows:
        print(
            f"{row['cell']:<5s} {row['resolution']:>4d} {row['wave']:>5d} {row['rc']:>3d} {row['wall_s']:>7.2f} "
            f"{str(row['gen_s']):>6s} {row['peak_mib']:>9d} {row['n_images']:>7d} "
            f"{str(row['corner_lap_min']):>10s}  {'COLLAPSED' if row['collapsed'] else 'ok'}"
        )
    for resolution in resolutions:
        for cell in cells:
            digests = {row["md5_first"] for row in rows if row["cell"] == cell and row["resolution"] == resolution}
            if len(digests) == 1 and rows:
                print(f"{cell} r{resolution}: bit-identical across waves")

    print(f"\nwrote {out_dir / 'summary.csv'} and {out_dir / 'environment.txt'}")
    failed = [row for row in rows if row["rc"] != 0 or row["collapsed"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
