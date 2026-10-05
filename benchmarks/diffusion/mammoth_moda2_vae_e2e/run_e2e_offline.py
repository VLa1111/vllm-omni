# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline E2E sweep for the MammothModa2 VAE memory modes.

``run_e2e.sh`` drives the same matrix through ``vllm-omni serve`` +
``vllm-omni bench serve``.  That path cannot work for MammothModa2 on this
head: the AR stage only generates visual tokens for requests that carry the
model_extras T2I scaffold (``omni_task=t2i`` + the AR grid metadata), and no
serving endpoint builds it -- every HTTP request falls into the text-chat
path and the DiT stage rejects the (text-only) AR output.  The offline
example path (``examples/offline_inference/text_to_image/text_to_image.py``)
is the one that builds the scaffold, and it is what the reference table in
``recipes/MammothModa2/MammothModa2.md`` was measured with.

This runner drives that same path in-process, per (size, config):

  1. writes a deploy config with the two flags injected into the DiT stage,
  2. builds the request exactly like the shared T2I example (model_extras
     prompt + declared extra-body params + AR max_tokens sizing),
  3. runs ``--num-warmups`` warmups and ``--num-prompts`` measured requests,
     in waves of the config's concurrency (4 for the ``-b4`` rows, so the
     DiT stage decodes a batch and slicing can act),
  4. samples device memory every 0.5 s for the whole run,
  5. saves one image per config (PSNR input for collect.py) -- the last
     measured wave's first output when the config runs waves,
  6. writes ``bench_<size>_<config>.log`` in the format ``collect.py`` parses,
     plus per-request timing details under ``raw/``.

Example::

    python run_e2e_offline.py --model /root/models/MammothModa2-Preview \
        --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
        --out ~/pro6000-vae-e2e --sizes 1536,1024 --gpu 2

    python collect.py --out ~/pro6000-vae-e2e
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

CONFIGS = {  # name -> (slicing, tiling, concurrency)
    "baseline": (False, False, 1),
    "slicing": (True, False, 1),
    "tiling": (False, True, 1),
    "both": (True, True, 1),
    "slicing-b4": (True, False, 4),
    "both-b4": (True, True, 4),
    # Batch-4 controls for the slicing A/B (see FOLLOWUP_RUN.md): the same
    # concurrency with the flags off, so a batch>1 row can be compared against
    # its own batch rather than against the single-request baseline.
    "baseline-b4": (False, False, 4),
    "tiling-b4": (False, True, 4),
}
DEFAULT_CONFIGS = "baseline,slicing,tiling,both,slicing-b4"
PROMPT = "A stylish woman riding a motorcycle in NYC, movie poster style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local MammothModa2 checkpoint directory")
    parser.add_argument("--deploy-config", required=True, help="Base deploy config to patch per (size, config)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--sizes", default="1536,1024", help="Comma-separated square sizes")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, help=f"Subset of {','.join(CONFIGS)}")
    parser.add_argument("--num-prompts", type=int, default=5, help="Measured requests per config")
    parser.add_argument("--num-warmups", type=int, default=4, help="Warmup requests per config")
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--guidance", type=float, default=9.0, help="text_guidance_scale")
    parser.add_argument("--cfg-range", default="0,1", help="cfg_range as lo,hi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without loading the model")
    parser.add_argument(
        "--cell",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def write_deploy_config(src: Path, dst: Path, slicing: bool, tiling: bool) -> None:
    import yaml

    with src.open() as handle:
        cfg = yaml.safe_load(handle)
    dit = [stage for stage in cfg["stages"] if stage.get("stage_id") == 1]
    if not dit:
        raise SystemExit(f"no stage_id: 1 in {src}")
    dit[0]["vae_use_slicing"] = slicing
    dit[0]["vae_use_tiling"] = tiling
    with dst.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print(f"[{dst}] stage 1: vae_use_slicing={slicing} vae_use_tiling={tiling}")


class MemorySampler:
    """Sample whole-device memory every 0.5 s until stopped."""

    def __init__(self, gpu: int, path: Path) -> None:
        self.gpu = gpu
        self.path = path
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        with self.path.open("w") as handle:
            while True:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if out.stdout.strip():
                        handle.write(out.stdout)
                        handle.flush()
                except Exception:
                    pass
                time.sleep(0.5)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # The thread is a daemon that only stops with the process; give it a
        # moment to flush the final sample before the file is read.
        time.sleep(0.2)


def build_request(omni, prompt_dict: dict, size: int, args: argparse.Namespace):
    """Mirror the shared T2I example's request construction."""
    from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
    from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import (
        get_extra_body_params,
        get_model_class_name,
        should_init_extra_args_for_non_diffusion_stages,
    )

    model_class_name = get_model_class_name(omni)
    declared_extra_body_params = get_extra_body_params(model_class_name)

    diffusion_params = OmniDiffusionSamplingParams(
        height=size,
        width=size,
        seed=args.seed,
        num_inference_steps=args.steps,
        num_outputs_per_prompt=1,
    )
    lo, hi = (float(v) for v in args.cfg_range.split(","))
    user_extra = {
        "text_guidance_scale": args.guidance,
        "cfg_range": [lo, hi],
        "num_inference_steps": args.steps,
    }
    if declared_extra_body_params:
        apply_declared_extra_args(diffusion_params, declared_extra_body_params, user_extra)

    init_non_diffusion = should_init_extra_args_for_non_diffusion_stages(model_class_name)
    defaults = list(omni.default_sampling_params_list or [])
    sampling_params_list = [clone_sampling_params(p) for p in defaults]
    if not sampling_params_list:
        sampling_params_list = [diffusion_params]

    diffusion_replaced = False
    for idx, params in enumerate(sampling_params_list):
        if isinstance(params, OmniDiffusionSamplingParams):
            sampling_params_list[idx] = diffusion_params
            diffusion_replaced = True
        elif init_non_diffusion and hasattr(params, "extra_args"):
            if params.extra_args is None:
                params.extra_args = {}
            params.extra_args.update(diffusion_params.extra_args or {})
            if hasattr(params, "seed"):
                params.seed = args.seed
            prompt_info = prompt_dict.get("additional_information", {})
            if idx == 0 and prompt_info.get("omni_task") == ["t2i"]:
                ar_width = int(prompt_info.get("ar_width", [0])[0])
                ar_height = int(prompt_info.get("ar_height", [0])[0])
                if ar_width > 0 and ar_height > 0:
                    params.max_tokens = ar_height * (ar_width + 1) + 1

    if not diffusion_replaced and len(sampling_params_list) == 1:
        sampling_params_list = [diffusion_params]
    return sampling_params_list


def run_wave(omni, prompts: list[dict], sampling_params_list, records: list[dict], phase: str, tag: str) -> list:
    prompts = [copy.deepcopy(p) for p in prompts]
    start = time.perf_counter()
    outputs = omni.generate(prompts, sampling_params_list=copy.deepcopy(sampling_params_list), use_tqdm=False)
    elapsed = time.perf_counter() - start
    per_request = elapsed / max(len(prompts), 1)
    for output in outputs:
        stage_durations = getattr(output, "stage_durations", None) or {}
        records.append(
            {
                "phase": phase,
                "batch": len(prompts),
                "wall_s": per_request,
                "stage_durations": {str(k): float(v) for k, v in stage_durations.items()},
                "peak_memory_mb": float(getattr(output, "peak_memory_mb", 0.0) or 0.0),
                "request_id": getattr(output, "request_id", ""),
            }
        )
    return outputs


def extract_image(outputs) -> object | None:
    for output in outputs or []:
        images = getattr(output, "images", None)
        if images:
            return images[0]
    return None


def write_bench_log(path: Path, records: list[dict], tag: str) -> None:
    """Emit a bench-style log: global E2E metrics plus per-stage sections."""
    measured = [r for r in records if r["phase"] == "measured"]
    times_ms = sorted(r["wall_s"] * 1000.0 for r in measured)
    lines: list[str] = []
    if times_ms:
        mean = statistics.fmean(times_ms)
        median = statistics.median(times_ms)
        lines.append(f"Mean E2E latency: {mean:.2f}")
        lines.append(f"Median E2E latency: {median:.2f}")
        for pct in (50, 95):
            idx = min(len(times_ms) - 1, max(0, round(pct / 100.0 * len(times_ms)) - 1))
            lines.append(f"P{pct} E2E latency: {times_ms[idx]:.2f}")
        lines.append(f"Successful requests: {len(times_ms)}")

    # The engine reports per-stage generation time under ``stage_<id>_gen_ms``;
    # other keys (pipeline timings such as queue_wait_ms / <stage>.diffuse /
    # <stage>.vae.decode) stay in raw/<tag>.json only.
    stage_key_re = re.compile(r"^stage_(\d+)_gen_ms$")
    stage_ids = sorted(
        {int(m.group(1)) for r in records for key in r["stage_durations"] if (m := stage_key_re.match(key))}
    )
    stage_names = {0: "llm", 1: "diffusion"}
    for sid in stage_ids:
        values = [
            float(r["stage_durations"][f"stage_{sid}_gen_ms"])
            for r in measured
            if f"stage_{sid}_gen_ms" in r["stage_durations"]
        ]
        title = f" Stage {sid} ({stage_names.get(sid, f'stage_{sid}')}) "
        lines.append("")
        lines.append(f"{title:=^50}")
        if values:
            lines.append(f"Mean stage_gen_time: {statistics.fmean(values):.2f}")
            lines.append(f"Median stage_gen_time: {statistics.median(values):.2f}")
    path.write_text("\n".join(lines) + "\n")


def run_one(omni_module, cfgfile: Path, size: int, config: str, args: argparse.Namespace, out: Path) -> dict:
    slicing, tiling, concurrency = CONFIGS[config]
    tag = f"{size}_{config}"
    print(f"=== {tag}: slicing={int(slicing)} tiling={int(tiling)} concurrency={concurrency} ===")

    from vllm_omni.model_extras import build_text_to_image_prompt, get_model_class_name

    sampler = MemorySampler(args.gpu, out / f"mem_{tag}.txt")
    omni = None
    records: list[dict] = []
    image_path = None
    # Sample from before the engine is built so the peak covers model load as
    # well as generation, matching how the reference table was measured.
    sampler.start()
    try:
        omni = omni_module.Omni(model=args.model, deploy_config=str(cfgfile), log_stats=True)
        model_class_name = get_model_class_name(omni)
        prompt_dict = {"prompt": args.prompt, "modalities": ["image"]}
        prompt_dict = build_text_to_image_prompt(model_class_name, prompt_dict, height=size, width=size)
        sampling_params_list = build_request(omni, prompt_dict, size, args)

        warmups_done = 0
        while warmups_done < args.num_warmups:
            wave = min(concurrency, args.num_warmups - warmups_done)
            run_wave(omni, [prompt_dict] * wave, sampling_params_list, records, "warmup", tag)
            warmups_done += wave
            print(f"  warmup {warmups_done}/{args.num_warmups} done")
        remaining = args.num_prompts
        last_outputs = None
        while remaining > 0:
            wave = min(concurrency, remaining)
            last_outputs = run_wave(omni, [prompt_dict] * wave, sampling_params_list, records, "measured", tag)
            remaining -= wave
            measured = [r for r in records if r["phase"] == "measured"]
            print(f"  measured {len(measured)}/{args.num_prompts} done")
        # Wave configs save the first image of the last measured wave: every
        # request in a wave carries the same prompt and seed, so any member
        # stands in for the config's output.
        if last_outputs is not None:
            image = extract_image(last_outputs)
            if image is not None:
                image_path = out / f"img_{tag}.png"
                image.save(image_path)
                print(f"  saved {image_path}")
        sampler.stop()
    finally:
        sampler.stop()
        if omni is not None:
            try:
                omni.close()
            except Exception as exc:  # pragma: no cover - best effort teardown
                print(f"  warning: engine close failed: {exc}", file=sys.stderr)

    measured = [r for r in records if r["phase"] == "measured"]
    write_bench_log(out / f"bench_{tag}.log", records, tag)
    (out / "raw").mkdir(exist_ok=True)
    (out / "raw" / f"{tag}.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "size": size,
                "config": config,
                "slicing": slicing,
                "tiling": tiling,
                "concurrency": concurrency,
                "image": str(image_path) if image_path else None,
                "records": records,
            },
            indent=2,
        )
    )
    return {"tag": tag, "measured": len(measured)}


def main() -> None:
    args = parse_args()
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    configs = [c for c in args.configs.split(",") if c.strip()]
    for config in configs:
        if config not in CONFIGS:
            raise SystemExit(f"unknown config: {config}")

    deploy = Path(args.deploy_config)
    print(
        f"model={args.model} deploy={deploy} sizes={sizes} configs={configs} "
        f"prompts={args.num_prompts} warmups={args.num_warmups} steps={args.steps} "
        f"guidance={args.guidance} seed={args.seed} gpu={args.gpu}"
    )
    (out / "runs.txt").write_text(
        json.dumps(
            {
                "model": args.model,
                "deploy": str(deploy),
                "sizes": sizes,
                "configs": configs,
                "num_prompts": args.num_prompts,
                "num_warmups": args.num_warmups,
                "steps": args.steps,
                "guidance": args.guidance,
                "cfg_range": args.cfg_range,
                "seed": args.seed,
                "gpu": args.gpu,
                "prompt": args.prompt,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
    )

    plan = [(size, config) for size in sizes for config in configs]
    if args.dry_run:
        for size, config in plan:
            slicing, tiling, concurrency = CONFIGS[config]
            print(
                f"would run {size}_{config}: slicing={int(slicing)} tiling={int(tiling)} "
                f"concurrency={concurrency} (deploy_{size}_{config}.yaml)"
            )
        return

    if args.cell:
        # Internal: run exactly one cell in this process. Each cell gets a
        # fresh process so a previous engine's allocator/cache can never
        # inflate the next cell's device peak.
        size_str, config = args.cell.split("_", 1)
        size = int(size_str)
        slicing, tiling, _ = CONFIGS[config]
        cfgfile = out / f"deploy_{args.cell}.yaml"
        write_deploy_config(deploy, cfgfile, slicing, tiling)
        import vllm_omni.entrypoints.omni as omni_module

        result = run_one(omni_module, cfgfile, size, config, args, out)
        print(f"exit={result['tag']} measured={result['measured']}")
        return

    for size, config in plan:
        env = dict(os.environ)
        cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--cell", f"{size}_{config}"]
        print(f"--- launching cell {size}_{config} ---")
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"!! cell {size}_{config} exited with rc={proc.returncode}", file=sys.stderr)

    print()
    print(f"raw output in {out}; now run:")
    print(f"  {sys.executable} {Path(__file__).with_name('collect.py')} --out {out}")


if __name__ == "__main__":
    main()
