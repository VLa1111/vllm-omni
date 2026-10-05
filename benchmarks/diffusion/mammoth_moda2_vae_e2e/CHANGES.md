# Changes: offline E2E runner for the MammothModa2 VAE modes

This directory originally contained `run_e2e.sh`, which drives the sweep through
`vllm-omni serve` + `vllm-omni bench serve`. On this head that path cannot
measure MammothModa2, so this change adds `run_e2e_offline.py`, which drives the
same matrix through the offline pipeline instead, plus the re-measured
`results.md` / `results.json` and `BENCHMARK_REPORT.md`.

## Why the server path cannot drive MammothModa2 here

MammothModa2's AR stage only generates visual tokens for a request that carries
the T2I scaffold built by `vllm_omni.model_extras`:

- the prompt string
  `<|im_start|>system\nYou are a helpful image generator.<|im_end|>...<|image start|>W*H<|image token|>`, and
- `additional_information` with `omni_task=["t2i"]`, `ar_width` / `ar_height`,
  `image_height` / `image_width`, `eol_token_id`, `visual_token_start_id` /
  `visual_token_end_id`.

Only the offline T2I example path (`build_text_to_image_prompt`, used by
`examples/offline_inference/text_to_image/text_to_image.py`) applies that
builder. No serving endpoint does: the chat path with `modalities=["image"]`
and `/v1/images/generations` both forward a plain prompt and never set
`omni_task`, and without it the AR stage's sampler forbidden-mask treats the
request as text chat and cannot emit visual tokens — every request ends in
`MammothModa2 AR stage produced no visual-token hidden states` from the DiT
stage. Verified with a hostile probe: an intentionally invalid `eol_token_id`
sent through HTTP changed nothing, i.e. the request's `additional_information`
is never consumed by the AR stage over HTTP. `vllm-omni bench serve` also
feeds `--dataset-name random`, a synthetic text-only dataset, so its prompts do
not trigger image generation either.

`run_e2e.sh` is left in place for the day the serving path learns the scaffold;
it is not usable for this model on this head.

## What `run_e2e_offline.py` does

- builds the request exactly like the shared T2I example (model_extras prompt,
  declared extra-body params, AR `max_tokens` sized from the prompt metadata);
- one subprocess per (size, config) cell, so allocator state from a previous
  engine can never inflate the next cell's device peak;
- starts the 0.5 s device-memory sampler *before* the engine is built, so the
  peak covers model load as well as generation (matching the reference table);
- issues batch-4 rows as waves of 4 concurrent requests
  (`Omni.generate(list_of_prompts, ...)` submits them together, letting the DiT
  stage decode a batch) and reports the amortized per-request latency;
- saves one image per batch-1 config and writes `bench_<size>_<config>.log` in
  the format `collect.py` already parses, so results land in the same
  `results.md` / `results.json` shape as the server-based sweep.

## Preconditions

- **vLLM 0.30.0.** The head contains main's *Rebase to vLLM 0.30.0* (#7820);
  with vLLM 0.29.0 the DiT stage fails at init with
  `pydantic ValidationError ... IrOpPriorityConfig ... gelu_and_mul_sparse`.
  The measurement container had 0.29.0 and was upgraded in place
  (`pip install vllm==0.30.0`); torch stayed at 2.13.0+cu130.
- `PYTHONPATH=<repo>` so the checkout's `vllm_omni` shadows the container's
  installed release.
- PyYAML / Pillow / numpy for the config patch and the PSNR column.

## Usage

```bash
cd /app/vllm_omni
export PYTHONPATH=$PWD
python benchmarks/diffusion/mammoth_moda2_vae_e2e/run_e2e_offline.py \
    --model /root/models/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --out ~/pro6000-vae-e2e --sizes 1536,1024 --gpu 2
python benchmarks/diffusion/mammoth_moda2_vae_e2e/collect.py --out ~/pro6000-vae-e2e
```

All flags (`--sizes`, `--configs`, `--num-prompts`, `--num-warmups`, `--steps`,
`--guidance`, `--seed`, `--gpu`, `--dry-run`) mirror `run_e2e.sh`; `--cell` is
internal (one cell per process).

Measured results and their analysis are in `BENCHMARK_REPORT.md`.
