# MammothModa2 VAE slicing/tiling — offline end-to-end re-measurement

Re-measurement of the recipe's end-to-end table (`recipes/MammothModa2/MammothModa2.md`,
section *Measured end-to-end (RTX PRO 6000 Blackwell 96 GB)*) on the head to be
merged for PR #7774, including the batch > 1 row requested in review.

- Head measured: `pro6000/mammothmoda2-vae-e2e` @ `6873f69b8`
  (PR head `b9b0ae663` plus the scratch tooling in this directory).
- Driver: `run_e2e_offline.py` in this directory (see `CHANGES.md` for why the
  server-based `run_e2e.sh` cannot drive this model).
- Raw output: `results.json`, `results.md` (copied here) and the per-cell
  logs/images under `/root/pro6000-vae-e2e-offline/` on the measurement box.

## Environment

| | |
| --- | --- |
| OS | Ubuntu 24.04.3 LTS (container `docker.m.daocloud.io/vllm/vllm-omni:nightly`), Linux 7.0.0-31-generic |
| GPU | one NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB (physical index 2) |
| vLLM | 0.30.0 (upgraded in place from 0.29.0; the head contains main's *Rebase to vLLM 0.30.0* #7820, whose `IrOpPriorityConfig.gelu_and_mul_sparse` does not exist in 0.29.0) |
| PyTorch | 2.13.0+cu130 |
| Model | `/root/models/MammothModa2-Preview` (local checkpoint) |
| Code loading | `PYTHONPATH=/app/vllm_omni` (the container's installed `vllm_omni` metadata predates this branch) |

The reference table in the recipe was measured at `624ebea` with vLLM 0.29.0 on
the same box; nothing about the flags changed, only the code around them.

## Protocol

Fixed prompt `A stylish woman riding a motorcycle in NYC, movie poster style`,
`seed=42`, `num_inference_steps=50`, `text_guidance_scale=9.0`,
`cfg_range=[0,1]`. Per (size, config): one fresh process, one model load
(memory sampled from before the load), then 4 warmups + 5 measured requests in
waves of the config's concurrency; device memory sampled every 0.5 s; one image
captured per batch-1 config for the PSNR comparison. The batch-4 rows issue a
wave of 4 concurrent requests (then 1) so the DiT stage decodes a batch and
slicing can act; the reported E2E value for those rows is the amortized
per-request latency of the wave.

`vae_use_slicing` / `vae_use_tiling` are injected into stage 1 of the deploy
config, matching the recipe's documented placement of these stage fields.

## Results

### 1536x1536 (above the tiling threshold)

| Config | Slice | Tile | Conc | Peak MiB | E2E mean s | Stage 0 (AR) s | Stage 1 (DiT+VAE) s | PSNR vs baseline |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | no | no | 1 | 67,766 | 217.8 | 193.0 | 24.6 | — |
| slicing | yes | no | 1 | 67,766 | 214.4 | 189.6 | 24.7 | identical |
| tiling | no | yes | 1 | 61,656 | 218.7 | 193.8 | 24.8 | 47.04 dB |
| both | yes | yes | 1 | 61,654 | 215.3 | 190.4 | 24.8 | 47.04 dB |
| slicing-b4 | yes | no | 4 | 67,200 | 106.2 (amortized) | 203.7 | 74.4 | — |

### 1024x1024 (at / below the tiling threshold)

| Config | Slice | Tile | Conc | Peak MiB | E2E mean s | Stage 0 (AR) s | Stage 1 (DiT+VAE) s | PSNR vs baseline |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | no | no | 1 | 60,844 | 105.4 | 96.3 | 9.1 | — |
| slicing | yes | no | 1 | 60,844 | 105.7 | 96.6 | 9.1 | identical |
| tiling | no | yes | 1 | 60,844 | 106.0 | 96.9 | 9.1 | identical |
| both | yes | yes | 1 | 60,844 | 106.5 | 97.2 | 9.2 | identical |
| slicing-b4 | yes | no | 4 | 60,848 | 50.0 (amortized) | 103.2 | 25.2 | — |

All numbers are means over the 5 measured requests, taken from `results.json`
(`e2e_mean_ms`, `stage_gen_mean_ms`, `peak_mib`, `psnr_db`). "identical" means
byte-identical to the baseline image at that size.

## Comparison with the reference table (measured at `624ebea`)

| Metric | Reference | This run | Delta |
| --- | ---: | ---: | ---: |
| 1536 baseline E2E | 225.9 s | 217.8 s | +3.6 % faster |
| 1536 baseline peak | 67,354 MiB | 67,766 MiB | +412 MiB (+0.6 %) |
| 1536 tiling peak | 61,814 MiB | 61,656 MiB | −158 MiB |
| 1536 tiling saving vs baseline | 5,540 MiB | 6,110 MiB | ~5.4 → ~6.0 GiB |
| 1024 peak (all configs) | 61,004 MiB | 60,844 MiB | −160 MiB |
| 1536 tiling PSNR | 46.75 dB | 47.04 dB | within run-to-run noise |

The headline findings of the reference table reproduce:

- **Tiling bounds the device peak at 1536x1536** (67,766 → 61,656 MiB, a 6.0 GiB
  saving) and is a no-op below the threshold: at 1024x1024 the latent (128) does
  not exceed `tile_latent_min_size`, so all five configs peak within 4 MiB of
  each other and the tiled images are byte-identical to the baseline.
- **Slicing alone changes nothing at batch 1**: same peak, byte-identical output.
- **End-to-end latency is dominated by the AR stage** (~89 % of the wall time)
  and is not a useful instrument for this option: at 1536 the four batch-1
  configs sit within 4.3 s of each other end to end while stage 1 varies by only
  0.14 s. The E2E delta between reference and this run tracks AR-stage drift.

### What the batch > 1 row adds (the review request)

`slicing-b4` (4 concurrent requests, slicing on) is the configuration slicing
was designed for, and it shows two things the batch-1 rows could not:

- **Throughput**: at 1536 a wave of 4 images completes in ~314 s
  (78.6 s amortized per image) against 216.7 s for a single request in the same
  run — ~2.8x per-image throughput. At 1024 the wave costs ~144 s
  (36.0 s amortized) against 105.9 s single — ~2.9x. The AR stage batches the
  four sequences (per-request AR time grows from 191.9 s to ~206.6 s for four
  requests), and stage 1 decodes the batch in ~107 s (1536) / ~39 s (1024) for
  all four images instead of 4 x 24.7 s / 4 x 9.0 s serially.
- **Peak stays flat**: 67,200 MiB at 1536 vs 67,766 MiB for the single-image
  baseline — four decoded images in flight cost *less* whole-device peak than
  one image without slicing. At 1024 the difference is 4 MiB. Slicing splits
  the VAE decode along the batch dimension, so the peak does not grow with the
  batch, which is exactly the property the batch>1 request was probing.

Limitation: this sweep does not include a batch-4 *without*-slicing control row
(the runner supports `both-b4`; `baseline-b4` is not defined), so the b4 numbers
above are compared against the single-request baseline rather than against the
same batch without the flag. Adding a batch-4 control is the obvious follow-up
if the reviewer wants the in-batch A/B.

## Failures

None. All 10 cells completed with 5/5 measured requests and 4/4 warmups; no cell
was skipped by the driver.

## Artifacts

`results.json` keeps every metric per run (global latency mean/median/p50/p95,
per-stage gen-time means, device peaks, PSNR); `results.md` is the table above.
The measurement box retains the per-cell `bench_*.log` (bench-style stage
sections), `mem_*.txt` (0.5 s device samples), `deploy_*.yaml` (the config
actually served) and `img_*.png` under `/root/pro6000-vae-e2e-offline/`.

## How to reproduce

```bash
cd /app/vllm_omni
export PYTHONPATH=$PWD
python benchmarks/diffusion/mammoth_moda2_vae_e2e/run_e2e_offline.py \
    --model /root/models/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --out ~/pro6000-vae-e2e --sizes 1536,1024 --gpu 2
python benchmarks/diffusion/mammoth_moda2_vae_e2e/collect.py --out ~/pro6000-vae-e2e
```

Requires vLLM 0.30.0 (see `CHANGES.md`).
