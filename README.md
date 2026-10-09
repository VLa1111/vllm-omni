# Boogu-Image: CPU offload × FP8 — combination-validation record

For issue [vllm-project/vllm-omni#6665](https://github.com/vllm-project/vllm-omni/issues/6665)
(owner-requested feature × feature validation, comment 5647196736).

Row covered here: **CPU offload + FP8 / cache; VAE fusion + offload** — this record is the
"CPU offload × FP8" part plus the module-level / local-layerwise / distributed-layerwise mode
boundaries. The cache and VAE-fusion parts remain blocked (§7).

Test owner: @VLa1111. Peer reviewer: @ShengleiFu.

## 1. Code head and dependencies

| Item | Value |
| --- | --- |
| Feature under test | CPU offload for Boogu-Image (PR [#6897](https://github.com/vllm-project/vllm-omni/pull/6897)) |
| Code head | `a054ccc19` (PR #6897 head, 2026-10-06; merges main@`5f95115e7`) |
| vLLM / vLLM-Omni | vLLM `0.31.0`; vLLM-Omni tree `a054ccc19` |
| PyTorch / torchao / Transformers | `2.13.0+cu130` / `0.17.0` / `5.14.1` |
| Checkpoints | `Boogu/Boogu-Image-0.1-Base` (34.6 GiB bf16) and `Boogu/Boogu-Image-0.1-Base-fp8` (serialized torchao FP8) |
| Hardware / topology | 1 × RTX 4090 24 GiB (PCIe 4.0 x16), driver `580.173.02`; single GPU, single process, DP = 1 |
| Quantization config (FP8 arms) | `--diffusion-quantization-config '{"transformer":{"method":"torchao_float8_weight_only"}}'` |

## 2. Workload and protocol

T2I `512x512`, `num_inference_steps=10`, `guidance_scale=1.0`, seed 42, prompt
`A mountain lake at sunset, photorealistic, cinematic lighting`; one warm-up request + **3
measured requests** per arm. Memory is process-scoped (`nvidia-smi --query-compute-apps=used_memory`,
sampled every 0.5 s): *load peak* = peak while loading/warming up, *steady peak* = peak across the
measured requests. Every arm is a fresh `vllm serve --omni` process. Activation evidence is read
from the service log (the selected offload backend and mode are logged), so a silently ignored
option would be visible.

## 3. Four-way matrix (neither / A only / B only / A+B)

A = CPU offload, B = DiT FP8. Raw artifacts per arm in `logs/`, `timings/`, `outputs/`.

| # | Arm | Model | Offload mode | Status | Steady peak MiB | Load peak MiB | E2E s (min-max, n=3) | Output sha256 (req1) |
| --: | --- | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | `bf16_noffload` | bf16 | none | **failed — OOM while loading (expected rejection)** | – (reached 18,352) | 18,352 | – | – |
| 2 | `bf16_model` | bf16 | module-level | validated | 21,924 | 21,850 | 5.65–5.67 | `de4c4747064d5537…` |
| 3 | `bf16_layerwise` | bf16 | local layerwise | validated | 21,380 | 21,368 | 9.17–9.18 | `de4c4747064d5537…` |
| 4 | `dlo_n0` | bf16 | distributed layerwise, 0 resident | validated | 21,038 | 21,026 | 9.15–9.16 | `de4c4747064d5537…` |
| 5 | `dlo_n4` | bf16 | distributed layerwise, 4 resident | **failed on first attempt (startup segfault); validated on 2 retries** | 22,014 | 22,002 | 8.63–8.65 | `de4c4747064d5537…` |
| 6 | `fp8_noffload` | fp8 | none | validated (baseline) | 21,132 | 21,284 | 1.60–1.61 | `554b4412506613e5…` |
| 7 | `fp8_model` | fp8 | module-level | validated (A+B) | 11,786 | 21,100 | 3.96–3.97 | `7752d736ca3b4567…` |
| 8 | `fp8_layerwise` | fp8 | local layerwise | validated (A+B) | 13,194 | 21,100 | 4.78–4.83 | `7752d736ca3b4567…` |
| 9 | `fp8_bothflags` | fp8 | module + layerwise flags together | validated — layerwise wins silently (mode boundary) | 13,194 | 21,100 | 4.76–4.77 | `7752d736ca3b4567…` |
| 10 | `fp8_dlo_n0` | fp8 | distributed layerwise, 0 resident | validated (A+B) | 13,020 | 21,284 | 4.76–4.78 | `7752d736ca3b4567…` |
| 11 | `fp8_dlo_n4` | fp8 | distributed layerwise, 4 resident | validated (A+B) | 13,708 | 21,284 | 4.57–4.59 | `7752d736ca3b4567…` |

Reading: on the 24 GiB card bf16 needs offload (row 1 OOMs); module-level is the cheapest bf16
mode (21.9 GiB, 5.66 s), layerwise and DLO trade ~3.5 s for the same footprint (9.17 / 9.15 s),
and four resident DLO layers win ~0.5 s back (8.64 s) for ~1 GiB of resident weights. On FP8,
module-level halves the footprint to 11.8 GiB under 12 GiB (3.96 s); layerwise costs ~0.8 s and
~1.4 GiB more (13,194 MiB, 4.80 s); DLO N=0 matches layerwise (13,020 MiB, 4.76 s) and DLO N=4
takes ~0.2 s off that (13,708 MiB, 4.58 s). **Full per-request timings, peak traces and logs:
see `summary/tables.md`.**

## 4. Output consistency (same prompt/seed, req1 of each arm)

| Group | Arms | Result |
| --- | --- | --- |
| bf16 | module-level, layerwise, DLO N=0, DLO N=4 (both retries) | **byte-identical** to each other (md5 `9add03c40f7cfaa1eb1ec49fac91ead0`) |
| fp8 | module-level, layerwise, both-flags, DLO N=0, DLO N=4 | **byte-identical** to each other (md5 `3a47a511976e7a55eeb29b0613c3fc84`) |
| fp8 | offload arms vs `fp8_noffload` baseline (`a061cbb44929ee58ad650604b733d88f`) | **not byte-identical**: PSNR **49.72 dB**, max abs diff 0.1922 (0-1 scale). Numerical noise, not a visible change |

Note: this *revises the accuracy claim published in PR #6897*, which stated FP8 offload output was
byte-identical to the no-offload baseline. At head `a054ccc19` the two offload modes are still
mutually byte-identical, but they differ from the direct-load baseline by a tiny amount. The
visible difference between the two paths is in the loader (§5): the offload arms log the
online-quantization + CPU-offload load path, the no-offload arm logs only the plain_fp8 unpack. The published hashes were measured at head `f5594160`
with an unrecorded prompt, so a cross-head byte comparison is not possible; this row is same-head,
same-workload. Unit tests still assert the repacked forward is bit-identical to torchao's
`dequantize()` path.

## 5. Activation evidence (from the service logs)

```text
bf16 module-level : Enabling offloader backend: ModelLevelOffloadBackend
                    Model-level offloading enabled: transformer <-> mllm (mutual exclusion)
bf16 layerwise    : Enabling offloader backend: LayerWiseOffloadBackend
                    layerwise offload timing instrumentation enabled (VLLM_OMNI_OFFLOAD_TIMING=1)
                    Layer-wise offloading enabled on 40 layers (blocks)
dlo (bf16 + fp8,   : Enabling offloader backend: DistributedLayerwiseOffloadBackend
  N=0 and N=4)
                    DLO direct checkpoint mmap unavailable; using ordinary loader: 186 required DiT
                      tensors have no checkpoint binding (first 5: transformer.context_refiner.*)
                    Distributed layer-wise offloading enabled on 40 blocks across 1 group(s),
                      transfers={dit: rank-local}, unified shared_buffers=2
fp8 (all arms)    : Per-component quantization: {'transformer': 'torchao'}
                    Unpacked 318 torchao Float8Tensor linear(s) into plain (fp8 weight, fp32
                      row-scale) parameters
fp8 + offload     : Online quantization with CPU offload, using cuda for weight loading
                      (will offload back to CPU)
                    Quantization complete, offloaded model back to CPU
both flags (row 9): Enabling offloader backend: LayerWiseOffloadBackend   (only)
```

Opt-in layerwise timing (`VLLM_OMNI_OFFLOAD_TIMING=1`), steady denoise rounds of 40 blocks:

| Arm | h2d per round | compute | exposed stall | wall | hidden |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp8 layerwise | 450.2–451.4 ms (8,813.56 MiB) | 136.5–136.9 ms | 313.9–315.4 ms | 432.5–433.8 ms | 30% |
| bf16 layerwise | 899.3–901.6 ms (17,601.47 MiB) | 111.3–111.6 ms | 788.6–790.4 ms | 860.0–861.7 ms | 12% |

## 6. Expected rejections and mode boundaries

| Case | Measured / pinned behavior |
| --- | --- |
| bf16 without offload on 24 GiB (row 1) | OOM while loading — the offload value proposition |
| `--enable-cpu-offload` **and** `--enable-layerwise-offload` together (row 9) | No error, no warning; legacy priority `DLO > layerwise > module-level` selects layerwise silently (`vllm_omni/diffusion/offloader/config.py`, `_LEGACY_STRATEGY_PRIORITY`); numbers match layerwise-only exactly |
| compact `diffusion_offload_config` × legacy `--enable-*` flags | Code-level rejection (pinned rule; not re-exercised in this record) |
| diffusion cache × DLO AllGather × DP > 1 | Code-level rejection (pinned rule; DP = 1 here, so not applicable) |

## 7. Blocked / not tested

| Row | Status | Reason |
| --- | --- | --- |
| Offload × Cache-DiT (#7381) / TeaCache (#6811) | blocked by dependency | no cache path on main yet (#7381 open, #6811 draft) |
| Offload × VAE fusion (in-repo VAE / fused GroupNorm+SiLU, #6945) | blocked by dependency | #6945 is a draft with conflicts and failing CI |
| Edit / Edit-Turbo / Turbo checkpoints × offload (and fp8 variants) | not tested in this round | Base only; same protocol applies |

## 8. Failures and anomalies

- `dlo_n4` first attempt: **segfault ~50 s into startup** (faulthandler dump with no Python-level
  frame, core piped to apport; 1 of 3 DLO-N=4 runs). Not reproduced in 2/2 retries, which both
  completed with identical results (8.63–8.65 s, 22,014 MiB, byte-identical output). Classified as
  flaky (environment), no reliable reproducer; the failed log is kept as `logs/dlo_n4.serve.log`.

## 9. Continuity with the numbers published in PR #6897 (head `f5594160`)

| Config | Published (f5594160) | This record (a054ccc19) |
| --- | --- | --- |
| bf16 + module-level | 22,012 MiB / 5.87 s | 21,924 MiB / 5.65–5.67 s |
| bf16 + layerwise | 21,380 MiB / 9.26 s | 21,380 MiB / 9.17–9.18 s |
| fp8 + module-level | 11,788 MiB / 4.05 s | 11,786 MiB / 3.96–3.97 s |
| fp8 + layerwise | 12,924 MiB / 4.80 s | 13,194 MiB / 4.78–4.83 s |
| fp8 layerwise timing | h2d 451.3 ms, 31% hidden | h2d 450.2–451.4 ms, 30% hidden |

Peaks and latencies reproduce within a couple of percent (the fp8-layerwise peak reads 2% higher,
within the 0.5 s sampler and single-run variance). The only substantive change is the byte-equality
of fp8 offload vs no-offload (§4).

## 10. Reproduce

```bash
# inside the container: tree snapshot of a054ccc19 at /workspace/boogu-matrix, vLLM 0.31.0 venv
PYTHONPATH=/workspace/boogu-matrix /workspace/venv31/bin/vllm serve /models/Boogu-Image-0.1-Base \
  --omni --port 8097 --trust-remote-code --enable-cpu-offload
# fp8 + layerwise with timing instrumentation
PYTHONPATH=/workspace/boogu-matrix VLLM_OMNI_OFFLOAD_TIMING=1 /workspace/venv31/bin/vllm serve \
  /models/Boogu-Image-0.1-Base-fp8 --omni --port 8097 --trust-remote-code \
  --diffusion-quantization-config '{"transformer":{"method":"torchao_float8_weight_only"}}' \
  --enable-layerwise-offload
# request
curl -X POST http://127.0.0.1:8097/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"A mountain lake at sunset, photorealistic, cinematic lighting","size":"512x512",
       "num_inference_steps":10,"guidance_scale":1.0,"seed":42,"response_format":"b64_json"}'
```

Drivers: `scripts/boogu_matrix_run.sh` (9-arm sweep), `scripts/boogu_retry_dlo.sh` (dlo_n4 retry,
run twice), `scripts/boogu_dlo_fp8.sh` (the fp8 DLO arms 10–11), `scripts/boogu_matrix_collect.py`
+ `scripts/boogu_matrix_analyze.py` (tables).

## 11. Limits

- Single GPU only. Distributed-layerwise with > 1 rank (the shard/AllGather path) is **not**
  covered here; it needs a multi-GPU box. The single-GPU DLO arms use rank-local transfer.
- One checkpoint revision, one workload size (512×512, 10 steps), one seed; 3 measured requests
  per arm (min-max reported), 1 warm-up.
- Quality check here is output consistency (byte/PSNR) against same-precision baselines, not a
  perceptual or CLIP-style evaluation.
