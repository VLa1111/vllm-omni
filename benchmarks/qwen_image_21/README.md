# Qwen-Image-2.1 Edit Benchmarks

Single-GPU profiling and output validation for Qwen-Image-2.1 text-to-image and image edit
(0 / 1 / 4 reference images) on vLLM-Omni.

| File | Purpose |
| --- | --- |
| `profile_edit_matrix.py` | Runs the 0/1/4-reference cells for N waves in fresh processes. Records wall time, generation time, peak device memory, gate result and output hash. |
| `quality_gate.py` | Collapse detector: flags edit outputs that were re-rendered instead of edited. |

## Quick start

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/qwen_image_21/profile_edit_matrix.py \
    --model /path/to/Qwen-Image-2.1 --out-dir /tmp/qwen21_profile --waves 3

python benchmarks/qwen_image_21/quality_gate.py /tmp/qwen21_profile/*.png
```

The runner writes `summary.csv`, `summary.json`, `environment.txt`, one log and one image per
cell and wave, and a `sources/` directory with the T2I images the edit cells consume (generated
on first use, reused afterwards). `environment.txt` records the revision, interpreter and
versions, model path, flags and the sampled device, so a number is only meaningful next to it.

Default cells (1024x1024, 50 steps, `--cfg-scale 1.0`, i.e. true CFG off):

| Cell | Workload |
| --- | --- |
| `0ref` | T2I: "A ceramic teapot on a wooden table" |
| `1ref` | Edit: "Change the background to a moonlit beach with floating stars" |
| `4ref` | Edit: "Combine these four images into a single cozy still-life scene on a wooden table" |

## Baseline (recorded on 2026-10-09)

| Item | Value |
| --- | --- |
| vllm-omni | `3dc35694b3d458fc1461a8fe71832ac12e8c8cea` |
| vLLM / torch | 0.31.0 / 2.13.0+cu130 |
| Model | Qwen-Image-2.1, local snapshot `d26bb61231c349cf6b7896fa83353113880e1ba3` |
| Device | 1x RTX PRO 6000 Blackwell (sm_120), 97,887 MiB, driver 595.84 |
| Parallelism | ulysses=1, ring=1, tp=1, cfg_parallel=1 |
| Flags | `--color-format RGBA` (edits); no cache backend, no `--enforce-eager`, no CPU or layerwise offload, no VAE slicing/tiling |

3 waves per cell, same seeds in every wave:

| Cell | Wall s (engine start + generation) | Generation s | Peak MiB |
| --- | --- | --- | --- |
| `0ref` | 53 - 54 | 10.56 / 10.66 / 10.66 | 40,277 |
| `1ref` | 58 - 59 | 16.98 / 17.09 / 17.08 | 42,921 |
| `4ref` | 75 | 32.52 / 32.38 / 32.69 | 55,893 |

All 9 outputs were byte-identical across waves, and every output passed the gate
(corner Laplacian 12.8 - 19.9).

## Known trap: self-seed lock-in

Reusing the seed that generated the source image as the edit seed collapses the edit: the
sampler starts from the same initial noise as the source generation, re-renders the source and
drops the instruction (dense, over-saturated high-frequency texture). It is reproducible
bit-for-bit and identical in the diffusers reference implementation, so it is a property of the
protocol, not a vLLM-Omni defect.

| Source image (generated with) | Edit seed | Result |
| --- | --- | --- |
| self-generated T2I, seed 42 | 42 | collapse, 5/5 |
| self-generated T2I, seed 42 | 0 / 1 / 1234 / 7 | correct, 4/4 |
| self-generated T2I, generated with seed 7 | 42 | correct |
| self-generated T2I, seed 42, output at 1536x1536 | 42 | correct (different noise tensor shape) |
| `tests/assets/qwen_image_edit/qwen_image_edit_2511_test1.png` | 42 | correct |

Therefore: record the source generation seed next to the edit seed, never reuse it, and gate
every edit output. `profile_edit_matrix.py` refuses to run when `--edit-seed` equals
`--source-seed`.

## Validating an edit

- **Gate.** `quality_gate.py` measures the mean Laplacian standard deviation over the four
  160x160 corner patches, where the background lives and the collapse signal is concentrated
  (a full-image Laplacian does not separate). Calibration sample: healthy 12.8 - 22.7, collapsed
  26.7 - 47.0, threshold 25. This is a detector, not a quality certificate: it cannot tell an
  edit that followed the instruction from one that quietly did something else.
- **Reference parity.** Against diffusers 0.41.0 `QwenImage21Pipeline` with the same seed,
  vLLM-Omni matched at 45.47 dB (repo asset, 1 ref), 39.44 dB (1 ref) and 25.95 dB (4 ref,
  where the free background detail differs). Collapsed outputs compare at 46 - 48 dB because
  they are pinned to the source.
- **PSNR alone is not enough.** Cross-stack PSNR is highest exactly where the edit failed, so
  use it together with a gate and a visual check, never as the only edit-accuracy metric.
