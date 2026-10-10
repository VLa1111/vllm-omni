# Qwen-Image-2.1 Edit Benchmarks

Single-GPU profiling and output validation for Qwen-Image-2.1 text-to-image and image edit
(0 / 1 / 4 reference images) on vLLM-Omni.

| File | Purpose |
| --- | --- |
| `profile_edit_matrix.py` | Runs the 0/1/4-reference cells for N waves at one or more resolutions, optionally batched, in fresh processes. Records wall time, generation time, peak device memory, gate result and output hashes. |
| `quality_gate.py` | Collapse detector: flags edit outputs that were re-rendered instead of edited. |

## Quick start

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/qwen_image_21/profile_edit_matrix.py \
    --model /path/to/Qwen-Image-2.1 --out-dir /tmp/qwen21_profile --waves 3

# native resolution, and the batching cell (flag name differs per example)
CUDA_VISIBLE_DEVICES=0 python benchmarks/qwen_image_21/profile_edit_matrix.py \
    --model /path/to/Qwen-Image-2.1 --out-dir /tmp/qwen21_2048 --resolution 1024,2048 --waves 3
CUDA_VISIBLE_DEVICES=0 python benchmarks/qwen_image_21/profile_edit_matrix.py \
    --model /path/to/Qwen-Image-2.1 --out-dir /tmp/qwen21_batch4 \
    --num-outputs-per-prompt 4 --waves 3

python benchmarks/qwen_image_21/quality_gate.py /tmp/qwen21_profile/*.png
```

The runner writes `summary.csv`, `summary.json`, `environment.txt`, one log and per-cell wave
images, and a `sources/r<resolution>/` directory with the T2I images the edit cells consume
(generated on first use, reused afterwards). With `--num-outputs-per-prompt N` each run
produces `N` images (`{stem}_{idx}.png`) and the summary records `n_images`, `n_distinct_md5`
and the gate range over the batch. `environment.txt` records the revision, interpreter and
versions, model path, flags and the sampled device, so a number is only meaningful next to it.
The runner exits `1` if any cell failed or was flagged by the gate; a flagged cell still needs
a visual check (see [Validating an edit](#validating-an-edit)).

Default cells (50 steps, `--cfg-scale 1.0`, i.e. true CFG off):

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

### Native resolution (recorded on 2026-10-10)

Same settings at 2048x2048, same 3 waves:

| Cell | Wall s | Generation s | Peak MiB |
| --- | --- | --- | --- |
| `0ref` | 101.9 - 102.7 | 57.19 / 57.56 / 57.52 | 66,547 |
| `1ref` | 114.1 - 115.6 | 69.40 / 69.48 / 69.40 | 70,985 |
| `4ref` | 142.8 - 143.8 | 97.62 / 97.86 / 98.49 | 78,541 |

Bit-identical across waves, all passed the gate (18.04 / 5.95 / 11.24). Quadrupling the pixel
count costs 5.4x / 4.1x / 3.0x generation time and 1.7x / 1.7x / 1.4x memory versus 1024; the
worst cell still leaves ~19 GiB free on a 96 GiB card.

### Batching (recorded on 2026-10-10)

1024x1024, `--num-outputs-per-prompt 4`, 3 waves. Generation time is for all four images:

| Cell | Wall s | Generation s (w1 / w2 / w3) | Per image s | Peak MiB | Gate range |
| --- | --- | --- | --- | --- | --- |
| `0ref` | 96.6 - 104.5 | 58.86 / 50.96 / 51.03 | 12.7 - 14.7 | 63,764 | 12.58 - 24.16, ok |
| `1ref` | 104.7 - 113.0 | 68.35 / 60.60 / 60.52 | 15.1 - 17.1 | 75,356 | 19.42 - 25.16, flagged&sup1; |
| `4ref` | 136.3 - 152.3 | 106.61 / 92.55 / 91.39 | 22.8 - 26.7 | 96,676 | 8.40 - 13.08, ok |

All 36 outputs were bit-identical across waves and `n_distinct_md5 = 4` for every cell, so the
four samples per prompt are genuinely distinct and reproducible.

Within each cell the first wave runs 8 - 15% slower than waves 2 and 3 (this run shared the
host with a second job during its first two waves), so compare the steady-state waves.

Batching is not free: steady-state total generation time is ~3.0 - 3.3x a single output, so
per-image time only improves on the heavier cells (`4ref` -29%, `1ref` -11%) and gets 19%
worse for `0ref` (10.66 s single). Peak memory grows 1.5 - 1.8x, and `4ref` reaches
**96,676 MiB of the card's 97,887 MiB** -- 4 is the practical maximum for a 4-reference edit at
1024x1024 on this device before OOM, with no headroom for a larger batch.

&sup1; Adjudicated healthy by inspection; the gate threshold was tripped by legitimate
high-frequency background detail. See below.

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
- **False positives are expected, so a flagged output needs a visual check.** Any edit that
  fills the corners with high-frequency detail (wood grain, a star field, glitter) raises the
  metric on its own. In the batching run the `1ref` cell was flagged in all three waves
  (max 25.16 vs threshold 25) and every flagged image turned out correct -- the prompt asks for
  "floating stars". The same run shows how wide the healthy band is within one batch for one
  prompt and one seed: 12.58 - 24.16. With healthy outputs observed up to 25.16 and the
  lowest collapsed output at 26.7, the separation margin is ~1.5 units: treat the threshold as
  a review trigger, and judge the flagged image by eye.
- **Reference parity.** Against diffusers 0.41.0 `QwenImage21Pipeline` with the same seed,
  vLLM-Omni matched at 45.47 dB (repo asset, 1 ref), 39.44 dB (1 ref) and 25.95 dB (4 ref,
  where the free background detail differs). Collapsed outputs compare at 46 - 48 dB because
  they are pinned to the source.
- **PSNR alone is not enough.** Cross-stack PSNR is highest exactly where the edit failed, so
  use it together with a gate and a visual check, never as the only edit-accuracy metric.
