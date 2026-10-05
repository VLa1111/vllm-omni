# 1536x1536 batch-4 wave images (to fill the PSNR column of the E2E batch-4 rows)

Box: one NVIDIA RTX PRO 6000 Blackwell (97,887 MiB), GPU index 2.
Branch `pro6000/mammothmoda2-vae-e2e` @ `381b1c1d4`.
Command (one fresh engine per cell):

```bash
PYTHONPATH=$PWD python benchmarks/diffusion/mammoth_moda2_vae_e2e/run_e2e_offline.py \
    --model /root/models/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --out <outdir> --sizes 1536 --gpu 2 \
    --configs baseline-b4,tiling-b4,slicing-b4,both-b4 \
    --num-prompts 4 --num-warmups 4
```

A wave is 4 concurrent requests that share prompt and seed; the runner saves the
**first output of the last measured wave** (`outputs[0]`), so every cell saves
wave member 0.

## run1 — the requested matrix (`/root/pro6000-vae-e2e-b4img`)

| Size | Config | Slice | Tile | Conc | Peak MiB | E2E mean ms | Stage 0 gen ms | Stage 1 gen ms | PSNR dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1536x1536 | baseline-b4 | no | no | 4 | 87078 | 78414.4 | 205919.8 | 86786.2 | - |
| 1536x1536 | both-b4 | yes | yes | 4 | 66020 | 77565.8 | 202071.3 | 87140.8 | 29.01 |
| 1536x1536 | slicing-b4 | yes | no | 4 | 67200 | 77343.3 | 201649.6 | 86746.7 | identical |
| 1536x1536 | tiling-b4 | no | yes | 4 | 74090 | 78175.1 | 205141.7 | 86642.2 | 47.10 |

PSNR is against `img_1536_baseline-b4.png`; "identical" is byte-identical.

Peaks and e2e reproduce the published batch-4 cells: 87,078 / 74,090 / 67,200 /
66,020 MiB and 78.1 / 77.7 / 78.0 / 77.6 s per image (measured here: 78.4 /
78.2 / 77.3 / 77.6; all within 0.9 %).

## Reproducibility — two re-runs of the same command

`recheck` re-ran `tiling-b4,both-b4` (`/root/pro6000-vae-e2e-b4img-recheck`);
`full2` re-ran the full four-cell matrix (`/root/pro6000-vae-e2e-b4img-full2`).
Image md5 and PSNR vs the untiled baseline:

| cell | run1 | recheck | full2 |
| --- | --- | --- | --- |
| baseline-b4 | a3bc2a7f · — | — | a3bc2a7f · — |
| tiling-b4 | e02b3a8f · 47.10 dB | e02b3a8f · 47.10 dB | 9547eab6 · **29.01 dB** |
| slicing-b4 | a3bc2a7f · identical | — | a3bc2a7f · identical |
| both-b4 | 9547eab6 · **29.01 dB** | e02b3a8f · 47.10 dB | e02b3a8f · 47.10 dB |

Peaks (87,078 / 74,090 / 67,200 / 66,020 MiB) and e2e (78.2–78.9 s per image)
are the same in all three runs.

Exactly **three** distinct images exist across the batch-4 cells of the three
runs (the four PNGs committed next to this file are ~10.7 MB; baseline and
slicing are byte-identical copies):

| md5 | what | how to get it |
| --- | --- | --- |
| `a3bc2a7f546394f97f0e8fe46598258e` | untiled decode; baseline and slicing are byte-identical | `img_1536_baseline-b4.png` == `img_1536_slicing-b4.png` |
| `e02b3a8fe05dff8e63c1caad2f29715b` | tiled decode, mode **T** (47.10 dB vs untiled) | `img_1536_tiling-b4.png` |
| `9547eab62178b37467d6697f9fc52479` | tiled decode, mode **X** (29.01 dB vs untiled) | `img_1536_both-b4.png` |

## Finding: the tiled batch-4 decode is bimodal across engine processes

- `slicing-b4` is byte-identical to `baseline-b4` in every run — "identical" is
  safe to state.
- `tiling-b4` and `both-b4` each produced both T and X (T in 4 of 6 tiled
  cell-runs, X in 2 of 6). The flag combination does **not** determine which:
  T appeared for tiling-only (run1, recheck) and for combined (recheck, full2);
  X appeared for combined (run1) and for tiling-only (full2).
- Both outcomes are byte-stable per process, and the committed images show the
  same scene with edge/fine-detail differences — this is not a wrong member
  being saved (see the runner note above) and not run-to-run drift in the
  untiled path.
- The decode-only benchmark stays bit-stable across its own runs: 1536 batch 4
  `slicing+tiling` = `tiling` = 55.35 dB vs the flags-off row (and identical
  max-abs 0.0938), so the bimodality is specific to the E2E engine process,
  not to the tiled decode as measured on seeded latents.
- Consequence for the PR body: the batch-4 tiled rows cannot carry one fixed
  PSNR. Suggested wording: slicing `identical`; tiling and combined ~47.1 dB,
  with a footnote that the tiled batch-4 capture is not bit-reproducible
  across engine processes (a second stable output, 29.0 dB, appears in some
  runs), while the deterministic decode-only benchmark gives 55.4 dB with
  combined == tiling.
