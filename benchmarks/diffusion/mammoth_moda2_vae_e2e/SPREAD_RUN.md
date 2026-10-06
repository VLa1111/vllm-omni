# Spread follow-up: sample count and per-cell min-max for the end-to-end tables

The 2026-10-06 review of PR #7774 (MrlixiangWE) asked the end-to-end tables for
the sample count and the per-cell min-max "as in the decode tables", and for the
1024x1024 figures to have a table of their own. The 1024 table is committed with
its means; what is missing is the min-max -- in both 1536 tables and in the 1024
table. This note specifies the extraction.

Unlike `FOLLOWUP_RUN.md` this is **not a GPU run**: every measured sample is
already on disk. `run_e2e_offline.py` keeps one record per measured request in
`<out>/raw/<size>_<config>.json`, with `wall_s` = wave wall time / wave size and
the per-request `stage_durations`; the published means in `results.md` and
`results_b4.md` were computed from those same records.

## Run

On the PRO 6000 box, in the checkout the sweeps ran from (`/app/vllm_omni`).
It reads two JSON trees and prints two tables -- stdlib only, no GPU, under a
minute:

```bash
cd /app/vllm_omni
python benchmarks/diffusion/mammoth_moda2_vae_e2e/spread.py \
    --out ~/pro6000-vae-e2e --out ~/pro6000-vae-e2e-b4 2>&1 | tee ~/pro6000-spread.txt
```

Point it at exactly those two directories: they are the sweeps whose numbers are
published (`results.md`, `results_b4.md`). Do **not** add the `-b4img*`
re-captures -- they re-measure some of the same 1536 batch-4 cells under the
image/PSNR protocol (`results_b4img.md`), so their rows would duplicate cells
with slightly different means.

## How the samples are counted

- Batch-1 cells: five measured single requests, so `n = 5`, one sample each.
- Batch-4 cells: four measured waves of four concurrent requests; the four
  members of a wave share one `wall_s`, so `n = 4`, one sample per wave -- the
  amortized per-image latency the table's `s/image` column is the mean of.
- The two `slicing-b4` rows in the **first** directory are the superseded
  five-prompt protocol (one wave of four plus one solo request, `Batch` reads
  `4+1`). The script flags them under the table; ignore them -- the `-b4`
  directory replaces both.

## Expected output -- self-check against the published means

16 current cells. If a row is missing, that `raw/<tag>.json` is gone; say so
rather than reading the min-max off anything else.

| Dir | Cell | Conc | n | Published mean ms |
| --- | --- | ---: | ---: | ---: |
| `~/pro6000-vae-e2e` | `1024_baseline` | 1 | 5 | 105,404.1 |
| `~/pro6000-vae-e2e` | `1024_both` | 1 | 5 | 106,480.8 |
| `~/pro6000-vae-e2e` | `1024_slicing` | 1 | 5 | 105,724.2 |
| `~/pro6000-vae-e2e` | `1024_tiling` | 1 | 5 | 106,028.0 |
| `~/pro6000-vae-e2e` | `1536_baseline` | 1 | 5 | 217,780.5 |
| `~/pro6000-vae-e2e` | `1536_both` | 1 | 5 | 215,329.5 |
| `~/pro6000-vae-e2e` | `1536_slicing` | 1 | 5 | 214,378.1 |
| `~/pro6000-vae-e2e` | `1536_tiling` | 1 | 5 | 218,680.6 |
| `~/pro6000-vae-e2e-b4` | `1024_baseline-b4` | 4 | 4 | 36,306.3 |
| `~/pro6000-vae-e2e-b4` | `1024_both-b4` | 4 | 4 | 36,497.1 |
| `~/pro6000-vae-e2e-b4` | `1024_slicing-b4` | 4 | 4 | 36,446.2 |
| `~/pro6000-vae-e2e-b4` | `1024_tiling-b4` | 4 | 4 | 35,992.6 |
| `~/pro6000-vae-e2e-b4` | `1536_baseline-b4` | 4 | 4 | 78,120.6 |
| `~/pro6000-vae-e2e-b4` | `1536_both-b4` | 4 | 4 | 77,624.8 |
| `~/pro6000-vae-e2e-b4` | `1536_slicing-b4` | 4 | 4 | 78,003.3 |
| `~/pro6000-vae-e2e-b4` | `1536_tiling-b4` | 4 | 4 | 77,710.7 |

Plus two rows to ignore: `1024_slicing-b4` and `1536_slicing-b4` in
`~/pro6000-vae-e2e` (the superseded blends above).

## If a directory is gone

`raw/` is the only record of the individual samples; `results.md` has means
only. If one of the two trees was cleaned up, the honest fallback is to re-run
that sweep and say so -- `README.md` ("What to run", first sweep, ~3 h) and
`FOLLOWUP_RUN.md` (wave run, ~1.5-2 h) have the exact commands. Do not estimate
a min-max from the mean.

## To record

- The script's full output: both tables and the notes under them.
- `ls ~/pro6000-vae-e2e/raw ~/pro6000-vae-e2e-b4/raw`, so a missing cell is
  visible rather than silently absent from the table.
- `git rev-parse HEAD` and the vLLM version of this checkout, for the
  environment block (the raw files come from the sweeps at `6873f69b8`, batch 1,
  and `1c87b798d`, batch 4; this script is read-only).

## Where the numbers go

- `recipes/MammothModa2/MammothModa2.md`: the `End-to-end s/image` column of
  the two 1536 tables and the 1024 table, as `217.8 (217.7-217.9)` -- the
  script prints ms, the recipe column is ms/1000.
- The same three tables in the PR description.
- The reply to the `MammothModa2.md:501` review thread (R3), which currently
  carries a TODO for this run.
