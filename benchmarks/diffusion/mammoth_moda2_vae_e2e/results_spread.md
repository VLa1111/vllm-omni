# Sample count and per-cell spread for the end-to-end tables

The extraction `SPREAD_RUN.md` specifies, recorded on the PRO 6000 box.
Nothing new was measured: every sample below is a record already on disk
(batch-1 sweep `6873f69b8`, batch-4 sweep `1c87b798d`; the latter is verified by
`~/pro6000-vae-e2e-b4/run_env.txt`, the batch-1 output directory has no
`run_env.txt` and is taken from the run note).

Extraction ran at branch `pro6000/mammothmoda2-vae-e2e`, `HEAD = 0bebd0c34`,
vLLM 0.30.0, PyTorch 2.13.0+cu130, one RTX PRO 6000 Blackwell (97,887 MiB),
GPU index 2 — read-only, no GPU work.

## Run

The note's command points the first `--out` at `~/pro6000-vae-e2e`. On this box
that directory holds the **online (HTTP) benchmark** of 2026-10-05 05:0x —
1536 baseline and slicing only, in `benchmark_serving` output format, with no
`records` key — so the literal command stops on it:

```
$ python benchmarks/diffusion/mammoth_moda2_vae_e2e/spread.py \
      --out ~/pro6000-vae-e2e --out ~/pro6000-vae-e2e-b4
# /root/pro6000-vae-e2e
Traceback (most recent call last):
  File "/app/vllm_omni/benchmarks/diffusion/mammoth_moda2_vae_e2e/spread.py", line 116, in <module>
    main()
  File "/app/vllm_omni/benchmarks/diffusion/mammoth_moda2_vae_e2e/spread.py", line 84, in main
    records = measured_records(data["records"])
                               ~~~~~~~~~~~~~~~^
KeyError: 'records'
```

The tree the note describes — the eight batch-1 cells plus the two superseded
`4+1` `slicing-b4` blends, ten files and nothing else — is
`~/pro6000-vae-e2e-offline`. With that directory:

```bash
cd /app/vllm_omni
python benchmarks/diffusion/mammoth_moda2_vae_e2e/spread.py \
    --out ~/pro6000-vae-e2e-offline --out ~/pro6000-vae-e2e-b4 \
    2>&1 | tee ~/pro6000-spread.txt
```

## Output

```
# /root/pro6000-vae-e2e-offline
| Cell | Conc | Batch | n | E2E ms/image mean (min-max) | Stage 0 gen ms | Stage 1 gen ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1024_baseline | 1 | 1 | 5 | 105,404.1 (104,789.3-105,874.5) | 96,260.4 | 9,089.4 |
| 1024_both | 1 | 1 | 5 | 106,480.8 (106,081.4-107,238.5) | 97,216.2 | 9,191.7 |
| 1024_slicing | 1 | 1 | 5 | 105,724.2 (104,500.2-107,889.0) | 96,550.8 | 9,120.2 |
| 1024_tiling | 1 | 1 | 5 | 106,028.0 (104,286.4-107,676.5) | 96,858.7 | 9,115.8 |
| 1024_slicing-b4 | 4 | 4+1 | 2 | 70,947.3 (35,954.7-105,939.9) | 103,210.5 | 25,219.2 |
| 1536_baseline | 1 | 1 | 5 | 217,780.5 (216,798.2-218,651.0) | 193,029.6 | 24,642.7 |
| 1536_both | 1 | 1 | 5 | 215,329.5 (213,960.7-216,714.0) | 190,429.5 | 24,783.3 |
| 1536_slicing | 1 | 1 | 5 | 214,378.1 (212,859.0-215,336.6) | 189,610.4 | 24,664.0 |
| 1536_tiling | 1 | 1 | 5 | 218,680.6 (217,797.3-219,824.5) | 193,796.6 | 24,780.7 |
| 1536_slicing-b4 | 4 | 4+1 | 2 | 147,675.7 (78,622.6-216,728.8) | 203,686.4 | 74,396.5 |

Mixed-batch cells -- not whole waves, the mean is not per-request (superseded; ignore):
- 1024_slicing-b4 (batches 4+1)
- 1536_slicing-b4 (batches 4+1)

# /root/pro6000-vae-e2e-b4
| Cell | Conc | Batch | n | E2E ms/image mean (min-max) | Stage 0 gen ms | Stage 1 gen ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1024_baseline-b4 | 4 | 4 | 1 | 36,306.3 (36,306.3-36,306.3) | 104,579.1 | 32,681.2 |
| 1024_both-b4 | 4 | 4 | 1 | 36,497.1 (36,497.1-36,497.1) | 105,741.8 | 32,396.7 |
| 1024_slicing-b4 | 4 | 4 | 1 | 36,446.1 (36,446.1-36,446.1) | 105,410.9 | 32,443.8 |
| 1024_tiling-b4 | 4 | 4 | 1 | 35,992.6 (35,992.6-35,992.6) | 103,871.7 | 32,248.8 |
| 1536_baseline-b4 | 4 | 4 | 1 | 78,120.6 (78,120.6-78,120.6) | 204,983.6 | 86,563.4 |
| 1536_both-b4 | 4 | 4 | 1 | 77,624.8 (77,624.8-77,624.8) | 202,544.4 | 86,919.4 |
| 1536_slicing-b4 | 4 | 4 | 1 | 78,003.3 (78,003.3-78,003.3) | 204,345.0 | 86,684.7 |
| 1536_tiling-b4 | 4 | 4 | 1 | 77,710.7 (77,710.7-77,710.7) | 203,405.5 | 86,559.9 |

Paste the tables back: the min-max goes into the recipe's end-to-end tables as e.g. 217.8 (217.7-217.9).
```

On the two mixed rows: their deduplicated mean (70,947.3 / 147,675.7 ms) is
*not* the published mean either — `results.md` averages the five records
(49,951.8 / 106,243.8), the extractor averages wave + solo. Both are blends of a
wave member and a solo request; the `-b4` directory replaces them.

## Self-check against the published means

All 16 published means reproduce (largest difference 0.05 ms):

| Cell | n | Extracted mean ms | Published ms |
| --- | ---: | ---: | ---: |
| `1024_baseline` | 5 | 105,404.1 | 105,404.1 |
| `1024_both` | 5 | 106,480.8 | 106,480.8 |
| `1024_slicing` | 5 | 105,724.2 | 105,724.2 |
| `1024_tiling` | 5 | 106,028.0 | 106,028.0 |
| `1536_baseline` | 5 | 217,780.5 | 217,780.5 |
| `1536_both` | 5 | 215,329.5 | 215,329.5 |
| `1536_slicing` | 5 | 214,378.1 | 214,378.1 |
| `1536_tiling` | 5 | 218,680.6 | 218,680.6 |
| `1024_baseline-b4` | 1 | 36,306.3 | 36,306.3 |
| `1024_both-b4` | 1 | 36,497.1 | 36,497.1 |
| `1024_slicing-b4` | 1 | 36,446.1 | 36,446.2 |
| `1024_tiling-b4` | 1 | 35,992.6 | 35,992.6 |
| `1536_baseline-b4` | 1 | 78,120.6 | 78,120.6 |
| `1536_both-b4` | 1 | 77,624.8 | 77,624.8 |
| `1536_slicing-b4` | 1 | 78,003.3 | 78,003.3 |
| `1536_tiling-b4` | 1 | 77,710.7 | 77,710.7 |

The stage columns reproduce `results.md` / `results_b4.md` as well.

Three printed values differ in the last digit from the published tables —
`1024_slicing-b4` E2E 36,446.1 vs 36,446.2, `1024_slicing` stage 1 9,120.2 vs
9,120.1, `1536_slicing-b4` stage 1 86,684.7 vs 86,684.8. All three come from the
same records and are a formatting-path artifact: `collect.py` formats the bench
log's two-decimal line, this extractor formats the raw mean, and the underlying
values sit on the half-millisecond boundary (36,446.1456 / 9,120.1503 /
86,684.7456 ms). Not a different measurement.

## Finding: the batch-4 cells are one measured wave each — `n = 1`, no min-max

`SPREAD_RUN.md` counts the batch-4 cells as "four measured waves of four
concurrent requests, so `n = 4`". The data has **one** measured wave per cell,
so the extractor collapses each to a single sample and prints a degenerate
min-max (mean == min == max). Evidence:

- `FOLLOWUP_RUN.md` (the run note for that sweep) says so itself: "Four measured
  requests over a concurrency of four is exactly one measured wave (and four
  warmups are one warmup wave), so every row is wave-only."
- `~/pro6000-vae-e2e-b4/runs.txt`: `num_prompts: 4`, `num_warmups: 4`;
  `run_env.txt`: `--num-prompts 4 --num-warmups 4`.
- `console.log`: eight cells launched, eight `measured 4/4 done` lines — one
  measured wave per cell.
- Each raw file holds 4 warmup + 4 measured records, and the four measured
  records share one `wall_s` exactly (e.g. `1024_baseline-b4`: four records at
  36.306343965261476 s), i.e. four members of one wave, not four waves. The
  published mean is that single wave's wall time per image.

A spread for the batch-4 rows therefore cannot be produced from what is on
disk; it needs a new sweep with several measured waves per cell (e.g.
`--num-prompts 16 --num-warmups 4`: one warmup wave plus four measured waves,
~2.5-3 h for the eight cells, no new code — `run_wave` already divides each wave
by its size).

## Numbers for the recipe's `End-to-end` columns (ms / 1000)

Batch 1 carries a real five-sample min-max:

| Table | Row | End-to-end s |
| --- | --- | --- |
| 1536 batch 1 | baseline | 217.8 (216.8-218.7) |
| 1536 batch 1 | slicing | 214.4 (212.9-215.3) |
| 1536 batch 1 | tiling | 218.7 (217.8-219.8) |
| 1536 batch 1 | slicing + tiling | 215.3 (214.0-216.7) |
| 1024 batch 1 | baseline | 105.4 (104.8-105.9) |
| 1024 batch 1 | both | 106.5 (106.1-107.2) |
| 1024 batch 1 | slicing | 105.7 (104.5-107.9) |
| 1024 batch 1 | tiling | 106.0 (104.3-107.7) |

Batch 4 rows are single waves (`n = 1`), unchanged from the published values:
1536 78.1 / 77.7 / 78.0 / 77.6, 1024 36.3 / 36.0 / 36.4 / 36.5 s per image.
They can carry the sample count (`n = 1 measured wave`), not a min-max.

The recipe in this checkout (`recipes/MammothModa2/MammothModa2.md`) still has
the 1536 batch-1 and batch-4 tables as single values and no 1024 table; the
1024 table the run note calls committed is not in this checkout's copy, so the
edits are left to the PR branch.

## To record

- Script output: this file, and `~/pro6000-spread.txt` on the box (the literal
  command's failure is `/root/pro6000-spread-literal.txt`).
- Raw trees (`/root/pro6000-vae-e2e/raw` is the online tree named by the note;
  `/root/pro6000-vae-e2e-offline/raw` is the batch-1 tree, 8 cells + 2 blends;
  `/root/pro6000-vae-e2e-b4/raw`, 8 cells):

```
/root/pro6000-vae-e2e/raw:                                   (online, not used)
  1536_baseline.json  1536_slicing.json

/root/pro6000-vae-e2e-offline/raw:
  1024_baseline.json   1024_both.json   1024_slicing-b4.json
  1024_slicing.json    1024_tiling.json
  1536_baseline.json   1536_both.json   1536_slicing-b4.json
  1536_slicing.json    1536_tiling.json

/root/pro6000-vae-e2e-b4/raw:
  1024_baseline-b4.json  1024_both-b4.json  1024_slicing-b4.json
  1024_tiling-b4.json    1536_baseline-b4.json  1536_both-b4.json
  1536_slicing-b4.json   1536_tiling-b4.json
```

- `git rev-parse HEAD` of this checkout: `0bebd0c3483756cdd466e72faf760e8b09646302`;
  vLLM 0.30.0.
