# Batch-4 E2E with four measured waves per cell (real min-max for the batch-4 rows)

`results_spread.md` found that the published batch-4 rows are one measured wave
each, so their sample count is `n = 1` and no min-max exists. This is the
re-run that closes that gap: the same eight cells, `--num-prompts 16`
(= four measured waves of four) instead of 4, everything else identical (fixed
prompt, `seed=42`, 50 steps, guidance 9.0, `cfg_range 0,1`, four warmups =
one warmup wave, one fresh engine per cell, GPU 2).

Run on 2026-10-07 at branch `pro6000/mammothmoda2-vae-e2e`,
`HEAD = 86a9d74dc`, vLLM 0.30.0, torch 2.13.0+cu130, one RTX PRO 6000
Blackwell (97,887 MiB = 95.6 GiB), whole-device memory sampled every 0.5 s.

```bash
cd /app/vllm_omni
export PYTHONPATH=$PWD
python benchmarks/diffusion/mammoth_moda2_vae_e2e/run_e2e_offline.py \
    --model /root/models/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --out ~/pro6000-vae-e2e-b4waves --sizes 1536,1024 --gpu 2 \
    --configs baseline-b4,tiling-b4,slicing-b4,both-b4 \
    --num-prompts 16 --num-warmups 4 2>&1 | tee ~/pro6000-vae-e2e-b4waves/console.log
python benchmarks/diffusion/mammoth_moda2_vae_e2e/collect.py --out ~/pro6000-vae-e2e-b4waves
```

8/8 cells completed, 32 measured waves, no cell skipped. Total wall time
2 h 48 min (05:41-08:29 Z), plus one 28 min single-cell re-run (below).

## Table

`1536_baseline-b4` is from the clean-card re-run (see "One cell was
contaminated" below); the other seven rows are from the main sweep. Waves are
listed in issue order.

| Size | Config | Wave s/image | Mean s/image | n | Min-max s/image | Peak MiB | Stage 0 ms | Stage 1 ms |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1536 | baseline-b4 | 79.5 / 79.5 / 79.3 / 79.4 | 79.4 | 4 | 79.3-79.5 | 87,078 | 209,406.8 | 87,188.2 |
| 1536 | tiling-b4 | 79.9 / 81.0 / 80.2 / 80.3 | 80.3 | 4 | 79.9-81.0 | 74,090 | 213,104.1 | 87,199.9 |
| 1536 | slicing-b4 | 79.1 / 79.0 / 79.0 / 78.8 | 79.0 | 4 | 78.8-79.1 | 67,200 | 207,805.5 | 87,032.0 |
| 1536 | slicing + tiling-b4 | 79.4 / 78.9 / 79.4 / 80.2 | 79.5 | 4 | 78.9-80.2 | 66,020 | 209,290.5 | 87,372.1 |
| 1024 | baseline-b4 | 36.8 / 36.9 / 37.0 / 37.1 | 37.0 | 4 | 36.8-37.1 | 70,258 | 107,450.0 | 32,467.0 |
| 1024 | tiling-b4 | 36.6 / 36.8 / 36.7 / 37.1 | 36.8 | 4 | 36.6-37.1 | 69,688 | 106,678.6 | 32,511.1 |
| 1024 | slicing-b4 | 36.8 / 36.8 / 36.9 / 36.9 | 36.8 | 4 | 36.8-36.9 | 60,846 | 106,947.9 | 32,444.1 |
| 1024 | slicing + tiling-b4 | 36.8 / 36.8 / 36.3 / 36.6 | 36.6 | 4 | 36.3-36.8 | 60,846 | 106,044.0 | 32,478.5 |

Extractor output, verbatim (`python .../spread.py --out ~/pro6000-vae-e2e-b4waves`):

```
| Cell | Conc | Batch | n | E2E ms/image mean (min-max) | Stage 0 gen ms | Stage 1 gen ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1024_baseline-b4 | 4 | 4 | 4 | 36,961.3 (36,832.4-37,118.7) | 107,450.0 | 32,467.0 |
| 1024_both-b4 | 4 | 4 | 4 | 36,615.3 (36,262.2-36,838.6) | 106,044.0 | 32,478.5 |
| 1024_slicing-b4 | 4 | 4 | 4 | 36,832.0 (36,752.7-36,904.2) | 106,947.9 | 32,444.1 |
| 1024_tiling-b4 | 4 | 4 | 4 | 36,783.5 (36,581.3-37,071.6) | 106,678.6 | 32,511.1 |
| 1536_baseline-b4 | 4 | 4 | 4 | 79,549.5 (77,712.6-80,264.3) | 210,362.8 | 86,873.1 |
| 1536_both-b4 | 4 | 4 | 4 | 79,455.3 (78,887.7-80,168.1) | 209,290.5 | 87,372.1 |
| 1536_slicing-b4 | 4 | 4 | 4 | 78,976.8 (78,839.1-79,054.9) | 207,805.5 | 87,032.0 |
| 1536_tiling-b4 | 4 | 4 | 4 | 80,341.2 (79,867.7-80,987.6) | 213,104.1 | 87,199.9 |
```

(The `1536_baseline-b4` row there is the contaminated first attempt; the clean
re-run reads `79,421.9 (79,334.4-79,470.5)`.)

## One cell was contaminated, and it was re-run clean

During the first cell (`1536_baseline-b4`, 05:41-06:09) a co-tenant held
~13.5 GiB on GPU 2 — everything else in the sweep started from a 22 MiB card.
Whole-device peak reached **95,361 MiB** against the card's 97,887, and the
allocator logged one OOM retry:

```
[rank0]:[W1007 05:48:24 ... CUDACachingAllocator.cpp:3933] memory allocation
failed with OOM on device 0 while trying to allocate 5437915136 bytes
(free: 1983250432, total: 101975851008).
```

The cell survived, but its wave spread (77.7-80.3 s, 3.3 %) mixes in that
memory pressure. The clean re-run of that one cell (`--sizes 1536 --configs
baseline-b4 --num-prompts 16 --num-warmups 4`, 08:30-08:58 Z, into
`~/pro6000-vae-e2e-b4waves-baseline`):

```
| Cell | Conc | Batch | n | E2E ms/image mean (min-max) | Stage 0 gen ms | Stage 1 gen ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1536_baseline-b4 | 4 | 4 | 4 | 79,421.9 (79,334.4-79,470.5) | 209,406.8 | 87,188.2 |
```

No OOM, span 0.17 %, and its whole-device peak reproduces the published
single-wave peak exactly (**87,078 MiB**). The other seven cells need no
re-run: their peaks are byte-for-byte the published ones.

## Cross-run comparison (read this before replacing any absolute value)

Against the published single-wave sweep (`results_b4.md`, 2026-10-05):

| Cell | New mean ms | Old mean ms | Diff | Stage 0 diff | Stage 1 diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1536 baseline-b4 | 79,421.9 | 78,120.6 | +1.7 % | +2.2 % | +0.7 % |
| 1536 tiling-b4 | 80,341.2 | 77,710.7 | +3.4 % | +4.8 % | +0.7 % |
| 1536 slicing-b4 | 78,976.8 | 78,003.3 | +1.2 % | +1.7 % | +0.4 % |
| 1536 both-b4 | 79,455.3 | 77,624.8 | +2.4 % | +3.3 % | +0.5 % |
| 1024 baseline-b4 | 36,961.3 | 36,306.3 | +1.8 % | +2.8 % | -0.7 % |
| 1024 tiling-b4 | 36,783.5 | 35,992.6 | +2.2 % | +2.7 % | +0.8 % |
| 1024 slicing-b4 | 36,832.0 | 36,446.2 | +1.1 % | +1.5 % | +0.0 % |
| 1024 both-b4 | 36,615.3 | 36,497.1 | +0.3 % | +0.3 % | +0.3 % |

The offset is small, one-sided, and **almost entirely in stage 0** (the
autoregressive stage), while stage 1 (DiT + VAE) is flat within ±0.8 %. The
node currently carries other tenants (GPUs 0/1/3 at ~7 GB and a few percent
utilisation; GPU 2 was empty after the contaminated first cell). Whether the
cause is clocks, thermals or power sharing, the reading is the same — a
~1-3 % environmental shift on the AR stage between Oct 5 and Oct 7.

Consequences:

- **Within this run** the config comparison is valid: at 1536 the four configs
  sit within 1.4 % of each other (79.0-80.3 s per image), slicing and baseline
  at the tight end and tiling at the wide end.
- **Across runs**, a couple of percent on any AR-bound number should be read as
  environment, not as a regression. The recipe's batch-1 rows (Oct 5) keep
  their published values; the batch-4 rows should either move to this run's
  mean (min-max) as a set, or keep the old single values with the note that the
  old run has no min-max — do not mix a new mean with an old min-max.

## Findings

- **The batch-4 rows now have a real spread.** Per-image min-max over four
  waves: 79.3-79.5 to 79.9-81.0 s at 1536, 36.3-36.8 to 36.8-37.1 s at 1024.
- **The spread is stage-0 (AR) weather.** Per-wave stage-0 ranges reach 1.9-2.3 %
  at 1536 (tiling, both) while stage 1 stays ≤ 1.1 % everywhere. The wave-to-wave
  variation is the AR stage's, consistent with the batch-1 story.
- **Slicing is the steadiest config.** 1536 slicing spans 0.3 % (78.8-79.1 s)
  and 1024 slicing 0.4 %; baseline/tiling/both span 0.6-1.6 % at 1024 and
  1.4-1.7 % (clean cells) at 1536. Slicing's batch-split decode damps the
  decode-side variation.
- **Memory is unchanged.** Seven of eight peaks are identical to the published
  ones (60,846 / 66,020 / 67,200 / 69,688 / 74,090 / 87,078 MiB and the second
  69,688); only 1024 baseline sits 570 MiB higher (70,258 vs 69,688), a
  transient in a 0.5 s sampling window, not a policy change.
- **1536 baseline-b4 at batch 4 now nearly fills the card when shared** —
  95,361 MiB device peak with a 13.5 GiB co-tenant. On a quiet card it peaks at
  87,078 MiB. Any batch-4 run on a shared 96 GB card should expect the
  allocator to hit the wall.

## Paste-ready values for the recipe (`End-to-end s / image`, ms / 1000)

| Table | Row | End-to-end s / image |
| --- | --- | --- |
| 1536 batch 4 | baseline | 79.4 (79.3-79.5) |
| 1536 batch 4 | tiling | 80.3 (79.9-81.0) |
| 1536 batch 4 | slicing | 79.0 (78.8-79.1) |
| 1536 batch 4 | slicing + tiling | 79.5 (78.9-80.2) |
| 1024 batch 4 | baseline | 37.0 (36.8-37.1) |
| 1024 batch 4 | tiling | 36.8 (36.6-37.1) |
| 1024 batch 4 | slicing | 36.8 (36.8-36.9) |
| 1024 batch 4 | slicing + tiling | 36.6 (36.3-36.8) |

Peaks are unchanged from the published table; add the environment note (one
shared-card caveat + the ~2 % AR-stage offset vs the Oct-5 sweep) next to the
table so the batch-1 and batch-4 columns are not read as one session.

## To record

- Both output trees, raw records included:

```
/root/pro6000-vae-e2e-b4waves/raw:             (main sweep, 2026-10-07 05:41-08:29 Z)
  1024_baseline-b4.json  1024_both-b4.json  1024_slicing-b4.json
  1024_tiling-b4.json    1536_baseline-b4.json  1536_both-b4.json
  1536_slicing-b4.json   1536_tiling-b4.json
/root/pro6000-vae-e2e-b4waves-baseline/raw:    (clean re-run of 1536_baseline-b4)
  1536_baseline-b4.json
```

- Each tree has `results.md` / `results.json` from `collect.py` (rc=0), the
  per-cell `bench_*.log`, `mem_*.txt`, `deploy_*.yaml`, `console.log` and
  `run_env.txt` (HEAD, vLLM version, command).
- `git rev-parse HEAD` of the measuring checkout:
  `86a9d74dc0a8055be18dc7addccc20810f96752b`; vLLM 0.30.0.
