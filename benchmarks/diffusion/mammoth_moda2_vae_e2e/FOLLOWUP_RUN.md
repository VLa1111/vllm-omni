# Follow-up run: batch-4 controls for the slicing rows

The sweep in `BENCHMARK_REPORT.md` answered the review request for a decode batch
larger than one, but it left two problems that the review of PR #7774 will hit.
This note specifies the run that closes them. It is the only outstanding
measurement for the PR.

## Why another run

1. **No control row.** `slicing-b4` is the only batch-4 row, and it is compared
   against the *single-request* baseline. The claim "slicing keeps the peak flat
   when a batch is decoded" therefore has no counterpart measured at the same
   concurrency with the flag off, so the table cannot isolate slicing from the
   batch itself. The VAE-level benchmark
   (`benchmarks/diffusion/mammoth_moda2_vae_decode.py`) does isolate it, but the
   end-to-end table does not.
2. **The batch-4 numbers are a blend.** `--num-prompts 5` with `-b4` issues a
   wave of four requests and then one solo request, so the reported mean mixes
   wave members (`78622 ms`) with a solo (`216728 ms`) — `1536_slicing-b4` reads
   mean 106244 ms against a median of 78622 ms. Only the median describes the
   wave. Running whole waves removes the ambiguity: `run_wave` divides the wave
   wall time by the number of requests, so every measured request carries the
   amortized per-image latency and mean == median.

## Run

`baseline-b4` and `tiling-b4` are defined for this run (see `CONFIGS` in
`run_e2e_offline.py`). Four measured requests over a concurrency of four is
exactly one measured wave (and four warmups are one warmup wave), so every row
is wave-only.

```bash
cd /app/vllm_omni
export PYTHONPATH=$PWD
python benchmarks/diffusion/mammoth_moda2_vae_e2e/run_e2e_offline.py \
    --model /root/models/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --out ~/pro6000-vae-e2e-b4 --sizes 1536,1024 --gpu 2 \
    --configs baseline-b4,tiling-b4,slicing-b4,both-b4 \
    --num-prompts 4 --num-warmups 4 2>&1 | tee ~/pro6000-vae-e2e-b4/console.log
python benchmarks/diffusion/mammoth_moda2_vae_e2e/collect.py --out ~/pro6000-vae-e2e-b4
```

Everything else keeps the published protocol (fixed prompt, `seed=42`,
`num_inference_steps=50`, `text_guidance_scale=9.0`, `cfg_range=0,1`, one fresh
process per cell, memory sampled from before the model load).

Use a **new output directory** — `collect.py` globs every `bench_*.log` under
`--out`, so reusing `~/pro6000-vae-e2e` would mix the new cells into the
published `results.json`.

`--configs` re-runs `slicing-b4` on purpose: the published row came from the
five-prompt protocol and has to be replaced by a wave-only one before the two
are compared in a table.

## Expected result

At 1536x1536 tiling is above its threshold and slicing splits the batch, so the
prediction is:

| Row | Peak vs single-request baseline (67766 MiB) |
| --- | --- |
| `baseline-b4` | grows by the three extra untiled decodes (~15 GiB) — may exceed the card |
| `tiling-b4` | grows by ~3 tiled decodes (~8 GiB): tiling caps the spatial extent, not the batch |
| `slicing-b4` | flat (~67200 MiB measured) |
| `both-b4` | flat, at the tiling peak (~61656 MiB) |

At 1024x1024 tiling is below its threshold (single tile, a no-op), so
`baseline-b4` and `tiling-b4` should agree and both should sit above the slicing
rows. A flat `baseline-b4` at 1536 would mean the batch is not being decoded as
a batch — check `stage_1_gen_ms` and the `max_num_seqs` in the generated
`deploy_*.yaml` before believing it.

If `baseline-b4` fails with an OOM, that is a result, not a broken run: record
the `rc` and the console traceback, and continue. The remaining three cells
still give the slicing A/B with tiling engaged (`both-b4` vs `tiling-b4`).

## Runtime

Roughly 1.5–2 h on the box: per cell, one model load plus a warmup wave plus a
measured wave — a 1536 wave of four takes ~314 s, a 1024 wave ~144 s.

## To record

- `~/pro6000-vae-e2e-b4/{results.md,results.json,runs.txt,console.log}` and the
  per-cell `bench_*.log` / `mem_*.txt` / `deploy_*.yaml`, as before.
- The commit actually measured (`git rev-parse HEAD` at run time) and the vLLM
  version, for the recipe's environment block.
- Fold the table into `BENCHMARK_REPORT.md` and then into the recipe: the
  published `624ebea` table stays, marked historical, and the current-source
  table carries these rows with the source and configuration recorded.
