# MammothModa2 VAE memory modes: end-to-end measurement

Scratch branch `pro6000/mammothmoda2-vae-e2e` (based on PR #7774's head, plus the
scripts in this directory).  It exists to answer the second review comment on
that PR:

> These numbers come from `624ebea` ... Could you rerun the table on the head you
> want merged, with a batch > 1 row for slicing?

The task is one long batch run on the RTX PRO 6000 box that produced the current
table, then a table for the PR.  Nothing here is meant to be merged upstream as
is: the upstream-facing part of the PR is
`benchmarks/diffusion/bench_mammoth_moda2_vae_decode.py` (VAE-level, no server).

## What to run

```bash
# on the PRO 6000 box, in the PR checkout (branch pro6000/mammothmoda2-vae-e2e)
git fetch origin && git checkout pro6000/mammothmoda2-vae-e2e

export PYTHONPATH=$PWD            # the image's installed vllm_omni predates the change
OUT=~/pro6000-vae-e2e

benchmarks/diffusion/mammoth_moda2_vae_e2e/run_e2e.sh \
  --model /models/MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --out $OUT --sizes 1536,1024

python benchmarks/diffusion/mammoth_moda2_vae_e2e/collect.py --out $OUT
```

`--dry-run` prints every command without touching the GPU; do that first.

## What it measures

Per size, one full server start per config, `--num-warmups 4` warmups, then
`--num-prompts 5` measured requests at concurrency 1 (batch 4 for the `-b4`
rows), fixed prompt, `seed=42`, `num_inference_steps=50`,
`text_guidance_scale=9.0`, device peak sampled every 0.5 s:

| Config | `vae_use_slicing` | `vae_use_tiling` | Concurrency |
| --- | --- | --- | --- |
| `baseline` | off | off | 1 |
| `slicing` | on | off | 1 |
| `tiling` | off | on | 1 |
| `both` | on | on | 1 |
| `slicing-b4` | on | off | 4 |

`--configs` selects a subset; `both-b4` (slicing + tiling at batch 4) is also
defined but left out of the default sweep.

Slicing splits the decode along the batch dimension, so it can only do anything
when a request carries more than one image -- that is what the `-b4` rows show,
and it is why the old table, measured at batch 1, could not show it.  Tiling is
a per-image spatial split and engages above the model's tile threshold
(`tile_latent_min_size = 128`, i.e. above 1024x1024 output).

## Output

Everything lands in `--out`:

- `deploy_<size>_<config>.yaml` -- the config actually served (the two flags
  injected into stage 1; the YAML round-trip drops the base file's comments);
- `server_*.log`, `bench_*.log`, `mem_*.txt` -- server log, the full
  `vllm-omni bench serve --print-stage` output, and the 0.5 s device samples;
- `raw/<size>_<config>.json` -- the bench client's own result JSON
  (`--save-detailed`, so per-request timings are in there too);
- `img_<size>_<config>.png` -- one image per batch-1 config, same prompt and
  seed, for the PSNR comparison;
- `results.md`, `results.json` -- `collect.py`'s summary.  `results.md` is the
  table to paste into the PR/recipe, `results.json` keeps every metric the
  client printed, per stage.

## Time and cost

Each config is one model load plus `(warmups + prompts) x per-request latency`:
at 1536x1536 the current table shows ~225 s per request end to end, so about
35 minutes per config, ~3 hours for the 1536 rows alone, plus the 1024 rows and
the model loads.  Run it as a batch job.  `--sizes 1536 --configs baseline,tiling,slicing-b4`
is the short version if only the headline numbers are needed.

## Notes

- The server must be the code under test: `PYTHONPATH=$PWD` (or an editable
  install) -- the nightly image's `vllm_omni` package metadata predates this
  branch, and without the override the flags never reach `gen_vae`.
- The driver patches the deploy config with a small inline Python script, so
  that interpreter needs PyYAML; `PYTHON=/path/to/python` overrides the
  interpreter it picks (`python3` first, then `python`).
- `--log-stats` is passed for parity with the original run; the per-stage split
  in the table actually comes from the response metrics the client requests with
  `--print-stage`.
- PSNR needs Pillow and numpy; without them the column stays `-` and the rest of
  the table is unaffected.
- The recipe's `vae_use_slicing` / `vae_use_tiling` keys go on **stage 1**; the
  script injects them there and fails loudly if stage 1 is missing.
