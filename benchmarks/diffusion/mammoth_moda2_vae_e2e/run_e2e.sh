#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# End-to-end sweep for the MammothModa2 VAE memory modes (vae_use_slicing /
# vae_use_tiling, PR #7774 review).  For every (size, config) pair it
#
#   1. writes a deploy config with the two flags injected into the DiT stage,
#   2. starts `vllm-omni serve` with that config,
#   3. waits for /health, then runs `vllm-omni bench serve --print-stage`
#      (--num-warmups warmups, then --num-prompts measured requests),
#   4. samples device memory every 0.5 s for the whole run,
#   5. captures one image for the quality comparison (batch-1 configs only),
#   6. stops the server and records the run.
#
# `collect.py` then turns the output directory into results.md/results.json.
#
# Example -- the run that fills the recipe's end-to-end table:
#
#   benchmarks/diffusion/mammoth_moda2_vae_e2e/run_e2e.sh \
#     --model /models/MammothModa2-Preview \
#     --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
#     --out ~/pro6000-vae-e2e --sizes 1536,1024
#
#   python benchmarks/diffusion/mammoth_moda2_vae_e2e/collect.py --out ~/pro6000-vae-e2e
#
# Run `--dry-run` first: it prints every command without executing anything.
# Each (size, config) pair takes about (warmups + prompts) x per-request latency
# plus one model load, so budget hours, not minutes, and prefer `--configs` to
# cut the sweep into pieces.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
DEPLOY=""
OUT=""
SIZES="1536,1024"
CONFIGS="baseline,slicing,tiling,both,slicing-b4"
NUM_PROMPTS=5
NUM_WARMUPS=4
PORT=8099
GPU=0
SERVER_TIMEOUT_MIN=45
NUM_INFERENCE_STEPS=50
TEXT_GUIDANCE_SCALE=9.0
SEED=42
CLI="vllm-omni"
DRY_RUN=0
# Some images only ship `python3` on the host side; the vllm-omni image has both.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
[ -n "$PYTHON" ] || { echo "no python interpreter found; set PYTHON=/path/to/python" >&2; exit 2; }

usage() {
    sed -n '5,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --model DIR              Local MammothModa2 checkpoint directory (required)
  --deploy-config FILE     Base deploy config (required), e.g. vllm_omni/deploy/mammoth_moda2.yaml
  --out DIR                Output directory (required)
  --sizes LIST             Comma-separated square sizes (default: 1536,1024)
  --configs LIST           Subset of baseline,slicing,tiling,both,slicing-b4,both-b4
                           (default: baseline,slicing,tiling,both,slicing-b4)
  --num-prompts N          Measured requests per run (default: 5)
  --num-warmups N          Warmup requests per run (default: 4)
  --port N                 Server port (default: 8099)
  --gpu N                  GPU index to sample (default: 0)
  --server-timeout-min N   Server startup timeout (default: 45)
  --steps N                num_inference_steps (default: 50)
  --guidance F             text_guidance_scale (default: 9.0)
  --seed N                 Request seed (default: 42)
  --cli CMD                How to reach the CLI (default: vllm-omni)
  --dry-run                Print the commands instead of running them
  -h, --help               This message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --deploy-config) DEPLOY="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --configs) CONFIGS="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --num-warmups) NUM_WARMUPS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --server-timeout-min) SERVER_TIMEOUT_MIN="$2"; shift 2 ;;
        --steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --guidance) TEXT_GUIDANCE_SCALE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --cli) CLI="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$MODEL" ] || { echo "--model is required" >&2; exit 2; }
[ -n "$DEPLOY" ] || { echo "--deploy-config is required" >&2; exit 2; }
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
[ -f "$DEPLOY" ] || { echo "no such deploy config: $DEPLOY" >&2; exit 2; }

# name -> "slicing tiling concurrency".  Slicing only splits when a request
# carries more than one image, so the -b4 rows are where it acts; tiling is a
# per-image spatial split and shows up at any batch size above the threshold.
config_flags() {
    case "$1" in
        baseline)   echo "0 0 1" ;;
        slicing)    echo "1 0 1" ;;
        tiling)     echo "0 1 1" ;;
        both)       echo "1 1 1" ;;
        slicing-b4) echo "1 0 4" ;;
        both-b4)    echo "1 1 4" ;;
        *) echo "unknown config: $1" >&2; exit 2 ;;
    esac
}

SERVER_PID=""
MEM_PID=""
cleanup() {
    [ -n "$MEM_PID" ] && kill "$MEM_PID" 2>/dev/null || true
    stop_server
}
stop_server() {
    if [ -n "$SERVER_PID" ]; then
        # setsid made the server a session leader, so its process group is -PID.
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT INT TERM

extra_body() {
    printf '{"height":%s,"width":%s,"num_inference_steps":%s,"text_guidance_scale":%s,"seed":%s}' \
        "$1" "$1" "$NUM_INFERENCE_STEPS" "$TEXT_GUIDANCE_SCALE" "$SEED"
}

run_one() {
    local size="$1" cfg="$2"
    read -r slicing tiling conc <<<"$(config_flags "$cfg")"
    local tag="${size}_${cfg}"
    local cfgfile="$OUT/deploy_${tag}.yaml"
    local log="$OUT/server_${tag}.log"
    local benchlog="$OUT/bench_${tag}.log"
    local memfile="$OUT/mem_${tag}.txt"
    local model_url="http://127.0.0.1:${PORT}"

    echo "=== ${tag}: slicing=${slicing} tiling=${tiling} concurrency=${conc} ==="
    "$PYTHON" - "$DEPLOY" "$cfgfile" "$slicing" "$tiling" <<'PY'
import sys

import yaml

src, dst, slicing, tiling = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"
with open(src) as handle:
    cfg = yaml.safe_load(handle)
dit = [stage for stage in cfg["stages"] if stage.get("stage_id") == 1]
if not dit:
    raise SystemExit(f"no stage_id: 1 in {src}")
dit[0]["vae_use_slicing"] = slicing
dit[0]["vae_use_tiling"] = tiling
with open(dst, "w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(f"[{dst}] stage 1: vae_use_slicing={slicing} vae_use_tiling={tiling}")
PY
    [ -f "$cfgfile" ] || { echo "${tag}: could not write $cfgfile" >&2; return 1; }

    if [ "$DRY_RUN" = 1 ]; then
        echo "  would start: $CLI serve $MODEL --omni --deploy-config $cfgfile --port $PORT --log-stats"
        echo "  would bench: $CLI bench serve --omni --model $MODEL --base-url $model_url \\"
        echo "      --endpoint /v1/chat/completions --dataset-name random --max-concurrency $conc \\"
        echo "      --num-warmups $NUM_WARMUPS --num-prompts $NUM_PROMPTS --metric-percentiles 50,95 \\"
        echo "      --extra-body '$(extra_body "$size")' --print-stage --save-detailed --save-result"
        return 0
    fi

    setsid $CLI serve "$MODEL" --omni --deploy-config "$cfgfile" --port "$PORT" --log-stats \
        >"$log" 2>&1 &
    SERVER_PID=$!

    local deadline=$((SECONDS + SERVER_TIMEOUT_MIN * 60))
    until curl -fsS "${model_url}/health" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "${tag}: server exited during startup; last log lines:" >&2
            tail -n 40 "$log" >&2
            SERVER_PID=""
            return 1
        fi
        if [ "$SECONDS" -gt "$deadline" ]; then
            echo "${tag}: server not ready after ${SERVER_TIMEOUT_MIN} min; last log lines:" >&2
            tail -n 40 "$log" >&2
            stop_server
            return 1
        fi
        sleep 5
    done
    echo "${tag}: server ready after $((SECONDS - deadline + SERVER_TIMEOUT_MIN * 60))s"

    : >"$memfile"
    ( while :; do nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true; sleep 0.5; done ) >>"$memfile" &
    MEM_PID=$!

    rm -f "$OUT/raw/${tag}.json"
    $CLI bench serve --omni --model "$MODEL" --base-url "$model_url" \
        --endpoint /v1/chat/completions --dataset-name random \
        --num-warmups "$NUM_WARMUPS" --num-prompts "$NUM_PROMPTS" --max-concurrency "$conc" \
        --metric-percentiles 50,95 \
        --extra-body "$(extra_body "$size")" \
        --print-stage --save-detailed --save-result --result-dir "$OUT/raw" --result-filename "${tag}.json" \
        2>&1 | tee "$benchlog"

    if [ "$conc" = 1 ]; then
        "$PYTHON" "$HERE/capture_image.py" --url "$model_url" --size "$size" --steps "$NUM_INFERENCE_STEPS" \
            --guidance "$TEXT_GUIDANCE_SCALE" --seed "$SEED" --out "$OUT/img_${tag}.png" || \
            echo "${tag}: image capture failed (the numbers above are unaffected)" >&2
    fi

    kill "$MEM_PID" 2>/dev/null || true
    MEM_PID=""
    stop_server
    echo "exit=${tag}" >>"$OUT/runs.txt"
}

mkdir -p "$OUT/raw"
echo "model=$MODEL deploy=$DEPLOY sizes=$SIZES configs=$CONFIGS" | tee "$OUT/runs.txt"
echo "prompts=$NUM_PROMPTS warmups=$NUM_WARMUPS steps=$NUM_INFERENCE_STEPS guidance=$TEXT_GUIDANCE_SCALE seed=$SEED" >>"$OUT/runs.txt"
echo "cli=$CLI port=$PORT gpu=$GPU started=$(date -Is)" >>"$OUT/runs.txt"

for size in ${SIZES//,/ }; do
    for cfg in ${CONFIGS//,/ }; do
        run_one "$size" "$cfg" || echo "!! ${size}_${cfg} failed; continuing" >&2
    done
done

echo
echo "raw output in $OUT; now run:"
echo "  $PYTHON $HERE/collect.py --out $OUT"
