#!/bin/bash
# Combination-validation sweep for #6665 (CPU offload x FP8), PR #6897 head a054ccc19.
# Runs inside the vllm-omni container. One serve per arm; 1 warmup + 3 measured requests.
set -u

ROOT=/workspace/boogu-matrix
RUNS="$ROOT/runs"
PORT=8097
VLLM_BIN=${VLLM_BIN:-/workspace/venv31/bin/vllm}
BASE=/models/Boogu-Image-0.1-Base
FP8=/models/Boogu-Image-0.1-Base-fp8
QUANT='{"transformer":{"method":"torchao_float8_weight_only"}}'
PROMPT="A mountain lake at sunset, photorealistic, cinematic lighting"
BODY=$(printf '{"prompt": "%s", "size": "512x512", "num_inference_steps": 10, "guidance_scale": 1.0, "seed": 42, "response_format": "b64_json"}' "$PROMPT")

mkdir -p "$RUNS"

sampler() {  # $1 = peak file; writes the running max of process-scoped used MiB
    local max=0 v
    while :; do
        v=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
        [ "${v:-0}" -gt "$max" ] && max=$v
        echo "$max" > "$1"
        sleep 0.5
    done
}

stop_arm() {  # $1 = pgid
    kill -- -"$1" 2>/dev/null
    kill "$1" 2>/dev/null
    sleep 5
    pkill -9 -f "vllm serve .*--port $PORT" 2>/dev/null
    for _ in $(seq 1 60); do
        local v
        v=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
        [ "${v:-0}" -lt 500 ] && break
        sleep 5
    done
}

serve_arm() {  # $1 name  $2 model  $3 timing  rest = extra args
    local name=$1 model=$2 timing=$3; shift 3
    local log="$RUNS/$name.serve.log"
    local ready=0

    echo "[$(date +%H:%M:%S)] === arm $name (timing=$timing) extra: $* ==="
    rm -f "$RUNS/$name.peak_mib" "$RUNS/$name.load_peak_mib" "$RUNS/$name.OOM"

    setsid env PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1 VLLM_OMNI_OFFLOAD_TIMING="$timing" \
        "$VLLM_BIN" serve "$model" --omni --port "$PORT" --trust-remote-code "$@" \
        > "$log" 2>&1 &
    local pgid=$!
    sampler "$RUNS/$name.load_peak_mib" &
    local load_sampler=$!

    for _ in $(seq 1 240); do
        if grep -q -E "CUDA out of memory|OutOfMemoryError" "$log" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] arm $name: OOM during load"; touch "$RUNS/$name.OOM"
            kill "$load_sampler" 2>/dev/null; stop_arm "$pgid"; return 1
        fi
        if ! kill -0 "$pgid" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] arm $name: server process gone"; tail -5 "$log"
            kill "$load_sampler" 2>/dev/null; stop_arm "$pgid"; return 1
        fi
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
           || curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            ready=1; break
        fi
        sleep 5
    done
    kill "$load_sampler" 2>/dev/null
    if [ "$ready" != 1 ]; then
        echo "[$(date +%H:%M:%S)] arm $name: not ready in 20 min"; stop_arm "$pgid"; return 1
    fi
    echo "[$(date +%H:%M:%S)] arm $name: ready (load peak $(cat "$RUNS/$name.load_peak_mib" 2>/dev/null) MiB)"

    sampler "$RUNS/$name.peak_mib" &
    local steady_sampler=$!

    for i in 0 1 2 3; do
        local t0 t1 code
        t0=$(date +%s.%N)
        code=$(curl -s -m 600 -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
            -H 'Content-Type: application/json' -d "$BODY" \
            -w '%{http_code}' -o "$RUNS/$name.req$i.json")
        t1=$(date +%s.%N)
        awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.0f\n", (b-a)*1000}' > "$RUNS/$name.req$i.ms"
        jq -r '.data[0].b64_json' "$RUNS/$name.req$i.json" 2>/dev/null | base64 -d > "$RUNS/$name.req$i.png" 2>/dev/null
        if [ "$i" = 0 ]; then
            echo "[$(date +%H:%M:%S)] arm $name warmup: http $code, $(cat "$RUNS/$name.req$i.ms") ms"
        else
            echo "[$(date +%H:%M:%S)] arm $name req$i: http $code, $(cat "$RUNS/$name.req$i.ms") ms, png $(sha256sum "$RUNS/$name.req$i.png" 2>/dev/null | cut -c1-16)"
        fi
    done

    kill "$steady_sampler" 2>/dev/null
    grep -m8 -E "OffloadBackend|offload|resident|streaming|mutual exclusion" "$log" > "$RUNS/$name.activation.txt" 2>/dev/null
    grep -m2 "layerwise offload timing" "$log" >> "$RUNS/$name.activation.txt" 2>/dev/null

    echo "[$(date +%H:%M:%S)] arm $name steady peak $(cat "$RUNS/$name.peak_mib" 2>/dev/null) MiB"
    stop_arm "$pgid"
    echo "[$(date +%H:%M:%S)] arm $name done"
    return 0
}

{
    echo "sweep start $(date -u +%FT%TZ)"
    echo "code head: a054ccc19 (boogu-merge-20261006), vllm 0.31.0 (/workspace/venv31)"
    echo

    # FP8 arms first (fast load): B-only, then A+B for both offload modes
    serve_arm fp8_noffload "$FP8" 0 --diffusion-quantization-config "$QUANT"
    serve_arm fp8_model    "$FP8" 0 --diffusion-quantization-config "$QUANT" --enable-cpu-offload
    serve_arm fp8_layerwise "$FP8" 1 --diffusion-quantization-config "$QUANT" --enable-layerwise-offload

    # bf16 arms: A-only (both modes) and the expected-rejection no-offload arm
    serve_arm bf16_noffload "$BASE" 0
    serve_arm bf16_model    "$BASE" 0 --enable-cpu-offload
    serve_arm bf16_layerwise "$BASE" 1 --enable-layerwise-offload

    # distributed layerwise (DLO), single GPU, resident-layers N = 0 / 4
    serve_arm dlo_n0 "$BASE" 1 --enable-distributed-layerwise-offload --dlo-no-use-allgather
    serve_arm dlo_n4 "$BASE" 1 --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 4

    # mode-boundary check: both legacy flags at once (module + local layerwise)
    serve_arm fp8_bothflags "$FP8" 0 --diffusion-quantization-config "$QUANT" --enable-cpu-offload --enable-layerwise-offload

    echo
    echo "sweep end $(date -u +%FT%TZ)"
} > "$RUNS/sweep.log" 2>&1
