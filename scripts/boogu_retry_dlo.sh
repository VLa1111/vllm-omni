#!/bin/bash
# Re-test dlo_n4 (DLO + 4 resident layers) twice, to classify the startup segfault
# seen in the main sweep as flaky vs reproducible. Same serve/measure pattern.
set -u

ROOT=/workspace/boogu-matrix
RUNS="$ROOT/runs"
PORT=8097
VLLM_BIN=${VLLM_BIN:-/workspace/venv31/bin/vllm}
BASE=/models/Boogu-Image-0.1-Base
PROMPT="A mountain lake at sunset, photorealistic, cinematic lighting"
BODY=$(printf '{"prompt": "%s", "size": "512x512", "num_inference_steps": 10, "guidance_scale": 1.0, "seed": 42, "response_format": "b64_json"}' "$PROMPT")

mkdir -p "$RUNS"

sampler() {
    local max=0 v
    while :; do
        v=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
        [ "${v:-0}" -gt "$max" ] && max=$v
        echo "$max" > "$1"
        sleep 0.5
    done
}

stop_arm() {
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

serve_arm() {  # $1 name
    local name=$1
    local log="$RUNS/$name.serve.log"
    local ready=0

    echo "[$(date +%H:%M:%S)] === retry arm $name ==="
    rm -f "$RUNS/$name.peak_mib" "$RUNS/$name.load_peak_mib" "$RUNS/$name.OOM" "$RUNS/$name.SEGV"

    setsid env PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1 VLLM_OMNI_OFFLOAD_TIMING=1 \
        "$VLLM_BIN" serve "$BASE" --omni --port "$PORT" --trust-remote-code \
        --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 4 \
        > "$log" 2>&1 &
    local pgid=$!
    sampler "$RUNS/$name.load_peak_mib" &
    local load_sampler=$!

    for _ in $(seq 1 240); do
        if grep -q "Segfault encountered" "$log" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] retry $name: SEGFAULT"; touch "$RUNS/$name.SEGV"
            kill "$load_sampler" 2>/dev/null; stop_arm "$pgid"; wait "$pgid" 2>/dev/null; return 1
        fi
        if grep -q -E "CUDA out of memory|OutOfMemoryError" "$log" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] retry $name: OOM"; touch "$RUNS/$name.OOM"
            kill "$load_sampler" 2>/dev/null; stop_arm "$pgid"; wait "$pgid" 2>/dev/null; return 1
        fi
        if ! kill -0 "$pgid" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] retry $name: server process gone (exit $?)"; tail -3 "$log"
            kill "$load_sampler" 2>/dev/null; stop_arm "$pgid"; wait "$pgid" 2>/dev/null; return 1
        fi
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
           || curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            ready=1; break
        fi
        sleep 5
    done
    kill "$load_sampler" 2>/dev/null
    if [ "$ready" != 1 ]; then
        echo "[$(date +%H:%M:%S)] retry $name: not ready in 20 min"; stop_arm "$pgid"; wait "$pgid" 2>/dev/null; return 1
    fi
    echo "[$(date +%H:%M:%S)] retry $name: ready (load peak $(cat "$RUNS/$name.load_peak_mib" 2>/dev/null) MiB)"

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
        echo "[$(date +%H:%M:%S)] retry $name req$i: http $code, $(cat "$RUNS/$name.req$i.ms") ms, png $(sha256sum "$RUNS/$name.req$i.png" 2>/dev/null | cut -c1-16)"
    done
    kill "$steady_sampler" 2>/dev/null
    grep -m8 -E "OffloadBackend|offload|resident|streaming|mutual exclusion" "$log" > "$RUNS/$name.activation.txt" 2>/dev/null
    grep -m2 "layerwise offload timing" "$log" >> "$RUNS/$name.activation.txt" 2>/dev/null
    echo "[$(date +%H:%M:%S)] retry $name steady peak $(cat "$RUNS/$name.peak_mib" 2>/dev/null) MiB"
    stop_arm "$pgid"
    wait "$pgid" 2>/dev/null
    echo "[$(date +%H:%M:%S)] retry $name done"
    return 0
}

{
    echo "dlo_n4 retry start $(date -u +%FT%TZ)"
    serve_arm dlo_n4_retry1
    echo
    serve_arm dlo_n4_retry2
    echo
    echo "dlo_n4 retry end $(date -u +%FT%TZ)"
} > "$RUNS/retry.log" 2>&1
