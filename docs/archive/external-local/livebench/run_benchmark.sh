#!/bin/bash
# LiveBench runner for local GGUF models
# Uses port 19000 to avoid conflict with nemotron (18000)

set -e

MODEL="${1:-/home/simon/models/GLM-4.6V-Flash-Q4_K_M.gguf}"
PORT="${2:-19000}"
MODEL_NAME=$(basename "$MODEL" .gguf | tr '[:upper:]' '[:lower:]')
BENCH_CATS="${3:-reasoning}"  # reasoning, math, coding, language, data_analysis, instruction_following

echo "=== LiveBench Runner ==="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Categories: $BENCH_CATS"
echo ""

# Check if container already running on this port
if docker ps --format '{{.Ports}}' | grep -q ":${PORT}->"; then
    echo "Warning: Something already running on port $PORT"
    echo "Stop it first or use a different port"
    exit 1
fi

# Start llama.cpp server
echo "Starting llama.cpp server..."
docker run -d --rm \
    --name livebench-llama \
    --gpus all \
    -p ${PORT}:8080 \
    -v /home/simon/models:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    --host 0.0.0.0 \
    --port 8080 \
    -m "$MODEL" \
    -c 8192 \
    -ngl 99

echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
    echo -n "."
done

# Run LiveBench
echo ""
echo "Running LiveBench..."
cd /home/simon/github/livebench

uv run python -m livebench.gen_model_answer \
    --model-path "http://localhost:${PORT}/v1" \
    --model-id "$MODEL_NAME" \
    --bench-name "live_bench/${BENCH_CATS}" \
    --max-new-token 4096 \
    --question-source huggingface

echo ""
echo "=== Results saved to ==="
echo "livebench/data/live_bench/${BENCH_CATS}/*/model_answer/${MODEL_NAME}.jsonl"

# Cleanup
echo ""
echo "Stopping server..."
docker stop livebench-llama 2>/dev/null || true

echo "Done! View results with:"
echo "  cat livebench/data/live_bench/${BENCH_CATS}/*/model_answer/${MODEL_NAME}.jsonl | jq -s 'length'"
