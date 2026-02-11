#!/bin/bash
# OCRBench evaluation runner - runs each model one at a time
# Usage: ./run_benchmarks.sh

set -e

VLLM_DIR="/home/simon/vllm"
OCRBENCH_DIR="/home/simon/github/MultimodalOCR/OCRBench"
RESULTS_DIR="$OCRBENCH_DIR/results"

# Models to evaluate (full precision, no quantization)
declare -A MODELS=(
    ["zai-org_GLM-OCR"]="zai-org/GLM-OCR"
    ["lightonai_LightOnOCR-2-1B"]="lightonai/LightOnOCR-2-1B"
    ["lightonai_LightOnOCR-1B-1025"]="lightonai/LightOnOCR-1B-1025"
    ["nanonets_Nanonets-OCR2-1.5B-exp"]="nanonets/Nanonets-OCR2-1.5B-exp"
    ["echo840_MonkeyOCR-pro-1.2B"]="echo840/MonkeyOCR-pro-1.2B"
)

# Model-specific vLLM arguments
declare -A MODEL_ARGS=(
    ["zai-org_GLM-OCR"]="--limit-mm-per-prompt '{\"image\": 1}' --mm-processor-cache-gb 0 --no-enable-prefix-caching"
    ["lightonai_LightOnOCR-2-1B"]="--limit-mm-per-prompt '{\"image\": 1}' --mm-processor-cache-gb 0 --no-enable-prefix-caching"
    ["lightonai_LightOnOCR-1B-1025"]="--limit-mm-per-prompt '{\"image\": 1}' --mm-processor-cache-gb 0 --no-enable-prefix-caching"
    ["nanonets_Nanonets-OCR2-1.5B-exp"]=""
    ["echo840_MonkeyOCR-pro-1.2B"]="--trust-remote-code"
)

wait_for_vllm() {
    echo "Waiting for vLLM to be ready..."
    local max_attempts=60
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "  Attempt $attempt/$max_attempts..."
        sleep 10
    done
    echo "ERROR: vLLM failed to start after $max_attempts attempts"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM..."
    cd "$VLLM_DIR"
    docker compose down --timeout 30 2>/dev/null || true
    sleep 5
}

start_vllm() {
    local model_name=$1
    local model_path=$2
    local extra_args=$3

    echo "Starting vLLM with model: $model_path"

    cd "$VLLM_DIR"

    # Create temporary .env for this run
    cat > .env.benchmark << EOF
MODEL=$model_path
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.95
QUANTIZATION=none
EXTRA_ARGS=$extra_args
VLLM_PORT=8000
HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
EOF

    # Use the benchmark env
    cp .env .env.backup
    cp .env.benchmark .env

    docker compose up -d

    wait_for_vllm
}

run_eval() {
    local model_name=$1
    local output_file="$RESULTS_DIR/${model_name}.json"

    echo "Running OCRBench evaluation..."
    echo "Output: $output_file"

    cd "$OCRBENCH_DIR"
    uv run eval_openai.py \
        --base-url http://localhost:8000/v1 \
        --output "$output_file" \
        --image-folder ./OCRBench_Images \
        --benchmark-file ./OCRBench/OCRBench.json \
        --resume

    echo "Evaluation complete: $output_file"
}

print_results() {
    local output_file=$1
    echo ""
    echo "=== Results Summary ==="
    python3 -c "
import json
with open('$output_file') as f:
    data = json.load(f)
scores = {}
for item in data:
    t = item.get('type', 'Unknown')
    if t not in scores:
        scores[t] = {'correct': 0, 'total': 0}
    scores[t]['total'] += 1
    scores[t]['correct'] += item.get('result', 0)
total = sum(s['correct'] for s in scores.values())
print(f'FINAL SCORE: {total}/1000')
"
}

# Main loop
echo "========================================"
echo "OCRBench Multi-Model Evaluation"
echo "========================================"
echo ""

for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    extra_args="${MODEL_ARGS[$model_name]}"
    output_file="$RESULTS_DIR/${model_name}.json"

    echo ""
    echo "========================================"
    echo "Model: $model_name"
    echo "Path:  $model_path"
    echo "========================================"

    # Skip if already completed
    if [ -f "$output_file" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$output_file'))))" 2>/dev/null || echo "0")
        if [ "$count" = "1000" ]; then
            echo "Already completed, skipping..."
            print_results "$output_file"
            continue
        fi
    fi

    stop_vllm
    start_vllm "$model_name" "$model_path" "$extra_args"
    run_eval "$model_name"
    print_results "$output_file"
done

# Restore original .env
cd "$VLLM_DIR"
if [ -f .env.backup ]; then
    mv .env.backup .env
fi

stop_vllm

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
echo ""
echo "Results saved in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
