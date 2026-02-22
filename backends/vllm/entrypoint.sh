#!/bin/bash
set -e

VLLM_DIR="/opt/vllm"
VLLM_INSTALLED_FLAG="/models/.vllm_installed"

echo "=== vLLM Server Startup ==="
echo "Model: $MODEL"
echo "Max context: $MAX_MODEL_LEN"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"

# Install vLLM if not already done
if [ ! -f "$VLLM_INSTALLED_FLAG" ]; then
    echo "=== First run: Installing vLLM (this takes ~10-15 min) ==="

    apt-get update && apt-get install -y ccache curl

    if [ ! -d "$VLLM_DIR" ]; then
        git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
    fi

    cd "$VLLM_DIR"
    git pull origin main

    python use_existing_torch.py
    pip install -r requirements/build.txt
    pip install setuptools_scm ninja

    echo "Building vLLM with ccache (be patient)..."
    export CCACHE_DIR=/models/.ccache
    export MAX_JOBS=8

    # Use pip install which handles torch detection better
    pip install -e . --no-build-isolation -v

    touch "$VLLM_INSTALLED_FLAG"
    echo "=== vLLM installation complete ==="
else
    echo "vLLM already installed, skipping build..."
    cd "$VLLM_DIR"
    pip install -e . --no-build-isolation 2>/dev/null || true
fi

# Build the vllm serve command
CMD="vllm serve $MODEL"
CMD="$CMD --host 0.0.0.0"
CMD="$CMD --port 8000"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

# Add quantization if specified
if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
    echo "Quantization: $QUANTIZATION"
fi

# Add any extra args
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
    echo "Extra args: $EXTRA_ARGS"
fi

echo "=== Starting vLLM server ==="
echo "Command: $CMD"
echo ""

exec $CMD
