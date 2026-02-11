#!/bin/bash
# Run SWE-bench Lite evaluation with local models
#
# Usage:
#   ./run_swebench_lite.sh ollama    # Use Ollama (localhost:11434)
#   ./run_swebench_lite.sh vllm      # Use vLLM (localhost:8000)
#   ./run_swebench_lite.sh llamacpp  # Use llama.cpp (localhost:8090)
#
# Options:
#   LIMIT=5 ./run_swebench_lite.sh ollama   # Only run 5 instances
#   MODEL=ministral-3:8b ./run_swebench_lite.sh ollama  # Specify model

set -e

BACKEND="${1:-ollama}"
LIMIT="${LIMIT:-5}"  # Default to 5 instances for quick testing

case "$BACKEND" in
  ollama)
    CONFIG="config/local_ollama.yaml"
    MODEL="${MODEL:-qwen3:4b}"
    MODEL_ARG="ollama/$MODEL"
    ;;
  vllm)
    CONFIG="config/local_vllm.yaml"
    MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
    MODEL_ARG="openai/$MODEL"
    ;;
  llamacpp)
    CONFIG="config/local_llamacpp.yaml"
    MODEL="${MODEL:-local}"
    MODEL_ARG="openai/$MODEL"
    ;;
  *)
    echo "Unknown backend: $BACKEND"
    echo "Usage: $0 [ollama|vllm|llamacpp]"
    exit 1
    ;;
esac

echo "=================================="
echo "SWE-bench Lite Evaluation"
echo "=================================="
echo "Backend: $BACKEND"
echo "Model: $MODEL_ARG"
echo "Limit: $LIMIT instances"
echo "=================================="

# Activate venv
source .venv/bin/activate

# Run evaluation
sweagent run-batch \
  --config config/default.yaml \
  --config "$CONFIG" \
  --agent.model.name "$MODEL_ARG" \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  --instances.slice ":$LIMIT" \
  --output_dir "results/swebench_lite_${BACKEND}_$(date +%Y%m%d_%H%M%S)"

echo "Done! Check results/ for outputs"
