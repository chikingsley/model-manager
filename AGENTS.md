# AGENTS.md - model-manager

## Benchmarking Canonical Workflow

Use these in order for OCR/vLLM performance checks.

1. `mm` sanity benchmark (quick health/perf snapshot)
```bash
cd /home/simon/github/model-manager
uv run mm benchmark run
```

2. Official vLLM load benchmark (OpenAI-compatible endpoint)
```bash
docker exec vllm vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model zai-org/GLM-OCR \
  --dataset-name random \
  --num-prompts 200 \
  --max-concurrency 16 \
  --request-rate inf \
  --random-input-len 32 \
  --random-output-len 64
```

3. OCR-heavy concurrency sweep (real image workload)
```bash
python3 scripts/ocr_concurrency_sweep.py \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --model zai-org/GLM-OCR \
  --image /home/simon/github/MultimodalOCR/OCRBench/OCRBench_Images/SROIE/image/X51006414427.jpg \
  --prompt "when was this receipt issued? Answer this question using the text in the image directly." \
  --max-tokens 128 \
  --timeout-s 120 \
  --concurrency 8,16,24,32 \
  --output-json benchmarks/results/ocr-load/custom_run.json \
  --output-csv benchmarks/results/ocr-load/custom_run.csv
```

## Current OCR vLLM Defaults (GLM-OCR)

- `--max-num-seqs 16`
- `--max-num-batched-tokens 8192`
- `--kv-cache-dtype fp8`
- `--kv-offloading-backend native --kv-offloading-size 62`
- `--no-enable-prefix-caching`
- `--mm-processor-cache-gb 0`

Rationale:
- `8192` fixed large-image encoder-cache failures seen at lower budget.
- `16` keeps multimodal latency stable while preserving concurrency headroom on 12GB VRAM.

## SAM3 (Segmentation) Workflow

Use SAM3 as a dedicated backend service (not vLLM/llama.cpp):

```bash
cd /home/simon/github/model-manager
uv run mm sam3
```

Notes:
- SAM3 runs from `services/sam3/` using the official Meta repo package.
- Activation switches GPU ownership (conflicting services are stopped by `mm`).
- Health endpoint: `http://localhost:8095/health`
