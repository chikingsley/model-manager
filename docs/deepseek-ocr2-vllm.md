# DeepSeek-OCR-2 on vLLM

This project is now configured to use `deepseek-ai/DeepSeek-OCR-2` as the default OCR model for `mm ocr`.

## What Changed

- `mm ocr` now defaults to `deepseek-ai/DeepSeek-OCR-2`.
- OCR mode uses the `max_performance` runtime profile (single-backend, single-model throughput).
- OCR startup flags include:
  - `--trust-remote-code`
  - `--limit-mm-per-prompt 'image=1'`
  - `--no-enable-prefix-caching`
  - `--disable-log-requests`

No services are started by these changes. vLLM only starts when you run an activation command.

## Launch (when ready)

```bash
cd /home/simon/github/model-manager
uv run mm ocr
```

Optional explicit model override:

```bash
uv run mm ocr deepseek-ai/DeepSeek-OCR-2
```

## Throughput Check

```bash
uv run scripts/benchmark.py
```

For stress testing concurrent requests:

```bash
uv run scripts/bench.py --url http://localhost:8000
```

For official vLLM load testing (OpenAI-compatible benchmarking):

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

For OCR-specific heavy-image concurrency sweeps:

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

## Why These Defaults

- Multi-step scheduling and async scheduling are already enabled in runtime config.
- Prefix caching is disabled for OCR because document workloads usually have low prefix reuse.
- KV offloading remains enabled for single-GPU stability when concurrency increases.
- KV connector / disaggregated prefill-decoding features are intentionally not enabled here because this setup is single-node, single-backend operation.
