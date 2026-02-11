# vLLM Optimization Testing Log

Hardware: RTX 5070 (12GB VRAM) + 94GB RAM
vLLM Version: 0.14.0

---

## 2026-01-25: Initial Max Performance Testing

### Configuration Tested

```bash
# mm perf config (final working version)
MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.85

VLLM_USE_V1=1
VLLM_ATTENTION_BACKEND=FLASHINFER

EXTRA_ARGS=--async-scheduling --kv-offloading-backend native --kv-offloading-size 62 --kv-cache-dtype fp8 --disable-hybrid-kv-cache-manager --max-num-seqs 64 --quantization awq
```

### Issues Encountered & Fixes

| Issue | Error | Fix |
|-------|-------|-----|
| Wrong flag format | `--kv_offloading_backend` unrecognized | Use hyphens: `--kv-offloading-backend` |
| HMA incompatibility | "OffloadingConnector does not support HMA" | Add `--disable-hybrid-kv-cache-manager` |
| OOM during warmup | "CUDA out of memory... 256 dummy requests" | Reduce `MAX_MODEL_LEN` to 16K, `gpu_memory_utilization` to 0.85, add `--max-num-seqs 64` |
| Model name "default" | 404 error | Must use actual model name from /v1/models |

### Active Optimizations (Confirmed in Logs)

- [x] V1 Engine (vLLM 0.14.0)
- [x] FlashInfer attention backend
- [x] Asynchronous scheduling
- [x] KV Offloading (CPUOffloadingSpec, 62GB)
- [x] FP8 KV Cache
- [x] Prefix caching enabled
- [x] Chunked prefill enabled

### Optimizations NOT Available in vLLM 0.14.0

- `--num-scheduler-steps` (multi-step scheduling) - flag doesn't exist in this version

### Memory Usage

| Resource | Usage |
|----------|-------|
| GPU VRAM | 10,968 MB / 12,227 MB (89.7%) |
| System RAM | 77.2 GB (includes 62GB KV offload allocation) |

### Benchmark Results

**Model**: Qwen/Qwen2.5-7B-Instruct-AWQ @ 16K context

| Metric | Value | Notes |
|--------|-------|-------|
| TTFT | 224.5ms | Time to first token |
| Single Request | ~10.8 t/s | 300 tokens in 27.8s |
| 10 Concurrent | 43.3 t/s | Total throughput |
| Per-Request (batched) | ~4.3 t/s | When batched with 10 others |
| Max Concurrent | 64 | Limited by --max-num-seqs |

**Interpretation**: With batching, total throughput increases ~4x (10.8 → 43.3 t/s), but individual request latency increases. This is the expected behavior with GPU batching.

---

## Baseline vs Max Performance Comparison

Tested: 2026-01-26

| Metric | Baseline | Max Perf | Change |
|--------|----------|----------|--------|
| TTFT | 10,979ms | 224.5ms | **49x faster** |
| Throughput (10 conc) | 84.3 t/s | 43.3 t/s | 0.5x slower |
| GPU Memory | 11,102 MB | 10,968 MB | Similar |
| RAM Usage | 13 GB | 77.2 GB | +64GB (KV) |

### Analysis

**Why is throughput LOWER with optimizations?**

1. **KV offloading adds latency** — moving KV cache between GPU and CPU takes time
2. **Baseline uses pure GPU** — faster for simple benchmarks but:
   - First request takes 11 seconds (compile/warmup)
   - Would OOM with longer contexts or more users
   - Not suitable for production

**When to use each:**

| Use Case | Best Config |
|----------|-------------|
| Single user, short context | Baseline |
| Production, multiple users | Max Perf |
| Long context (>16K) | Max Perf (only option) |
| Low latency TTFT | Max Perf |

---

## 2026-01-26: Official vLLM Benchmark Results

Using `vllm bench serve` for standardized measurements.

### Single User Latency (1 req/s, 256 in / 256 out)

| Metric | Value |
|--------|-------|
| Mean TTFT | **220.44ms** |
| P99 TTFT | 275.38ms |
| Mean TPOT | 91.45ms |
| Output throughput | 76.20 tok/s |
| Generation speed | ~11 tok/s |

### Burst Load (50 concurrent, 128 in / 256 out)

| Metric | Value |
|--------|-------|
| Output throughput | 503.31 tok/s |
| Peak throughput | 550 tok/s |
| Mean TTFT | 1,231ms |
| P99 TTFT | 1,878ms |

### Heavy Load (100 concurrent, 512 in / 512 out)

| Metric | Value |
|--------|-------|
| Output throughput | 481.29 tok/s |
| Peak throughput | **706 tok/s** |
| Total throughput | 962.58 tok/s |
| Mean TTFT | 22,852ms |
| P99 TTFT | 58,934ms |

### Key Insights

- **Single user**: ~220ms TTFT, ~11 tok/s generation - excellent for chat
- **Batch capacity**: 500-700 tok/s sustained with 50-100 concurrent requests
- **KV offloading enables**: 100 concurrent 512-token requests without OOM
- **Tradeoff**: Burst requests queue up (high TTFT variance at high concurrency)

Full results: `benchmark_results/official/summary.md`

---

## 2026-01-26: Quality Benchmarks (lm-evaluation-harness)

Using `lm_eval` with `local-chat-completions` model type against vLLM server.

### Qwen2.5-7B-Instruct-AWQ Results

| Benchmark | Metric | Score | Notes |
|-----------|--------|-------|-------|
| **GSM8K** (math) | exact_match | **0%** | Complete failure |
| **IFEval** (instruction following) | strict_acc | 33% | Poor |
| **IFEval** (instruction following) | loose_acc | 44% | Below average |

### Analysis

**The AWQ quantization appears to have severely degraded math capabilities.** The model scores 0% on GSM8K (chain-of-thought math) while achieving mediocre instruction following.

This suggests:

1. AWQ quantization may not preserve reasoning ability well
2. Need to test non-quantized or different quant methods (GGUF Q8, etc.)
3. The "speed vs quality" tradeoff is real

### Benchmark Types

**Works with chat API (`generate_until`):**

- `gsm8k_cot` - Math reasoning
- `ifeval` - Instruction following
- `gpqa_main_cot_zeroshot` - Graduate-level QA

**Needs completions API (`loglikelihood`) - can't use with vLLM chat:**

- `mmlu` - Knowledge
- `hellaswag` - Commonsense
- `winogrande` - Commonsense

### Commands

```bash
# Quick quality test (30 samples)
uv run scripts/quality_bench.py --quick

# Full benchmark
uv run scripts/quality_bench.py --full --save
```

---

## Next Steps

1. [x] Test baseline without optimizations for comparison
2. [x] Run official vLLM throughput benchmarks
3. [x] Run quality benchmarks with lm-evaluation-harness
4. [ ] **Test different quantization methods** - AWQ shows 0% math, try Q8 GGUF
5. [ ] Test Qwen3-4B (smaller model, might allow 32K context)
6. [ ] Test models from MODELS_TO_TEST.md with quality benchmarks
7. [ ] Compare llama.cpp vs vLLM quality at same quant level

---

## Configuration Reference

### Max Performance (Single Model)

Best for: Running one model at max throughput

```bash
./mm perf [model-name]
```

### Multi-Model Switching

Best for: Switching between models quickly (uses sleep mode)

```bash
./mm chat [model-name]  # Basic vLLM
./mm voice              # Nemotron stack
./mm ocr                # LightOn OCR
```

---

## Hardware Notes

- RTX 5070 is Blackwell architecture
- FlashInfer is optimal attention backend for Blackwell
- 94GB RAM allows large KV offloading (up to ~80GB theoretically)
- 12GB VRAM limits model size + context + batch size combination
