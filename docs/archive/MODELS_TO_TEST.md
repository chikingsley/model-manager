# Models to Test

Hardware: RTX 5070 (12GB VRAM) + 94GB RAM

---

## vLLM Compatible (AWQ/GPTQ/safetensors)

These get ALL optimizations (async scheduling, KV offloading, FP8 KV cache, FlashInfer).

| Model | Size | Format | Fits 12GB? | Notes |
|-------|------|--------|------------|-------|
| **Qwen/Qwen3-4B** | ~8GB | safetensors | ✓ Likely | Small, fast, good quality |
| **cyankiwi/GLM-4.7-Flash-AWQ-4bit** | ~5GB? | AWQ | ✓ Likely | Vision capable |
| **Alibaba-Apsara/DASD-4B-Thinking** | ~8GB? | safetensors? | ✓ Check | Reasoning model |
| **LiquidAI/LFM2.5-1.2B-Instruct** | ~2.5GB | safetensors | ✓ Yes | Very small, fast |
| **Qwen/Qwen2.5-7B-Instruct-AWQ** | ~4GB | AWQ | ✓ Yes | Current benchmark model |

### To Test First (vLLM)

1. **Qwen3-4B** — smaller than 7B, might allow 32K context
2. **LiquidAI/LFM2.5-1.2B** — tiny, should be very fast
3. **cyankiwi/GLM-4.7-Flash-AWQ** — vision + chat in one

---

## llama.cpp Only (GGUF)

These run on llama.cpp, NOT vLLM. Different optimization path.

| Model | Quant | Size | Fits 12GB? |
|-------|-------|------|------------|
| **Devstral-Small-2-24B** | Q2_K | 5.25GB | ✓ Tight |
| **Devstral-Small-2-24B** | IQ2_M | 4.91GB | ✓ Yes |
| **Ministral-3-14B-Instruct** | Q4_K_M | 6.17GB | ✓ Yes |
| **Ministral-3-14B-Instruct** | Q3_K_M | 4.97GB | ✓ Yes |
| **Ministral-3-14B-Reasoning** | Q4_K_M | ~6GB | ✓ Yes |
| **GLM-4.6V-Flash** | Q4_K_M | 8.24GB | ⚠️ Tight |

### GGUF Notes

- 24B model at Q2_K (5.25GB) = heavily quantized, quality hit
- 14B at Q4_K_M (6GB) = good balance
- These run via `~/github/calling-nemo/docker-compose.yml` (the qwen3 container)

---

## Test Priority

### Phase 1: vLLM Baseline vs Optimized

1. [ ] Qwen2.5-7B-AWQ baseline (no optimizations)
2. [ ] Qwen2.5-7B-AWQ optimized (current config) ✓ Done: 43.3 t/s

### Phase 2: Smaller vLLM Models (try 32K context)

3. [ ] Qwen3-4B with max optimizations
2. [ ] LFM2.5-1.2B (speed test)

### Phase 3: Vision Models

5. [ ] GLM-4.7-Flash-AWQ (vision + chat)

### Phase 4: llama.cpp Comparison

6. [ ] Ministral-3-14B Q4_K_M via llama.cpp
2. [ ] Compare llama.cpp vs vLLM for same model (if available in both formats)

---

## Size Estimation for 12GB VRAM

```text
Available VRAM:     12,227 MB
System overhead:    ~500 MB
KV cache (FP8):     ~1-4 GB (depends on context)
Model weights:      ??? MB
---------------------------------
Safe model size:    ~7-10 GB max
```

With FP8 KV cache and 16K context:

- 4B model: Easy fit, room for 32K context
- 7B AWQ: Fits at 16K context
- 14B: Too big for vLLM with optimizations
