# Optimization Test Plan

> Goal: Find the optimal single-model configuration for RTX 5070 (12GB) + 94GB RAM.
> We're optimizing for **throughput and context length**, not model switching.

---

## Hardware Context

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | RTX 5070 | 12GB VRAM, Blackwell architecture |
| RAM | 94GB | Can dedicate 60-80GB to KV offloading |
| Storage | NVMe | Fast model loading |

---

## Optimization Compatibility Matrix

| Optimization | Requires | Conflicts With | Best For |
|--------------|----------|----------------|----------|
| **KV Offloading** | CPU RAM, vLLM 0.11+ | Nothing (works with all) | Long context, high throughput |
| **Async Scheduling** | vLLM V1 | Nothing | Always use |
| **Multi-Step Scheduling** | vLLM 0.10+ | Nothing | Always use |
| **FlashInfer Attention** | Blackwell/Ampere+ | FLASH_ATTN (pick one) | RTX 5070 optimal |
| **V1 Engine** | vLLM 0.10+ | Nothing | Always use |
| **Sleep Mode** | vLLM 0.11+ | Nothing, but wastes RAM if not switching | Only for multi-model |
| **FP8 KV Cache** | FP8-capable GPU | Nothing | 2x context window |

**Key insight**: All performance optimizations are compatible. Sleep mode is orthogonal — skip it for single-model deployment.

---

## KV Offloading Sizing

With 94GB RAM:

- System needs: ~4-8GB
- Docker/other services: ~4-8GB
- Available for KV: **70-80GB**

The RECOMMENDATIONS.md suggests 32GB as a default, but that's conservative. Let's test:

| Test | KV Size | Expected Benefit |
|------|---------|------------------|
| Baseline | 0 | Reference |
| Conservative | 32GB | Safe, good gains |
| Aggressive | 64GB | Near-max, great for long context |
| Maximum | 80GB | Risk of OOM under load |

---

## Test Configurations

### Baseline (Current Setup)

```bash
# Your current docker-compose.yml settings
VLLM_ATTENTION_BACKEND=FLASH_ATTN
# No other optimizations
```

### Config A: V1 Engine + Async Scheduling

```bash
VLLM_USE_V1=1
--async-scheduling
```

**Expected**: Reduced GPU idle time

### Config B: Multi-Step Scheduling

```bash
--num-scheduler-steps 10
```

**Expected**: ~28% throughput boost

### Config C: FlashInfer Attention

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER
```

**Expected**: Better performance on Blackwell

### Config D: KV Offloading (32GB)

```bash
--kv_offloading_backend native
--kv_offloading_size 32
```

**Expected**: 2-9x throughput depending on cache hit rate

### Config E: KV Offloading (64GB)

```bash
--kv_offloading_backend native
--kv_offloading_size 64
```

**Expected**: More headroom for long conversations

### Config F: FP8 KV Cache

```bash
--kv-cache-dtype fp8
```

**Expected**: 2x context window in GPU VRAM

### Config MAX: All Optimizations Combined

```bash
VLLM_USE_V1=1
VLLM_ATTENTION_BACKEND=FLASHINFER
--async-scheduling
--num-scheduler-steps 10
--kv_offloading_backend native
--kv_offloading_size 64
--kv-cache-dtype fp8
```

---

## Test Procedures

### Test 1: Startup Time

```bash
time docker compose up -d
# Wait for health check
curl http://localhost:8000/health
```

Measure: Time from start to healthy

### Test 2: Single Request Latency (TTFT)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_NAME",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

Measure: Time to first token

### Test 3: Throughput (Concurrent Requests)

```python
# Use vLLM's benchmark script or custom load test
# Send 100 concurrent requests, measure tokens/second
```

### Test 4: Long Context Performance

```bash
# Send request with 8K, 16K, 32K token prompts
# Measure TTFT and tokens/second
```

### Test 5: Cache Hit Rate Impact

```bash
# Send same prompt prefix repeatedly
# Measure improvement from KV cache hits
```

### Test 6: Memory Usage

```bash
nvidia-smi --query-gpu=memory.used --format=csv
free -h
```

---

## Test Matrix

Run each test (1-6) for each config (Baseline, A-F, MAX):

| Config | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 | Test 6 |
|--------|--------|--------|--------|--------|--------|--------|
| Baseline | | | | | | |
| A: V1+Async | | | | | | |
| B: Multi-Step | | | | | | |
| C: FlashInfer | | | | | | |
| D: KV 32GB | | | | | | |
| E: KV 64GB | | | | | | |
| F: FP8 KV | | | | | | |
| MAX | | | | | | |

---

## Model Compatibility

### Full Optimization Support (vLLM)

- Qwen2.5 series (AWQ, safetensors)
- Llama 3.x series
- Mistral series
- Most HuggingFace transformers

### Partial Support

- Vision models: May not support all KV optimizations
- MoE models: Need expert parallelism flags
- Very new architectures: May fall back to transformers backend

### GGUF (llama.cpp)

- Only supports: `-ngl 999`, `--flash-attn`
- No KV offloading, no async scheduling
- Use for: Models only available as GGUF

---

## Implementation: Updated docker-compose.yml

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    restart: unless-stopped
    ports:
      - "${VLLM_PORT:-8000}:8000"
    volumes:
      - ./cache:/root/.cache/huggingface
      - ./vllm-cache:/root/.cache/vllm
    environment:
      - HF_TOKEN=${HF_TOKEN:-}
      - VLLM_USE_V1=1
      - VLLM_ATTENTION_BACKEND=${ATTENTION_BACKEND:-FLASHINFER}
    command: >
      --model ${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}
      --host 0.0.0.0
      --port 8000
      --max-model-len ${MAX_MODEL_LEN:-32768}
      --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.90}
      --async-scheduling
      --num-scheduler-steps 10
      --kv_offloading_backend native
      --kv_offloading_size ${KV_OFFLOAD_SIZE:-64}
      ${EXTRA_ARGS:-}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: 16gb
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
```

### .env for testing

```bash
# Config MAX
MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.90
KV_OFFLOAD_SIZE=64
ATTENTION_BACKEND=FLASHINFER
VLLM_PORT=8000
```

---

## Success Criteria

1. **Stability**: No OOM, no crashes under load
2. **Throughput**: >1.5x baseline tokens/sec
3. **Latency**: TTFT <2s for 8K context
4. **Memory**: GPU <11GB, RAM <85GB under load

---

## Next Steps

1. [ ] Update docker-compose.yml with optimized config
2. [ ] Run baseline tests with current setup
3. [ ] Test each optimization individually (A-F)
4. [ ] Test MAX config
5. [ ] Find optimal KV offload size (32 vs 64 vs 80)
6. [ ] Document final recommended config
