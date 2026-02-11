# Model Manager — Specification

> Local LLM infrastructure manager. Download a model, it just works optimally.

## Philosophy

**Zero configuration. Zero flags. Zero bullshit.**

When you download or add a model:

1. System detects format (GGUF, AWQ, safetensors, etc.)
2. System picks the right backend (vLLM vs llama.cpp)
3. System applies optimal settings for your hardware
4. You just say "use this model" — everything else is automatic

You never see CLI flags. You never edit config files. The system bootstraps everything.

---

## Problem

Running local LLMs on a single GPU requires:

- Knowing which backend works with which format
- Remembering 15 different optimization flags
- Manual HuggingFace downloads
- Stopping/starting containers constantly

## Solution

A manager that handles all of this invisibly:

- **Auto-bootstrap**: Detects model → picks backend → applies optimizations
- **Smart switching**: Sleep/wake for instant model swaps
- **HF integration**: Search, download, auto-configure
- **One interface**: TUI or API, your choice

---

## Hardware Context

| Resource | Value |
|----------|-------|
| GPU | RTX 5070 (Blackwell, 12GB VRAM) |
| RAM | 94GB |
| Models | `/home/simon/models/` |

---

## Auto-Bootstrap Rules

When a model is added, the system applies these rules automatically:

### Backend Selection

| Format | Backend | Reason |
|--------|---------|--------|
| `.gguf` | llama.cpp | Native format |
| `awq` | vLLM | Best AWQ support |
| `gptq` | vLLM | Best GPTQ support |
| `safetensors` | vLLM | Native HF format |
| `exl2` | exllamav2 | (future) |

### Model Type Detection

| Pattern | Type | Special Config |
|---------|------|----------------|
| `embed` in name | embedding | Task=embed, port 8085 |
| `rerank` in name | reranker | Task=score |
| `vision`, `vl` in name | vision | Multimodal enabled |
| Default | chat | Standard config |

### Hardware Optimizations (always applied)

These are applied based on YOUR hardware, not model choice:

**RTX 5070 + 94GB RAM:**

- vLLM: Sleep mode enabled (instant switching)
- vLLM: KV offload 32GB to RAM (you have plenty)
- vLLM: Async scheduling (Blackwell optimization)
- vLLM: FlashInfer attention (Blackwell optimal)
- vLLM: Multi-step scheduling (28% throughput boost)
- llama.cpp: Full GPU offload
- llama.cpp: Flash attention enabled

### Quantization Detection

| Model name contains | Quant type |
|---------------------|------------|
| `AWQ`, `awq` | AWQ |
| `GPTQ`, `gptq` | GPTQ |
| `Q4_K_M`, `Q5_K_M`, etc. | GGUF quant |
| `FP8`, `fp8` | FP8 |
| `INT8`, `int8` | INT8 |

### VRAM Estimation

System estimates VRAM from:

- Model file size
- Quant type (4-bit ≈ 0.5 bytes/param, 8-bit ≈ 1 byte/param, fp16 ≈ 2 bytes/param)
- Context length overhead

If model won't fit → recommend smaller quant or warn user.

---

## Model Registry

Instead of "presets", we have a **model registry** — every downloaded model gets auto-configured:

```yaml
# models.yaml (auto-generated, user doesn't edit)
models:
  qwen2.5-7b-awq:
    source: Qwen/Qwen2.5-7B-Instruct-AWQ
    path: ~/.cache/huggingface/...
    format: awq
    backend: vllm
    type: chat
    vram_estimate: 10
    # Config is computed, not stored

  qwen3-embed-4b:
    source: Qwen/Qwen3-Embedding-4B
    path: ~/.cache/huggingface/...
    format: safetensors
    backend: vllm
    type: embedding
    vram_estimate: 8

  mistral-7b-q5:
    source: /home/simon/models/mistral-7b-v0.3.Q5_K_M.gguf
    path: /home/simon/models/mistral-7b-v0.3.Q5_K_M.gguf
    format: gguf
    backend: llama.cpp
    type: chat
    vram_estimate: 5
```

---

## Named Setups (optional convenience)

For common workflows, you can name a setup:

```yaml
setups:
  voice:
    model: nemotron-nano
    expose_tunnel: true
    note: Voice assistant

  rag:
    models:
      - mistral-7b-q5
      - qwen3-embed-0.6b
    note: Chat + embeddings together
```

But these are just shortcuts. The system can run any model with zero setup.

---

## User Interactions

### Download a model

```yaml
TUI: Search "qwen 7b instruct"
     → Shows options with VRAM estimates
     → Pick one
     → Downloads, auto-registers, ready to use
```

### Switch models

```yaml
TUI: See list of registered models
     → Pick one
     → Current model sleeps (or stops)
     → New model starts with optimal config
     → ~2 seconds
```

### API equivalent

```text
POST /models/activate
body: { "model": "qwen2.5-7b-awq" }
→ Done. No flags, no config.
```

---

## API Design

```text
GET  /status
     → { active: "qwen2.5-7b-awq", sleeping: [...], resources: {...} }

GET  /models
     → [{ id: "qwen2.5-7b-awq", type: "chat", vram: 10, status: "active" }, ...]

POST /models/activate
     body: { model: "qwen3-embed-4b" }
     → { success: true }

POST /models/sleep
     → { success: true }

GET  /hf/search?q=qwen+embedding
     → [{ repo, files, recommended_file, vram_estimate }]

POST /hf/download
     body: { repo: "...", file: "..." }
     → Downloads, auto-registers, returns model id

GET  /resources
     → { vram: { used, total }, ram: { used, total } }
```

---

## Internal: How Config Gets Built

When activating a model, the system builds config from rules:

```python
def build_config(model: Model, hardware: Hardware) -> Config:
    config = Config()

    # Backend from format
    config.backend = BACKEND_RULES[model.format]

    # Base optimizations for hardware
    if config.backend == "vllm":
        config.add("--enable-sleep-mode")  # Always, single GPU
        config.add(f"--kv_offloading_size {hardware.ram_gb // 3}")
        config.add("--async-scheduling")
        config.add("--num-scheduler-steps 10")
        config.env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

        # Quant-specific
        if model.quant == "awq":
            config.add("--quantization awq")

        # Type-specific
        if model.type == "embedding":
            config.add("--task embed")
            config.port = 8085

    elif config.backend == "llama.cpp":
        config.add(f"-ngl {999}")  # Full GPU offload
        config.add("--flash-attn")

        if model.type == "embedding":
            config.add("--embedding")
            config.port = 8085

    return config
```

User never sees this. They just pick a model.

---

## TUI Mockup

```text
┌─ Model Manager ─────────────────────────────────────────────────┐
│                                                                 │
│  ACTIVE                              VRAM ████████░░ 9.2/12 GB  │
│  ● qwen2.5-7b-awq                    RAM  ██░░░░░░░░ 18/94 GB   │
│    chat • vLLM • :8000                                          │
│                                                                 │
│  SLEEPING (instant wake)                                        │
│  ◐ nemotron-nano — ~0.8s                                       │
│                                                                 │
├─ MODELS ────────────────────────────────────────────────────────┤
│  qwen2.5-7b-awq        chat       10GB   ● active              │
│  nemotron-nano         chat        9GB   ◐ sleeping            │
│  qwen3-embed-4b        embedding   8GB   ○                     │
│  mistral-7b-q5         chat        5GB   ○                     │
└─────────────────────────────────────────────────────────────────┘
 [enter] activate  [d]ownload  [q]uit
```

---

## Existing Infrastructure Integration

| Existing | How it integrates |
|----------|-------------------|
| `~/vllm/` | Manager generates docker-compose or env from model config |
| `~/github/calling-nemo/` | Can register as "nemotron" model |
| Cloudflare tunnel | Manager can toggle tunnel exposure per model |

---

## References

### Optimization Sources (in ~/github/vllm-articles/)

- `sleep-mode.md` — 18-200x faster than restart
- `kv-offloading-connector.md` — Use your 94GB RAM
- `blackwell-inferencemax.md` — RTX 5070 optimizations
- `RECOMMENDATIONS.md` — All the quick wins

### External

- [vLLM Docs](https://docs.vllm.ai/)
- [OpenTUI React](https://github.com/anomalyco/opentui/tree/main/packages/react)

---

## Changelog

- **2026-01-22**: Initial spec
- **2026-01-22**: Rewrote around zero-config philosophy
