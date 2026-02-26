# Model Manager

Zero config local LLM infrastructure. Download a model, it works.

## Philosophy

**Zero configuration. Zero flags. Zero bullshit.**

- Download a model from HuggingFace
- System auto-detects format, picks backend, applies optimal settings
- You just say "activate this model"

## Quick Start

```bash
# Show status
mm

# Activate Ollama
mm ollama

# Activate with specific model
mm ollama ministral-3:14b

# Activate llama.cpp with GGUF
mm llama Ministral-3-14B-Instruct-2512-Q4_K_M.gguf

# Activate vLLM
mm chat

# Activate SAM3 segmentation service
mm sam3

# MAX PERFORMANCE mode
mm perf

# Stop all
mm stop

# List available GGUF models
mm models
```

## Architecture

```text
                    ┌─────────────┐
                    │   TUI       │
                    │ (OpenTUI)   │
                    └──────┬──────┘
                           │ HTTP
    ┌──────────────┐       │       ┌─────────────┐
    │   mm CLI     │       ▼       │  External   │
    │  (Python)    │─────► API ◄───│  (curl/etc) │
    └──────────────┘       │       └─────────────┘
                           │
            ┌──────────────┴───────────────┐
            │      Core Python Modules     │
            │  containers.py  modes.py     │
            │  state.py       config.py    │
            └──────────────────────────────┘
```

## Usage

### CLI

```bash
mm                  # Show status
mm voice            # Activate voice stack (nemotron)
mm llama [model]    # Activate llama.cpp (GGUF models)
mm ollama [model]   # Activate Ollama
mm ocr [model]      # Activate OCR via vLLM (default: DeepSeek-OCR-2, also: GLM-OCR, Nemotron Parse)
mm chat [model]     # Activate vLLM chat
mm perf [model]     # MAX PERFORMANCE mode (all optimizations)
mm embed            # Activate embeddings
mm sam3             # Activate SAM3 segmentation service (official Meta SAM3)
mm stop             # Stop all model services
mm models           # List available GGUF models
mm serve            # Start API server (port 4001)
mm ollama-context <model>  # Test/save Ollama context profile
mm benchmark run    # Benchmark active model endpoint
mm benchmark compare  # Compare saved benchmark results
mm benchmark sources  # List tracked benchmark repos
mm benchmark sync all # Pull latest benchmark repos
mm benchmark swebench ollama --limit 5  # Run SWE-bench Lite
```

### API Server

```bash
# Start API in container (recommended)
docker compose -f docker-compose.api.yml up -d --build

# Stop
docker compose -f docker-compose.api.yml down

# Local endpoint
curl http://localhost:4001/health
```

Endpoints:

```text
GET  /                    # API metadata
GET  /health              # Health check
GET  /docs                # Swagger UI (interactive API docs)
GET  /redoc               # ReDoc API docs
GET  /openapi.json        # OpenAPI schema (for tooling/agents)
GET  /llms.txt            # LLM-oriented API manifest
GET  /llms-full.txt       # Expanded LLM-oriented context

GET  /status              # Current status
GET  /resources           # GPU/RAM usage
GET  /models              # List registered models
GET  /models/gguf         # List available GGUF files
GET  /capabilities        # Tested operational limits + docs pointers
GET  /capabilities/ocr    # OCR throughput breakpoints from benchmark runs

POST /activate/{mode}     # Activate a mode (voice, llama, ollama, ocr, chat, perf, embed, sam3, stop)
POST /stop                # Stop all services

GET  /ollama/models       # List Ollama models
POST /ollama/load/{model} # Load an Ollama model
```

### SAM3 Service

`mm sam3` starts a dedicated SAM3 segmentation backend from `services/sam3/` using the official Meta repository (`facebookresearch/sam3`).

- Local endpoint: `http://localhost:8095`
- Health: `GET /health`
- Inference: `POST /segment`
- Request body: `{"image_url"|"image_path", "prompt", "top_k"}`

Example:

```bash
curl -s http://localhost:8095/segment \
  -H 'content-type: application/json' \
  -d '{
    "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
    "prompt": "beignets",
    "top_k": 3
  }' | jq
```

Notes:

- `GET /docs` is the main Swagger page to share with teammates/agents.
- `GET /capabilities/ocr` returns the currently documented heavy-OCR concurrency envelope (including failure breakpoint data).
- `GET /llms.txt` and `GET /llms-full.txt` are model-facing docs endpoints for agent discovery.

### TUI

```bash
just tui
```

## Project Structure

```text
model-manager/
├── src/model_manager/
│   ├── __init__.py
│   ├── cli.py           # CLI entry point
│   ├── containers.py    # Docker operations
│   ├── state.py         # models.yaml management
│   ├── config.py        # Hardware detection, config building
│   ├── modes.py         # Mode activation logic
│   ├── benchmark_hub.py # Benchmark source + SWE-bench orchestration
│   └── api/
│       └── server.py    # FastAPI server
├── benchmarks/
│   ├── sources.yaml      # Benchmark repository tracking
│   └── README.md         # Benchmark workflow guide
├── tui/                  # Terminal UI (OpenTUI/Solid.js)
├── tests/
├── models.yaml           # Model registry + state
├── config.yaml           # Hardware + paths
├── Justfile
└── pyproject.toml
```

## Development

```bash
# Install dependencies
uv sync

# Run tests
just test

# Lint
just lint

# Format
just fmt

# Type check
just check

# All checks
just ci
```

## Configuration

### config.yaml

Hardware and paths (auto-detected, but can override):

```yaml
hardware:
  gpu: RTX 5070
  vram_gb: 12
  ram_gb: 94

paths:
  models: /home/simon/models
  vllm_compose: /home/simon/docker/vllm/docker-compose.yml
```

### models.yaml

Model registry (auto-managed):

```yaml
models:
  qwen3-4b:
    source: local
    path: /home/simon/models/Qwen3-4B-Q4_K_M.gguf
    format: gguf
    backend: llama.cpp

state:
  active: ollama
```

## See Also

- `docs/README.md` — Documentation map (active + archived)
- `docs/archive/SPEC.md` — Historical specification
- `~/github/vllm-articles/` — vLLM optimization notes
