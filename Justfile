# Model Manager - Justfile
# Usage: just <recipe> or just <alias>

set shell := ["bash", "-uc"]

# Default recipe - show status
default: status

# ═══════════════════════════════════════════════════════════════════════════
# Model Switching (via Python CLI)
# ═══════════════════════════════════════════════════════════════════════════

# Activate voice stack (nemotron: LLM + ASR + TTS)
voice:
    uv run mm voice
alias v := voice

# Activate OCR (DeepSeek-OCR-2 by default, pass model to override)
ocr:
    uv run mm ocr
alias o := ocr

# Activate embeddings (Qwen3-Embedding-4B)
embed:
    uv run mm embed
alias e := embed

# Activate chat (vLLM with default model)
chat model="":
    uv run mm chat {{ model }}
alias c := chat

# Activate MAX PERFORMANCE mode (all optimizations)
perf model="":
    uv run mm perf {{ model }}
alias p := perf

# Activate llama.cpp (GGUF models)
llama model="":
    uv run mm llama {{ model }}
alias l := llama

# Activate Ollama
ollama model="":
    uv run mm ollama {{ model }}
alias ol := ollama

# Stop model services
stop:
    uv run mm stop

# Show current status
status:
    uv run mm
alias s := status

# List available GGUF models
models:
    uv run mm models

# ═══════════════════════════════════════════════════════════════════════════
# API Server
# ═══════════════════════════════════════════════════════════════════════════

# Run API server (port 8888)
serve:
    uv run mm serve

# Run API server in Docker (host port 8890)
serve-container:
    docker compose -f docker-compose.api.yml up -d --build

# Stop API server container
serve-container-stop:
    docker compose -f docker-compose.api.yml down

# Run API server with reload
serve-dev:
    uv run uvicorn model_manager.api.server:app --reload --host 0.0.0.0 --port 8888

# ═══════════════════════════════════════════════════════════════════════════
# Development
# ═══════════════════════════════════════════════════════════════════════════

# Install dependencies
install:
    uv sync

# Run tests
test *args:
    uv run pytest {{ args }}

# Run tests with coverage
test-cov:
    uv run pytest --cov=src/model_manager --cov-report=term-missing

# Lint with ruff
lint:
    uv run ruff check src/

# Format with ruff
fmt:
    uv run ruff format src/

# Type check
check:
    uv run ty check --error-on-warning src/model_manager/

# All checks (lint + type + test)
ci: lint check test

# ═══════════════════════════════════════════════════════════════════════════
# TUI
# ═══════════════════════════════════════════════════════════════════════════

# Run the TUI
tui:
    cd tui && ./mm-tui

# Build the TUI
tui-build:
    cd tui && bun run build

# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

# Show GPU usage
gpu:
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv

# List docker containers
ps:
    docker ps --format "table {{{{.Names}}}}\t{{{{.Status}}}}\t{{{{.Ports}}}}" | grep -E "nemotron|vllm|qwen|llama|ollama"

# Tail nemotron logs
logs-nemotron:
    docker logs -f nemotron --tail 50

# Tail vllm logs
logs-vllm:
    docker logs -f vllm --tail 50

# Tail ollama logs
logs-ollama:
    docker logs -f ollama --tail 50

# Quick health check on endpoints
health:
    @echo "Checking endpoints..."
    @curl -sf http://localhost:8000/health && echo "localhost:8000 (vllm) ✓" || echo "localhost:8000 (vllm) ✗"
    @curl -sf http://localhost:8090/health && echo "localhost:8090 (llama) ✓" || echo "localhost:8090 (llama) ✗"
    @curl -sf http://localhost:11434/api/tags && echo "localhost:11434 (ollama) ✓" || echo "localhost:11434 (ollama) ✗"
    @curl -sf http://localhost:8888/health && echo "localhost:8888 (api) ✓" || echo "localhost:8888 (api) ✗"

# Run optimization benchmark
benchmark:
    uv run mm benchmark run

# Sync benchmark sources (all by default)
benchmark-sync sources="all":
    uv run mm benchmark sync {{ sources }}
