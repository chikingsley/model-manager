# Model Manager
# Run `just` for status, `just --list` for all recipes

set shell := ["bash", "-uc"]
set dotenv-load := false

default: status

# ─── Modes ────────────────────────────────────────────────────────────────

# Show current status
status:
    uv run mm
alias s := status

# Activate Ollama
ollama model="":
    uv run mm ollama {{ model }}
alias ol := ollama

# Activate OCR (vLLM)
ocr:
    uv run mm ocr
alias o := ocr

# Activate chat (vLLM)
chat model="":
    uv run mm chat {{ model }}
alias c := chat

# Activate llama.cpp (GGUF)
llama model="":
    uv run mm llama {{ model }}
alias l := llama

# Activate voice stack
voice:
    uv run mm voice
alias v := voice

# Activate embeddings
embed:
    uv run mm embed
alias e := embed

# Activate max performance mode
perf model="":
    uv run mm perf {{ model }}
alias p := perf

# Stop all GPU services
stop:
    uv run mm stop

# List registered models
models:
    uv run mm models

# ─── API ──────────────────────────────────────────────────────────────────

# Deploy API container (build + start)
deploy:
    docker compose -f docker-compose.api.yml up -d --build --force-recreate
    @sleep 5
    @just smoke

# Start API container
up:
    docker compose -f docker-compose.api.yml up -d

# Stop API container
down:
    docker compose -f docker-compose.api.yml down

# Rebuild API container
rebuild:
    docker compose -f docker-compose.api.yml build --no-cache

# API container logs
logs-api:
    docker logs -f model-manager-api --tail 50

# Run API locally (dev mode with reload)
dev:
    uv run uvicorn model_manager.api.server:app --reload --host 0.0.0.0 --port 8890

# ─── Checks ───────────────────────────────────────────────────────────────

# Smoke test the live API
smoke:
    uv run scripts/smoke_test.py

# Run all checks (lint + types + tests)
ci: lint check test

# Lint
lint:
    uv run ruff check src/

# Format
fmt:
    uv run ruff format src/

# Type check
check:
    uv run ty check src/

# Run tests
test *args:
    uv run pytest {{ args }}

# Tests with coverage
test-cov:
    uv run pytest --cov=src/model_manager --cov-report=term-missing

# Lint + format + type check (fix what you can)
fix:
    uv run ruff check --fix src/
    uv run ruff format src/

# ─── TUI ──────────────────────────────────────────────────────────────────

# Run the TUI
tui:
    cd tui && ./mm-tui

# Build the TUI
tui-build:
    cd tui && bun run build

# ─── Info ─────────────────────────────────────────────────────────────────

# GPU usage
gpu:
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv

# Running model containers
ps:
    docker ps --format "table {{{{.Names}}}}\t{{{{.Status}}}}\t{{{{.Ports}}}}" | grep -E "nemotron|vllm|llama|ollama|model-manager"

# Quick endpoint health check
health:
    @curl -sf http://localhost:8890/health && echo "API      ✓" || echo "API      ✗"
    @curl -sf http://localhost:8000/health && echo "vLLM     ✓" || echo "vLLM     ✗"
    @curl -sf http://localhost:8090/health && echo "llama    ✓" || echo "llama    ✗"
    @curl -sf http://localhost:11434/api/tags > /dev/null && echo "Ollama   ✓" || echo "Ollama   ✗"

# Tail vLLM logs
logs-vllm:
    docker logs -f vllm --tail 50

# Tail Ollama logs
logs-ollama:
    docker logs -f ollama --tail 50

# Install dependencies
install:
    uv sync

# Run benchmark
benchmark:
    uv run mm benchmark run
