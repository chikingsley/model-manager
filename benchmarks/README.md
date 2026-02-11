# Model Benchmarks

Benchmark operations are managed through the `mm` CLI.

## Quick Reference

| Benchmark | What it tests | Run time | Command |
|-----------|---------------|----------|---------|
| Runtime benchmark | Throughput + latency on active model | ~1-3 min | `uv run mm benchmark run` |
| SWE-bench Lite | Code agent issue resolution | ~5-15 min/issue | `uv run mm benchmark swebench ollama --limit 5` |
| Benchmark source sync | Pull latest benchmark repos | ~seconds | `uv run mm benchmark sync all` |
| Benchmark drift check | Check if repos are behind | ~seconds | `uv run mm benchmark sync --check all` |

## Benchmark Source Management

Tracked sources are declared in `benchmarks/sources.yaml`.

```bash
cd /home/simon/github/model-manager

# Show tracked repositories
uv run mm benchmark sources

# Pull latest from origin for all sources
uv run mm benchmark sync all

# Check whether tracked repos are behind (no pull)
uv run mm benchmark sync --check all

# Sync a subset
uv run mm benchmark sync swe-agent livebench
```

Default tracked sources:

- `swe-agent` -> `/home/simon/github/SWE-agent`
- `livebench` -> `/home/simon/github/livebench`
- `multimodal-ocr` -> `/home/simon/github/MultimodalOCR`

## Runtime Benchmarking

```bash
cd /home/simon/github/model-manager

# Benchmark whichever model is currently active in mm
uv run mm benchmark run

# Compare all saved benchmark/profile data in models.yaml
uv run mm benchmark compare
```

## SWE-bench Lite

Real GitHub issue resolution benchmark via SWE-agent.

```bash
cd /home/simon/github/model-manager

# Quick run
uv run mm benchmark swebench ollama

# Backend/model/limit control
uv run mm benchmark swebench vllm --limit 5
uv run mm benchmark swebench ollama --model ministral-3:8b --limit 10
uv run mm benchmark swebench llamacpp --limit 3
```

Backends:

- `ollama` -> `localhost:11434`
- `vllm` -> `localhost:8000`
- `llamacpp` -> `localhost:8090`

Note: First SWE-bench run will download Docker environments.

## OCRBench

OCRBench scripts remain in `benchmarks/ocrbench/`:

```bash
cd /home/simon/github/model-manager
uv run benchmarks/ocrbench/eval_openai.py --base-url http://localhost:8000/v1
```

Results notes: `benchmarks/ocrbench/BENCHMARK_NOTES.md`.
