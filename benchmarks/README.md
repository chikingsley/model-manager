# Model Benchmarks

Benchmark operations are managed through the `mm` CLI.

## Quick Reference

| Benchmark | What it tests | Run time | Command |
|-----------|---------------|----------|---------|
| Runtime benchmark | Throughput + latency on active model | ~1-3 min | `mm benchmark run` |
| OCR suite | OCR across 5 benchmarks (text, tables, formulas, KIE) | ~4-8 hours | `mm benchmark ocr` |
| OCR suite (quick) | Same, limited samples | ~5-10 min | `mm benchmark ocr --limit 5` |
| SWE-bench Lite | Code agent issue resolution | ~5-15 min/issue | `mm benchmark swebench ollama --limit 5` |

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

## OCR Benchmark Suite

Comprehensive OCR evaluation covering 5 benchmarks:

| Benchmark | Tests | Samples |
|-----------|-------|---------|
| OCRBench v1 | Text recognition, VQA, KIE, math | 1,000 |
| OmniDocBench v1.5 | End-to-end doc parsing | 1,355 |
| UniMER-Test | Formula recognition | 23,757 |
| PubTabNet | Table recognition (TEDS) | 9,115 |
| KIE | Key info extraction | 1,432 |

### First-Time Setup

```bash
cd /home/simon/docker/model-manager
mm benchmark ocr setup    # Clones repos + downloads HF datasets (~10GB)
```

### Running

```bash
# Run all benchmarks against active model
mm benchmark ocr

# Run specific benchmark
mm benchmark ocr --bench ocrbench
mm benchmark ocr --bench omnidoc
mm benchmark ocr --bench unimer
mm benchmark ocr --bench tables
mm benchmark ocr --bench kie

# Quick smoke test (5 samples per benchmark)
mm benchmark ocr --limit 5

# Resume interrupted run
mm benchmark ocr --resume

# Compare results across models
mm benchmark ocr compare
```

Results saved to `results/ocr-suite/{model}_{timestamp}/results.json`.

## OCRBench (standalone)

The original standalone OCRBench scripts remain in `benchmarks/ocrbench/`:

```bash
cd /home/simon/docker/model-manager
uv run benchmarks/ocrbench/eval_openai.py --base-url http://localhost:8000/v1
```

Results notes: `benchmarks/ocrbench/BENCHMARK_NOTES.md`.
