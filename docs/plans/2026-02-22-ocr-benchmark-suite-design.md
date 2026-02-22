# OCR Benchmark Suite Design

**Date**: 2026-02-22
**Status**: Approved

## Goal

Unified OCR benchmark suite that runs the same evaluations used by GLM-OCR and the IDP Leaderboard against any model served via the model-manager. Triggered via `mm benchmark ocr`.

## Benchmarks

| Benchmark | Tests | Samples | Source |
|-----------|-------|---------|--------|
| OmniDocBench v1.5 | End-to-end doc parsing (text + tables + formulas) | 1,355 pages | opendatalab/OmniDocBench |
| OCRBench v1 | Text recognition, VQA, KIE, handwritten math | 1,000 QA pairs | Yuliang-Liu/MultimodalOCR |
| UniMER-Test | Formula recognition (image → LaTeX) | 23,757 formulas | opendatalab/UniMERNet |
| PubTabNet | Table recognition (image → HTML), TEDS scoring | 9,115 val tables | ibm-aur-nlp/PubTabNet |
| KIE (docext) | Key info extraction: Nanonets-KIE + Handwritten-Forms | 1,432 samples | NanoNets/docext |

## Architecture

Thin wrapper around upstream evaluation toolkits. We write the inference layer (send images to OpenAI-compatible endpoint, collect predictions). Each benchmark's own eval scripts compute metrics.

### CLI

```text
mm benchmark ocr                     # Run all against active model
mm benchmark ocr --bench omnidoc     # Single benchmark
mm benchmark ocr --bench ocrbench
mm benchmark ocr --bench unimer
mm benchmark ocr --bench tables
mm benchmark ocr --bench kie
mm benchmark ocr compare             # Compare scores across models
mm benchmark ocr setup               # Download datasets + clone repos
```

Flags: `--limit N` (cap samples per benchmark), `--resume` (skip completed predictions).

### File Layout

```text
benchmarks/
├── ocr-suite/
│   ├── __init__.py
│   ├── runner.py              # Main orchestrator
│   ├── inference.py           # Unified image → model → prediction
│   ├── setup.py               # Dataset download + repo cloning
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── omnidocbench.py
│   │   ├── ocrbench.py
│   │   ├── unimer.py
│   │   ├── tables.py
│   │   └── kie.py
│   └── configs/
│       └── suite.yaml
├── repos/                     # Upstream git clones
│   ├── OmniDocBench/
│   ├── UniMERNet/
│   ├── docext/
│   └── MultimodalOCR/         # Already exists
├── datasets/                  # HF dataset downloads
│   ├── omnidocbench/
│   ├── unimer-test/
│   └── pubtabnet/
└── ocrbench/                  # Already exists (kept as-is)
```

### Inference Layer

Single async function shared by all benchmarks:

```python
async def run_inference(client, base_url, model, image_path, prompt, max_tokens=4096):
    """Send image + prompt to OpenAI-compatible endpoint, return text."""
```

Each benchmark module implements:

- `get_samples(dataset_path, limit)` — yields (image_path, prompt, sample_id)
- `format_prediction(raw_response)` — benchmark-specific output formatting
- `run_eval(predictions_dir, ground_truth_path)` — calls upstream eval, returns scores
- `get_score_summary(scores)` — human-readable summary

### Results

Stored in `results/ocr-suite/{model}_{date}.json`:

```json
{
  "model": "glm-ocr:latest",
  "backend": "ollama",
  "date": "2026-02-22T14:30:00",
  "benchmarks": {
    "omnidocbench": {"overall": 94.62, "text": 97.1, "table": 91.2, "formula": 95.5},
    "ocrbench": {"score": 789, "max": 1000},
    "unimer": {"cdm": 96.5, "edit_dist": 0.04},
    "tables": {"teds": 88.3, "teds_s": 91.2},
    "kie": {"nanonets_kie": 85.2, "handwritten_forms": 72.1}
  }
}
```

### Eval Methods Per Benchmark

| Benchmark | Inference Prompt | Output Format | Eval Script |
|-----------|-----------------|---------------|-------------|
| OmniDocBench | "Convert this page to markdown" | .md files | upstream pdf_validation.py |
| OCRBench | Per-sample questions from JSON | Text answers | Containment check (existing) |
| UniMER-Test | "Convert this formula to LaTeX" | LaTeX strings | Edit distance + CDM (Docker) |
| PubTabNet | "Convert this table to HTML" | HTML strings | TEDS via table-recognition-metric |
| KIE (docext) | docext-managed prompts | Structured extraction | docext eval (LiteLLM) |

### Dependencies

New Python deps: `table-recognition-metric`, `rapidfuzz`, `Levenshtein`, `datasets` (HuggingFace).
docext installed from git clone. OmniDocBench CDM uses Docker image `sunyuefeng/omnidocbench-env:v1.5`.
