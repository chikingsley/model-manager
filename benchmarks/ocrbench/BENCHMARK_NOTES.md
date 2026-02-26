# OCRBench Evaluation Notes

**Hardware**: RTX 5070 (12GB VRAM)
**Last Updated**: 2026-02-03

## Results Summary

### Closed-Source API Models

| Model | Score | Notes |
|-------|-------|-------|
| **gemini-3-flash-preview** | **907/1000** | Best overall, dominates most categories |
| mistral-large-latest (VQA) | 460/1000 | 345 samples failed due to rate limits (70% on working samples) |
| mistral-ocr-latest | 392/1000 | Pure extraction, not VQA |

### Local Models

## Strategy

1. Start with model's max supported context length
2. Back off if CUDA OOM occurs
3. Find sweet spot that minimizes both token-limit errors and OOM errors

| Model | Context | Errors | Score | Notes |
|-------|---------|--------|-------|-------|
| **zai-org/GLM-OCR** | 4096 | 0 | **789/1000** | Ollama, 0.9B params, best local |
| nanonets/Nanonets-OCR-s | 2048 | 0 | 729/1000 | GGUF via llama.cpp |
| nanonets/Nanonets-OCR2-1.5B-exp | 8192 | 24 | 727/1000 | vLLM, see tuning below |
| lightonai/LightOnOCR-2-1B | 8192 | 0 | 679/1000 | vLLM, 1B model fits easily |
| lightonai/LightOnOCR-1B-1025 | 8192 | 0 | 586/1000 | vLLM, older v1 model |
| deepseek-ai/DeepSeek-OCR | 2048 | 0 | 431/1000 | GGUF via llama.cpp |
| nvidia/Nemotron-Parse-v1.2 | 9000 | 0 | 396/1000 | Custom backend, document parser (not VQA) |
| mistral-ocr-latest | N/A | 299 | 392/1000 | Mistral API, pure extraction (not VQA) |

## Models That Failed to Load in vLLM

| Model | Issue |
|-------|-------|
| echo840/MonkeyOCR-pro-1.2B | Custom architecture not recognized by vLLM |

## Context Length Tuning: Nanonets-OCR2-1.5B-exp

Model supports up to 32K (Qwen2-VL base). Tested on 12GB VRAM:

| Context | GPU Util | Errors | Score | Error Type |
|---------|----------|--------|-------|------------|
| 4096 | 0.95 | 76 | 680 | Token limit exceeded |
| 8192 | 0.90 | 24 | 727 | Token limit exceeded |
| 16384 | 0.95 | 75 | 715 | CUDA OOM |

**Conclusion**: 8192 is optimal for this model on 12GB VRAM.

Some images produce 10K-16K+ tokens due to high resolution. These always fail regardless of settings.

## Context Length Tuning: LightOnOCR-2-1B

1B model with Mistral base. Smaller footprint = more KV cache headroom.

| Context | GPU Util | Errors | Score | Error Type |
|---------|----------|--------|-------|------------|
| 16384 | 0.95 | N/A | N/A | OOM during warmup |
| 8192 | 0.90 | 0 | 679 | None |

**Conclusion**: 8192 works well. Model produces fewer tokens per image than Qwen-based models.

Note: LightOnOCR weak on handwriting (29/50), digit strings (28/50), and math (14/100). Strong on key info extraction (160/200).

## Mistral OCR (API)

Mistral OCR is a **pure document extraction tool**, not a VQA model. It doesn't answer questions - it extracts text.

| Category | Score | Notes |
|----------|-------|-------|
| Key Information Extraction | **190/200** | 95%! Its specialty |
| Doc-oriented VQA | 63/200 | Doesn't answer, just extracts |
| Scene Text-centric VQA | 62/200 | Same issue |
| Regular Text Recognition | 33/50 | Decent |
| Handwriting | 3/50 | Poor |
| Math | 1/100 | Very poor |

**Conclusion**: Use Mistral OCR for structured document parsing (invoices, forms, receipts). For general OCR + VQA, use the Nanonets models.

## Model-Specific vLLM Args

```bash
# Nanonets-OCR2-1.5B-exp (Qwen2-VL based)
MODEL=nanonets/Nanonets-OCR2-1.5B-exp
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

# LightOnOCR-2-1B (Mistral based)
MODEL=lightonai/LightOnOCR-2-1B
EXTRA_ARGS=--limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 --no-enable-prefix-caching

# MonkeyOCR-pro-1.2B
MODEL=echo840/MonkeyOCR-pro-1.2B
EXTRA_ARGS=--trust-remote-code
```

## Pending Evaluations

- [x] **gemini-3-flash-preview** (907/1000) - Best overall!
- [x] **zai-org/GLM-OCR** (789/1000) - Best local model!
- [x] mistral-large-latest VQA (460/1000) - Rate limited, needs re-run
- [x] lightonai/LightOnOCR-2-1B (679/1000)
- [x] lightonai/LightOnOCR-1B-1025 (586/1000)
- [ ] ~~echo840/MonkeyOCR-pro-1.2B~~ (vLLM incompatible)
- [ ] docling (pipeline OCR - needs separate eval script)
- [ ] paddleocr (pipeline OCR - needs separate eval script)

## GLM-OCR (Ollama)

Requires Ollama 0.15.5+ (pre-release as of 2026-02-03).

```bash
docker run -d --gpus all --name ollama -p 11434:11434 \
  -v /home/simon/docker/ollama:/root/.ollama ollama/ollama:0.15.5-rc1
docker exec ollama ollama pull glm-ocr
```

**GLM-OCR Results (789/1000)**:

| Category | Score | Max | % |
|----------|-------|-----|---|
| Regular Text Recognition | 49 | 50 | 98% |
| Irregular Text Recognition | 49 | 50 | 98% |
| Artistic Text Recognition | 48 | 50 | 96% |
| Handwriting Recognition | 43 | 50 | 86% |
| Digit String Recognition | 47 | 50 | 94% |
| Non-Semantic Text Recognition | 50 | 50 | 100% |
| **Text Recognition Total** | **286** | **300** | **95%** |
| Scene Text-centric VQA | 168 | 200 | 84% |
| Doc-oriented VQA | 102 | 200 | 51% |
| Key Information Extraction | 152 | 200 | 76% |
| Handwritten Math Expression | 81 | 100 | 81% |

---

## Gemini 3 Flash (API)

**Score: 907/1000** - Best overall across all models tested.

| Category | Score | Max | % |
|----------|-------|-----|---|
| Regular Text Recognition | 49 | 50 | 98% |
| Irregular Text Recognition | 48 | 50 | 96% |
| Artistic Text Recognition | 49 | 50 | 98% |
| Handwriting Recognition | 46 | 50 | 92% |
| Digit String Recognition | 39 | 50 | 78% |
| Non-Semantic Text Recognition | 50 | 50 | 100% |
| **Text Recognition Total** | **281** | **300** | **94%** |
| Scene Text-centric VQA | 187 | 200 | 93.5% |
| Doc-oriented VQA | 193 | 200 | 96.5% |
| Key Information Extraction | 189 | 200 | 94.5% |
| Handwritten Math Expression | 57 | 100 | 57% |

**Strengths**: Dominates VQA tasks (93-97%), excellent text recognition across all types.
**Weaknesses**: Handwritten math (57%) - GLM-OCR is significantly better here (81%).

---

## NVIDIA Nemotron-Parse v1.2 (Custom Backend)

**Score: 396/1000** — Document parser, not a VQA model. Runs via custom FastAPI backend on port 8097.

Settings: `max_new_tokens=9000`, `repetition_penalty=1.1`, `do_sample=false`, server-side postprocessing via NVIDIA's `postprocessing.py`.

| Category | Score | Max | % |
|----------|-------|-----|---|
| Regular Text Recognition | 36 | 50 | 72% |
| Irregular Text Recognition | 17 | 50 | 34% |
| Artistic Text Recognition | 2 | 50 | 4% |
| Handwriting Recognition | 5 | 50 | 10% |
| Digit String Recognition | 28 | 50 | 56% |
| Non-Semantic Text Recognition | 47 | 50 | 94% |
| **Text Recognition Total** | **135** | **300** | **45%** |
| Scene Text-centric VQA | 44 | 200 | 22% |
| Doc-oriented VQA | 54 | 200 | 27% |
| Key Information Extraction | **162** | 200 | **81%** |
| Handwritten Math Expression | 1 | 100 | 1% |

**Strengths**: Key Information Extraction (81%, beats GLM-OCR's 76%), Non-Semantic Text (94%).
**Weaknesses**: Not designed for VQA, handwriting, or artistic text. Outputs structured document markdown, not answers to questions.

**Note**: OmniDocBench, UniMER, Tables, KIE benchmarks not completed (run interrupted). 143/1355 OmniDoc predictions saved. Full results in `results/ocr-suite/nvidia_NVIDIA-Nemotron-Parse-v1.2_20260224_202151/`.

---

## Mistral Large VQA (API)

**Score: 460/1000** (impacted by rate limits - 345/1000 samples failed with 429 errors)

Effective accuracy on working samples: 460/655 = **70.2%**

| Category | Score | Max | % |
|----------|-------|-----|---|
| Regular Text Recognition | 47 | 50 | 94% |
| Irregular Text Recognition | 7 | 50 | 14% |
| Artistic Text Recognition | 30 | 50 | 60% |
| Handwriting Recognition | 17 | 50 | 34% |
| Digit String Recognition | 13 | 50 | 26% |
| Non-Semantic Text Recognition | 22 | 50 | 44% |
| **Text Recognition Total** | **136** | **300** | **45%** |
| Scene Text-centric VQA | 142 | 200 | 71% |
| Doc-oriented VQA | 103 | 200 | 51.5% |
| Key Information Extraction | 71 | 200 | 35.5% |
| Handwritten Math Expression | 8 | 100 | 8% |

**Note**: Score heavily impacted by rate limiting. Re-run with `--resume` after limits reset to get accurate results.

---

## Category-by-Category Comparison (Best in Category)

| Category | Best Model | Score |
|----------|------------|-------|
| Regular Text | Nanonets-OCR2-1.5B | 50/50 |
| Irregular Text | GLM-OCR, Nanonets-OCR-s | 49/50 |
| Artistic Text | **Gemini 3 Flash**, GLM-OCR | 49/50 |
| Handwriting | Nanonets-OCR2-1.5B | 47/50 |
| Digit String | GLM-OCR | 47/50 |
| Non-Semantic Text | **Gemini 3 Flash**, GLM-OCR, Nanonets-OCR2 | 50/50 |
| Scene Text VQA | **Gemini 3 Flash** | 187/200 |
| Doc-oriented VQA | **Gemini 3 Flash** | 193/200 |
| Key Info Extraction | **Gemini 3 Flash** | 189/200 |
| Handwritten Math | **GLM-OCR** | 81/100 |

**Takeaways**:

- **Gemini 3 Flash**: Best overall (907), dominates VQA categories (93-97%)
- **GLM-OCR**: Best local model (789), dominates handwritten math (81% vs 57% Gemini)
- **Nanonets-OCR2-1.5B**: Best local model for document VQA (148/200)
- **LightOnOCR-1B-1025**: Best at key info extraction among local models (175/200)
