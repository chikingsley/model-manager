# LiveBench Results

## GLM-4.6V-Flash-Q4_K_M.gguf

**Status: INCOMPATIBLE** - Uses "thinking mode" that outputs to `reasoning_content` instead of `content`. LiveBench and standard OpenAI API tooling expects responses in `content` field, so benchmarking fails.

Manual spot-check on 6 questions (before discovering the issue):

- Zebra Puzzle: ~50%
- Spatial: ~33%

Note: GLM-4.6V has vision capabilities but the thinking mode architecture makes it incompatible with standard text benchmarks

---

## Ministral-3-14B-Instruct-2512-Q4_K_M.gguf

### Reasoning (100 questions, full LiveBench set)

| Task | Correct | Total | Accuracy |
|------|---------|-------|----------|
| Spatial | 46 | 50 | **92.0%** |
| Zebra Puzzle | 14 | 50 | **28.0%** |
| **Overall Reasoning** | **60** | **100** | **60.0%** |

Notes:

- Q4_K_M quantization (7.7GB), run via llama.cpp on RTX 5070
- ~47s avg per question
- Spatial score is strong; zebra puzzles are harder (multi-constraint logic)

---

## Qwen2.5-7B-Instruct-AWQ (vLLM)

**Status: BROKEN** - Produces garbage output (repeated "5 5 5 5...")

- GSM8K (lm-eval): 0%
- IFEval strict accuracy (lm-eval): 33%
- LiveBench: Not usable (garbage output)

---

## Running Benchmarks

```bash
# Run reasoning benchmarks on a GGUF model (uses port 19000)
./run_benchmark.sh /home/simon/models/GLM-4.6V-Flash-Q4_K_M.gguf 19000 reasoning

# Run math benchmarks
./run_benchmark.sh /home/simon/models/Qwen3-4B-Q4_K_M.gguf 19000 math

# Available categories: reasoning, math, coding, language, data_analysis, instruction_following
```

Note: Port 18000 is reserved for nemotron. Use 19000+ for benchmarks.
