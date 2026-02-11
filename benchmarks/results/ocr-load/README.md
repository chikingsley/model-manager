# OCR Load Results (GLM-OCR on vLLM)

Date: 2026-02-11  
Hardware: RTX 5070 12GB (single GPU)  
Workload: OCR-heavy receipt image (`SROIE/image/X51006414427.jpg`)

## Files

- `glm_ocr_budget_4096.csv` / `.json`
- `glm_ocr_budget_8192_compact.csv` / `.json`
- `glm_ocr_budget_8192_breakpoint_timeout90.csv` / `.json`
- `glm_ocr_budget_12288_short.csv` / `.json`
- `glm_ocr_budget_16384_short.csv` / `.json`

## Key Findings

1. `max-num-batched-tokens=4096` is invalid for this heavy OCR workload:
   - 100% failures (HTTP 400 encoder cache overflow), all tested concurrency points.
2. `8192` is the best balanced production setting in this run:
   - 0 errors through concurrency 32.
   - Best or near-best req/s versus higher budgets.
3. Increasing budget above `8192` did not improve throughput on this workload:
   - `12288` was slower at both 8 and 16 concurrency.
   - `16384` recovered some speed versus 12288, but did not clearly beat 8192.

## Snapshot Table

| Budget | Concurrency | Errors | req/s |
|---|---:|---:|---:|
| 4096 | 8 | 8/8 | 0.962 (all failed) |
| 4096 | 16 | 16/16 | 2.044 (all failed) |
| 8192 | 8 | 0/8 | 0.434 |
| 8192 | 16 | 0/16 | 0.493 |
| 8192 | 24 | 0/24 | 0.396 |
| 8192 | 32 | 0/32 | 0.504 |
| 12288 | 8 | 0/8 | 0.241 |
| 12288 | 16 | 0/16 | 0.299 |
| 16384 | 8 | 0/8 | 0.406 |
| 16384 | 16 | 0/16 | 0.448 |

## Production Breakpoint (8192 budget, timeout 90s)

Workload:
- Same heavy OCR image (SROIE receipt)
- Concurrency sweep: 32, 40, 48, 56, 64
- Timeout: 90s/request

Results:

| Budget | Concurrency | Errors | req/s | Notes |
|---|---:|---:|---:|---|
| 8192 | 32 | 0/32 | 0.459 | Stable |
| 8192 | 40 | 0/40 | 0.586 | Stable |
| 8192 | 48 | 0/48 | 0.619 | Stable |
| 8192 | 56 | 0/56 | 0.627 | Stable (near cap) |
| 8192 | 64 | 19/64 | 0.710 | Timeout-limited; practical failure starts |

Interpretation:
- For this heavy OCR class, practical limit is around **56 concurrent** with a 90s SLA.
- At 64, throughput rises, but reliability degrades (timeouts).
- Recommended production cap for heavy documents: **40-56 concurrent**, depending on latency SLA.

## Current Recommendation

Keep:
- `--max-num-batched-tokens 8192`
- `--max-num-seqs 16`

This remains stable for heavy OCR while preserving throughput and low error risk.
