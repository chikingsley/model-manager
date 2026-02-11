# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lm_eval[vllm,api]",
#     "tenacity",
#     "langdetect",
#     "hf_transfer",
#     "immutabledict",
# ]
# ///
"""
Quality Benchmark using lm-evaluation-harness

Runs quality benchmarks against a running vLLM server.

Usage:
    uv run scripts/quality_bench.py                          # Quick test (30 samples)
    uv run scripts/quality_bench.py --full                   # Full benchmark
    uv run scripts/quality_bench.py --url http://localhost:18000  # Different server
    uv run scripts/quality_bench.py --tasks gsm8k_cot,ifeval      # Specific tasks

Available Tasks (work with chat API):
    gsm8k_cot       - Math reasoning with chain-of-thought
    ifeval          - Instruction following evaluation
    gpqa_main_cot_zeroshot  - Graduate-level science QA
    arc_challenge_llama     - Reasoning (needs llama template)

Tasks that DON'T work with chat API (need loglikelihood):
    mmlu, hellaswag, winogrande, arc_challenge
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx

RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results" / "quality"

# Tasks that work with generate_until (chat API compatible)
DEFAULT_TASKS = "gsm8k_cot,ifeval"
FULL_TASKS = "gsm8k_cot,ifeval,gpqa_main_cot_zeroshot"


def get_model_name(url: str) -> str:
    """Get model name from vLLM server."""
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=10)
        data = resp.json()
        if data.get("data"):
            return data["data"][0]["id"]
    except Exception:
        pass
    return "unknown"


def run_benchmark(
    url: str,
    model: str,
    tasks: str,
    limit: int | None,
    output_path: Path | None,
) -> int:
    """Run lm-eval benchmark."""
    cmd = [
        "lm_eval",
        "--model", "local-chat-completions",
        "--tasks", tasks,
        "--model_args", f"model={model},base_url={url}/v1/chat/completions,num_concurrent=8,tokenized_requests=False",
        "--apply_chat_template",
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    if output_path:
        cmd.extend(["--output_path", str(output_path)])
        cmd.append("--log_samples")

    print(f"\n{'='*60}")
    print(f"Quality Benchmark: {model}")
    print(f"Tasks: {tasks}")
    print(f"Limit: {limit or 'full'}")
    print(f"{'='*60}\n")

    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Quality Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated tasks")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (no limit)")
    parser.add_argument("--quick", action="store_true", help="Quick test (30 samples)")
    parser.add_argument("--limit", type=int, help="Number of samples per task")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    args = parser.parse_args()

    # Get model name
    model = get_model_name(args.url)
    if model == "unknown":
        print(f"Error: Could not connect to {args.url}")
        sys.exit(1)

    print(f"Detected model: {model}")

    # Determine limit
    if args.full:
        limit = None
        tasks = FULL_TASKS
    elif args.quick:
        limit = 30
        tasks = args.tasks
    elif args.limit:
        limit = args.limit
        tasks = args.tasks
    else:
        limit = 30  # Default to quick
        tasks = args.tasks

    # Output path
    output_path = None
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model.split("/")[-1].replace(".", "_")
        output_path = RESULTS_DIR / f"{model_short}_{timestamp}"

    # Run benchmark
    ret = run_benchmark(args.url, model, tasks, limit, output_path)

    if output_path and ret == 0:
        print(f"\nResults saved to: {output_path}")

    return ret


if __name__ == "__main__":
    sys.exit(main())
