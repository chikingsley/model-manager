#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lm_eval[vllm,api]",
#     "tenacity",
#     "langdetect",
#     "hf_transfer",
#     "immutabledict",
#     "httpx",
# ]
# ///
"""Run lm-eval quality benchmarks across all Ollama models.

Usage:
    uv run scripts/eval-all-ollama.py              # Full eval (gsm8k_cot + ifeval)
    uv run scripts/eval-all-ollama.py --quick       # 30 samples per task (testing)
    uv run scripts/eval-all-ollama.py --limit 100   # Custom sample limit
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.stdout.reconfigure(line_buffering=True)

OLLAMA_URL = "http://localhost:11434"
RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results" / "quality"
SUMMARY_FILE = RESULTS_DIR / "summary.json"

TASKS = "gsm8k_cot,ifeval"

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"


def get_ollama_models() -> list[str]:
    """Get all Ollama models."""
    resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    data = resp.json()
    return [m["name"] for m in data.get("models", [])]


def load_model(model: str) -> bool:
    """Load a model into Ollama and wait for it to be ready."""
    print(f"  Loading {model}...")
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
            timeout=120,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"  {RED}Failed to load: {e}{NC}")
        return False


def unload_model(model: str) -> None:
    """Unload a model from Ollama."""
    try:
        httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass
    time.sleep(3)


def run_eval(model: str, tasks: str, limit: int | None, output_path: Path) -> int:
    """Run lm-eval for a single model."""
    cmd = [
        "lm_eval",
        "--model", "local-chat-completions",
        "--tasks", tasks,
        "--model_args", f"model={model},base_url={OLLAMA_URL}/v1/chat/completions,num_concurrent=1,tokenized_requests=False",
        "--apply_chat_template",
        "--output_path", str(output_path),
        "--log_samples",
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    return subprocess.call(cmd)


def parse_results(output_path: Path) -> dict | None:
    """Parse lm-eval results JSON."""
    # lm-eval saves results in a nested directory structure
    for results_file in output_path.rglob("results.json"):
        with open(results_file) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Eval all Ollama models")
    parser.add_argument("--quick", action="store_true", help="30 samples per task")
    parser.add_argument("--limit", type=int, help="Samples per task")
    parser.add_argument("--models", help="Comma-separated model list (default: all)")
    args = parser.parse_args()

    limit = 30 if args.quick else args.limit

    # Get models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = get_ollama_models()

    if not models:
        print(f"{RED}No Ollama models found.{NC}")
        return 1

    print(f"{BOLD}{'═' * 60}{NC}")
    print(f"{BOLD}  lm-eval Quality Benchmark — All Ollama Models{NC}")
    print(f"{BOLD}{'═' * 60}{NC}")
    print(f"  Tasks: {TASKS}")
    print(f"  Limit: {limit or 'full'}")
    print(f"  Models ({len(models)}):")
    for m in models:
        print(f"    - {m}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}
    succeeded = 0
    failed = 0

    for i, model in enumerate(models, 1):
        print(f"\n{CYAN}[{i}/{len(models)}]{NC}")
        print(f"{BOLD}{'═' * 60}{NC}")
        print(f"{BOLD}  Evaluating: {model}{NC}")
        print(f"{BOLD}{'═' * 60}{NC}\n")

        # Load model
        if not load_model(model):
            print(f"  {RED}Skipping {model} — failed to load{NC}")
            failed += 1
            continue

        # Run eval
        model_safe = model.replace(":", "-").replace("/", "-")
        output_path = RESULTS_DIR / f"{model_safe}_{timestamp}"

        start = time.time()
        ret = run_eval(model, TASKS, limit, output_path)
        elapsed = time.time() - start

        if ret == 0:
            succeeded += 1
            results = parse_results(output_path)
            if results:
                # Extract key metrics
                r = results.get("results", {})
                model_summary = {
                    "model": model,
                    "gsm8k_cot_flexible": r.get("gsm8k_cot", {}).get("exact_match,flexible-extract", None),
                    "gsm8k_cot_strict": r.get("gsm8k_cot", {}).get("exact_match,strict-match", None),
                    "ifeval_inst_loose": r.get("ifeval", {}).get("inst_level_loose_acc,none", None),
                    "ifeval_inst_strict": r.get("ifeval", {}).get("inst_level_strict_acc,none", None),
                    "ifeval_prompt_loose": r.get("ifeval", {}).get("prompt_level_loose_acc,none", None),
                    "ifeval_prompt_strict": r.get("ifeval", {}).get("prompt_level_strict_acc,none", None),
                    "duration_s": round(elapsed),
                    "limit": limit,
                    "date": datetime.now().isoformat(),
                }
                summary[model] = model_summary

                # Print summary
                gsm8k = model_summary["gsm8k_cot_flexible"]
                ifeval = model_summary["ifeval_inst_strict"]
                gsm8k_str = f"{gsm8k*100:.1f}%" if gsm8k is not None else "N/A"
                ifeval_str = f"{ifeval*100:.1f}%" if ifeval is not None else "N/A"
                print(f"\n  {GREEN}Results for {model}:{NC}")
                print(f"    GSM8K (flex):     {gsm8k_str}")
                print(f"    IFEval (strict):  {ifeval_str}")
                print(f"    Time:             {int(elapsed)}s")
        else:
            failed += 1
            print(f"  {RED}Eval failed for {model}{NC}")

        # Unload before next model
        print(f"  Unloading {model}...")
        unload_model(model)

    # Save summary
    if summary:
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)

    # Final report
    print(f"\n\n{BOLD}{'═' * 60}{NC}")
    print(f"{BOLD}  Evaluation Complete{NC}")
    print(f"{BOLD}{'═' * 60}{NC}")
    print(f"  {GREEN}Succeeded:{NC} {succeeded}")
    if failed:
        print(f"  {RED}Failed:{NC} {failed}")

    if summary:
        print(f"\n  {'Model':<25} {'GSM8K':>7} {'IFEval':>7} {'Time':>6}")
        print(f"  {'─' * 25} {'─' * 7} {'─' * 7} {'─' * 6}")
        for model, s in summary.items():
            gsm8k = f"{s['gsm8k_cot_flexible']*100:.1f}%" if s["gsm8k_cot_flexible"] is not None else "N/A"
            ifeval = f"{s['ifeval_inst_strict']*100:.1f}%" if s["ifeval_inst_strict"] is not None else "N/A"
            dur = f"{s['duration_s']}s"
            print(f"  {model:<25} {gsm8k:>7} {ifeval:>7} {dur:>6}")

    print(f"\n  Results: {RESULTS_DIR}")
    if SUMMARY_FILE.exists():
        print(f"  Summary: {SUMMARY_FILE}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
