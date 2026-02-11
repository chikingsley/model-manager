#!/usr/bin/env python3
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
"""Context-Quality Profiler: measures quality at each context size.

For each model, at each context level (4K, 8K, 16K, 32K, ...):
  1. Load model with that num_ctx
  2. Run quick quality eval (gsm8k_cot + ifeval, 100 samples each)
  3. If OOM → stop, that context size is the real limit under load
  4. Record: context, gsm8k%, ifeval%, tok/s, vram

Usage:
    uv run scripts/context-quality-profile.py                        # All models
    uv run scripts/context-quality-profile.py --models granite4:latest
    uv run scripts/context-quality-profile.py --limit 50             # Fewer samples (faster)
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
RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results" / "context_quality"

TASKS = "gsm8k_cot,ifeval"
SAMPLES_PER_TASK = 100  # default — enough to see trends, fast enough to be practical

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"


def get_ollama_models() -> list[str]:
    resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    return [m["name"] for m in resp.json().get("models", [])]


def get_model_max_context(model: str) -> int:
    """Get model's claimed max context from Ollama."""
    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/show", json={"name": model}, timeout=30)
        data = resp.json()
        model_info = data.get("model_info", {})
        for key in model_info:
            if "context" in key.lower():
                val = model_info[key]
                if isinstance(val, int):
                    return val
    except Exception:
        pass
    return 32768  # fallback


def generate_context_sizes(max_ctx: int) -> list[int]:
    """Generate power-of-2 context sizes up to max."""
    sizes = []
    current = 4096
    while current <= max_ctx:
        sizes.append(current)
        current *= 2
    if sizes and sizes[-1] < max_ctx:
        sizes.append(max_ctx)
    return sizes


def unload_model(model: str) -> None:
    try:
        httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass
    time.sleep(5)


def load_model_with_context(model: str, num_ctx: int) -> bool:
    """Load model with a specific num_ctx. Returns False if OOM."""
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": "Reply with only: OK",
                "stream": False,
                "options": {"num_ctx": num_ctx, "num_predict": 5},
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return True
        # Check for OOM in error
        text = resp.text.lower()
        if "out of memory" in text or "cuda error" in text:
            return False
        return False
    except Exception:
        return False


def get_vram_usage() -> int:
    """Get current GPU VRAM usage in MB."""
    try:
        import subprocess as sp
        out = sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip())
    except Exception:
        return 0


def run_eval_at_context(
    model: str, num_ctx: int, limit: int, output_path: Path
) -> dict | None:
    """Run lm-eval at a specific context size. Returns results or None on OOM."""

    # The model is already loaded with the right num_ctx from load_model_with_context.
    # lm-eval will use the already-loaded model via chat completions API.
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--tasks", TASKS,
        "--model_args", f"model={model},base_url={OLLAMA_URL}/v1/chat/completions,num_concurrent=1,tokenized_requests=False",
        "--apply_chat_template",
        "--limit", str(limit),
        "--output_path", str(output_path),
        "--log_samples",
    ]

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=None)
    elapsed = time.time() - start

    # Check for OOM in output
    combined = proc.stdout + proc.stderr
    oom_count = combined.lower().count("out of memory") + combined.lower().count("cuda error")

    if proc.returncode != 0:
        if oom_count > 0 and oom_count > limit // 2:
            return {"oom": True, "oom_count": oom_count}
        # Some other error — show details
        print(f"    {RED}lm-eval failed (exit {proc.returncode}){NC}")
        last_lines = combined.strip().split("\n")[-10:]
        for line in last_lines:
            line = line.strip()
            if line and ("error" in line.lower() or "traceback" in line.lower() or "|" in line):
                print(f"    {line[:120]}")
        # Still try to parse results even on non-zero exit
        for results_file in output_path.rglob("results_*.json"):
            with open(results_file) as f:
                results = json.load(f)
            r = results.get("results", {})
            return {
                "oom": False,
                "oom_count": oom_count,
                "gsm8k_flex": r.get("gsm8k_cot", {}).get("exact_match,flexible-extract"),
                "gsm8k_strict": r.get("gsm8k_cot", {}).get("exact_match,strict-match"),
                "ifeval_strict": r.get("ifeval", {}).get("inst_level_strict_acc,none"),
                "ifeval_loose": r.get("ifeval", {}).get("inst_level_loose_acc,none"),
                "duration_s": round(elapsed),
                "unreliable": oom_count > 5,
            }
        return None

    # Parse results
    results = None
    for results_file in output_path.rglob("results_*.json"):
        with open(results_file) as f:
            results = json.load(f)
        break

    if not results:
        if oom_count > 5:
            # Too many OOMs — this context size is unreliable
            return {"oom": True, "oom_count": oom_count, "unreliable": True}
        return None

    r = results.get("results", {})
    return {
        "oom": False,
        "oom_count": oom_count,
        "gsm8k_flex": r.get("gsm8k_cot", {}).get("exact_match,flexible-extract"),
        "gsm8k_strict": r.get("gsm8k_cot", {}).get("exact_match,strict-match"),
        "ifeval_strict": r.get("ifeval", {}).get("inst_level_strict_acc,none"),
        "ifeval_loose": r.get("ifeval", {}).get("inst_level_loose_acc,none"),
        "duration_s": round(elapsed),
        "unreliable": oom_count > 5,  # too many OOMs to trust scores
    }


def profile_model(model: str, limit: int, timestamp: str) -> dict:
    """Run context-quality profile for one model."""
    print(f"\n{BOLD}{'═' * 60}{NC}")
    print(f"{BOLD}  Context-Quality Profile: {model}{NC}")
    print(f"{BOLD}{'═' * 60}{NC}\n")

    max_ctx = get_model_max_context(model)
    sizes = generate_context_sizes(max_ctx)
    print(f"  Max context: {max_ctx:,}")
    print(f"  Testing: {', '.join(f'{s:,}' for s in sizes)}")
    print(f"  Samples per task: {limit}")
    print()

    profile = []

    for num_ctx in sizes:
        print(f"  {CYAN}num_ctx={num_ctx:,}{NC}")

        # Unload first
        print(f"    Unloading...")
        unload_model(model)

        # Load with specific context
        print(f"    Loading with num_ctx={num_ctx:,}...")
        if not load_model_with_context(model, num_ctx):
            print(f"    {RED}OOM on load → stopping{NC}")
            profile.append({
                "num_ctx": num_ctx,
                "status": "oom_load",
            })
            break

        vram = get_vram_usage()
        print(f"    VRAM: {vram}MB")

        # Run eval
        model_safe = model.replace(":", "-").replace("/", "-")
        output_path = RESULTS_DIR / f"{model_safe}" / f"ctx_{num_ctx}_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"    Running eval...")
        result = run_eval_at_context(model, num_ctx, limit, output_path)

        if result is None:
            print(f"    {RED}Eval failed{NC}")
            profile.append({
                "num_ctx": num_ctx,
                "status": "error",
                "vram_mb": vram,
            })
            continue

        if result.get("oom") and not result.get("gsm8k_flex"):
            print(f"    {RED}OOM during eval ({result['oom_count']} errors) → stopping{NC}")
            profile.append({
                "num_ctx": num_ctx,
                "status": "oom_eval",
                "oom_count": result["oom_count"],
                "vram_mb": vram,
            })
            break

        # Got results (possibly with some OOMs)
        gsm8k = result.get("gsm8k_flex")
        ifeval = result.get("ifeval_strict")
        gsm8k_str = f"{gsm8k*100:.1f}%" if gsm8k is not None else "N/A"
        ifeval_str = f"{ifeval*100:.1f}%" if ifeval is not None else "N/A"
        oom_str = f" ({result['oom_count']} OOMs)" if result.get("oom_count", 0) > 0 else ""
        reliable = "⚠" if result.get("unreliable") else "✓"

        print(f"    {GREEN}{reliable} GSM8K: {gsm8k_str}  IFEval: {ifeval_str}  VRAM: {vram}MB  {result['duration_s']}s{oom_str}{NC}")

        entry = {
            "num_ctx": num_ctx,
            "status": "ok" if not result.get("unreliable") else "unreliable",
            "gsm8k_flex": gsm8k,
            "gsm8k_strict": result.get("gsm8k_strict"),
            "ifeval_strict": ifeval,
            "ifeval_loose": result.get("ifeval_loose"),
            "vram_mb": vram,
            "duration_s": result["duration_s"],
            "oom_count": result.get("oom_count", 0),
        }
        profile.append(entry)

    # Unload when done
    unload_model(model)

    return {"model": model, "profile": profile, "max_ctx": max_ctx, "limit": limit}


def print_summary(all_results: list[dict]) -> None:
    """Print final summary table."""
    print(f"\n\n{BOLD}{'═' * 70}{NC}")
    print(f"{BOLD}  Context-Quality Summary{NC}")
    print(f"{BOLD}{'═' * 70}{NC}\n")

    for result in all_results:
        model = result["model"]
        profile = result["profile"]
        ok_entries = [p for p in profile if p.get("status") == "ok"]

        if not ok_entries:
            print(f"  {model}: no successful evaluations")
            continue

        print(f"  {BOLD}{model}{NC}")
        print(f"    {'Context':>10}  {'GSM8K':>7}  {'IFEval':>7}  {'VRAM':>7}  {'Time':>6}  {'OOMs':>5}  Status")
        print(f"    {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*10}")

        for p in profile:
            ctx = f"{p['num_ctx']:,}"
            status = p.get("status", "?")

            if status in ("oom_load", "oom_eval"):
                print(f"    {ctx:>10}  {'—':>7}  {'—':>7}  {p.get('vram_mb','—'):>7}  {'—':>6}  {'—':>5}  {RED}OOM{NC}")
                break

            gsm8k = f"{p['gsm8k_flex']*100:.1f}%" if p.get("gsm8k_flex") is not None else "N/A"
            ifeval = f"{p['ifeval_strict']*100:.1f}%" if p.get("ifeval_strict") is not None else "N/A"
            vram = f"{p.get('vram_mb', 0)}MB"
            dur = f"{p.get('duration_s', 0)}s"
            ooms = str(p.get("oom_count", 0))
            flag = "⚠ noisy" if status == "unreliable" else "OK"

            print(f"    {ctx:>10}  {gsm8k:>7}  {ifeval:>7}  {vram:>7}  {dur:>6}  {ooms:>5}  {flag}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Context-Quality Profiler")
    parser.add_argument("--models", help="Comma-separated model list (default: all)")
    parser.add_argument("--limit", type=int, default=SAMPLES_PER_TASK, help=f"Samples per task (default: {SAMPLES_PER_TASK})")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR/vision models")
    args = parser.parse_args()

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = get_ollama_models()

    if args.skip_ocr:
        models = [m for m in models if "ocr" not in m.lower()]

    if not models:
        print(f"{RED}No models found.{NC}")
        return 1

    print(f"{BOLD}{'═' * 60}{NC}")
    print(f"{BOLD}  Context-Quality Profiler{NC}")
    print(f"{BOLD}{'═' * 60}{NC}")
    print(f"  Tasks: {TASKS}")
    print(f"  Samples per task: {args.limit}")
    print(f"  Models ({len(models)}):")
    for m in models:
        print(f"    - {m}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    for i, model in enumerate(models, 1):
        print(f"\n{CYAN}[{i}/{len(models)}]{NC}")
        result = profile_model(model, args.limit, timestamp)
        all_results.append(result)

        # Save incremental results
        summary_file = RESULTS_DIR / f"summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print_summary(all_results)

    print(f"\n  Results saved: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
