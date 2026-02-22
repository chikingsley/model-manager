# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///
"""
Quick benchmark for Ollama models.

Usage:
    uv run scripts/bench_ollama.py                     # Test all models
    uv run scripts/bench_ollama.py ministral-3:14b    # Test specific model
    uv run scripts/bench_ollama.py --quick            # Just math tests
"""

import argparse
import time
import httpx
from rich.console import Console
from rich.table import Table

console = Console()

OLLAMA_URL = "http://localhost:11434"

TESTS = [
    # Math reasoning
    {"name": "math_simple", "prompt": "What is 17 * 23? Just give the number.", "answer": "391", "category": "math"},
    {"name": "math_word", "prompt": "If I have 15 apples and give away 7, then buy 12 more, how many do I have?", "answer": "20", "category": "math"},
    {"name": "math_multi", "prompt": "What is (8 + 4) * 3 - 6? Just the number.", "answer": "30", "category": "math"},

    # Coding
    {"name": "code_reverse", "prompt": "Write a Python one-liner to reverse a string s.", "answer": "[::-1]", "category": "code"},
    {"name": "code_sum", "prompt": "Write Python: sum of list [1,2,3,4,5]. Just the code.", "answer": "15", "category": "code"},

    # Instruction following
    {"name": "inst_caps", "prompt": "Say HELLO WORLD in all caps. Nothing else.", "answer": "HELLO WORLD", "category": "instruction"},
    {"name": "inst_count", "prompt": "List exactly 3 fruits, one per line.", "answer": None, "category": "instruction", "check": lambda r: len([l for l in r.strip().split("\n") if l.strip()]) >= 3},
]

MATH_ONLY = [t for t in TESTS if t["category"] == "math"]


def get_models() -> list[str]:
    """Get list of available Ollama models."""
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def run_inference(model: str, prompt: str, num_ctx: int = 4096) -> tuple[str, float, float]:
    """Run inference, return (response, ttft_ms, total_ms)."""
    start = time.perf_counter()
    first_token_time = None
    response_text = ""

    with httpx.stream(
        "POST",
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True, "options": {"num_ctx": num_ctx}},
        timeout=60,
    ) as resp:
        for line in resp.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "response" in data:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    response_text += data["response"]

    total_time = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else total_time

    return response_text.strip(), ttft * 1000, total_time * 1000


def check_answer(response: str, test: dict) -> bool:
    """Check if response is correct."""
    if "check" in test:
        return test["check"](response)
    if test["answer"]:
        return test["answer"].lower() in response.lower()
    return True


def benchmark_model(model: str, tests: list[dict]) -> dict:
    """Run benchmark on a model."""
    results = {"model": model, "passed": 0, "total": len(tests), "ttft_avg": 0, "details": []}
    ttfts = []

    for test in tests:
        try:
            response, ttft, total = run_inference(model, test["prompt"])
            passed = check_answer(response, test)
            ttfts.append(ttft)

            if passed:
                results["passed"] += 1

            results["details"].append({
                "name": test["name"],
                "passed": passed,
                "ttft_ms": round(ttft, 1),
                "total_ms": round(total, 1),
                "response": response[:100] + "..." if len(response) > 100 else response,
            })
        except Exception as e:
            results["details"].append({"name": test["name"], "passed": False, "error": str(e)})

    results["ttft_avg"] = round(sum(ttfts) / len(ttfts), 1) if ttfts else 0
    return results


def main():
    parser = argparse.ArgumentParser(description="Ollama benchmark")
    parser.add_argument("models", nargs="*", help="Models to test (default: all)")
    parser.add_argument("--quick", action="store_true", help="Just math tests")
    args = parser.parse_args()

    models = args.models or get_models()
    tests = MATH_ONLY if args.quick else TESTS

    if not models:
        console.print("[red]No models found. Is Ollama running?[/red]")
        return

    # Skip OCR models
    models = [m for m in models if "ocr" not in m.lower()]

    console.print(f"\n[bold]Ollama Benchmark[/bold]")
    console.print(f"Models: {', '.join(models)}")
    console.print(f"Tests: {len(tests)} ({', '.join(set(t['category'] for t in tests))})\n")

    all_results = []
    for model in models:
        console.print(f"[cyan]Testing {model}...[/cyan]")
        result = benchmark_model(model, tests)
        all_results.append(result)

        # Print individual results
        for d in result["details"]:
            status = "[green]✓[/green]" if d.get("passed") else "[red]✗[/red]"
            console.print(f"  {status} {d['name']}: {d.get('ttft_ms', '?')}ms TTFT")

    # Summary table
    console.print("\n")
    table = Table(title="Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("TTFT (avg)", justify="right")

    for r in sorted(all_results, key=lambda x: x["passed"], reverse=True):
        score = f"{r['passed']}/{r['total']}"
        score_pct = r['passed'] / r['total'] * 100
        color = "green" if score_pct >= 80 else "yellow" if score_pct >= 50 else "red"
        table.add_row(r["model"], f"[{color}]{score}[/{color}]", f"{r['ttft_avg']}ms")

    console.print(table)


if __name__ == "__main__":
    main()
