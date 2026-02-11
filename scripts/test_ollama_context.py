# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
# ]
# ///
"""
Test Ollama model context limits on your GPU.
Finds the maximum context length each model can handle before OOM.

Usage:
    uv run test_ollama_context.py glm-ocr
    uv run test_ollama_context.py glm-ocr ministral
    uv run test_ollama_context.py --all
"""

import argparse
import subprocess
import sys
import time

import httpx

OLLAMA_URL = "http://localhost:11434"

# Models to test with their claimed max context
MODELS = {
    "glm-ocr": {"claimed_max": 131072, "test_sizes": [4096, 8192, 16384, 32768, 65536, 131072]},
    "ministral-3:8b": {"claimed_max": 262144, "test_sizes": [4096, 8192, 16384, 32768, 65536]},
    "qwen3:4b": {"claimed_max": 32768, "test_sizes": [4096, 8192, 16384, 32768]},
    "llama3.2:3b": {"claimed_max": 131072, "test_sizes": [4096, 8192, 16384, 32768, 65536]},
}


def get_gpu_memory() -> tuple[int, int]:
    """Get GPU memory usage (used_mb, total_mb)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        used, total = result.stdout.strip().split(",")
        return int(used.strip()), int(total.strip())
    except Exception:
        return 0, 12227


def unload_model(model: str) -> None:
    """Unload a model from memory."""
    try:
        httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def test_context_size(model: str, num_ctx: int) -> dict:
    """Test if a model can handle a specific context size."""
    # Create a prompt that will use significant context
    # We'll ask it to repeat back a long string to verify it's actually using the context
    test_prompt = "Reply with only: OK"

    print(f"  Testing num_ctx={num_ctx:,}...", end=" ", flush=True)

    used_before, total = get_gpu_memory()
    start = time.time()

    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": test_prompt,
                "stream": False,
                "options": {"num_ctx": num_ctx, "num_predict": 10},
            },
            timeout=120,
        )

        elapsed = time.time() - start
        used_after, _ = get_gpu_memory()

        if response.status_code == 200:
            data = response.json()
            print(f"✓ OK ({elapsed:.1f}s, VRAM: {used_after}MB)")
            return {
                "success": True,
                "num_ctx": num_ctx,
                "vram_mb": used_after,
                "vram_delta": used_after - used_before,
                "load_time": elapsed,
                "response": data.get("response", "")[:50],
            }
        else:
            print(f"✗ HTTP {response.status_code}")
            return {"success": False, "num_ctx": num_ctx, "error": f"HTTP {response.status_code}"}

    except httpx.TimeoutException:
        print("✗ Timeout")
        return {"success": False, "num_ctx": num_ctx, "error": "timeout"}
    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            print("✗ OOM")
        else:
            print(f"✗ {error_msg[:50]}")
        return {"success": False, "num_ctx": num_ctx, "error": error_msg}


def test_model(model: str) -> dict:
    """Test a model at various context sizes."""
    config = MODELS.get(model, {"claimed_max": 32768, "test_sizes": [4096, 8192, 16384, 32768]})

    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"Claimed max context: {config['claimed_max']:,}")
    print(f"{'='*60}")

    # First unload any existing model
    print("Unloading existing models...")
    unload_model(model)
    time.sleep(2)

    results = []
    max_working = 0

    for ctx_size in config["test_sizes"]:
        result = test_context_size(model, ctx_size)
        results.append(result)

        if result["success"]:
            max_working = ctx_size
        else:
            # Stop testing larger sizes after first failure
            print(f"  Stopping at {ctx_size:,} - larger sizes will likely fail too")
            break

        # Brief pause between tests
        time.sleep(1)

    # Unload after testing
    unload_model(model)

    return {
        "model": model,
        "claimed_max": config["claimed_max"],
        "tested_max": max_working,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Ollama model context limits")
    parser.add_argument("models", nargs="*", help="Models to test (or --all)")
    parser.add_argument("--all", action="store_true", help="Test all known models")
    args = parser.parse_args()

    # Check Ollama is running
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except Exception:
        print("Error: Ollama not running at", OLLAMA_URL)
        print("Start with: docker start ollama")
        sys.exit(1)

    models_to_test = list(MODELS.keys()) if args.all else args.models

    if not models_to_test:
        print("Usage: uv run test_ollama_context.py <model> [model2 ...]")
        print("       uv run test_ollama_context.py --all")
        print(f"\nAvailable models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    all_results = []

    for model in models_to_test:
        result = test_model(model)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Recommended num_ctx settings")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Claimed':<12} {'Tested Max':<12} {'Recommendation'}")
    print("-" * 60)

    for r in all_results:
        claimed = f"{r['claimed_max']:,}"
        tested = f"{r['tested_max']:,}" if r['tested_max'] > 0 else "failed"
        # Recommend 80% of tested max for safety margin
        if r['tested_max'] > 0:
            recommended = int(r['tested_max'] * 0.8)
            rec_str = f"{recommended:,}"
        else:
            rec_str = "N/A"
        print(f"{r['model']:<20} {claimed:<12} {tested:<12} {rec_str}")

    print(f"\nAdd these to your models.yaml under each model's config.")


if __name__ == "__main__":
    main()
