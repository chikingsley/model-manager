# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
#     "pyyaml",
# ]
# ///
"""
Unified Model Benchmark

Tests both vLLM and llama.cpp backends for:
- Performance: TTFT, throughput, memory
- Quality: reasoning, coding, instruction following

Usage:
    uv run scripts/bench.py                    # Test whatever is running
    uv run scripts/bench.py --url http://localhost:18000  # Test llama.cpp
    uv run scripts/bench.py --full             # Full quality + perf test
"""

import argparse
import asyncio
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

# Default endpoints
VLLM_URL = "http://localhost:8000"
LLAMACPP_URL = "http://localhost:18000"

RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results"


# ═══════════════════════════════════════════════════════════════════════════════
# Quality Test Cases
# ═══════════════════════════════════════════════════════════════════════════════

QUALITY_TESTS = [
    {
        "name": "math_simple",
        "prompt": "What is 17 * 23? Just give the number.",
        "check": lambda r: "391" in r,
        "category": "reasoning",
    },
    {
        "name": "math_word",
        "prompt": "If I have 15 apples and give away 7, then buy 12 more, how many do I have? Just the number.",
        "check": lambda r: "20" in r,
        "category": "reasoning",
    },
    {
        "name": "code_reverse",
        "prompt": "Write a Python function called reverse_string that reverses a string. Just the code, no explanation.",
        "check": lambda r: "def reverse_string" in r and ("[::-1]" in r or "reversed" in r or "for" in r),
        "category": "coding",
    },
    {
        "name": "code_fizzbuzz",
        "prompt": "Write Python code for FizzBuzz from 1-15. Just print the output, no function needed.",
        "check": lambda r: "Fizz" in r and "Buzz" in r,
        "category": "coding",
    },
    {
        "name": "instruction_list",
        "prompt": "List exactly 5 colors, one per line. Nothing else.",
        "check": lambda r: len([l for l in r.strip().split("\n") if l.strip()]) >= 4,
        "category": "instruction",
    },
    {
        "name": "instruction_format",
        "prompt": "Reply with exactly: HELLO WORLD (all caps, nothing else)",
        "check": lambda r: "HELLO WORLD" in r.upper(),
        "category": "instruction",
    },
]


@dataclass
class BenchResult:
    """Benchmark result."""
    timestamp: str = ""
    url: str = ""
    model: str = ""
    backend: str = ""  # "vllm" or "llamacpp"

    # Performance
    ttft_ms: float | None = None
    throughput_tps: float | None = None
    gpu_mb: int | None = None
    ram_gb: float | None = None

    # Quality (0-100%)
    quality_score: float | None = None
    quality_details: dict = field(default_factory=dict)

    # Errors
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# API Helpers
# ═══════════════════════════════════════════════════════════════════════════════

async def detect_backend(url: str) -> tuple[str, str]:
    """Detect if URL is vLLM or llama.cpp, return (backend, model_name)."""
    async with httpx.AsyncClient() as client:
        # Try vLLM endpoint
        try:
            resp = await client.get(f"{url}/v1/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    return "vllm", data["data"][0]["id"]
        except Exception:
            pass

        # Try llama.cpp endpoint
        try:
            resp = await client.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                # llama.cpp health returns {"status": "ok"}
                return "llamacpp", "unknown"
        except Exception:
            pass

    return "unknown", "unknown"


async def chat_completion(url: str, prompt: str, model: str, max_tokens: int = 200) -> tuple[str, float]:
    """Send chat completion, return (response_text, time_seconds)."""
    async with httpx.AsyncClient() as client:
        start = time.time()

        # Try OpenAI-compatible endpoint (works for both vLLM and llama.cpp)
        resp = await client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Low temp for consistent quality tests
            },
            timeout=120,
        )

        elapsed = time.time() - start
        data = resp.json()

        if "error" in data:
            raise Exception(data["error"].get("message", str(data["error"])))

        content = data["choices"][0]["message"]["content"]
        return content, elapsed


async def measure_ttft(url: str, model: str) -> float:
    """Measure time to first token in ms."""
    async with httpx.AsyncClient() as client:
        start = time.time()
        async with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
                "stream": True,
            },
            timeout=60,
        ) as response:
            async for _ in response.aiter_bytes():
                return (time.time() - start) * 1000
    return 0


async def measure_throughput(url: str, model: str, num_requests: int = 5) -> float:
    """Measure throughput with concurrent requests."""
    prompt = "Write a short paragraph about technology."

    async def single(client: httpx.AsyncClient) -> int:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            },
            timeout=120,
        )
        data = resp.json()
        return data.get("usage", {}).get("completion_tokens", 0)

    async with httpx.AsyncClient() as client:
        # Warmup
        await single(client)

        # Concurrent
        start = time.time()
        tasks = [single(client) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start

        tokens = sum(r for r in results if isinstance(r, int))
        return tokens / elapsed if elapsed > 0 else 0


def get_gpu_memory() -> int:
    """Get GPU memory in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def get_ram_usage() -> float:
    """Get RAM usage in GB."""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        total = avail = 0
        for line in lines:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1])
        return round((total - avail) / 1024 / 1024, 1)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

async def run_quality_tests(url: str, model: str) -> tuple[float, dict]:
    """Run quality tests, return (score_percent, details)."""
    details = {}
    passed = 0

    for test in QUALITY_TESTS:
        name = test["name"]
        console.print(f"    {name}...", end=" ")

        try:
            response, _ = await chat_completion(url, test["prompt"], model, max_tokens=300)
            success = test["check"](response)
            details[name] = {
                "passed": success,
                "response": response[:200],  # Truncate for storage
            }
            if success:
                passed += 1
                console.print("[green]✓[/green]")
            else:
                console.print("[red]✗[/red]")
        except Exception as e:
            details[name] = {"passed": False, "error": str(e)}
            console.print(f"[red]error: {e}[/red]")

    score = (passed / len(QUALITY_TESTS)) * 100
    return score, details


async def run_benchmark(url: str, full: bool = False) -> BenchResult:
    """Run full benchmark."""
    result = BenchResult(
        timestamp=datetime.now().isoformat(),
        url=url,
    )

    # Detect backend
    console.print(f"\n[bold cyan]Benchmarking: {url}[/bold cyan]")
    backend, model = await detect_backend(url)
    result.backend = backend
    result.model = model
    console.print(f"  Backend: {backend}, Model: {model}")

    if backend == "unknown":
        result.error = "Could not connect or detect backend"
        return result

    # Memory
    result.gpu_mb = get_gpu_memory()
    result.ram_gb = get_ram_usage()
    console.print(f"  Memory: GPU {result.gpu_mb}MB, RAM {result.ram_gb}GB")

    # TTFT
    console.print("  Measuring TTFT...")
    try:
        ttfts = [await measure_ttft(url, model) for _ in range(3)]
        result.ttft_ms = round(sum(ttfts) / len(ttfts), 1)
        console.print(f"    TTFT: {result.ttft_ms}ms")
    except Exception as e:
        console.print(f"    [red]Failed: {e}[/red]")

    # Throughput
    console.print("  Measuring throughput (5 concurrent)...")
    try:
        result.throughput_tps = round(await measure_throughput(url, model, 5), 1)
        console.print(f"    Throughput: {result.throughput_tps} t/s")
    except Exception as e:
        console.print(f"    [red]Failed: {e}[/red]")

    # Quality (if full mode)
    if full:
        console.print("  Running quality tests...")
        try:
            result.quality_score, result.quality_details = await run_quality_tests(url, model)
            console.print(f"    Quality: {result.quality_score:.0f}%")
        except Exception as e:
            console.print(f"    [red]Failed: {e}[/red]")

    return result


def save_result(result: BenchResult, name: str | None = None) -> Path:
    """Save result to YAML."""
    RESULTS_DIR.mkdir(exist_ok=True)

    if name is None:
        name = f"bench_{result.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"

    path = RESULTS_DIR / name
    with open(path, "w") as f:
        yaml.dump(asdict(result), f, default_flow_style=False)

    return path


def print_result(result: BenchResult) -> None:
    """Print result table."""
    table = Table(title=f"Benchmark: {result.model}")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Backend", result.backend)
    table.add_row("TTFT", f"{result.ttft_ms}ms" if result.ttft_ms else "-")
    table.add_row("Throughput", f"{result.throughput_tps} t/s" if result.throughput_tps else "-")
    table.add_row("GPU Memory", f"{result.gpu_mb}MB" if result.gpu_mb else "-")
    table.add_row("RAM", f"{result.ram_gb}GB" if result.ram_gb else "-")
    if result.quality_score is not None:
        table.add_row("Quality", f"{result.quality_score:.0f}%")

    console.print("\n")
    console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="Unified Model Benchmark")
    parser.add_argument("--url", default=VLLM_URL, help="API endpoint URL")
    parser.add_argument("--full", action="store_true", help="Include quality tests")
    parser.add_argument("--save", help="Output filename")
    parser.add_argument("--both", action="store_true", help="Test both vLLM and llama.cpp")
    args = parser.parse_args()

    console.print("[bold]Unified Model Benchmark[/bold]")

    results = []

    if args.both:
        # Test both endpoints
        for url in [VLLM_URL, LLAMACPP_URL]:
            try:
                async with httpx.AsyncClient() as c:
                    await c.get(f"{url}/health", timeout=3)
                result = await run_benchmark(url, full=args.full)
                results.append(result)
            except Exception:
                console.print(f"[yellow]Skipping {url} (not running)[/yellow]")
    else:
        result = await run_benchmark(args.url, full=args.full)
        results.append(result)

    # Print and save
    for result in results:
        print_result(result)
        path = save_result(result, args.save)
        console.print(f"[green]Saved: {path}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
