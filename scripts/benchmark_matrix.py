# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
#     "pyyaml",
# ]
# ///
"""
Model Manager Benchmark Matrix

Runs benchmarks across multiple models and context lengths,
saves results to YAML for comparison.

Usage:
    uv run scripts/benchmark_matrix.py                    # Benchmark current config
    uv run scripts/benchmark_matrix.py --model qwen3-4b   # Specific model
    uv run scripts/benchmark_matrix.py --all              # Full matrix (takes a while)
"""

import argparse
import asyncio
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

VLLM_URL = "http://localhost:8000"
RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results"
MM_SCRIPT = Path(__file__).parent.parent / "mm"

# Models to test (id -> HuggingFace path)
MODELS = {
    "qwen2.5-7b-awq": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct",  # If available locally
}

# Context lengths to test
CONTEXT_LENGTHS = [4096, 8192, 16384]


@dataclass
class BenchmarkRun:
    """Single benchmark run result."""
    timestamp: str = ""
    model: str = ""
    context_length: int = 0
    config_mode: str = ""  # "perf" or "baseline"

    # Metrics
    startup_time_s: float | None = None
    ttft_ms: float | None = None
    throughput_tps: float | None = None
    gpu_memory_mb: int | None = None
    ram_used_gb: float | None = None

    # Test params
    num_requests: int = 10
    max_tokens: int = 500

    # Errors
    error: str | None = None

    # vLLM info
    vllm_version: str = ""
    optimizations: list[str] = field(default_factory=list)


async def wait_for_health(timeout: int = 300) -> float:
    """Wait for vLLM to become healthy, return time in seconds."""
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                resp = await client.get(f"{VLLM_URL}/health", timeout=5)
                if resp.status_code == 200:
                    return time.time() - start
            except Exception:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"vLLM didn't become healthy within {timeout}s")


async def get_model_name() -> str:
    """Get the currently loaded model name from vLLM."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{VLLM_URL}/v1/models", timeout=10)
            data = resp.json()
            if data.get("data"):
                return data["data"][0]["id"]
        except Exception:
            pass
    return "unknown"


async def measure_ttft(prompt: str, max_tokens: int = 100, model: str | None = None) -> float:
    """Measure time to first token in milliseconds."""
    if model is None:
        model = await get_model_name()

    async with httpx.AsyncClient() as client:
        start = time.time()
        async with client.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=120,
        ) as response:
            async for _ in response.aiter_bytes():
                return (time.time() - start) * 1000
    return 0


async def measure_throughput(
    prompt: str,
    max_tokens: int = 500,
    num_requests: int = 10,
    model: str | None = None,
) -> float:
    """Measure throughput in tokens/second with concurrent requests."""
    if model is None:
        model = await get_model_name()

    async def single_request(client: httpx.AsyncClient) -> tuple[int, float]:
        start = time.time()
        resp = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=180,
        )
        elapsed = time.time() - start
        data = resp.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return tokens, elapsed

    async with httpx.AsyncClient() as client:
        # Warmup
        await single_request(client)

        # Concurrent requests
        start = time.time()
        tasks = [single_request(client) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start

        # Filter out exceptions
        valid = [r for r in results if isinstance(r, tuple)]
        total_tokens = sum(r[0] for r in valid)

        return total_tokens / total_time if total_time > 0 else 0


def get_gpu_memory() -> int:
    """Get GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def get_ram_usage() -> float:
    """Get RAM usage in GB."""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        total_kb = available_kb = 0
        for line in lines:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
        return round((total_kb - available_kb) / 1024 / 1024, 1)
    except Exception:
        return 0.0


def get_vllm_info() -> tuple[str, list[str]]:
    """Get vLLM version and active optimizations from logs."""
    version = "unknown"
    optimizations = []

    try:
        result = subprocess.run(
            ["docker", "logs", "vllm", "--tail", "100"],
            capture_output=True,
            text=True,
        )
        logs = result.stdout + result.stderr

        # Extract version
        if "vLLM API server version" in logs:
            for line in logs.split("\n"):
                if "vLLM API server version" in line:
                    version = line.split("version")[-1].strip()
                    break

        # Check for optimizations
        if "FLASHINFER" in logs:
            optimizations.append("FlashInfer")
        if "Asynchronous scheduling is enabled" in logs:
            optimizations.append("AsyncScheduling")
        if "OffloadingConnector" in logs:
            optimizations.append("KVOffloading")
        if "fp8" in logs.lower():
            optimizations.append("FP8_KV")

    except Exception:
        pass

    return version, optimizations


async def run_single_benchmark(
    model_name: str,
    context_length: int,
    config_mode: str,
) -> BenchmarkRun:
    """Run benchmark for a single configuration."""

    result = BenchmarkRun(
        timestamp=datetime.now().isoformat(),
        model=model_name,
        context_length=context_length,
        config_mode=config_mode,
    )

    console.print(f"\n[bold cyan]Benchmarking: {model_name} @ {context_length} tokens ({config_mode})[/bold cyan]")

    # Check health
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{VLLM_URL}/health", timeout=5)
            if resp.status_code != 200:
                result.error = "vLLM not healthy"
                return result
    except Exception as e:
        result.error = f"vLLM not running: {e}"
        return result

    # Get vLLM info
    result.vllm_version, result.optimizations = get_vllm_info()

    # Memory
    result.gpu_memory_mb = get_gpu_memory()
    result.ram_used_gb = get_ram_usage()
    console.print(f"  Memory: GPU {result.gpu_memory_mb}MB, RAM {result.ram_used_gb}GB")

    # Generate prompt that's roughly the target context length
    # (actual tokens will be less, but this gives consistent load)
    base_prompt = "Explain the concept of " + "artificial intelligence " * 50
    prompt = base_prompt[:min(len(base_prompt), context_length * 2)]

    # TTFT (3 runs, average)
    console.print("  Measuring TTFT...")
    try:
        ttfts = []
        for _ in range(3):
            ttft = await measure_ttft(prompt, max_tokens=50)
            ttfts.append(ttft)
        result.ttft_ms = round(sum(ttfts) / len(ttfts), 1)
        console.print(f"    TTFT: {result.ttft_ms}ms")
    except Exception as e:
        console.print(f"    [red]TTFT failed: {e}[/red]")

    # Throughput
    console.print("  Measuring throughput (10 concurrent)...")
    try:
        result.throughput_tps = round(await measure_throughput(prompt), 1)
        console.print(f"    Throughput: {result.throughput_tps} t/s")
    except Exception as e:
        console.print(f"    [red]Throughput failed: {e}[/red]")

    return result


def save_results(results: list[BenchmarkRun], filename: str | None = None) -> Path:
    """Save results to YAML file."""
    RESULTS_DIR.mkdir(exist_ok=True)

    if filename is None:
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"

    filepath = RESULTS_DIR / filename

    data = {
        "benchmark_date": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]Results saved to: {filepath}[/green]")
    return filepath


def print_results_table(results: list[BenchmarkRun]) -> None:
    """Print results as a table."""
    table = Table(title="Benchmark Results")
    table.add_column("Model")
    table.add_column("Context")
    table.add_column("Mode")
    table.add_column("TTFT (ms)")
    table.add_column("Throughput (t/s)")
    table.add_column("GPU (MB)")
    table.add_column("RAM (GB)")

    for r in results:
        table.add_row(
            r.model,
            str(r.context_length),
            r.config_mode,
            str(r.ttft_ms) if r.ttft_ms else "-",
            str(r.throughput_tps) if r.throughput_tps else "-",
            str(r.gpu_memory_mb) if r.gpu_memory_mb else "-",
            str(r.ram_used_gb) if r.ram_used_gb else "-",
        )

    console.print("\n")
    console.print(table)


async def benchmark_current() -> list[BenchmarkRun]:
    """Benchmark whatever is currently running."""
    result = await run_single_benchmark(
        model_name="current",
        context_length=16384,  # Default
        config_mode="current",
    )
    return [result]


async def main():
    parser = argparse.ArgumentParser(description="Model Manager Benchmark Matrix")
    parser.add_argument("--model", help="Specific model to test")
    parser.add_argument("--all", action="store_true", help="Run full matrix (slow)")
    parser.add_argument("--save", help="Output filename (default: auto)")
    args = parser.parse_args()

    console.print("[bold]Model Manager Benchmark Matrix[/bold]")
    console.print(f"Results will be saved to: {RESULTS_DIR}/\n")

    results: list[BenchmarkRun] = []

    if args.all:
        console.print("[yellow]Full matrix mode - this will take a while![/yellow]")
        # TODO: Implement full matrix with model switching
        # For now, just benchmark current
        results = await benchmark_current()
    else:
        # Just benchmark current config
        results = await benchmark_current()

    # Print and save
    print_results_table(results)
    save_results(results, args.save)

    # Show optimizations detected
    if results and results[0].optimizations:
        console.print(f"\n[cyan]Active optimizations:[/cyan] {', '.join(results[0].optimizations)}")
        console.print(f"[cyan]vLLM version:[/cyan] {results[0].vllm_version}")


if __name__ == "__main__":
    asyncio.run(main())
