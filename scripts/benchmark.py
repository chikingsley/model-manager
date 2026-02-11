# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///
"""
vLLM Optimization Benchmark Script

Run with: uv run scripts/benchmark.py

Tests:
1. Startup time (measures time to healthy)
2. TTFT (time to first token)
3. Throughput (tokens/sec)
4. Memory usage
"""

import asyncio
import subprocess
import time
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

VLLM_URL = "http://localhost:8000"


@dataclass
class BenchmarkResult:
    name: str
    startup_time: float | None = None
    ttft_ms: float | None = None
    throughput_tps: float | None = None
    gpu_memory_mb: int | None = None
    ram_used_gb: float | None = None
    error: str | None = None


async def wait_for_health(timeout: int = 300) -> float:
    """Wait for vLLM to become healthy, return startup time in seconds."""
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                resp = await client.get(f"{VLLM_URL}/health", timeout=5)
                if resp.status_code == 200:
                    return time.time() - start
            except Exception:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"vLLM didn't become healthy within {timeout}s")


async def measure_ttft(prompt: str = "Hello, how are you?", max_tokens: int = 100) -> float:
    """Measure time to first token in milliseconds."""
    async with httpx.AsyncClient() as client:
        start = time.time()

        # Use streaming to measure TTFT
        async with client.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": "default",  # vLLM uses whatever is loaded
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=60,
        ) as response:
            async for chunk in response.aiter_bytes():
                # First chunk = first token
                ttft = (time.time() - start) * 1000  # ms
                break

        return ttft


async def measure_throughput(
    prompt: str = "Write a detailed essay about artificial intelligence.",
    max_tokens: int = 500,
    num_requests: int = 10,
) -> float:
    """Measure throughput in tokens/second."""

    async def single_request(client: httpx.AsyncClient) -> tuple[int, float]:
        start = time.time()
        resp = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=120,
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
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start

        total_tokens = sum(r[0] for r in results)
        throughput = total_tokens / total_time

        return throughput


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
        total_kb = 0
        available_kb = 0
        for line in lines:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
        used_gb = (total_kb - available_kb) / 1024 / 1024
        return round(used_gb, 1)
    except Exception:
        return 0.0


async def run_benchmark(name: str) -> BenchmarkResult:
    """Run full benchmark suite."""
    result = BenchmarkResult(name=name)

    console.print(f"\n[bold blue]Running benchmark: {name}[/bold blue]")

    # Check if vLLM is already running
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{VLLM_URL}/health", timeout=5)
            if resp.status_code == 200:
                console.print("[green]vLLM is running[/green]")
                result.startup_time = 0  # Already running
    except Exception:
        console.print("[yellow]vLLM not running - skipping startup test[/yellow]")
        result.error = "vLLM not running"
        return result

    # Memory usage
    console.print("Measuring memory...")
    result.gpu_memory_mb = get_gpu_memory()
    result.ram_used_gb = get_ram_usage()
    console.print(f"  GPU: {result.gpu_memory_mb}MB, RAM: {result.ram_used_gb}GB")

    # TTFT
    console.print("Measuring TTFT...")
    try:
        # Average of 3 runs
        ttfts = []
        for _ in range(3):
            ttft = await measure_ttft()
            ttfts.append(ttft)
        result.ttft_ms = round(sum(ttfts) / len(ttfts), 1)
        console.print(f"  TTFT: {result.ttft_ms}ms")
    except Exception as e:
        console.print(f"[red]TTFT failed: {e}[/red]")

    # Throughput
    console.print("Measuring throughput (10 concurrent requests)...")
    try:
        result.throughput_tps = round(await measure_throughput(), 1)
        console.print(f"  Throughput: {result.throughput_tps} tokens/sec")
    except Exception as e:
        console.print(f"[red]Throughput failed: {e}[/red]")

    return result


def print_results(results: list[BenchmarkResult]) -> None:
    """Print results table."""
    table = Table(title="Benchmark Results")
    table.add_column("Config")
    table.add_column("Startup (s)")
    table.add_column("TTFT (ms)")
    table.add_column("Throughput (t/s)")
    table.add_column("GPU (MB)")
    table.add_column("RAM (GB)")

    for r in results:
        table.add_row(
            r.name,
            str(r.startup_time) if r.startup_time else "-",
            str(r.ttft_ms) if r.ttft_ms else "-",
            str(r.throughput_tps) if r.throughput_tps else "-",
            str(r.gpu_memory_mb) if r.gpu_memory_mb else "-",
            str(r.ram_used_gb) if r.ram_used_gb else "-",
        )

    console.print("\n")
    console.print(table)


async def main():
    console.print("[bold]vLLM Optimization Benchmark[/bold]")
    console.print("Make sure vLLM is running with the config you want to test.\n")

    # Run single benchmark
    result = await run_benchmark("Current Config")
    print_results([result])

    console.print("\n[bold]To test different configs:[/bold]")
    console.print("1. Edit ~/vllm/.env with new settings")
    console.print("2. Run: cd ~/vllm && docker compose up -d --force-recreate")
    console.print("3. Wait for health check, then run this script again")


if __name__ == "__main__":
    asyncio.run(main())
