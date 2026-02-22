#!/usr/bin/env -S uv run
"""Model Manager CLI."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from model_manager.benchmark import _median, run_benchmark
from model_manager.containers import (
    get_gpu_info,
    get_running_services,
    get_running_tunnels,
    list_gguf_models,
    ollama_get_loaded,
    ollama_is_running,
)
from model_manager.modes import Mode, _get_model_key, _get_ollama_model_key, activate
from model_manager.ollama import test_model_context
from model_manager.state import ModelEntry, StateManager

# Ensure unbuffered output (fixes piped/background execution)
reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_stdout):
    reconfigure_stdout(line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"  # No Color

BenchBackend = Literal["ollama", "vllm", "llama.cpp"]

ACTIVE_BENCHMARK_ENDPOINTS: dict[str, tuple[BenchBackend, str]] = {
    "ollama": ("ollama", "http://localhost:11434/v1"),
    "llama": ("llama.cpp", "http://localhost:8090/v1"),
    "chat": ("vllm", "http://localhost:8000/v1"),
    "perf": ("vllm", "http://localhost:8000/v1"),
    "ocr": ("vllm", "http://localhost:8000/v1"),
    "voice": ("llama.cpp", "http://localhost:18000/v1"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Status Display
# ─────────────────────────────────────────────────────────────────────────────


async def show_status() -> None:
    """Display current status."""
    state = StateManager()
    active = state.get_active()
    gpu = get_gpu_info()
    services = get_running_services()
    tunnels = get_running_tunnels()

    ollama_model = None
    if await ollama_is_running():
        ollama_model = await ollama_get_loaded()

    print()
    print(f"{BOLD}═══ Model Manager ═══{NC}")
    print()
    print(f"  {CYAN}Active:{NC} {GREEN}{active}{NC}")

    vram_pct = int(gpu.percent)
    bar_len = 20
    filled = vram_pct * bar_len // 100
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  {CYAN}VRAM:{NC}   {bar} {gpu.used_mb}/{gpu.total_mb} MB ({vram_pct}%)")
    print(f"  {CYAN}GPU:{NC}    {gpu.util_percent}% utilization")
    print()

    print(f"  {CYAN}Services:{NC}")
    for service in services:
        model_str = f" ({service.model})" if service.model else ""
        if service.name == "ollama" and ollama_model:
            model_str = f" ({ollama_model})"
        print(f"    {GREEN}●{NC} {service.name}{model_str}")
        if service.endpoint:
            print(f"      Local: localhost:{service.port}   Endpoint: {service.endpoint}")

    if tunnels:
        print(f"    {GREEN}●{NC} tunnels: {' '.join(tunnels)}")

    print()
    print(f"  {CYAN}Commands:{NC} mm ollama | mm llama | mm chat | mm sam3 | mm voice | mm stop")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Mode Activation
# ─────────────────────────────────────────────────────────────────────────────


async def run_activate(mode: Mode, model: str | None = None) -> int:
    """Run mode activation with progress output."""

    def on_progress(progress) -> None:
        health = f" [{progress.health}]" if progress.health else ""
        print(f"  {progress.step}{health}")

    print(f"{CYAN}Activating {mode}...{NC}")
    result = await activate(mode, model, progress=on_progress)

    if not result.success:
        print()
        print(f"{RED}✗ {result.message}{NC}")
        if result.details:
            print(f"  {result.details}")
        return 1

    print()
    print(f"{GREEN}{result.message}{NC}")
    if result.details:
        if "endpoint" in result.details:
            print(f"  Endpoint: {result.details['endpoint']}")
        if "model" in result.details:
            print(f"  Model: {result.details['model']}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Model Listing
# ─────────────────────────────────────────────────────────────────────────────


def show_models() -> None:
    """List available GGUF models."""
    models = list_gguf_models()

    print(f"{CYAN}Available GGUF models:{NC}")
    print()
    for model in models:
        print(f"  {model}")
    print()
    print("Switch with: mm llama <model-filename>")


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Context Testing
# ─────────────────────────────────────────────────────────────────────────────


async def run_ollama_context_test(model: str) -> int:
    """Test Ollama model context limits and persist results."""
    print(f"{CYAN}Testing context limits for {model}...{NC}")
    print()

    def on_progress(message: str) -> None:
        print(f"  {message}")

    result = await test_model_context(model, on_progress)

    model_key = _get_ollama_model_key(model)
    entry = ModelEntry(
        source="ollama",
        model=model,
        backend="ollama",
        context_tested=True,
        claimed_num_ctx=result.claimed_max_ctx,
        tested_num_ctx=result.tested_max_ctx,
        num_ctx=result.recommended_ctx,
        vram_estimate=round(result.vram_at_max_mb / 1024, 1),
        notes=f"Auto-tested: max {result.tested_max_ctx:,}, using {result.recommended_ctx:,}",
    )
    StateManager().register_model(model_key, entry)

    print()
    print(f"{GREEN}Results:{NC}")
    print(f"  Claimed max:    {result.claimed_max_ctx:,}")
    print(f"  Tested max:     {result.tested_max_ctx:,}")
    print(f"  Recommended:    {result.recommended_ctx:,}")
    print(f"  VRAM at max:    {result.vram_at_max_mb:,} MB")

    speed_results = [item for item in result.results if item.success and item.tok_s > 0]
    if speed_results:
        print()
        print(f"  {CYAN}Context-Speed Profile:{NC}")
        print(f"  {'Context':>10}  {'Tok/s':>7}  {'TTFT':>7}  {'VRAM':>8}")
        print(f"  {'─' * 10}  {'─' * 7}  {'─' * 7}  {'─' * 8}")
        for row in speed_results:
            context_str = f"{row.num_ctx:,}"
            print(
                f"  {context_str:>10}  {row.tok_s:>6.1f}  {row.ttft_ms:>5.0f}ms  {row.vram_mb:>6}MB"
            )

    print()
    print(f"{GREEN}Saved to models.yaml as '{model_key}'{NC}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Benchmarking
# ─────────────────────────────────────────────────────────────────────────────


async def run_benchmark_current() -> int:
    """Benchmark the currently active chat/completion endpoint."""
    state = StateManager()
    active = state.get_active()

    if active == "none":
        print(f"{RED}Nothing running. Start a model first: mm ollama/llama/chat{NC}")
        return 1

    if active == "embed":
        print(f"{RED}Cannot benchmark embedding mode with chat completions.{NC}")
        return 1

    if active not in ACTIVE_BENCHMARK_ENDPOINTS:
        print(f"{RED}Active mode '{active}' cannot be benchmarked with this command.{NC}")
        return 1

    backend, base_url = ACTIVE_BENCHMARK_ENDPOINTS[active]
    print(f"{CYAN}Benchmarking {active} ({backend})...{NC}")
    print()

    def on_progress(message: str) -> None:
        print(f"  {message}")

    try:
        result = await run_benchmark(
            base_url=base_url,
            num_requests=10,
            max_tokens=128,
            on_progress=on_progress,
            backend=backend,
        )
    except Exception as error:
        print(f"{RED}Benchmark failed: {error}{NC}")
        return 1

    model_key = _get_model_key(backend, result.model)
    existing = state.get_model(model_key)
    if existing:
        existing.bench_tok_s = result.tok_s
        existing.bench_ttft_ms = result.ttft_ms
        existing.bench_itl_ms = result.itl_ms
        existing.bench_p95_ms = result.p95_itl_ms
        existing.bench_date = date.today().isoformat()
        existing.benchmarked = True
        state.register_model(model_key, existing)
    else:
        state.register_model(
            model_key,
            ModelEntry(
                source=result.model,
                backend=backend,
                bench_tok_s=result.tok_s,
                bench_ttft_ms=result.ttft_ms,
                bench_itl_ms=result.itl_ms,
                bench_p95_ms=result.p95_itl_ms,
                bench_date=date.today().isoformat(),
                benchmarked=True,
            ),
        )

    print()
    print(f"{GREEN}Results:{NC}")
    print(f"  Model:     {result.model}")
    print(f"  Backend:   {result.backend}")
    print(f"  Tok/s:     {BOLD}{result.tok_s}{NC}")
    print(f"  TTFT:      {result.ttft_ms} ms")
    print(f"  ITL:       {result.itl_ms} ms (median)")
    print(f"  P95 ITL:   {result.p95_itl_ms} ms")
    print(
        f"  Tokens:    {result.total_tokens} in {result.duration_s}s ({result.num_requests} requests)"
    )
    print()
    print(f"{GREEN}Saved to models.yaml as '{model_key}'{NC}")
    return 0


def show_benchmarks() -> None:
    """Show all stored benchmark results side by side."""
    registry = StateManager().load()

    rows = {
        name: entry
        for name, entry in registry.models.items()
        if entry.benchmarked or entry.context_profile
    }

    if not rows:
        print(f"{YELLOW}No benchmark data yet. Run 'mm benchmark run' first.{NC}")
        return

    print()
    print(f"{BOLD}═══ Model Performance ═══{NC}")
    print()
    print(f"  {'Model':<25} {'Tok/s':>7} {'TTFT':>7} {'Context':>9} {'VRAM':>7}  {'Source'}")
    print(f"  {'─' * 25} {'─' * 7} {'─' * 7} {'─' * 9} {'─' * 7}  {'─' * 10}")

    for key, entry in sorted(rows.items()):
        model_name = entry.model or entry.source or key
        if len(model_name) > 25:
            model_name = f"...{model_name[-22:]}"

        if entry.benchmarked and entry.bench_tok_s:
            tok_s = f"{entry.bench_tok_s:.1f}"
            ttft = f"{entry.bench_ttft_ms:.0f}ms" if entry.bench_ttft_ms else "—"
            source = "bench"
        elif entry.context_profile:
            speeds = [point.tok_s for point in entry.context_profile]
            ttfts = [point.ttft_ms for point in entry.context_profile]
            tok_s = f"{_median(speeds):.1f}"
            ttft = f"{_median(ttfts):.0f}ms"
            source = "profile"
        else:
            tok_s = "—"
            ttft = "—"
            source = "—"

        context = f"{entry.tested_num_ctx:,}" if entry.tested_num_ctx else "—"
        vram = f"{entry.vram_estimate:.0f}GB" if entry.vram_estimate else "—"
        print(f"  {model_name:<25} {tok_s:>7} {ttft:>7} {context:>9} {vram:>7}  {source}")

    profiled = {name: entry for name, entry in rows.items() if entry.context_profile}
    if profiled:
        print()
        print(f"  {CYAN}Context-Speed Details:{NC}")
        for key, entry in sorted(profiled.items()):
            print(f"\n  {BOLD}{entry.model or key}{NC}")
            print(f"    {'Context':>10}  {'Tok/s':>7}  {'TTFT':>7}  {'VRAM':>8}")
            print(f"    {'─' * 10}  {'─' * 7}  {'─' * 7}  {'─' * 8}")
            for point in entry.context_profile or []:
                context_str = f"{point.num_ctx:,}"
                print(
                    f"    {context_str:>10}  {point.tok_s:>6.1f}  "
                    f"{point.ttft_ms:>5.0f}ms  {point.vram_mb:>6}MB"
                )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────


def run_server(host: str = "0.0.0.0", port: int = 8888) -> None:
    """Run the API server."""
    import uvicorn

    from .api.server import app

    print(f"{CYAN}Starting API server on {host}:{port}...{NC}")
    uvicorn.run(app, host=host, port=port)


# ─────────────────────────────────────────────────────────────────────────────
# Help + Command Dispatch
# ─────────────────────────────────────────────────────────────────────────────


def show_help() -> None:
    """Show usage information."""
    print("mm - Model Manager")
    print()
    print("Core:")
    print("  mm                                Show status")
    print("  mm voice                          Activate voice stack")
    print("  mm llama [model]                  Activate llama.cpp")
    print("  mm ollama [model]                 Activate Ollama")
    print("  mm ocr [model]                    Activate OCR mode (default: DeepSeek-OCR-2)")
    print("  mm chat [model]                   Activate vLLM chat")
    print("  mm perf [model]                   Activate max-performance vLLM")
    print("  mm embed                          Activate embeddings")
    print("  mm sam3                           Activate SAM3 segmentation service")
    print("  mm pronunciation                  Activate pronunciation lab (Qwen3-ASR)")
    print("  mm stop                           Stop all model services")
    print("  mm models                         List GGUF models")
    print("  mm serve [host] [port]            Start API server")
    print("  mm ollama-context <model>         Test Ollama context and save profile")
    print()
    print("Benchmark CLI:")
    print("  mm benchmark run                  Benchmark current active model")
    print("  mm benchmark compare              Compare saved benchmark results")
    print()
    print("OCR Benchmark Suite:")
    print("  mm benchmark ocr                  Run all OCR benchmarks against active model")
    print("  mm benchmark ocr --bench NAME     Run specific (ocrbench/omnidoc/unimer/tables/kie)")
    print("  mm benchmark ocr --limit N        Limit samples per benchmark")
    print("  mm benchmark ocr --resume         Resume from existing predictions")
    print("  mm benchmark ocr compare          Compare OCR results across models")
    print("  mm benchmark ocr setup            Download datasets and clone repos")


def _dispatch_ocr_benchmark(args: list[str]) -> int:
    """Dispatch OCR benchmark suite commands."""
    import subprocess

    runner = Path(__file__).parent.parent.parent / "benchmarks" / "ocr_suite" / "runner.py"

    if not runner.exists():
        print(f"{RED}OCR suite runner not found at {runner}{NC}")
        return 1

    # Parse the OCR-specific args
    parser = argparse.ArgumentParser(prog="mm benchmark ocr")
    parser.add_argument("ocr_command", nargs="?", default="run",
                        choices=["run", "compare", "setup"])
    parser.add_argument("--bench", type=str, default="all")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    parsed, extra = parser.parse_known_args(args)

    if parsed.ocr_command == "setup":
        setup_script = runner.parent / "setup_datasets.py"
        cmd = ["uv", "run", str(setup_script), *extra]
    elif parsed.ocr_command == "compare":
        cmd = ["uv", "run", str(runner), "compare"]
    else:
        cmd = ["uv", "run", str(runner), "run", "--bench", parsed.bench]
        if parsed.base_url:
            cmd.extend(["--base-url", parsed.base_url])
        if parsed.api_key != "EMPTY":
            cmd.extend(["--api-key", parsed.api_key])
        if parsed.limit:
            cmd.extend(["--limit", str(parsed.limit)])
        if parsed.resume:
            cmd.append("--resume")

    return subprocess.run(cmd).returncode


def _dispatch_benchmark(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mm benchmark", add_help=True)
    subparsers = parser.add_subparsers(dest="subcommand")

    subparsers.add_parser("run")
    subparsers.add_parser("compare")
    subparsers.add_parser("ocr")

    # Parse only the first arg to get the subcommand
    parsed, remaining = parser.parse_known_args(args)

    if parsed.subcommand == "run":
        return asyncio.run(run_benchmark_current())

    if parsed.subcommand == "compare":
        show_benchmarks()
        return 0

    if parsed.subcommand == "ocr":
        return _dispatch_ocr_benchmark(remaining)

    parser.print_help()
    return 1


def main() -> int:
    """CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] in {"status", "st"}:
        asyncio.run(show_status())
        return 0

    command = args[0]
    model = args[1] if len(args) > 1 else None

    if command in {"help", "-h", "--help"}:
        show_help()
        return 0

    if command in {"models", "list"}:
        show_models()
        return 0

    if command == "serve":
        host = args[1] if len(args) > 1 else "0.0.0.0"
        port = int(args[2]) if len(args) > 2 else 8888
        run_server(host, port)
        return 0

    if command == "benchmark":
        return _dispatch_benchmark(args[1:])

    if command == "ollama-context":
        if not model:
            print(f"{RED}Usage: mm ollama-context <model>{NC}")
            return 1
        return asyncio.run(run_ollama_context_test(model))

    mode_map: dict[str, Mode] = {
        "voice": "voice",
        "v": "voice",
        "llama": "llama",
        "l": "llama",
        "gguf": "llama",
        "ollama": "ollama",
        "ol": "ollama",
        "ocr": "ocr",
        "o": "ocr",
        "chat": "chat",
        "c": "chat",
        "perf": "perf",
        "p": "perf",
        "max": "perf",
        "embed": "embed",
        "embeddings": "embed",
        "e": "embed",
        "sam3": "sam3",
        "sam": "sam3",
        "pronunciation": "pronunciation",
        "pron": "pronunciation",
        "stop": "stop",
    }

    if command in mode_map:
        return asyncio.run(run_activate(mode_map[command], model))

    print(f"{RED}Unknown command: {command}{NC}")
    print("Run 'mm help' for usage")
    return 1


if __name__ == "__main__":
    sys.exit(main())
