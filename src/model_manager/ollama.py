"""
Ollama-specific functionality.

Handles context testing, model info, and Ollama API interactions.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from model_manager.containers import get_gpu_info

if TYPE_CHECKING:
    from collections.abc import Callable

OLLAMA_URL = "http://localhost:11434"

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""

    name: str
    size_bytes: int
    parameter_size: str  # e.g., "8B", "14B"
    quantization: str  # e.g., "Q4_K_M"
    family: str  # e.g., "llama", "qwen"
    context_length: int  # Model's claimed max context


@dataclass
class ContextTestResult:
    """Result of testing a specific context size."""

    num_ctx: int
    success: bool
    vram_mb: int = 0
    load_time_s: float = 0.0
    tok_s: float = 0.0
    ttft_ms: float = 0.0
    error: str | None = None


@dataclass
class ContextTestSummary:
    """Summary of context testing for a model."""

    model: str
    claimed_max_ctx: int
    tested_max_ctx: int
    recommended_ctx: int  # 80-90% of tested max for safety
    vram_at_max_mb: int
    results: list[ContextTestResult]


# ─────────────────────────────────────────────────────────────────────────────
# Ollama API
# ─────────────────────────────────────────────────────────────────────────────


async def get_ollama_model_info(model: str) -> OllamaModelInfo | None:
    """Get detailed info about an Ollama model."""
    async with httpx.AsyncClient() as client:
        try:
            # First check if model exists
            resp = await client.post(
                f"{OLLAMA_URL}/api/show",
                json={"name": model},
                timeout=30,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            details = data.get("details", {})
            model_info = data.get("model_info", {})

            # Extract context length from model info
            # Different models store this differently
            context_length = 32768  # Default
            for key in model_info:
                if "context" in key.lower():
                    val = model_info[key]
                    if isinstance(val, int):
                        context_length = val
                        break

            # Also check parameters
            params = data.get("parameters", "")
            if "num_ctx" in params:
                # Parse from parameters string
                for line in params.split("\n"):
                    if "num_ctx" in line:
                        with contextlib.suppress(ValueError, IndexError):
                            context_length = int(line.split()[-1])

            return OllamaModelInfo(
                name=model,
                size_bytes=data.get("size", 0),
                parameter_size=details.get("parameter_size", "unknown"),
                quantization=details.get("quantization_level", "unknown"),
                family=details.get("family", "unknown"),
                context_length=context_length,
            )
        except Exception:
            return None


async def unload_ollama_model(model: str) -> None:
    """Unload a model from Ollama's memory."""
    async with httpx.AsyncClient() as client:
        with contextlib.suppress(Exception):
            await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=30,
            )

    # Wait for GPU memory to actually free up
    await asyncio.sleep(5)


async def get_ollama_loaded_model() -> str | None:
    """Get currently loaded model in Ollama."""
    async with httpx.AsyncClient() as client:
        with contextlib.suppress(Exception):
            resp = await client.get(f"{OLLAMA_URL}/api/ps", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                if models:
                    return models[0].get("name")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Context Testing
# ─────────────────────────────────────────────────────────────────────────────


def generate_test_sizes(max_context: int) -> list[int]:
    """Generate context sizes to test, up to the model's claimed max."""
    sizes = []
    current = 4096

    while current <= max_context:
        sizes.append(current)
        current *= 2

    # Make sure we include the actual max if it's not a power of 2
    if sizes and sizes[-1] < max_context:
        sizes.append(max_context)

    return sizes


async def test_context_size(
    model: str,
    num_ctx: int,
    timeout: float = 120.0,
    speed_test: bool = True,
) -> ContextTestResult:
    """Test if a model can handle a specific context size, and measure speed."""
    start = time.time()

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": "Reply with only: OK",
                    "stream": False,
                    "options": {"num_ctx": num_ctx, "num_predict": 5},
                },
                timeout=timeout,
            )

            elapsed = time.time() - start
            gpu_after = get_gpu_info()

            if resp.status_code == 200:
                # Quick speed benchmark (3 requests via OpenAI-compatible API)
                tok_s = 0.0
                ttft_ms = 0.0
                if speed_test:
                    tok_s, ttft_ms = await _quick_speed_test(model)

                return ContextTestResult(
                    num_ctx=num_ctx,
                    success=True,
                    vram_mb=gpu_after.used_mb,
                    load_time_s=elapsed,
                    tok_s=tok_s,
                    ttft_ms=ttft_ms,
                )

            error = resp.text[:100] if resp.text else f"HTTP {resp.status_code}"
            return ContextTestResult(
                num_ctx=num_ctx,
                success=False,
                error=error,
            )

        except httpx.TimeoutException:
            return ContextTestResult(
                num_ctx=num_ctx,
                success=False,
                error="timeout",
            )
        except Exception as e:
            error = str(e)
            return ContextTestResult(
                num_ctx=num_ctx,
                success=False,
                error=error[:100],
            )


async def _quick_speed_test(model: str) -> tuple[float, float]:
    """Run 3 quick streaming requests to measure tok/s and TTFT."""
    from model_manager.benchmark import _bench_single, _median

    prompts = [
        "Explain gravity in two sentences.",
        "What is the capital of France?",
        "Write a haiku about rain.",
    ]
    tok_counts = []
    gen_times = []
    ttfts = []

    async with httpx.AsyncClient() as client:
        for prompt in prompts:
            try:
                result = await _bench_single(
                    client,
                    f"{OLLAMA_URL}/v1",
                    model,
                    prompt,
                    max_tokens=64,
                )
                if result.token_count > 0:
                    gen_time = result.total_s - (result.ttft_ms / 1000)
                    if gen_time > 0:
                        tok_counts.append(result.token_count)
                        gen_times.append(gen_time)
                        ttfts.append(result.ttft_ms)
            except Exception:
                continue

    if not tok_counts:
        return 0.0, 0.0

    total_tok_s = sum(tok_counts) / sum(gen_times)
    median_ttft = _median(ttfts)
    return round(total_tok_s, 1), round(median_ttft)


async def test_model_context(
    model: str,
    on_progress: Callable[[str], None] | None = None,
) -> ContextTestSummary:
    """
    Test a model to find its maximum working context on this hardware.

    Args:
        model: Ollama model name (e.g., "glm-ocr:latest")
        on_progress: Optional callback(num_ctx, success, message)

    Returns:
        ContextTestSummary with results
    """

    def report(msg: str):
        if on_progress:
            on_progress(msg)

    report(f"Getting model info for {model}...")

    # Get model info to determine max context
    info = await get_ollama_model_info(model)
    if info:
        claimed_max = info.context_length
        report(f"Model claims max context: {claimed_max:,}")
    else:
        # Default to 128K if we can't determine
        claimed_max = 131072
        report(f"Could not get model info, assuming max context: {claimed_max:,}")

    # Generate test sizes
    test_sizes = generate_test_sizes(claimed_max)
    report(f"Will test: {', '.join(f'{s:,}' for s in test_sizes)}")

    # Unload any existing model first
    report("Unloading existing models...")
    await unload_ollama_model(model)

    results: list[ContextTestResult] = []
    max_working = 0
    vram_at_max = 0

    for ctx_size in test_sizes:
        report(f"Testing num_ctx={ctx_size:,}...")

        result = await test_context_size(model, ctx_size)
        results.append(result)

        if result.success:
            max_working = ctx_size
            vram_at_max = result.vram_mb
            speed_str = f", {result.tok_s} tok/s" if result.tok_s else ""
            report(
                f"  ✓ {ctx_size:,} OK ({result.load_time_s:.1f}s, VRAM: {result.vram_mb}MB{speed_str})"
            )
        else:
            report(f"  ✗ {ctx_size:,} failed: {result.error}")
            # Stop testing larger sizes
            break

        # Pause between tests to let Ollama settle
        await asyncio.sleep(3)

    # Calculate recommended (80% of max for safety margin)
    recommended = int(max_working * 0.8) if max_working > 0 else 4096

    # Round to nice number
    if recommended > 65536:
        recommended = (recommended // 8192) * 8192
    elif recommended > 8192:
        recommended = (recommended // 4096) * 4096
    else:
        recommended = (recommended // 1024) * 1024

    report(f"Testing complete. Max: {max_working:,}, Recommended: {recommended:,}")

    return ContextTestSummary(
        model=model,
        claimed_max_ctx=claimed_max,
        tested_max_ctx=max_working,
        recommended_ctx=recommended,
        vram_at_max_mb=vram_at_max,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sync wrapper
# ─────────────────────────────────────────────────────────────────────────────


def test_model_context_sync(
    model: str,
    on_progress: Callable[[str], None] | None = None,
) -> ContextTestSummary:
    """Synchronous wrapper for test_model_context."""
    return asyncio.run(test_model_context(model, on_progress))
