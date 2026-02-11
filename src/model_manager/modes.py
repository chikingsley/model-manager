"""
Mode activation and switching.

This is the core orchestration logic - handles switching between backends,
stopping conflicting services, and starting new ones.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from model_manager.config import build_runtime_config, build_vllm_env, load_config
from model_manager.containers import (
    LLAMA_DIR,
    MODELS_DIR,
    VLLM_DIR,
    compose_up,
    docker_kill,
    docker_start,
    get_llama_env,
    get_vllm_env,
    is_running,
    list_gguf_models,
    ollama_list_models,
    ollama_load_model,
    ollama_model_exists,
    ollama_pull_model,
    wait_for_healthy,
    write_env_file,
)
from model_manager.state import ContextSpeedPoint, ModelEntry, StateManager

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

Mode = Literal["voice", "llama", "ollama", "ocr", "chat", "perf", "embed", "stop"]


@dataclass
class ActivationResult:
    """Result of a mode activation."""

    success: bool
    mode: Mode
    message: str
    details: dict | None = None


@dataclass
class ActivationProgress:
    """Progress update during activation."""

    step: str
    elapsed: float = 0.0
    health: str | None = None


ProgressCallback = Callable[[ActivationProgress], None]


# ─────────────────────────────────────────────────────────────────────────────
# Model Key + Auto-Benchmark
# ─────────────────────────────────────────────────────────────────────────────

BenchmarkBackend = Literal["ollama", "vllm", "llama.cpp"]


def _get_model_key(backend: BenchmarkBackend, model: str) -> str:
    """Generate a registry key for any backend."""
    if backend == "ollama":
        return "ollama-" + model.replace(":", "-").replace("/", "-")
    clean = model.lower()
    if "/" in clean:
        clean = clean.split("/")[-1]
    if clean.endswith(".gguf"):
        clean = clean[:-5]
    clean = clean.replace("_", "-").replace(":", "-")
    backend_key = backend.replace(".", "-")
    return f"{backend_key}-{clean}"


BACKEND_URLS = {
    "ollama": "http://localhost:11434/v1",
    "llama.cpp": "http://localhost:8090/v1",
    "vllm": "http://localhost:8000/v1",
}


async def _auto_benchmark_if_needed(
    backend: BenchmarkBackend,
    model: str,
    progress: ProgressCallback | None = None,
    port: int | None = None,
) -> None:
    """Run benchmark on first use, like context testing for Ollama."""
    from model_manager.benchmark import run_benchmark

    state = StateManager()
    model_key = _get_model_key(backend, model)

    # Check if already benchmarked
    existing = state.get_model(model_key)
    if existing and existing.benchmarked:
        if progress:
            tok_s = existing.bench_tok_s or 0
            progress(ActivationProgress(step=f"Benchmark: {tok_s} tok/s (cached)"))
        return

    def report(msg: str):
        if progress:
            progress(ActivationProgress(step=msg))

    report("Auto-benchmarking (first use)...")

    # Determine URL
    if port:
        base_url = f"http://localhost:{port}/v1"
    else:
        base_url = BACKEND_URLS.get(backend, "http://localhost:8000/v1")

    try:
        result = await run_benchmark(
            base_url=base_url,
            model=model if backend != "llama.cpp" else None,  # llama.cpp auto-detects
            num_requests=10,
            max_tokens=128,
            on_progress=report,
            backend=backend,
        )
    except Exception as e:
        report(f"Benchmark failed: {e}")
        return

    # Update existing entry or create new one
    if existing:
        existing.bench_tok_s = result.tok_s
        existing.bench_ttft_ms = result.ttft_ms
        existing.bench_itl_ms = result.itl_ms
        existing.bench_p95_ms = result.p95_itl_ms
        existing.bench_date = date.today().isoformat()
        existing.benchmarked = True
        state.register_model(model_key, existing)
    else:
        entry = ModelEntry(
            source=model,
            backend=backend,
            bench_tok_s=result.tok_s,
            bench_ttft_ms=result.ttft_ms,
            bench_itl_ms=result.itl_ms,
            bench_p95_ms=result.p95_itl_ms,
            bench_date=date.today().isoformat(),
            benchmarked=True,
        )
        state.register_model(model_key, entry)

    report(f"Benchmark: {result.tok_s} tok/s | TTFT {result.ttft_ms}ms | P95 {result.p95_itl_ms}ms")


# ─────────────────────────────────────────────────────────────────────────────
# Stop Operations
# ─────────────────────────────────────────────────────────────────────────────


def stop_gpu_services(exclude: list[str] | None = None) -> list[str]:
    """Stop all GPU services except those in exclude list. Returns stopped services."""
    exclude = exclude or []
    stopped = []

    gpu_services = ["nemotron", "vllm", "llama-server", "ollama"]

    for service in gpu_services:
        if service not in exclude and is_running(service):
            docker_kill(service)
            stopped.append(service)

    return stopped


def stop_all() -> ActivationResult:
    """Stop all model services."""
    stopped = stop_gpu_services()

    state = StateManager()
    state.set_active("none")

    return ActivationResult(
        success=True,
        mode="stop",
        message=f"Stopped: {', '.join(stopped)}" if stopped else "No services running",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Voice Mode
# ─────────────────────────────────────────────────────────────────────────────


async def activate_voice(progress: ProgressCallback | None = None) -> ActivationResult:
    """Activate voice stack (nemotron: LLM + ASR + TTS)."""

    def report(step: str, health: str | None = None):
        if progress:
            progress(ActivationProgress(step=step, health=health))

    report("Stopping conflicting services...")

    # Stop vLLM if running (can't have both on GPU)
    if is_running("vllm"):
        docker_kill("vllm")
        await asyncio.sleep(1)

    # Start nemotron if not running
    if not is_running("nemotron"):
        report("Starting nemotron...")
        if not docker_start("nemotron"):
            return ActivationResult(
                success=False,
                mode="voice",
                message="Failed to start nemotron",
            )

    report("Waiting for model to load...", "starting")

    # Wait for healthy
    healthy = await wait_for_healthy("nemotron", timeout=120)

    if healthy:
        report("Ready!", "healthy")
        StateManager().set_active("voice")
        return ActivationResult(
            success=True,
            mode="voice",
            message="Voice stack active",
            details={"endpoint": "https://llm-voice.peacockery.studio"},
        )
    report("Timeout waiting for health", "unhealthy")
    return ActivationResult(
        success=False,
        mode="voice",
        message="nemotron started but not healthy",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Mode
# ─────────────────────────────────────────────────────────────────────────────


def _get_ollama_model_key(model: str) -> str:
    """Convert model name to a valid key for models.yaml."""
    # "glm-ocr:latest" -> "ollama-glm-ocr-latest"
    return "ollama-" + model.replace(":", "-").replace("/", "-")


async def _get_or_test_context(
    model: str,
    progress: ProgressCallback | None = None,
) -> int:
    """
    Get the context size for an Ollama model.
    If not tested before, automatically run context test and save results.
    """
    from model_manager.ollama import test_model_context

    def report(step: str):
        if progress:
            progress(ActivationProgress(step=step))

    state = StateManager()
    model_key = _get_ollama_model_key(model)

    # Check if we have tested this model before
    registry = state.load()
    existing = registry.models.get(model_key)

    # Only trust previous test if it actually succeeded (tested_num_ctx > 0)
    if existing and existing.context_tested and existing.num_ctx and existing.tested_num_ctx:
        report(f"Using tested context: {existing.num_ctx:,}")
        return existing.num_ctx

    # Need to test this model
    report(f"First time using {model}, testing context limits...")

    test_result = await test_model_context(
        model,
        on_progress=lambda msg: report(msg),
    )

    # Only save successful test results (tested_num_ctx > 0)
    if test_result.tested_max_ctx > 0:
        # Build context-speed profile from successful results
        profile = [
            ContextSpeedPoint(
                num_ctx=r.num_ctx,
                tok_s=r.tok_s,
                ttft_ms=r.ttft_ms,
                vram_mb=r.vram_mb,
            )
            for r in test_result.results
            if r.success and r.tok_s > 0
        ]

        entry = ModelEntry(
            source="ollama",
            model=model,
            backend="ollama",
            context_tested=True,
            claimed_num_ctx=test_result.claimed_max_ctx,
            tested_num_ctx=test_result.tested_max_ctx,
            num_ctx=test_result.recommended_ctx,
            vram_estimate=round(test_result.vram_at_max_mb / 1024, 1),
            notes=f"Auto-tested: max {test_result.tested_max_ctx:,}, using {test_result.recommended_ctx:,}",
            context_profile=profile or None,
        )
        state.register_model(model_key, entry)
        report(f"Context test complete: using {test_result.recommended_ctx:,}")
        return test_result.recommended_ctx

    # Test failed completely - use conservative default
    report("Context test failed, using default 32K")
    return 32768


async def activate_ollama(
    model: str | None = None,
    progress: ProgressCallback | None = None,
    auto_pull: bool = True,
) -> ActivationResult:
    """
    Activate Ollama backend.

    If a model is specified and hasn't been tested before, automatically
    runs context testing to find the max context this GPU can handle.

    Args:
        model: Model to load (e.g., "granite4:latest")
        progress: Optional callback for progress updates
        auto_pull: Automatically pull model if not found locally
    """

    def report(step: str, health: str | None = None):
        if progress:
            progress(ActivationProgress(step=step, health=health))

    report("Stopping conflicting services...")

    # Stop other GPU services
    stop_gpu_services(exclude=["ollama", "ollama-tunnel"])

    await asyncio.sleep(1)

    # Start ollama if not running
    if not is_running("ollama"):
        report("Starting Ollama...")
        if not docker_start("ollama"):
            return ActivationResult(
                success=False,
                mode="ollama",
                message="Failed to start Ollama",
            )
        await asyncio.sleep(2)

    # Load model if specified
    num_ctx_used = None
    if model:
        # Check if model exists locally
        if not await ollama_model_exists(model):
            if auto_pull:
                report(f"Model {model} not found locally, pulling...")
                pull_success = await ollama_pull_model(
                    model,
                    on_progress=lambda msg: report(msg),
                )
                if not pull_success:
                    available = await ollama_list_models()
                    return ActivationResult(
                        success=False,
                        mode="ollama",
                        message=f"Failed to pull model: {model}",
                        details={"available_models": available},
                    )
            else:
                available = await ollama_list_models()
                return ActivationResult(
                    success=False,
                    mode="ollama",
                    message=f"Model not found: {model}",
                    details={"available_models": available},
                )

        # Get context (auto-test if needed)
        num_ctx = await _get_or_test_context(model, progress)
        num_ctx_used = num_ctx
        report(f"Loading {model} with {num_ctx:,} context...")
        success = await ollama_load_model(model, num_ctx)
        if not success:
            return ActivationResult(
                success=False,
                mode="ollama",
                message=f"Failed to load model: {model}",
            )

    # Auto-benchmark if model specified
    if model:
        await _auto_benchmark_if_needed("ollama", model, progress)

    # Start tunnel if not running
    if not is_running("ollama-tunnel"):
        docker_start("ollama-tunnel")

    report("Ready!", "healthy")
    StateManager().set_active("ollama")

    return ActivationResult(
        success=True,
        mode="ollama",
        message=f"Ollama active{f' ({model})' if model else ''}",
        details={
            "endpoint": "https://ollama.peacockery.studio",
            "model": model,
            "num_ctx": num_ctx_used,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# llama.cpp Mode
# ─────────────────────────────────────────────────────────────────────────────


async def activate_llama(
    model: str | None = None,
    progress: ProgressCallback | None = None,
) -> ActivationResult:
    """Activate llama.cpp backend with a GGUF model."""

    def report(step: str, health: str | None = None):
        if progress:
            progress(ActivationProgress(step=step, health=health))

    # Default model
    if not model:
        model = "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf"

    # Validate model exists
    model_path = MODELS_DIR / model
    if not model_path.exists():
        available = list_gguf_models()
        return ActivationResult(
            success=False,
            mode="llama",
            message=f"Model not found: {model}",
            details={"available": available},
        )

    report("Stopping conflicting services...")
    stop_gpu_services(exclude=["llama-server", "llama-tunnel"])
    await asyncio.sleep(1)

    # Update .env with model
    report(f"Configuring {model}...")
    current_env = get_llama_env()
    tunnel_token = current_env.get("CLOUDFLARE_LLAMA_TOKEN")
    context = current_env.get("CONTEXT", "8192")

    new_env = {"MODEL": model, "CONTEXT": context}
    if tunnel_token:
        new_env["CLOUDFLARE_LLAMA_TOKEN"] = tunnel_token

    write_env_file(LLAMA_DIR / ".env", new_env)

    # Start or restart
    report("Starting llama-server...")
    if is_running("llama-server"):
        compose_up(LLAMA_DIR, "llama", recreate=True)
    else:
        compose_up(LLAMA_DIR)

    report("Waiting for model to load...", "starting")
    healthy = await wait_for_healthy("llama-server", timeout=120)

    if healthy:
        await _auto_benchmark_if_needed("llama.cpp", model, progress)
        report("Ready!", "healthy")
        StateManager().set_active("llama")
        return ActivationResult(
            success=True,
            mode="llama",
            message=f"llama.cpp active ({model})",
            details={"endpoint": "https://llama.peacockery.studio", "model": model},
        )
    return ActivationResult(
        success=False,
        mode="llama",
        message="llama-server started but not healthy",
    )


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Modes (OCR, Chat, Perf, Embed)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VllmModeConfig:
    model: str
    max_model_len: int
    description: str
    extra_args: tuple[str, ...] = ()
    port: int | None = None
    gpu_memory_utilization: float | None = None


def get_vllm_mode_config(
    mode: Literal["ocr", "chat", "perf", "embed"], model: str | None
) -> VllmModeConfig:
    """Get configuration for a vLLM mode."""
    selected_model = model or ""
    selected_model_lc = selected_model.lower()

    if mode == "ocr":
        if not model or "deepseek-ocr-2" in selected_model_lc:
            return VllmModeConfig(
                model=model or "deepseek-ai/DeepSeek-OCR-2",
                max_model_len=8192,
                description="OCR (DeepSeek-OCR-2)",
                extra_args=(
                    "--trust-remote-code",
                    "--max-num-seqs 32",
                    "--no-enable-prefix-caching",
                    "--disable-log-requests",
                ),
                gpu_memory_utilization=0.82,
            )

        if "lightonocr-2-1b" in selected_model_lc:
            return VllmModeConfig(
                model=model,
                max_model_len=8192,
                description="OCR (LightOnOCR-2-1B)",
                extra_args=(
                    "--max-num-seqs 64",
                    "--mm-processor-cache-gb 0",
                    "--no-enable-prefix-caching",
                    "--disable-log-requests",
                ),
                gpu_memory_utilization=0.85,
            )

        if "glm-ocr" in selected_model_lc:
            return VllmModeConfig(
                model=model,
                max_model_len=8192,
                description="OCR (GLM-OCR)",
                extra_args=(
                    "--max-num-seqs 16",
                    "--max-num-batched-tokens 8192",
                    "--mm-processor-cache-gb 0",
                    "--no-enable-prefix-caching",
                    "--disable-log-requests",
                ),
                gpu_memory_utilization=0.90,
            )

        return VllmModeConfig(
            model=model,
            max_model_len=8192,
            description="OCR (vLLM)",
            extra_args=(
                "--max-num-seqs 64",
                "--no-enable-prefix-caching",
                "--disable-log-requests",
            ),
            gpu_memory_utilization=0.85,
        )

    configs = {
        "embed": VllmModeConfig(
            model="Qwen/Qwen3-Embedding-4B",
            max_model_len=8192,
            description="Embeddings (Qwen3-4B)",
            extra_args=("--task embed",),
            port=8085,
        ),
        "chat": VllmModeConfig(
            model=model or "Qwen/Qwen2.5-7B-Instruct-AWQ",
            max_model_len=8192,
            description="Chat (vLLM)",
        ),
        "perf": VllmModeConfig(
            model=model or "Qwen/Qwen2.5-7B-Instruct-AWQ",
            max_model_len=16384,
            description="Max Performance",
        ),
    }
    return configs[mode]


def get_vllm_runtime_mode(mode: Literal["ocr", "chat", "perf", "embed"]) -> Literal[
    "max_performance", "multi_model"
]:
    """Pick runtime profile for vLLM modes."""
    return "max_performance" if mode in ("ocr", "perf") else "multi_model"


async def activate_vllm_mode(
    mode: Literal["ocr", "chat", "perf", "embed"],
    model: str | None = None,
    progress: ProgressCallback | None = None,
) -> ActivationResult:
    """Activate a vLLM-based mode."""

    def report(step: str, health: str | None = None):
        if progress:
            progress(ActivationProgress(step=step, health=health))

    config = get_vllm_mode_config(mode, model)
    target_model = model or config.model

    report(f"Activating {config.description}...")

    # Stop conflicting services
    stop_gpu_services(exclude=["vllm", "vllm-tunnel"])
    await asyncio.sleep(1)

    # Build runtime config
    sys_config = load_config()
    hardware = sys_config.hardware

    # Create model entry for config building
    model_entry = ModelEntry(
        source=target_model,
        backend="vllm",
        type="vision" if mode == "ocr" else "embedding" if mode == "embed" else "chat",
        quant="awq" if "awq" in target_model.lower() else None,
    )

    runtime_mode = get_vllm_runtime_mode(mode)
    runtime = build_runtime_config(model_entry, hardware, mode=runtime_mode)

    # Add mode-specific extra args
    runtime.flags.extend(list(config.extra_args))

    # Build .env
    report(f"Configuring {target_model}...")
    current_env = get_vllm_env()
    tunnel_token = current_env.get("CLOUDFLARE_VLLM_TOKEN")

    env = build_vllm_env(
        model=target_model,
        runtime=runtime,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization or 0.90,
    )

    if tunnel_token:
        env["CLOUDFLARE_VLLM_TOKEN"] = tunnel_token

    if config.port:
        env["VLLM_PORT"] = str(config.port)

    write_env_file(VLLM_DIR / ".env", env)

    # Start or restart vLLM
    report("Starting vLLM...")
    if is_running("vllm"):
        compose_up(VLLM_DIR, "vllm", recreate=True)
    else:
        compose_up(VLLM_DIR, "vllm")

    # Wait for healthy (vLLM can take a while)
    timeout = 300 if mode == "perf" else 180
    report("Waiting for model to load...", "starting")
    healthy = await wait_for_healthy("vllm", timeout=timeout)

    if healthy:
        # Auto-benchmark (skip embedding models — different API)
        if mode != "embed":
            vllm_port = config.port or 8000
            await _auto_benchmark_if_needed("vllm", target_model, progress, port=vllm_port)
        report("Ready!", "healthy")
        StateManager().set_active(mode)
        return ActivationResult(
            success=True,
            mode=mode,
            message=f"{config.description} active ({target_model})",
            details={"endpoint": "https://vllm.peacockery.studio", "model": target_model},
        )
    return ActivationResult(
        success=False,
        mode=mode,
        message="vLLM started but not healthy",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Activation Entry Point
# ─────────────────────────────────────────────────────────────────────────────


async def activate(
    mode: Mode,
    model: str | None = None,
    progress: ProgressCallback | None = None,
) -> ActivationResult:
    """
    Main entry point for activating a mode.

    Args:
        mode: The mode to activate (voice, llama, ollama, ocr, chat, perf, embed, stop)
        model: Optional model name (for llama, ollama, chat, perf)
        progress: Optional callback for progress updates

    Returns:
        ActivationResult with success status and details
    """
    if mode == "stop":
        return stop_all()

    if mode == "voice":
        return await activate_voice(progress)

    if mode == "ollama":
        return await activate_ollama(model, progress)

    if mode == "llama":
        return await activate_llama(model, progress)

    if mode in ("ocr", "chat", "perf", "embed"):
        return await activate_vllm_mode(mode, model, progress)

    return ActivationResult(
        success=False,
        mode=mode,
        message=f"Unknown mode: {mode}",
    )


# Sync wrapper for CLI usage
def activate_sync(mode: Mode, model: str | None = None) -> ActivationResult:
    """Synchronous wrapper for activate()."""
    return asyncio.run(activate(mode, model))
