"""
Model Manager API Server.

FastAPI server that exposes all model management functionality via HTTP.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from textwrap import dedent
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from model_manager.config import (
    build_runtime_config,
    detect_backend,
    detect_format,
    detect_quant,
    detect_type,
    estimate_vram,
    load_config,
)
from model_manager.containers import (
    get_gpu_info,
    get_ram_info,
    get_running_services,
    get_running_tunnels,
    list_gguf_models,
    ollama_get_loaded,
    ollama_is_running,
    ollama_list_models,
)
from model_manager.modes import Mode, _get_ollama_model_key, activate
from model_manager.state import ActiveState, ModelEntry, StateManager

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Model Manager",
    description="Zero config local LLM infrastructure. Download a model, it works.",
    version="0.2.0",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OCR_BREAKPOINT_RESULT_FILE = (
    REPO_ROOT / "benchmarks" / "results" / "ocr-load" / "glm_ocr_budget_8192_breakpoint_timeout90.json"
)
OCR_BREAKPOINT_RESULT_FILE = Path(
    os.getenv("MM_OCR_BREAKPOINT_FILE", str(DEFAULT_OCR_BREAKPOINT_RESULT_FILE))
)


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────


class ActivateRequest(BaseModel):
    """Request to activate a mode."""

    model: str | None = None


class RegisterRequest(BaseModel):
    """Request to register a new model."""

    model_id: str
    source: str
    path: str | None = None


class VramInfo(BaseModel):
    """VRAM usage info."""

    used_gb: float
    total_gb: float
    free_gb: float
    percent: float


class RamInfo(BaseModel):
    """RAM usage info."""

    used_gb: float
    total_gb: float
    free_gb: float


class ResourcesResponse(BaseModel):
    """Resource usage response."""

    vram: VramInfo
    gpu_util_percent: int
    gpu_temperature: int
    ram: RamInfo


class ServiceInfo(BaseModel):
    """Running service info."""

    name: str
    running: bool
    healthy: str
    model: str | None = None
    port: int | None = None
    endpoint: str | None = None


class StatusResponse(BaseModel):
    """Full status response."""

    active: ActiveState
    services: list[ServiceInfo]
    tunnels: list[str]
    resources: ResourcesResponse
    ollama_model: str | None = None


class ModelInfo(BaseModel):
    """Model info for listing."""

    id: str
    source: str | None = None
    format: str | None = None
    backend: str | None = None
    type: str | None = None
    vram_gb: float | None = None
    status: Literal["active", "sleeping", "stopped"]


class ActivateResponse(BaseModel):
    """Activation result."""

    success: bool
    mode: str
    message: str
    details: dict | None = None


class RegisterResponse(BaseModel):
    """Registration result."""

    success: bool
    model_id: str
    detected: dict


class OcrLoadPoint(BaseModel):
    """Single OCR load-test datapoint."""

    concurrency: int
    ok: int
    total: int
    errors: int
    req_s: float
    p95_s: float | None = None
    timeout_errors: int = 0


class OcrLoadCapabilities(BaseModel):
    """OCR throughput profile derived from recorded benchmark data."""

    model: str
    timeout_s: int
    no_error_max_concurrency: int | None = None
    first_failure_concurrency: int | None = None
    recommended_concurrency_min: int
    recommended_concurrency_max: int
    recommendation_reason: str
    source_file: str
    points: list[OcrLoadPoint]
    sla_profiles: list[dict[str, str | int | bool]]


class CapabilitiesResponse(BaseModel):
    """High-level, machine-readable API capabilities."""

    docs: dict[str, str]
    openapi: dict[str, str]
    mode_behavior: dict[str, str | bool]
    ocr_load: OcrLoadCapabilities | None = None


def _load_ocr_breakpoint_capabilities() -> OcrLoadCapabilities | None:
    """Load OCR load-test summary from committed benchmark results."""
    if not OCR_BREAKPOINT_RESULT_FILE.exists():
        return None

    payload = json.loads(OCR_BREAKPOINT_RESULT_FILE.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    rows = payload.get("results", [])
    if not rows:
        return None

    points = [
        OcrLoadPoint(
            concurrency=int(row["concurrency"]),
            ok=int(row["ok"]),
            total=int(row["total"]),
            errors=int(row["errors"]),
            req_s=float(row["req_s"]),
            p95_s=float(row["p95_s"]) if row.get("p95_s") is not None else None,
            timeout_errors=int(row.get("timeout_errors", 0)),
        )
        for row in rows
    ]
    no_error = [p.concurrency for p in points if p.errors == 0]
    failures = [p.concurrency for p in points if p.errors > 0]

    try:
        source_file = str(OCR_BREAKPOINT_RESULT_FILE.relative_to(REPO_ROOT))
    except ValueError:
        source_file = str(OCR_BREAKPOINT_RESULT_FILE)

    return OcrLoadCapabilities(
        model=str(meta.get("model", "unknown")),
        timeout_s=int(meta.get("timeout_s", 90)),
        no_error_max_concurrency=max(no_error) if no_error else None,
        first_failure_concurrency=min(failures) if failures else None,
        recommended_concurrency_min=40,
        recommended_concurrency_max=56,
        recommendation_reason="Heavy OCR workload tested on RTX 5070 12GB with 90s timeout.",
        source_file=source_file,
        points=points,
        sla_profiles=[
            {
                "name": "safe",
                "timeout_s": int(meta.get("timeout_s", 90)),
                "recommended_concurrency": 40,
                "retry_on_timeout": False,
                "notes": "Headroom-first default for stable production latency.",
            },
            {
                "name": "balanced",
                "timeout_s": int(meta.get("timeout_s", 90)),
                "recommended_concurrency": 56,
                "retry_on_timeout": False,
                "notes": "Highest tested zero-error concurrency for heavy OCR.",
            },
            {
                "name": "aggressive",
                "timeout_s": int(meta.get("timeout_s", 90)),
                "recommended_concurrency": 64,
                "retry_on_timeout": True,
                "notes": "Higher throughput target; expect timeout retries on heavy docs.",
            },
        ],
    )


def _build_llms_txt(base_url: str) -> str:
    """Build concise llms.txt manifest for model-manager."""
    return dedent(
        f"""\
        # Model Manager API

        > Local model orchestration API for backend switching, OCR throughput controls, and benchmark-aware operations.

        Use this service as the canonical control plane for local model operations. It exposes OpenAPI docs, machine-readable capabilities, and OCR throughput limits tested on current hardware.

        ## API

        - [Swagger UI]({base_url}/docs): Interactive API docs for humans and agents.
        - [OpenAPI JSON]({base_url}/openapi.json): Canonical schema for code generation and tool integration.
        - [Capabilities]({base_url}/capabilities): Operational metadata and mode behavior guarantees.
        - [OCR Capabilities]({base_url}/capabilities/ocr): Heavy OCR throughput breakpoints and SLA profiles.
        - [Status]({base_url}/status): Active mode, running services, and endpoint visibility.
        - [Health]({base_url}/health): Liveness endpoint.

        ## Optional

        - [OCR Capabilities JSON]({base_url}/capabilities/ocr): Machine-readable throughput breakpoints and SLA profiles.
        - [Service Status JSON]({base_url}/status): Active backend/model and service endpoints.
        """
    ).strip()


def _build_llms_full_txt(base_url: str) -> str:
    """Build expanded llms-full style context file."""
    ocr = _load_ocr_breakpoint_capabilities()
    if ocr is None:
        ocr_summary = "No OCR breakpoint benchmark data is currently available."
    else:
        ocr_summary = dedent(
            f"""\
            OCR Throughput Envelope (Heavy Docs):
            - Model: {ocr.model}
            - Timeout: {ocr.timeout_s}s
            - Max tested zero-error concurrency: {ocr.no_error_max_concurrency}
            - First tested failure concurrency: {ocr.first_failure_concurrency}
            - Recommended operating range: {ocr.recommended_concurrency_min}-{ocr.recommended_concurrency_max}
            - Source file: {ocr.source_file}
            """
        ).strip()

    return dedent(
        f"""\
        # Model Manager API (Full Context)

        > Expanded model-facing context for integrating with this model-manager deployment.

        Core discovery endpoints:
        - Swagger: {base_url}/docs
        - OpenAPI: {base_url}/openapi.json
        - Capabilities: {base_url}/capabilities
        - OCR capabilities: {base_url}/capabilities/ocr

        Mode behavior guarantees:
        - One backend active at a time.
        - Activation stops previous services for deterministic switching.
        - OCR load guidance is based on measured heavy-document benchmarks.

        {ocr_summary}

        SLA profiles (OCR):
        - safe: concurrency 40, no timeout retries
        - balanced: concurrency 56, no timeout retries
        - aggressive: concurrency 64, retries expected on timeout
        """
    ).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Health & Root
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict:
    """API info."""
    return {
        "name": "model-manager",
        "version": "0.2.0",
        "philosophy": "Zero configuration. Zero flags. Zero bullshit.",
    }


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "ok"}


@app.get("/llms.txt", response_class=PlainTextResponse)
async def llms_txt(request: Request) -> str:
    """LLM-oriented manifest for this API."""
    base_url = str(request.base_url).rstrip("/")
    return _build_llms_txt(base_url)


@app.get("/llms-full.txt", response_class=PlainTextResponse)
async def llms_full_txt(request: Request) -> str:
    """Expanded LLM-oriented context for this API."""
    base_url = str(request.base_url).rstrip("/")
    return _build_llms_full_txt(base_url)


@app.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities() -> CapabilitiesResponse:
    """Get API docs links and tested operational capabilities."""
    return CapabilitiesResponse(
        docs={
            "project_readme": "README.md",
            "ocr_runbook": "docs/deepseek-ocr2-vllm.md",
            "sam3_runbook": "docs/sam3.md",
            "ocr_results": "benchmarks/results/ocr-load/README.md",
        },
        openapi={
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        mode_behavior={
            "single_backend_at_a_time": True,
            "single_active_model_per_backend": True,
            "activation_stops_previous_services": True,
            "notes": "Use POST /activate/{mode} or POST /stop for deterministic switching.",
        },
        ocr_load=_load_ocr_breakpoint_capabilities(),
    )


@app.get("/capabilities/ocr", response_model=OcrLoadCapabilities)
async def get_ocr_capabilities() -> OcrLoadCapabilities:
    """Get tested OCR throughput envelope and breakpoints."""
    data = _load_ocr_breakpoint_capabilities()
    if data is None:
        raise HTTPException(status_code=404, detail="OCR load benchmark data not found")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Status & Resources
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get current status: active mode, services, resources."""
    active: ActiveState = "none"
    services: list[ServiceInfo] = []
    tunnels: list[str] = []
    ollama_model: str | None = None

    try:
        active = StateManager().get_active()
    except Exception:
        logger.exception("status: failed to read active state")

    gpu = get_gpu_info()
    ram = get_ram_info()

    try:
        services = [
            ServiceInfo(
                name=s.name,
                running=s.running,
                healthy=s.healthy,
                model=s.model,
                port=s.port,
                endpoint=s.endpoint,
            )
            for s in get_running_services()
        ]
    except Exception:
        logger.exception("status: failed to enumerate running services")
        services = []

    try:
        tunnels = get_running_tunnels()
    except Exception:
        logger.exception("status: failed to enumerate running tunnels")
        tunnels = []

    try:
        if await ollama_is_running():
            ollama_model = await ollama_get_loaded()
    except Exception:
        logger.exception("status: failed to read ollama loaded model")
        ollama_model = None

    return StatusResponse(
        active=active,
        services=services,
        tunnels=tunnels,
        resources=ResourcesResponse(
            vram=VramInfo(
                used_gb=gpu.used_gb,
                total_gb=gpu.total_gb,
                free_gb=gpu.free_gb,
                percent=gpu.percent,
            ),
            gpu_util_percent=gpu.util_percent,
            gpu_temperature=gpu.temperature,
            ram=RamInfo(
                used_gb=ram.used_gb,
                total_gb=ram.total_gb,
                free_gb=ram.free_gb,
            ),
        ),
        ollama_model=ollama_model,
    )


@app.get("/resources", response_model=ResourcesResponse)
async def get_resources() -> ResourcesResponse:
    """Get current resource usage."""
    gpu = get_gpu_info()
    ram = get_ram_info()

    return ResourcesResponse(
        vram=VramInfo(
            used_gb=gpu.used_gb,
            total_gb=gpu.total_gb,
            free_gb=gpu.free_gb,
            percent=gpu.percent,
        ),
        gpu_util_percent=gpu.util_percent,
        gpu_temperature=gpu.temperature,
        ram=RamInfo(
            used_gb=ram.used_gb,
            total_gb=ram.total_gb,
            free_gb=ram.free_gb,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode Activation
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/activate/{mode}", response_model=ActivateResponse)
async def activate_mode(mode: Mode, req: ActivateRequest | None = None) -> ActivateResponse:
    """Activate a mode (voice, llama, ollama, ocr, chat, perf, embed, sam3, stop)."""
    model = req.model if req else None

    try:
        result = await activate(mode, model)
    except Exception as e:
        logger.exception("activate: mode=%s model=%s failed", mode, model)
        return ActivateResponse(
            success=False,
            mode=mode,
            message=f"Activation failed: {type(e).__name__}: {e}",
            details={"model": model},
        )

    return ActivateResponse(
        success=result.success,
        mode=result.mode,
        message=result.message,
        details=result.details,
    )


@app.post("/stop", response_model=ActivateResponse)
async def stop_all() -> ActivateResponse:
    """Stop all model services."""
    result = await activate("stop")

    return ActivateResponse(
        success=result.success,
        mode=result.mode,
        message=result.message,
        details=result.details,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Models Registry
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List all registered models."""
    state = StateManager()
    registry = state.load()
    active = registry.state.active
    sleeping = registry.state.sleeping

    result = []
    for model_id, model in registry.models.items():
        status: Literal["active", "sleeping", "stopped"] = "stopped"
        if model_id == active:
            status = "active"
        elif model_id in sleeping:
            status = "sleeping"

        result.append(
            ModelInfo(
                id=model_id,
                source=model.source,
                format=model.format,
                backend=model.backend,
                type=model.type,
                vram_gb=model.vram_estimate,
                status=status,
            )
        )

    return result


@app.get("/models/{model_id}")
async def get_model(
    model_id: str,
    mode: Literal["max_performance", "multi_model"] = "max_performance",
) -> dict:
    """Get model details + computed runtime config."""
    state = StateManager()
    model = state.get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    config = load_config()
    runtime = build_runtime_config(model, config.hardware, mode=mode)

    return {
        **model.model_dump(exclude_none=True),
        "runtime_config": runtime.model_dump(),
    }


@app.post("/models/register", response_model=RegisterResponse)
async def register_model(req: RegisterRequest) -> RegisterResponse:
    """Register a new model. Auto-detects format, backend, etc."""
    model_format = detect_format(req.model_id, req.path)
    backend = detect_backend(model_format)
    model_type = detect_type(req.model_id)
    quant = detect_quant(req.model_id)
    vram = estimate_vram(req.model_id, model_format)

    entry = ModelEntry(
        source=req.source,
        path=req.path,
        format=model_format,
        backend=backend,
        type=model_type,
        quant=quant,
        vram_estimate=vram,
    )

    state = StateManager()
    state.register_model(req.model_id, entry)

    return RegisterResponse(
        success=True,
        model_id=req.model_id,
        detected={
            "format": model_format,
            "backend": backend,
            "type": model_type,
            "quant": quant,
            "vram_estimate": vram,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GGUF Models (for llama.cpp)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/models/gguf")
async def get_gguf_models() -> list[str]:
    """List available GGUF models in ~/models/."""
    return list_gguf_models()


# ─────────────────────────────────────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/ollama/models")
async def get_ollama_models() -> dict:
    """List Ollama models (available + tested configurations)."""
    available = await ollama_list_models()
    loaded = await ollama_get_loaded()

    # Get tested configurations from models.yaml
    state = StateManager()
    registry = state.load()
    tested_configs = {}
    for _model_id, entry in registry.models.items():
        if entry.backend == "ollama" and entry.model and entry.context_tested:
            tested_configs[entry.model] = {
                "num_ctx": entry.num_ctx,
                "tested_num_ctx": entry.tested_num_ctx,
                "claimed_num_ctx": entry.claimed_num_ctx,
                "vram_gb": entry.vram_estimate,
            }

    return {
        "available": available,
        "loaded": loaded,
        "tested_configurations": tested_configs,
    }


@app.post("/ollama/load/{model}")
async def load_ollama_model(model: str) -> dict:
    """
    Load an Ollama model with auto-tested context.

    If the model hasn't been tested yet, it will be tested first to find
    the optimal context size for this hardware.
    """
    from model_manager.containers import ollama_load_model, ollama_model_exists, ollama_pull_model
    from model_manager.modes import _get_or_test_context

    # Check if model exists
    if not await ollama_model_exists(model):
        # Try to pull it
        success = await ollama_pull_model(model)
        if not success:
            available = await ollama_list_models()
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model}' not found and failed to pull. Available: {available}",
            )

    # Get tested context (auto-tests if needed)
    num_ctx = await _get_or_test_context(model)

    success = await ollama_load_model(model, num_ctx)

    if success:
        return {"success": True, "model": model, "num_ctx": num_ctx}
    raise HTTPException(status_code=500, detail=f"Failed to load {model}")


@app.post("/ollama/test-context/{model}")
async def test_ollama_context(model: str) -> dict:
    """
    Test an Ollama model to find max working context on this hardware.
    Results are saved to models.yaml for future use.
    """
    from model_manager.containers import ollama_model_exists, ollama_pull_model
    from model_manager.ollama import test_model_context

    # Ensure model exists first
    if not await ollama_model_exists(model):
        success = await ollama_pull_model(model)
        if not success:
            available = await ollama_list_models()
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model}' not found and failed to pull. Available: {available}",
            )

    result = await test_model_context(model)

    # Only save if test succeeded
    if result.tested_max_ctx > 0:
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

    return {
        "model": model,
        "claimed_max_ctx": result.claimed_max_ctx,
        "tested_max_ctx": result.tested_max_ctx,
        "recommended_ctx": result.recommended_ctx,
        "vram_at_max_mb": result.vram_at_max_mb,
        "saved_to_models_yaml": result.tested_max_ctx > 0,
        "results": [
            {
                "num_ctx": r.num_ctx,
                "success": r.success,
                "vram_mb": r.vram_mb,
                "load_time_s": r.load_time_s,
                "error": r.error,
            }
            for r in result.results
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Config Generation
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/config/max-performance")
async def get_max_performance_config() -> dict:
    """Get recommended max performance config for this hardware."""
    config = load_config()
    hardware = config.hardware
    kv_size = min(int(hardware.ram_gb * 0.66), 80)

    return {
        "hardware": hardware.model_dump(),
        "optimizations": {
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "--async-scheduling": "Eliminates GPU idle time",
            "--num-scheduler-steps 10": "28% throughput boost",
            f"--kv-offloading-backend native --kv-offloading-size {kv_size}": f"KV cache to {kv_size}GB RAM",
            "--kv-cache-dtype fp8": "2x context in GPU VRAM",
        },
        "recommended_env": f"""# Max Performance Config for {hardware.gpu} + {hardware.ram_gb}GB RAM
MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.90
VLLM_PORT=8000
VLLM_USE_V1=1
VLLM_ATTENTION_BACKEND=FLASHINFER
EXTRA_ARGS=--async-scheduling --num-scheduler-steps 10 --kv-offloading-backend native --kv-offloading-size {kv_size} --kv-cache-dtype fp8 --quantization awq
""",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main (for direct running)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
