"""
Configuration and hardware detection.

Handles loading config.yaml and building runtime configurations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from model_manager.state import ModelEntry

ModelFormat = Literal["gguf", "awq", "gptq", "safetensors", "exl2"]
Backend = Literal["llama.cpp", "vllm", "ollama", "sam3", "exllamav2"]
ModelType = Literal["chat", "embedding", "reranker", "vision"]

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


class HardwareConfig(BaseModel):
    """Hardware configuration."""

    gpu: str = "RTX 5070"
    vram_gb: int = 12
    ram_gb: int = 94
    compute_cap: float = 12.0


class PathsConfig(BaseModel):
    """Path configuration."""

    models: Path = Path("/home/simon/models")
    hf_cache: Path = Path.home() / ".cache/huggingface"
    vllm_compose: Path = Path("/home/simon/docker/model-manager/backends/vllm/docker-compose.yml")
    llama_compose: Path = Path("/home/simon/docker/model-manager/backends/llama/docker-compose.yml")
    ollama_compose: Path = Path("/home/simon/docker/model-manager/backends/ollama/docker-compose.yml")


class TunnelConfig(BaseModel):
    """Cloudflare tunnel configuration."""

    enabled: bool = True
    domain: str = "peacockery.studio"


class SystemConfig(BaseModel):
    """Complete system configuration from config.yaml."""

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    tunnel: TunnelConfig = Field(default_factory=TunnelConfig)


class RuntimeConfig(BaseModel):
    """Generated runtime configuration for a model."""

    backend: Backend
    flags: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    port: int = 8000


Mode = Literal["max_performance", "multi_model"]


# ─────────────────────────────────────────────────────────────────────────────
# Config Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: Path | None = None) -> SystemConfig:
    """Load system configuration from config.yaml."""
    path = config_path or Path(__file__).parent.parent.parent / "config.yaml"

    if not path.exists():
        return SystemConfig()

    with path.open(encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle) or {}

    return SystemConfig(
        hardware=HardwareConfig(**data.get("hardware", {})),
        paths=PathsConfig(**{k: Path(v) for k, v in data.get("paths", {}).items()}),
        tunnel=TunnelConfig(**data.get("tunnel", {})),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_format(model_id: str, path: str | None = None) -> ModelFormat:
    """Detect model format from name or path."""
    name = (path or model_id).lower()

    if name.endswith(".gguf") or ".gguf" in name:
        return "gguf"
    if "awq" in name:
        return "awq"
    if "gptq" in name:
        return "gptq"
    if "exl2" in name:
        return "exl2"
    return "safetensors"


def detect_backend(model_format: ModelFormat) -> Backend:
    """Pick backend based on format."""
    rules: dict[ModelFormat, Backend] = {
        "gguf": "llama.cpp",
        "awq": "vllm",
        "gptq": "vllm",
        "safetensors": "vllm",
        "exl2": "exllamav2",
    }
    return rules.get(model_format, "vllm")


def detect_type(model_id: str) -> ModelType:
    """Detect model type from name."""
    name = model_id.lower()

    if "embed" in name:
        return "embedding"
    if "rerank" in name:
        return "reranker"
    if "vision" in name or "-vl" in name or "ocr" in name:
        return "vision"
    return "chat"


def detect_quant(model_id: str) -> str | None:
    """Detect quantization from name."""
    name = model_id.lower()

    if "awq" in name:
        return "awq"
    if "gptq" in name:
        return "gptq"
    if "fp8" in name:
        return "fp8"
    if "int8" in name:
        return "int8"

    # GGUF quantization patterns
    gguf_match = re.search(r"q[0-9]+_[a-z0-9_]+", name, re.IGNORECASE)
    if gguf_match:
        return gguf_match.group(0).upper()

    return None


def estimate_vram(model_id: str, model_format: ModelFormat) -> float:
    """Estimate VRAM in GB. Rough approximation."""
    name = model_id.lower()

    # Extract parameter count
    param_match = re.search(r"(\d+\.?\d*)b", name)
    params_b = float(param_match.group(1)) if param_match else 7.0

    # Bytes per parameter based on quantization
    quant = detect_quant(model_id)
    if quant:
        q = quant.lower()
        if "q4" in q or "4bit" in q:
            bytes_per_param = 0.5
        elif "q5" in q:
            bytes_per_param = 0.625
        elif "q6" in q:
            bytes_per_param = 0.75
        elif "q8" in q or "int8" in q:
            bytes_per_param = 1.0
        elif "awq" in q or "gptq" in q:
            bytes_per_param = 0.5
        elif "fp8" in q:
            bytes_per_param = 1.0
        else:
            bytes_per_param = 2.0
    else:
        bytes_per_param = 2.0  # FP16 default

    # Add 20% overhead for KV cache, activations, etc.
    vram_gb = params_b * bytes_per_param * 1.2
    return round(vram_gb, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Config Building
# ─────────────────────────────────────────────────────────────────────────────


def build_runtime_config(
    model: ModelEntry,
    hardware: HardwareConfig,
    mode: Mode = "max_performance",
) -> RuntimeConfig:
    """
    Build runtime configuration from model + hardware.

    Modes:
    - max_performance: Single model, max throughput. No sleep mode, max KV offload.
    - multi_model: Multiple models, fast switching. Sleep mode enabled, smaller KV.
    """
    flags: list[str] = []
    env: dict[str, str] = {}
    port = 8000

    backend = model.backend or "vllm"
    model_type = model.type or "chat"
    quant = model.quant

    if backend == "vllm":
        # V1 Engine + Core Optimizations (always enabled)
        env["VLLM_USE_V1"] = "1"
        env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        flags.append("--async-scheduling")

        if mode == "max_performance":
            # No sleep mode - dedicate RAM to KV cache
            kv_size = min(int(hardware.ram_gb * 0.66), 80)
            flags.append("--kv-offloading-backend native")
            flags.append(f"--kv-offloading-size {kv_size}")
            flags.append("--disable-hybrid-kv-cache-manager")
            flags.append("--kv-cache-dtype fp8")
        else:  # multi_model
            flags.append("--enable-sleep-mode")
            kv_size = min(hardware.ram_gb // 4, 32)
            flags.append("--kv-offloading-backend native")
            flags.append(f"--kv-offloading-size {kv_size}")
            flags.append("--disable-hybrid-kv-cache-manager")

        # Quantization flags
        if quant == "awq":
            flags.append("--quantization awq")
        elif quant == "gptq":
            flags.append("--quantization gptq")
        elif quant == "fp8":
            flags.append("--quantization fp8")

        # Model type specific
        if model_type == "embedding":
            flags.append("--task embed")
            port = 8085
        elif model_type == "reranker":
            flags.append("--task score")
            port = 8085
        elif model_type == "vision":
            flags.append("--limit-mm-per-prompt '{\"image\": 1}'")

    elif backend == "llama.cpp":
        flags.append("-ngl 999")  # Full GPU offload
        flags.append("--flash-attn")

        if model_type == "embedding":
            flags.append("--embedding")
            port = 8085

    return RuntimeConfig(
        backend=backend,
        flags=flags,
        env=env,
        port=port,
    )


def build_vllm_env(
    model: str,
    runtime: RuntimeConfig,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.90,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a complete .env file for vLLM docker-compose."""
    env = {
        "MODEL": model,
        "MAX_MODEL_LEN": str(max_model_len),
        "GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),
        "VLLM_PORT": str(runtime.port),
        **runtime.env,
        "EXTRA_ARGS": " ".join(runtime.flags),
    }

    if extra_env:
        env.update(extra_env)

    return env
