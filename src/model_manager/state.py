"""
State management for model registry.

Handles reading/writing models.yaml and tracking active state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

ModelFormat = Literal["gguf", "awq", "gptq", "safetensors", "exl2"]
Backend = Literal["llama.cpp", "vllm", "ollama", "sam3", "exllamav2"]
ModelType = Literal["chat", "embedding", "reranker", "vision"]
ActiveState = Literal["voice", "llama", "ollama", "ocr", "chat", "perf", "embed", "sam3", "none"]


class ContextSpeedPoint(BaseModel):
    """Speed measurement at a specific context size."""

    num_ctx: int
    tok_s: float
    ttft_ms: float
    vram_mb: int


class ModelEntry(BaseModel):
    """A registered model."""

    source: str
    path: str | None = None
    format: ModelFormat | None = None
    backend: Backend | None = None
    type: ModelType = "chat"
    quant: str | None = None
    vram_estimate: float | None = None
    notes: str | None = None
    port: int | None = None
    # Ollama-specific
    model: str | None = None  # ollama model name
    num_ctx: int | None = None  # context to use when loading
    # Context testing results (Ollama)
    tested_num_ctx: int | None = None  # max context that worked
    claimed_num_ctx: int | None = None  # model's claimed max
    context_tested: bool = False  # whether we've run context test
    # Benchmark results (all backends)
    bench_tok_s: float | None = None  # tokens per second
    bench_ttft_ms: float | None = None  # time to first token (median, ms)
    bench_itl_ms: float | None = None  # inter-token latency (median, ms)
    bench_p95_ms: float | None = None  # P95 inter-token latency (ms)
    bench_date: str | None = None  # ISO date of last benchmark
    benchmarked: bool = False
    # Context-speed profile (Ollama — tok/s at each tested context size)
    context_profile: list[ContextSpeedPoint] | None = None


class SetupEntry(BaseModel):
    """A named setup configuration."""

    description: str | None = None
    model: str | None = None
    models: list[str] | dict[str, str] | None = None
    type: str | None = None
    num_ctx: int | None = None


class RegistryState(BaseModel):
    """Current state of the model manager."""

    active: ActiveState = "none"
    sleeping: list[str] = Field(default_factory=list)


class ModelsRegistry(BaseModel):
    """The complete models.yaml structure."""

    models: dict[str, ModelEntry] = Field(default_factory=dict)
    setups: dict[str, SetupEntry] = Field(default_factory=dict)
    state: RegistryState = Field(default_factory=RegistryState)


# ─────────────────────────────────────────────────────────────────────────────
# State Manager
# ─────────────────────────────────────────────────────────────────────────────


class StateManager:
    """Manages the models.yaml registry."""

    def __init__(self, registry_path: Path | None = None):
        if registry_path:
            self.registry_path = registry_path
        elif env := os.getenv("MM_MODELS_FILE"):
            self.registry_path = Path(env)
        else:
            self.registry_path = Path(__file__).parent.parent.parent / "models.yaml"

    def load(self) -> ModelsRegistry:
        """Load the registry from disk."""
        if not self.registry_path.exists():
            return ModelsRegistry()

        with self.registry_path.open(encoding="utf-8") as file_handle:
            data = yaml.safe_load(file_handle) or {}

        # Parse with defaults
        return ModelsRegistry(
            models={k: ModelEntry(**v) for k, v in data.get("models", {}).items()},
            setups={k: SetupEntry(**v) for k, v in data.get("setups", {}).items()},
            state=RegistryState(**data.get("state", {})),
        )

    def save(self, registry: ModelsRegistry) -> None:
        """Save the registry to disk."""
        # Convert to dict, excluding None values for cleaner YAML
        data = {
            "models": {
                k: {kk: vv for kk, vv in v.model_dump().items() if vv is not None}
                for k, v in registry.models.items()
            },
            "setups": {
                k: {kk: vv for kk, vv in v.model_dump().items() if vv is not None}
                for k, v in registry.setups.items()
            },
            "state": {k: v for k, v in registry.state.model_dump().items() if v},
        }

        with self.registry_path.open("w", encoding="utf-8") as file_handle:
            yaml.dump(data, file_handle, default_flow_style=False, sort_keys=False)

    def get_active(self) -> ActiveState:
        """Get the current active state."""
        return self.load().state.active

    def set_active(self, state: ActiveState) -> None:
        """Set the active state."""
        registry = self.load()
        registry.state.active = state
        self.save(registry)

    def get_model(self, model_id: str) -> ModelEntry | None:
        """Get a specific model entry."""
        registry = self.load()
        return registry.models.get(model_id)

    def register_model(self, model_id: str, entry: ModelEntry) -> None:
        """Register or update a model."""
        registry = self.load()
        registry.models[model_id] = entry
        self.save(registry)
