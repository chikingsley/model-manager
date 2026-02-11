"""
Container management for model backends.

Handles docker operations: start, stop, kill, health checks, compose.
This is the core logic extracted from the mm bash script.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import docker as docker_sdk
import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

ContainerName = Literal[
    "nemotron",
    "vllm",
    "llama-server",
    "ollama",
    "voice-tunnel",
    "vllm-tunnel",
    "llama-tunnel",
    "ollama-tunnel",
]

HealthStatus = Literal["healthy", "starting", "unhealthy", "none", "not_running"]


@dataclass
class GpuInfo:
    """GPU memory and utilization info."""

    used_mb: int
    total_mb: int
    util_percent: int
    temperature: int = 0

    @property
    def used_gb(self) -> float:
        return round(self.used_mb / 1024, 1)

    @property
    def total_gb(self) -> float:
        return round(self.total_mb / 1024, 1)

    @property
    def free_gb(self) -> float:
        return round((self.total_mb - self.used_mb) / 1024, 1)

    @property
    def percent(self) -> float:
        return round(self.used_mb / self.total_mb * 100, 1) if self.total_mb else 0.0


@dataclass
class RamInfo:
    """System RAM info."""

    used_gb: float
    total_gb: float

    @property
    def free_gb(self) -> float:
        return round(self.total_gb - self.used_gb, 1)


@dataclass
class ServiceStatus:
    """Status of a running service."""

    name: str
    running: bool
    healthy: HealthStatus
    model: str | None = None
    port: int | None = None
    endpoint: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

VLLM_DIR = Path("/home/simon/vllm")
LLAMA_DIR = Path("/home/simon/llama-server")
MODELS_DIR = Path("/home/simon/models")


# ─────────────────────────────────────────────────────────────────────────────
# Docker Client (lazy singleton — talks to /var/run/docker.sock)
# ─────────────────────────────────────────────────────────────────────────────

_client: docker_sdk.DockerClient | None = None

# Docker binary path for compose commands (compose has no SDK support)
_DOCKER_BIN: str = shutil.which("docker") or "docker"


def _get_client() -> docker_sdk.DockerClient:
    """Get or create a shared Docker client."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = docker_sdk.from_env()
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Docker Operations (sync, via SDK)
# ─────────────────────────────────────────────────────────────────────────────


def is_running(container: str) -> bool:
    """Check if a container is running."""
    try:
        c = _get_client().containers.get(container)
        return c.status == "running"
    except (docker_sdk.errors.NotFound, docker_sdk.errors.APIError):
        return False


def container_exists(container: str) -> bool:
    """Check if a container exists (running or stopped)."""
    try:
        _get_client().containers.get(container)
        return True
    except docker_sdk.errors.NotFound:
        return False
    except docker_sdk.errors.APIError:
        return False


def get_health(container: str) -> HealthStatus:
    """Get container health status."""
    try:
        c = _get_client().containers.get(container)
    except (docker_sdk.errors.NotFound, docker_sdk.errors.APIError):
        return "not_running"

    if c.status != "running":
        return "not_running"

    health = c.attrs.get("State", {}).get("Health", {}).get("Status")

    if health == "healthy":
        return "healthy"
    if health == "starting":
        return "starting"
    if health == "unhealthy":
        return "unhealthy"
    return "none"  # No health check configured


def docker_start(container: str) -> bool:
    """Start a stopped container."""
    try:
        _get_client().containers.get(container).start()
        return True
    except (docker_sdk.errors.NotFound, docker_sdk.errors.APIError):
        return False


def docker_stop(container: str, timeout: int = 10) -> bool:
    """Stop a container gracefully."""
    try:
        _get_client().containers.get(container).stop(timeout=timeout)
        return True
    except (docker_sdk.errors.NotFound, docker_sdk.errors.APIError):
        return False


def docker_kill(container: str) -> bool:
    """Kill a container immediately (fast stop)."""
    try:
        _get_client().containers.get(container).kill()
        return True
    except (docker_sdk.errors.NotFound, docker_sdk.errors.APIError):
        return False


def compose_up(compose_dir: Path, service: str | None = None, recreate: bool = False) -> bool:
    """Run docker compose up."""
    cmd = [_DOCKER_BIN, "compose", "up", "-d"]
    if recreate:
        cmd.append("--force-recreate")
    if service:
        cmd.append(service)

    result = subprocess.run(cmd, cwd=compose_dir, capture_output=True)
    return result.returncode == 0


def compose_down(compose_dir: Path) -> bool:
    """Run docker compose down."""
    result = subprocess.run(
        [_DOCKER_BIN, "compose", "down"],
        cwd=compose_dir,
        capture_output=True,
    )
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Resource Monitoring
# ─────────────────────────────────────────────────────────────────────────────


def get_gpu_info() -> GpuInfo:
    """Get GPU memory and utilization."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = result.stdout.strip().split(", ")
        return GpuInfo(
            used_mb=int(parts[0]),
            total_mb=int(parts[1]),
            util_percent=int(parts[2]),
            temperature=int(parts[3]) if len(parts) > 3 else 0,
        )
    except Exception:
        return GpuInfo(used_mb=0, total_mb=12288, util_percent=0, temperature=0)


def get_ram_info() -> RamInfo:
    """Get system RAM usage."""
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as file_handle:
            lines = file_handle.readlines()

        total_kb = available_kb = 0
        for line in lines:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])

        return RamInfo(
            used_gb=round((total_kb - available_kb) / 1024 / 1024, 1),
            total_gb=round(total_kb / 1024 / 1024, 1),
        )
    except Exception:
        return RamInfo(used_gb=0.0, total_gb=94.0)


# ─────────────────────────────────────────────────────────────────────────────
# Service-Specific Operations
# ─────────────────────────────────────────────────────────────────────────────


def write_env_file(path: Path, env: dict[str, str]) -> None:
    """Write a .env file."""
    content = "\n".join(f"{k}={v}" for k, v in env.items())
    path.write_text(content + "\n")


def read_env_file(path: Path) -> dict[str, str]:
    """Read a .env file."""
    if not path.exists():
        return {}

    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def get_vllm_env() -> dict[str, str]:
    """Get current vLLM .env configuration."""
    return read_env_file(VLLM_DIR / ".env")


def get_llama_env() -> dict[str, str]:
    """Get current llama-server .env configuration."""
    return read_env_file(LLAMA_DIR / ".env")


def list_gguf_models() -> list[str]:
    """List available GGUF models."""
    if not MODELS_DIR.exists():
        return []
    return [f.name for f in MODELS_DIR.glob("*.gguf")]


# ─────────────────────────────────────────────────────────────────────────────
# Async HTTP Operations (for health checks and API calls)
# ─────────────────────────────────────────────────────────────────────────────


async def check_http_health(url: str, timeout: float = 5.0) -> bool:
    """Check if an HTTP endpoint is healthy."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False


async def wait_for_healthy(
    container: str,
    timeout: int = 120,
    poll_interval: float = 2.0,
) -> bool:
    """Wait for a container to become healthy."""
    elapsed = 0.0
    while elapsed < timeout:
        health = get_health(container)
        if health == "healthy":
            return True
        if health in ("unhealthy", "not_running"):
            return False
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    return False


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Sleep/Wake
# ─────────────────────────────────────────────────────────────────────────────


async def vllm_sleep(port: int = 8000, level: int = 2) -> bool:
    """Put vLLM to sleep."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"http://localhost:{port}/sleep?level={level}",
                timeout=30,
            )
            return resp.status_code == 200
        except Exception:
            return False


async def vllm_wake(port: int = 8000) -> bool:
    """Wake vLLM from sleep."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"http://localhost:{port}/wake_up",
                timeout=60,
            )
            return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Operations
# ─────────────────────────────────────────────────────────────────────────────


async def ollama_list_models() -> list[str]:
    """List available Ollama models."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
    return []


async def ollama_get_loaded() -> str | None:
    """Get the currently loaded Ollama model."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:11434/api/ps", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                if models:
                    return models[0].get("name")
        except Exception:
            pass
    return None


async def ollama_load_model(model: str, num_ctx: int = 32768) -> bool:
    """Load an Ollama model into memory."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": "test",
                    "stream": False,
                    "keep_alive": -1,  # Keep loaded indefinitely
                    "options": {"num_ctx": num_ctx, "num_predict": 1},
                },
                timeout=120,  # Model loading can take time
            )
            return resp.status_code == 200
        except Exception:
            return False


async def ollama_is_running() -> bool:
    """Check if Ollama is running."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False


async def ollama_model_exists(model: str) -> bool:
    """Check if a model exists locally in Ollama."""
    models = await ollama_list_models()
    # Check exact match or with :latest suffix
    if model in models:
        return True
    # Handle "model" matching "model:latest"
    if ":" not in model and f"{model}:latest" in models:
        return True
    # Handle "model:latest" matching "model"
    if model.endswith(":latest"):
        base = model.rsplit(":", 1)[0]
        if base in models:
            return True
    return False


async def ollama_pull_model(
    model: str,
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """
    Pull an Ollama model from the registry.

    Args:
        model: Model name (e.g., "granite4:latest")
        on_progress: Optional callback for progress updates

    Returns:
        True if pull succeeded, False otherwise
    """

    def report(msg: str):
        if on_progress:
            on_progress(msg)

    report(f"Pulling {model}...")

    async with httpx.AsyncClient() as client:
        try:
            # Use streaming to track progress
            async with client.stream(
                "POST",
                "http://localhost:11434/api/pull",
                json={"name": model, "stream": True},
                timeout=600,  # 10 min timeout for large models
            ) as resp:
                if resp.status_code != 200:
                    report(f"Pull failed: HTTP {resp.status_code}")
                    return False

                last_status = ""
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json

                        data = json.loads(line)
                        status = data.get("status", "")
                        if status != last_status:
                            # Report status changes (downloading, verifying, etc.)
                            if "pulling" in status or "downloading" in status:
                                completed = data.get("completed", 0)
                                total = data.get("total", 0)
                                if total > 0:
                                    pct = int(completed / total * 100)
                                    report(f"  {status}: {pct}%")
                                else:
                                    report(f"  {status}")
                            elif status:
                                report(f"  {status}")
                            last_status = status
                    except Exception:
                        pass

            report(f"Pull complete: {model}")
            return True

        except httpx.TimeoutException:
            report("Pull timed out")
            return False
        except Exception as e:
            report(f"Pull failed: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Status Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def get_running_services() -> list[ServiceStatus]:
    """Get status of all known services."""
    services = []

    if is_running("nemotron"):
        services.append(
            ServiceStatus(
                name="nemotron",
                running=True,
                healthy=get_health("nemotron"),
                port=18000,
                endpoint="https://llm-voice.peacockery.studio",
            )
        )

    if is_running("vllm"):
        env = get_vllm_env()
        services.append(
            ServiceStatus(
                name="vllm",
                running=True,
                healthy=get_health("vllm"),
                model=env.get("MODEL"),
                port=int(env.get("VLLM_PORT", 8000)),
                endpoint="https://vllm.peacockery.studio",
            )
        )

    if is_running("llama-server"):
        env = get_llama_env()
        services.append(
            ServiceStatus(
                name="llama-server",
                running=True,
                healthy=get_health("llama-server"),
                model=env.get("MODEL"),
                port=8090,
                endpoint="https://llama.peacockery.studio",
            )
        )

    if is_running("ollama"):
        services.append(
            ServiceStatus(
                name="ollama",
                running=True,
                healthy="healthy" if is_running("ollama") else "not_running",
                port=11434,
                endpoint="https://ollama.peacockery.studio",
            )
        )

    return services


def get_running_tunnels() -> list[str]:
    """Get list of running tunnels."""
    tunnels = []
    for name in ["voice-tunnel", "vllm-tunnel", "llama-tunnel", "ollama-tunnel"]:
        if is_running(name):
            tunnels.append(name.replace("-tunnel", ""))
    return tunnels
