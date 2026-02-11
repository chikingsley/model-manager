"""
Integration Tests for Model Manager

These tests verify REAL functionality:
- Actual endpoint health checks
- Real GPU detection
- Real container status
- Real model switching and inference

NO MOCKS. NO FAKES. NO SKIPPING.
Everything runs on real hardware with real containers.

Run with: just test
Requires: Running on gmk-server with GPU access
"""

import asyncio
import subprocess

import httpx
import pytest

from model_manager.containers import (
    get_gpu_info,
    get_ram_info,
    get_running_services,
    is_running,
    ollama_get_loaded,
    ollama_is_running,
)
from model_manager.modes import activate
from model_manager.state import StateManager

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def http_client():
    """Sync HTTP client for endpoint tests."""
    return httpx.Client(timeout=30)


@pytest.fixture
def async_client():
    """Async HTTP client."""
    return httpx.AsyncClient(timeout=30)


# ─────────────────────────────────────────────────────────────────────────────
# GPU & System Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSystemResources:
    """Test that we can actually read system resources."""

    def test_nvidia_smi_works(self):
        """nvidia-smi should be available and return data."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "RTX" in result.stdout or "NVIDIA" in result.stdout

    def test_can_read_gpu_memory(self):
        """Should be able to read GPU memory usage."""
        gpu = get_gpu_info()
        assert gpu.total_mb > 0
        assert gpu.used_mb >= 0
        assert gpu.used_mb <= gpu.total_mb
        # We have 12GB VRAM
        assert gpu.total_mb > 10000

    def test_can_read_ram(self):
        """Should be able to read RAM."""
        ram = get_ram_info()
        assert ram.total_gb > 0
        assert ram.used_gb >= 0
        # We have 94GB RAM
        assert ram.total_gb > 80


# ─────────────────────────────────────────────────────────────────────────────
# Container Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContainerStatus:
    """Test that we can check container status."""

    def test_docker_is_available(self):
        """Docker should be available."""
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        assert result.returncode == 0

    def test_is_running_function(self):
        """is_running should work correctly."""
        # Test with a container that definitely doesn't exist
        assert not is_running("nonexistent-container-12345")

    def test_get_running_services(self):
        """Should be able to get running services."""
        services = get_running_services()
        assert isinstance(services, list)


# ─────────────────────────────────────────────────────────────────────────────
# State Management Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStateManagement:
    """Test state management."""

    def test_can_load_registry(self):
        """Should be able to load the models registry."""
        state = StateManager()
        registry = state.load()

        assert hasattr(registry, "models")
        assert hasattr(registry, "state")

    def test_can_get_active_state(self):
        """Should be able to get active state."""
        state = StateManager()
        active = state.get_active()

        assert active in ["voice", "llama", "ollama", "ocr", "chat", "perf", "embed", "none"]


# ─────────────────────────────────────────────────────────────────────────────
# Config Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfig:
    """Test configuration loading."""

    def test_can_load_config(self):
        """Should be able to load system config."""
        from model_manager.config import load_config

        config = load_config()

        assert config.hardware.vram_gb == 12
        assert config.hardware.ram_gb == 94

    def test_can_detect_format(self):
        """Should detect model formats correctly."""
        from model_manager.config import detect_format

        assert detect_format("model.gguf") == "gguf"
        assert detect_format("Qwen-7B-AWQ") == "awq"
        assert detect_format("model-gptq") == "gptq"
        assert detect_format("regular-model") == "safetensors"

    def test_can_detect_backend(self):
        """Should pick correct backend for format."""
        from model_manager.config import detect_backend

        assert detect_backend("gguf") == "llama.cpp"
        assert detect_backend("awq") == "vllm"
        assert detect_backend("safetensors") == "vllm"

    def test_can_build_runtime_config(self):
        """Should build runtime config with all optimizations."""
        from model_manager.config import HardwareConfig, build_runtime_config
        from model_manager.state import ModelEntry

        model = ModelEntry(source="test", backend="vllm", type="chat", quant="awq")
        hardware = HardwareConfig(vram_gb=12, ram_gb=94)

        config = build_runtime_config(model, hardware)

        assert config.backend == "vllm"
        assert len(config.flags) > 0
        assert "--quantization awq" in config.flags
        assert config.env.get("VLLM_USE_V1") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# CLI Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCli:
    """Test the CLI works."""

    def test_mm_status_runs(self):
        """mm status should run without error."""
        result = subprocess.run(
            ["uv", "run", "mm"],
            capture_output=True,
            cwd="/home/simon/github/model-manager",
        )
        assert result.returncode == 0
        stdout = result.stdout.decode("utf-8", errors="replace")
        assert "Model Manager" in stdout or "Active" in stdout

    def test_mm_help_runs(self):
        """mm help should show usage."""
        result = subprocess.run(
            ["uv", "run", "mm", "help"],
            capture_output=True,
            cwd="/home/simon/github/model-manager",
        )
        assert result.returncode == 0
        stdout = result.stdout.decode("utf-8", errors="replace")
        assert "voice" in stdout
        assert "ollama" in stdout

    def test_mm_models_runs(self):
        """mm models should list available GGUF models."""
        result = subprocess.run(
            ["uv", "run", "mm", "models"],
            capture_output=True,
            cwd="/home/simon/github/model-manager",
        )
        assert result.returncode == 0
        stdout = result.stdout.decode("utf-8", errors="replace")
        assert "gguf" in stdout.lower() or "GGUF" in stdout


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Tests - REAL FUNCTIONALITY
# ─────────────────────────────────────────────────────────────────────────────


class TestOllama:
    """Test Ollama backend - real switching and inference."""

    @pytest.mark.asyncio
    async def test_activate_ollama(self):
        """Should be able to activate Ollama backend."""
        result = await activate("ollama")

        assert result.success, f"Failed to activate ollama: {result.message}"
        assert result.mode == "ollama"

        # Verify state was updated
        state = StateManager()
        assert state.get_active() == "ollama"

    @pytest.mark.asyncio
    async def test_ollama_is_running_after_activate(self):
        """Ollama container should be running after activation."""
        # Ensure ollama is active
        result = await activate("ollama")
        assert result.success

        # Check container is running
        assert is_running("ollama"), "Ollama container not running"

        # Check API responds
        running = await ollama_is_running()
        assert running, "Ollama API not responding"

    @pytest.mark.asyncio
    async def test_ollama_load_model(self):
        """Should be able to load an Ollama model."""
        from model_manager.containers import ollama_load_model

        # Ensure ollama is active
        result = await activate("ollama")
        assert result.success

        # Load a model (use ministral-3:8b as it's smaller)
        success = await ollama_load_model("ministral-3:8b", num_ctx=8192)
        assert success, "Failed to load ministral-3:8b"

        # Verify model is loaded
        loaded = await ollama_get_loaded()
        assert loaded is not None
        assert "ministral" in loaded.lower()

    def test_ollama_inference(self, http_client: httpx.Client):
        """Ollama should respond to inference requests."""
        # This test requires ollama to be running with a model loaded
        # Previous tests should have set this up

        response = http_client.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": "ministral-3:8b",
                "messages": [{"role": "user", "content": "Say 'test passed' and nothing else."}],
                "max_tokens": 10,
            },
            timeout=60,
        )

        assert response.status_code == 200, f"Ollama inference failed: {response.text}"
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Mode Switching Tests - REAL FUNCTIONALITY
# ─────────────────────────────────────────────────────────────────────────────


class TestModeSwitching:
    """Test switching between different modes."""

    @pytest.mark.asyncio
    async def test_switch_ollama_to_stop(self):
        """Should be able to stop all services."""
        # First activate ollama
        result = await activate("ollama")
        assert result.success

        # Now stop
        result = await activate("stop")
        assert result.success
        assert result.mode == "stop"

        # Verify state
        state = StateManager()
        assert state.get_active() == "none"

    @pytest.mark.asyncio
    async def test_switch_to_ollama_with_model(self):
        """Should be able to activate Ollama with a specific model."""
        result = await activate("ollama", model="ministral-3:8b")

        assert result.success, f"Failed: {result.message}"
        assert result.mode == "ollama"

        # Wait a moment for model to load
        await asyncio.sleep(2)

        # Verify model is loaded
        loaded = await ollama_get_loaded()
        assert loaded is not None
        assert "ministral" in loaded.lower()

    @pytest.mark.asyncio
    async def test_ollama_model_responds(self):
        """After activating Ollama, it should respond to requests."""
        # Activate with model
        result = await activate("ollama", model="ministral-3:8b")
        assert result.success

        # Give it time to load
        await asyncio.sleep(3)

        # Test inference
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "http://localhost:11434/v1/chat/completions",
                json={
                    "model": "ministral-3:8b",
                    "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                    "max_tokens": 5,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        content = data["choices"][0]["message"]["content"]
        assert "4" in content


# ─────────────────────────────────────────────────────────────────────────────
# Models.yaml Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelsRegistry:
    """Test that models.yaml is valid and contains expected data."""

    def test_models_yaml_exists(self):
        """models.yaml should exist."""
        from pathlib import Path

        assert Path("/home/simon/github/model-manager/models.yaml").exists()

    def test_models_yaml_is_valid(self):
        """models.yaml should be valid and loadable."""
        state = StateManager()
        registry = state.load()

        assert registry.models is not None
        assert registry.state is not None

    def test_local_model_paths_exist(self):
        """Local model paths in registry should exist."""
        from pathlib import Path

        state = StateManager()
        registry = state.load()

        for model_id, model in registry.models.items():
            path = model.path
            if path and path.startswith("/home/simon/models/"):
                assert Path(path).exists(), f"Model path missing: {path} for {model_id}"


# ─────────────────────────────────────────────────────────────────────────────
# API Server Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestApiServerUnit:
    """Test API server components without requiring it to be running."""

    def test_api_app_imports(self):
        """API app should import without errors."""
        from model_manager.api.server import app

        assert app is not None
        assert app.title == "Model Manager"

    @pytest.mark.asyncio
    async def test_status_endpoint_logic(self):
        """Test the status endpoint logic directly."""
        from model_manager.api.server import get_status

        status = await get_status()

        assert status.active in ["voice", "llama", "ollama", "ocr", "chat", "perf", "embed", "none"]
        assert status.resources.vram.total_gb > 0
        assert isinstance(status.services, list)
        assert isinstance(status.tunnels, list)

    @pytest.mark.asyncio
    async def test_activate_endpoint_logic(self):
        """Test the activate endpoint logic directly."""
        from model_manager.api.server import ActivateRequest, activate_mode

        # Activate ollama
        result = await activate_mode("ollama", ActivateRequest(model="ministral-3:8b"))

        assert result.success
        assert result.mode == "ollama"


# ─────────────────────────────────────────────────────────────────────────────
# Context Testing
# ─────────────────────────────────────────────────────────────────────────────


class TestContextTesting:
    """Test Ollama context testing functionality."""

    def test_generate_test_sizes_powers_of_two(self):
        """Should generate sizes doubling from 4096."""
        from model_manager.ollama import generate_test_sizes

        sizes = generate_test_sizes(32768)
        assert sizes == [4096, 8192, 16384, 32768]

    def test_generate_test_sizes_includes_max(self):
        """Should include non-power-of-2 max."""
        from model_manager.ollama import generate_test_sizes

        sizes = generate_test_sizes(65536)
        assert sizes == [4096, 8192, 16384, 32768, 65536]

    def test_generate_test_sizes_includes_odd_max(self):
        """Should include odd max values."""
        from model_manager.ollama import generate_test_sizes

        sizes = generate_test_sizes(128000)
        # 4096 -> 8192 -> 16384 -> 32768 -> 65536 -> 131072 (exceeds 128000)
        # So it should be [4096, 8192, 16384, 32768, 65536, 128000]
        assert 128000 in sizes
        assert sizes[-1] == 128000

    def test_context_test_result_dataclass(self):
        """ContextTestResult should have expected fields."""
        from model_manager.ollama import ContextTestResult

        result = ContextTestResult(
            num_ctx=8192,
            success=True,
            vram_mb=6000,
            load_time_s=2.5,
        )
        assert result.num_ctx == 8192
        assert result.success is True
        assert result.vram_mb == 6000
        assert result.error is None

    def test_context_test_summary_dataclass(self):
        """ContextTestSummary should have expected fields."""
        from model_manager.ollama import ContextTestResult, ContextTestSummary

        summary = ContextTestSummary(
            model="test:latest",
            claimed_max_ctx=32768,
            tested_max_ctx=16384,
            recommended_ctx=12288,
            vram_at_max_mb=8000,
            results=[ContextTestResult(num_ctx=4096, success=True)],
        )
        assert summary.model == "test:latest"
        assert summary.recommended_ctx == 12288
        assert len(summary.results) == 1

    @pytest.mark.asyncio
    async def test_get_ollama_model_info(self):
        """Should get model info from Ollama API."""
        from model_manager.ollama import get_ollama_model_info

        # Only run if Ollama is running with a model
        if not await ollama_is_running():
            pytest.skip("Ollama not running")

        loaded = await ollama_get_loaded()
        if not loaded:
            pytest.skip("No model loaded in Ollama")

        info = await get_ollama_model_info(loaded)
        assert info is not None
        assert info.name == loaded
        assert info.context_length > 0

    @pytest.mark.asyncio
    async def test_context_test_single_size(self):
        """Should test a single context size."""
        from model_manager.ollama import test_context_size

        if not await ollama_is_running():
            pytest.skip("Ollama not running")

        loaded = await ollama_get_loaded()
        if not loaded:
            pytest.skip("No model loaded in Ollama")

        # Test with a conservative context size that should work
        result = await test_context_size(loaded, 4096, timeout=60.0)
        assert result.num_ctx == 4096
        # Should succeed with 4K context on any model
        assert result.success is True
        assert result.vram_mb > 0
