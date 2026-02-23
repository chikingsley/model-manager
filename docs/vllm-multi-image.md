# vLLM Multi-Image Architecture

Some models need specific vLLM versions or different Python dependencies. Instead of one Dockerfile for all models, the vLLM backend supports **per-model Docker images** selected via environment variables.

## How It Works

The `docker-compose.yml` reads three env vars from `.env`:

| Env Var | Default | Purpose |
|---------|---------|---------|
| `VLLM_IMAGE` | `local/vllm-openai:ocr2-nightly` | Docker image to run |
| `VLLM_DOCKERFILE` | `Dockerfile` | Which Dockerfile for `docker compose build` |
| `VLLM_PYTHONPATH` | _(empty)_ | Extra Python import path inside container |

When the model-manager activates a model, `build_vllm_env()` writes these to `.env` based on the `VllmModeConfig` for that model. Models without image overrides use the defaults.

## Available Images

| Image Tag | Dockerfile | Base | Models |
|-----------|------------|------|--------|
| `local/vllm-openai:ocr2-nightly` | `Dockerfile` | `vllm/vllm-openai:nightly` | GLM-OCR, DeepSeek-OCR-2, LightOnOCR, Qwen chat models |
| `local/vllm-openai:nemotron` | `Dockerfile.nemotron` | `vllm/vllm-openai:v0.14.1` | Nemotron Parse v1.2 |

## Building Images

```bash
cd /home/simon/docker/model-manager/backends/vllm

# Default image (nightly, for most models)
docker compose build

# Nemotron image
VLLM_DOCKERFILE=Dockerfile.nemotron VLLM_IMAGE=local/vllm-openai:nemotron docker compose build
```

Or directly:

```bash
docker build -t local/vllm-openai:ocr2-nightly -f Dockerfile .
docker build -t local/vllm-openai:nemotron -f Dockerfile.nemotron .
```

## Adding a New Image Variant

1. Create `Dockerfile.<variant>` in `backends/vllm/` with the required base version and deps.

2. Build and tag: `docker build -t local/vllm-openai:<variant> -f Dockerfile.<variant> .`

3. Add a model config in `modes.py` → `get_vllm_mode_config()`:

```python
if "my-model" in selected_model_lc:
    return VllmModeConfig(
        model=model or "org/My-Model",
        max_model_len=8192,
        description="OCR (My Model)",
        extra_args=("--trust-remote-code", "--disable-log-requests"),
        gpu_memory_utilization=0.90,
        image="local/vllm-openai:<variant>",
        dockerfile="Dockerfile.<variant>",
        pythonpath="/vllm-workspace",  # only if mounting custom Python modules
    )
```

1. The model-manager handles the rest — writes the image/dockerfile to `.env` and restarts the container.

## Nemotron Parse v1.2

### Activation

```bash
mm ocr nvidia/NVIDIA-Nemotron-Parse-v1.2
```

### Architecture Notes

Nemotron Parse is an encoder-decoder model (RADIO vision encoder + mBART decoder) — different from the decoder-only models that GLM-OCR and DeepSeek-OCR-2 use. It needs:

- **vLLM v0.14.1** (pinned in `Dockerfile.nemotron`)
- **Extra deps**: `albumentations`, `timm`, `open_clip_torch` for the RADIO encoder
- **Custom logits processors** mounted at `/vllm-workspace/` and loaded via `--logits-processors`
- **PYTHONPATH=/vllm-workspace** so the EngineCore subprocess can import the logits processors

### Known Issue: Pixel Values Transfer Bug

As of vLLM v0.14.1, the V1 engine's inter-process architecture (APIServer → EngineCore) does not correctly transfer `pixel_values` tensors for encoder-decoder multimodal models. Images are preprocessed correctly in the APIServer process but arrive as all-white in EngineCore.

**Status**: Upstream vLLM bug. When fixed, update the `FROM` line in `Dockerfile.nemotron`.

**Evidence**: Debug logging showed APIServer pixel_values `mean=238.4` (correct) vs EngineCore pixel_values `mean=1.0` (all white). Different input images produce identical model output.
