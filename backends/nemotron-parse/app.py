"""Nemotron Parse v1.2 OCR service.

Wraps the HuggingFace Transformers model in an OpenAI-compatible
chat completions API so it works with the benchmark suite, model-manager,
and any OpenAI SDK client.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("nemotron-parse")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("NEMOTRON_MODEL_ID", "nvidia/NVIDIA-Nemotron-Parse-v1.2")
DEVICE = os.getenv("NEMOTRON_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.getenv("NEMOTRON_PORT", "8097"))

# Nemotron Parse uses special token sequences, not natural language prompts.
DEFAULT_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

app = FastAPI(title="nemotron-parse", version="0.1.0")

_model = None
_processor = None
_generation_config = None
_load_error: str | None = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    global _model, _processor, _generation_config, _load_error
    try:
        from transformers import AutoModel, AutoProcessor, GenerationConfig

        logger.info("Loading model '%s' on %s", MODEL_ID, DEVICE)

        _model = (
            AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .to(DEVICE)
            .eval()
        )

        _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        _generation_config = GenerationConfig.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )

        _load_error = None
        logger.info("Model loaded (%s)", DEVICE)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Startup failed")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    if _load_error:
        return {"status": "error", "detail": _load_error}
    if _model is None:
        return {"status": "loading"}
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/models
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "nvidia",
            }
        ],
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/chat/completions
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | list = ""


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1, le=9000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


def _extract_image_and_prompt(messages: list[ChatMessage]) -> tuple[Image.Image | None, str]:
    """Pull the first image and text from OpenAI-format messages."""
    image = None
    text_parts: list[str] = []

    for msg in messages:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
            continue
        for part in msg.content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif part_type == "image_url" and image is None:
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # data:image/png;base64,AAAA...
                    header, _, b64 = url.partition(",")
                    raw = base64.b64decode(b64)
                    image = Image.open(io.BytesIO(raw)).convert("RGB")

    prompt = " ".join(text_parts).strip()
    return image, prompt


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict:
    if _load_error:
        raise HTTPException(status_code=500, detail=f"Model not available: {_load_error}")
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    image, _user_prompt = _extract_image_and_prompt(req.messages)
    if image is None:
        raise HTTPException(status_code=400, detail="No image found in messages")

    # Always use the model's internal prompt format.
    prompt = DEFAULT_PROMPT

    t0 = time.monotonic()
    try:
        inputs = _processor(
            images=[image], text=prompt, return_tensors="pt", add_special_tokens=False
        ).to(DEVICE)

        with torch.inference_mode():
            outputs = _model.generate(
                **inputs,
                generation_config=_generation_config,
                max_new_tokens=req.max_tokens,
            )

        text = _processor.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as exc:
        import traceback
        logger.error("Inference error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - t0

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "timing": {"inference_s": round(elapsed, 3)},
    }
