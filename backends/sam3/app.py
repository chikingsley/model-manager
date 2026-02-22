from __future__ import annotations

import io
import logging
import os
from typing import Any

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

logger = logging.getLogger("sam3-service")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
DEVICE = os.getenv("SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="sam3-service", version="0.1.0")

_processor: Sam3Processor | None = None
_load_error: str | None = None


class SegmentRequest(BaseModel):
    image_url: str | None = None
    image_path: str | None = None
    prompt: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


async def _load_remote_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url)
        response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _load_local_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@app.on_event("startup")
async def startup() -> None:
    global _processor, _load_error
    try:
        logger.info("Loading SAM3 model '%s' on %s", MODEL_ID, DEVICE)
        model = build_sam3_image_model(model_id=MODEL_ID)
        model = model.to(DEVICE)
        model.eval()
        _processor = Sam3Processor(model)
        _load_error = None
        logger.info("SAM3 model loaded")
    except Exception as exc:  # pragma: no cover - runtime-only path
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("SAM3 startup failed")


@app.get("/health")
async def health() -> dict[str, str]:
    if _load_error:
        return {"status": "error", "detail": _load_error}
    if _processor is None:
        return {"status": "loading"}
    return {"status": "ok"}


@app.post("/segment")
async def segment(req: SegmentRequest) -> dict[str, Any]:
    if _load_error:
        raise HTTPException(status_code=500, detail=f"SAM3 not available: {_load_error}")
    if _processor is None:
        raise HTTPException(status_code=503, detail="SAM3 still loading")

    if not req.image_url and not req.image_path:
        raise HTTPException(status_code=400, detail="Provide image_url or image_path")

    if req.image_url and req.image_path:
        raise HTTPException(status_code=400, detail="Provide only one of image_url or image_path")

    try:
        image = await _load_remote_image(req.image_url) if req.image_url else _load_local_image(req.image_path or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {exc}") from exc

    try:
        state = _processor.set_image(image)
        output = _processor.set_text_prompt(state=state, prompt=req.prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SAM3 inference failed: {exc}") from exc

    masks = _to_numpy(output.get("masks", []))
    boxes = _to_numpy(output.get("boxes", []))
    scores = _to_numpy(output.get("scores", []))

    if masks.ndim == 4:
        masks = masks[0]
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 2:
        scores = scores[0]

    if scores.size:
        order = np.argsort(scores)[::-1][: req.top_k]
        masks = masks[order] if masks.size else masks
        boxes = boxes[order] if boxes.size else boxes
        scores = scores[order]

    mask_pixels = [int(np.count_nonzero(mask)) for mask in masks] if masks.size else []

    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "model": MODEL_ID,
        "prompt": req.prompt,
        "num_masks": int(len(mask_pixels)),
        "scores": [float(x) for x in scores.tolist()] if scores.size else [],
        "boxes": boxes.tolist() if boxes.size else [],
        "mask_pixels": mask_pixels,
    }
