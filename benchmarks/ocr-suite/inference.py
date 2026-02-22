"""Shared inference module for OCR suite benchmarks.

Provides a common interface for sending images to OpenAI-compatible
vision endpoints (vLLM, Ollama, llama.cpp, etc.) and collecting results.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    text: str
    elapsed_s: float
    error: str | None


def encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its contents as a base64 string."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _get_mime_type(image_path: Path) -> str:
    """Determine MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime = _MIME_TYPES.get(ext)
    if mime is None:
        raise ValueError(f"Unsupported image extension: {ext}")
    return mime


def detect_model(client: OpenAI) -> str:
    """Auto-detect the model name by querying /v1/models.

    Returns the ID of the first model listed by the endpoint.
    Raises RuntimeError if no models are available.
    """
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models available at the endpoint")
    return models.data[0].id


def run_inference(
    client: OpenAI,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> InferenceResult:
    """Send an image + prompt to an OpenAI-compatible vision endpoint.

    Args:
        client: OpenAI SDK client configured with the target base_url.
        model: Model name/ID to use for the completion.
        image_path: Path to the image file on disk.
        prompt: Text prompt to send alongside the image.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.

    Returns:
        InferenceResult with the model's text response, wall-clock time,
        and an error string (None on success).
    """
    try:
        b64 = encode_image_base64(image_path)
        mime = _get_mime_type(image_path)
        data_url = f"data:{mime};base64,{b64}"

        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.monotonic() - t0

        text = response.choices[0].message.content.strip()
        return InferenceResult(text=text, elapsed_s=elapsed, error=None)

    except Exception as exc:
        elapsed = time.monotonic() - t0 if "t0" in locals() else 0.0
        return InferenceResult(text="", elapsed_s=elapsed, error=str(exc))
