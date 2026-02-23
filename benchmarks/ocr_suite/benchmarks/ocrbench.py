"""OCRBench benchmark module.

Wraps the OCRBench evaluation logic into a clean module interface that
uses the shared inference layer from ``..inference``.

Supports two data loading paths:
1. HuggingFace dataset (preferred) — images embedded as PIL objects.
2. Raw JSON + images on disk — ``OCRBench.json`` + ``OCRBench_Images/``.

Usage::

    from openai import OpenAI
    from pathlib import Path
    from ocr_suite.benchmarks.ocrbench import run

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    result = run(client, "my-model", Path("datasets/ocrbench"), Path("results"))
    print(result.score, "/", result.max_score)
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openai import OpenAI

from ..inference import run_inference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring categories (10 total, max 1000)
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
    "Scene Text-centric VQA",
    "Doc-oriented VQA",
    "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OCRBenchResult:
    """Aggregated result from an OCRBench evaluation run."""

    score: int
    max_score: int = 1000
    categories: dict[str, int] = field(default_factory=dict)
    errors: int = 0


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_prediction(predict: str, answers: str | list, dataset_name: str) -> int:
    """Evaluate whether *predict* matches any of *answers*.

    * **HME100k** (math expressions): strip all whitespace, case-sensitive
      containment check.
    * **Everything else**: case-insensitive, strip leading/trailing whitespace
      and collapse newlines, containment check.

    Returns 1 on match, 0 otherwise.
    """
    if not isinstance(answers, list):
        answers = [answers]

    if dataset_name == "HME100k":
        predict_clean = predict.strip().replace("\n", " ").replace(" ", "")
        for answer in answers:
            answer_clean = answer.strip().replace("\n", " ").replace(" ", "")
            if answer_clean in predict_clean:
                return 1
    else:
        predict_clean = predict.lower().strip().replace("\n", " ")
        for answer in answers:
            answer_clean = answer.lower().strip().replace("\n", " ")
            if answer_clean in predict_clean:
                return 1

    return 0


def _compute_scores(data: list[dict]) -> dict[str, int]:
    """Tally per-category scores from evaluated items."""
    scores: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for item in data:
        cat = item.get("type")
        if "result" in item and cat in scores:
            scores[cat] += item["result"]
    return scores


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_from_hf(dataset_dir: Path, limit: int | None = None) -> list[dict] | None:
    """Try to load from HuggingFace save_to_disk format at *dataset_dir*.

    Returns list of dicts with keys: id, dataset_name, question, answers,
    type, image_bytes.  Returns None if the HF dataset isn't found.
    """
    # Check for save_to_disk marker files
    if not (dataset_dir / "dataset_dict.json").is_file() and not (
        dataset_dir / "dataset_info.json"
    ).is_file():
        return None

    try:
        from datasets import load_from_disk
    except ImportError:
        return None

    try:
        ds = load_from_disk(str(dataset_dir))
    except Exception:
        return None

    # Get the right split
    if hasattr(ds, "keys"):
        if "test" in ds:
            ds = ds["test"]
        elif "train" in ds:
            ds = ds["train"]
        else:
            ds = ds[list(ds.keys())[0]]

    items = []
    count = len(ds) if limit is None else min(limit, len(ds))
    for i in range(count):
        sample = ds[i]

        # Convert PIL image to bytes
        img = sample.get("image")
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        items.append({
            "id": i,
            "dataset_name": sample.get("dataset", ""),
            "question": sample.get("question", ""),
            "answers": sample.get("answer", []),
            "type": sample.get("question_type", ""),
            "image_bytes": image_bytes,
        })

    return items if items else None


def _load_from_json(dataset_dir: Path, limit: int | None = None) -> list[dict]:
    """Load from raw OCRBench.json + OCRBench_Images/ on disk."""
    # Resolve JSON
    candidates_json = [
        dataset_dir / "OCRBench.json",
        dataset_dir / "OCRBench" / "OCRBench.json",
    ]
    json_path = None
    for p in candidates_json:
        if p.is_file():
            json_path = p
            break
    if json_path is None:
        raise FileNotFoundError(
            f"OCRBench.json not found. Tried: {', '.join(str(c) for c in candidates_json)}"
        )

    # Resolve images dir
    candidates_img = [
        dataset_dir / "OCRBench_Images",
        dataset_dir / "OCRBench" / "OCRBench_Images",
    ]
    images_dir = None
    for p in candidates_img:
        if p.is_dir():
            images_dir = p
            break
    if images_dir is None:
        raise FileNotFoundError(
            f"OCRBench_Images not found. Tried: {', '.join(str(c) for c in candidates_img)}"
        )

    with open(json_path, "r") as f:
        data: list[dict] = json.load(f)

    if limit is not None:
        data = data[:limit]

    # Add image paths and read bytes
    for item in data:
        image_path = images_dir / item["image_path"]
        if image_path.is_file():
            item["image_bytes"] = image_path.read_bytes()
        else:
            item["image_bytes"] = None
            logger.warning("Image not found: %s", image_path)

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    client: OpenAI,
    model: str,
    dataset_dir: Path,
    results_dir: Path,
    limit: int | None = None,
    resume: bool = False,
    on_progress: Callable | None = None,
) -> OCRBenchResult:
    """Run the OCRBench evaluation.

    Args:
        client: OpenAI SDK client pointed at the target endpoint.
        model: Model name/ID for chat completions.
        dataset_dir: Directory containing either a HF dataset cache or
            ``OCRBench.json`` + ``OCRBench_Images/``.
        results_dir: Where to write ``ocrbench_predictions.json``.
        limit: If set, only evaluate the first *limit* items.
        resume: If *True*, skip items that already have predictions saved
            in the output file.
        on_progress: Optional callback invoked every 50 items (and on
            completion) with ``(completed: int, total: int)``.

    Returns:
        An :class:`OCRBenchResult` with the final score breakdown.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "ocrbench_predictions.json"

    # -- load data (try HF first, then JSON) ---------------------------------
    data = _load_from_hf(dataset_dir, limit)
    if data is None:
        data = _load_from_json(dataset_dir, limit)

    total = len(data)

    # -- resume handling -----------------------------------------------------
    completed_ids: set = set()
    if resume and output_path.is_file():
        with open(output_path, "r") as f:
            existing: list[dict] = json.load(f)
        for item in existing:
            if "predict" in item:
                completed_ids.add(item["id"])
                for d in data:
                    if d["id"] == item["id"]:
                        d["predict"] = item["predict"]
                        d["result"] = item.get("result", 0)
        logger.info("Resuming: %d items already completed", len(completed_ids))

    # -- inference loop ------------------------------------------------------
    errors = 0
    completed = 0

    for item in data:
        if item.get("id") in completed_ids:
            completed += 1
            continue

        image_bytes = item.get("image_bytes")
        if image_bytes is None:
            item["predict"] = ""
            item["result"] = 0
            errors += 1
            completed += 1
            continue

        result = run_inference(
            client=client,
            model=model,
            image_bytes=image_bytes,
            prompt=item["question"],
            max_tokens=512,
        )

        if result.error is not None:
            logger.warning("Inference error on item %s: %s", item.get("id"), result.error)
            item["predict"] = ""
            item["result"] = 0
            errors += 1
        else:
            item["predict"] = result.text
            item["result"] = _evaluate_prediction(
                result.text, item["answers"], item["dataset_name"]
            )

        completed += 1

        # -- incremental save ------------------------------------------------
        save_data = [{k: v for k, v in d.items() if k != "image_bytes"} for d in data]
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)

        # -- progress callback -----------------------------------------------
        if on_progress is not None and (completed % 50 == 0 or completed == total):
            on_progress(completed, total)

    # -- final save ----------------------------------------------------------
    save_data = [{k: v for k, v in d.items() if k != "image_bytes"} for d in data]
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    # -- compute scores ------------------------------------------------------
    categories = _compute_scores(data)
    score = sum(categories.values())

    return OCRBenchResult(
        score=score,
        max_score=1000,
        categories=categories,
        errors=errors,
    )
