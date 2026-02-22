"""OCRBench benchmark module.

Wraps the OCRBench evaluation logic into a clean module interface that
uses the shared inference layer from ``..inference``.

Usage::

    from openai import OpenAI
    from pathlib import Path
    from benchmarks.ocrbench import run

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    result = run(client, "my-model", Path("datasets/ocrbench"), Path("results"))
    print(result.score, "/", result.max_score)
"""

from __future__ import annotations

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
# Path resolution helpers
# ---------------------------------------------------------------------------


def _resolve_json(dataset_dir: Path) -> Path:
    """Locate ``OCRBench.json`` trying multiple conventional paths."""
    candidates = [
        dataset_dir / "OCRBench.json",
        dataset_dir / "OCRBench" / "OCRBench.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"OCRBench.json not found. Tried: {tried}"
    )


def _resolve_images_dir(dataset_dir: Path) -> Path:
    """Locate the ``OCRBench_Images/`` folder trying multiple paths."""
    candidates = [
        dataset_dir / "OCRBench_Images",
        dataset_dir / "OCRBench" / "OCRBench_Images",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"OCRBench_Images directory not found. Tried: {tried}"
    )


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
        dataset_dir: Directory containing ``OCRBench.json`` and the
            ``OCRBench_Images/`` folder (or an ``OCRBench/`` sub-directory
            with both).
        results_dir: Where to write ``ocrbench_predictions.json``.
        limit: If set, only evaluate the first *limit* items.
        resume: If *True*, skip items that already have predictions saved
            in the output file.
        on_progress: Optional callback invoked every 50 items (and on
            completion) with ``(completed: int, total: int)``.

    Returns:
        An :class:`OCRBenchResult` with the final score breakdown.
    """
    # -- resolve paths -------------------------------------------------------
    json_path = _resolve_json(dataset_dir)
    images_dir = _resolve_images_dir(dataset_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "ocrbench_predictions.json"

    # -- load benchmark data -------------------------------------------------
    with open(json_path, "r") as f:
        data: list[dict] = json.load(f)

    if limit is not None:
        data = data[:limit]

    total = len(data)

    # -- resume handling -----------------------------------------------------
    completed_ids: set = set()
    if resume and output_path.is_file():
        with open(output_path, "r") as f:
            existing: list[dict] = json.load(f)
        for item in existing:
            if "predict" in item:
                completed_ids.add(item["id"])
                # Merge existing predictions back into data
                for d in data:
                    if d["id"] == item["id"]:
                        d["predict"] = item["predict"]
                        d["result"] = item.get("result", 0)
        logger.info("Resuming: %d items already completed", len(completed_ids))

    # -- inference loop ------------------------------------------------------
    errors = 0
    completed = 0

    for idx, item in enumerate(data):
        if item.get("id") in completed_ids:
            completed += 1
            continue

        image_path = images_dir / item["image_path"]
        if not image_path.is_file():
            logger.warning("Image not found: %s", image_path)
            item["predict"] = ""
            item["result"] = 0
            errors += 1
            completed += 1
            continue

        result = run_inference(
            client=client,
            model=model,
            image_path=image_path,
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
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        # -- progress callback -----------------------------------------------
        if on_progress is not None and (completed % 50 == 0 or completed == total):
            on_progress(completed, total)

    # -- final save (ensure everything is written) ---------------------------
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # -- compute scores ------------------------------------------------------
    categories = _compute_scores(data)
    score = sum(categories.values())

    return OCRBenchResult(
        score=score,
        max_score=1000,
        categories=categories,
        errors=errors,
    )
