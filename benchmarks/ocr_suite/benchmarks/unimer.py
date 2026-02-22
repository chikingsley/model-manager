"""UniMER-Test benchmark — formula recognition evaluation.

Evaluates a vision model's ability to convert images of mathematical
expressions into LaTeX.  Scoring uses normalised Levenshtein edit distance
(lower is better).

Categories
----------
spe  Simple Printed Expressions
cpe  Complex Printed Expressions
sce  Screen-Captured Expressions
hwe  Handwritten Expressions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openai import OpenAI
from rapidfuzz.distance import Levenshtein

from ..inference import run_inference

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

CATEGORIES: dict[str, str] = {
    "spe": "Simple Printed Expressions",
    "cpe": "Complex Printed Expressions",
    "sce": "Screen-Captured Expressions",
    "hwe": "Handwritten Expressions",
}

PROMPT = (
    "Convert this mathematical expression to LaTeX. "
    "Output ONLY the LaTeX code, nothing else."
)

PREDICTIONS_FILE = "unimer_predictions.json"
PROGRESS_INTERVAL = 200


@dataclass
class UniMERResult:
    """Aggregated benchmark result."""

    edit_distance: float  # normalised average, lower=better
    categories: dict[str, float] = field(default_factory=dict)
    total_samples: int = 0
    errors: int = 0


# ------------------------------------------------------------------
# Sample representation
# ------------------------------------------------------------------


@dataclass
class _Sample:
    id: str
    image_path: Path
    ground_truth: str
    category: str


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------


def _load_on_disk(dataset_dir: Path) -> list[_Sample]:
    """Load from UniMERNet repo layout.

    Expected structure:
        {dataset_dir}/{cat}/   — image files
        {dataset_dir}/{cat}.txt — one LaTeX label per line, matching
                                   sorted image filenames
    """
    samples: list[_Sample] = []
    for cat in CATEGORIES:
        img_dir = dataset_dir / cat
        label_file = dataset_dir / f"{cat}.txt"
        if not img_dir.is_dir() or not label_file.is_file():
            continue

        images = sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".tif"}
        )
        labels = label_file.read_text(encoding="utf-8").splitlines()

        if len(images) != len(labels):
            log.warning(
                "Category %s: %d images vs %d labels — using min",
                cat,
                len(images),
                len(labels),
            )

        for idx, (img, label) in enumerate(zip(images, labels)):
            samples.append(
                _Sample(
                    id=f"{cat}_{idx:05d}",
                    image_path=img,
                    ground_truth=label.strip(),
                    category=cat,
                )
            )

    return samples


def _load_huggingface(dataset_dir: Path) -> list[_Sample]:
    """Load from a HuggingFace ``datasets.load_from_disk`` directory.

    Expects columns ``image`` (PIL Image) and ``label`` (str).
    Images are saved to ``{dataset_dir}/images/sample_{i}.png``.
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_dir))

    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples: list[_Sample] = []
    for i, row in enumerate(ds):
        img_path = images_dir / f"sample_{i}.png"
        if not img_path.exists():
            row["image"].save(str(img_path))

        # Determine category from the row if available, else default to "spe"
        category = row.get("category", "spe")
        if category not in CATEGORIES:
            category = "spe"

        samples.append(
            _Sample(
                id=f"hf_{i:05d}",
                image_path=img_path,
                ground_truth=row["label"].strip(),
                category=category,
            )
        )

    return samples


def _load_samples(dataset_dir: Path) -> list[_Sample]:
    """Try on-disk layout first, then fall back to HuggingFace."""
    samples = _load_on_disk(dataset_dir)
    if samples:
        log.info("Loaded %d samples from on-disk layout", len(samples))
        return samples

    log.info("On-disk layout not found, trying HuggingFace load_from_disk")
    samples = _load_huggingface(dataset_dir)
    log.info("Loaded %d samples from HuggingFace dataset", len(samples))
    return samples


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


def _normalised_edit_distance(pred: str, gt: str) -> float:
    """Normalised Levenshtein distance (0 = perfect, 1 = worst)."""
    dist = Levenshtein.distance(pred, gt)
    return dist / max(len(pred), len(gt), 1)


def _strip_delimiters(text: str) -> str:
    """Remove surrounding ``$`` / ``$$`` delimiters from LaTeX."""
    text = text.strip()
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()
    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    return text


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------


def _load_predictions(path: Path) -> list[dict]:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def run(
    client: OpenAI,
    model: str,
    dataset_dir: Path,
    results_dir: Path,
    limit: int | None = None,
    resume: bool = False,
    on_progress: Callable | None = None,
) -> UniMERResult:
    """Run the UniMER-Test benchmark.

    Args:
        client: OpenAI-compatible client.
        model: Model name / ID.
        dataset_dir: Root directory containing UniMER-Test data.
        results_dir: Where to write prediction results.
        limit: Cap the number of samples evaluated (``None`` = all).
        resume: If ``True``, skip samples that already have predictions.
        on_progress: Optional callback ``(completed, total) -> None``,
                     invoked every ``PROGRESS_INTERVAL`` items.

    Returns:
        Aggregated :class:`UniMERResult`.
    """
    samples = _load_samples(dataset_dir)
    if not samples:
        raise RuntimeError(f"No samples found in {dataset_dir}")

    if limit is not None:
        samples = samples[:limit]

    predictions_path = results_dir / PREDICTIONS_FILE

    # Load existing predictions for resume support
    predictions: list[dict] = []
    completed_ids: set[str] = set()
    if resume:
        predictions = _load_predictions(predictions_path)
        completed_ids = {p["id"] for p in predictions}
        log.info("Resuming — %d existing predictions loaded", len(completed_ids))

    total = len(samples)
    errors = 0
    processed = len(completed_ids)

    for i, sample in enumerate(samples):
        if sample.id in completed_ids:
            continue

        result = run_inference(
            client=client,
            model=model,
            image_path=sample.image_path,
            prompt=PROMPT,
            max_tokens=1024,
        )

        if result.error is not None:
            log.warning("Inference error on %s: %s", sample.id, result.error)
            errors += 1
            pred_text = ""
        else:
            pred_text = _strip_delimiters(result.text)

        edit_dist = _normalised_edit_distance(pred_text, sample.ground_truth)

        predictions.append(
            {
                "id": sample.id,
                "ground_truth": sample.ground_truth,
                "category": sample.category,
                "predict": pred_text,
                "edit_dist": edit_dist,
            }
        )

        processed += 1

        # Incremental save + progress callback
        if processed % PROGRESS_INTERVAL == 0 or processed == total:
            _save_predictions(predictions_path, predictions)
            if on_progress is not None:
                on_progress(processed, total)
            log.info("Progress: %d / %d", processed, total)

    # Final save (in case total wasn't a multiple of PROGRESS_INTERVAL)
    _save_predictions(predictions_path, predictions)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    cat_dists: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}
    all_dists: list[float] = []

    for pred in predictions:
        d = pred["edit_dist"]
        all_dists.append(d)
        cat = pred.get("category")
        if cat in cat_dists:
            cat_dists[cat].append(d)

    avg_edit = sum(all_dists) / max(len(all_dists), 1)
    cat_avg = {
        cat: (sum(ds) / len(ds)) if ds else 0.0
        for cat, ds in cat_dists.items()
    }

    return UniMERResult(
        edit_distance=avg_edit,
        categories=cat_avg,
        total_samples=len(predictions),
        errors=errors,
    )
