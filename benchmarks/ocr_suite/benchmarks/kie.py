"""KIE (Key Information Extraction) benchmark module.

Runs two KIE benchmarks using direct inference + edit similarity scoring:

1. **Nanonets-KIE** — 987 receipt images from ``nanonets/key_information_extraction``
2. **Handwritten-Forms** — 89 death certificate images (validation split) from
   ``Rasi1610/DeathSe43_44_checkbox``

Usage::

    from openai import OpenAI
    from pathlib import Path
    from benchmarks.kie import run

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    result = run(client, "my-model", Path("datasets"), Path("results"))
    print(result.overall)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from datasets import load_from_disk
from openai import OpenAI
from rapidfuzz.distance import Levenshtein

from ..inference import run_inference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NANONETS_FIELDS: list[str] = [
    "date",
    "doc_no_receipt_no",
    "seller_address",
    "seller_gst_id",
    "seller_name",
    "seller_phone",
    "total_amount",
    "total_tax",
]

HANDWRITTEN_FIELDS: list[str] = [
    "name_of_deceased",
    "deceased_gender",
    "deceased_race",
    "deceased_status",
    "deceased_age",
    "birth_place",
    "place_of_death_county",
    "place_of_death_city",
    "State file #",
    "father_name",
    "mother_name",
]

# Mapping from ground-truth nested JSON paths to flat field names.
# ground_truth -> gt_parse -> {person, person_data, relation}
_HANDWRITTEN_FIELD_MAP: dict[str, tuple[str, str]] = {
    "name_of_deceased": ("person", "name"),
    "place_of_death_county": ("person", "county"),
    "place_of_death_city": ("person", "city"),
    "State file #": ("person", "State file #"),
    "deceased_gender": ("person_data", "Gender"),
    "deceased_race": ("person_data", "Race"),
    "deceased_status": ("person_data", "status"),
    "deceased_age": ("person_data", "Age"),
    "birth_place": ("person_data", "birth_place"),
    "father_name": ("relation", "Father"),
    "mother_name": ("relation", "Mother"),
}

_PROMPT_TEMPLATE = (
    "Extract the following fields from this document image.\n"
    "Return a JSON object with these keys: {fields}\n"
    "If a field is not found, use an empty string.\n"
    "Output ONLY the JSON, no explanations."
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class KIEResult:
    """Aggregated result from the KIE benchmarks."""

    nanonets_kie_score: float
    handwritten_forms_score: float
    overall: float
    errors: int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_prompt(fields: list[str]) -> str:
    """Build the extraction prompt for a given set of fields."""
    return _PROMPT_TEMPLATE.format(fields=", ".join(fields))


def _parse_json_response(text: str) -> dict:
    """Parse a JSON object from model output, stripping markdown fences.

    Returns an empty dict on failure.
    """
    cleaned = text.strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        # Remove opening fence (with optional language tag)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].rstrip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _normalized_edit_similarity(pred: str, gt: str) -> float:
    """Compute normalized edit similarity between two strings.

    Returns ``1 - levenshtein(pred, gt) / max(len(pred), len(gt), 1)``.
    """
    dist = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt), 1)
    return 1.0 - dist / max_len


def _score_prediction(pred: dict, gt: dict, fields: list[str]) -> float:
    """Score a single prediction against ground truth.

    Returns the average normalized edit similarity across all fields.
    """
    if not fields:
        return 0.0

    total = 0.0
    for f in fields:
        pred_val = str(pred.get(f, "")).strip()
        gt_val = str(gt.get(f, "")).strip()
        total += _normalized_edit_similarity(pred_val, gt_val)

    return total / len(fields)


def _save_image(image, path: Path) -> None:
    """Save a PIL image to disk as JPEG, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert RGBA/P mode to RGB for JPEG compatibility
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")
    image.save(str(path), "JPEG")


def _load_predictions(path: Path) -> dict[int, dict]:
    """Load existing predictions from a JSON file.

    Returns a dict mapping sample index to the predicted fields dict.
    """
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return {entry["index"]: entry for entry in data}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _save_predictions(preds: dict[int, dict], path: Path) -> None:
    """Save predictions dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort by index for deterministic output
    entries = [preds[k] for k in sorted(preds)]
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Benchmark 1: Nanonets-KIE
# ---------------------------------------------------------------------------


def _flatten_nanonets_gt(sample: dict) -> dict[str, str]:
    """Extract ground truth fields from a Nanonets-KIE sample.

    Each field column contains the ground truth string directly.
    """
    gt: dict[str, str] = {}
    for field_name in NANONETS_FIELDS:
        val = sample.get(field_name)
        gt[field_name] = str(val).strip() if val is not None else ""
    return gt


def _run_nanonets(
    client: OpenAI,
    model: str,
    dataset_dir: Path,
    results_dir: Path,
    limit: int | None,
    resume: bool,
    on_progress: Callable | None,
) -> tuple[float, int]:
    """Run the Nanonets-KIE benchmark.

    Returns (score, error_count).
    """
    bench = "nanonets_kie"
    images_dir = results_dir / f"{bench}_images"
    preds_path = results_dir / f"{bench}_predictions.json"

    logger.info("Loading Nanonets-KIE dataset from %s", dataset_dir)
    ds = load_from_disk(str(dataset_dir))
    split = ds["test"]

    total = len(split) if limit is None else min(limit, len(split))
    prompt = _build_prompt(NANONETS_FIELDS)

    # Resume support
    existing_preds = _load_predictions(preds_path) if resume else {}

    errors = 0
    completed = 0
    score_sum = 0.0

    for i in range(total):
        sample = split[i]

        # Check resume
        if i in existing_preds and existing_preds[i].get("prediction") is not None:
            pred_fields = existing_preds[i]["prediction"]
            gt = _flatten_nanonets_gt(sample)
            score_sum += _score_prediction(pred_fields, gt, NANONETS_FIELDS)
            completed += 1
            if on_progress and (completed % 50 == 0 or completed == total):
                on_progress(completed, total)
            continue

        # Cache image to disk
        image = sample["image"]
        img_path = images_dir / f"sample_{i}.jpg"
        _save_image(image, img_path)

        # Run inference
        result = run_inference(
            client=client,
            model=model,
            image_path=img_path,
            prompt=prompt,
            max_tokens=512,
        )

        gt = _flatten_nanonets_gt(sample)

        if result.error is not None:
            logger.warning("Inference error on Nanonets-KIE sample %d: %s", i, result.error)
            pred_fields: dict = {}
            errors += 1
        else:
            pred_fields = _parse_json_response(result.text)

        sample_score = _score_prediction(pred_fields, gt, NANONETS_FIELDS)
        score_sum += sample_score

        # Save incrementally
        existing_preds[i] = {
            "index": i,
            "prediction": pred_fields,
            "ground_truth": gt,
            "score": sample_score,
            "raw_response": result.text,
            "error": result.error,
        }
        _save_predictions(existing_preds, preds_path)

        completed += 1
        if on_progress and (completed % 50 == 0 or completed == total):
            on_progress(completed, total)

    avg_score = score_sum / total if total > 0 else 0.0
    logger.info("Nanonets-KIE: score=%.4f, errors=%d, samples=%d", avg_score, errors, total)
    return avg_score, errors


# ---------------------------------------------------------------------------
# Benchmark 2: Handwritten-Forms
# ---------------------------------------------------------------------------


def _flatten_handwritten_gt(sample: dict) -> dict[str, str]:
    """Extract and flatten ground truth from a Handwritten-Forms sample.

    The ground truth is nested JSON in the ``ground_truth`` column::

        {"gt_parse": {"person": {...}, "person_data": {...}, "relation": {...}}}
    """
    gt_raw = sample.get("ground_truth", "")
    if isinstance(gt_raw, str):
        try:
            gt_parsed = json.loads(gt_raw)
        except (json.JSONDecodeError, ValueError):
            return dict.fromkeys(HANDWRITTEN_FIELDS, "")
    elif isinstance(gt_raw, dict):
        gt_parsed = gt_raw
    else:
        return dict.fromkeys(HANDWRITTEN_FIELDS, "")

    gt_parse = gt_parsed.get("gt_parse", {})

    flat: dict[str, str] = {}
    for field_name, (group, key) in _HANDWRITTEN_FIELD_MAP.items():
        val = gt_parse.get(group, {}).get(key, "")
        flat[field_name] = str(val).strip() if val is not None else ""

    return flat


def _run_handwritten(
    client: OpenAI,
    model: str,
    dataset_dir: Path,
    results_dir: Path,
    limit: int | None,
    resume: bool,
    on_progress: Callable | None,
) -> tuple[float, int]:
    """Run the Handwritten-Forms benchmark.

    Returns (score, error_count).
    """
    bench = "handwritten_forms"
    images_dir = results_dir / f"{bench}_images"
    preds_path = results_dir / f"{bench}_predictions.json"

    logger.info("Loading Handwritten-Forms dataset from %s", dataset_dir)
    ds = load_from_disk(str(dataset_dir))
    split = ds["validation"]

    total = len(split) if limit is None else min(limit, len(split))
    prompt = _build_prompt(HANDWRITTEN_FIELDS)

    # Resume support
    existing_preds = _load_predictions(preds_path) if resume else {}

    errors = 0
    completed = 0
    score_sum = 0.0

    for i in range(total):
        sample = split[i]

        # Check resume
        if i in existing_preds and existing_preds[i].get("prediction") is not None:
            pred_fields = existing_preds[i]["prediction"]
            gt = _flatten_handwritten_gt(sample)
            score_sum += _score_prediction(pred_fields, gt, HANDWRITTEN_FIELDS)
            completed += 1
            if on_progress and (completed % 20 == 0 or completed == total):
                on_progress(completed, total)
            continue

        # Cache image to disk
        image = sample["image"]
        img_path = images_dir / f"sample_{i}.jpg"
        _save_image(image, img_path)

        # Run inference
        result = run_inference(
            client=client,
            model=model,
            image_path=img_path,
            prompt=prompt,
            max_tokens=512,
        )

        gt = _flatten_handwritten_gt(sample)

        if result.error is not None:
            logger.warning("Inference error on Handwritten-Forms sample %d: %s", i, result.error)
            pred_fields: dict = {}
            errors += 1
        else:
            pred_fields = _parse_json_response(result.text)

        sample_score = _score_prediction(pred_fields, gt, HANDWRITTEN_FIELDS)
        score_sum += sample_score

        # Save incrementally
        existing_preds[i] = {
            "index": i,
            "prediction": pred_fields,
            "ground_truth": gt,
            "score": sample_score,
            "raw_response": result.text,
            "error": result.error,
        }
        _save_predictions(existing_preds, preds_path)

        completed += 1
        if on_progress and (completed % 20 == 0 or completed == total):
            on_progress(completed, total)

    avg_score = score_sum / total if total > 0 else 0.0
    logger.info("Handwritten-Forms: score=%.4f, errors=%d, samples=%d", avg_score, errors, total)
    return avg_score, errors


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
) -> KIEResult:
    """Run both KIE benchmarks and return combined results.

    Args:
        client: OpenAI SDK client pointed at the target endpoint.
        model: Model name/ID for chat completions.
        dataset_dir: Parent directory containing ``nanonets-kie/`` and
            ``handwritten-forms/`` subdirectories (HF datasets saved with
            ``save_to_disk``).
        results_dir: Where to write predictions, cached images, and results.
        limit: If set, cap the number of samples per benchmark.
        resume: If *True*, skip samples that already have predictions saved.
        on_progress: Optional callback invoked periodically with
            ``(completed: int, total: int)``.

    Returns:
        A :class:`KIEResult` with per-benchmark and overall scores.
    """
    dataset_dir = Path(dataset_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total_errors = 0
    scores: list[float] = []

    # -- Benchmark 1: Nanonets-KIE ------------------------------------------
    nanonets_dir = dataset_dir / "nanonets-kie"
    nanonets_score = 0.0
    if nanonets_dir.exists():
        logger.info("Running Nanonets-KIE benchmark")
        nanonets_score, errs = _run_nanonets(
            client=client,
            model=model,
            dataset_dir=nanonets_dir,
            results_dir=results_dir,
            limit=limit,
            resume=resume,
            on_progress=on_progress,
        )
        total_errors += errs
        scores.append(nanonets_score)
    else:
        logger.warning("Nanonets-KIE dataset not found at %s — skipping", nanonets_dir)

    # -- Benchmark 2: Handwritten-Forms -------------------------------------
    handwritten_dir = dataset_dir / "handwritten-forms"
    handwritten_score = 0.0
    if handwritten_dir.exists():
        logger.info("Running Handwritten-Forms benchmark")
        handwritten_score, errs = _run_handwritten(
            client=client,
            model=model,
            dataset_dir=handwritten_dir,
            results_dir=results_dir,
            limit=limit,
            resume=resume,
            on_progress=on_progress,
        )
        total_errors += errs
        scores.append(handwritten_score)
    else:
        logger.warning("Handwritten-Forms dataset not found at %s — skipping", handwritten_dir)

    # -- Overall score -------------------------------------------------------
    overall = sum(scores) / len(scores) if scores else 0.0

    return KIEResult(
        nanonets_kie_score=nanonets_score,
        handwritten_forms_score=handwritten_score,
        overall=overall,
        errors=total_errors,
    )
