"""OmniDocBench v1.5 benchmark module.

Evaluates end-to-end document parsing: the model converts page images to
markdown, then the upstream eval script (pdf_validation.py) computes text
edit distance, table TEDS, and formula CDM.
"""

from __future__ import annotations

import json
import logging
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from openai import OpenAI

from ..inference import run_inference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT = """\
Convert this document page to markdown. Rules:
- Output ONLY the markdown, no explanations
- Use $...$ for inline math and $$...$$ for display math
- Use HTML <table> tags for tables with proper colspan/rowspan
- Preserve reading order"""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OmniDocBenchResult:
    """Aggregated results from the OmniDocBench evaluation."""

    overall: float = 0.0
    text_edit_dist: float = 0.0
    table_teds: float = 0.0
    formula_cdm: float = 0.0
    pages_processed: int = 0
    errors: int = 0
    raw_results: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pages(dataset_dir: Path, limit: int | None = None) -> list[dict]:
    """Load page entries from OmniDocBench.json."""
    json_path = dataset_dir / "OmniDocBench.json"
    if not json_path.exists():
        raise FileNotFoundError(f"OmniDocBench.json not found in {dataset_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    if limit is not None:
        pages = pages[:limit]

    return pages


def _image_path_for_page(dataset_dir: Path, page: dict) -> Path:
    """Resolve the image path for a given page entry."""
    rel_path = page["page_info"]["image_path"]
    img = dataset_dir / rel_path
    if not img.exists():
        # Try under images/ subdirectory as fallback
        img_alt = dataset_dir / "images" / Path(rel_path).name
        if img_alt.exists():
            return img_alt
    return img


def _write_eval_config(
    config_path: Path,
    dataset_dir: Path,
    predictions_dir: Path,
) -> None:
    """Write the YAML configuration file for pdf_validation.py."""
    config = {
        "dataset_path": str(dataset_dir / "OmniDocBench.json"),
        "predictions_dir": str(predictions_dir),
        "images_dir": str(dataset_dir),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def _parse_eval_results(result_dir: Path) -> dict:
    """Parse evaluation results from the result/ directory created by pdf_validation.py.

    Looks for JSON or YAML result files and extracts metrics.
    """
    results: dict = {}

    # Check common result file patterns
    for pattern in ["*.json", "*.yaml", "*.yml"]:
        for result_file in result_dir.glob(pattern):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    if result_file.suffix == ".json":
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
                if isinstance(data, dict):
                    results.update(data)
            except Exception as exc:
                logger.warning("Failed to parse result file %s: %s", result_file, exc)

    return results


def _extract_metric(raw: dict, keys: list[str], default: float = 0.0) -> float:
    """Try multiple key names to extract a metric from raw results."""
    for key in keys:
        # Check top-level
        if key in raw:
            val = raw[key]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict) and "score" in val:
                return float(val["score"])
        # Check nested under common containers
        for container in ["metrics", "scores", "overall"]:
            if container in raw and isinstance(raw[container], dict):
                if key in raw[container]:
                    return float(raw[container][key])
    return default


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    client: OpenAI,
    model: str,
    dataset_dir: Path,
    results_dir: Path,
    repo_dir: Path,
    limit: int | None = None,
    resume: bool = False,
    on_progress: callable = None,
) -> OmniDocBenchResult:
    """Run the OmniDocBench benchmark.

    Phase 1: Inference -- convert each page image to markdown via the model.
    Phase 2: Evaluation -- run upstream pdf_validation.py to score results.

    Args:
        client: OpenAI SDK client configured for the target endpoint.
        model: Model name/ID.
        dataset_dir: HF dataset directory containing OmniDocBench.json and images/.
        results_dir: Directory to write predictions and evaluation results.
        repo_dir: Cloned OmniDocBench repo (contains pdf_validation.py).
        limit: Optional cap on number of pages to process.
        resume: If True, skip pages that already have a non-empty .md file.
        on_progress: Optional callback(pages_done, total_pages).

    Returns:
        OmniDocBenchResult with scores and metadata.
    """
    dataset_dir = Path(dataset_dir)
    results_dir = Path(results_dir)
    repo_dir = Path(repo_dir)

    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1 — Inference
    # ------------------------------------------------------------------
    logger.info("Phase 1: Inference — loading pages from %s", dataset_dir)
    pages = _load_pages(dataset_dir, limit=limit)
    total = len(pages)
    logger.info("Processing %d pages (limit=%s, resume=%s)", total, limit, resume)

    pages_processed = 0
    errors = 0

    for i, page in enumerate(pages):
        image_path = _image_path_for_page(dataset_dir, page)
        image_stem = image_path.stem
        output_path = predictions_dir / f"{image_stem}.md"

        # Resume: skip if output already exists and is non-empty
        if resume and output_path.exists() and output_path.stat().st_size > 0:
            pages_processed += 1
            if on_progress and (pages_processed % 25 == 0 or pages_processed == total):
                on_progress(pages_processed, total)
            continue

        if not image_path.exists():
            logger.warning("Image not found: %s — skipping", image_path)
            errors += 1
            continue

        result = run_inference(
            client=client,
            model=model,
            image_path=image_path,
            prompt=_PROMPT,
            max_tokens=4096,
            temperature=0.0,
        )

        if result.error:
            logger.warning(
                "Inference error for %s: %s", image_path.name, result.error
            )
            errors += 1
            # Still write empty file so we can distinguish "tried and failed"
            # from "never attempted" on resume
            output_path.write_text("", encoding="utf-8")
        else:
            output_path.write_text(result.text, encoding="utf-8")

        pages_processed += 1

        # Progress callback every 25 pages
        if on_progress and (pages_processed % 25 == 0 or pages_processed == total):
            on_progress(pages_processed, total)

    logger.info(
        "Phase 1 complete: %d pages processed, %d errors", pages_processed, errors
    )

    # ------------------------------------------------------------------
    # Phase 2 — Evaluation
    # ------------------------------------------------------------------
    eval_script = repo_dir / "pdf_validation.py"
    if not eval_script.exists():
        warnings.warn(
            f"Evaluation script not found at {eval_script}. "
            "Returning inference-only results without scores. "
            "Clone the OmniDocBench repo to enable evaluation.",
            stacklevel=2,
        )
        return OmniDocBenchResult(
            pages_processed=pages_processed,
            errors=errors,
            raw_results={"warning": "eval script not found"},
        )

    logger.info("Phase 2: Evaluation — running pdf_validation.py")

    config_path = results_dir / "eval_config.yaml"
    _write_eval_config(config_path, dataset_dir, predictions_dir)

    eval_result_dir = results_dir / "result"
    eval_result_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use uv to run the eval script with essential dependencies.
        # The upstream requirements.txt has conflicting version pins,
        # so we install key packages individually instead.
        _EVAL_DEPS = [
            "beautifulsoup4", "lxml", "pylatexenc", "rapidfuzz",
            "python-Levenshtein", "apted", "nltk", "mmeval",
            "tqdm", "pandas", "numpy", "pyyaml", "Pillow",
            "scikit-learn", "scipy", "tabulate", "matplotlib",
            "opencv-python",
        ]
        cmd = ["uv", "run"]
        for dep in _EVAL_DEPS:
            cmd += ["--with", dep]
        cmd += [str(eval_script), "--config", str(config_path)]

        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            logger.error("pdf_validation.py failed (exit %d):\n%s", proc.returncode, proc.stderr)
            return OmniDocBenchResult(
                pages_processed=pages_processed,
                errors=errors,
                raw_results={
                    "eval_error": f"exit code {proc.returncode}",
                    "stderr": proc.stderr,
                    "stdout": proc.stdout,
                },
            )
        logger.info("pdf_validation.py completed successfully")
    except subprocess.TimeoutExpired:
        logger.error("pdf_validation.py timed out after 600s")
        return OmniDocBenchResult(
            pages_processed=pages_processed,
            errors=errors,
            raw_results={"eval_error": "timeout after 600s"},
        )
    except Exception as exc:
        logger.error("Failed to run pdf_validation.py: %s", exc)
        return OmniDocBenchResult(
            pages_processed=pages_processed,
            errors=errors,
            raw_results={"eval_error": str(exc)},
        )

    # Parse results
    raw_results = _parse_eval_results(eval_result_dir)

    text_edit_dist = _extract_metric(
        raw_results,
        ["text_edit_distance", "edit_dist", "edit_distance", "text_edit_dist"],
    )
    table_teds = _extract_metric(
        raw_results,
        ["table_teds", "teds", "table_TEDS", "TEDS"],
    )
    formula_cdm = _extract_metric(
        raw_results,
        ["formula_cdm", "cdm", "CDM", "formula_CDM"],
    )

    # Overall = ((1 - text_edit_dist) * 100 + table_teds + formula_cdm) / 3
    overall = ((1.0 - text_edit_dist) * 100.0 + table_teds + formula_cdm) / 3.0

    return OmniDocBenchResult(
        overall=overall,
        text_edit_dist=text_edit_dist,
        table_teds=table_teds,
        formula_cdm=formula_cdm,
        pages_processed=pages_processed,
        errors=errors,
        raw_results=raw_results,
    )
