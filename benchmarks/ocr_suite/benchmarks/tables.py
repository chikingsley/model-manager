"""PubTabNet / TEDS benchmark module.

Evaluates a vision model's ability to convert table images into HTML by
computing Tree Edit Distance-based Similarity (TEDS) scores against
ground-truth PubTabNet annotations.

Usage::

    from openai import OpenAI
    from pathlib import Path
    from benchmarks.tables import run

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    result = run(client, "my-model", Path("datasets/pubtabnet"), Path("results"))
    print(f"TEDS: {result.teds:.1f}%  TEDS-struct: {result.teds_struct:.1f}%")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable

from openai import OpenAI

from ..inference import run_inference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TABLE_PROMPT = "Table Recognition:"

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TablesResult:
    """Aggregated result from a PubTabNet / TEDS evaluation run."""

    teds: float
    teds_struct: float
    total_samples: int
    errors: int


# ---------------------------------------------------------------------------
# HTML reconstruction from PubTabNet annotations
# ---------------------------------------------------------------------------


def _reconstruct_html(sample: dict) -> str:
    """Reconstruct full HTML table from PubTabNet structure tokens + cells.

    The PubTabNet annotation stores:
    - ``html.structure.tokens``: a list of HTML tag strings that form the
      table skeleton (e.g. ``['<thead>', '<tr>', '<td>', '</td>', ...]``).
    - ``html.cell``: a list of cell dicts, each with a ``tokens`` list
      containing the textual content (character-level + HTML tag tokens).

    Cell content is inserted after each ``<td>`` or ``>`` token that opens
    a cell, following the original IBM reconstruction logic.
    """
    html_obj = sample["html"]
    structure_tokens: list[str] = list(html_obj["structure"]["tokens"])
    cells: list[dict] = html_obj.get("cell", html_obj.get("cells", []))

    # Find insertion points: positions of '<td>' or '>' that open a cell.
    # The convention from the original PubTabNet notebook is to look for
    # '<td>' and '>' tokens; cell content goes right after them.
    to_insert = [
        i for i, tag in enumerate(structure_tokens)
        if tag in ("<td>", ">")
    ]

    # Insert cell content in reverse so indices stay valid.
    for pos, cell in zip(to_insert[::-1], cells[::-1]):
        cell_tokens = cell.get("tokens", [])
        if cell_tokens:
            # Escape single-character tokens (plain text chars) but keep
            # multi-character tokens as-is (they are HTML tags like <b>).
            escaped = [
                escape(tok) if len(tok) == 1 else tok
                for tok in cell_tokens
            ]
            content = "".join(escaped)
            structure_tokens.insert(pos + 1, content)

    table_body = "".join(structure_tokens)
    # Wrap in minimal valid HTML that lxml/TEDS can parse.
    return f"<html><body><table>{table_body}</table></body></html>"


# ---------------------------------------------------------------------------
# Predicted HTML cleanup
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(
    r"<table[^>]*>.*?</table>",
    re.DOTALL | re.IGNORECASE,
)


def _markdown_table_to_html(md: str) -> str:
    """Convert a Markdown table to an HTML ``<table>`` string.

    Handles standard GFM pipe-delimited tables.  The separator line
    (``|---|---|``) is detected and skipped; the first row becomes
    ``<thead>`` and the rest ``<tbody>``.
    """
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]
    if not lines:
        return "<table></table>"

    def _parse_row(line: str) -> list[str]:
        # Remove leading/trailing pipes and split
        line = line.strip("|").strip()
        return [cell.strip() for cell in line.split("|")]

    # Detect separator line (e.g. |---|---|)
    sep_idx = None
    for i, line in enumerate(lines):
        stripped = line.replace("|", "").replace("-", "").replace(":", "").strip()
        if not stripped:
            sep_idx = i
            break

    rows_html: list[str] = []
    header_cells = _parse_row(lines[0])

    # Build header
    header_tds = "".join(f"<td>{escape(c)}</td>" for c in header_cells)
    thead = f"<thead><tr>{header_tds}</tr></thead>"

    # Body rows: skip the separator line
    body_start = (sep_idx + 1) if sep_idx is not None else 1
    body_rows = []
    for line in lines[body_start:]:
        # Skip any additional separator lines
        stripped = line.replace("|", "").replace("-", "").replace(":", "").strip()
        if not stripped:
            continue
        cells = _parse_row(line)
        tds = "".join(f"<td>{escape(c)}</td>" for c in cells)
        body_rows.append(f"<tr>{tds}</tr>")

    tbody = f"<tbody>{''.join(body_rows)}</tbody>" if body_rows else ""
    return f"<table>{thead}{tbody}</table>"


def _normalise_html_table(html: str) -> str:
    """Normalise an HTML table for fairer TEDS comparison.

    - Strips ``<head>`` block (with ``<style>``, ``<meta>``, etc.)
    - Converts ``<th>`` → ``<td>``
    - Strips all attributes from tags
    - Removes inline formatting wrappers
    - Collapses whitespace inside cell text
    """
    # Remove <head>...</head> entirely
    html = re.sub(r"<head>.*?</head>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Convert <th> to <td>
    html = re.sub(r"<th(\s|>)", r"<td\1", html, flags=re.IGNORECASE)
    html = re.sub(r"</th>", "</td>", html, flags=re.IGNORECASE)
    # Strip ALL attributes from tags (frame, rules, style, class, etc.)
    html = re.sub(r"(<\w+)\s+[^>]*(/?>)", r"\1\2", html)
    # Unwrap inline formatting tags
    for tag in ("b", "i", "strong", "em", "font", "sup", "sub", "span"):
        html = re.sub(rf"</?{tag}[^>]*>", "", html, flags=re.IGNORECASE)
    # Collapse whitespace (normalize newlines and spaces inside cells)
    html = re.sub(r"\s+", " ", html)
    # Remove spaces around tags
    html = re.sub(r"\s*(<[^>]+>)\s*", r"\1", html)
    return html


def _clean_predicted_html(raw: str) -> str:
    """Convert model output to normalised HTML for TEDS scoring.

    Handles both direct HTML output and Markdown table output.  Strips
    markdown fences, converts Markdown→HTML if needed, and normalises
    the result.
    """
    # Strip markdown code fences if present.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

    # Check if output contains an HTML table
    match = _TABLE_RE.search(cleaned)
    if match:
        table_html = match.group(0)
    elif "|" in cleaned and "\n" in cleaned:
        # Looks like a Markdown table — convert it
        table_html = _markdown_table_to_html(cleaned)
    else:
        table_html = f"<table>{cleaned}</table>"

    table_html = _normalise_html_table(table_html)
    return f"<html><body>{table_html}</body></html>"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset(dataset_dir: Path) -> list[dict]:
    """Load PubTabNet validation samples.

    Loads from a ``save_to_disk`` formatted dataset (apoidea/pubtabnet-html).

    Returns a list of sample dicts.
    """
    from datasets import load_from_disk  # type: ignore[import-untyped]

    dataset_dir = Path(dataset_dir)

    logger.info("Loading PubTabNet from %s", dataset_dir)
    ds = load_from_disk(str(dataset_dir))
    # If it's a DatasetDict, pick the val/validation split.
    if hasattr(ds, "keys"):
        for key in ("val", "validation"):
            if key in ds:
                ds = ds[key]
                break
        else:
            first_key = next(iter(ds.keys()))
            logger.warning("No val split found, using '%s'", first_key)
            ds = ds[first_key]
    logger.info("Loaded %d samples from disk", len(ds))
    return list(ds)


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


def _save_image(sample: dict, images_dir: Path, index: int) -> Path:
    """Save the sample image to disk and return the path.

    The HF dataset stores images under ``image`` (PIL) or ``png`` (PIL).
    We save as PNG to ``{images_dir}/{filename}.png``.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename.
    filename = sample.get("filename") or sample.get("__key__") or f"table_{index:06d}"
    if not filename.endswith(".png"):
        filename = f"{filename}.png"

    image_path = images_dir / filename
    if image_path.is_file():
        return image_path

    # Get PIL image from the sample.
    pil_image = sample.get("image") or sample.get("png")
    if pil_image is None:
        raise ValueError(f"Sample {index} has no 'image' or 'png' field")

    pil_image.save(str(image_path), format="PNG")
    return image_path


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
) -> TablesResult:
    """Run the PubTabNet / TEDS table recognition evaluation.

    Args:
        client: OpenAI SDK client pointed at the target endpoint.
        model: Model name/ID for chat completions.
        dataset_dir: Directory for the HuggingFace PubTabNet dataset
            (used with ``load_from_disk`` or as ``cache_dir``).
        results_dir: Where to write ``tables_predictions.json``.
        limit: If set, only evaluate the first *limit* samples.
        resume: If *True*, skip samples that already have predictions
            saved in the output file.
        on_progress: Optional callback invoked every 100 items (and on
            completion) with ``(completed: int, total: int)``.

    Returns:
        A :class:`TablesResult` with TEDS and TEDS-struct scores.
    """
    from table_recognition_metric import TEDS  # type: ignore[import-untyped]

    # -- setup ---------------------------------------------------------------
    dataset_dir = Path(dataset_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "tables_predictions.json"
    images_dir = dataset_dir / "images"

    # -- load dataset --------------------------------------------------------
    samples = _load_dataset(dataset_dir)
    if limit is not None:
        samples = samples[:limit]
    total = len(samples)
    logger.info("Evaluating %d PubTabNet samples", total)

    # -- resume handling -----------------------------------------------------
    predictions: list[dict] = []
    completed_indices: set[int] = set()
    if resume and output_path.is_file():
        with open(output_path, "r") as f:
            predictions = json.load(f)
        for pred in predictions:
            if "predicted_html" in pred:
                completed_indices.add(pred["index"])
        logger.info("Resuming: %d items already completed", len(completed_indices))

    # -- TEDS scorers --------------------------------------------------------
    teds_full = TEDS(structure_only=False)
    teds_struct = TEDS(structure_only=True)

    # -- inference loop ------------------------------------------------------
    errors = 0
    completed = len(completed_indices)
    teds_scores: list[float] = []
    teds_struct_scores: list[float] = []

    # Collect scores from already-completed predictions.
    for pred in predictions:
        if pred.get("teds_score") is not None:
            teds_scores.append(pred["teds_score"])
        if pred.get("teds_struct_score") is not None:
            teds_struct_scores.append(pred["teds_struct_score"])

    for idx, sample in enumerate(samples):
        if idx in completed_indices:
            continue

        # -- save image to disk ----------------------------------------------
        try:
            image_path = _save_image(sample, images_dir, idx)
        except Exception as exc:
            logger.warning("Failed to save image for sample %d: %s", idx, exc)
            errors += 1
            completed += 1
            predictions.append({
                "index": idx,
                "error": str(exc),
            })
            continue

        # -- get ground truth HTML --------------------------------------------
        try:
            # Prefer pre-rendered html_table (apoidea/pubtabnet-html format)
            if "html_table" in sample and sample["html_table"]:
                gt_html = sample["html_table"]
                # Ensure it has the wrapper TEDS expects
                if "<html>" not in gt_html.lower():
                    gt_html = f"<html><body>{gt_html}</body></html>"
            else:
                gt_html = _reconstruct_html(sample)
        except Exception as exc:
            logger.warning("Failed to reconstruct GT HTML for sample %d: %s", idx, exc)
            errors += 1
            completed += 1
            predictions.append({
                "index": idx,
                "error": f"gt_reconstruction: {exc}",
            })
            continue

        # -- normalise ground-truth HTML for fair comparison ------------------
        gt_html = _normalise_html_table(gt_html)

        # -- run inference ---------------------------------------------------
        result = run_inference(
            client=client,
            model=model,
            image_path=image_path,
            prompt=TABLE_PROMPT,
            max_tokens=8192,
            temperature=0.01,
        )

        if result.error is not None:
            logger.warning("Inference error on sample %d: %s", idx, result.error)
            errors += 1
            completed += 1
            predictions.append({
                "index": idx,
                "predicted_html": "",
                "ground_truth_html": gt_html,
                "teds_score": 0.0,
                "teds_struct_score": 0.0,
                "error": result.error,
                "elapsed_s": result.elapsed_s,
            })
            teds_scores.append(0.0)
            teds_struct_scores.append(0.0)
            continue

        # -- clean and score -------------------------------------------------
        pred_html = _clean_predicted_html(result.text)

        try:
            score_full = teds_full(pred_html, gt_html)
        except Exception as exc:
            logger.warning("TEDS scoring error on sample %d: %s", idx, exc)
            score_full = 0.0

        try:
            score_struct = teds_struct(pred_html, gt_html)
        except Exception as exc:
            logger.warning("TEDS-struct scoring error on sample %d: %s", idx, exc)
            score_struct = 0.0

        teds_scores.append(score_full)
        teds_struct_scores.append(score_struct)

        completed += 1
        predictions.append({
            "index": idx,
            "predicted_html": pred_html,
            "ground_truth_html": gt_html,
            "teds_score": score_full,
            "teds_struct_score": score_struct,
            "elapsed_s": result.elapsed_s,
        })

        # -- incremental save ------------------------------------------------
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

        # -- progress callback -----------------------------------------------
        if on_progress is not None and (completed % 100 == 0 or completed == total):
            on_progress(completed, total)

    # -- final save ----------------------------------------------------------
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # -- aggregate scores (multiply by 100 for percentage) -------------------
    avg_teds = (sum(teds_scores) / len(teds_scores) * 100) if teds_scores else 0.0
    avg_teds_struct = (
        (sum(teds_struct_scores) / len(teds_struct_scores) * 100)
        if teds_struct_scores
        else 0.0
    )

    return TablesResult(
        teds=avg_teds,
        teds_struct=avg_teds_struct,
        total_samples=total,
        errors=errors,
    )
