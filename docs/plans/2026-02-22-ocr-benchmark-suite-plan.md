# OCR Benchmark Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `mm benchmark ocr` command that runs 5 OCR benchmarks (OmniDocBench, OCRBench, UniMER-Test, PubTabNet/TEDS, KIE/docext) against any model served via model-manager endpoints.

**Architecture:** Thin wrapper over upstream eval toolkits. A shared inference module sends images to OpenAI-compatible endpoints. Each benchmark module adapts predictions to its upstream eval format. Results stored as JSON in `results/ocr-suite/`.

**Tech Stack:** Python 3.11+, httpx (async HTTP), openai (sync inference), datasets (HuggingFace downloads), table-recognition-metric (TEDS), rapidfuzz (edit distance), pyyaml.

---

### Task 1: Setup Script — Clone Repos + Download Datasets

**Files:**

- Create: `benchmarks/ocr-suite/setup_datasets.py`

**Step 1: Write the setup script**

This is a standalone PEP 723 script (like `eval_openai.py`) that clones repos and downloads datasets. It will be called by `mm benchmark ocr setup`.

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
# ]
# ///
"""Download OCR benchmark datasets and clone evaluation repos."""

import argparse
import subprocess
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent.parent
REPOS_DIR = BENCHMARKS_DIR / "repos"
DATASETS_DIR = BENCHMARKS_DIR / "datasets"

REPOS = {
    "MultimodalOCR": "https://github.com/Yuliang-Liu/MultimodalOCR.git",
    "OmniDocBench": "https://github.com/opendatalab/OmniDocBench.git",
    "UniMERNet": "https://github.com/opendatalab/UniMERNet.git",
    "docext": "https://github.com/NanoNets/docext.git",
}

HF_DATASETS = {
    "omnidocbench": "opendatalab/OmniDocBench",
    "ocrbench": "echo840/OCRBench",
    "unimer-test": "wanderkid/UniMER_Dataset",
    "pubtabnet": "ajimeno/PubTabNet",
    "nanonets-kie": "nanonets/key_information_extraction",
    "handwritten-forms": "Rasi1610/DeathSe43_44_checkbox",
}


def clone_repos():
    """Clone upstream evaluation repos."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        dest = REPOS_DIR / name
        if dest.exists():
            print(f"  [skip] {name} already cloned")
            continue
        print(f"  [clone] {name} ...")
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
        print(f"  [done] {name}")


def download_datasets():
    """Download HuggingFace datasets."""
    from huggingface_hub import snapshot_download

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, repo_id in HF_DATASETS.items():
        dest = DATASETS_DIR / name
        if dest.exists() and any(dest.iterdir()):
            print(f"  [skip] {name} already downloaded")
            continue
        print(f"  [download] {name} from {repo_id} ...")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest))
        print(f"  [done] {name}")


def main():
    parser = argparse.ArgumentParser(description="Setup OCR benchmark datasets")
    parser.add_argument("--repos-only", action="store_true", help="Only clone repos")
    parser.add_argument("--datasets-only", action="store_true", help="Only download datasets")
    args = parser.parse_args()

    print("=== OCR Benchmark Suite Setup ===\n")

    if not args.datasets_only:
        print("Cloning evaluation repos...")
        clone_repos()
        print()

    if not args.repos_only:
        print("Downloading HuggingFace datasets...")
        download_datasets()
        print()

    print("Setup complete!")
    # Print sizes
    for d in [REPOS_DIR, DATASETS_DIR]:
        if d.exists():
            size = subprocess.run(
                ["du", "-sh", str(d)], capture_output=True, text=True
            ).stdout.strip()
            print(f"  {size}")


if __name__ == "__main__":
    main()
```

**Step 2: Test that it runs**

Run: `cd /home/simon/docker/model-manager && uv run benchmarks/ocr-suite/setup_datasets.py --repos-only`
Expected: Clones 4 repos into `benchmarks/repos/`

**Step 3: Fix the broken OCRBench symlinks**

The existing symlinks in `benchmarks/ocrbench/` point to `../repos/MultimodalOCR/...` which will now be valid once repos are cloned.

**Step 4: Commit**

```bash
git add benchmarks/ocr-suite/setup_datasets.py
git commit -m "feat(ocr-suite): add dataset setup script"
```

---

### Task 2: Shared Inference Module

**Files:**

- Create: `benchmarks/ocr-suite/__init__.py`
- Create: `benchmarks/ocr-suite/inference.py`

**Step 1: Write the inference module**

```python
# benchmarks/ocr-suite/__init__.py
# (empty)
```

```python
# benchmarks/ocr-suite/inference.py
"""Shared inference for OCR benchmarks — sends images to OpenAI-compatible endpoints."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

MIME_TYPES = {
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
    text: str
    elapsed_s: float
    error: str | None = None


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def detect_model(client: OpenAI) -> str:
    """Auto-detect model name from endpoint."""
    models = client.models.list()
    return models.data[0].id


def run_inference(
    client: OpenAI,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0,
) -> InferenceResult:
    """Send image + prompt to OpenAI-compatible endpoint, return response text."""
    mime = MIME_TYPES.get(image_path.suffix.lower(), "image/png")
    b64 = encode_image_base64(image_path)

    start = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content.strip()
        return InferenceResult(text=text, elapsed_s=time.monotonic() - start)
    except Exception as e:
        return InferenceResult(text="", elapsed_s=time.monotonic() - start, error=str(e))
```

**Step 2: Commit**

```bash
git add benchmarks/ocr-suite/__init__.py benchmarks/ocr-suite/inference.py
git commit -m "feat(ocr-suite): add shared inference module"
```

---

### Task 3: OCRBench Benchmark Module

Wraps the existing `eval_openai.py` logic into the benchmark module interface.

**Files:**

- Create: `benchmarks/ocr-suite/benchmarks/__init__.py`
- Create: `benchmarks/ocr-suite/benchmarks/ocrbench.py`

**Step 1: Write the benchmark module**

```python
# benchmarks/ocr-suite/benchmarks/__init__.py
# (empty)
```

```python
# benchmarks/ocr-suite/benchmarks/ocrbench.py
"""OCRBench v1 — 1000 QA pairs testing text recognition, VQA, KIE, math."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from ..inference import InferenceResult, run_inference

BENCHMARK_FILE = "OCRBench/OCRBench.json"
IMAGES_DIR = "OCRBench/OCRBench_Images"

CATEGORIES = [
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


@dataclass
class OCRBenchResult:
    score: int
    max_score: int = 1000
    categories: dict[str, int] = field(default_factory=dict)
    errors: int = 0


def _evaluate_prediction(predict: str, answers: str | list, dataset_name: str) -> int:
    if dataset_name == "HME100k":
        predict_clean = predict.strip().replace("\n", " ").replace(" ", "")
        if isinstance(answers, list):
            for answer in answers:
                if answer.strip().replace("\n", " ").replace(" ", "") in predict_clean:
                    return 1
        elif answers.strip().replace("\n", " ").replace(" ", "") in predict_clean:
            return 1
    else:
        predict_clean = predict.lower().strip().replace("\n", " ")
        if isinstance(answers, list):
            for answer in answers:
                if answer.lower().strip().replace("\n", " ") in predict_clean:
                    return 1
        elif answers.lower().strip().replace("\n", " ") in predict_clean:
            return 1
    return 0


def run(
    client: OpenAI,
    model: str,
    dataset_dir: Path,
    results_dir: Path,
    limit: int | None = None,
    resume: bool = False,
    on_progress: callable = None,
) -> OCRBenchResult:
    """Run OCRBench evaluation."""
    bench_file = dataset_dir / BENCHMARK_FILE
    images_dir = dataset_dir / IMAGES_DIR
    output_file = results_dir / "ocrbench_predictions.json"

    with open(bench_file) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    # Resume support
    completed_ids: set = set()
    if resume and output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        for item in existing:
            if "predict" in item:
                completed_ids.add(item["id"])
                for d in data:
                    if d["id"] == item["id"]:
                        d["predict"] = item["predict"]
                        d["result"] = item.get("result", 0)
        if on_progress:
            on_progress(f"Resuming: {len(completed_ids)}/{len(data)} already done")

    errors = 0
    for i, item in enumerate(data):
        if item["id"] in completed_ids:
            continue

        image_path = images_dir / item["image_path"]
        if not image_path.exists():
            errors += 1
            continue

        result = run_inference(client, model, image_path, item["question"], max_tokens=512)
        if result.error:
            item["predict"] = ""
            item["result"] = 0
            errors += 1
        else:
            item["predict"] = result.text
            item["result"] = _evaluate_prediction(result.text, item["answers"], item["dataset_name"])

        # Save incrementally
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        if on_progress and (i + 1) % 50 == 0:
            done = len(completed_ids) + i + 1
            on_progress(f"OCRBench: {done}/{len(data)}")

    # Score
    scores = {cat: 0 for cat in CATEGORIES}
    for item in data:
        if "result" in item and item.get("type") in scores:
            scores[item["type"]] += item["result"]

    total = sum(scores.values())
    return OCRBenchResult(score=total, categories=scores, errors=errors)
```

**Step 2: Commit**

```bash
git add benchmarks/ocr-suite/benchmarks/
git commit -m "feat(ocr-suite): add OCRBench benchmark module"
```

---

### Task 4: OmniDocBench Benchmark Module

**Files:**

- Create: `benchmarks/ocr-suite/benchmarks/omnidocbench.py`

**Step 1: Write the module**

OmniDocBench expects one `.md` file per page image. The model is prompted to convert each page to markdown. Then the upstream `pdf_validation.py` computes text edit distance, table TEDS, and formula CDM.

```python
# benchmarks/ocr-suite/benchmarks/omnidocbench.py
"""OmniDocBench v1.5 — end-to-end document parsing (text + tables + formulas)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from ..inference import InferenceResult, run_inference

PROMPT = """Convert this document page to markdown. Rules:
- Output ONLY the markdown, no explanations
- Use $...$ for inline math and $$...$$ for display math
- Use HTML <table> tags for tables with proper colspan/rowspan
- Preserve reading order"""


@dataclass
class OmniDocBenchResult:
    overall: float = 0.0
    text_edit_dist: float = 0.0
    table_teds: float = 0.0
    formula_cdm: float = 0.0
    pages_processed: int = 0
    errors: int = 0
    raw_results: dict = field(default_factory=dict)


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
    """Run OmniDocBench evaluation."""
    gt_file = dataset_dir / "OmniDocBench.json"
    images_dir = dataset_dir / "images"
    preds_dir = results_dir / "omnidocbench_predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth to get image list
    with open(gt_file) as f:
        gt_data = json.load(f)

    if limit:
        gt_data = gt_data[:limit]

    # Phase 1: Inference — generate markdown for each page
    errors = 0
    for i, page in enumerate(gt_data):
        image_name = page["page_info"]["image_path"]
        image_path = images_dir / Path(image_name).name
        md_path = preds_dir / (Path(image_name).stem + ".md")

        if resume and md_path.exists() and md_path.stat().st_size > 0:
            continue

        if not image_path.exists():
            errors += 1
            continue

        result = run_inference(client, model, image_path, PROMPT, max_tokens=4096)
        if result.error:
            md_path.write_text("")
            errors += 1
        else:
            md_path.write_text(result.text)

        if on_progress and (i + 1) % 25 == 0:
            on_progress(f"OmniDocBench inference: {i + 1}/{len(gt_data)}")

    pages_done = sum(1 for p in preds_dir.glob("*.md") if p.stat().st_size > 0)
    if on_progress:
        on_progress(f"OmniDocBench inference complete: {pages_done} pages")

    # Phase 2: Evaluation — call upstream OmniDocBench eval
    eval_result = _run_upstream_eval(gt_file, preds_dir, repo_dir, on_progress)
    eval_result.pages_processed = pages_done
    eval_result.errors = errors
    return eval_result


def _run_upstream_eval(
    gt_file: Path, preds_dir: Path, repo_dir: Path, on_progress: callable = None
) -> OmniDocBenchResult:
    """Run OmniDocBench's pdf_validation.py to compute metrics."""
    eval_script = repo_dir / "pdf_validation.py"
    config_dir = repo_dir / "configs"

    if not eval_script.exists():
        if on_progress:
            on_progress("WARNING: OmniDocBench eval script not found, skipping metric computation")
        return OmniDocBenchResult()

    # Write a temporary config for this run
    config = {
        "task_name": "end2end_eval",
        "metrics": {
            "text_block": ["Edit_dist"],
            "display_formula": ["Edit_dist", "CDM_plain"],
            "table": ["TEDS", "Edit_dist"],
            "reading_order": ["Edit_dist"],
        },
        "dataset": {
            "name": "end2end_dataset",
            "ground_truth": {"data_path": str(gt_file)},
            "prediction": {"data_path": str(preds_dir)},
            "match_method": "quick_match",
        },
    }

    import yaml

    config_path = preds_dir.parent / "omnidocbench_eval_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    if on_progress:
        on_progress("Running OmniDocBench evaluation (this may take a while)...")

    try:
        proc = subprocess.run(
            [sys.executable, str(eval_script), "--config", str(config_path)],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            if on_progress:
                on_progress(f"OmniDocBench eval error: {proc.stderr[:500]}")
            return OmniDocBenchResult()
    except subprocess.TimeoutExpired:
        if on_progress:
            on_progress("OmniDocBench eval timed out after 10 minutes")
        return OmniDocBenchResult()

    # Parse results from the result/ directory created by OmniDocBench
    return _parse_results(repo_dir / "result")


def _parse_results(result_dir: Path) -> OmniDocBenchResult:
    """Parse OmniDocBench result JSON files."""
    result = OmniDocBenchResult()

    # Look for the metric_result.json files
    for f in result_dir.glob("*_metric_result.json"):
        with open(f) as fh:
            data = json.load(fh)
        # Extract key metrics
        if "text_block" in data:
            result.text_edit_dist = data["text_block"].get("Edit_dist", 0.0)
        if "table" in data:
            result.table_teds = data["table"].get("TEDS", 0.0)
        if "display_formula" in data:
            result.formula_cdm = data["display_formula"].get("CDM_plain", 0.0)

    # Overall = ((1 - text_edit_dist) * 100 + table_teds + formula_cdm) / 3
    text_score = (1 - result.text_edit_dist) * 100
    result.overall = (text_score + result.table_teds + result.formula_cdm) / 3

    return result
```

**Step 2: Commit**

```bash
git add benchmarks/ocr-suite/benchmarks/omnidocbench.py
git commit -m "feat(ocr-suite): add OmniDocBench benchmark module"
```

---

### Task 5: UniMER-Test Benchmark Module

**Files:**

- Create: `benchmarks/ocr-suite/benchmarks/unimer.py`

**Step 1: Write the module**

UniMER-Test evaluates formula recognition. Images of math formulas → model generates LaTeX → scored by normalized edit distance (and optionally CDM via Docker).

```python
# benchmarks/ocr-suite/benchmarks/unimer.py
"""UniMER-Test — formula recognition benchmark (23,757 samples across 4 categories)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from ..inference import run_inference

PROMPT = "Convert this mathematical expression to LaTeX. Output ONLY the LaTeX code, nothing else."

CATEGORIES = {
    "spe": "Simple Printed Expressions",
    "cpe": "Complex Printed Expressions",
    "sce": "Screen-Captured Expressions",
    "hwe": "Handwritten Expressions",
}


@dataclass
class UniMERResult:
    edit_distance: float = 0.0  # normalized, lower is better
    categories: dict[str, float] = field(default_factory=dict)
    total_samples: int = 0
    errors: int = 0


def _normalized_edit_distance(pred: str, gt: str) -> float:
    """Compute normalized Levenshtein edit distance."""
    from rapidfuzz.distance import Levenshtein

    if not pred and not gt:
        return 0.0
    dist = Levenshtein.distance(pred, gt)
    return dist / max(len(pred), len(gt), 1)


def _load_samples(dataset_dir: Path) -> list[dict]:
    """Load UniMER-Test samples from the on-disk layout."""
    samples = []

    for cat_key, cat_name in CATEGORIES.items():
        img_dir = dataset_dir / cat_key
        label_file = dataset_dir / f"{cat_key}.txt"

        if not img_dir.exists() or not label_file.exists():
            continue

        labels = label_file.read_text().strip().splitlines()
        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

        for img_path, label in zip(images, labels):
            samples.append({
                "id": f"{cat_key}/{img_path.name}",
                "image_path": img_path,
                "ground_truth": label.strip(),
                "category": cat_key,
            })

    return samples


def _load_from_hf(dataset_dir: Path) -> list[dict]:
    """Load from HuggingFace dataset download (parquet)."""
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(dataset_dir))
        samples = []
        for i, item in enumerate(ds):
            # Save image to temp file for inference
            img_path = dataset_dir / "images" / f"sample_{i}.png"
            if not img_path.exists():
                img_path.parent.mkdir(parents=True, exist_ok=True)
                item["image"].save(img_path)
            samples.append({
                "id": f"hf_{i}",
                "image_path": img_path,
                "ground_truth": item["label"],
                "category": "unknown",
            })
        return samples
    except Exception:
        return []


def run(
    client: OpenAI,
    model: str,
    dataset_dir: Path,
    results_dir: Path,
    limit: int | None = None,
    resume: bool = False,
    on_progress: callable = None,
) -> UniMERResult:
    """Run UniMER-Test evaluation."""
    output_file = results_dir / "unimer_predictions.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Try on-disk layout first (from UniMERNet repo), then HF
    samples = _load_samples(dataset_dir)
    if not samples:
        samples = _load_from_hf(dataset_dir)

    if limit:
        samples = samples[:limit]

    # Resume support
    completed: dict[str, dict] = {}
    if resume and output_file.exists():
        with open(output_file) as f:
            for item in json.load(f):
                if "predict" in item:
                    completed[item["id"]] = item
        if on_progress:
            on_progress(f"Resuming: {len(completed)}/{len(samples)} already done")

    errors = 0
    for i, sample in enumerate(samples):
        sid = sample["id"]
        if sid in completed:
            sample["predict"] = completed[sid]["predict"]
            sample["edit_dist"] = completed[sid].get("edit_dist", 0.0)
            continue

        result = run_inference(client, model, sample["image_path"], PROMPT, max_tokens=1024)
        if result.error:
            sample["predict"] = ""
            sample["edit_dist"] = 1.0
            errors += 1
        else:
            pred = result.text.strip().strip("$").strip()
            sample["predict"] = pred
            sample["edit_dist"] = _normalized_edit_distance(pred, sample["ground_truth"])

        # Save incrementally
        saveable = [
            {"id": s["id"], "ground_truth": s["ground_truth"], "category": s["category"],
             "predict": s.get("predict", ""), "edit_dist": s.get("edit_dist", 1.0)}
            for s in samples if "predict" in s
        ]
        with open(output_file, "w") as f:
            json.dump(saveable, f, indent=2)

        if on_progress and (i + 1) % 200 == 0:
            done = len(completed) + i + 1
            on_progress(f"UniMER: {done}/{len(samples)}")

    # Aggregate scores
    cat_scores: dict[str, list[float]] = {}
    all_dists = []
    for s in samples:
        if "edit_dist" in s:
            all_dists.append(s["edit_dist"])
            cat = s["category"]
            cat_scores.setdefault(cat, []).append(s["edit_dist"])

    avg_edit = sum(all_dists) / len(all_dists) if all_dists else 1.0
    cat_avgs = {k: sum(v) / len(v) for k, v in cat_scores.items()}

    return UniMERResult(
        edit_distance=avg_edit,
        categories=cat_avgs,
        total_samples=len(samples),
        errors=errors,
    )
```

**Step 2: Commit**

```bash
git add benchmarks/ocr-suite/benchmarks/unimer.py
git commit -m "feat(ocr-suite): add UniMER-Test formula benchmark module"
```

---

### Task 6: PubTabNet/TEDS Benchmark Module

**Files:**

- Create: `benchmarks/ocr-suite/benchmarks/tables.py`

**Step 1: Write the module**

PubTabNet evaluation: table images → model generates HTML → scored by TEDS metric.

```python
# benchmarks/ocr-suite/benchmarks/tables.py
"""PubTabNet — table recognition benchmark scored with TEDS."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from ..inference import run_inference

PROMPT = """Convert this table image to HTML. Rules:
- Use <table>, <thead>, <tbody>, <tr>, <th>, <td> tags
- Include colspan and rowspan attributes where needed
- Output ONLY the HTML table, no explanations"""


@dataclass
class TablesResult:
    teds: float = 0.0
    teds_struct: float = 0.0
    total_samples: int = 0
    errors: int = 0


def _load_samples(dataset_dir: Path, limit: int | None = None) -> list[dict]:
    """Load PubTabNet validation samples from HF dataset download."""
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(dataset_dir))
        # Use validation split
        val = ds["val"] if "val" in ds else ds.get("validation", ds.get("test", None))
        if val is None:
            # Try loading directly
            from datasets import load_dataset
            val = load_dataset("ajimeno/PubTabNet", split="val")

        samples = []
        img_cache = dataset_dir / "images"
        img_cache.mkdir(parents=True, exist_ok=True)

        count = 0
        for item in val:
            if limit and count >= limit:
                break

            # Reconstruct HTML from structure tokens + cell tokens
            html_str = _reconstruct_html(item["html"])
            if not html_str:
                continue

            # Save image
            img_path = img_cache / f"{item.get('filename', f'table_{count}')}"
            if not img_path.exists():
                item["image"].save(img_path)

            samples.append({
                "id": item.get("filename", f"table_{count}"),
                "image_path": img_path,
                "ground_truth_html": html_str,
            })
            count += 1

        return samples
    except Exception as e:
        print(f"Error loading PubTabNet: {e}")
        return []


def _reconstruct_html(html_data: dict) -> str:
    """Reconstruct HTML table from PubTabNet structure + cell tokens."""
    structure_tokens = html_data.get("structure", {}).get("tokens", [])
    cells = html_data.get("cells", html_data.get("cell", []))

    if not structure_tokens:
        return ""

    html_parts = []
    cell_idx = 0
    for token in structure_tokens:
        if token == "</td>" and cell_idx < len(cells):
            cell_tokens = cells[cell_idx].get("tokens", [])
            html_parts.append("".join(cell_tokens))
            html_parts.append("</td>")
            cell_idx += 1
        else:
            html_parts.append(token)

    return "<table>" + "".join(html_parts) + "</table>"


def run(
    client: OpenAI,
    model: str,
    dataset_dir: Path,
    results_dir: Path,
    limit: int | None = None,
    resume: bool = False,
    on_progress: callable = None,
) -> TablesResult:
    """Run PubTabNet/TEDS evaluation."""
    from table_recognition_metric import TEDS

    output_file = results_dir / "tables_predictions.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(dataset_dir, limit=limit)
    if not samples and on_progress:
        on_progress("WARNING: No PubTabNet samples loaded")
        return TablesResult()

    # Resume support
    completed: dict[str, dict] = {}
    if resume and output_file.exists():
        with open(output_file) as f:
            for item in json.load(f):
                if "predict" in item:
                    completed[item["id"]] = item
        if on_progress:
            on_progress(f"Resuming: {len(completed)}/{len(samples)} already done")

    errors = 0
    for i, sample in enumerate(samples):
        sid = sample["id"]
        if sid in completed:
            sample["predict"] = completed[sid]["predict"]
            continue

        result = run_inference(client, model, sample["image_path"], PROMPT, max_tokens=4096)
        if result.error:
            sample["predict"] = ""
            errors += 1
        else:
            sample["predict"] = result.text

        # Save incrementally
        saveable = [
            {"id": s["id"], "ground_truth_html": s["ground_truth_html"],
             "predict": s.get("predict", "")}
            for s in samples if "predict" in s
        ]
        with open(output_file, "w") as f:
            json.dump(saveable, f, indent=2)

        if on_progress and (i + 1) % 100 == 0:
            done = len(completed) + i + 1
            on_progress(f"PubTabNet: {done}/{len(samples)}")

    # Compute TEDS scores
    teds_full = TEDS(structure_only=False)
    teds_struct = TEDS(structure_only=True)

    full_scores = []
    struct_scores = []
    for s in samples:
        pred = s.get("predict", "")
        gt = s["ground_truth_html"]
        if not pred or not gt:
            full_scores.append(0.0)
            struct_scores.append(0.0)
            continue
        try:
            full_scores.append(teds_full(pred, gt))
            struct_scores.append(teds_struct(pred, gt))
        except Exception:
            full_scores.append(0.0)
            struct_scores.append(0.0)

    avg_teds = sum(full_scores) / len(full_scores) if full_scores else 0.0
    avg_struct = sum(struct_scores) / len(struct_scores) if struct_scores else 0.0

    return TablesResult(
        teds=round(avg_teds * 100, 2),
        teds_struct=round(avg_struct * 100, 2),
        total_samples=len(samples),
        errors=errors,
    )
```

**Step 2: Commit**

```bash
git add benchmarks/ocr-suite/benchmarks/tables.py
git commit -m "feat(ocr-suite): add PubTabNet/TEDS table benchmark module"
```

---

### Task 7: KIE Benchmark Module (docext wrapper)

**Files:**

- Create: `benchmarks/ocr-suite/benchmarks/kie.py`

**Step 1: Write the module**

Wraps the Nanonets docext toolkit for KIE evaluation. Uses HuggingFace datasets directly since docext handles prompt construction.

```python
# benchmarks/ocr-suite/benchmarks/kie.py
"""KIE benchmarks via Nanonets IDP Leaderboard — Nanonets-KIE + Handwritten-Forms."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from openai import OpenAI

from ..inference import run_inference

NANONETS_KIE_FIELDS = [
    "date", "doc_no_receipt_no", "seller_address", "seller_gst_id",
    "seller_name", "seller_phone", "total_amount", "total_tax",
]

HANDWRITTEN_FIELDS = [
    "name_of_deceased", "deceased_gender", "deceased_race", "deceased_status",
    "deceased_age", "birth_place", "place_of_death_county", "place_of_death_city",
    "State file #", "father_name", "mother_name",
]


@dataclass
class KIEResult:
    nanonets_kie_score: float = 0.0
    handwritten_forms_score: float = 0.0
    overall: float = 0.0
    errors: int = 0


def _normalized_edit_similarity(pred: str, gt: str) -> float:
    """1 - normalized_edit_distance. Higher is better."""
    from rapidfuzz.distance import Levenshtein

    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    dist = Levenshtein.distance(pred.strip(), gt.strip())
    return 1.0 - dist / max(len(pred.strip()), len(gt.strip()), 1)


def _make_extraction_prompt(fields: list[str]) -> str:
    fields_str = ", ".join(fields)
    return f"""Extract the following fields from this document image.
Return a JSON object with these keys: {fields_str}
If a field is not found, use an empty string.
Output ONLY the JSON, no explanations."""


def _parse_json_response(text: str) -> dict:
    """Best-effort parse JSON from model response."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _run_nanonets_kie(
    client: OpenAI, model: str, dataset_dir: Path, results_dir: Path,
    limit: int | None, resume: bool, on_progress: callable,
) -> float:
    """Run Nanonets-KIE benchmark."""
    from datasets import load_from_disk

    output_file = results_dir / "nanonets_kie_predictions.json"
    ds = load_from_disk(str(dataset_dir))
    samples = list(ds["test"] if "test" in ds else ds)
    if limit:
        samples = samples[:limit]

    completed: dict[int, dict] = {}
    if resume and output_file.exists():
        with open(output_file) as f:
            completed = {item["idx"]: item for item in json.load(f)}

    prompt = _make_extraction_prompt(NANONETS_KIE_FIELDS)
    all_scores = []

    img_cache = results_dir / "nanonets_kie_images"
    img_cache.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        if i in completed:
            all_scores.append(completed[i].get("score", 0.0))
            continue

        # Save image
        img_path = img_cache / f"sample_{i}.jpg"
        if not img_path.exists():
            sample["image"].save(img_path)

        result = run_inference(client, model, img_path, prompt, max_tokens=1024)
        pred = _parse_json_response(result.text) if not result.error else {}
        gt = sample.get("annotations", {})

        # Score: average edit similarity across fields
        field_scores = []
        for field_name in NANONETS_KIE_FIELDS:
            gt_val = str(gt.get(field_name, ""))
            pred_val = str(pred.get(field_name, ""))
            field_scores.append(_normalized_edit_similarity(pred_val, gt_val))

        score = sum(field_scores) / len(field_scores) if field_scores else 0.0
        all_scores.append(score)
        completed[i] = {"idx": i, "pred": pred, "score": score}

        # Save incrementally
        with open(output_file, "w") as f:
            json.dump(list(completed.values()), f, indent=2)

        if on_progress and (i + 1) % 50 == 0:
            on_progress(f"Nanonets-KIE: {i + 1}/{len(samples)}")

    return (sum(all_scores) / len(all_scores) * 100) if all_scores else 0.0


def _run_handwritten_forms(
    client: OpenAI, model: str, dataset_dir: Path, results_dir: Path,
    limit: int | None, resume: bool, on_progress: callable,
) -> float:
    """Run Handwritten-Forms benchmark."""
    from datasets import load_from_disk

    output_file = results_dir / "handwritten_forms_predictions.json"
    ds = load_from_disk(str(dataset_dir))
    # Use validation split (89 samples)
    samples = list(ds["validation"] if "validation" in ds else ds)
    if limit:
        samples = samples[:limit]

    completed: dict[int, dict] = {}
    if resume and output_file.exists():
        with open(output_file) as f:
            completed = {item["idx"]: item for item in json.load(f)}

    prompt = _make_extraction_prompt(HANDWRITTEN_FIELDS)
    all_scores = []

    img_cache = results_dir / "handwritten_forms_images"
    img_cache.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        if i in completed:
            all_scores.append(completed[i].get("score", 0.0))
            continue

        img_path = img_cache / f"sample_{i}.jpg"
        if not img_path.exists():
            sample["image"].save(img_path)

        result = run_inference(client, model, img_path, prompt, max_tokens=1024)
        pred = _parse_json_response(result.text) if not result.error else {}

        # Parse ground truth from nested JSON
        gt_raw = json.loads(sample.get("ground_truth", "{}")) if isinstance(sample.get("ground_truth"), str) else sample.get("ground_truth", {})
        gt_parse = gt_raw.get("gt_parse", {})
        gt = {}
        # Flatten nested structure
        person = gt_parse.get("person", {})
        person_data = gt_parse.get("person_data", {})
        relation = gt_parse.get("relation", {})
        gt["name_of_deceased"] = person.get("name", "")
        gt["deceased_gender"] = person_data.get("Gender", "")
        gt["deceased_race"] = person_data.get("Race", "")
        gt["deceased_status"] = person_data.get("status", "")
        gt["deceased_age"] = person_data.get("Age", "")
        gt["birth_place"] = person_data.get("birth_place", "")
        gt["place_of_death_county"] = person.get("county", "")
        gt["place_of_death_city"] = person.get("city", "")
        gt["State file #"] = person.get("State file #", "")
        gt["father_name"] = relation.get("Father", "")
        gt["mother_name"] = relation.get("Mother", "")

        field_scores = []
        for field_name in HANDWRITTEN_FIELDS:
            gt_val = str(gt.get(field_name, ""))
            pred_val = str(pred.get(field_name, ""))
            field_scores.append(_normalized_edit_similarity(pred_val, gt_val))

        score = sum(field_scores) / len(field_scores) if field_scores else 0.0
        all_scores.append(score)
        completed[i] = {"idx": i, "pred": pred, "score": score}

        with open(output_file, "w") as f:
            json.dump(list(completed.values()), f, indent=2)

        if on_progress and (i + 1) % 20 == 0:
            on_progress(f"Handwritten-Forms: {i + 1}/{len(samples)}")

    return (sum(all_scores) / len(all_scores) * 100) if all_scores else 0.0


def run(
    client: OpenAI,
    model: str,
    dataset_dir: Path,  # parent dir containing nanonets-kie/ and handwritten-forms/
    results_dir: Path,
    limit: int | None = None,
    resume: bool = False,
    on_progress: callable = None,
) -> KIEResult:
    """Run KIE benchmarks."""
    errors = 0

    nanonets_dir = dataset_dir / "nanonets-kie"
    handwritten_dir = dataset_dir / "handwritten-forms"

    kie_score = 0.0
    hw_score = 0.0

    if nanonets_dir.exists():
        kie_score = _run_nanonets_kie(client, model, nanonets_dir, results_dir, limit, resume, on_progress)
    elif on_progress:
        on_progress("WARNING: Nanonets-KIE dataset not found, skipping")

    if handwritten_dir.exists():
        hw_score = _run_handwritten_forms(client, model, handwritten_dir, results_dir, limit, resume, on_progress)
    elif on_progress:
        on_progress("WARNING: Handwritten-Forms dataset not found, skipping")

    overall = (kie_score + hw_score) / 2 if kie_score and hw_score else kie_score or hw_score

    return KIEResult(
        nanonets_kie_score=round(kie_score, 2),
        handwritten_forms_score=round(hw_score, 2),
        overall=round(overall, 2),
        errors=errors,
    )
```

**Step 2: Commit**

```bash
git add benchmarks/ocr-suite/benchmarks/kie.py
git commit -m "feat(ocr-suite): add KIE benchmark module (Nanonets-KIE + Handwritten-Forms)"
```

---

### Task 8: Runner — Orchestrates All Benchmarks

**Files:**

- Create: `benchmarks/ocr-suite/runner.py`

**Step 1: Write the runner**

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
#     "pyyaml",
#     "rapidfuzz",
#     "table-recognition-metric",
#     "datasets",
#     "huggingface-hub",
# ]
# ///
"""OCR Benchmark Suite runner — orchestrates all benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# Resolve paths
SUITE_DIR = Path(__file__).parent
BENCHMARKS_DIR = SUITE_DIR.parent
REPOS_DIR = BENCHMARKS_DIR / "repos"
DATASETS_DIR = BENCHMARKS_DIR / "datasets"
RESULTS_DIR = BENCHMARKS_DIR.parent / "results" / "ocr-suite"

ALL_BENCHMARKS = ["ocrbench", "omnidoc", "unimer", "tables", "kie"]


def _detect_endpoint() -> tuple[str, str]:
    """Auto-detect active model endpoint from model-manager state."""
    try:
        sys.path.insert(0, str(BENCHMARKS_DIR.parent / "src"))
        from model_manager.state import StateManager

        state = StateManager()
        active = state.get_active()

        endpoints = {
            "ollama": "http://localhost:11434/v1",
            "llama": "http://localhost:8090/v1",
            "chat": "http://localhost:8000/v1",
            "ocr": "http://localhost:8000/v1",
            "perf": "http://localhost:8000/v1",
            "voice": "http://localhost:18000/v1",
        }
        base_url = endpoints.get(active)
        if not base_url:
            print(f"ERROR: No model active (state={active}). Start one with mm ollama/chat/llama")
            sys.exit(1)
        return active, base_url
    except Exception:
        return "unknown", "http://localhost:11434/v1"


def run_suite(
    benchmarks: list[str],
    base_url: str,
    api_key: str = "EMPTY",
    limit: int | None = None,
    resume: bool = False,
) -> dict:
    """Run selected benchmarks and return combined results."""
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Auto-detect model
    from benchmarks.ocr_suite.inference import detect_model

    model = detect_model(client)
    print(f"Model: {model}")
    print(f"Endpoint: {base_url}")
    if limit:
        print(f"Sample limit: {limit}")
    print()

    # Create model-specific results dir
    model_slug = model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = RESULTS_DIR / f"{model_slug}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": model,
        "base_url": base_url,
        "date": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "benchmarks": {},
    }

    def on_progress(msg: str):
        print(f"  {msg}")

    # Run each benchmark
    if "ocrbench" in benchmarks:
        print("=" * 60)
        print("Running OCRBench...")
        print("=" * 60)
        from benchmarks.ocr_suite.benchmarks.ocrbench import run as run_ocrbench

        ocr_dir = DATASETS_DIR / "ocrbench"
        if not ocr_dir.exists():
            # Fall back to the repos dir
            ocr_dir = REPOS_DIR / "MultimodalOCR" / "OCRBench"

        r = run_ocrbench(client, model, ocr_dir, run_dir, limit=limit, resume=resume, on_progress=on_progress)
        results["benchmarks"]["ocrbench"] = {
            "score": r.score, "max": r.max_score,
            "categories": r.categories, "errors": r.errors,
        }
        print(f"  Score: {r.score}/{r.max_score}")
        print()

    if "omnidoc" in benchmarks:
        print("=" * 60)
        print("Running OmniDocBench v1.5...")
        print("=" * 60)
        from benchmarks.ocr_suite.benchmarks.omnidocbench import run as run_omnidoc

        r = run_omnidoc(
            client, model,
            dataset_dir=DATASETS_DIR / "omnidocbench",
            results_dir=run_dir,
            repo_dir=REPOS_DIR / "OmniDocBench",
            limit=limit, resume=resume, on_progress=on_progress,
        )
        results["benchmarks"]["omnidocbench"] = {
            "overall": r.overall, "text_edit_dist": r.text_edit_dist,
            "table_teds": r.table_teds, "formula_cdm": r.formula_cdm,
            "pages_processed": r.pages_processed, "errors": r.errors,
        }
        print(f"  Overall: {r.overall:.2f}")
        print()

    if "unimer" in benchmarks:
        print("=" * 60)
        print("Running UniMER-Test...")
        print("=" * 60)
        from benchmarks.ocr_suite.benchmarks.unimer import run as run_unimer

        # Try repo layout first, then HF dataset
        ds_dir = REPOS_DIR / "UniMERNet" / "data" / "UniMER-Test"
        if not ds_dir.exists():
            ds_dir = DATASETS_DIR / "unimer-test"

        r = run_unimer(client, model, ds_dir, run_dir, limit=limit, resume=resume, on_progress=on_progress)
        results["benchmarks"]["unimer"] = {
            "edit_distance": round(r.edit_distance, 4),
            "accuracy": round((1 - r.edit_distance) * 100, 2),
            "categories": r.categories,
            "total_samples": r.total_samples, "errors": r.errors,
        }
        print(f"  Edit Distance: {r.edit_distance:.4f} (Accuracy: {(1 - r.edit_distance) * 100:.2f}%)")
        print()

    if "tables" in benchmarks:
        print("=" * 60)
        print("Running PubTabNet/TEDS...")
        print("=" * 60)
        from benchmarks.ocr_suite.benchmarks.tables import run as run_tables

        r = run_tables(
            client, model,
            dataset_dir=DATASETS_DIR / "pubtabnet",
            results_dir=run_dir,
            limit=limit, resume=resume, on_progress=on_progress,
        )
        results["benchmarks"]["tables"] = {
            "teds": r.teds, "teds_struct": r.teds_struct,
            "total_samples": r.total_samples, "errors": r.errors,
        }
        print(f"  TEDS: {r.teds:.2f}%  TEDS-S: {r.teds_struct:.2f}%")
        print()

    if "kie" in benchmarks:
        print("=" * 60)
        print("Running KIE (Nanonets-KIE + Handwritten-Forms)...")
        print("=" * 60)
        from benchmarks.ocr_suite.benchmarks.kie import run as run_kie

        r = run_kie(
            client, model,
            dataset_dir=DATASETS_DIR,
            results_dir=run_dir,
            limit=limit, resume=resume, on_progress=on_progress,
        )
        results["benchmarks"]["kie"] = {
            "nanonets_kie": r.nanonets_kie_score,
            "handwritten_forms": r.handwritten_forms_score,
            "overall": r.overall, "errors": r.errors,
        }
        print(f"  Nanonets-KIE: {r.nanonets_kie_score:.2f}%  Handwritten-Forms: {r.handwritten_forms_score:.2f}%")
        print()

    # Save combined results
    results_file = run_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    return results


def compare_results():
    """Show comparison table of all OCR suite runs."""
    if not RESULTS_DIR.exists():
        print("No OCR suite results yet. Run 'mm benchmark ocr' first.")
        return

    runs = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        results_file = run_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                runs.append(json.load(f))

    if not runs:
        print("No results found.")
        return

    print()
    print("=" * 80)
    print("OCR Benchmark Suite — Results Comparison")
    print("=" * 80)
    print()

    header = f"{'Model':<30} {'OCRBench':>10} {'OmniDoc':>10} {'UniMER':>10} {'TEDS':>10} {'KIE':>10}"
    print(header)
    print("-" * 80)

    for r in runs:
        model = r["model"]
        if len(model) > 28:
            model = "..." + model[-25:]

        b = r.get("benchmarks", {})
        ocr = f"{b['ocrbench']['score']}/1000" if "ocrbench" in b else "—"
        omni = f"{b['omnidocbench']['overall']:.1f}" if "omnidocbench" in b else "—"
        uni = f"{b['unimer']['accuracy']:.1f}%" if "unimer" in b else "—"
        teds = f"{b['tables']['teds']:.1f}%" if "tables" in b else "—"
        kie = f"{b['kie']['overall']:.1f}%" if "kie" in b else "—"

        print(f"{model:<30} {ocr:>10} {omni:>10} {uni:>10} {teds:>10} {kie:>10}")

    print()


def main():
    parser = argparse.ArgumentParser(description="OCR Benchmark Suite")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--bench", type=str, default="all",
                           help=f"Benchmark to run: {', '.join(ALL_BENCHMARKS)}, or 'all'")
    run_parser.add_argument("--base-url", type=str, default=None, help="API base URL (auto-detected)")
    run_parser.add_argument("--api-key", type=str, default="EMPTY", help="API key")
    run_parser.add_argument("--limit", type=int, default=None, help="Max samples per benchmark")
    run_parser.add_argument("--resume", action="store_true", help="Resume from existing predictions")

    # Compare command
    subparsers.add_parser("compare", help="Compare results across models")

    # Setup command
    subparsers.add_parser("setup", help="Download datasets and clone repos")

    args = parser.parse_args()

    if args.command == "compare":
        compare_results()
        return

    if args.command == "setup":
        from benchmarks.ocr_suite.setup_datasets import main as setup_main
        setup_main()
        return

    if args.command == "run" or args.command is None:
        if not hasattr(args, "bench"):
            args.bench = "all"
        benchmarks = ALL_BENCHMARKS if args.bench == "all" else [args.bench]

        # Validate benchmark names
        for b in benchmarks:
            if b not in ALL_BENCHMARKS:
                print(f"Unknown benchmark: {b}")
                print(f"Available: {', '.join(ALL_BENCHMARKS)}")
                sys.exit(1)

        if args.base_url is None:
            _, args.base_url = _detect_endpoint()

        run_suite(benchmarks, args.base_url, args.api_key,
                  limit=getattr(args, "limit", None),
                  resume=getattr(args, "resume", False))


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add benchmarks/ocr-suite/runner.py
git commit -m "feat(ocr-suite): add main runner orchestrator"
```

---

### Task 9: Wire Into mm CLI

**Files:**

- Modify: `src/model_manager/cli.py`

**Step 1: Add the ocr subcommand to `_dispatch_benchmark`**

In `cli.py`, modify `_dispatch_benchmark()` to handle `ocr` subcommand, and update `show_help()`.

Changes to `_dispatch_benchmark`:

```python
def _dispatch_benchmark(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mm benchmark", add_help=True)
    subparsers = parser.add_subparsers(dest="subcommand")

    subparsers.add_parser("run")
    subparsers.add_parser("compare")

    # OCR suite
    ocr_parser = subparsers.add_parser("ocr")
    ocr_parser.add_argument("ocr_subcommand", nargs="?", default="run",
                            choices=["run", "compare", "setup"])
    ocr_parser.add_argument("--bench", type=str, default="all")
    ocr_parser.add_argument("--base-url", type=str, default=None)
    ocr_parser.add_argument("--api-key", type=str, default="EMPTY")
    ocr_parser.add_argument("--limit", type=int, default=None)
    ocr_parser.add_argument("--resume", action="store_true")

    parsed = parser.parse_args(args)

    if parsed.subcommand == "run":
        return asyncio.run(run_benchmark_current())

    if parsed.subcommand == "compare":
        show_benchmarks()
        return 0

    if parsed.subcommand == "ocr":
        return _dispatch_ocr_benchmark(parsed)

    parser.print_help()
    return 1
```

Add new function:

```python
def _dispatch_ocr_benchmark(args) -> int:
    """Dispatch OCR benchmark suite commands."""
    import subprocess

    runner = Path(__file__).parent.parent.parent / "benchmarks" / "ocr-suite" / "runner.py"

    cmd = ["uv", "run", str(runner)]

    if args.ocr_subcommand == "setup":
        setup_script = runner.parent / "setup_datasets.py"
        cmd = ["uv", "run", str(setup_script)]
    elif args.ocr_subcommand == "compare":
        cmd.extend(["compare"])
    else:
        cmd.extend(["run", "--bench", args.bench])
        if args.base_url:
            cmd.extend(["--base-url", args.base_url])
        if args.api_key != "EMPTY":
            cmd.extend(["--api-key", args.api_key])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.resume:
            cmd.append("--resume")

    return subprocess.run(cmd).returncode
```

Add to `show_help()`:

```python
    print()
    print("OCR Benchmark Suite:")
    print("  mm benchmark ocr                  Run all OCR benchmarks against active model")
    print("  mm benchmark ocr --bench NAME     Run specific benchmark (ocrbench/omnidoc/unimer/tables/kie)")
    print("  mm benchmark ocr --limit N        Limit samples per benchmark")
    print("  mm benchmark ocr --resume         Resume from existing predictions")
    print("  mm benchmark ocr compare          Compare OCR results across models")
    print("  mm benchmark ocr setup            Download datasets and clone repos")
```

**Step 2: Commit**

```bash
git add src/model_manager/cli.py
git commit -m "feat(ocr-suite): wire OCR benchmarks into mm CLI"
```

---

### Task 10: Update README and Benchmark Docs

**Files:**

- Modify: `benchmarks/README.md`

**Step 1: Add OCR suite section to the benchmarks README**

Add after the OCRBench section:

```markdown
## OCR Benchmark Suite

Comprehensive OCR evaluation covering 5 benchmarks:

| Benchmark | Tests | Samples |
|-----------|-------|---------|
| OCRBench v1 | Text recognition, VQA, KIE, math | 1,000 |
| OmniDocBench v1.5 | End-to-end doc parsing | 1,355 |
| UniMER-Test | Formula recognition | 23,757 |
| PubTabNet | Table recognition (TEDS) | 9,115 |
| KIE | Key info extraction | 1,432 |

### First-Time Setup

```bash
cd /home/simon/github/model-manager
mm benchmark ocr setup    # Clones repos + downloads HF datasets (~10GB)
```

### Running

```bash
# Run all benchmarks against active model
mm benchmark ocr

# Run specific benchmark
mm benchmark ocr --bench ocrbench
mm benchmark ocr --bench omnidoc
mm benchmark ocr --bench unimer
mm benchmark ocr --bench tables
mm benchmark ocr --bench kie

# Quick smoke test (5 samples per benchmark)
mm benchmark ocr --limit 5

# Resume interrupted run
mm benchmark ocr --resume

# Compare results across models
mm benchmark ocr compare
```

Results saved to `results/ocr-suite/{model}_{timestamp}/results.json`.

```text

**Step 2: Commit**

```bash
git add benchmarks/README.md
git commit -m "docs: add OCR benchmark suite to benchmarks README"
```

---

### Task 11: Smoke Test

**Step 1: Run setup to clone repos and download datasets**

```bash
cd /home/simon/docker/model-manager
mm benchmark ocr setup
```

Expected: Repos cloned to `benchmarks/repos/`, datasets downloaded to `benchmarks/datasets/`.

**Step 2: Run a quick smoke test with limit=2**

```bash
mm benchmark ocr --bench ocrbench --limit 2
```

Expected: Runs 2 OCRBench samples against the active model, prints scores, saves results.

**Step 3: Verify results file**

```bash
ls results/ocr-suite/
cat results/ocr-suite/*/results.json | python3 -m json.tool | head -20
```

**Step 4: Test compare command**

```bash
mm benchmark ocr compare
```

**Step 5: Commit any fixes needed**

```bash
git add -A
git commit -m "fix(ocr-suite): smoke test fixes"
```
