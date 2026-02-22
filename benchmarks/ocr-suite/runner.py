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
"""OCR Benchmark Suite — Runner.

Orchestrates all OCR benchmarks: run, compare results, or set up datasets.

Usage:
    uv run benchmarks/ocr-suite/runner.py run --bench ocrbench
    uv run benchmarks/ocr-suite/runner.py run --bench all --limit 100
    uv run benchmarks/ocr-suite/runner.py compare
    uv run benchmarks/ocr-suite/runner.py setup
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

SUITE_DIR = Path(__file__).resolve().parent              # benchmarks/ocr-suite/
BENCHMARKS_DIR = SUITE_DIR.parent                        # benchmarks/
REPOS_DIR = BENCHMARKS_DIR / "repos"                     # benchmarks/repos/
DATASETS_DIR = BENCHMARKS_DIR / "datasets"               # benchmarks/datasets/
RESULTS_DIR = BENCHMARKS_DIR.parent / "results" / "ocr-suite"  # results/ocr-suite/

VALID_BENCHES = ("ocrbench", "omnidoc", "unimer", "tables", "kie")

logger = logging.getLogger("ocr-suite.runner")

# ---------------------------------------------------------------------------
# Auto-detect endpoint from model-manager state
# ---------------------------------------------------------------------------

_BACKEND_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "llama": "http://localhost:8090/v1",
    "chat": "http://localhost:8000/v1",
    "ocr": "http://localhost:8000/v1",
    "perf": "http://localhost:8000/v1",
}


def _detect_endpoint() -> tuple[str, str]:
    """Return (backend_name, base_url) from model-manager state.

    Imports StateManager from the model-manager src/ directory and reads
    the current active state to determine which backend is running.

    Raises RuntimeError if no backend is active.
    """
    src_dir = str(BENCHMARKS_DIR.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        from model_manager.state import StateManager
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import StateManager (looked in {src_dir}). "
            "Use --base-url to specify the endpoint manually."
        ) from exc

    sm = StateManager()
    active = sm.get_active()

    if active == "none":
        raise RuntimeError(
            "No active backend. Start one with 'mm ollama' (etc.) "
            "or pass --base-url explicitly."
        )

    url = _BACKEND_URLS.get(active)
    if url is None:
        raise RuntimeError(
            f"Active state '{active}' has no known URL mapping. "
            "Pass --base-url explicitly."
        )

    return active, url


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------


def _make_progress_callback(bench_name: str):
    """Return a progress callback that prints to stderr."""
    def _on_progress(completed: int, total: int) -> None:
        pct = completed / total * 100 if total else 0
        print(
            f"  [{bench_name}] {completed}/{total} ({pct:.0f}%)",
            file=sys.stderr,
            flush=True,
        )
    return _on_progress


# ---------------------------------------------------------------------------
# Individual benchmark runners
# ---------------------------------------------------------------------------


def _run_ocrbench(client, model, results_dir, limit, resume):
    """Run OCRBench and return a serialisable result dict."""
    from benchmarks.ocrbench import run as run_bench

    # Try dataset locations in priority order
    dataset_dir = None
    candidates = [
        DATASETS_DIR / "ocrbench",
        REPOS_DIR / "MultimodalOCR" / "OCRBench",
    ]
    for c in candidates:
        if c.is_dir():
            dataset_dir = c
            break

    if dataset_dir is None:
        raise FileNotFoundError(
            f"OCRBench dataset not found. Tried: {', '.join(str(c) for c in candidates)}"
        )

    result = run_bench(
        client=client,
        model=model,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        limit=limit,
        resume=resume,
        on_progress=_make_progress_callback("ocrbench"),
    )

    return {
        "score": result.score,
        "max": result.max_score,
        "categories": result.categories,
        "errors": result.errors,
    }


def _run_omnidoc(client, model, results_dir, limit, resume):
    """Run OmniDocBench and return a serialisable result dict."""
    from benchmarks.omnidocbench import run as run_bench

    dataset_dir = DATASETS_DIR / "omnidocbench"
    repo_dir = REPOS_DIR / "OmniDocBench"

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"OmniDocBench dataset not found at {dataset_dir}")

    result = run_bench(
        client=client,
        model=model,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        repo_dir=repo_dir,
        limit=limit,
        resume=resume,
        on_progress=_make_progress_callback("omnidoc"),
    )

    return {
        "overall": result.overall,
        "text_edit_dist": result.text_edit_dist,
        "table_teds": result.table_teds,
        "formula_cdm": result.formula_cdm,
        "pages_processed": result.pages_processed,
        "errors": result.errors,
    }


def _run_unimer(client, model, results_dir, limit, resume):
    """Run UniMER-Test and return a serialisable result dict."""
    from benchmarks.unimer import run as run_bench

    dataset_dir = None
    candidates = [
        REPOS_DIR / "UniMERNet" / "data" / "UniMER-Test",
        DATASETS_DIR / "unimer-test",
    ]
    for c in candidates:
        if c.is_dir():
            dataset_dir = c
            break

    if dataset_dir is None:
        raise FileNotFoundError(
            f"UniMER-Test dataset not found. Tried: {', '.join(str(c) for c in candidates)}"
        )

    result = run_bench(
        client=client,
        model=model,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        limit=limit,
        resume=resume,
        on_progress=_make_progress_callback("unimer"),
    )

    accuracy = (1.0 - result.edit_distance) * 100.0

    return {
        "edit_distance": result.edit_distance,
        "accuracy": accuracy,
        "categories": result.categories,
        "total_samples": result.total_samples,
        "errors": result.errors,
    }


def _run_tables(client, model, results_dir, limit, resume):
    """Run PubTabNet/TEDS and return a serialisable result dict."""
    from benchmarks.tables import run as run_bench

    dataset_dir = DATASETS_DIR / "pubtabnet"

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"PubTabNet dataset not found at {dataset_dir}")

    result = run_bench(
        client=client,
        model=model,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        limit=limit,
        resume=resume,
        on_progress=_make_progress_callback("tables"),
    )

    return {
        "teds": result.teds,
        "teds_struct": result.teds_struct,
        "total_samples": result.total_samples,
        "errors": result.errors,
    }


def _run_kie(client, model, results_dir, limit, resume):
    """Run KIE benchmarks and return a serialisable result dict."""
    from benchmarks.kie import run as run_bench

    # KIE looks for nanonets-kie/ and handwritten-forms/ under dataset_dir
    dataset_dir = DATASETS_DIR

    result = run_bench(
        client=client,
        model=model,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        limit=limit,
        resume=resume,
        on_progress=_make_progress_callback("kie"),
    )

    return {
        "nanonets_kie": result.nanonets_kie_score * 100.0,
        "handwritten_forms": result.handwritten_forms_score * 100.0,
        "overall": result.overall * 100.0,
        "errors": result.errors,
    }


# Registry of all benchmarks and their runner functions
_BENCH_RUNNERS = {
    "ocrbench": _run_ocrbench,
    "omnidoc": _run_omnidoc,
    "unimer": _run_unimer,
    "tables": _run_tables,
    "kie": _run_kie,
}


# ---------------------------------------------------------------------------
# `run` command
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    """Execute benchmarks."""
    from inference import detect_model

    # Resolve endpoint
    if args.base_url:
        base_url = args.base_url
    else:
        backend_name, base_url = _detect_endpoint()
        print(f"Auto-detected backend: {backend_name} -> {base_url}", file=sys.stderr)

    # Create client
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    # Detect model
    model = detect_model(client)
    print(f"Model: {model}", file=sys.stderr)

    # Create model slug (filesystem-safe)
    model_slug = model.replace("/", "_").replace(":", "_").replace(" ", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{model_slug}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Determine which benchmarks to run
    if args.bench == "all":
        benches = list(VALID_BENCHES)
    else:
        benches = [args.bench]

    # Ensure benchmarks/ package is importable
    if str(SUITE_DIR) not in sys.path:
        sys.path.insert(0, str(SUITE_DIR))

    # Run each benchmark
    results: dict[str, dict] = {}
    total_start = time.monotonic()

    for bench_name in benches:
        runner_fn = _BENCH_RUNNERS[bench_name]
        bench_results_dir = run_dir / bench_name
        bench_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Running: {bench_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        bench_start = time.monotonic()
        try:
            result = runner_fn(client, model, bench_results_dir, args.limit, args.resume)
            elapsed = time.monotonic() - bench_start
            result["elapsed_s"] = round(elapsed, 1)
            results[bench_name] = result
            print(
                f"  {bench_name} completed in {elapsed:.1f}s",
                file=sys.stderr,
            )
        except Exception as exc:
            elapsed = time.monotonic() - bench_start
            logger.error("Benchmark %s failed: %s", bench_name, exc, exc_info=True)
            results[bench_name] = {"error": str(exc), "elapsed_s": round(elapsed, 1)}
            print(
                f"  {bench_name} FAILED after {elapsed:.1f}s: {exc}",
                file=sys.stderr,
            )

    total_elapsed = time.monotonic() - total_start

    # Build combined results JSON
    combined = {
        "model": model,
        "base_url": base_url,
        "date": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        "elapsed_s": round(total_elapsed, 1),
        "benchmarks": results,
    }

    # Save results.json
    results_path = run_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}", file=sys.stderr)

    # Print summary
    _print_summary(combined)


def _print_summary(data: dict) -> None:
    """Print a human-readable summary of a single run."""
    print(f"\n{'='*60}")
    print(f"OCR Benchmark Suite — Results Summary")
    print(f"{'='*60}")
    print(f"Model:    {data['model']}")
    print(f"Endpoint: {data['base_url']}")
    print(f"Date:     {data['date']}")
    if data.get("limit"):
        print(f"Limit:    {data['limit']} samples/bench")
    if data.get("elapsed_s"):
        print(f"Duration: {data['elapsed_s']:.0f}s")
    print(f"{'-'*60}")

    benchmarks = data.get("benchmarks", {})

    if "ocrbench" in benchmarks:
        b = benchmarks["ocrbench"]
        if "error" in b and "score" not in b:
            print(f"  OCRBench:    ERROR — {b['error']}")
        else:
            print(f"  OCRBench:    {b['score']}/{b['max']}")

    if "omnidoc" in benchmarks:
        b = benchmarks["omnidoc"]
        if "error" in b and "overall" not in b:
            print(f"  OmniDoc:     ERROR — {b['error']}")
        else:
            print(f"  OmniDoc:     {b['overall']:.1f} (text={b['text_edit_dist']:.3f}, table={b['table_teds']:.1f}, formula={b['formula_cdm']:.1f})")

    if "unimer" in benchmarks:
        b = benchmarks["unimer"]
        if "error" in b and "accuracy" not in b:
            print(f"  UniMER:      ERROR — {b['error']}")
        else:
            print(f"  UniMER:      {b['accuracy']:.1f}% (edit_dist={b['edit_distance']:.4f})")

    if "tables" in benchmarks:
        b = benchmarks["tables"]
        if "error" in b and "teds" not in b:
            print(f"  TEDS:        ERROR — {b['error']}")
        else:
            print(f"  TEDS:        {b['teds']:.1f}% (struct={b['teds_struct']:.1f}%)")

    if "kie" in benchmarks:
        b = benchmarks["kie"]
        if "error" in b and "overall" not in b:
            print(f"  KIE:         ERROR — {b['error']}")
        else:
            print(f"  KIE:         {b['overall']:.1f}% (nanonets={b['nanonets_kie']:.1f}%, handwritten={b['handwritten_forms']:.1f}%)")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# `compare` command
# ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare results across all saved runs."""
    if not RESULTS_DIR.is_dir():
        print(f"No results directory found at {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Collect all results.json files
    runs: list[dict] = []
    for results_json in sorted(RESULTS_DIR.glob("*/results.json")):
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_path"] = str(results_json.parent.name)
            runs.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", results_json, exc)

    if not runs:
        print("No results found. Run some benchmarks first.", file=sys.stderr)
        sys.exit(1)

    # Print comparison table
    print()
    print("=" * 80)
    print("OCR Benchmark Suite — Results Comparison")
    print("=" * 80)
    print()

    header = f"{'Model':<30s}  {'OCRBench':>8s}  {'OmniDoc':>8s}  {'UniMER':>8s}  {'TEDS':>8s}  {'KIE':>8s}"
    print(header)
    print("-" * 80)

    for run in runs:
        model = run.get("model", "unknown")
        # Truncate long model names
        if len(model) > 30:
            model = model[:27] + "..."

        benchmarks = run.get("benchmarks", {})

        # OCRBench
        ocr = benchmarks.get("ocrbench", {})
        if "score" in ocr:
            ocr_str = f"{ocr['score']}/{ocr['max']}"
        else:
            ocr_str = "-"

        # OmniDoc
        omnidoc = benchmarks.get("omnidoc", {})
        if "overall" in omnidoc:
            omnidoc_str = f"{omnidoc['overall']:.1f}"
        else:
            omnidoc_str = "-"

        # UniMER
        unimer = benchmarks.get("unimer", {})
        if "accuracy" in unimer:
            unimer_str = f"{unimer['accuracy']:.1f}%"
        else:
            unimer_str = "-"

        # TEDS
        tables = benchmarks.get("tables", {})
        if "teds" in tables:
            teds_str = f"{tables['teds']:.1f}%"
        else:
            teds_str = "-"

        # KIE
        kie = benchmarks.get("kie", {})
        if "overall" in kie:
            kie_str = f"{kie['overall']:.1f}%"
        else:
            kie_str = "-"

        print(f"{model:<30s}  {ocr_str:>8s}  {omnidoc_str:>8s}  {unimer_str:>8s}  {teds_str:>8s}  {kie_str:>8s}")

    print()

    # Show limit info if any runs were limited
    limited_runs = [r for r in runs if r.get("limit")]
    if limited_runs:
        print("Note: runs with sample limits:")
        for r in limited_runs:
            print(f"  {r.get('model', '?')}: limit={r['limit']}")
        print()


# ---------------------------------------------------------------------------
# `setup` command
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> None:
    """Delegate to setup_datasets.py."""
    setup_script = SUITE_DIR / "setup_datasets.py"
    if not setup_script.is_file():
        print(f"Setup script not found at {setup_script}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(setup_script)]
    # Forward any extra args (--repos-only, --datasets-only)
    if hasattr(args, "setup_args") and args.setup_args:
        cmd.extend(args.setup_args)

    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR Benchmark Suite — run, compare, or set up benchmarks",
        prog="runner.py",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run -----------------------------------------------------------------
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--bench",
        required=True,
        choices=[*VALID_BENCHES, "all"],
        help="Which benchmark(s) to run",
    )
    run_parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint URL (auto-detected if not given)",
    )
    run_parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the endpoint (default: EMPTY)",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples per benchmark",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed predictions",
    )
    run_parser.set_defaults(func=cmd_run)

    # -- compare -------------------------------------------------------------
    compare_parser = subparsers.add_parser("compare", help="Compare results across runs")
    compare_parser.set_defaults(func=cmd_compare)

    # -- setup ---------------------------------------------------------------
    setup_parser = subparsers.add_parser("setup", help="Set up datasets and repos")
    setup_parser.add_argument(
        "setup_args",
        nargs="*",
        help="Extra args forwarded to setup_datasets.py (e.g. --repos-only)",
    )
    setup_parser.set_defaults(func=cmd_setup)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
