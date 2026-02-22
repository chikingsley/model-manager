# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
# ]
# ///
"""
Setup script for OCR benchmark suite — clones repos and downloads datasets.

Usage:
    uv run benchmarks/ocr-suite/setup_datasets.py              # everything
    uv run benchmarks/ocr-suite/setup_datasets.py --repos-only  # git repos only
    uv run benchmarks/ocr-suite/setup_datasets.py --datasets-only  # HF datasets only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
REPOS_DIR = BENCHMARKS_DIR / "repos"
DATASETS_DIR = BENCHMARKS_DIR / "datasets"

REPOS: dict[str, str] = {
    "MultimodalOCR": "https://github.com/Yuliang-Liu/MultimodalOCR.git",
    "OmniDocBench": "https://github.com/opendatalab/OmniDocBench.git",
    "UniMERNet": "https://github.com/opendatalab/UniMERNet.git",
    "docext": "https://github.com/NanoNets/docext.git",
}

# name -> HuggingFace dataset ID
DATASETS: dict[str, str] = {
    "omnidocbench": "opendatalab/OmniDocBench",
    "ocrbench": "echo840/OCRBench",
    "unimer-test": "wanderkid/UniMER_Dataset",
    "pubtabnet": "ajimeno/PubTabNet",
    "nanonets-kie": "nanonets/key_information_extraction",
    "handwritten-forms": "Rasi1610/DeathSe43_44_checkbox",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command, printing it first."""
    cmd = ["git", *args]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def dir_size_mb(path: Path) -> float:
    """Return total size of a directory tree in MiB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def fmt_size(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GiB"
    return f"{mb:.1f} MiB"


# ---------------------------------------------------------------------------
# Clone repos
# ---------------------------------------------------------------------------


def clone_repos() -> None:
    """Shallow-clone benchmark git repos into benchmarks/repos/."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("Cloning git repos into", REPOS_DIR)
    print(f"{'='*60}")

    for name, url in REPOS.items():
        dest = REPOS_DIR / name
        if dest.exists():
            print(f"\n[skip] {name} — already exists at {dest}")
            continue
        print(f"\n[clone] {name}")
        try:
            run_git(["clone", "--depth", "1", url, str(dest)])
            print(f"  -> OK ({fmt_size(dir_size_mb(dest))})")
        except subprocess.CalledProcessError as exc:
            print(f"  !! FAILED: {exc.stderr.strip()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Download HuggingFace datasets
# ---------------------------------------------------------------------------


def download_datasets() -> None:
    """Download HuggingFace datasets into benchmarks/datasets/."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("Downloading HuggingFace datasets into", DATASETS_DIR)
    print(f"{'='*60}")

    # Import here so --repos-only works without the datasets package
    from datasets import load_dataset

    for local_name, hf_id in DATASETS.items():
        dest = DATASETS_DIR / local_name
        if dest.exists() and any(dest.iterdir()):
            print(f"\n[skip] {local_name} — already exists at {dest}")
            continue
        print(f"\n[download] {local_name}  ({hf_id})")
        try:
            ds = load_dataset(hf_id, cache_dir=str(dest), trust_remote_code=True)
            print(f"  -> OK  splits: {list(ds.keys())}  ({fmt_size(dir_size_mb(dest))})")
        except Exception as exc:
            print(f"  !! FAILED: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Disk usage summary
# ---------------------------------------------------------------------------


def print_disk_usage() -> None:
    """Print disk usage for repos and datasets directories."""
    print(f"\n{'='*60}")
    print("Disk usage summary")
    print(f"{'='*60}")

    if REPOS_DIR.exists():
        total_repos = 0.0
        for name in sorted(REPOS):
            p = REPOS_DIR / name
            if p.exists():
                sz = dir_size_mb(p)
                total_repos += sz
                print(f"  repos/{name:<20s}  {fmt_size(sz):>10s}")
        print(f"  {'':20s}  {'─'*12}")
        print(f"  {'repos total':<20s}  {fmt_size(total_repos):>10s}")

    if DATASETS_DIR.exists():
        total_ds = 0.0
        for name in sorted(DATASETS):
            p = DATASETS_DIR / name
            if p.exists():
                sz = dir_size_mb(p)
                total_ds += sz
                print(f"  datasets/{name:<20s}  {fmt_size(sz):>10s}")
        print(f"  {'':20s}  {'─'*12}")
        print(f"  {'datasets total':<20s}  {fmt_size(total_ds):>10s}")

    grand = 0.0
    if REPOS_DIR.exists():
        grand += dir_size_mb(REPOS_DIR)
    if DATASETS_DIR.exists():
        grand += dir_size_mb(DATASETS_DIR)
    print(f"\n  TOTAL: {fmt_size(grand)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup OCR benchmark suite — clone repos & download datasets",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--repos-only",
        action="store_true",
        help="Only clone git repos (skip dataset downloads)",
    )
    group.add_argument(
        "--datasets-only",
        action="store_true",
        help="Only download HF datasets (skip git clones)",
    )
    args = parser.parse_args()

    if not args.datasets_only:
        clone_repos()
    if not args.repos_only:
        download_datasets()

    print_disk_usage()
    print("\nDone.")


if __name__ == "__main__":
    main()
