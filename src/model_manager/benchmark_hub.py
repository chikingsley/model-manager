"""Benchmark repository and SWE-bench workflows for the mm CLI."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SOURCES_FILE = REPO_ROOT / "benchmarks" / "sources.yaml"
SWEBENCH_RESULTS_DIR = REPO_ROOT / "benchmarks" / "swebench" / "results"


@dataclass(frozen=True)
class BenchmarkSource:
    """A benchmark repository tracked outside the model-manager tree."""

    name: str
    path: Path
    description: str


DEFAULT_BENCHMARK_SOURCES: tuple[BenchmarkSource, ...] = (
    BenchmarkSource(
        name="livebench",
        path=Path("/home/simon/github/livebench"),
        description="LiveBench benchmark suite",
    ),
    BenchmarkSource(
        name="multimodal-ocr",
        path=Path("/home/simon/github/MultimodalOCR"),
        description="Multimodal OCR benchmark suite",
    ),
    BenchmarkSource(
        name="swe-agent",
        path=Path("/home/simon/github/SWE-agent"),
        description="SWE-agent harness for SWE-bench Lite",
    ),
)


class RawSourceConfig(TypedDict, total=False):
    """Typed shape for benchmark source entries loaded from YAML."""

    path: str
    description: str


RepoSyncStatus = Literal[
    "up_to_date",
    "updated",
    "behind",
    "ahead",
    "missing",
    "not_git",
    "no_upstream",
    "error",
]


@dataclass(frozen=True)
class RepoSyncResult:
    """Result of checking or syncing one benchmark repository."""

    name: str
    path: Path
    status: RepoSyncStatus
    message: str


SwebenchBackend = Literal["ollama", "vllm", "llamacpp"]


class SwebenchBackendConfig(TypedDict):
    """Backend-specific SWE-bench configuration."""

    config: str
    default_model: str
    model_prefix: str


SWEBENCH_BACKENDS: dict[SwebenchBackend, SwebenchBackendConfig] = {
    "ollama": {
        "config": "local_ollama.yaml",
        "default_model": "qwen3:4b",
        "model_prefix": "ollama/",
    },
    "vllm": {
        "config": "local_vllm.yaml",
        "default_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "model_prefix": "openai/",
    },
    "llamacpp": {
        "config": "local_llamacpp.yaml",
        "default_model": "local",
        "model_prefix": "openai/",
    },
}


@dataclass(frozen=True)
class SwebenchRunResult:
    """Result metadata for a SWE-bench batch run."""

    backend: SwebenchBackend
    model: str
    output_dir: Path
    command: list[str]
    return_code: int


def _defaults_as_map() -> dict[str, BenchmarkSource]:
    return {source.name: source for source in DEFAULT_BENCHMARK_SOURCES}


def load_benchmark_sources(config_path: Path | None = None) -> dict[str, BenchmarkSource]:
    """Load benchmark source definitions from YAML with defaults as fallback."""
    path = config_path or BENCHMARK_SOURCES_FILE
    defaults = _defaults_as_map()

    if not path.exists():
        return defaults

    data = yaml.safe_load(path.read_text()) or {}
    raw_sources = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(raw_sources, dict):
        return defaults

    loaded: dict[str, BenchmarkSource] = {}
    for raw_name, raw_value in raw_sources.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, dict):
            continue

        entry = raw_value
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue

        description = entry.get("description", "")
        if not isinstance(description, str):
            description = ""

        source_path = Path(raw_path).expanduser()
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()

        loaded[raw_name] = BenchmarkSource(
            name=raw_name,
            path=source_path,
            description=description or "No description provided",
        )

    return loaded or defaults


def resolve_source_names(
    available: dict[str, BenchmarkSource],
    requested: list[str] | None,
) -> list[str]:
    """Resolve source names from CLI args and validate unknown names."""
    if not requested or "all" in requested:
        return sorted(available)

    unique: list[str] = []
    for source_name in requested:
        if source_name not in unique:
            unique.append(source_name)

    unknown = [source_name for source_name in unique if source_name not in available]
    if unknown:
        known = ", ".join(sorted(available))
        unknown_names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown source(s): {unknown_names}. Known: {known}")

    return unique


def _run_git(path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else ""


def _git_branch(path: Path) -> str:
    result = _run_git(path, ["rev-parse", "--abbrev-ref", "HEAD"])
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_dirty(path: Path) -> bool:
    result = _run_git(path, ["status", "--porcelain"])
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def _git_ahead_behind(path: Path) -> tuple[int, int] | None:
    upstream = _run_git(path, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
    if upstream.returncode != 0:
        return None

    upstream_ref = upstream.stdout.strip()
    if not upstream_ref:
        return None

    counts = _run_git(path, ["rev-list", "--left-right", "--count", f"HEAD...{upstream_ref}"])
    if counts.returncode != 0:
        return None

    parts = counts.stdout.strip().split()
    if len(parts) != 2:
        return None

    try:
        ahead = int(parts[0])
        behind = int(parts[1])
    except ValueError:
        return None

    return ahead, behind


def sync_benchmark_source(source: BenchmarkSource, check_only: bool = False) -> RepoSyncResult:
    """Check or sync a single benchmark source repository."""
    if not source.path.exists():
        return RepoSyncResult(
            name=source.name,
            path=source.path,
            status="missing",
            message="Path does not exist",
        )

    repo_check = _run_git(source.path, ["rev-parse", "--is-inside-work-tree"])
    if repo_check.returncode != 0 or repo_check.stdout.strip() != "true":
        return RepoSyncResult(
            name=source.name,
            path=source.path,
            status="not_git",
            message="Directory is not a git repository",
        )

    fetch = _run_git(source.path, ["fetch", "--prune", "--quiet"])
    if fetch.returncode != 0:
        return RepoSyncResult(
            name=source.name,
            path=source.path,
            status="error",
            message=f"git fetch failed: {_first_line(fetch.stderr) or 'unknown error'}",
        )

    branch = _git_branch(source.path)
    ahead_behind = _git_ahead_behind(source.path)
    dirty = _git_dirty(source.path)
    dirty_suffix = " (dirty tree)" if dirty else ""

    if ahead_behind is None:
        return RepoSyncResult(
            name=source.name,
            path=source.path,
            status="no_upstream",
            message=f"{branch}: no upstream tracking branch{dirty_suffix}",
        )

    ahead, behind = ahead_behind

    if check_only:
        if behind > 0:
            return RepoSyncResult(
                name=source.name,
                path=source.path,
                status="behind",
                message=f"{branch}: behind by {behind} commit(s){dirty_suffix}",
            )
        if ahead > 0:
            return RepoSyncResult(
                name=source.name,
                path=source.path,
                status="ahead",
                message=f"{branch}: ahead by {ahead} commit(s){dirty_suffix}",
            )
        return RepoSyncResult(
            name=source.name,
            path=source.path,
            status="up_to_date",
            message=f"{branch}: up to date{dirty_suffix}",
        )

    if behind > 0:
        pull = _run_git(source.path, ["pull", "--ff-only"])
        if pull.returncode != 0:
            pull_error = _first_line(pull.stderr) or _first_line(pull.stdout)
            return RepoSyncResult(
                name=source.name,
                path=source.path,
                status="error",
                message=f"{branch}: pull failed: {pull_error or 'unknown error'}",
            )

        return RepoSyncResult(
            name=source.name,
            path=source.path,
            status="updated",
            message=f"{branch}: fast-forwarded {behind} commit(s){dirty_suffix}",
        )

    if ahead > 0:
        return RepoSyncResult(
            name=source.name,
            path=source.path,
            status="ahead",
            message=f"{branch}: ahead by {ahead} commit(s){dirty_suffix}",
        )

    return RepoSyncResult(
        name=source.name,
        path=source.path,
        status="up_to_date",
        message=f"{branch}: already current{dirty_suffix}",
    )


def sync_benchmark_sources(
    available: dict[str, BenchmarkSource],
    selected: list[str],
    check_only: bool = False,
) -> list[RepoSyncResult]:
    """Check or sync multiple benchmark source repositories."""
    results: list[RepoSyncResult] = []
    for source_name in selected:
        result = sync_benchmark_source(available[source_name], check_only=check_only)
        results.append(result)
    return results


def format_command(command: list[str]) -> str:
    """Return a shell-safe, human-readable command string."""
    return " ".join(shlex.quote(part) for part in command)


def get_default_sweagent_dir(sources: dict[str, BenchmarkSource] | None = None) -> Path:
    """Resolve the default SWE-agent directory from benchmark sources."""
    active_sources = sources or load_benchmark_sources()
    swe_agent = active_sources.get("swe-agent")
    if swe_agent:
        return swe_agent.path
    return Path("/home/simon/github/SWE-agent")


def run_swebench_batch(
    backend: SwebenchBackend,
    model: str | None = None,
    limit: int = 1,
    sweagent_dir: Path | None = None,
    results_dir: Path | None = None,
) -> SwebenchRunResult:
    """Run SWE-bench Lite through SWE-agent for a chosen backend."""
    if limit < 1:
        raise ValueError("limit must be >= 1")

    backend_config = SWEBENCH_BACKENDS[backend]
    selected_model = model or backend_config["default_model"]
    model_arg = f"{backend_config['model_prefix']}{selected_model}"

    sweagent_home = (sweagent_dir or get_default_sweagent_dir()).expanduser()
    sweagent_bin = sweagent_home / ".venv" / "bin" / "sweagent"
    if not sweagent_bin.exists():
        raise FileNotFoundError(f"Missing SWE-agent binary: {sweagent_bin}")

    default_config = sweagent_home / "config" / "default.yaml"
    if not default_config.exists():
        raise FileNotFoundError(f"Missing SWE-agent config: {default_config}")

    local_config = REPO_ROOT / "benchmarks" / "swebench" / backend_config["config"]
    if not local_config.exists():
        raise FileNotFoundError(f"Missing local config: {local_config}")

    output_base = (results_dir or SWEBENCH_RESULTS_DIR).expanduser()
    output_base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"{backend}_{timestamp}"

    command = [
        str(sweagent_bin),
        "run-batch",
        "--config",
        str(default_config),
        "--config",
        str(local_config),
        "--agent.model.name",
        model_arg,
        "--instances.type",
        "swe_bench",
        "--instances.subset",
        "lite",
        "--instances.split",
        "test",
        "--instances.slice",
        f":{limit}",
        "--output_dir",
        str(output_dir),
    ]

    completed = subprocess.run(command, check=False, cwd=sweagent_home)

    return SwebenchRunResult(
        backend=backend,
        model=model_arg,
        output_dir=output_dir,
        command=command,
        return_code=completed.returncode,
    )
