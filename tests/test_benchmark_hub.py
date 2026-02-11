from __future__ import annotations

from pathlib import Path

import pytest

from model_manager.benchmark_hub import (
    BenchmarkSource,
    format_command,
    load_benchmark_sources,
    resolve_source_names,
    sync_benchmark_source,
)


def test_load_benchmark_sources_uses_defaults_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing_sources.yaml"
    sources = load_benchmark_sources(missing)

    assert "swe-agent" in sources
    assert "livebench" in sources
    assert "multimodal-ocr" in sources


def test_load_benchmark_sources_resolves_relative_paths(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    target_dir = tmp_path / "tracked"
    target_dir.mkdir()

    config.write_text(
        """
source: ignored
sources:
  local:
    path: tracked
    description: Local benchmark repo
""".strip()
        + "\n",
        encoding="utf-8",
    )

    sources = load_benchmark_sources(config)

    assert "local" in sources
    assert sources["local"].path == target_dir.resolve()
    assert sources["local"].description == "Local benchmark repo"


def test_resolve_source_names_rejects_unknown() -> None:
    available = {
        "alpha": BenchmarkSource("alpha", Path("/tmp/alpha"), "a"),
        "beta": BenchmarkSource("beta", Path("/tmp/beta"), "b"),
    }

    with pytest.raises(ValueError, match="Unknown source"):
        resolve_source_names(available, ["alpha", "gamma"])


def test_sync_benchmark_source_reports_missing_path(tmp_path: Path) -> None:
    source = BenchmarkSource("missing", tmp_path / "does-not-exist", "none")
    result = sync_benchmark_source(source, check_only=True)

    assert result.status == "missing"


def test_format_command_quotes_special_characters() -> None:
    rendered = format_command(["python", "script.py", "--name", "model with spaces"])
    assert "'model with spaces'" in rendered
