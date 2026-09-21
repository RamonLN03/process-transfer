"""Generated files respect PT_DATA_DIR and never land in tracked directories."""

import shutil
import subprocess
from pathlib import Path

import pytest

from process_transfer.data.paths import data_dir, output_dir, repository_root


def test_repository_root_is_the_directory_with_pyproject() -> None:
    assert (repository_root() / "pyproject.toml").is_file()
    assert (repository_root() / "AGENTS.md").is_file()


def test_default_is_the_ignored_data_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PT_DATA_DIR", raising=False)
    assert data_dir() == repository_root() / "data"
    monkeypatch.setenv("PT_DATA_DIR", "")
    assert data_dir() == repository_root() / "data"


def test_a_relative_value_is_resolved_against_the_repository_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", "somewhere/else")
    assert data_dir() == repository_root() / "somewhere" / "else"


def test_an_absolute_value_is_used_as_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    created = output_dir("m0_e03", "figures")
    assert created == tmp_path / "m0_e03" / "figures"
    assert created.is_dir()


def test_data_dir_does_not_create_anything(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "not_created"
    monkeypatch.setenv("PT_DATA_DIR", str(target))
    assert data_dir() == target
    assert not target.exists()


def _is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path], cwd=repository_root(), check=False
    )
    return result.returncode == 0


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_generated_data_is_ignored_but_the_data_package_is_not() -> None:
    """Regression: an unanchored ``data/`` pattern once ignored the source package
    ``src/process_transfer/data`` together with the generated files."""
    if not (repository_root() / ".git").exists():
        pytest.skip("not a git checkout")
    assert _is_ignored("data/m0_e03/summary.json")
    assert _is_ignored("data/raw/source.parquet")
    assert not _is_ignored("src/process_transfer/data/paths.py")
    assert not _is_ignored("src/process_transfer/data/__init__.py")
