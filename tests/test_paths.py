"""Generated files respect PT_DATA_DIR and never land in tracked directories."""

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
