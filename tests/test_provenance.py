"""A run is identified honestly, and never overwrites another."""

import hashlib
import shutil
from pathlib import Path

import pytest

from process_transfer.data import provenance
from process_transfer.data.paths import repository_root
from process_transfer.data.provenance import (
    copy_with_fingerprints,
    environment,
    file_fingerprint,
    git_state,
    new_run_directory,
)


def test_runs_never_share_a_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path))
    state = {"commit": "0123456789abcdef", "dirty": False}
    first = new_run_directory("m0_e03", state)
    second = new_run_directory("m0_e03", state)  # within the same second
    third = new_run_directory("m0_e03", state)
    assert len({first, second, third}) == 3
    for directory in (first, second, third):
        assert directory.is_dir()
        assert directory.parent == tmp_path / "experiments" / "m0_e03"
        assert "_0123456" in directory.name

    (first / "summary.json").write_text("{}", encoding="utf-8")
    assert not (second / "summary.json").exists()  # nothing is shared or overwritten


def test_the_run_id_says_when_the_code_is_not_identified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path))
    assert new_run_directory("x", {"commit": "abcdef0123", "dirty": True}).name.endswith(
        "_abcdef0-dirty"
    )
    assert "_nogit" in new_run_directory("x", {"commit": None, "dirty": None}).name


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_git_state_of_this_checkout() -> None:
    if not (repository_root() / ".git").exists():
        pytest.skip("not a git checkout")
    state = git_state()
    assert state["available"] is True
    assert isinstance(state["commit"], str) and len(state["commit"]) == 40
    assert int(state["commit"], 16) >= 0  # hexadecimal
    assert state["code_identified"] == (not state["dirty"])
    assert (state["diff_sha256"] is None) == (not state["dirty"])


def test_without_git_no_commit_is_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    state = git_state()
    assert state["available"] is False
    assert state["commit"] is None
    assert state["code_identified"] is False
    assert "not found" in str(state["reason"])


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_outside_a_checkout_no_commit_is_invented(tmp_path: Path) -> None:
    state = git_state(tmp_path)
    assert state["available"] is False
    assert state["commit"] is None
    assert state["code_identified"] is False
    assert "not a git checkout" in str(state["reason"])


def test_fingerprints_verify_a_copy_and_detect_a_change(tmp_path: Path, configs_dir: Path) -> None:
    sources = [configs_dir / "source_cstr.yaml", configs_dir / "target_cstr.yaml"]
    fingerprints = copy_with_fingerprints(sources, tmp_path / "configs")
    for source, fingerprint in zip(sources, fingerprints, strict=True):
        copy = tmp_path / "configs" / source.name
        assert copy.read_bytes() == source.read_bytes()
        assert fingerprint == file_fingerprint(source)
        assert fingerprint["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()

    altered = tmp_path / "configs" / "source_cstr.yaml"
    altered.write_bytes(altered.read_bytes().replace(b"0.005", b"0.006"))
    assert file_fingerprint(altered)["sha256"] != fingerprints[0]["sha256"]


def test_environment_records_the_versions_that_affect_the_numbers() -> None:
    found = environment()
    assert found["python"].count(".") == 2
    for name in ("numpy", "scipy", "pydantic"):
        assert found["packages"][name] is not None
