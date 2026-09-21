"""A git query that fails is not an answer.

Regression tests for a confirmed defect, found in the review of ``325cb93``: ``git_state``
read the empty output of a ``git status`` that had failed as "nothing has changed", and
reported the code as identified by its commit.

git itself is replaced here by a stand-in that answers each query with a chosen exit
code, output and error text, so the failures can be produced on demand. The clean and
the modified working tree are also checked against a real repository.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from process_transfer.data import provenance
from process_transfer.data.provenance import git_state, new_run_directory

COMMIT = "0123456789abcdef0123456789abcdef01234567"
Answer = tuple[int, bytes, bytes]  # exit code, stdout, stderr


def stand_in_for_git(monkeypatch: pytest.MonkeyPatch, answers: dict[str, Answer]) -> list[str]:
    """Replace the git calls of ``provenance``; returns the list of queries made."""
    asked: list[str] = []

    def fake(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        query = " ".join(arguments)
        asked.append(query)
        code, out, err = answers[query]
        return subprocess.CompletedProcess(["git", *arguments], code, out, err)

    monkeypatch.setattr(provenance.shutil, "which", lambda name: "git")
    monkeypatch.setattr(provenance, "_git", fake)
    return asked


def checkout() -> dict[str, Answer]:
    """The answers of a clean checkout at ``COMMIT``; a test overwrites what it needs."""
    return {
        "rev-parse --is-inside-work-tree": (0, b"true\n", b""),
        "rev-parse HEAD": (0, COMMIT.encode() + b"\n", b""),
        "status --porcelain": (0, b"", b""),
        "diff HEAD": (0, b"", b""),
    }


def test_a_failed_status_is_not_a_clean_working_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reproduction of the review: a commit is found, ``git status`` exits with 128
    and prints nothing. It used to give dirty = False and code_identified = True."""
    answers = checkout()
    answers["status --porcelain"] = (128, b"", b"fatal: unable to read index file\n")
    stand_in_for_git(monkeypatch, answers)

    state = git_state(Path("."))
    assert state["available"] is True
    assert state["commit"] == COMMIT  # what is reliable is kept
    assert state["dirty"] is None  # unknown, neither clean nor modified
    assert state["changed_files"] is None
    assert state["diff_sha256"] is None
    assert state["code_identified"] is False
    assert "git status failed with exit code 128" in str(state["reason"])
    assert "unable to read index file" in str(state["reason"])


def test_a_failed_diff_leaves_the_changes_without_a_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = checkout()
    answers["status --porcelain"] = (0, b" M src/a.py\n?? notes.txt\n", b"")
    answers["diff HEAD"] = (2, b"", b"error: could not read object\n")
    stand_in_for_git(monkeypatch, answers)

    state = git_state(Path("."))
    assert state["commit"] == COMMIT
    assert state["dirty"] is True
    assert state["changed_files"] == [" M src/a.py", "?? notes.txt"]
    assert state["diff_sha256"] is None  # not the hash of an empty output
    assert state["diff_sha256"] != hashlib.sha256(b"").hexdigest()
    assert state["code_identified"] is False
    assert "git diff failed with exit code 2" in str(state["reason"])


@pytest.mark.parametrize("output", [b"", b"\n", b"HEAD\n", b"0123abc\n", b"zz" * 20 + b"\n"])
def test_a_commit_that_is_not_an_object_name_is_not_recorded(
    output: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same kind of mistake one query earlier: an exit code of zero with an empty or
    malformed answer must not become the identity of the code."""
    answers = checkout()
    answers["rev-parse HEAD"] = (0, output, b"")
    asked = stand_in_for_git(monkeypatch, answers)

    state = git_state(Path("."))
    assert state["commit"] is None
    assert state["code_identified"] is False
    assert "not a commit" in str(state["reason"])
    assert "status --porcelain" not in asked  # nothing further is claimed


def test_a_clean_and_a_modified_tree_as_answered_by_git(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_for_git(monkeypatch, checkout())
    clean = git_state(Path("."))
    assert clean["dirty"] is False and clean["code_identified"] is True
    assert clean["changed_files"] == [] and clean["diff_sha256"] is None

    answers = checkout()
    answers["status --porcelain"] = (0, b" M src/a.py\n", b"")
    answers["diff HEAD"] = (0, b"diff --git a/src/a.py b/src/a.py\n", b"")
    stand_in_for_git(monkeypatch, answers)
    modified = git_state(Path("."))
    assert modified["dirty"] is True and modified["code_identified"] is False
    assert modified["diff_sha256"] == hashlib.sha256(answers["diff HEAD"][1]).hexdigest()


def test_the_run_id_says_when_the_working_tree_could_not_be_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run whose working tree is unknown must not look like a clean one."""
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path))
    unknown = new_run_directory("x", {"commit": COMMIT, "dirty": None})
    assert unknown.name.endswith("_0123456-unverified")
    assert new_run_directory("x", {"commit": COMMIT, "dirty": False}).name.endswith("_0123456")


# --------------------------------------------------------------------------- #
# Against a real repository
# --------------------------------------------------------------------------- #


def run_git(root: Path, *arguments: str) -> None:
    # A throwaway repository under tmp_path: it gets its own identity and does not ask
    # the developer's signing key for anything.
    settings = [
        "-c", "user.name=Test",
        "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false",
    ]  # fmt: skip
    subprocess.run(["git", *settings, *arguments], cwd=root, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_clean_modified_and_untracked_in_a_real_repository(tmp_path: Path) -> None:
    run_git(tmp_path, "init", "--quiet")
    assert git_state(tmp_path)["commit"] is None  # a checkout without a commit yet
    assert git_state(tmp_path)["code_identified"] is False

    tracked = tmp_path / "model.py"
    tracked.write_text("x = 1\n", encoding="utf-8")
    run_git(tmp_path, "add", "model.py")
    run_git(tmp_path, "commit", "--quiet", "-m", "first")

    clean = git_state(tmp_path)
    assert clean["available"] is True and clean["dirty"] is False
    assert clean["code_identified"] is True
    assert len(str(clean["commit"])) == 40

    tracked.write_text("x = 2\n", encoding="utf-8")
    modified = git_state(tmp_path)
    assert modified["dirty"] is True and modified["code_identified"] is False
    assert modified["commit"] == clean["commit"]
    assert isinstance(modified["diff_sha256"], str)
    assert any("model.py" in line for line in modified["changed_files"])

    tracked.write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "scratch.txt").write_text("not tracked\n", encoding="utf-8")
    untracked = git_state(tmp_path)
    assert untracked["dirty"] is True and untracked["code_identified"] is False
    assert any("scratch.txt" in line for line in untracked["changed_files"])
