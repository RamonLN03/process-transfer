"""Identify a run: which code, which configuration, which environment.

Layout under ``PT_DATA_DIR``, all of it ignored by git:

    experiments/<experiment>/<run id>/      one directory per run, never reused
        summary.json                        results and the provenance block
        configs/                            copies of the configuration files used
        *.png                               figures

Everything under ``experiments/`` is a diagnostic artefact of the simulator. It may
contain hidden parameters, since the true-plant configurations are copied there in
full. It must stay apart from the data that will later be made available for
training or adaptation, which never carries ground truth (``AGENTS.md``).

A run is only as well identified as the information available. When git is missing,
a git query fails, or the working tree has uncommitted changes, that is recorded as
such: no commit is invented, a failed query is never read as a clean answer, and the
run is not presented as fully identified.
"""

from __future__ import annotations

import hashlib
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from process_transfer.data.paths import output_dir, repository_root

PACKAGES = (
    "process-transfer",
    "numpy",
    "scipy",
    "pydantic",
    "pyyaml",
    "matplotlib",
    "pyarrow",
    "duckdb",
    "pandas",
)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False)


def _failure(query: str, result: subprocess.CompletedProcess[bytes]) -> str:
    """One line saying which query failed, with its exit code and git's own words."""
    said = result.stderr.decode("utf-8", "replace").strip().splitlines()
    detail = f": {said[0]}" if said else ""
    return f"git {query} failed with exit code {result.returncode}{detail}"


# A full object name: 40 hexadecimal digits with SHA-1, 64 with SHA-256.
_OBJECT_NAME = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")


def git_state(root: Path | None = None) -> dict[str, object]:
    """The commit and the state of the working tree, or an explicit statement of what
    could not be determined.

    ``code_identified`` is true only when the commit is known and git has answered that
    nothing is modified or untracked, so that the commit alone reproduces the code. With
    local changes the commit, the list of changed files and a hash of ``git diff HEAD``
    are recorded, but the run is not claimed to be reproducible from the commit. That
    hash covers tracked files only: untracked files are listed, not fingerprinted.

    A query that fails is not an answer. Every exit code is checked, and an empty output
    is read as "nothing to report" only from a query that succeeded. When the working
    tree cannot be read, ``dirty`` is ``None``, which means unknown, not clean; when the
    changes cannot be hashed, ``diff_sha256`` is ``None``. What was obtained reliably,
    such as the commit, is kept, and ``reason`` names the query that failed.
    """
    root = repository_root() if root is None else root
    unknown: dict[str, object] = {
        "available": False,
        "commit": None,
        "dirty": None,
        "changed_files": None,
        "diff_sha256": None,
        "code_identified": False,
    }
    if shutil.which("git") is None:
        return {**unknown, "reason": "the git executable was not found"}
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != b"true":
        said = inside.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = f" ({said[0]})" if said else ""
        return {**unknown, "reason": f"{root} is not a git checkout{detail}"}
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        # The usual cause is a checkout without a commit yet; git's words say which.
        return {
            **unknown,
            "reason": f"{_failure('rev-parse HEAD', head)}; no commit identifies this checkout",
        }
    commit = head.stdout.decode("ascii", "replace").strip()
    if _OBJECT_NAME.fullmatch(commit) is None:
        return {**unknown, "reason": f"git rev-parse HEAD answered {commit!r}, not a commit"}

    known: dict[str, object] = {**unknown, "available": True, "commit": commit}
    status = _git(root, "status", "--porcelain")
    if status.returncode != 0:
        return {
            **known,
            "reason": _failure("status", status)
            + "; whether the working tree matches the commit is unknown",
        }
    changed_files = status.stdout.decode("utf-8", "replace").splitlines()
    if not changed_files:
        return {
            **known,
            "dirty": False,
            "changed_files": [],
            "code_identified": True,
            "reason": "clean working tree",
        }

    reason = "uncommitted or untracked changes: the commit alone does not identify the code"
    diff = _git(root, "diff", "HEAD")
    if diff.returncode != 0:
        return {
            **known,
            "dirty": True,
            "changed_files": changed_files,
            "reason": f"{reason}; {_failure('diff', diff)}, so the changes have no fingerprint",
        }
    return {
        **known,
        "dirty": True,
        "changed_files": changed_files,
        "diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "reason": reason,
    }


def file_fingerprint(path: Path) -> dict[str, object]:
    """SHA-256 and size of a file, to verify later that a copy is the file that was used."""
    data = Path(path).read_bytes()
    return {"name": Path(path).name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def copy_with_fingerprints(paths: list[Path], destination: Path) -> list[dict[str, object]]:
    """Copy configuration files into ``destination`` and return their fingerprints."""
    destination.mkdir(parents=True, exist_ok=True)
    fingerprints = []
    for path in paths:
        shutil.copyfile(path, destination / Path(path).name)
        fingerprints.append(file_fingerprint(destination / Path(path).name))
    return fingerprints


def environment() -> dict[str, object]:
    """Interpreter, platform and the versions of the packages that affect the numbers."""
    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None  # recorded as unknown, not guessed
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def new_run_directory(experiment: str, state: dict[str, object] | None = None) -> Path:
    """A fresh directory ``experiments/<experiment>/<run id>`` that did not exist before.

    The run id is the UTC time to the second, the short commit (or ``nogit``) and a
    mark: ``-dirty`` when the working tree has changes, ``-unverified`` when the commit
    is known but the working tree could not be read. Only a working tree that git
    reported as clean carries no mark. Two runs in the same second get a numeric
    suffix. An existing directory is never reused, so a run never overwrites another.
    """
    return new_directory(output_dir("experiments", experiment), state)


def attempt_stamp(state: dict[str, object] | None = None) -> str:
    """UTC time to the second, the short commit (or ``nogit``) and the mark of the working
    tree: what identifies one attempt at running something."""
    state = git_state() if state is None else state
    commit = str(state["commit"])[:7] if state.get("commit") else "nogit"
    dirty = state.get("dirty")
    if dirty is None:  # unknown is not clean
        mark = "-unverified" if state.get("commit") else ""
    else:
        mark = "-dirty" if dirty else ""
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{commit}{mark}"


def new_directory(parent: Path, state: dict[str, object] | None = None) -> Path:
    """A fresh directory under ``parent`` named after this attempt. An existing directory
    is never reused: two attempts in the same second get a numeric suffix."""
    parent.mkdir(parents=True, exist_ok=True)
    stamp = attempt_stamp(state)
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"-{attempt}"
        candidate = parent / f"{stamp}{suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        return candidate
