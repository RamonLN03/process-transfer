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
or the working tree has uncommitted changes, that is recorded as such: no commit is
invented and the run is not presented as fully identified.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from process_transfer.data.paths import output_dir, repository_root

PACKAGES = ("process-transfer", "numpy", "scipy", "pydantic", "pyyaml", "matplotlib")


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False)


def git_state(root: Path | None = None) -> dict[str, object]:
    """The commit and the state of the working tree, or an explicit statement that they
    could not be determined.

    ``code_identified`` is true only when the commit is known and nothing is modified
    or untracked, so that the commit alone reproduces the code. With local changes the
    commit, the list of changed files and a hash of ``git diff HEAD`` are recorded, but
    the run is not claimed to be reproducible from the commit.
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
        return {**unknown, "reason": f"{root} is not a git checkout"}
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        return {**unknown, "reason": "the checkout has no commit yet"}

    status = _git(root, "status", "--porcelain").stdout.decode("utf-8", "replace").splitlines()
    dirty = len(status) > 0
    diff = _git(root, "diff", "HEAD").stdout if dirty else b""
    return {
        "available": True,
        "commit": head.stdout.decode("ascii").strip(),
        "dirty": dirty,
        "changed_files": status,
        "diff_sha256": hashlib.sha256(diff).hexdigest() if dirty else None,
        "code_identified": not dirty,
        "reason": (
            "uncommitted or untracked changes: the commit alone does not identify the code"
            if dirty
            else "clean working tree"
        ),
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
    ``-dirty`` mark. Two runs in the same second get a numeric suffix. An existing
    directory is never reused, so a run never overwrites another.
    """
    state = git_state() if state is None else state
    commit = str(state["commit"])[:7] if state.get("commit") else "nogit"
    mark = "-dirty" if state.get("dirty") else ""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parent = output_dir("experiments", experiment)
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"-{attempt}"
        candidate = parent / f"{stamp}_{commit}{mark}{suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        return candidate
