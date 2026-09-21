"""Where generated files go.

Everything the project generates (figures, summaries, and later Parquet files and
the DuckDB database) lives under one directory, set by the ``PT_DATA_DIR``
environment variable and ignored by git. A relative value is resolved against the
repository root; the default is ``data``.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_VARIABLE = "PT_DATA_DIR"
DEFAULT_DATA_DIR = "data"


def repository_root() -> Path:
    """The directory that holds ``pyproject.toml``, found by walking up from here.

    Falls back to the current working directory when the package is installed
    outside a checkout."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def data_dir() -> Path:
    """The root directory for generated files. It is not created here."""
    configured = Path(os.environ.get(DATA_DIR_VARIABLE) or DEFAULT_DATA_DIR)
    return configured if configured.is_absolute() else repository_root() / configured


def output_dir(*parts: str) -> Path:
    """A directory for generated files under ``data_dir()``, created if missing."""
    path = data_dir().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
