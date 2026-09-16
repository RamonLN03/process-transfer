"""Shared pytest fixtures."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    """Directory holding the committed YAML configurations."""
    return REPO_ROOT / "configs"
