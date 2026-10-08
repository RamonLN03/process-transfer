"""The provenance of a run that trains records what decides the numbers of JAX, and the
paths that do not train never import it (I4 of M1, amendment of the registration of
M1-E02, point E)."""

import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import pytest

from process_transfer.data.provenance import environment
from process_transfer.models import training


def test_a_run_that_trains_records_jax_jaxlib_the_backend_and_the_precision() -> None:
    found = training.learning_environment()
    assert {key: found[key] for key in environment()} == environment()
    learning = found["learning"]
    assert set(learning) == {"jax", "jaxlib", "backend", "devices", "x64", "float_dtype"}
    assert learning["jax"] and learning["jaxlib"] and learning["backend"] and learning["devices"]
    assert learning["x64"] is True and learning["float_dtype"] == "float64"


def test_a_version_that_cannot_be_read_is_unknown_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(training, "version", missing)
    learning = training.learning_environment()["learning"]
    assert learning["jax"] is None and learning["jaxlib"] is None


def test_the_paths_that_do_not_train_do_not_import_jax() -> None:
    code = (
        "import sys\n"
        "import process_transfer.data.provenance, process_transfer.generation.pipeline\n"
        "import process_transfer.evaluation.windows, process_transfer.models.rollout\n"
        "import process_transfer.models.fitting, process_transfer.simulation.protocols\n"
        "print(sorted(m for m in sys.modules if m == 'jax' or m.startswith('jax.')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
