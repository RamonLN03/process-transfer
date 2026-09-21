"""Shared pytest fixtures."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.simulation import cstr_true
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.steady_state import find_steady_states

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    """Directory holding the committed YAML configurations."""
    return REPO_ROOT / "configs"


@dataclass(frozen=True)
class PlantUnderTest:
    """A true plant with its nominal inputs and its nominal steady state, in SI."""

    parameters: TrueCSTRParameters
    nominal_inputs: np.ndarray
    nominal_state: np.ndarray

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return cstr_true.rhs(0.0, x, u, self.parameters)


@pytest.fixture(scope="session")
def true_plants(configs_dir: Path) -> dict[str, PlantUnderTest]:
    plants: dict[str, PlantUnderTest] = {}
    for name in ("source", "target"):
        cfg = load_true_plant(configs_dir / f"{name}_cstr.yaml")
        p, u = TrueCSTRParameters.from_config(cfg), nominal_inputs(cfg.plant)
        (steady,) = find_steady_states(
            lambda x, u=u, p=p: cstr_true.rhs(0.0, x, u, p), c_a_upper=u[1]
        )
        plants[name] = PlantUnderTest(p, u, steady.state)
    return plants
