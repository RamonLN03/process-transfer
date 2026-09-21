"""Shared pytest fixtures."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.measurement.observations import Observations
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


def synthetic_observations(
    plant: str = "target",
    run: str = "target.p3.e0.x1.n0",
    n: int = 21,
    offset: float = 0.0,
    sample_period: float = 6.0,
) -> Observations:
    """A small, deterministic observation set that needs no simulation: a reading that
    wanders, one negative reading, and inputs that switch once, at row ``n // 2``."""
    k = np.arange(n, dtype=np.float64)
    measured = np.column_stack([190.0 + 25.0 * np.sin(0.7 * k) + offset, 355.0 + np.cos(0.3 * k)])
    measured[3, 0] = -2.5  # a noisy concentration reading may be negative; it is stored as it is
    inputs = np.tile([1.0e-3 / 0.6, 500.0, 350.0, 337.5], (n, 1))
    inputs[n // 2 :, 3] = 342.5  # the coolant temperature steps up, from that row on
    return Observations(
        plant=plant,
        run=run,
        times=sample_period * k,
        measured=measured,
        inputs=inputs,
        measured_names=("C_A", "T"),
        measured_units=("mol/m^3", "K"),
        input_names=("q", "C_Af", "T_f", "T_c"),
        input_units=("m^3/s", "mol/m^3", "K", "K"),
        sample_period=sample_period,
        noise_std=(5.0, 0.5),
    )
