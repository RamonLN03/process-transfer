"""D-009 at the nominal inputs: the operating points of source and target are unique,
stable with margin, and reached by time integration."""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.simulation import cstr_true
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.steady_state import SteadyState, find_steady_states

PER_MINUTE = 60.0
STABILITY_MARGIN = 0.5 / PER_MINUTE  # 0.5 1/min in SI (docs/decisions.md D-017)

Plant = tuple[TrueCSTRParameters, np.ndarray, list[SteadyState]]


@pytest.fixture(scope="module")
def plants(configs_dir: Path) -> dict[str, Plant]:
    result: dict[str, Plant] = {}
    for name in ("source", "target"):
        cfg = load_true_plant(configs_dir / f"{name}_cstr.yaml")
        p, u = TrueCSTRParameters.from_config(cfg), nominal_inputs(cfg.plant)
        steady_states = find_steady_states(
            lambda x, u=u, p=p: cstr_true.rhs(0.0, x, u, p), c_a_upper=u[1]
        )
        result[name] = (p, u, steady_states)
    return result


@pytest.mark.parametrize("name", ["source", "target"])
def test_operating_point_is_unique_between_280_and_480_kelvin(
    name: str, plants: dict[str, Plant]
) -> None:
    assert len(plants[name][2]) == 1


@pytest.mark.parametrize("name", ["source", "target"])
def test_operating_point_is_stable_with_margin(name: str, plants: dict[str, Plant]) -> None:
    (steady,) = plants[name][2]
    assert steady.is_stable(STABILITY_MARGIN)


@pytest.mark.parametrize("name", ["source", "target"])
def test_operating_point_closes_both_balances(name: str, plants: dict[str, Plant]) -> None:
    _, u, (steady,) = plants[name]
    dilution = u[0] / 0.1  # q / V, 1/s
    assert abs(steady.residual[0]) < 1e-10 * u[1] * dilution
    assert abs(steady.residual[1]) < 1e-10 * steady.temperature * dilution


def test_source_sits_at_the_designed_nominal_point(plants: dict[str, Plant]) -> None:
    """D-005 designs the source for C_A = 250 mol/m^3 at 350 K. The book's k0 and E/R
    give k(350 K) = 0.99993 1/min rather than exactly 1, so the point is met to a few
    hundredths of a mol/m^3 and a millikelvin, not exactly."""
    (steady,) = plants["source"][2]
    assert steady.c_a == pytest.approx(250.0, abs=0.05)
    assert steady.temperature == pytest.approx(350.0, abs=0.002)


def test_target_runs_hotter_and_more_converted_than_source(plants: dict[str, Plant]) -> None:
    """Same inputs, less cooling (D-008): the target must settle hotter, with less A left."""
    (source,), (target,) = plants["source"][2], plants["target"][2]
    assert target.temperature > source.temperature
    assert target.c_a < source.c_a


@pytest.mark.parametrize("name", ["source", "target"])
def test_time_integration_converges_to_the_steady_state(
    name: str, plants: dict[str, Plant]
) -> None:
    """An independent route to the same point: integrate the ODE from a perturbed
    state and let it settle. Root finding and time integration must agree."""
    p, u, (steady,) = plants[name]
    perturbed = steady.state + np.array([15.0, 3.0])
    solution = solve_ivp(
        lambda t, x: cstr_true.rhs(t, x, u, p),
        (0.0, 3600.0),
        perturbed,
        method="LSODA",
        rtol=1e-10,
        atol=1e-10,
    )
    assert solution.success
    np.testing.assert_allclose(solution.y[:, -1], steady.state, rtol=1e-7)


def test_repository_values_reproduce_the_preliminary_calculation(
    plants: dict[str, Plant],
) -> None:
    """The scratch calculation quoted in the first version of docs/assumptions.md gave,
    at its printed precision: source (0.25 mol/L, 350 K), -1.60 +- 1.36i 1/min; target
    (0.190 mol/L, 355.2 K), -0.96 +- 1.80i 1/min. Repository code reproduces them."""
    preliminary = {
        "source": (0.25, 350.0, -1.60, 1.36),
        "target": (0.190, 355.2, -0.96, 1.80),
    }
    for name, (c_a, temperature, real, imag) in preliminary.items():
        (steady,) = plants[name][2]
        eigenvalue = steady.eigenvalues[np.argmax(steady.eigenvalues.imag)] * PER_MINUTE
        assert steady.c_a / 1000.0 == pytest.approx(c_a, abs=5e-4)
        assert steady.temperature == pytest.approx(temperature, abs=0.05)
        assert eigenvalue.real == pytest.approx(real, abs=5e-3)
        assert eigenvalue.imag == pytest.approx(imag, abs=5e-3)


def test_operating_points_regression(plants: dict[str, Plant]) -> None:
    """Pins of the repository-computed values (M0-E01), to catch accidental changes
    to the physics or to the configuration."""
    pinned = {
        "source": (250.020898, 349.999197, -1.604980 + 1.362469j),
        "target": (189.672763, 355.168666, -0.963586 + 1.804133j),
    }
    for name, (c_a, temperature, eigenvalue) in pinned.items():
        (steady,) = plants[name][2]
        found = steady.eigenvalues[np.argmax(steady.eigenvalues.imag)] * PER_MINUTE
        assert steady.c_a == pytest.approx(c_a, rel=1e-7)
        assert steady.temperature == pytest.approx(temperature, rel=1e-8)
        assert found == pytest.approx(eigenvalue, abs=1e-5)
