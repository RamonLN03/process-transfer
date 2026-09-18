"""Steady-state search, numerical Jacobian and eigenvalues, validated on known cases."""

import math
from pathlib import Path

import numpy as np
import pytest

from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.modeller import cstr_first_order as first_order
from process_transfer.modeller.cstr_first_order import ModellerCSTRParameters
from process_transfer.simulation import cstr_true
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.steady_state import (
    SteadyState,
    find_steady_states,
    numerical_jacobian,
)

PER_MINUTE = 60.0  # multiply an SI rate (1/s) by this to read it in 1/min


# --------------------------------------------------------------------------- #
# The textbook case: Seborg, Edgar, Mellichamp and Doyle, Example 2.5
# --------------------------------------------------------------------------- #

BOOK = ModellerCSTRParameters(
    volume=0.1,  # 100 L
    density=1000.0,
    heat_capacity=239.0,  # 0.239 J/(g K)
    reaction_enthalpy=-5.0e4,
    k0=7.2e10 / 60.0,  # 7.2e10 1/min
    activation_temperature=8750.0,
    ua=5.0e4 / 60.0,  # 5e4 J/(min K)
)
BOOK_INPUTS = np.array([100e-3 / 60.0, 1000.0, 350.0, 300.0])  # q, C_Af, T_f, T_c


def book_rhs(x: np.ndarray) -> np.ndarray:
    return first_order.rhs(0.0, x, BOOK_INPUTS, BOOK)


def first_order_jacobian(x: np.ndarray, u: np.ndarray, p: ModellerCSTRParameters) -> np.ndarray:
    """Hand-derived Jacobian of the first-order model, independent of the code under test."""
    c_a, temperature = x
    q = u[0]
    k = p.k0 * math.exp(-p.activation_temperature / temperature)
    dk_dt = k * p.activation_temperature / temperature**2
    beta = -p.reaction_enthalpy / (p.density * p.heat_capacity)
    gamma = p.ua / (p.volume * p.density * p.heat_capacity)
    return np.array(
        [
            [-q / p.volume - k, -c_a * dk_dt],
            [beta * k, -q / p.volume + beta * c_a * dk_dt - gamma],
        ]
    )


def true_jacobian(x: np.ndarray, u: np.ndarray, p: TrueCSTRParameters) -> np.ndarray:
    """Hand-derived Jacobian of the true plant (saturating kinetics, UA(T))."""
    c_a, temperature = x
    q, _, _, t_c = u
    k = p.k0 * math.exp(-p.activation_temperature / temperature)
    saturation = 1.0 + p.saturation_constant * c_a
    rate = k * c_a / saturation
    dr_dc = k / saturation**2
    dr_dt = rate * p.activation_temperature / temperature**2
    beta = -p.reaction_enthalpy / (p.density * p.heat_capacity)
    thermal_mass = p.volume * p.density * p.heat_capacity
    ua = p.ua_ref * (1.0 + p.alpha * (temperature - p.t_ref))
    dcooling_dt = (p.ua_ref * p.alpha * (temperature - t_c) + ua) / thermal_mass
    return np.array(
        [
            [-q / p.volume - dr_dc, -dr_dt],
            [beta * dr_dc, -q / p.volume + beta * dr_dt - dcooling_dt],
        ]
    )


@pytest.mark.parametrize("x", [[500.0, 350.0], [877.0, 324.5], [209.0, 369.7], [50.0, 410.0]])
def test_numerical_jacobian_matches_analytical_first_order(x: list[float]) -> None:
    state = np.array(x)
    np.testing.assert_allclose(
        numerical_jacobian(book_rhs, state),
        first_order_jacobian(state, BOOK_INPUTS, BOOK),
        rtol=1e-7,
        atol=1e-12,
    )


@pytest.mark.parametrize("plant_file", ["source_cstr.yaml", "target_cstr.yaml"])
@pytest.mark.parametrize("x", [[250.0, 350.0], [150.0, 362.0], [400.0, 341.0]])
def test_numerical_jacobian_matches_analytical_true_plant(
    plant_file: str, x: list[float], configs_dir: Path
) -> None:
    cfg = load_true_plant(configs_dir / plant_file)
    p, u = TrueCSTRParameters.from_config(cfg), nominal_inputs(cfg.plant)
    state = np.array(x)
    np.testing.assert_allclose(
        numerical_jacobian(lambda s: cstr_true.rhs(0.0, s, u, p), state),
        true_jacobian(state, u, p),
        rtol=1e-7,
        atol=1e-12,
    )


@pytest.fixture(scope="module")
def book_steady_states() -> list[SteadyState]:
    return find_steady_states(book_rhs, c_a_upper=BOOK_INPUTS[1])


def test_book_case_has_three_steady_states_with_the_known_stability_pattern(
    book_steady_states: list[SteadyState],
) -> None:
    """At T_c = 300 K the textbook reactor has a stable low-conversion state, a
    saddle at the book's nominal point and an unstable focus at high conversion.
    The book's own operating point is therefore open-loop unstable, which is why
    D-005 redesigns the plant."""
    low, middle, high = book_steady_states
    assert len(book_steady_states) == 3

    assert low.is_stable()
    assert np.all(np.abs(low.eigenvalues.imag) > 0)  # stable focus

    assert np.all(middle.eigenvalues.imag == 0)  # saddle: real eigenvalues of opposite sign
    assert np.min(middle.eigenvalues.real) < 0 < np.max(middle.eigenvalues.real)

    assert not high.is_stable()
    assert np.all(high.eigenvalues.real > 0)  # unstable focus
    assert np.all(np.abs(high.eigenvalues.imag) > 0)


def test_book_case_agrees_with_an_independent_implementation(
    book_steady_states: list[SteadyState],
) -> None:
    """Cross-check against a separate pure-Python calculation (Newton iteration with
    forward differences, outside the repository). Agreement of two independent
    implementations is the evidence; neither is assumed correct beforehand."""
    expected = [  # C_A mol/L, T K, eigenvalues 1/min
        (0.8773, 324.48, [-1.049 + 0.539j, -1.049 - 0.539j]),
        (0.4999, 350.01, [2.834, -0.454]),
        (0.2088, 369.70, [1.357 + 1.540j, 1.357 - 1.540j]),
    ]
    for found, (c_a, temperature, eigenvalues) in zip(book_steady_states, expected, strict=True):
        assert found.c_a / 1000.0 == pytest.approx(c_a, abs=1e-4)
        assert found.temperature == pytest.approx(temperature, abs=0.01)
        found_per_min = np.sort_complex(found.eigenvalues * PER_MINUTE)
        np.testing.assert_allclose(found_per_min, np.sort_complex(eigenvalues), atol=2e-3)


def test_steady_states_close_both_balances_and_the_analytical_mass_balance(
    book_steady_states: list[SteadyState],
) -> None:
    """Residuals vanish, and C_A satisfies the closed form C_Af / (1 + tau k(T))
    of a first-order reaction, which the search never uses."""
    q, c_af = BOOK_INPUTS[0], BOOK_INPUTS[1]
    tau = BOOK.volume / q
    for steady in book_steady_states:
        assert abs(steady.residual[0]) < 1e-10 * c_af / tau
        assert abs(steady.residual[1]) < 1e-10 * steady.temperature / tau
        k = BOOK.k0 * math.exp(-BOOK.activation_temperature / steady.temperature)
        assert steady.c_a == pytest.approx(c_af / (1.0 + tau * k), rel=1e-9)


def test_search_returns_nothing_outside_the_roots() -> None:
    found = find_steady_states(book_rhs, c_a_upper=BOOK_INPUTS[1], temperature_range=(400.0, 480.0))
    assert found == []


def test_stability_margin_semantics(book_steady_states: list[SteadyState]) -> None:
    low = book_steady_states[0]
    assert low.is_stable(margin=0.0)
    assert not low.is_stable(margin=-low.max_real_part + 1e-6)
    assert low.max_real_part == pytest.approx(-1.049 / PER_MINUTE, abs=1e-4)
