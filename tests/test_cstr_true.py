"""The true CSTR right-hand side: SI parameters, constitutive laws and balances."""

import math
from pathlib import Path

import numpy as np
import pytest

from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import INPUT_NAMES, STATE_NAMES, nominal_inputs
from process_transfer.simulation.cstr_true import (
    TrueCSTRParameters,
    conductance,
    reaction_rate,
    rhs,
)


@pytest.fixture(scope="module")
def source(configs_dir: Path) -> TrueCSTRParameters:
    return TrueCSTRParameters.from_config(load_true_plant(configs_dir / "source_cstr.yaml"))


@pytest.fixture(scope="module")
def target(configs_dir: Path) -> TrueCSTRParameters:
    return TrueCSTRParameters.from_config(load_true_plant(configs_dir / "target_cstr.yaml"))


@pytest.fixture(scope="module")
def u_nominal(configs_dir: Path) -> np.ndarray:
    return nominal_inputs(load_true_plant(configs_dir / "source_cstr.yaml").plant)


def test_parameters_are_converted_to_si(source: TrueCSTRParameters) -> None:
    assert math.isclose(source.volume, 0.1)  # 100 L
    assert math.isclose(source.density, 1000.0)  # 1000 g/L
    assert math.isclose(source.heat_capacity, 239.0)  # 0.239 J/(g K)
    assert math.isclose(source.reaction_enthalpy, -5.0e4)
    assert math.isclose(source.k0, 1.44e11 / 60.0)  # 1/min -> 1/s
    assert math.isclose(source.activation_temperature, 8750.0)
    assert math.isclose(source.saturation_constant, 4.0e-3)  # 4 L/mol
    assert math.isclose(source.ua_ref, 1.0e5 / 60.0)  # J/(min K) -> W/K
    assert math.isclose(source.alpha, 0.005)
    assert math.isclose(source.t_ref, 350.0)


def test_nominal_inputs_are_si_and_ordered(u_nominal: np.ndarray) -> None:
    assert STATE_NAMES == ("C_A", "T")
    assert INPUT_NAMES == ("q", "C_Af", "T_f", "T_c")
    np.testing.assert_allclose(u_nominal, [100e-3 / 60.0, 500.0, 350.0, 337.5], rtol=1e-12)


def test_target_differs_from_source_only_in_heat_transfer(
    source: TrueCSTRParameters, target: TrueCSTRParameters
) -> None:
    differing = {
        name
        for name in source.__dataclass_fields__
        if getattr(source, name) != getattr(target, name)
    }
    assert differing == {"ua_ref", "alpha"}


def test_rate_is_zero_without_reactant(source: TrueCSTRParameters) -> None:
    assert reaction_rate(0.0, 350.0, source) == 0.0


def test_rate_increases_with_concentration_and_temperature(source: TrueCSTRParameters) -> None:
    assert reaction_rate(300.0, 350.0, source) > reaction_rate(200.0, 350.0, source)
    assert reaction_rate(250.0, 360.0, source) > reaction_rate(250.0, 350.0, source)


def test_rate_saturates_at_high_concentration(source: TrueCSTRParameters) -> None:
    """For K_sat C_A >> 1 the rate tends to k(T) / K_sat, independent of C_A."""
    k = source.k0 * math.exp(-source.activation_temperature / 350.0)
    limit = k / source.saturation_constant
    assert math.isclose(reaction_rate(1.0e9, 350.0, source), limit, rel_tol=1e-6)
    assert reaction_rate(1.0e9, 350.0, source) < limit


def test_conductance_is_linear_in_temperature(source: TrueCSTRParameters) -> None:
    assert math.isclose(conductance(350.0, source), source.ua_ref)
    assert math.isclose(conductance(360.0, source), source.ua_ref * 1.05)
    assert math.isclose(conductance(340.0, source), source.ua_ref * 0.95)


def test_rhs_matches_hand_calculation_in_engineering_units(
    source: TrueCSTRParameters, u_nominal: np.ndarray
) -> None:
    """Evaluate the documented balances in the book's units (L, min, mol/L) and
    convert the derivatives to SI. This checks the equations and the unit
    handling through an independent path."""
    c_a, temperature = 0.3, 355.0  # mol/L, K
    q, volume, c_af, t_f, t_c = 100.0, 100.0, 0.5, 350.0, 337.5  # L/min, L, mol/L, K, K
    rho, cp, d_h = 1000.0, 0.239, -5.0e4  # g/L, J/(g K), J/mol
    k0, theta, k_sat = 1.44e11, 8750.0, 4.0  # 1/min, K, L/mol
    ua_ref, alpha, t_ref = 1.0e5, 0.005, 350.0  # J/(min K), 1/K, K

    rate = k0 * math.exp(-theta / temperature) * c_a / (1.0 + k_sat * c_a)  # mol/(L min)
    dc_a_dt = (q / volume) * (c_af - c_a) - rate  # mol/(L min)
    ua = ua_ref * (1.0 + alpha * (temperature - t_ref))
    dt_dt = (
        (q / volume) * (t_f - temperature)
        + (-d_h) / (rho * cp) * rate
        - ua / (volume * rho * cp) * (temperature - t_c)
    )  # K/min

    expected_si = np.array([dc_a_dt * 1000.0 / 60.0, dt_dt / 60.0])
    actual = rhs(0.0, np.array([c_a * 1000.0, temperature]), u_nominal, source)
    np.testing.assert_allclose(actual, expected_si, rtol=1e-12)


def test_rhs_response_directions(source: TrueCSTRParameters, u_nominal: np.ndarray) -> None:
    x = np.array([250.0, 350.0])
    base = rhs(0.0, x, u_nominal, source)

    def perturbed(index: int, delta: float) -> np.ndarray:
        u = u_nominal.copy()
        u[index] += delta
        return rhs(0.0, x, u, source)

    assert perturbed(3, +5.0)[1] > base[1]  # warmer coolant: temperature rises faster
    assert perturbed(3, +5.0)[0] == base[0]  # coolant does not enter the mass balance
    assert perturbed(2, +5.0)[1] > base[1]  # warmer feed: temperature rises faster
    assert perturbed(1, +50.0)[0] > base[0]  # richer feed: concentration rises faster
    assert perturbed(0, +1e-4)[0] > base[0]  # more flow of richer feed (C_Af > C_A)


def test_concentration_cannot_become_negative(
    source: TrueCSTRParameters, u_nominal: np.ndarray
) -> None:
    """At C_A = 0 the reaction term vanishes and the feed can only raise C_A."""
    derivative = rhs(0.0, np.array([0.0, 350.0]), u_nominal, source)
    assert derivative[0] > 0.0


def test_rhs_is_pure_and_returns_float64(source: TrueCSTRParameters, u_nominal: np.ndarray) -> None:
    x = np.array([250.0, 350.0])
    x_before, u_before = x.copy(), u_nominal.copy()
    first = rhs(0.0, x, u_nominal, source)
    second = rhs(123.0, x, u_nominal, source)  # autonomous: time does not matter
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(u_nominal, u_before)
    assert first.shape == (2,)
    assert first.dtype == np.float64
