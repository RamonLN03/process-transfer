"""The modeller's first-order CSTR and its designed mismatch with the true plant."""

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from process_transfer.config import load_modeller, load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.modeller import cstr_first_order as modeller_model
from process_transfer.modeller.cstr_first_order import ModellerCSTRParameters
from process_transfer.simulation import cstr_true as true_model
from process_transfer.simulation.cstr_true import TrueCSTRParameters


@pytest.fixture(scope="module")
def modeller(configs_dir: Path) -> ModellerCSTRParameters:
    plant = load_true_plant(configs_dir / "source_cstr.yaml").plant  # known part only
    return ModellerCSTRParameters.from_config(
        plant, load_modeller(configs_dir / "modeller_cstr.yaml")
    )


@pytest.fixture(scope="module")
def true_source(configs_dir: Path) -> TrueCSTRParameters:
    return TrueCSTRParameters.from_config(load_true_plant(configs_dir / "source_cstr.yaml"))


@pytest.fixture(scope="module")
def u_nominal(configs_dir: Path) -> np.ndarray:
    return nominal_inputs(load_true_plant(configs_dir / "source_cstr.yaml").plant)


def test_parameters_are_converted_to_si(modeller: ModellerCSTRParameters) -> None:
    assert math.isclose(modeller.volume, 0.1)
    assert math.isclose(modeller.density, 1000.0)
    assert math.isclose(modeller.heat_capacity, 239.0)
    assert math.isclose(modeller.reaction_enthalpy, -5.0e4)
    assert math.isclose(modeller.k0, 7.2e10 / 60.0)
    assert math.isclose(modeller.activation_temperature, 8750.0)
    assert math.isclose(modeller.ua, 1.0e5 / 60.0)


def test_modeller_parameters_are_built_from_known_information_only() -> None:
    """The constructor takes the known plant specification and the modeller's
    configuration; it has no argument through which hidden physics could enter."""
    signature = inspect.signature(ModellerCSTRParameters.from_config)
    assert list(signature.parameters) == ["plant", "modeller"]
    fields = set(ModellerCSTRParameters.__dataclass_fields__)
    assert fields.isdisjoint({"saturation_constant", "alpha", "ua_ref", "t_ref"})


def test_rhs_matches_hand_calculation_in_engineering_units(
    modeller: ModellerCSTRParameters, u_nominal: np.ndarray
) -> None:
    c_a, temperature = 0.3, 355.0  # mol/L, K
    q, volume, c_af, t_f, t_c = 100.0, 100.0, 0.5, 350.0, 337.5
    rho, cp, d_h = 1000.0, 0.239, -5.0e4
    k0, theta, ua = 7.2e10, 8750.0, 1.0e5  # 1/min, K, J/(min K)

    rate = k0 * math.exp(-theta / temperature) * c_a  # mol/(L min)
    dc_a_dt = (q / volume) * (c_af - c_a) - rate
    dt_dt = (
        (q / volume) * (t_f - temperature)
        + (-d_h) / (rho * cp) * rate
        - ua / (volume * rho * cp) * (temperature - t_c)
    )

    expected_si = np.array([dc_a_dt * 1000.0 / 60.0, dt_dt / 60.0])
    actual = modeller_model.rhs(0.0, np.array([c_a * 1000.0, temperature]), u_nominal, modeller)
    np.testing.assert_allclose(actual, expected_si, rtol=1e-12)


def test_rates_coincide_at_the_nominal_point(
    modeller: ModellerCSTRParameters, true_source: TrueCSTRParameters
) -> None:
    """D-006: at C_A = 250 mol/m^3 (K_sat C_A = 1) the two rate laws agree at any
    temperature, because they share E/R and k0_true = 2 k0."""
    for temperature in (340.0, 350.0, 360.0):
        assert math.isclose(
            true_model.reaction_rate(250.0, temperature, true_source),
            modeller_model.reaction_rate(250.0, temperature, modeller),
            rel_tol=1e-12,
        )


@pytest.mark.parametrize("c_a", [100.0, 150.0, 200.0, 300.0, 350.0, 400.0])
def test_kinetic_mismatch_is_a_function_of_concentration_only(
    c_a: float, modeller: ModellerCSTRParameters, true_source: TrueCSTRParameters
) -> None:
    """Away from the nominal concentration r_true / r_model = 2 / (1 + K_sat C_A):
    above one when dilute, below one when concentrated, the same at every temperature.
    No choice of k0 and E/R in the first-order law can remove it."""
    expected = 2.0 / (1.0 + true_source.saturation_constant * c_a)
    for temperature in (340.0, 350.0, 360.0):
        ratio = true_model.reaction_rate(c_a, temperature, true_source) / (
            modeller_model.reaction_rate(c_a, temperature, modeller)
        )
        assert math.isclose(ratio, expected, rel_tol=1e-12)
    assert (expected > 1.0) == (c_a < 250.0)


def test_models_share_the_balances_and_differ_only_in_hidden_laws(
    modeller: ModellerCSTRParameters, u_nominal: np.ndarray
) -> None:
    """With saturation and the temperature dependence of UA switched off, and the
    modeller's parameter values, the true right-hand side reduces exactly to the
    modeller's. The two models therefore differ only in the hidden constitutive laws."""
    degenerate_truth = TrueCSTRParameters(
        volume=modeller.volume,
        density=modeller.density,
        heat_capacity=modeller.heat_capacity,
        reaction_enthalpy=modeller.reaction_enthalpy,
        k0=modeller.k0,
        activation_temperature=modeller.activation_temperature,
        saturation_constant=0.0,
        ua_ref=modeller.ua,
        alpha=0.0,
        t_ref=350.0,
    )
    for x in (np.array([250.0, 350.0]), np.array([120.0, 362.0]), np.array([410.0, 341.0])):
        np.testing.assert_allclose(
            true_model.rhs(0.0, x, u_nominal, degenerate_truth),
            modeller_model.rhs(0.0, x, u_nominal, modeller),
            rtol=1e-13,
        )


def test_the_mismatch_is_real_away_from_the_nominal_point(
    modeller: ModellerCSTRParameters, true_source: TrueCSTRParameters, u_nominal: np.ndarray
) -> None:
    """Guard against truth and model silently becoming the same equations."""
    x = np.array([150.0, 358.0])
    true_derivative = true_model.rhs(0.0, x, u_nominal, true_source)
    model_derivative = modeller_model.rhs(0.0, x, u_nominal, modeller)
    relative_gap = np.abs(true_derivative - model_derivative) / np.abs(true_derivative)
    assert np.all(relative_gap > 0.05)
