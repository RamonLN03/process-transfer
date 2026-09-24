"""MN is the modeller's model, bit for bit; the parameterisation by k_350, E/R and UA is a
change of coordinates and nothing else; the derivatives agree with finite differences."""

import math
from pathlib import Path

import numpy as np
import pytest

from m1_support import NOMINAL
from process_transfer.config import load_modeller, load_true_plant
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.modeller import cstr_first_order
from process_transfer.modeller.cstr_first_order import ModellerCSTRParameters
from process_transfer.models.mechanistic import (
    REFERENCE_TEMPERATURE,
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
    nominal_model,
)
from process_transfer.simulation.steady_state import find_steady_states

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, NOMINAL)
POINTS = [
    (np.array([189.67, 355.17]), NOMINAL),
    (np.array([67.0, 376.2]), NOMINAL * np.array([1.1, 1.1, 1.0, 1.0]) + [0, 0, 5.0, 5.0]),
    (np.array([347.0, 342.4]), NOMINAL * np.array([0.9, 0.9, 1.0, 1.0]) - [0, 0, 5.0, 5.0]),
    (np.array([250.0, 337.5]), NOMINAL),  # T = T_c
]


def test_mn_is_the_modeller_model_with_the_textbook_values(configs_dir: Path) -> None:
    spec = load_true_plant(configs_dir / "target_cstr.yaml").plant
    modeller = load_modeller(configs_dir / "modeller_cstr.yaml")
    reference = ModellerCSTRParameters.from_config(spec, modeller)
    known = KnownPlant(
        "target",
        spec.design.volume.si,
        spec.properties.density.si,
        spec.properties.heat_capacity.si,
        spec.properties.reaction_enthalpy.si,
        NOMINAL,
    )
    mn = nominal_model(known, modeller_values())
    assert mn.name == "MN"
    assert mn.modeller_parameters == reference
    for x, u in POINTS:
        np.testing.assert_array_equal(mn.rhs(x, u), cstr_first_order.rhs(0.0, x, u, reference))


def test_the_textbook_rate_constant_at_350_k() -> None:
    textbook = MechanisticParameters.textbook(modeller_values())
    assert textbook.k0 == 7.2e10 / 60.0
    assert textbook.activation_temperature == 8750.0
    assert textbook.ua == pytest.approx(1.0e5 / 60.0, rel=1e-15)
    # k0 exp(-(E/R) / 350 K), about 1 / 60 1/s
    assert textbook.k_350 == pytest.approx(7.2e10 / 60.0 * math.exp(-25.0), rel=1e-14)
    assert textbook.k_350 == pytest.approx(0.016665532637956827, rel=1e-14)


def test_mn_puts_the_target_steady_state_where_the_source_is() -> None:
    """Section 2 of the plan: the textbook model's one steady state at the nominal inputs is
    at 250.01 mol/m^3 and 350.00 K."""
    mn = nominal_model(KNOWN, modeller_values())
    (steady,) = find_steady_states(lambda x: mn.rhs(x, NOMINAL), c_a_upper=NOMINAL[1])
    assert steady.state == pytest.approx([250.01, 350.00], abs=0.005)


def test_the_coordinates_are_a_change_of_units() -> None:
    parameters = MechanisticParameters.from_k_350(0.0217, 9200.0, 1416.7)
    theta = parameters.coordinates(None)
    assert theta[1] == 9200.0 / REFERENCE_TEMPERATURE
    back = MechanisticParameters.from_coordinates(theta, None)
    assert back.k0 == pytest.approx(parameters.k0, rel=1e-14)
    assert back.activation_temperature == pytest.approx(9200.0, rel=1e-15)
    assert back.ua == pytest.approx(1416.7, rel=1e-15)
    assert back.k_350 == pytest.approx(0.0217, rel=1e-14)


def test_with_e_over_r_fixed_there_are_two_coordinates() -> None:
    parameters = MechanisticParameters.from_k_350(0.02, 8750.0, 1500.0)
    theta = parameters.coordinates(8750.0)
    assert theta.shape == (2,)
    back = MechanisticParameters.from_coordinates(theta, 8750.0)
    assert back.activation_temperature == 8750.0
    assert back.k_350 == pytest.approx(0.02, rel=1e-14)
    with pytest.raises(ValueError, match="fixed at"):
        parameters.coordinates(9000.0)
    model = MechanisticModel("MR", KNOWN, back, fixed_activation=8750.0)
    assert model.parameter_names == ("ln_k_350", "ln_ua")
    assert model.jacobian_parameters(*POINTS[0]).shape == (2, 2)


def test_parameters_outside_the_domain_are_refused_not_clipped() -> None:
    with pytest.raises(OverflowError):
        MechanisticParameters.from_coordinates(np.array([700.0, 25.0, 7.0]), None)
    with pytest.raises(ValueError, match="activation_temperature"):
        MechanisticParameters(1.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="logarithms"):
        MechanisticParameters(0.0, 8750.0, 1.0).coordinates(None)


def test_the_model_cannot_start_at_a_temperature_that_is_not_positive() -> None:
    mn = nominal_model(KNOWN, modeller_values())
    assert mn.initial_state_problem(np.array([-3.0, 350.0])) is None  # a noisy estimate
    assert "divides by the temperature" in mn.initial_state_problem(np.array([200.0, 0.0]))


@pytest.mark.parametrize("fixed", [None, 8750.0])
def test_the_derivatives_agree_with_finite_differences(fixed: float | None) -> None:
    parameters = MechanisticParameters.from_k_350(0.0217, 8750.0, 1416.7)
    model = MechanisticModel("MR", KNOWN, parameters, fixed)
    theta = parameters.coordinates(fixed)
    for x, u in POINTS:
        numeric = np.empty((2, 2))
        for j in range(2):
            step = 1e-6 * abs(x[j])
            up, down = x.copy(), x.copy()
            up[j] += step
            down[j] -= step
            numeric[:, j] = (model.rhs(up, u) - model.rhs(down, u)) / (2 * step)
        np.testing.assert_allclose(model.jacobian_state(x, u), numeric, rtol=1e-6, atol=1e-12)
        numeric = np.empty((2, len(theta)))
        for j in range(len(theta)):
            up, down = theta.copy(), theta.copy()
            up[j] += 1e-6
            down[j] -= 1e-6
            f_up = MechanisticModel(
                "MR", KNOWN, MechanisticParameters.from_coordinates(up, fixed), fixed
            ).rhs(x, u)
            f_down = MechanisticModel(
                "MR", KNOWN, MechanisticParameters.from_coordinates(down, fixed), fixed
            ).rhs(x, u)
            numeric[:, j] = (f_up - f_down) / 2e-6
        np.testing.assert_allclose(model.jacobian_parameters(x, u), numeric, rtol=1e-6, atol=1e-12)
