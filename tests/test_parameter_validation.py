"""Model parameters are validated once, at construction, not inside the right-hand side.

Regression tests: parameters built directly, bypassing the configuration schema, used
to accept a zero volume (the right-hand side then returned inf and NaN) and a negative
rate constant (which silently reversed the reaction).
"""

import dataclasses

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.modeller.cstr_first_order import ModellerCSTRParameters
from process_transfer.simulation import cstr_true

MODELLER = dict(
    volume=0.1,
    density=1000.0,
    heat_capacity=239.0,
    reaction_enthalpy=-5.0e4,
    k0=1.2e9,
    activation_temperature=8750.0,
    ua=1666.7,
)


@pytest.mark.parametrize("name", ["volume", "density", "heat_capacity"])
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_quantities_that_divide_must_be_positive_and_finite(
    name: str, bad: float, true_plants: dict[str, PlantUnderTest]
) -> None:
    with pytest.raises(ValueError, match=name):
        dataclasses.replace(true_plants["source"].parameters, **{name: bad})
    with pytest.raises(ValueError, match=name):
        ModellerCSTRParameters(**{**MODELLER, name: bad})


@pytest.mark.parametrize("name", ["k0", "activation_temperature", "saturation_constant", "ua_ref"])
def test_negative_physical_constants_are_rejected_but_zero_is_a_valid_limit(
    name: str, true_plants: dict[str, PlantUnderTest]
) -> None:
    parameters = true_plants["source"].parameters
    with pytest.raises(ValueError, match="must not be negative"):
        dataclasses.replace(parameters, **{name: -1.0})
    limit = dataclasses.replace(parameters, **{name: 0.0})
    derivative = cstr_true.rhs(
        0.0, np.array([250.0, 350.0]), true_plants["source"].nominal_inputs, limit
    )
    assert np.all(np.isfinite(derivative))


@pytest.mark.parametrize("name", ["reaction_enthalpy", "alpha", "t_ref"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_signed_parameters_must_still_be_finite(
    name: str, bad: float, true_plants: dict[str, PlantUnderTest]
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        dataclasses.replace(true_plants["source"].parameters, **{name: bad})


def test_either_sign_of_the_enthalpy_is_accepted(true_plants: dict[str, PlantUnderTest]) -> None:
    """D-015 holds for the parameter objects as it does for the schema."""
    for enthalpy in (-5.0e4, 0.0, 5.0e4):
        p = dataclasses.replace(true_plants["source"].parameters, reaction_enthalpy=enthalpy)
        assert np.sign(p.adiabatic_coefficient) == -np.sign(enthalpy)


def test_finite_factors_with_a_useless_product_are_rejected() -> None:
    """Finite inputs do not guarantee a finite, non-zero result: the thermal mass
    V rho cp can overflow or underflow even though each factor is acceptable."""
    with pytest.raises(ValueError, match=r"volume \* density \* heat_capacity"):
        ModellerCSTRParameters(**{**MODELLER, "volume": 1e200, "density": 1e200})
    with pytest.raises(ValueError, match=r"volume \* density \* heat_capacity"):
        ModellerCSTRParameters(**{**MODELLER, "volume": 1e-200, "density": 1e-200})
    with pytest.raises(ValueError, match="reaction_enthalpy /"):
        ModellerCSTRParameters(
            **{**MODELLER, "reaction_enthalpy": 1e300, "density": 1e-150, "heat_capacity": 1e-150}
        )


def test_the_committed_plants_pass(true_plants: dict[str, PlantUnderTest]) -> None:
    for plant in true_plants.values():
        assert plant.parameters.thermal_mass == pytest.approx(0.1 * 1000.0 * 239.0)
        assert plant.parameters.adiabatic_coefficient == pytest.approx(5.0e4 / (1000.0 * 239.0))
