"""Dimensional and finiteness validation of configuration quantities.

Regression tests for a confirmed defect: the configuration accepted any known
unit in any field (a volume in kelvin) and non-finite values (an infinite volume).
"""

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from process_transfer.config import (
    InverseTemperature,
    ModellerConfig,
    MolarEnergy,
    Quantity,
    TruePlantConfig,
    Volume,
    load_yaml,
)
from process_transfer.units import dimension_of, known_units, si_unit_of


def _set(raw: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    patched = copy.deepcopy(raw)
    node = patched
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return patched


@pytest.fixture(scope="module")
def source_raw(configs_dir: Path) -> dict[str, Any]:
    return load_yaml(configs_dir / "source_cstr.yaml")


@pytest.fixture(scope="module")
def modeller_raw(configs_dir: Path) -> dict[str, Any]:
    return load_yaml(configs_dir / "modeller_cstr.yaml")


def test_the_committed_configurations_still_load(
    source_raw: dict[str, Any], modeller_raw: dict[str, Any]
) -> None:
    TruePlantConfig.model_validate(source_raw)
    ModellerConfig.model_validate(modeller_raw)


TRUE_PLANT_WRONG_DIMENSIONS = [
    # field path, a known unit of the wrong dimension, the dimension the field requires
    (("plant", "design", "volume"), "K", "volume"),
    (("plant", "design", "volume"), "L/min", "volume"),
    (("plant", "properties", "density"), "mol/L", "density"),
    (("plant", "properties", "heat_capacity"), "J/mol", "specific_heat_capacity"),
    (("plant", "properties", "reaction_enthalpy"), "K", "molar_energy"),
    (("plant", "nominal_inputs", "feed_flow"), "L", "volumetric_flow"),
    (("plant", "nominal_inputs", "feed_concentration"), "K", "concentration"),
    (("plant", "nominal_inputs", "feed_temperature"), "L", "temperature"),
    (("plant", "nominal_inputs", "coolant_temperature"), "1/K", "temperature"),
    (("true_physics", "kinetics", "k0"), "K", "inverse_time"),
    (("true_physics", "kinetics", "activation_temperature"), "J/mol", "temperature"),
    (("true_physics", "kinetics", "saturation_constant"), "1/K", "inverse_concentration"),
    (("true_physics", "heat_transfer", "UA_ref"), "1/min", "thermal_conductance"),
    (("true_physics", "heat_transfer", "alpha"), "L/mol", "inverse_temperature"),
    (("true_physics", "heat_transfer", "T_ref"), "min", "temperature"),
]


@pytest.mark.parametrize(("path", "wrong_unit", "required"), TRUE_PLANT_WRONG_DIMENSIONS)
def test_true_plant_rejects_a_known_unit_of_the_wrong_dimension(
    path: tuple[str, ...], wrong_unit: str, required: str, source_raw: dict[str, Any]
) -> None:
    patched = _set(source_raw, (*path, "unit"), wrong_unit)
    with pytest.raises(ValidationError, match=f"requires {required}"):
        TruePlantConfig.model_validate(patched)


@pytest.mark.parametrize(
    ("path", "wrong_unit", "required"),
    [
        (("kinetics", "k0"), "W/K", "inverse_time"),
        (("kinetics", "activation_temperature"), "1/K", "temperature"),
        (("heat_transfer", "UA"), "J/mol", "thermal_conductance"),
    ],
)
def test_modeller_rejects_a_known_unit_of_the_wrong_dimension(
    path: tuple[str, ...], wrong_unit: str, required: str, modeller_raw: dict[str, Any]
) -> None:
    patched = _set(modeller_raw, (*path, "unit"), wrong_unit)
    with pytest.raises(ValidationError, match=f"requires {required}"):
        ModellerConfig.model_validate(patched)


def test_an_alternative_unit_of_the_right_dimension_is_accepted(
    source_raw: dict[str, Any],
) -> None:
    patched = _set(source_raw, ("plant", "design", "volume"), {"value": 0.1, "unit": "m^3"})
    cfg = TruePlantConfig.model_validate(patched)
    assert cfg.plant.design.volume.si == pytest.approx(0.1)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize(
    "path",
    [
        ("plant", "design", "volume"),  # positive field
        ("plant", "properties", "reaction_enthalpy"),  # signed field
        ("true_physics", "heat_transfer", "alpha"),  # signed field
        ("true_physics", "kinetics", "k0"),
    ],
)
def test_non_finite_values_are_rejected(
    path: tuple[str, ...], bad: float, source_raw: dict[str, Any]
) -> None:
    patched = _set(source_raw, (*path, "value"), bad)
    with pytest.raises(ValidationError):
        TruePlantConfig.model_validate(patched)


def test_yaml_infinity_is_rejected(tmp_path: Path, source_raw: dict[str, Any]) -> None:
    """The defect as first reported: ``.inf`` written in the YAML file itself."""
    text = (
        "plant:\n  name: bad\n  process_type: cstr\n"
        "  design:\n    volume: {value: .inf, unit: L}\n"
    )
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    raw = load_yaml(path)
    assert raw["plant"]["design"]["volume"]["value"] == float("inf")
    patched = _set(source_raw, ("plant", "design", "volume"), raw["plant"]["design"]["volume"])
    with pytest.raises(ValidationError, match="finite"):
        TruePlantConfig.model_validate(patched)


def test_signed_quantities_accept_both_signs() -> None:
    """D-015: the enthalpy sign is not constrained; the conductance slope is signed too."""
    assert MolarEnergy(value=5.0e4, unit="J/mol").si == 5.0e4
    assert MolarEnergy(value=-5.0e4, unit="J/mol").si == -5.0e4
    assert InverseTemperature(value=-0.002, unit="1/K").si == -0.002


def test_positive_quantities_reject_zero_and_negative_values() -> None:
    for value in (0.0, -1.0):
        with pytest.raises(ValidationError, match="greater than 0"):
            Volume(value=value, unit="L")


def test_the_generic_quantity_accepts_any_known_unit_but_no_unknown_one() -> None:
    assert Quantity(value=1.0, unit="K").si == 1.0
    with pytest.raises(ValidationError, match="unknown unit"):
        Quantity(value=1.0, unit="furlong")


def test_every_unit_has_the_same_dimension_as_its_si_target() -> None:
    for unit in known_units():
        assert dimension_of(unit) == dimension_of(si_unit_of(unit)), unit
