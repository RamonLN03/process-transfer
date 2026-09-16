"""Configuration loading, validation and the ground-truth boundary."""

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from process_transfer.config import (
    ModellerConfig,
    Quantity,
    TruePlantConfig,
    load_modeller,
    load_true_plant,
    load_yaml,
)


def test_source_plant_loads_with_expected_si_values(configs_dir: Path) -> None:
    cfg = load_true_plant(configs_dir / "source_cstr.yaml")
    plant = cfg.plant
    assert plant.name == "source"
    assert plant.process_type == "cstr"
    assert math.isclose(plant.design.volume.si, 0.1)  # 100 L
    assert math.isclose(plant.nominal_inputs.feed_flow.si, 100e-3 / 60)  # 100 L/min
    assert math.isclose(plant.nominal_inputs.feed_concentration.si, 500.0)  # 0.5 mol/L
    assert math.isclose(plant.properties.heat_capacity.si, 239.0)  # 0.239 J/(g K)
    assert math.isclose(cfg.true_physics.kinetics.k0.si, 1.44e11 / 60)
    assert math.isclose(cfg.true_physics.kinetics.saturation_constant.si, 4.0e-3)
    assert math.isclose(cfg.true_physics.heat_transfer.UA_ref.si, 1.0e5 / 60)
    assert cfg.true_physics.heat_transfer.UA_ref.si_unit == "W/K"


def test_yaml_numbers_are_floats_not_strings(configs_dir: Path) -> None:
    """PyYAML only parses exponents with an explicit sign; guard the config files."""
    raw = load_yaml(configs_dir / "source_cstr.yaml")
    assert isinstance(raw["true_physics"]["kinetics"]["k0"]["value"], float)
    assert isinstance(raw["plant"]["properties"]["reaction_enthalpy"]["value"], float)


def test_target_differs_from_source_only_in_hidden_heat_transfer(configs_dir: Path) -> None:
    """D-008: in M0 the target plant differs only in hidden heat-transfer behaviour."""
    source = load_true_plant(configs_dir / "source_cstr.yaml")
    target = load_true_plant(configs_dir / "target_cstr.yaml")
    assert source.plant.model_dump(exclude={"name"}) == target.plant.model_dump(exclude={"name"})
    assert source.true_physics.kinetics == target.true_physics.kinetics
    assert source.true_physics.heat_transfer != target.true_physics.heat_transfer


def test_nominal_rates_of_true_and_modeller_kinetics_coincide(configs_dir: Path) -> None:
    """D-006: at the nominal point (50 % conversion) the saturating law and the
    modeller's first-order law give the same rate, so a modeller fitting there
    recovers the textbook parameters and is wrong away from that point."""
    true = load_true_plant(configs_dir / "source_cstr.yaml")
    modeller = load_modeller(configs_dir / "modeller_cstr.yaml")
    temperature = true.plant.nominal_inputs.feed_temperature.si  # 350 K by design
    c_a = 0.5 * true.plant.nominal_inputs.feed_concentration.si  # 50 % conversion

    kin = true.true_physics.kinetics
    k_true = kin.k0.si * math.exp(-kin.activation_temperature.si / temperature)
    r_true = k_true * c_a / (1.0 + kin.saturation_constant.si * c_a)

    mod = modeller.kinetics
    k_model = mod.k0.si * math.exp(-mod.activation_temperature.si / temperature)
    r_model = k_model * c_a

    assert math.isclose(r_true, r_model, rel_tol=1e-12)
    assert math.isclose(kin.saturation_constant.si * c_a, 1.0, rel_tol=1e-12)


def test_modeller_config_contains_nothing_hidden(configs_dir: Path) -> None:
    modeller = load_modeller(configs_dir / "modeller_cstr.yaml")
    dumped = modeller.model_dump()
    assert "saturation_constant" not in str(dumped)
    assert "alpha" not in str(dumped)
    assert modeller.kinetics.form == "first_order"
    assert modeller.heat_transfer.form == "constant"


def test_true_plant_file_cannot_be_loaded_as_modeller_knowledge(configs_dir: Path) -> None:
    """The strict schemas are the ground-truth boundary: hidden physics never fits
    into the modeller's model, and the modeller's file never passes as truth."""
    with pytest.raises(ValidationError):
        ModellerConfig.model_validate(load_yaml(configs_dir / "source_cstr.yaml"))
    with pytest.raises(ValidationError):
        TruePlantConfig.model_validate(load_yaml(configs_dir / "modeller_cstr.yaml"))


def test_unknown_unit_in_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown unit"):
        Quantity(value=25.0, unit="degC")


def test_negative_volume_is_rejected(configs_dir: Path) -> None:
    raw = load_yaml(configs_dir / "source_cstr.yaml")
    raw["plant"]["design"]["volume"]["value"] = -100.0
    with pytest.raises(ValidationError, match="greater than 0"):
        TruePlantConfig.model_validate(raw)


def test_unknown_field_is_rejected(configs_dir: Path) -> None:
    raw = load_yaml(configs_dir / "source_cstr.yaml")
    raw["plant"]["design"]["diameter"] = {"value": 0.5, "unit": "m"}
    with pytest.raises(ValidationError, match="diameter"):
        TruePlantConfig.model_validate(raw)


def test_configuration_is_immutable(configs_dir: Path) -> None:
    cfg = load_true_plant(configs_dir / "source_cstr.yaml")
    with pytest.raises(ValidationError):
        cfg.plant.design.volume.value = 200.0  # type: ignore[misc]
