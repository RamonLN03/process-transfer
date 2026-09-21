"""The instrument specification accepted in D-020, as configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from process_transfer.config import SensorsConfig, load_sensors, load_true_plant


def sensor(variable: str, value: float, unit: str) -> dict:
    return {
        "variable": variable,
        "form": "additive_gaussian",
        "noise_std": {"value": value, "unit": unit},
    }


VALID = {
    "process_type": "cstr",
    "sampling_period": {"value": 0.1, "unit": "min"},
    "sensors": [sensor("C_A", 5.0, "mol/m^3"), sensor("T", 0.5, "K")],
}


def with_sensor(index: int, **changes: object) -> dict:
    sensors = [dict(sensor) for sensor in VALID["sensors"]]
    sensors[index].update(changes)
    return {**VALID, "sensors": sensors}


def test_the_committed_specification_is_the_one_of_d020(configs_dir: Path) -> None:
    cfg = load_sensors(configs_dir / "sensors_cstr.yaml")
    assert cfg.sampling_period.si == pytest.approx(6.0)  # 0.1 min, in seconds
    by_variable = {sensor.variable: sensor for sensor in cfg.sensors}
    assert set(by_variable) == {"C_A", "T"}

    c_a = by_variable["C_A"].noise_std
    assert (c_a.value, c_a.unit) == (0.005, "mol/L")  # as written, in engineering units
    assert c_a.si == pytest.approx(5.0) and c_a.si_unit == "mol/m^3"
    assert by_variable["T"].noise_std.si == 0.5 and by_variable["T"].noise_std.si_unit == "K"


def test_one_specification_serves_both_plants(configs_dir: Path) -> None:
    """The decision is the same absolute noise on source and target. It is expressed by
    there being a single sensor file, and none of the plant files mentioning sensors."""
    assert sorted(path.name for path in configs_dir.glob("sensors*.yaml")) == ["sensors_cstr.yaml"]
    for name in ("source", "target"):
        plant = load_true_plant(configs_dir / f"{name}_cstr.yaml")
        assert "sensor" not in plant.model_dump_json().lower()
        assert "noise" not in plant.model_dump_json().lower()


def test_the_absolute_noise_differs_from_two_percent_of_the_target_concentration(
    configs_dir: Path,
) -> None:
    """The reading of D-010 that was not chosen: 2 % of each plant's nominal C_A is
    5.0 mol/m^3 on the source but 3.8 mol/m^3 on the target."""
    sigma = {
        s.variable: s.noise_std.si for s in load_sensors(configs_dir / "sensors_cstr.yaml").sensors
    }
    assert sigma["C_A"] == pytest.approx(0.02 * 250.0, rel=1e-3)
    assert sigma["C_A"] != pytest.approx(0.02 * 189.67, rel=0.2)


def test_zero_noise_is_a_valid_limit_and_negative_noise_is_not() -> None:
    exact = SensorsConfig.model_validate(with_sensor(0, noise_std={"value": 0.0, "unit": "mol/L"}))
    assert exact.sensors[0].noise_std.si == 0.0
    for bad in (-0.005, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            SensorsConfig.model_validate(with_sensor(0, noise_std={"value": bad, "unit": "mol/L"}))


@pytest.mark.parametrize(
    ("index", "unit", "message"),
    [
        (0, "K", "the noise of the C_A sensor must be a concentration"),
        (1, "mol/L", "the noise of the T sensor must be a temperature"),
        (1, "degC", "unknown unit"),
    ],
)
def test_the_noise_must_have_the_dimension_of_the_variable(
    index: int, unit: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SensorsConfig.model_validate(with_sensor(index, noise_std={"value": 0.5, "unit": unit}))


def test_a_noise_level_that_is_not_representable_in_si_is_rejected() -> None:
    """The check on the SI value applies here as to any quantity: a finite number of
    mol/L can overflow when it is converted to mol/m^3."""
    with pytest.raises(ValidationError, match="overflows"):
        SensorsConfig.model_validate(with_sensor(0, noise_std={"value": 1e306, "unit": "mol/L"}))


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({**VALID, "sensors": []}, "at least one sensor"),
        ({**VALID, "sensors": [VALID["sensors"][1], VALID["sensors"][1]]}, "one sensor only"),
        ({**VALID, "sampling_period": {"value": 0.0, "unit": "min"}}, "greater than 0"),
        ({**VALID, "sampling_period": {"value": 6.0, "unit": "K"}}, "requires time"),
        ({**VALID, "bias": 0.1}, "Extra inputs are not permitted"),
        (with_sensor(0, variable="q"), "Input should be 'C_A' or 'T'"),
        (with_sensor(0, form="uniform"), "additive_gaussian"),
        (with_sensor(0, bias={"value": 1.0, "unit": "mol/L"}), "Extra inputs are not permitted"),
    ],
)
def test_invalid_specifications_are_rejected(document: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SensorsConfig.model_validate(document)
