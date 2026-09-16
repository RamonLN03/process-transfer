"""Unit conversion at the configuration boundary."""

import math

import pytest

from process_transfer.units import UnknownUnitError, known_units, si_unit_of, to_si


@pytest.mark.parametrize(
    ("value", "unit", "expected_si", "expected_unit"),
    [
        (100.0, "L", 0.1, "m^3"),
        (100.0, "L/min", 100.0e-3 / 60.0, "m^3/s"),
        (0.5, "mol/L", 500.0, "mol/m^3"),
        (350.0, "K", 350.0, "K"),
        (1000.0, "g/L", 1000.0, "kg/m^3"),
        (0.239, "J/(g*K)", 239.0, "J/(kg*K)"),
        (-5.0e4, "J/mol", -5.0e4, "J/mol"),
        (7.2e10, "1/min", 1.2e9, "1/s"),
        (1.0e5, "J/(min*K)", 1.0e5 / 60.0, "W/K"),
        (4.0, "L/mol", 4.0e-3, "m^3/mol"),
        (0.005, "1/K", 0.005, "1/K"),
        (2.0, "h", 7200.0, "s"),
        (0.1, "min", 6.0, "s"),
    ],
)
def test_engineering_units_convert_to_si(
    value: float, unit: str, expected_si: float, expected_unit: str
) -> None:
    si_value, si_unit = to_si(value, unit)
    assert math.isclose(si_value, expected_si, rel_tol=1e-12)
    assert si_unit == expected_unit


def test_unknown_unit_is_an_error_not_a_passthrough() -> None:
    with pytest.raises(UnknownUnitError, match="degC"):
        to_si(25.0, "degC")


def test_every_si_target_is_itself_in_the_table_with_unit_factor() -> None:
    """The SI unit that each entry converts to must be accepted as an identity."""
    for unit in known_units():
        si_unit = si_unit_of(unit)
        value, roundtrip_unit = to_si(1.0, si_unit)
        assert value == 1.0, si_unit
        assert roundtrip_unit == si_unit
