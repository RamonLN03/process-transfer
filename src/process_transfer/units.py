"""Explicit unit conversion at the configuration and data boundaries.

Configuration files are written in engineering units (litres, minutes, mol/L).
Everything inside the simulator is a plain SI float. This module is the only
place where unit strings are interpreted, and it rejects anything it does not
know rather than guessing. Temperatures are absolute (kelvin) only, so every
conversion is a pure scale factor with no offset.

Each unit also carries the physical dimension it measures, so that a
configuration field can refuse a known unit of the wrong kind (a volume given
in kelvin, for instance).
"""

from __future__ import annotations

from typing import NamedTuple


class UnitDefinition(NamedTuple):
    factor: float  # multiply by this to obtain the SI value
    si_unit: str
    dimension: str


_UNITS: dict[str, UnitDefinition] = {
    # volume
    "L": UnitDefinition(1e-3, "m^3", "volume"),
    "m^3": UnitDefinition(1.0, "m^3", "volume"),
    # volumetric flow
    "L/min": UnitDefinition(1e-3 / 60.0, "m^3/s", "volumetric_flow"),
    "m^3/s": UnitDefinition(1.0, "m^3/s", "volumetric_flow"),
    # concentration
    "mol/L": UnitDefinition(1e3, "mol/m^3", "concentration"),
    "mol/m^3": UnitDefinition(1.0, "mol/m^3", "concentration"),
    # absolute temperature
    "K": UnitDefinition(1.0, "K", "temperature"),
    # density
    "g/L": UnitDefinition(1.0, "kg/m^3", "density"),
    "kg/m^3": UnitDefinition(1.0, "kg/m^3", "density"),
    # specific heat capacity
    "J/(g*K)": UnitDefinition(1e3, "J/(kg*K)", "specific_heat_capacity"),
    "J/(kg*K)": UnitDefinition(1.0, "J/(kg*K)", "specific_heat_capacity"),
    # molar energy
    "J/mol": UnitDefinition(1.0, "J/mol", "molar_energy"),
    "kJ/mol": UnitDefinition(1e3, "J/mol", "molar_energy"),
    # first-order rate constant and pre-exponential factor
    "1/min": UnitDefinition(1.0 / 60.0, "1/s", "inverse_time"),
    "1/s": UnitDefinition(1.0, "1/s", "inverse_time"),
    # heat-transfer conductance UA
    "J/(min*K)": UnitDefinition(1.0 / 60.0, "W/K", "thermal_conductance"),
    "W/K": UnitDefinition(1.0, "W/K", "thermal_conductance"),
    # inverse concentration (saturation constant)
    "L/mol": UnitDefinition(1e-3, "m^3/mol", "inverse_concentration"),
    "m^3/mol": UnitDefinition(1.0, "m^3/mol", "inverse_concentration"),
    # inverse temperature
    "1/K": UnitDefinition(1.0, "1/K", "inverse_temperature"),
    # time
    "s": UnitDefinition(1.0, "s", "time"),
    "min": UnitDefinition(60.0, "s", "time"),
    "h": UnitDefinition(3600.0, "s", "time"),
    # dimensionless
    "-": UnitDefinition(1.0, "-", "dimensionless"),
}


class UnknownUnitError(ValueError):
    """Raised for a unit string that the conversion table does not contain."""


def _definition(unit: str) -> UnitDefinition:
    try:
        return _UNITS[unit]
    except KeyError:
        known = ", ".join(sorted(_UNITS))
        raise UnknownUnitError(f"unknown unit {unit!r}; known units: {known}") from None


def to_si(value: float, unit: str) -> tuple[float, str]:
    """Convert ``value`` expressed in ``unit`` to SI.

    Returns the SI value and the SI unit string. Unknown units raise
    :class:`UnknownUnitError`; nothing is ever silently passed through.
    """
    definition = _definition(unit)
    return value * definition.factor, definition.si_unit


def si_unit_of(unit: str) -> str:
    """Return the SI unit string that ``unit`` converts to."""
    return _definition(unit).si_unit


def dimension_of(unit: str) -> str:
    """Return the physical dimension that ``unit`` measures, for example ``"volume"``."""
    return _definition(unit).dimension


def known_units() -> tuple[str, ...]:
    """All unit strings the table accepts, sorted."""
    return tuple(sorted(_UNITS))
