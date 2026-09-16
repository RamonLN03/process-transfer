"""Explicit unit conversion at the configuration and data boundaries.

Configuration files are written in engineering units (litres, minutes, mol/L).
Everything inside the simulator is a plain SI float. This module is the only
place where unit strings are interpreted, and it rejects anything it does not
know rather than guessing. Temperatures are absolute (kelvin) only, so every
conversion is a pure scale factor with no offset.
"""

from __future__ import annotations

# unit string as written in configuration -> (factor to SI, SI unit string)
_TO_SI: dict[str, tuple[float, str]] = {
    # volume
    "L": (1e-3, "m^3"),
    "m^3": (1.0, "m^3"),
    # volumetric flow
    "L/min": (1e-3 / 60.0, "m^3/s"),
    "m^3/s": (1.0, "m^3/s"),
    # concentration
    "mol/L": (1e3, "mol/m^3"),
    "mol/m^3": (1.0, "mol/m^3"),
    # absolute temperature
    "K": (1.0, "K"),
    # density
    "g/L": (1.0, "kg/m^3"),
    "kg/m^3": (1.0, "kg/m^3"),
    # specific heat capacity
    "J/(g*K)": (1e3, "J/(kg*K)"),
    "J/(kg*K)": (1.0, "J/(kg*K)"),
    # molar energy
    "J/mol": (1.0, "J/mol"),
    "kJ/mol": (1e3, "J/mol"),
    # first-order rate constant and pre-exponential factor
    "1/min": (1.0 / 60.0, "1/s"),
    "1/s": (1.0, "1/s"),
    # heat-transfer conductance UA
    "J/(min*K)": (1.0 / 60.0, "W/K"),
    "W/K": (1.0, "W/K"),
    # inverse concentration (saturation constant)
    "L/mol": (1e-3, "m^3/mol"),
    "m^3/mol": (1.0, "m^3/mol"),
    # inverse temperature
    "1/K": (1.0, "1/K"),
    # time
    "s": (1.0, "s"),
    "min": (60.0, "s"),
    "h": (3600.0, "s"),
    # dimensionless
    "-": (1.0, "-"),
}


class UnknownUnitError(ValueError):
    """Raised for a unit string that the conversion table does not contain."""


def to_si(value: float, unit: str) -> tuple[float, str]:
    """Convert ``value`` expressed in ``unit`` to SI.

    Returns the SI value and the SI unit string. Unknown units raise
    :class:`UnknownUnitError`; nothing is ever silently passed through.
    """
    try:
        factor, si_unit = _TO_SI[unit]
    except KeyError:
        known = ", ".join(sorted(_TO_SI))
        raise UnknownUnitError(f"unknown unit {unit!r}; known units: {known}") from None
    return value * factor, si_unit


def si_unit_of(unit: str) -> str:
    """Return the SI unit string that ``unit`` converts to."""
    return to_si(0.0, unit)[1]


def known_units() -> tuple[str, ...]:
    """All unit strings the table accepts, sorted."""
    return tuple(sorted(_TO_SI))
