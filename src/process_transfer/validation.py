"""Argument checks shared by the numerical modules.

Invalid input is rejected once, at the boundary where it enters, with a message
that names the argument and the value. Nothing here repairs a value: there is no
clipping, no substitution and no tolerance. The right-hand sides of the models
are deliberately left free of such checks; they run millions of times and rely on
the parameters and arguments having been validated before.
"""

from __future__ import annotations

import math


def require_finite(name: str, value: float) -> float:
    """``value`` as a float, or ``ValueError`` if it is NaN or infinite."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def require_positive(name: str, value: float) -> float:
    """``value`` as a float, or ``ValueError`` unless it is finite and greater than zero."""
    number = require_finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}")
    return number


def require_non_negative(name: str, value: float) -> float:
    """``value`` as a float, or ``ValueError`` unless it is finite and not negative.

    Zero is accepted where it is a valid physical limit: no reaction, no saturation,
    an adiabatic reactor."""
    number = require_finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return number
