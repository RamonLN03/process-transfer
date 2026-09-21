"""Which instants fall on a sampling clock, to the resolution of floating-point arithmetic.

A sampling clock is ``start + k * period`` for k = 0, 1, 2, ... An instant computed by
another route, such as ``start + h * j`` on the finer grid of a simulation, is the same
instant and yet can differ from the clock in its last bits: 0.1 * 3 is
0.30000000000000004. Each route is one product and one sum, each rounded by at most half
a unit in the last place, so the two agree to within a few such units. That is the
resolution of the arithmetic, not a tolerance on the data: at two hours it is of the
order of 1e-12 s, against a spacing of 0.1 s between stored samples, so it cannot make a
wrong sample pass for the right one.

Shared by the truth side, which picks the samples that the sensors read, and by the data
layer, which turns instants into integer ticks. It knows nothing about plants.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from process_transfer.cstr_variables import FloatArray
from process_transfer.validation import require_finite, require_positive

UNITS_IN_THE_LAST_PLACE = 4.0
# Ticks are stored as 64-bit integers and compared as floats on the way; beyond 2**53 a
# float no longer holds every integer. No run comes near it: 2**53 ticks of 6 s are 1.7e9
# years.
LARGEST_TICK = 2**53


def nearest_ticks(
    times: FloatArray, start: float, period: float
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """For every instant, the nearest tick of the clock and whether the instant is that
    tick to within the resolution of the arithmetic.

    ``times`` must be finite. A period so small, or instants so far from ``start``, that
    the tick number cannot be held exactly is an error and not a rounded answer.
    """
    period = require_positive("period", period)
    start = require_finite("start", start)
    instants = np.asarray(times, dtype=np.float64)
    if instants.ndim != 1:
        raise ValueError(f"times must be a vector, got shape {instants.shape}")
    if not np.all(np.isfinite(instants)):
        raise ValueError("times must be finite")

    # finite values can still overflow in the difference or in the ratio
    with np.errstate(over="ignore", invalid="ignore"):
        count = np.rint((instants - start) / period)
    if not np.all(np.isfinite(count)) or (count.size and np.max(np.abs(count)) > LARGEST_TICK):
        raise ValueError(
            f"with a period of {period!r} s the instants are more than {LARGEST_TICK} ticks "
            f"from the start, {start!r} s; a tick number that large cannot be held exactly"
        )
    nearest = start + count * period
    resolution = UNITS_IN_THE_LAST_PLACE * np.spacing(np.maximum(np.abs(instants), np.abs(nearest)))
    return count.astype(np.int64), np.abs(instants - nearest) <= resolution
