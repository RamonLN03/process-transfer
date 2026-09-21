"""Time grids for sampling simulated trajectories."""

from __future__ import annotations

import math

import numpy as np

from process_transfer.cstr_variables import FloatArray


def sample_times(duration: float, sample_period: float) -> FloatArray:
    """Sampling instants 0, h, 2h, ... and the final instant ``duration``, each once.

    The final instant is always included, whether or not ``duration`` is a whole
    number of periods, so that the last sample is the state at the end of the
    simulation. Instants are computed as multiples of the period, not by repeated
    addition, to avoid drift. A period longer than the duration gives ``[0, duration]``.
    """
    for name, value in (("duration", duration), ("sample_period", sample_period)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number of seconds, got {value!r}")

    whole_periods = int(math.floor(duration / sample_period))
    times = sample_period * np.arange(whole_periods + 1, dtype=np.float64)
    if duration - times[-1] > 1.0e-9 * sample_period:
        times = np.append(times, duration)
    else:
        times[-1] = duration  # absorb rounding in the last multiple
    return times
