"""Sampling grids and integration of a plant under piecewise-constant inputs.

Inputs change discontinuously in every excitation protocol of this project. An
adaptive integrator that steps across a discontinuity loses accuracy without
saying so, so the integration is restarted at every change of the inputs: one
``solve_ivp`` call per segment, each starting from the final state of the last.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from process_transfer.cstr_variables import FloatArray

RightHandSide = Callable[[FloatArray, FloatArray], FloatArray]  # f(x, u) -> dx/dt


class IntegrationError(RuntimeError):
    """The integrator reported a failure or produced a non-finite state."""


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


@dataclass(frozen=True)
class InputSegment:
    """Inputs u = [q, C_Af, T_f, T_c] held constant for ``duration`` seconds."""

    duration: float
    inputs: FloatArray


@dataclass(frozen=True)
class SegmentTrajectory:
    """Samples of one segment, both of its ends included."""

    times: FloatArray  # absolute time, s
    states: FloatArray  # shape (n_samples, n_states)
    inputs: FloatArray  # the constant input vector of the segment


@dataclass(frozen=True)
class Trajectory:
    """A simulated trajectory, kept segment by segment so that quantities which are
    discontinuous at the input changes can be integrated correctly."""

    segments: tuple[SegmentTrajectory, ...]
    method: str
    rtol: float
    atol: float
    n_rhs_evaluations: int

    @property
    def times(self) -> FloatArray:
        """All sampling instants, each switching instant appearing once."""
        parts = [segment.times[:-1] for segment in self.segments[:-1]]
        return np.concatenate([*parts, self.segments[-1].times])

    @property
    def states(self) -> FloatArray:
        """States at ``times``. The state is continuous across an input change."""
        parts = [segment.states[:-1] for segment in self.segments[:-1]]
        return np.concatenate([*parts, self.segments[-1].states])

    @property
    def inputs(self) -> FloatArray:
        """Inputs at ``times``, right-continuous: the sample taken at a switching
        instant carries the inputs applied from that instant on (zero-order hold)."""
        parts = [
            np.tile(segment.inputs, (len(segment.times) - 1, 1)) for segment in self.segments[:-1]
        ]
        last = self.segments[-1]
        return np.concatenate([*parts, np.tile(last.inputs, (len(last.times), 1))])

    @property
    def duration(self) -> float:
        return float(self.segments[-1].times[-1])

    def state_range(self, index: int) -> tuple[float, float]:
        """Minimum and maximum of one state over the sampled trajectory."""
        values = self.states[:, index]
        return float(values.min()), float(values.max())

    def peak(self, index: int) -> tuple[float, float]:
        """Maximum of one state over the samples and the time at which it occurs."""
        values = self.states[:, index]
        position = int(np.argmax(values))
        return float(values[position]), float(self.times[position])

    def refined_peak(self, index: int) -> tuple[float, float]:
        """Maximum of one state, refined by a parabola through the largest sample and
        its two neighbours inside the same segment.

        Sampling can only underestimate a maximum. The refinement estimates by how
        much; it is not applied when the largest sample sits on a segment end, where
        the derivative is discontinuous and a parabola would be meaningless.
        """
        best = max(self.segments, key=lambda segment: float(segment.states[:, index].max()))
        values, times = best.states[:, index], best.times
        i = int(np.argmax(values))
        if i == 0 or i == len(values) - 1:
            return float(values[i]), float(times[i])
        coefficients = np.polyfit(times[i - 1 : i + 2] - times[i], values[i - 1 : i + 2], 2)
        if coefficients[0] >= 0.0:
            return float(values[i]), float(times[i])
        shift = -coefficients[1] / (2.0 * coefficients[0])
        return float(np.polyval(coefficients, shift)), float(times[i] + shift)


def simulate_piecewise(
    f: RightHandSide,
    x0: FloatArray,
    segments: Sequence[InputSegment],
    sample_period: float = 0.1,
    method: str = "LSODA",
    rtol: float = 1.0e-9,
    atol: float = 1.0e-9,
) -> Trajectory:
    """Integrate ``dx/dt = f(x, u)`` under piecewise-constant inputs.

    The integrator is restarted at the start of every segment, so no step ever
    crosses a discontinuity of the inputs. Each segment is sampled every
    ``sample_period`` seconds and at its own end. Raises :class:`IntegrationError`
    if the integrator fails or a state stops being finite; nothing is clipped.
    """
    if len(segments) == 0:
        raise ValueError("at least one input segment is required")

    state = np.asarray(x0, dtype=np.float64)
    start = 0.0
    n_rhs_evaluations = 0
    simulated: list[SegmentTrajectory] = []
    for index, segment in enumerate(segments):
        local_times = sample_times(segment.duration, sample_period)
        u = np.asarray(segment.inputs, dtype=np.float64)
        solution = solve_ivp(
            lambda t, x, u=u: f(x, u),
            (0.0, segment.duration),
            state,
            method=method,
            t_eval=local_times,
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise IntegrationError(f"segment {index} ({method}): {solution.message}")
        states = solution.y.T
        if not np.all(np.isfinite(states)):
            raise IntegrationError(f"segment {index} ({method}): non-finite state")

        simulated.append(
            SegmentTrajectory(times=start + local_times, states=states, inputs=u.copy())
        )
        state = states[-1].copy()
        start += segment.duration
        n_rhs_evaluations += int(solution.nfev)

    return Trajectory(
        segments=tuple(simulated),
        method=method,
        rtol=rtol,
        atol=atol,
        n_rhs_evaluations=n_rhs_evaluations,
    )
