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
from process_transfer.validation import require_positive

RightHandSide = Callable[[FloatArray, FloatArray], FloatArray]  # f(x, u) -> dx/dt


# A guard against a mistyped period, not a physical limit: ten million samples of two
# states are 160 MB. A request beyond it is refused with a message, instead of ending
# in a MemoryError or an OverflowError from deep inside numpy.
MAX_SAMPLES_PER_SEGMENT = 10_000_000


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

    ratio = duration / sample_period  # finite inputs can still overflow here
    if not math.isfinite(ratio) or ratio > MAX_SAMPLES_PER_SEGMENT:
        raise ValueError(
            f"duration {duration!r} s with sample_period {sample_period!r} s would give "
            f"about {ratio:.3g} samples; the limit is {MAX_SAMPLES_PER_SEGMENT}. Use a "
            "longer sample_period or split the duration."
        )
    whole_periods = int(math.floor(ratio))
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


def _read_only_copy(values: FloatArray) -> FloatArray:
    """A float64 copy that cannot be written to. What is validated when a trajectory is
    built is then what every later reader sees: neither the caller's array nor an
    assignment in place can change it afterwards."""
    copy = np.array(values, dtype=np.float64)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class SegmentTrajectory:
    """Samples of one segment, both of its ends included.

    The sampling instants are finite and strictly increasing, there is one row of
    states per instant, and the inputs are one constant vector. Anything else is
    rejected here, where the samples enter. The arrays are stored as read-only copies.
    """

    times: FloatArray  # absolute time, s
    states: FloatArray  # shape (n_samples, n_states)
    inputs: FloatArray  # the constant input vector of the segment

    def __post_init__(self) -> None:
        times, states, inputs = (
            _read_only_copy(values) for values in (self.times, self.states, self.inputs)
        )
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("a segment needs at least its two end samples")
        if states.ndim != 2 or len(states) != len(times):
            raise ValueError(
                f"states must have one row per sample: {states.shape} against "
                f"{len(times)} sampling instants"
            )
        if inputs.ndim != 1:
            raise ValueError(
                f"inputs must be the one constant vector of the segment, got shape {inputs.shape}"
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("sampling instants must be finite")
        not_increasing = np.flatnonzero(times[1:] <= times[:-1])
        if not_increasing.size:
            first = int(not_increasing[0])
            raise ValueError(
                "sampling instants must be strictly increasing: sample "
                f"{first + 1} is at {float(times[first + 1])!r} s, after {float(times[first])!r} s"
            )
        for name, values in (("times", times), ("states", states), ("inputs", inputs)):
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class Trajectory:
    """A simulated trajectory, kept segment by segment so that quantities which are
    discontinuous at the input changes can be integrated correctly.

    Structure, checked when the trajectory is built. Consecutive segments share their
    switching instant: the next one starts at the instant and in the state at which the
    previous one ended, so there is no gap, no overlap and no jump of the state. The
    inputs may change discontinuously there, and usually do. A trajectory need not start
    at t = 0, so a run of consecutive segments cut out of a longer one is valid.

    The comparison at a junction is exact, not within a tolerance. The switching instant
    is one instant and the state there is one state, stored twice; ``times`` and
    ``states`` keep a single copy, which is only right if the two are the same number.
    ``simulate_piecewise`` stores them identically by construction. A tolerance would
    need an arbitrary scale, and would mean choosing between two values without saying
    so. A junction that differs, even in the last bit, is rejected; it is never repaired,
    interpolated or smoothed. Pieces computed separately are joined correctly by
    starting each from the final state of the previous one.
    """

    segments: tuple[SegmentTrajectory, ...]
    method: str
    rtol: float
    atol: float
    n_rhs_evaluations: int

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if len(segments) == 0:
            raise ValueError("a trajectory needs at least one segment")
        for index, (before, after) in enumerate(zip(segments[:-1], segments[1:], strict=True)):
            pair = f"segments {index} and {index + 1}"
            if (
                before.states.shape[1] != after.states.shape[1]
                or before.inputs.shape != after.inputs.shape
            ):
                raise ValueError(
                    f"{pair} differ in the number of states or of inputs: states "
                    f"{before.states.shape[1]} and {after.states.shape[1]}, inputs "
                    f"{before.inputs.shape[0]} and {after.inputs.shape[0]}"
                )
            end, start = float(before.times[-1]), float(after.times[0])
            if start != end:
                kind = "a gap" if start > end else "an overlap"
                raise ValueError(
                    f"{pair} leave {kind} in time: one ends at {end!r} s and the next "
                    f"starts at {start!r} s"
                )
            if not np.array_equal(after.states[0], before.states[-1]):
                raise ValueError(
                    f"the state jumps between {pair}, at t = {end!r} s: from "
                    f"{before.states[-1].tolist()} to {after.states[0].tolist()}. A step in "
                    "the inputs does not move the states instantaneously; the junction is "
                    "rejected, not repaired"
                )
        object.__setattr__(self, "segments", segments)

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
        """Seconds between the first and the last sample. It equals the final instant
        only for a trajectory that starts at t = 0."""
        return float(self.segments[-1].times[-1] - self.segments[0].times[0])

    def state_range(self, index: int) -> tuple[float, float]:
        """Minimum and maximum of one state over the sampled trajectory."""
        values = self.states[:, index]
        return float(values.min()), float(values.max())

    def _finite_column(self, index: int) -> None:
        if not np.all(np.isfinite(self.states[:, index])) or not np.all(np.isfinite(self.times)):
            raise ValueError("the trajectory contains non-finite times or states")

    def peak(self, index: int) -> tuple[float, float]:
        """Maximum of one state over the samples and the time at which it occurs."""
        self._finite_column(index)
        values = self.states[:, index]
        position = int(np.argmax(values))
        return float(values[position]), float(self.times[position])

    def refined_peak(self, index: int) -> tuple[float, float]:
        """Estimate of the maximum of one state between samples, and its time.

        Every segment is examined, and inside it every run of three consecutive
        samples: where the parabola through them is concave and its vertex falls in
        the part of the segment that the triple is responsible for, the vertex is a
        candidate. The result is the largest candidate, or the largest sample when no
        candidate exceeds it. No parabola is ever fitted across a change of inputs,
        where the derivative is discontinuous; the segment ends are samples already.

        This is an estimate, not a bound. It is never smaller than the largest
        sample, but a peak much narrower than the sampling period leaves no trace in
        the samples and cannot be recovered. Critical cases must also be recomputed
        with a finer sampling period.
        """
        self._finite_column(index)
        best = self.peak(index)
        for segment in self.segments:
            candidate = _largest_parabolic_vertex(segment.times, segment.states[:, index])
            if candidate is not None and candidate[0] > best[0]:
                best = candidate
        return best


def _largest_parabolic_vertex(times: FloatArray, values: FloatArray) -> tuple[float, float] | None:
    """Largest valid vertex among the parabolas through consecutive sample triples.

    A triple centred on sample i answers for the interval from the midpoint with
    its left neighbour to the midpoint with its right neighbour; the first and last
    triples also answer for the rest of the segment up to its ends. A vertex outside
    that interval is ignored, because a parabola is only trustworthy near its centre
    and the neighbouring triple covers the rest. Works for unevenly spaced samples.
    Returns ``None`` when there is nothing to refine: fewer than three samples,
    repeated sampling instants, flat or convex data (plateaus included).
    """
    if len(values) < 3:
        return None
    t0, t1, t2 = times[:-2], times[1:-1], times[2:]
    y0, y1, y2 = values[:-2], values[1:-1], values[2:]

    lower, upper = 0.5 * (t0 + t1), 0.5 * (t1 + t2)
    lower[0], upper[-1] = t0[0], t2[-1]

    spaced = (t1 > t0) & (t2 > t1)  # a parabola needs three distinct instants
    if not np.any(spaced):
        return None
    t0, t1, t2, y0, y1, y2 = (a[spaced] for a in (t0, t1, t2, y0, y1, y2))
    lower, upper = lower[spaced], upper[spaced]

    slope_left = (y1 - y0) / (t1 - t0)
    slope_right = (y2 - y1) / (t2 - t1)
    curvature = (slope_right - slope_left) / (t2 - t0)  # leading coefficient

    concave = curvature < 0.0  # flat and convex triples have no interior maximum
    if not np.any(concave):
        return None
    t0, t1, y0, slope_left, curvature = (a[concave] for a in (t0, t1, y0, slope_left, curvature))
    lower, upper = lower[concave], upper[concave]

    vertex_time = 0.5 * (t0 + t1) - slope_left / (2.0 * curvature)
    inside = (vertex_time >= lower) & (vertex_time <= upper)
    if not np.any(inside):
        return None
    t0, t1, y0, slope_left, curvature, vertex_time = (
        a[inside] for a in (t0, t1, y0, slope_left, curvature, vertex_time)
    )
    vertex_value = (
        y0 + slope_left * (vertex_time - t0) + curvature * (vertex_time - t0) * (vertex_time - t1)
    )
    position = int(np.argmax(vertex_value))
    return float(vertex_value[position]), float(vertex_time[position])


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

    # scipy replaces a tolerance it finds too small and only warns; reject it here.
    rtol = require_positive("rtol", rtol)
    atol = require_positive("atol", atol)
    state = np.asarray(x0, dtype=np.float64)
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError(f"x0 must be a finite vector, got {x0!r}")
    start = 0.0
    n_rhs_evaluations = 0
    simulated: list[SegmentTrajectory] = []
    for index, segment in enumerate(segments):
        local_times = sample_times(segment.duration, sample_period)
        u = np.asarray(segment.inputs, dtype=np.float64)
        if not np.all(np.isfinite(u)):
            raise ValueError(f"segment {index} has non-finite inputs: {u!r}")
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
        states = solution.y.T.copy()
        # The first sample of a segment is its initial condition, by definition. Taken
        # from the solver's interpolant at t = 0 it can differ from it in the last bit,
        # which would store the switching instant twice with two different states.
        states[0] = state
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
