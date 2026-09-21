"""From a validated true trajectory to observations, the two kept apart.

    true trajectory, finely sampled
        -> acceptance checks of the truth (simulation/checks.py)
        -> exact states at the sensor instants
        -> measurement, which knows no physics (process_transfer.measurement)
        -> Observations, for modelling    and    RunTruth, for diagnostics only

Two sampling periods are involved and must not be confused. The true trajectory is
sampled finely, 0.1 s in M0, because peaks and integrated balances are judged on it.
The sensors sample every 6 s. The sensors read stored samples of the true trajectory:
every sensor instant must coincide with one, and nothing is interpolated. A sensor
period that is not a whole multiple of the simulation period is an error.

Inputs are known without error (D-020), so every change of the inputs must fall on a
sensor instant. Otherwise the sampled inputs would misplace the change by up to one
sampling period, and the record would no longer describe what was applied.

This module is on the truth side: it uses the hidden parameters to validate the
trajectory. What it hands to ``process_transfer.measurement`` is only the values to be
measured and the instrument specification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from process_transfer.cstr_variables import (
    INPUT_NAMES,
    INPUT_UNITS,
    STATE_NAMES,
    STATE_UNITS,
    FloatArray,
)
from process_transfer.measurement.observations import Observations
from process_transfer.measurement.sensors import (
    MeasurementSpec,
    measure,
    noise_key,
    noise_stream,
)
from process_transfer.simulation.checks import TrajectoryCheck, check_trajectory
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.integration import Trajectory
from process_transfer.validation import require_positive

# Two floating-point routes to the same instant, start + period * k, each one product and
# one sum, each rounded by at most half a unit in the last place. They agree to within a
# few such units. This is the resolution of the arithmetic, not a tolerance on the data.
_UNITS_IN_THE_LAST_PLACE = 4.0


class TrajectoryNotAcceptedError(ValueError):
    """The true trajectory failed its acceptance checks; no observation is made of it."""

    def __init__(self, check: TrajectoryCheck) -> None:
        self.check = check
        failed = [
            name
            for name in ("values_finite", "states_physical", "inside_envelope", "balances_close")
            if not getattr(check, name)
        ]
        super().__init__(
            "the true trajectory was not accepted, so it is not turned into observations: "
            f"failed {', '.join(failed)} (peak {check.refined_peak_temperature!r} K)"
        )


def sensor_sample_indices(trajectory: Trajectory, sample_period: float) -> NDArray[np.intp]:
    """Positions in ``trajectory.times`` of the sensor instants t0 + k * sample_period.

    The sensor clock starts at the first instant of the trajectory, which need not be
    zero. Raises ``ValueError`` when a sensor instant is not a stored sample, or when the
    inputs change between two sensor instants. Nothing is interpolated.
    """
    period = require_positive("sample_period", sample_period)
    times = trajectory.times
    start = times[0]

    count = np.rint((times - start) / period)
    nearest = start + count * period
    resolution = _UNITS_IN_THE_LAST_PLACE * np.spacing(np.maximum(np.abs(times), np.abs(nearest)))
    on_the_clock = np.abs(times - nearest) <= resolution
    indices = np.flatnonzero(on_the_clock)
    ticks = count[indices].astype(np.int64)

    if np.any(ticks[1:] == ticks[:-1]):
        repeated = int(ticks[1:][ticks[1:] == ticks[:-1]][0])
        instant = float(start + repeated * period)
        raise ValueError(
            f"two stored samples fall on the sensor instant t = {instant!r} s; they are "
            "closer together than floating-point arithmetic can tell apart"
        )
    expected = np.arange(len(ticks), dtype=np.int64)
    if not np.array_equal(ticks, expected):
        missing = int(np.flatnonzero(ticks != expected)[0])
        raise ValueError(
            f"the sensor instant t = {float(start + missing * period)!r} s is not a stored "
            f"sample of the trajectory. The sensor period, {period!r} s, must be a whole "
            "multiple of the sampling period of the simulation; nothing is interpolated"
        )
    next_tick = start + len(ticks) * period
    if next_tick <= times[-1]:
        raise ValueError(
            f"the sensor instant t = {float(next_tick)!r} s is not a stored sample of the "
            f"trajectory, which ends at {float(times[-1])!r} s; nothing is interpolated"
        )

    sampled = set(indices.tolist())
    position = 0
    for before, after in zip(trajectory.segments[:-1], trajectory.segments[1:], strict=True):
        position += len(before.times) - 1  # index of the switching instant in ``times``
        if not np.array_equal(before.inputs, after.inputs) and position not in sampled:
            raise ValueError(
                f"the inputs change at t = {float(after.times[0])!r} s, between two sensor "
                f"instants {period!r} s apart. The sampled inputs would misplace the change; "
                "hold every input for a whole number of sensor periods"
            )
    return indices


@dataclass(frozen=True)
class RunTruth:
    """Everything about a run that no model may see. For diagnostics and tests only.

    It is never written next to the observations, and it is the only place where the
    noise seed of a run is kept: with the seed the noise can be regenerated and the
    exact states recovered from the readings.
    """

    plant: str
    run: str
    trajectory: Trajectory  # the true trajectory at the resolution of the simulation
    check: TrajectoryCheck  # its acceptance checks, all passed
    sample_indices: NDArray[np.intp]  # sensor instants as positions in trajectory.times
    exact: FloatArray  # exact values of the measured variables at the sensor instants
    errors: FloatArray  # reading minus exact value
    sensor_seed: int
    sensor_stream: tuple[int, ...]


@dataclass(frozen=True)
class OperatingRun:
    """One run of a plant: what may be modelled, and what may only be inspected."""

    observations: Observations
    truth: RunTruth


def state_columns(measurement: MeasurementSpec) -> list[int]:
    """The column of the state vector that each sensor reads, in the order of the sensors.

    This is where a generic sensor meets the states of the CSTR, so this is where the
    two are checked against each other: a sensor must name a state, and its unit must be
    the SI unit in which the simulation holds that state. A subset of the states, in any
    order, is valid. A mismatch is an error. Applying a noise level of 0.005, meant in
    mol/L, to a state held in mol/m^3, and labelling the result mol/m^3, is what this
    prevents; no number is converted or relabelled to make a specification fit.
    """
    columns = []
    for sensor in measurement.sensors:
        if sensor.variable not in STATE_NAMES:
            raise ValueError(
                f"no state is called {sensor.variable!r}; the states are {list(STATE_NAMES)}"
            )
        column = STATE_NAMES.index(sensor.variable)
        if sensor.unit != STATE_UNITS[column]:
            raise ValueError(
                f"the {sensor.variable} sensor is specified in {sensor.unit!r}, but the "
                f"simulation holds {sensor.variable} in {STATE_UNITS[column]!r}. The noise level "
                "must be given in that unit; nothing is converted or relabelled here"
            )
        columns.append(column)
    return columns


def observe_trajectory(
    trajectory: Trajectory,
    parameters: TrueCSTRParameters,
    measurement: MeasurementSpec,
    *,
    plant: str,
    run: str,
    sensor_seed: int,
    sensor_stream: Sequence[int],
) -> OperatingRun:
    """Observations of ``trajectory`` as the sensors of ``measurement`` would give them.

    The true trajectory is validated first; one that is not accepted raises
    :class:`TrajectoryNotAcceptedError` and is not observed. The acceptance criteria of
    true states, physical bounds and closed balances, are applied to the truth only.
    The readings are returned as the sensors gave them, unchecked and uncorrected.

    The seed and the stream are validated as they were given, by the one rule of
    ``measurement.sensors``, before anything is computed and whatever the noise levels
    are. They are never rounded or converted into valid keys.
    """
    seed = noise_key("sensor_seed", sensor_seed)
    stream = noise_stream(sensor_stream)
    columns = state_columns(measurement)

    check = check_trajectory(trajectory, parameters)
    if not check.accepted:
        raise TrajectoryNotAcceptedError(check)

    indices = sensor_sample_indices(trajectory, measurement.sample_period)
    exact = trajectory.states[indices][:, columns]
    # the noise channel of a sensor is the index of its state, not its position among the
    # sensors, so a variable keeps its noise whatever else is measured, in whatever order
    readings = measure(exact, measurement.sensors, seed, stream, channels=columns)

    observations = Observations(
        plant=plant,
        run=run,
        times=trajectory.times[indices],
        measured=readings,
        inputs=trajectory.inputs[indices],
        measured_names=measurement.variables,
        measured_units=tuple(sensor.unit for sensor in measurement.sensors),
        input_names=INPUT_NAMES,
        input_units=INPUT_UNITS,
        sample_period=measurement.sample_period,
        noise_std=measurement.noise_std,
    )
    exact_copy, errors, indices = exact.copy(), readings - exact, indices.copy()
    for values in (exact_copy, errors, indices):
        values.setflags(write=False)
    truth = RunTruth(
        plant=plant,
        run=run,
        trajectory=trajectory,
        check=check,
        sample_indices=indices,
        exact=exact_copy,
        errors=errors,
        sensor_seed=seed,
        sensor_stream=stream,
    )
    return OperatingRun(observations=observations, truth=truth)
