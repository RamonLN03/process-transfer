"""From a validated P3 trajectory to observations: grid, alignment, separation of truth
and observation, and reproducibility from explicit seeds.

The seeds below are fixed and arbitrary, and deliberately not those of M0-E04, so that
running the tests says nothing about the realisation that experiment registers.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.config import load_sensors
from process_transfer.measurement.noise_statistics import correlation, noise_statistics
from process_transfer.measurement.observations import Observations
from process_transfer.measurement.sensors import MeasurementSpec, SensorSpec, measure
from process_transfer.simulation.integration import (
    InputSegment,
    SegmentTrajectory,
    Trajectory,
    simulate_piecewise,
)
from process_transfer.simulation.operating_run import (
    OperatingRun,
    RunTruth,
    TrajectoryNotAcceptedError,
    observe_trajectory,
    sensor_sample_indices,
)
from process_transfer.simulation.protocols import (
    P3_HOLD,
    P3_REST,
    a10_amplitudes,
    p3_corners,
    p3_segments,
)

EXCITATION_SEED, SENSOR_SEED = 7, 11
STREAM = {"source": (0, 0), "target": (1, 0)}
N_EXCURSIONS = 3
Z_LIMIT = 4.0


@pytest.fixture(scope="module")
def measurement(configs_dir: Path) -> MeasurementSpec:
    return MeasurementSpec.from_config(load_sensors(configs_dir / "sensors_cstr.yaml"))


def generate(
    plant: PlantUnderTest,
    name: str,
    measurement: MeasurementSpec,
    excitation_seed: int = EXCITATION_SEED,
    sensor_seed: int = SENSOR_SEED,
    n_excursions: int = N_EXCURSIONS,
) -> OperatingRun:
    """A whole generation from nothing but the configuration and the two seeds."""
    corners = p3_corners(n_excursions, excitation_seed)
    trajectory = simulate_piecewise(
        plant.f, plant.nominal_state, p3_segments(plant.nominal_inputs, corners), 0.1
    )
    return observe_trajectory(
        trajectory,
        plant.parameters,
        measurement,
        plant=name,
        run="p3-test",
        sensor_seed=sensor_seed,
        sensor_stream=STREAM[name],
    )


@pytest.fixture(scope="module")
def runs(
    true_plants: dict[str, PlantUnderTest], measurement: MeasurementSpec
) -> dict[str, OperatingRun]:
    return {name: generate(plant, name, measurement) for name, plant in true_plants.items()}


# --------------------------------------------------------------------------- #
# Time grid and alignment
# --------------------------------------------------------------------------- #


def test_one_reading_every_six_seconds_without_duplicates(runs: dict[str, OperatingRun]) -> None:
    for run in runs.values():
        observed = run.observations
        duration = N_EXCURSIONS * (P3_HOLD + P3_REST)
        np.testing.assert_array_equal(observed.times, 6.0 * np.arange(duration / 6.0 + 1))
        assert observed.n_samples == 120 * N_EXCURSIONS + 1
        assert len(np.unique(observed.times)) == observed.n_samples
        assert observed.sample_period == 6.0
        assert observed.measured.shape == (observed.n_samples, 2)
        assert observed.inputs.shape == (observed.n_samples, 4)


def test_the_truth_keeps_its_own_finer_sampling(runs: dict[str, OperatingRun]) -> None:
    """Peaks and balances are judged every 0.1 s; the sensors read every sixtieth of
    those samples. The two periods are not confused."""
    truth = runs["target"].truth
    fine = truth.trajectory.times
    assert np.median(np.diff(fine)) == pytest.approx(0.1)
    assert len(fine) == 60 * (runs["target"].observations.n_samples - 1) + 1
    np.testing.assert_array_equal(np.diff(truth.sample_indices), 60)
    np.testing.assert_array_equal(fine[truth.sample_indices], runs["target"].observations.times)
    assert truth.check.accepted
    # the true peak falls between two readings: the sensor grid could not have judged it
    assert truth.check.refined_peak_temperature > truth.exact[:, 1].max()


def test_a_row_carries_the_inputs_applied_from_its_instant_on(
    runs: dict[str, OperatingRun], true_plants: dict[str, PlantUnderTest]
) -> None:
    observed, nominal = runs["target"].observations, true_plants["target"].nominal_inputs
    corners = p3_corners(N_EXCURSIONS, EXCITATION_SEED)
    amplitudes = a10_amplitudes(nominal)
    for k, corner in enumerate(corners):
        start = k * (P3_HOLD + P3_REST)
        at_corner = nominal + corner * amplitudes
        row = {t: int(np.flatnonzero(observed.times == t)[0]) for t in (start, start + P3_HOLD)}
        np.testing.assert_array_equal(observed.inputs[row[start]], at_corner)  # right-continuous
        np.testing.assert_array_equal(observed.inputs[row[start + P3_HOLD] - 1], at_corner)
        np.testing.assert_array_equal(observed.inputs[row[start + P3_HOLD]], nominal)
    np.testing.assert_array_equal(observed.inputs[-1], nominal)
    assert observed.input_names == ("q", "C_Af", "T_f", "T_c")
    assert observed.input_units == ("m^3/s", "mol/m^3", "K", "K")
    assert observed.measured_names == ("C_A", "T")
    assert observed.measured_units == ("mol/m^3", "K")


def test_both_plants_receive_the_same_inputs(runs: dict[str, OperatingRun]) -> None:
    """One excitation seed, equal nominal inputs (D-008): the same experiment on both."""
    np.testing.assert_array_equal(
        runs["source"].observations.inputs, runs["target"].observations.inputs
    )
    np.testing.assert_array_equal(
        runs["source"].observations.times, runs["target"].observations.times
    )


def lag(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.array([-0.1 * (x[0] - u[1]), -0.05 * (x[1] - u[2])])


def lag_trajectory(durations: list[float], period: float = 0.1) -> Trajectory:
    levels = ([0.0, 50.0, 300.0, 0.0], [0.0, 200.0, 450.0, 0.0])
    segments = [InputSegment(d, np.array(levels[k % 2])) for k, d in enumerate(durations)]
    return simulate_piecewise(lag, np.array([100.0, 400.0]), segments, period)


def test_a_sensor_instant_that_is_not_a_stored_sample_is_an_error_not_an_interpolation() -> None:
    trajectory = lag_trajectory([12.0, 12.0])
    assert len(sensor_sample_indices(trajectory, 6.0)) == 5
    with pytest.raises(ValueError, match=r"t = 0.25 s is not a stored sample"):
        sensor_sample_indices(trajectory, 0.25)  # between 0.2 and 0.3
    with pytest.raises(ValueError, match="not a stored sample"):
        sensor_sample_indices(trajectory, 0.05)  # finer than the simulation
    with pytest.raises(ValueError, match="sample_period"):
        sensor_sample_indices(trajectory, 0.0)


def test_an_input_change_between_two_readings_is_an_error() -> None:
    """With holds of 125 s and readings every 6 s the change at t = 125 s would be
    recorded at 126 s. The inputs are said to be known without error, so this is refused."""
    with pytest.raises(ValueError, match=r"the inputs change at t = 125.0 s, between two sensor"):
        sensor_sample_indices(lag_trajectory([125.0, 115.0]), 6.0)
    # the same junction is harmless when the inputs do not change there
    constant = [InputSegment(125.0, np.zeros(4)), InputSegment(115.0, np.zeros(4))]
    same = simulate_piecewise(lag, np.array([100.0, 400.0]), constant, 0.1)
    assert len(sensor_sample_indices(same, 6.0)) == 41


def test_periods_that_are_not_binary_fractions_still_meet() -> None:
    """0.1 * 3 is 0.30000000000000004 in floating point and 0.3 is not. They are the same
    instant computed by two routes, and are recognised as such."""
    trajectory = lag_trajectory([3.0, 3.0])
    indices = sensor_sample_indices(trajectory, 0.3)
    assert len(indices) == 21
    np.testing.assert_array_equal(np.diff(indices), 3)


def test_a_run_that_does_not_start_at_zero_and_a_tail_shorter_than_a_period() -> None:
    whole = lag_trajectory([12.0, 12.0, 8.0])
    tail = dataclasses.replace(whole, segments=whole.segments[1:])
    indices = sensor_sample_indices(tail, 6.0)
    np.testing.assert_array_equal(tail.times[indices], [12.0, 18.0, 24.0, 30.0])  # ends at 32 s


def test_two_samples_too_close_to_tell_apart_are_refused() -> None:
    times = np.array([0.0, 6.0, float(np.nextafter(6.0, np.inf)), 12.0])
    states = np.column_stack([np.linspace(250.0, 251.0, 4), np.full(4, 350.0)])
    segment = SegmentTrajectory(times=times, states=states, inputs=np.zeros(4))
    hand_built = Trajectory((segment,), "test", 1e-9, 1e-9, 0)
    with pytest.raises(ValueError, match="two stored samples fall on the sensor instant"):
        sensor_sample_indices(hand_built, 6.0)


# --------------------------------------------------------------------------- #
# The truth is validated first
# --------------------------------------------------------------------------- #


def test_a_trajectory_that_is_not_accepted_is_not_observed(
    true_plants: dict[str, PlantUnderTest], measurement: MeasurementSpec
) -> None:
    """The M0-E03 counterexample, cold then hot, takes the target to 395.6 K."""
    plant = true_plants["target"]
    amplitudes = a10_amplitudes(plant.nominal_inputs)
    cold = plant.nominal_inputs + np.array([1, 1, -1, -1]) * amplitudes
    hot = plant.nominal_inputs + np.array([1, 1, 1, 1]) * amplitudes
    rejected = simulate_piecewise(
        plant.f, plant.nominal_state, [InputSegment(120.0, cold), InputSegment(120.0, hot)]
    )
    with pytest.raises(TrajectoryNotAcceptedError, match="failed inside_envelope") as caught:
        observe_trajectory(
            rejected,
            plant.parameters,
            measurement,
            plant="target",
            run="rejected",
            sensor_seed=SENSOR_SEED,
            sensor_stream=(1, 0),
        )
    assert caught.value.check.peak_temperature == pytest.approx(395.63, abs=0.01)


def test_a_sensor_for_a_variable_that_is_not_a_state_is_refused(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    plant = true_plants["source"]
    rest = simulate_piecewise(
        plant.f, plant.nominal_state, [InputSegment(60.0, plant.nominal_inputs)]
    )
    with pytest.raises(ValueError, match="no state is called"):
        observe_trajectory(
            rest,
            plant.parameters,
            MeasurementSpec(6.0, (SensorSpec("pH", "-", 0.1),)),
            plant="source",
            run="x",
            sensor_seed=1,
            sensor_stream=(0, 0),
        )


# --------------------------------------------------------------------------- #
# Separation of truth and observation
# --------------------------------------------------------------------------- #


def test_observations_and_truth_are_two_objects_with_nothing_shared(
    runs: dict[str, OperatingRun],
) -> None:
    run = runs["target"]
    assert isinstance(run.observations, Observations) and isinstance(run.truth, RunTruth)
    assert {field.name for field in dataclasses.fields(OperatingRun)} == {"observations", "truth"}

    reachable = [getattr(run.observations, f.name) for f in dataclasses.fields(Observations)]
    for value in reachable:
        assert not isinstance(value, (RunTruth, Trajectory))
    truth_arrays = (run.truth.exact, run.truth.errors, run.truth.trajectory.segments[0].states)
    for array in (run.observations.times, run.observations.measured, run.observations.inputs):
        assert not any(np.shares_memory(array, hidden) for hidden in truth_arrays)
        assert not array.flags.writeable
    assert not run.truth.exact.flags.writeable and not run.truth.errors.flags.writeable


def test_the_readings_differ_from_the_exact_states_by_the_recorded_errors(
    runs: dict[str, OperatingRun],
) -> None:
    for run in runs.values():
        truth, observed = run.truth, run.observations
        np.testing.assert_array_equal(truth.exact, truth.trajectory.states[truth.sample_indices])
        np.testing.assert_array_equal(observed.measured - truth.exact, truth.errors)
        assert np.all(truth.errors != 0.0)  # every reading carries noise
        assert truth.sensor_seed == SENSOR_SEED and truth.sensor_stream == STREAM[observed.plant]


def test_the_sensors_are_handed_values_and_a_specification_and_nothing_else(
    runs: dict[str, OperatingRun], measurement: MeasurementSpec
) -> None:
    """The readings of a run are reproduced from the exact values alone, without the
    plant, its parameters or the trajectory: the sensor never needed them."""
    run = runs["target"]
    again = measure(run.truth.exact, measurement.sensors, SENSOR_SEED, STREAM["target"])
    np.testing.assert_array_equal(again, run.observations.measured)


# --------------------------------------------------------------------------- #
# Reproducibility and independence of the two kinds of randomness
# --------------------------------------------------------------------------- #


def test_the_same_configuration_and_seeds_reproduce_the_content(
    runs: dict[str, OperatingRun],
    true_plants: dict[str, PlantUnderTest],
    measurement: MeasurementSpec,
) -> None:
    for name, plant in true_plants.items():
        np.random.seed(999)  # a global generator moved in between changes nothing
        np.random.normal(size=100)
        again = generate(plant, name, measurement)
        first = runs[name]
        np.testing.assert_array_equal(again.observations.times, first.observations.times)
        np.testing.assert_array_equal(again.observations.measured, first.observations.measured)
        np.testing.assert_array_equal(again.observations.inputs, first.observations.inputs)
        np.testing.assert_array_equal(again.truth.trajectory.states, first.truth.trajectory.states)
        assert again.observations.content_digest() == first.observations.content_digest()


def test_another_sensor_seed_changes_the_readings_and_nothing_else(
    runs: dict[str, OperatingRun],
    true_plants: dict[str, PlantUnderTest],
    measurement: MeasurementSpec,
) -> None:
    for name, plant in true_plants.items():
        first = runs[name]
        other = generate(plant, name, measurement, sensor_seed=SENSOR_SEED + 1)
        np.testing.assert_array_equal(
            other.truth.trajectory.states, first.truth.trajectory.states
        )  # bit for bit
        np.testing.assert_array_equal(other.truth.exact, first.truth.exact)
        np.testing.assert_array_equal(other.observations.inputs, first.observations.inputs)
        np.testing.assert_array_equal(other.observations.times, first.observations.times)
        assert np.all(other.observations.measured != first.observations.measured)


def test_another_excitation_seed_changes_the_corners_and_not_the_noise(
    true_plants: dict[str, PlantUnderTest], measurement: MeasurementSpec
) -> None:
    """The two seeds feed two generators that never meet. With the sensor seed and the
    stream unchanged the noise is the same sequence whatever was excited; the errors
    agree to the rounding of adding them to different states."""
    assert not np.array_equal(p3_corners(8, EXCITATION_SEED), p3_corners(8, EXCITATION_SEED + 1))
    np.testing.assert_array_equal(
        p3_corners(8, EXCITATION_SEED)[:3], p3_corners(3, EXCITATION_SEED)
    )

    plant = true_plants["target"]
    first = generate(plant, "target", measurement, n_excursions=2)
    other = generate(
        plant, "target", measurement, excitation_seed=EXCITATION_SEED + 1, n_excursions=2
    )
    assert not np.array_equal(other.observations.inputs, first.observations.inputs)
    np.testing.assert_allclose(other.truth.errors, first.truth.errors, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("bad", [None, -1, 0.5, True])
def test_the_excitation_seed_must_be_named(bad: object) -> None:
    with pytest.raises(ValueError, match="seed must be an explicit non-negative integer"):
        p3_corners(4, bad)


# --------------------------------------------------------------------------- #
# The same absolute noise on both plants (D-020)
# --------------------------------------------------------------------------- #


def test_both_plants_carry_the_same_absolute_noise(
    true_plants: dict[str, PlantUnderTest], measurement: MeasurementSpec
) -> None:
    """Ten excursions, 1201 readings per variable and plant. The deviation is 5 mol/m^3
    and 0.5 K on both; 2 % of the target's nominal concentration, 3.8 mol/m^3, is not
    what the target gets. The errors of the two plants, and of the two sensors of a
    plant, are uncorrelated."""
    long_runs = {
        name: generate(plant, name, measurement, n_excursions=10)
        for name, plant in true_plants.items()
    }
    assert measurement.noise_std == (pytest.approx(5.0), 0.5)
    for name, run in long_runs.items():
        assert run.observations.noise_std == measurement.noise_std
        assert run.observations.n_samples == 1201
        for column, sigma in enumerate(measurement.noise_std):
            found = noise_statistics(run.truth.errors[:, column], sigma)
            assert all(abs(z) <= Z_LIMIT for z in found.z_scores.values()), (name, column, found)
    target_c_a = long_runs["target"].truth.errors[:, 0]
    assert noise_statistics(target_c_a, 3.8).spread_z > 8.0

    source, target = long_runs["source"].truth.errors, long_runs["target"].truth.errors
    pairs = [(source[:, 0], source[:, 1]), (target[:, 0], target[:, 1])]
    pairs += [(source[:, i], target[:, j]) for i in range(2) for j in range(2)]
    for a, b in pairs:
        assert abs(correlation(a, b)[1]) <= Z_LIMIT
