"""The two other operating runs of D-010 as experiments on the plant: steady operation
and single-input step tests. Their structure, their inputs and their independence; what
they do to the plants over the full durations is measured in M0-E06 and M0-E07."""

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.simulation.integration import simulate_piecewise
from process_transfer.simulation.protocols import (
    STEADY_DURATION,
    STEP_DIRECTIONS,
    STEP_HOLD,
    STEP_INPUTS,
    STEP_LEAD,
    STEP_RECOVERY,
    a10_amplitudes,
    single_step_segments,
    steady_segments,
)


def test_steady_operation_is_one_segment_at_the_nominal_inputs(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    u = true_plants["source"].nominal_inputs
    (segment,) = steady_segments(u)
    assert segment.duration == STEADY_DURATION == 7200.0
    np.testing.assert_array_equal(segment.inputs, u)
    (shorter,) = steady_segments(u, 120.0)
    assert shorter.duration == 120.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="clock must be a finite positive number"):
            steady_segments(u, bad)


@pytest.mark.parametrize("direction", STEP_DIRECTIONS)
@pytest.mark.parametrize(("input_name", "index"), list(zip(STEP_INPUTS, range(4), strict=True)))
def test_a_single_input_step_moves_one_input_by_its_a10_amplitude(
    input_name: str, index: int, direction: str, true_plants: dict[str, PlantUnderTest]
) -> None:
    u = true_plants["target"].nominal_inputs
    segments = single_step_segments(u, input_name, direction)
    assert [s.duration for s in segments] == [STEP_LEAD, STEP_HOLD, STEP_RECOVERY] == [600.0] * 3
    np.testing.assert_array_equal(segments[0].inputs, u)
    np.testing.assert_array_equal(segments[2].inputs, u)
    expected = u.copy()
    expected[index] += (1.0 if direction == "up" else -1.0) * a10_amplitudes(u)[index]
    np.testing.assert_allclose(segments[1].inputs, expected, rtol=0, atol=0)
    assert np.count_nonzero(segments[1].inputs != u) == 1  # the other three stay nominal


def test_unknown_inputs_directions_and_durations_are_refused(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    u = true_plants["target"].nominal_inputs
    with pytest.raises(ValueError, match="input_name must be one of"):
        single_step_segments(u, "T_c", "up")
    with pytest.raises(ValueError, match="direction must be one of"):
        single_step_segments(u, "tc", "plus")
    with pytest.raises(ValueError, match="clock must be a finite positive number"):
        single_step_segments(u, "tc", "up", hold=0.0)


def test_the_eight_tests_of_a_plant_are_independent_runs_from_the_nominal_state(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """Shortened to 12 s per segment: each test starts at the verified steady state and
    carries its state across its own three segments only."""
    plant = true_plants["target"]
    for input_name in STEP_INPUTS:
        for direction in STEP_DIRECTIONS:
            segments = single_step_segments(plant.nominal_inputs, input_name, direction, 12, 12, 12)
            trajectory = simulate_piecewise(plant.f, plant.nominal_state, segments, 0.1)
            assert len(trajectory.segments) == 3
            np.testing.assert_array_equal(trajectory.states[0], plant.nominal_state)
            assert trajectory.duration == 36.0
            # the state is carried: the second and third segments start where the previous ended
            for previous, following in zip(
                trajectory.segments, trajectory.segments[1:], strict=False
            ):
                np.testing.assert_array_equal(following.states[0], previous.states[-1])
