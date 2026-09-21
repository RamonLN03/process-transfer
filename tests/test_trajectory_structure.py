"""A trajectory is one continuous trajectory, or it is not built at all.

Regression tests for a confirmed defect, found in the review of ``325cb93``: two pieces
simulated apart, each from its own steady state, could be put side by side in a
``Trajectory``. The state jumped by -67 mol/m^3 and +6.9 K where they met, and
``check_trajectory`` accepted the result, because the balances close inside each piece.
"""

import dataclasses

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.simulation import cstr_true
from process_transfer.simulation.checks import check_trajectory
from process_transfer.simulation.integration import (
    InputSegment,
    SegmentTrajectory,
    Trajectory,
    simulate_piecewise,
)
from process_transfer.simulation.protocols import p3_segments
from process_transfer.simulation.steady_state import find_steady_states

INPUTS = np.array([1.0, 10.0, 100.0, 1000.0])


def segment(times: list[float], c_a: list[float], inputs: np.ndarray = INPUTS) -> SegmentTrajectory:
    states = np.column_stack([np.array(c_a, dtype=np.float64), np.full(len(c_a), 350.0)])
    return SegmentTrajectory(times=np.array(times, dtype=np.float64), states=states, inputs=inputs)


def trajectory(*segments: SegmentTrajectory) -> Trajectory:
    return Trajectory(segments=segments, method="test", rtol=1e-9, atol=1e-9, n_rhs_evaluations=0)


# --------------------------------------------------------------------------- #
# The counterexample of the review
# --------------------------------------------------------------------------- #


def test_two_steady_states_simulated_apart_cannot_be_joined(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """The target at rest at its nominal inputs for 60 s, then at rest at the steady
    state of T_c + 5 K for 60 s, the second piece moved to start at t = 60 s. Each piece
    is a valid trajectory and closes its balances; together they are not a trajectory."""
    plant = true_plants["target"]
    warmer = plant.nominal_inputs.copy()
    warmer[3] += 5.0
    (shifted,) = find_steady_states(
        lambda x: cstr_true.rhs(0.0, x, warmer, plant.parameters), c_a_upper=warmer[1]
    )
    jump = shifted.state - plant.nominal_state
    assert jump[0] == pytest.approx(-67.09, abs=0.01)  # mol/m^3
    assert jump[1] == pytest.approx(6.86, abs=0.01)  # K

    first = simulate_piecewise(
        plant.f, plant.nominal_state, [InputSegment(60.0, plant.nominal_inputs)]
    )
    second = simulate_piecewise(plant.f, shifted.state, [InputSegment(60.0, warmer)])
    assert check_trajectory(first, plant.parameters).accepted
    assert check_trajectory(second, plant.parameters).accepted

    (piece,) = second.segments
    moved = dataclasses.replace(piece, times=piece.times + 60.0)
    with pytest.raises(ValueError, match="the state jumps between segments 0 and 1"):
        dataclasses.replace(first, segments=(*first.segments, moved))


def test_the_same_inputs_simulated_as_one_trajectory_are_accepted(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """What the counterexample should have been: the step in T_c applied to the plant,
    the state carried across it. The inputs jump; the states do not."""
    plant = true_plants["target"]
    warmer = plant.nominal_inputs.copy()
    warmer[3] += 5.0
    found = simulate_piecewise(
        plant.f,
        plant.nominal_state,
        [InputSegment(60.0, plant.nominal_inputs), InputSegment(60.0, warmer)],
    )
    before, after = found.segments
    assert after.times[0] == before.times[-1] == 60.0
    np.testing.assert_array_equal(after.states[0], before.states[-1])
    assert after.inputs[3] - before.inputs[3] == 5.0  # discontinuous, as it should be

    switch = int(np.flatnonzero(found.times == 60.0)[0])
    assert len(found.times) == len(np.unique(found.times))  # the instant appears once
    np.testing.assert_array_equal(found.inputs[switch - 1], plant.nominal_inputs)
    np.testing.assert_array_equal(found.inputs[switch], warmer)  # right-continuous
    assert check_trajectory(found, plant.parameters).accepted


# --------------------------------------------------------------------------- #
# Junctions
# --------------------------------------------------------------------------- #


def test_a_gap_and_an_overlap_in_time_are_rejected() -> None:
    first = segment([0.0, 1.0, 2.0], [250.0, 251.0, 252.0])
    with pytest.raises(ValueError, match="leave a gap in time"):
        trajectory(first, segment([2.5, 3.0], [252.0, 253.0]))
    with pytest.raises(ValueError, match="leave an overlap in time"):
        trajectory(first, segment([1.5, 3.0], [252.0, 253.0]))
    with pytest.raises(ValueError, match="segments 1 and 2"):  # the message says where
        trajectory(first, segment([2.0, 3.0], [252.0, 253.0]), segment([4.0, 5.0], [253.0, 254.0]))
    assert trajectory(first, segment([2.0, 3.0], [252.0, 253.0])).duration == 3.0


def test_a_jump_of_either_state_is_rejected() -> None:
    first = segment([0.0, 1.0], [250.0, 252.0])
    with pytest.raises(ValueError, match="the state jumps"):
        trajectory(first, segment([1.0, 2.0], [251.0, 253.0]))  # C_A

    hotter = np.column_stack([[252.0, 253.0], [350.5, 351.0]])
    with pytest.raises(ValueError, match="the state jumps"):
        trajectory(first, SegmentTrajectory(np.array([1.0, 2.0]), hotter, INPUTS))  # T


def test_the_comparison_at_a_junction_is_exact() -> None:
    """One unit in the last place is a different number. The switching instant and its
    state are one sample stored twice, and ``simulate_piecewise`` stores them identically,
    so nothing legitimate is lost by refusing a tolerance."""
    first = segment([0.0, 1.0], [250.0, 252.0])
    with pytest.raises(ValueError, match="the state jumps"):
        trajectory(first, segment([1.0, 2.0], [float(np.nextafter(252.0, np.inf)), 253.0]))
    with pytest.raises(ValueError, match="in time"):
        trajectory(first, segment([float(np.nextafter(1.0, np.inf)), 2.0], [252.0, 253.0]))


def test_a_non_finite_state_at_a_junction_is_not_continuous() -> None:
    first = segment([0.0, 1.0], [250.0, np.nan])
    with pytest.raises(ValueError, match="the state jumps"):
        trajectory(first, segment([1.0, 2.0], [np.nan, 253.0]))


def test_segments_must_agree_on_the_number_of_states_and_inputs() -> None:
    first = segment([0.0, 1.0], [250.0, 252.0])
    one_state = SegmentTrajectory(np.array([1.0, 2.0]), np.array([[252.0], [253.0]]), INPUTS)
    with pytest.raises(ValueError, match="number of states or of inputs"):
        trajectory(first, one_state)
    with pytest.raises(ValueError, match="number of states or of inputs"):
        trajectory(first, segment([1.0, 2.0], [252.0, 253.0], inputs=INPUTS[:3]))


# --------------------------------------------------------------------------- #
# Inside a segment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ([0.0, 1.0, 1.0, 2.0], "strictly increasing: sample 2 is at 1.0 s, after 1.0 s"),
        ([0.0, 2.0, 1.0, 3.0], "strictly increasing: sample 2 is at 1.0 s, after 2.0 s"),
        ([3.0, 2.0, 1.0, 0.0], "strictly increasing"),
        ([0.0, np.nan, 2.0, 3.0], "must be finite"),
        ([0.0, 1.0, 2.0, np.inf], "must be finite"),
        ([-np.inf, 1.0, 2.0, 3.0], "must be finite"),
    ],
)
def test_sampling_instants_must_be_finite_and_strictly_increasing(
    times: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        segment(times, [250.0, 251.0, 252.0, 253.0])


def test_shapes_must_be_compatible() -> None:
    times, states = np.array([0.0, 1.0]), np.zeros((2, 2))
    with pytest.raises(ValueError, match="one constant vector"):
        SegmentTrajectory(times=times, states=states, inputs=np.tile(INPUTS, (2, 1)))
    with pytest.raises(ValueError, match="one row per sample"):
        SegmentTrajectory(times=times, states=np.zeros(2), inputs=INPUTS)
    with pytest.raises(ValueError, match="two end samples"):
        SegmentTrajectory(times=np.zeros((2, 1)), states=states, inputs=INPUTS)


# --------------------------------------------------------------------------- #
# Legitimate trajectories
# --------------------------------------------------------------------------- #


def test_a_trimmed_trajectory_need_not_start_at_zero(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """The second excursion of a P3 pair and its rest, cut out of the whole. Regression
    as well: ``duration`` used to return the final instant, 1440 s here."""
    plant = true_plants["target"]
    corners = np.array([[1, 1, -1, -1], [1, 1, 1, 1]])
    whole = simulate_piecewise(
        plant.f, plant.nominal_state, p3_segments(plant.nominal_inputs, corners)
    )
    second_half = dataclasses.replace(whole, segments=whole.segments[2:])
    assert second_half.times[0] == 720.0 and second_half.times[-1] == 1440.0
    assert second_half.duration == 720.0
    assert whole.duration == 1440.0
    assert check_trajectory(second_half, plant.parameters).accepted
    assert np.all(np.diff(second_half.times) > 0.0)

    with pytest.raises(ValueError, match="leave a gap in time"):  # not consecutive
        dataclasses.replace(whole, segments=(whole.segments[0], whole.segments[2]))


def test_negative_and_uneven_sampling_instants_are_valid() -> None:
    found = trajectory(
        segment([-5.0, -4.5, -1.0], [250.0, 251.0, 252.0]), segment([-1.0, 7.0], [252.0, 260.0])
    )
    np.testing.assert_array_equal(found.times, [-5.0, -4.5, -1.0, 7.0])
    assert found.duration == 12.0


def test_what_was_validated_cannot_be_changed_afterwards() -> None:
    """The arrays of a segment are read-only copies. An assignment in place fails, and
    the caller's own array no longer reaches the trajectory."""
    times = np.array([0.0, 1.0])
    states = np.array([[250.0, 350.0], [252.0, 351.0]])
    built = SegmentTrajectory(times=times, states=states, inputs=INPUTS)

    with pytest.raises(ValueError, match="read-only"):
        built.states[1, 0] = 300.0
    with pytest.raises(ValueError, match="read-only"):
        built.times[1] = 0.0

    states[1, 0] = 300.0  # the caller's array is theirs to change
    times[1] = -1.0
    assert built.states[1, 0] == 252.0
    assert built.times[1] == 1.0
    assert states.flags.writeable  # and it has not been frozen behind their back
