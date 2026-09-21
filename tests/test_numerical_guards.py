"""Invalid input is rejected at the boundary, with a useful message, not downstream.

Each test below is the regression of a failure that was reproduced on the code as it
stood at 50dc509 (docs/numerical_robustness.md). Most of them were silent: a result
was returned, and it was wrong.
"""

import numpy as np
import pytest

from process_transfer.simulation.envelope import input_cases, simulate_envelope
from process_transfer.simulation.excitation import levels_to_segments
from process_transfer.simulation.integration import (
    MAX_SAMPLES_PER_SEGMENT,
    InputSegment,
    IntegrationError,
    SegmentTrajectory,
    Trajectory,
    sample_times,
    simulate_piecewise,
)

NOMINAL = np.array([1.0, 10.0, 100.0, 1000.0])
AMPLITUDES = np.array([0.1, 1.0, 5.0, 50.0])
X0 = np.array([100.0, 400.0])


def lag(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.array([-0.1 * (x[0] - u[1]), -0.05 * (x[1] - u[2])])


def returns_nan(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.array([np.nan, 0.0])


# --------------------------------------------------------------------------- #
# Sampling grid
# --------------------------------------------------------------------------- #


def test_a_grid_too_large_to_build_is_refused_with_an_explanation() -> None:
    """Was a MemoryError for 7 PiB, or an OverflowError when the ratio itself overflowed:
    both inputs are finite and positive, their ratio is not usable."""
    with pytest.raises(ValueError, match="longer sample_period"):
        sample_times(1.0e6, 1.0e-9)
    with pytest.raises(ValueError, match="longer sample_period"):
        sample_times(1.0e308, 1.0e-308)
    assert len(sample_times(7200.0, 0.01)) == 720_001 < MAX_SAMPLES_PER_SEGMENT


# --------------------------------------------------------------------------- #
# Integration arguments
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["rtol", "atol"])
@pytest.mark.parametrize("bad", [0.0, -1.0e-9, np.nan, np.inf])
def test_tolerances_must_be_positive_and_finite(name: str, bad: float) -> None:
    """With rtol = 0 scipy substituted its own tolerance and only warned."""
    with pytest.raises(ValueError, match=name):
        simulate_piecewise(lag, X0, [InputSegment(1.0, NOMINAL)], **{name: bad})


def test_the_initial_state_and_the_inputs_must_be_finite() -> None:
    with pytest.raises(ValueError, match="x0 must be a finite vector"):
        simulate_piecewise(lag, np.array([np.nan, 400.0]), [InputSegment(1.0, NOMINAL)])
    bad_inputs = NOMINAL.copy()
    bad_inputs[2] = np.inf
    with pytest.raises(ValueError, match="segment 1 has non-finite inputs"):
        simulate_piecewise(lag, X0, [InputSegment(1.0, NOMINAL), InputSegment(1.0, bad_inputs)])


def test_hand_built_trajectories_are_checked_for_shape() -> None:
    times, states = np.array([0.0, 1.0]), np.zeros((2, 2))
    SegmentTrajectory(times=times, states=states, inputs=NOMINAL)  # the smallest valid segment
    with pytest.raises(ValueError, match="two end samples"):
        SegmentTrajectory(times=np.array([0.0]), states=np.zeros((1, 2)), inputs=NOMINAL)
    with pytest.raises(ValueError, match="one row per sample"):
        SegmentTrajectory(times=times, states=np.zeros((3, 2)), inputs=NOMINAL)
    with pytest.raises(ValueError, match="at least one segment"):
        Trajectory(segments=(), method="test", rtol=1e-9, atol=1e-9, n_rhs_evaluations=0)


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #


def test_the_envelope_raises_on_a_non_finite_right_hand_side() -> None:
    """Returned a temperature range of (350, 350) computed from the initial state."""
    with pytest.raises(IntegrationError):
        simulate_envelope(returns_nan, X0, [("case", NOMINAL)], duration=10.0)


def test_a_negative_deviation_does_not_swap_the_labels() -> None:
    """With a deviation of -0.1 the case labelled 'q+' lowered q and 'q-' raised it."""
    with pytest.raises(ValueError, match="must not be negative"):
        input_cases(NOMINAL, np.array([-0.1, 1.0, 5.0, 50.0]))
    for label, u in input_cases(NOMINAL, AMPLITUDES)[:2]:
        assert (u[0] > NOMINAL[0]) == label.endswith("+")


@pytest.mark.parametrize(
    ("nominal", "deviations", "message"),
    [
        (NOMINAL[:3], AMPLITUDES[:3], "shape"),
        (NOMINAL, np.array([0.1, np.nan, 5.0, 50.0]), "finite"),
        (np.array([1.0, np.inf, 100.0, 1000.0]), AMPLITUDES, "finite"),
    ],
)
def test_input_cases_validate_shape_and_finiteness(
    nominal: np.ndarray, deviations: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        input_cases(nominal, deviations)


def test_zero_deviation_is_a_valid_limit() -> None:
    cases = input_cases(NOMINAL, np.zeros(4))
    assert len(cases) == 24
    assert all(np.array_equal(u, NOMINAL) for _, u in cases)


# --------------------------------------------------------------------------- #
# Excitation
# --------------------------------------------------------------------------- #


def test_levels_outside_the_declared_range_are_rejected() -> None:
    """A level of 7 moved an input by seven amplitudes without a word."""
    with pytest.raises(ValueError, match="between -1 and \\+1"):
        levels_to_segments(NOMINAL, AMPLITUDES, np.array([[7, 0, 0, 0]]), clock=120.0)
    with pytest.raises(ValueError, match="nominal inputs must be finite"):
        levels_to_segments(
            np.array([1.0, np.nan, 100.0, 1000.0]), AMPLITUDES, np.zeros((1, 4), dtype=int), 120.0
        )
    (segment,) = levels_to_segments(NOMINAL, AMPLITUDES, np.array([[1, -1, 0, 1]]), clock=120.0)
    np.testing.assert_allclose(segment.inputs, [1.1, 9.0, 100.0, 1050.0])
