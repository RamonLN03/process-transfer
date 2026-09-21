"""Integration under piecewise-constant inputs, restarted at every input change."""

import math

import numpy as np
import pytest

from process_transfer.simulation.integration import (
    InputSegment,
    IntegrationError,
    simulate_piecewise,
)


def lag(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Two first-order lags towards u[1] and u[2], with rates 0.1 and 0.05 1/s."""
    return np.array([-0.1 * (x[0] - u[1]), -0.05 * (x[1] - u[2])])


X0 = np.array([100.0, 400.0])
SEGMENTS = [
    InputSegment(10.0, np.array([0.0, 50.0, 300.0, 0.0])),
    InputSegment(7.0, np.array([0.0, 200.0, 450.0, 0.0])),
]


def exact_lag(x0: float, target: float, rate: float, t: np.ndarray | float) -> np.ndarray:
    return target + (x0 - target) * np.exp(-rate * np.asarray(t))


def test_trajectory_matches_the_analytical_solution_across_an_input_change() -> None:
    trajectory = simulate_piecewise(lag, X0, SEGMENTS, sample_period=0.5)
    first, second = trajectory.segments

    np.testing.assert_allclose(
        first.states[:, 0], exact_lag(100.0, 50.0, 0.1, first.times), rtol=1e-7
    )
    np.testing.assert_allclose(
        first.states[:, 1], exact_lag(400.0, 300.0, 0.05, first.times), rtol=1e-7
    )

    c_switch = float(exact_lag(100.0, 50.0, 0.1, 10.0))
    t_switch = float(exact_lag(400.0, 300.0, 0.05, 10.0))
    local = second.times - 10.0
    np.testing.assert_allclose(
        second.states[:, 0], exact_lag(c_switch, 200.0, 0.1, local), rtol=1e-7
    )
    np.testing.assert_allclose(
        second.states[:, 1], exact_lag(t_switch, 450.0, 0.05, local), rtol=1e-7
    )


def test_switching_instants_appear_once_and_carry_the_new_inputs() -> None:
    trajectory = simulate_piecewise(lag, X0, SEGMENTS, sample_period=3.0)
    times, states, inputs = trajectory.times, trajectory.states, trajectory.inputs

    np.testing.assert_allclose(times, [0, 3, 6, 9, 10, 13, 16, 17])
    assert len(states) == len(inputs) == len(times)
    assert trajectory.duration == 17.0
    assert np.all(np.diff(times) > 0)

    switch = int(np.flatnonzero(times == 10.0)[0])
    np.testing.assert_array_equal(inputs[switch - 1], SEGMENTS[0].inputs)
    np.testing.assert_array_equal(inputs[switch], SEGMENTS[1].inputs)  # zero-order hold
    np.testing.assert_array_equal(inputs[-1], SEGMENTS[1].inputs)
    np.testing.assert_array_equal(  # the state is continuous across the change
        trajectory.segments[0].states[-1], trajectory.segments[1].states[0]
    )


def test_splitting_a_constant_input_run_changes_nothing() -> None:
    u = SEGMENTS[0].inputs
    whole = simulate_piecewise(lag, X0, [InputSegment(17.0, u)], sample_period=1.0)
    split = simulate_piecewise(
        lag, X0, [InputSegment(10.0, u), InputSegment(7.0, u)], sample_period=1.0
    )
    np.testing.assert_allclose(split.states[-1], whole.states[-1], rtol=1e-8)
    np.testing.assert_allclose(split.times, whole.times)


def test_ranges_and_peak_are_read_from_the_samples() -> None:
    trajectory = simulate_piecewise(lag, X0, SEGMENTS, sample_period=0.5)
    low, high = trajectory.state_range(0)
    assert low == pytest.approx(float(exact_lag(100.0, 50.0, 0.1, 10.0)), rel=1e-7)
    assert high == pytest.approx(trajectory.states[-1, 0])
    peak, when = trajectory.peak(0)  # the first state ends at its maximum
    assert when == 17.0
    assert peak == pytest.approx(trajectory.states[-1, 0])
    peak, when = trajectory.peak(1)  # the second state starts at its maximum
    assert when == 0.0
    assert peak == 400.0


@pytest.mark.parametrize("method", ["LSODA", "DOP853", "Radau"])
def test_integrators_agree(method: str) -> None:
    trajectory = simulate_piecewise(lag, X0, SEGMENTS, sample_period=1.0, method=method)
    c_switch = float(exact_lag(100.0, 50.0, 0.1, 10.0))
    expected = float(exact_lag(c_switch, 200.0, 0.1, 7.0))
    assert trajectory.states[-1, 0] == pytest.approx(expected, rel=1e-7)
    assert trajectory.method == method
    assert trajectory.n_rhs_evaluations > 0


def test_a_failing_right_hand_side_raises_instead_of_being_clipped() -> None:
    def blows_up(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([math.nan, 0.0])

    with pytest.raises(IntegrationError):
        simulate_piecewise(blows_up, X0, SEGMENTS)


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        simulate_piecewise(lag, X0, [])
    with pytest.raises(ValueError, match="sample_period"):
        simulate_piecewise(lag, X0, SEGMENTS, sample_period=0.0)
    with pytest.raises(ValueError, match="duration"):
        simulate_piecewise(lag, X0, [InputSegment(-1.0, SEGMENTS[0].inputs)])


def test_refined_peak_recovers_a_maximum_that_falls_between_samples() -> None:
    """A harmonic oscillator started at (0, 1) gives x = sin(t), with its maximum of 1 at pi/2.
    Sampled every 0.4 s, the largest sample is sin(1.6); the parabola recovers the rest."""

    def oscillator(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([x[1], -x[0]])  # x0 = sin t, x1 = cos t

    segments = [InputSegment(3.0, np.zeros(4))]
    trajectory = simulate_piecewise(oscillator, np.array([0.0, 1.0]), segments, sample_period=0.4)
    sampled, _ = trajectory.peak(0)
    refined, when = trajectory.refined_peak(0)
    assert sampled == pytest.approx(math.sin(1.6), rel=1e-7)
    assert sampled < refined <= 1.0 + 1e-3
    assert refined == pytest.approx(1.0, abs=2e-3)
    assert when == pytest.approx(math.pi / 2.0, abs=0.02)


def test_refined_peak_is_not_applied_at_a_segment_end() -> None:
    trajectory = simulate_piecewise(lag, X0, SEGMENTS, sample_period=0.5)
    assert trajectory.refined_peak(0) == trajectory.peak(0)  # maximum at the final sample
    assert trajectory.refined_peak(1) == trajectory.peak(1)  # maximum at the first sample


def test_the_stored_state_is_exactly_continuous_across_every_input_change() -> None:
    """Regression: on a nonlinear system the first stored sample of a segment came from
    the solver's interpolant at t = 0 and differed from the last sample of the previous
    segment in the last bit, so a switching instant was stored with two states."""

    def nonlinear(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        rate = 1.0e3 * np.exp(-3000.0 / x[1]) * x[0]
        return np.array([0.02 * (u[1] - x[0]) - rate, 0.02 * (u[2] - x[1]) + 0.2 * rate])

    segments = [
        InputSegment(37.0, np.array([0.0, 500.0, 345.0, 0.0])),
        InputSegment(41.0, np.array([0.0, 550.0, 355.0, 0.0])),
        InputSegment(29.0, np.array([0.0, 450.0, 350.0, 0.0])),
    ]
    start = np.array([250.0, 350.0])
    trajectory = simulate_piecewise(nonlinear, start, segments, sample_period=0.1)
    np.testing.assert_array_equal(trajectory.segments[0].states[0], start)
    for before, after in zip(trajectory.segments[:-1], trajectory.segments[1:], strict=True):
        np.testing.assert_array_equal(after.states[0], before.states[-1])
        assert after.times[0] == before.times[-1]
