"""Sampling grids. Regression tests for a confirmed defect: the envelope grid was
built with ``arange`` and dropped the final instant whenever the duration was not a
whole number of sampling periods, so the reported final state was too early."""

import math

import numpy as np
import pytest

from process_transfer.simulation.envelope import simulate_envelope
from process_transfer.simulation.integration import sample_times


@pytest.mark.parametrize(
    ("duration", "period", "expected"),
    [
        (10.0, 3.0, [0.0, 3.0, 6.0, 9.0, 10.0]),
        (10.0, 4.0, [0.0, 4.0, 8.0, 10.0]),
        (10.0, 5.0, [0.0, 5.0, 10.0]),
        (10.0, 10.0, [0.0, 10.0]),
        (10.0, 25.0, [0.0, 10.0]),  # a period longer than the duration
    ],
)
def test_grid_always_ends_at_the_duration(
    duration: float, period: float, expected: list[float]
) -> None:
    np.testing.assert_allclose(sample_times(duration, period), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("duration", "period", "count"), [(1.0, 0.1, 11), (0.3, 0.1, 4)])
def test_rounding_neither_duplicates_nor_drops_the_final_instant(
    duration: float, period: float, count: int
) -> None:
    times = sample_times(duration, period)
    assert len(times) == count
    assert times[0] == 0.0
    assert times[-1] == duration  # exactly, not approximately
    assert np.all(np.diff(times) > 0.5 * period)


def test_long_grids_do_not_drift() -> None:
    times = sample_times(7200.0, 0.1)
    assert len(times) == 72001
    assert times[-1] == 7200.0
    assert times[36000] == pytest.approx(3600.0, abs=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_duration_and_period_must_be_finite_and_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        sample_times(bad, 1.0)
    with pytest.raises(ValueError, match="sample_period"):
        sample_times(10.0, bad)


def _linear_decay(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """dC/dt = -0.1 C and dT/dt = -0.05 (T - 300): known exponential solutions."""
    return np.array([-0.1 * x[0], -0.05 * (x[1] - 300.0)])


@pytest.mark.parametrize("period", [3.0, 4.0, 5.0, 0.7])
def test_envelope_final_state_is_the_state_at_the_requested_duration(period: float) -> None:
    x0 = np.array([100.0, 400.0])
    (case,) = simulate_envelope(
        _linear_decay, x0, [("decay", np.zeros(4))], duration=10.0, sample_period=period
    )
    exact = np.array([100.0 * math.exp(-1.0), 300.0 + 100.0 * math.exp(-0.5)])
    np.testing.assert_allclose(case.final_state, exact, rtol=1e-7)
    assert case.c_a_range[0] == pytest.approx(exact[0], rel=1e-7)  # the minimum is at the end
    assert case.duration == 10.0


def test_envelope_rejects_invalid_duration_and_period() -> None:
    x0 = np.array([100.0, 400.0])
    cases = [("decay", np.zeros(4))]
    with pytest.raises(ValueError, match="duration"):
        simulate_envelope(_linear_decay, x0, cases, duration=-5.0)
    with pytest.raises(ValueError, match="sample_period"):
        simulate_envelope(_linear_decay, x0, cases, sample_period=0.0)
