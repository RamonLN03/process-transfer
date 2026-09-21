"""Instants on a sampling clock, to the resolution of floating-point arithmetic."""

import numpy as np
import pytest

from process_transfer.sampling_clock import LARGEST_TICK, nearest_ticks


def test_the_same_instant_by_two_routes_is_one_tick() -> None:
    """0.1 * 3 is 0.30000000000000004 and 0.3 is not; both are the third tick of 0.1 s."""
    grid = 0.1 * np.arange(61)  # a simulation grid of 0.1 s over 6 s
    ticks, on_clock = nearest_ticks(grid, 0.0, 0.3)
    assert grid[3] != 0.3
    assert on_clock[3] and ticks[3] == 1
    np.testing.assert_array_equal(np.flatnonzero(on_clock), np.arange(0, 61, 3))
    np.testing.assert_array_equal(ticks[on_clock], np.arange(21))


def test_an_instant_between_two_ticks_is_not_on_the_clock() -> None:
    ticks, on_clock = nearest_ticks(np.array([0.0, 5.9, 6.0, 6.1, 9.0, 12.0]), 0.0, 6.0)
    np.testing.assert_array_equal(on_clock, [True, False, True, False, False, True])
    np.testing.assert_array_equal(ticks, [0, 1, 1, 1, 2, 2])  # nearest tick, halves to even


def test_a_distance_far_above_the_resolution_is_never_absorbed() -> None:
    """At two hours the resolution is of the order of 1e-12 s. A sample 1e-9 s off the
    clock is not on it."""
    late = 7200.0 + 1.0e-9
    _, on_clock = nearest_ticks(np.array([7200.0, late]), 0.0, 6.0)
    assert on_clock[0] and not on_clock[1]
    _, near = nearest_ticks(np.array([float(np.nextafter(7200.0, np.inf))]), 0.0, 6.0)
    assert near[0]  # one unit in the last place is the arithmetic, not the data


def test_the_clock_may_start_anywhere() -> None:
    ticks, on_clock = nearest_ticks(np.array([720.0, 726.0, 732.0]), 720.0, 6.0)
    assert on_clock.all()
    np.testing.assert_array_equal(ticks, [0, 1, 2])
    ticks, on_clock = nearest_ticks(np.array([-12.0, -6.0, 0.0]), -12.0, 6.0)
    assert on_clock.all() and list(ticks) == [0, 1, 2]
    empty_ticks, empty_mask = nearest_ticks(np.array([]), 0.0, 6.0)
    assert empty_ticks.shape == empty_mask.shape == (0,)  # nothing to place, nothing placed


@pytest.mark.parametrize(
    ("times", "start", "period", "message"),
    [
        (np.array([0.0, 6.0]), 0.0, 0.0, "period"),
        (np.array([0.0, 6.0]), 0.0, -6.0, "period"),
        (np.array([0.0, 6.0]), 0.0, np.nan, "period"),
        (np.array([0.0, 6.0]), np.inf, 6.0, "start"),
        (np.array([0.0, np.nan]), 0.0, 6.0, "times must be finite"),
        (np.zeros((2, 2)), 0.0, 6.0, "must be a vector"),
        (np.array([0.0, 1.0e300]), 0.0, 1.0e-300, "cannot be held exactly"),
        (np.array([0.0, 1.0e20]), 0.0, 6.0, "cannot be held exactly"),
        (np.array([-1.7e308, 1.7e308]), -1.7e308, 6.0, "cannot be held exactly"),
    ],
)
def test_invalid_clocks_are_rejected(
    times: np.ndarray, start: float, period: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        nearest_ticks(times, start, period)


def test_the_largest_tick_is_the_last_integer_a_float_holds_exactly() -> None:
    assert float(LARGEST_TICK) + 1.0 == float(LARGEST_TICK)  # 2**53 + 1 is not representable
    assert float(LARGEST_TICK - 1) + 1.0 == float(LARGEST_TICK)
