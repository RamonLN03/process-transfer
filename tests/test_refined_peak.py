"""Refined estimate of a maximum between samples.

Regression tests for a confirmed defect: the refinement looked only at the segment
holding the largest sample, so a higher maximum between the samples of another
segment, or around another local maximum, was missed.
"""

import numpy as np
import pytest

from process_transfer.simulation.integration import SegmentTrajectory, Trajectory


def segment(times: list[float], temperatures: list[float]) -> SegmentTrajectory:
    t = np.array(times, dtype=np.float64)
    states = np.column_stack([np.full(len(t), 250.0), np.array(temperatures, dtype=np.float64)])
    return SegmentTrajectory(times=t, states=states, inputs=np.zeros(4))


def trajectory(*segments: SegmentTrajectory) -> Trajectory:
    return Trajectory(segments=segments, method="test", rtol=0.0, atol=0.0, n_rhs_evaluations=0)


def test_a_higher_maximum_in_another_segment_is_found() -> None:
    """The case reported in review. The largest sample, 379.9 K, is in the first
    segment; the parabola of the second segment peaks near 380.988 K, above the limit."""
    found = trajectory(
        segment([0, 1, 2], [370.0, 379.9, 370.0]), segment([2, 3, 4], [370.0, 379.8, 379.7])
    )
    assert found.peak(1) == (379.9, 1.0)
    value, when = found.refined_peak(1)
    assert value == pytest.approx(380.988005, abs=1e-6)
    assert when == pytest.approx(3.489899, abs=1e-6)


def test_a_higher_maximum_around_another_local_maximum_of_the_same_segment_is_found() -> None:
    found = trajectory(segment([0, 1, 2, 3, 4], [370.0, 379.9, 370.0, 379.8, 379.7]))
    value, when = found.refined_peak(1)
    assert value == pytest.approx(380.988005, abs=1e-6)
    assert when == pytest.approx(3.489899, abs=1e-6)


def test_no_parabola_is_fitted_across_a_change_of_inputs() -> None:
    """Both segments are convex, so neither has an interior maximum: the maximum is
    the sample at the switching instant. A parabola through 372, 380, 378, the three
    samples around the switch, would invent 380.45 K out of a kink."""
    found = trajectory(
        segment([0, 1, 2], [370.0, 372.0, 380.0]), segment([2, 3, 4], [380.0, 378.0, 377.5])
    )
    assert found.refined_peak(1) == (380.0, 2.0)


def test_a_plateau_of_two_samples_is_refined_and_a_flat_one_is_not() -> None:
    two = trajectory(segment([0, 1, 2, 3], [370.0, 380.0, 380.0, 370.0]))
    value, when = two.refined_peak(1)
    assert value == pytest.approx(381.25)
    assert when == pytest.approx(1.5)

    flat = trajectory(segment([0, 1, 2, 3, 4], [370.0, 380.0, 380.0, 380.0, 370.0]))
    value, when = flat.refined_peak(1)
    assert value == pytest.approx(381.25)  # from the concave ends of the plateau
    assert trajectory(segment([0, 1, 2], [380.0, 380.0, 380.0])).refined_peak(1) == (380.0, 0.0)


def test_unevenly_spaced_samples_recover_an_exact_parabola() -> None:
    times = [0.0, 1.0, 1.5, 3.0]
    found = trajectory(segment(times, [100.0 - (t - 1.3) ** 2 for t in times]))
    value, when = found.refined_peak(1)
    assert value == pytest.approx(100.0, abs=1e-12)
    assert when == pytest.approx(1.3, abs=1e-12)


def test_a_vertex_outside_the_segment_is_ignored() -> None:
    """Concave and still rising at the end of the segment: the vertex lies beyond the
    last sample, where nothing is known. The end of a segment is a sample already."""
    times = [0.0, 1.0, 2.0]
    rising = trajectory(segment(times, [379.0 - (t - 5.0) ** 2 for t in times]))
    assert rising.refined_peak(1) == rising.peak(1) == (370.0, 2.0)


def test_repeated_sampling_instants_are_skipped_not_divided_by() -> None:
    found = trajectory(segment([0, 1, 1, 2], [370.0, 380.0, 380.0, 370.0]))
    assert found.refined_peak(1) == (380.0, 1.0)


def test_fewer_than_three_samples_return_the_largest_sample() -> None:
    assert trajectory(segment([0, 1], [370.0, 375.0])).refined_peak(1) == (375.0, 1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_states_are_an_error_not_a_maximum(bad: float) -> None:
    found = trajectory(segment([0, 1, 2], [370.0, bad, 370.0]))
    with pytest.raises(ValueError, match="non-finite"):
        found.refined_peak(1)
    with pytest.raises(ValueError, match="non-finite"):
        found.peak(1)


def test_the_estimate_is_never_below_the_largest_sample() -> None:
    rng = np.random.default_rng(12345)
    for _ in range(200):
        n_first, n_second = rng.integers(2, 12, size=2)
        first = np.sort(rng.uniform(0.0, 10.0, n_first))
        second = first[-1] + np.sort(rng.uniform(0.0, 10.0, n_second))
        second[0] = first[-1]
        found = trajectory(
            segment(list(first), list(rng.normal(360.0, 10.0, n_first))),
            segment(list(second), list(rng.normal(360.0, 10.0, n_second))),
        )
        value, when = found.refined_peak(1)
        assert np.isfinite(value) and np.isfinite(when)
        assert value >= found.peak(1)[0]
        assert first[0] <= when <= second[-1]
