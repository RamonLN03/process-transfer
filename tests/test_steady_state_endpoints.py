"""Roots on grid points and at the ends of the scanned range.

Regression tests for a confirmed defect: the scan tested ``residual == 0`` only on
the left point of each interval, so a steady state lying exactly on the upper end of
the temperature range was never reported."""

import numpy as np
import pytest

from process_transfer.simulation.steady_state import find_steady_states

ROOT = (100.0, 400.0)  # C_A, T


def linear_rhs(x: np.ndarray) -> np.ndarray:
    """A synthetic system with a single steady state at C_A = 100, T = 400 and
    residuals that are exactly zero there."""
    return np.array([ROOT[0] - x[0], ROOT[1] - x[1]])


@pytest.mark.parametrize(
    ("temperature_range", "where"),
    [
        ((300.0, 400.0), "upper end"),
        ((400.0, 500.0), "lower end"),
        ((300.0, 500.0), "interior grid point"),
        ((300.5, 500.0), "between grid points"),
    ],
)
def test_a_root_is_found_exactly_once_wherever_it_lies(
    temperature_range: tuple[float, float], where: str
) -> None:
    found = find_steady_states(
        linear_rhs, c_a_upper=500.0, temperature_range=temperature_range, n_grid=101
    )
    assert len(found) == 1, where
    assert found[0].c_a == pytest.approx(ROOT[0], abs=1e-9)
    assert found[0].temperature == pytest.approx(ROOT[1], abs=1e-9)


def test_a_range_that_excludes_the_root_finds_nothing() -> None:
    assert find_steady_states(linear_rhs, 500.0, temperature_range=(300.0, 399.0), n_grid=100) == []
    assert find_steady_states(linear_rhs, 500.0, temperature_range=(401.0, 500.0), n_grid=100) == []


def test_results_are_sorted_by_temperature_with_roots_on_and_between_grid_points() -> None:
    """Three roots: one exactly on the lower end, one between grid points, one exactly
    on the upper end. The energy residual is a cubic in T; C_A is pinned at 100."""
    roots = (300.0, 350.25, 400.0)

    def cubic_rhs(x: np.ndarray) -> np.ndarray:
        energy = -(x[1] - roots[0]) * (x[1] - roots[1]) * (x[1] - roots[2])
        return np.array([100.0 - x[0], energy])

    found = find_steady_states(cubic_rhs, 500.0, temperature_range=(300.0, 400.0), n_grid=101)
    assert [s.temperature for s in found] == pytest.approx(list(roots), abs=1e-8)


def test_a_tangent_root_between_grid_points_is_a_documented_blind_spot() -> None:
    """A double root changes no sign. The scan misses it unless it lies on a grid
    point; this is a stated limitation, recorded here so that it is not forgotten."""

    def tangent_rhs(x: np.ndarray) -> np.ndarray:
        return np.array([100.0 - x[0], -((x[1] - 350.25) ** 2)])

    found = find_steady_states(tangent_rhs, 500.0, temperature_range=(300.0, 400.0), n_grid=101)
    assert found == []
