"""Arguments of the steady-state search and of the numerical Jacobian.

Regression tests. A right-hand side returning NaN used to yield an empty list, read
as "no steady state", because a NaN never changes sign. A temperature range starting
at 0 K divided by zero inside the rate law, a reversed range and a one-point grid
were accepted, and a zero step gave a Jacobian of NaN.
"""

import numpy as np
import pytest

from process_transfer.simulation.steady_state import find_steady_states, numerical_jacobian


def linear_rhs(x: np.ndarray) -> np.ndarray:
    return np.array([100.0 - x[0], 400.0 - x[1]])


def test_a_non_finite_right_hand_side_is_an_error_not_an_empty_result() -> None:
    def broken_energy_balance(x: np.ndarray) -> np.ndarray:
        return np.array([100.0 - x[0], np.nan])

    def broken_mass_balance(x: np.ndarray) -> np.ndarray:
        return np.array([np.nan, 400.0 - x[1]])

    for rhs in (broken_energy_balance, broken_mass_balance):
        with pytest.raises(ValueError, match="not finite at T ="):
            find_steady_states(rhs, 500.0, (300.0, 500.0), 11)


@pytest.mark.parametrize(
    ("temperature_range", "message"),
    [
        ((0.0, 400.0), "lower end of temperature_range"),
        ((-10.0, 400.0), "lower end of temperature_range"),
        ((400.0, 300.0), "must be increasing"),
        ((300.0, 300.0), "must be increasing"),
        ((300.0, np.inf), "upper end of temperature_range"),
        ((np.nan, 400.0), "lower end of temperature_range"),
    ],
)
def test_the_temperature_range_must_be_positive_finite_and_increasing(
    temperature_range: tuple[float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        find_steady_states(linear_rhs, 500.0, temperature_range, 11)


@pytest.mark.parametrize("n_grid", [0, 1, 2.5])
def test_the_grid_needs_at_least_two_points(n_grid: float) -> None:
    with pytest.raises(ValueError, match="n_grid"):
        find_steady_states(linear_rhs, 500.0, (300.0, 500.0), n_grid)


@pytest.mark.parametrize("c_a_upper", [0.0, -1.0, np.nan, np.inf])
def test_the_concentration_bound_must_be_positive_and_finite(c_a_upper: float) -> None:
    with pytest.raises(ValueError, match="c_a_upper"):
        find_steady_states(linear_rhs, c_a_upper)


def test_a_concentration_bound_below_the_root_explains_itself() -> None:
    with pytest.raises(ValueError, match="does not change sign"):
        find_steady_states(linear_rhs, 50.0, (300.0, 500.0), 11)  # the root is at C_A = 100


def test_the_jacobian_rejects_a_zero_step_a_non_finite_state_and_a_non_finite_rhs() -> None:
    state = np.array([100.0, 400.0])
    with pytest.raises(ValueError, match="rel_step"):
        numerical_jacobian(linear_rhs, state, rel_step=0.0)
    with pytest.raises(ValueError, match="state must be finite"):
        numerical_jacobian(linear_rhs, np.array([np.nan, 400.0]))
    with pytest.raises(ValueError, match="not finite around"):
        numerical_jacobian(lambda x: np.array([np.inf, 0.0]), state)
    np.testing.assert_allclose(numerical_jacobian(linear_rhs, state), -np.eye(2), atol=1e-9)
