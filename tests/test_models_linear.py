"""BL, the linear black box (docs/m1_plan.md, sections 8.3 and 8.7): its excited
directions, its convention on the directions the data do not excite, its derivatives and
its fit. The data are simulated on the truth side of the tests by a linear plant."""

import numpy as np
import pytest

from m1_support import AMPLITUDES, NOMINAL, corner, modeller_run
from process_transfer.evaluation.budgets import integer_rank
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.models.fitting import BUDGET_EXHAUSTED, CONVERGED
from process_transfer.models.linear import (
    LinearCoordinates,
    LinearModel,
    LinearParameters,
    excited_directions,
    fit_linear,
)
from process_transfer.models.rollout import rollout

# a stable linear plant, with an input matrix that moves the states along every input
A_TRUE = np.array([[-0.030, -0.8], [0.004, -0.012]])  # 1/s, and mol/m^3 per K per s
X_E = np.array([190.0, 355.0])
B_LEVEL = np.array([[-2.0, 1.5, -0.5, 0.3], [0.05, 0.02, 0.06, 0.08]])  # per level, per s
B_TRUE = B_LEVEL / AMPLITUDES  # per unit of each input


def linear_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return A_TRUE @ (x - X_E) + B_TRUE @ (u - NOMINAL)


COORDINATES = LinearCoordinates(2, np.array([5.0, 0.5]), np.array([180.0, 352.0]))

class LinearPlant:
    """The linear plant as a model of the rollout."""

    name = "linear plant"

    def rhs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return linear_plant(x, u)

    def initial_state_problem(self, x0: np.ndarray) -> None:
        return None


# two corners whose sign vectors span two of the four directions
RANK_TWO = [(1, 1, -1, -1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, 1, -1)]


def windows(corners, noise_seed=None):  # noqa: ANN001, ANN201
    run = modeller_run(linear_plant, corners, noise_seed=noise_seed, initial_state=X_E)
    return window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)


def test_the_excited_directions_are_the_span_of_the_levels_of_the_rows() -> None:
    data = windows(RANK_TWO)
    directions = excited_directions(data, NOMINAL)
    assert np.allclose(directions.amplitudes, AMPLITUDES, rtol=1e-12, atol=0.0)
    assert directions.rank == 2 == integer_rank(sorted(set(RANK_TWO)))
    q = directions.basis
    assert q.shape == (4, 2) and np.allclose(q.T @ q, np.eye(2), atol=1e-14)
    for signs in RANK_TWO:
        level = np.array(signs, dtype=np.float64)
        assert np.allclose(q @ (q.T @ level), level, atol=1e-14)
    # the nominal rows add the zero vector, and the corners their signs, exactly
    assert set(directions.levels) == {(0, 0, 0, 0), *RANK_TWO}


def test_an_input_that_never_moves_has_no_effect() -> None:
    """A design that never moves T_c: its amplitude is written as one, its level is zero,
    and its column of B is zero whatever C is."""
    data = windows([(1, 1, 1, 0), (-1, 1, -1, 0)])
    directions = excited_directions(data, NOMINAL)
    assert directions.amplitudes[3] == 1.0 and directions.rank == 2
    rng = np.random.default_rng(2)
    parameters = LinearParameters(A_TRUE, X_E, rng.standard_normal((2, 2)))
    model = LinearModel("BL", directions, parameters, COORDINATES)
    assert np.all(model.input_matrix[:, 3] == 0.0)


def test_b_is_zero_on_the_directions_the_data_do_not_excite() -> None:
    """The minimum-norm convention of section 8.7: whatever C, B moves the states only along
    the span of the excited levels."""
    directions = excited_directions(windows(RANK_TWO), NOMINAL)
    rng = np.random.default_rng(4)
    parameters = LinearParameters(A_TRUE, X_E, rng.standard_normal((2, 2)))
    model = LinearModel("BL", directions, parameters, COORDINATES)
    q = directions.basis
    unexcited = np.linalg.svd(q.T)[2][2:]  # an orthonormal basis of the complement
    x = np.array([150.0, 360.0])
    for v in unexcited:
        u = NOMINAL + AMPLITUDES * v
        assert np.allclose(model.rhs(x, u), A_TRUE @ (x - X_E), rtol=0.0, atol=1e-12)
    assert np.linalg.matrix_rank(model.input_matrix * AMPLITUDES) == 2


def test_the_derivatives_of_bl_agree_with_central_differences() -> None:
    directions = excited_directions(windows(RANK_TWO), NOMINAL)
    rng = np.random.default_rng(6)
    theta = COORDINATES.theta(LinearParameters(A_TRUE, X_E, rng.standard_normal((2, 2))))
    assert np.allclose(COORDINATES.theta(COORDINATES.parameters(theta)), theta, rtol=1e-14)
    x, u = np.array([170.0, 358.0]), corner((1, -1, 1, -1))

    def rhs(t: np.ndarray, x: np.ndarray = x) -> np.ndarray:
        model = LinearModel("BL", directions, COORDINATES.parameters(t), COORDINATES)
        return model.rhs(x, u)

    model = LinearModel("BL", directions, COORDINATES.parameters(theta), COORDINATES)
    jacobian = model.jacobian_parameters(x, u)
    for i in range(len(theta)):
        step = 1e-6 * max(1.0, abs(theta[i]))
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        difference = (rhs(up) - rhs(down)) / (2.0 * step)
        assert np.allclose(difference, jacobian[:, i], rtol=1e-7, atol=1e-9), i
    state = model.jacobian_state(x, u)
    for j in range(2):
        dx = np.zeros(2)
        dx[j] = 1e-3
        difference = (model.rhs(x + dx, u) - model.rhs(x - dx, u)) / 2e-3
        assert np.allclose(difference, state[:, j], rtol=1e-9, atol=1e-12)


def test_bl_recovers_a_linear_plant_along_what_its_data_excite() -> None:
    """Noise-free data of a linear plant, from two directions of the inputs: the fit finds
    A and x_e, and B on the span of the excited levels, which is all the data determine;
    on the complement B is zero by convention, whatever the plant does there."""
    data = windows(RANK_TWO)
    fit = fit_linear("BL", data, NOMINAL)
    # the data were integrated by the simulation's integrator and the fit by the reference
    # rollout: their difference, some 4e-6 sigma, is the floor of J
    assert fit.training_failure is None and fit.objective < 1e-4
    assert all(start.outcome == CONVERGED for start in fit.starts)
    found = fit.parameters
    assert np.allclose(found.a, A_TRUE, rtol=1e-5)
    assert np.allclose(found.equilibrium, X_E, rtol=1e-8)
    q = fit.directions.basis
    model = fit.model()
    assert np.allclose(model.input_matrix * AMPLITUDES, B_LEVEL @ q @ q.T, rtol=1e-5, atol=1e-8)
    # a window along a direction the data did not excite: BL predicts what it predicts with
    # the inputs at nominal, whatever the plant would do
    unexcited = np.linalg.svd(q.T)[2][2]
    inputs = np.tile(NOMINAL, (110, 1))
    inputs[:20] = NOMINAL + AMPLITUDES * unexcited
    start = np.array([185.0, 356.0])
    predicted = rollout(model, start, inputs, 6.0).states
    at_nominal = rollout(model, start, np.tile(NOMINAL, (110, 1)), 6.0).states
    # equal up to the integrator, which restarts at the changes of the inputs of one of them
    assert np.allclose(predicted, at_nominal, rtol=0.0, atol=1e-4)
    # the plant itself does respond along that direction
    assert not np.allclose(rollout(LinearPlant(), start, inputs, 6.0).states, at_nominal)


def test_a_fit_whose_starts_all_stop_short_is_a_training_failure() -> None:
    fit = fit_linear("BL", windows(RANK_TWO[:2], noise_seed=3), NOMINAL, max_evaluations=1)
    assert fit.parameters is None and fit.selected is None
    assert all(start.outcome == BUDGET_EXHAUSTED for start in fit.starts)
    assert "no start of BL converged" in fit.training_failure.reason
    with pytest.raises(ValueError, match="has no model"):
        fit.model()


def test_invalid_nominal_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="nominal inputs"):
        excited_directions(windows(RANK_TWO[:2]), NOMINAL[:3])
