"""The rollout of a window: its accuracy against far tighter integrations, the convention of
the inputs at their switching instants, its independence from the readings, sensitivities,
and failures returned as records with their causes."""

from collections.abc import Callable

import numpy as np
import pytest

from m1_support import NOMINAL, SIGMA, corner, observations, p3_inputs
from process_transfer.evaluation.physics import (
    check_implied_terms,
    implied_terms_along,
    validity_violations,
)
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, WindowData, find_windows, window_data
from process_transfer.models.mechanistic import (
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
    nominal_model,
)
from process_transfer.models.rollout import (
    EVALUATION_SETTINGS,
    RolloutFailure,
    RolloutSettings,
    predict_window,
    rollout,
)
from process_transfer.simulation.integration import InputSegment, simulate_piecewise

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, NOMINAL)
ALL_CORNERS = [[a, b, c, d] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1) for d in (-1, 1)]
# MR-like values: those that reproduce the target's nominal point better than the textbook
REESTIMATED = MechanisticParameters.from_k_350(0.0217, 9590.0, 1326.0)


def models():
    return [
        nominal_model(KNOWN, modeller_values()),
        MechanisticModel("MR", KNOWN, REESTIMATED),
    ]


def corner_windows(initial_state: np.ndarray) -> list[WindowData]:
    """The 16 corners of P3, each window started from ``initial_state``."""
    run = observations(p3_inputs(ALL_CORNERS), measured=np.tile(initial_state, (1931, 1)))
    return list(window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows))


def test_the_rollout_is_accurate_to_far_below_one_percent_of_sigma() -> None:
    """Against LSODA and DOP853 at rtol = 1e-12, over the 16 corners, for MN and MR."""
    tight = {
        "LSODA": RolloutSettings("LSODA", 1e-12, 1e-10, 10_000_000),
        "DOP853": RolloutSettings("DOP853", 1e-12, 1e-10, 10_000_000),
    }
    for model in models():
        for data in corner_windows(np.array([200.0, 355.0])):
            evaluated = predict_window(model, data)
            for settings in tight.values():
                reference = predict_window(model, data, settings)
                error = np.max(np.abs(evaluated - reference), axis=0) / np.array(SIGMA)
                assert np.all(error < 0.01), (model.name, data.key, error)


def test_the_rollout_agrees_with_the_integrator_of_the_simulation() -> None:
    """The same equation integrated by the code of M0, which restarts at every change of
    the inputs too, and stores every 0.1 s."""
    model = models()[1]
    x0 = np.array([180.0, 356.0])
    signs = [1, -1, 1, 1]
    segments = [InputSegment(120.0, corner(signs)), InputSegment(540.0, NOMINAL)]
    truth = simulate_piecewise(model.rhs, x0, segments, sample_period=0.1)
    on_sensor_instants = truth.states[60::60]  # 6 s, 12 s, ..., 660 s
    inputs = np.vstack([np.tile(corner(signs), (20, 1)), np.tile(NOMINAL, (90, 1))])
    predicted = rollout(model, x0, inputs, 6.0).states
    np.testing.assert_allclose(predicted, on_sensor_instants, rtol=1e-6, atol=0.0)


def test_an_input_acts_from_its_own_instant_on_and_not_before() -> None:
    """Row 50 acts over [300 s, 306 s). The states at 6 s ... 300 s do not see it; they are
    not equal bit for bit, because with a change at 300 s the first piece of the
    integration ends there and the adaptive integrator takes other steps, but they agree to
    the accuracy of the integration, while the first state after it moves by about a
    sigma."""
    model = models()[0]
    x0 = np.array([250.0, 350.0])
    inputs = np.tile(NOMINAL, (110, 1))
    base = rollout(model, x0, inputs, 6.0).states
    changed = inputs.copy()
    changed[50] = corner([1, 1, 1, 1])
    moved = rollout(model, x0, changed, 6.0).states
    sigma = np.array(SIGMA)
    assert np.all(np.abs(moved[:50] - base[:50]) < 1e-6 * sigma)
    assert np.all(np.abs(moved[50] - base[50]) > 0.1 * sigma)
    # a single row is a segment of its own
    one_row = np.tile(NOMINAL, (1, 1))
    assert rollout(model, x0, one_row, 6.0).states.shape == (1, 2)


def test_a_prediction_reads_the_initial_state_and_the_inputs_and_nothing_else() -> None:
    (data,) = corner_windows(np.array([200.0, 355.0]))[:1]
    moved = WindowData(
        window=data.window,
        sample_period=data.sample_period,
        noise_std=data.noise_std,
        onset_time=data.onset_time + 1000.0,
        context=data.context,
        scored=data.scored + 50.0,
        inputs=data.inputs,
    )
    model = models()[0]
    np.testing.assert_array_equal(predict_window(model, data), predict_window(model, moved))


def test_structured_models_satisfy_the_validity_bounds_and_the_implied_terms() -> None:
    for model in models():
        for data in corner_windows(np.array([200.0, 355.0])):
            predicted = predict_window(model, data)
            assert validity_violations(data, predicted, KNOWN) == ()
            terms = implied_terms_along(model.rhs, data, predicted, KNOWN)
            check = check_implied_terms(terms)
            assert (check.negative_rate, check.incompatible_heat_flow) == (0, 0)


def test_the_sensitivities_agree_with_finite_differences_of_rollouts() -> None:
    """Central differences of rollouts at rtol = 1e-12. Each step balances the truncation of
    the difference, which grows as its square, against the error of the two integrations
    divided by the step, about 350e-12 / h: the sensitivity to (E/R) / T_ref is small, a
    factor 1 - T_ref / T of about 0.014 below that to ln k_350, and needs the larger step.
    At a step of 1e-5 that noise reached 1 % of it (investigated while writing this test:
    the difference shrank as the step grew, the same with LSODA, DOP853 and Radau)."""
    model = models()[1]
    x0 = np.array([190.0, 355.0])
    inputs = np.vstack([np.tile(corner([1, -1, 1, 1]), (20, 1)), np.tile(NOMINAL, (90, 1))])
    tight = RolloutSettings("LSODA", 1e-12, 1e-10, 10_000_000)
    result = rollout(model, x0, inputs, 6.0, tight, True, True)
    theta = REESTIMATED.coordinates(None)
    for j, step in enumerate((1e-4, 3e-3, 1e-4)):
        up, down = theta.copy(), theta.copy()
        up[j] += step
        down[j] -= step
        states = [
            rollout(
                MechanisticModel("MR", KNOWN, MechanisticParameters.from_coordinates(t, None)),
                x0,
                inputs,
                6.0,
                tight,
            ).states
            for t in (up, down)
        ]
        numeric = (states[0] - states[1]) / (2 * step)
        np.testing.assert_allclose(
            result.parameter_sensitivities[:, :, j], numeric, rtol=1e-4, atol=1e-5
        )
    for j, step in enumerate((1e-2, 1e-3)):
        up, down = x0.copy(), x0.copy()
        up[j] += step
        down[j] -= step
        numeric = (
            rollout(model, up, inputs, 6.0, tight).states
            - rollout(model, down, inputs, 6.0, tight).states
        ) / (2 * step)
        np.testing.assert_allclose(
            result.initial_state_sensitivities[:, :, j], numeric, rtol=1e-4, atol=1e-6
        )
    # the states come out the same with or without sensitivities, to the tolerance
    plain = rollout(model, x0, inputs, 6.0, tight).states
    np.testing.assert_allclose(result.states, plain, rtol=1e-9)


class Toy:
    """A two-state model for the failures a rollout must record."""

    name = "toy"

    def __init__(self, f: Callable[[np.ndarray], np.ndarray]) -> None:
        self.f = f

    def rhs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.f(x)

    def initial_state_problem(self, x0: np.ndarray) -> str | None:
        return None


@pytest.mark.parametrize(
    ("f", "cause", "detail"),
    [
        (lambda x: np.array([np.float64(1e308) * x[0], 0.0]), "right-hand side", "overflow"),
        (lambda x: np.array([np.nan, 0.0]), "right-hand side", "not finite"),
        (lambda x: np.array([np.inf, 0.0]), "right-hand side", "not finite"),
        (lambda x: x**2, "right-hand side", "overflow"),  # finite-time blow-up
    ],
)
def test_a_failed_rollout_is_a_record_with_its_cause(
    f: Callable[[np.ndarray], np.ndarray], cause: str, detail: str
) -> None:
    result = rollout(Toy(f), np.array([10.0, 1.0]), np.tile(NOMINAL, (110, 1)), 6.0)
    assert isinstance(result, RolloutFailure)
    assert result.cause == cause and detail in result.detail


def test_the_limit_on_evaluations_ends_a_rollout_that_makes_no_progress() -> None:
    settings = RolloutSettings(max_rhs_evaluations=50)
    result = rollout(
        models()[0], np.array([200.0, 355.0]), np.tile(NOMINAL, (110, 1)), 6.0, settings
    )
    assert isinstance(result, RolloutFailure) and result.cause == "work limit"
    assert "50 evaluations" in result.detail


def test_an_initial_state_outside_the_domain_is_a_failure_before_any_step() -> None:
    result = rollout(models()[0], np.array([200.0, -1.0]), np.tile(NOMINAL, (5, 1)), 6.0)
    assert isinstance(result, RolloutFailure) and result.cause == "initial state"
    (data,) = corner_windows(np.array([200.0, 355.0]))[:1]
    frozen = WindowData(
        window=data.window,
        sample_period=data.sample_period,
        noise_std=data.noise_std,
        onset_time=data.onset_time,
        context=np.tile([200.0, -1.0], (10, 1)),
        scored=data.scored,
        inputs=data.inputs,
    )
    failure = predict_window(models()[0], frozen)
    assert failure.window == data.key and failure.cause == "initial state"


def test_arguments_and_settings_are_checked_where_they_enter() -> None:
    model = models()[0]
    with pytest.raises(ValueError, match="initial state"):
        rollout(model, np.array([1.0, 2.0, 3.0]), np.tile(NOMINAL, (5, 1)), 6.0)
    with pytest.raises(ValueError, match="4 columns"):
        rollout(model, np.array([1.0, 350.0]), np.ones((5, 3)), 6.0)
    with pytest.raises(ValueError, match="sample_period"):
        rollout(model, np.array([1.0, 350.0]), np.tile(NOMINAL, (5, 1)), 0.0)
    with pytest.raises(ValueError, match="no Jacobians"):
        rollout(
            Toy(lambda x: -x),
            np.array([1.0, 2.0]),
            np.tile(NOMINAL, (5, 1)),
            6.0,
            EVALUATION_SETTINGS,
            True,
        )
    for bad in (
        {"rtol": 0.0},
        {"atol": -1.0},
        {"max_rhs_evaluations": 0},
        {"max_rhs_evaluations": True},
    ):
        with pytest.raises(ValueError):
            RolloutSettings(**bad)
