"""The estimation of MR and MR_F (docs/m1_plan.md, sections 8.1, 8.7 and 13, I1).

Data are simulated with the modeller's own equations at values chosen here, on the truth
side of the tests, and reach the estimator only as observations. The values, the corners
and the seeds of the noise are declared below, before any fit was run on them.

Recovery is tested twice, as the plan asks: first without noise and from an exact initial
state, to the tolerance of the optimiser; then with the noise of D-020 and the rule of the
context mean, each estimate within four standard errors of its true value for every seed,
the standard errors being those of the sandwich, which include the error of the initial
state.
"""

import math

import numpy as np
import pytest

from m1_support import NOMINAL, modeller_run, observations, random_corners
from process_transfer.evaluation.budgets import split_budget
from process_transfer.evaluation.metrics import Role, evaluate
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import (
    CONTEXT_READINGS,
    P3_LAYOUT,
    Window,
    WindowData,
    find_windows,
    window_data,
)
from process_transfer.measurement.observations import Observations
from process_transfer.models.fitting import (
    BUDGET_EXHAUSTED,
    CONVERGED,
    DEFAULT_STARTS,
    NUMERICAL_FAILURE,
    FitSettings,
    Start,
    StartRecord,
    covariance,
    fit_mechanistic,
    fit_mr,
    fit_mr_f,
    select_start,
)
from process_transfer.models.mechanistic import (
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
)
from process_transfer.models.rollout import (
    EVALUATION_SETTINGS,
    RolloutSettings,
    predict_window,
    rollout,
)
from process_transfer.simulation.steady_state import find_steady_states

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, NOMINAL)
MODELLER = modeller_values()
TEXTBOOK = MechanisticParameters.textbook(MODELLER)
# Declared before any fit: values away from the textbook, which is where every fit starts
TRUE = MechanisticParameters.from_k_350(1.3 * TEXTBOOK.k_350, 9200.0, 0.85 * TEXTBOOK.ua)
CORNERS = random_corners(8, seed=2026)
NOISE_SEEDS = (101, 102, 103, 104, 105)
ONE_START = FitSettings(starts=DEFAULT_STARTS[:1])


def truth_rhs(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return MechanisticModel("truth", KNOWN, TRUE).rhs(x, u)


def windows_of(run: Observations) -> tuple[WindowData, ...]:
    return window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)


def coordinates(parameters: MechanisticParameters) -> np.ndarray:
    """ln k_350, E/R in K, ln UA: the units of the standard errors."""
    return np.array(
        [math.log(parameters.k_350), parameters.activation_temperature, math.log(parameters.ua)]
    )


def exact_context(run: Observations) -> Observations:
    """The run with every context reading replaced by the exact state at its onset, so that
    the context mean is the exact initial state."""
    measured = np.array(run.measured)
    for window in find_windows(run, NOMINAL, P3_LAYOUT).windows:
        measured[window.context_ticks.start : window.onset + 1] = run.measured[window.onset]
    return observations(run.inputs, measured=measured, run=run.run)


def test_without_noise_every_start_recovers_the_parameters_to_the_optimiser_tolerance():
    run = exact_context(modeller_run(truth_rhs, CORNERS, noise_seed=None))
    data = windows_of(run)
    assert len(data) == 8
    fit = fit_mechanistic("MR", data, KNOWN, MODELLER)
    assert fit.training_failure is None
    assert [record.outcome for record in fit.starts] == [CONVERGED] * len(DEFAULT_STARTS)
    for record in fit.starts:
        found = record.final.coordinates(None)
        assert np.all(np.abs(found - TRUE.coordinates(None)) < 1e-5), record.start.label
        assert record.objective < 1e-5
    assert fit.objective == min(record.objective for record in fit.starts)


def test_with_noise_every_estimate_is_within_four_sandwich_standard_errors():
    for seed in NOISE_SEEDS:
        data = windows_of(modeller_run(truth_rhs, CORNERS, noise_seed=seed))
        fit = fit_mechanistic("MR", data, KNOWN, MODELLER, ONE_START)
        assert fit.training_failure is None, seed
        spread = covariance(fit.parameters, data, KNOWN)
        errors = np.array(list(spread.standard_errors().values()))
        exact = np.array(list(spread.standard_errors(sandwich=False).values()))
        assert np.all(errors >= exact)  # the error of the initial state only adds
        z = (coordinates(fit.parameters) - coordinates(TRUE)) / errors
        assert np.all(np.abs(z) <= 4.0), (seed, z)


def test_with_e_over_r_fixed_the_other_two_are_recovered():
    run = exact_context(modeller_run(truth_rhs, CORNERS, noise_seed=None))
    settings = FitSettings(fixed_activation_temperature=9200.0, starts=DEFAULT_STARTS[:3])
    fit = fit_mechanistic("MR", windows_of(run), KNOWN, MODELLER, settings)
    assert fit.parameters.activation_temperature == 9200.0
    assert abs(math.log(fit.parameters.k_350 / TRUE.k_350)) < 1e-5
    assert abs(math.log(fit.parameters.ua / TRUE.ua)) < 1e-5
    assert fit.model(KNOWN).parameter_names == ("ln_k_350", "ln_ua")
    with pytest.raises(ValueError, match="move it"):
        FitSettings(fixed_activation_temperature=9200.0)


def test_the_loss_of_a_fit_is_the_score_of_the_evaluation_on_its_windows():
    data = windows_of(modeller_run(truth_rhs, CORNERS, noise_seed=NOISE_SEEDS[0]))
    fit = fit_mechanistic("MR", data, KNOWN, MODELLER, ONE_START)
    model = fit.model(KNOWN)
    scored = evaluate(data, [predict_window(model, d) for d in data], Role.FITTING)
    assert fit.objective == pytest.approx(scored.pooled.j, rel=1e-6)
    assert scored.pooled.excess_mse is None  # fitting data: no estimate against the truth
    assert fit.readings == 880


def steady_windows(state: np.ndarray, n: int) -> tuple[WindowData, ...]:
    """Windows of a run held at the nominal inputs from a steady state: nothing excites the
    plant, so only the steady state is seen."""
    return tuple(
        WindowData(
            window=Window("target.steady.d7200.n0", j, 10 + 120 * j, P3_LAYOUT),
            sample_period=6.0,
            noise_std=np.array([5.0, 0.5]),
            onset_time=60.0 + 720.0 * j,
            context=np.tile(state, (CONTEXT_READINGS, 1)),
            scored=np.tile(state, (110, 1)),
            inputs=np.tile(NOMINAL, (110, 1)),
        )
        for j in range(n)
    )


def test_a_steady_state_alone_does_not_determine_three_parameters():
    """At a steady state the two balances fix two combinations of k_350, E/R and UA; the
    third direction is not determined. The fit shows it: a singular value of the Jacobian
    at the level of the error of the integration, and endpoints that differ in E/R with the
    same objective."""
    (steady,) = find_steady_states(lambda x: truth_rhs(x, NOMINAL), c_a_upper=NOMINAL[1])
    data = steady_windows(steady.state, 3)
    fit = fit_mechanistic("MR", data, KNOWN, MODELLER)
    converged = [record for record in fit.starts if record.outcome == CONVERGED]
    assert len(converged) >= 2
    for record in converged:
        largest, *_, smallest = record.singular_values
        assert smallest / largest < 1e-6
        assert record.objective < 1e-6
    low, high = fit.spread()["activation_temperature"]
    assert high - low > 1000.0  # E/R is wherever each start took it


def test_every_start_that_fails_is_recorded_and_no_start_means_a_training_failure():
    data = windows_of(modeller_run(truth_rhs, CORNERS[:2], noise_seed=NOISE_SEEDS[0]))
    starved = FitSettings(rollout=RolloutSettings(max_rhs_evaluations=5))
    fit = fit_mechanistic("MR", data, KNOWN, MODELLER, starved)
    assert fit.parameters is None and fit.selected is None
    assert [record.outcome for record in fit.starts] == [NUMERICAL_FAILURE] * 5
    assert all("work limit" in record.message for record in fit.starts)
    assert "no start of MR converged" in fit.training_failure.reason
    with pytest.raises(ValueError, match="has no model"):
        fit.model(KNOWN)


def test_a_start_that_exhausts_its_budget_is_recorded_and_not_selected():
    data = windows_of(modeller_run(truth_rhs, CORNERS[:2], noise_seed=NOISE_SEEDS[0]))
    fit = fit_mechanistic("MR", data, KNOWN, MODELLER, FitSettings(max_evaluations=1))
    assert [record.outcome for record in fit.starts] == [BUDGET_EXHAUSTED] * 5
    assert all(record.final is not None for record in fit.starts)  # endpoints are kept
    assert fit.selected is None and fit.training_failure is not None


def record(outcome: str, objective: float | None) -> StartRecord:
    return StartRecord(Start("s"), TEXTBOOK, outcome, "", None, objective, None, 1, 1, 0.0)


def test_the_selection_rule_lowest_converged_objective_first_on_a_tie():
    records = [
        record(NUMERICAL_FAILURE, None),
        record(CONVERGED, 1.0),
        record(BUDGET_EXHAUSTED, 0.5),  # lower, and not selectable
        record(CONVERGED, 0.9),
        record(CONVERGED, 0.9),
    ]
    assert select_start(records) == 3
    assert select_start(records[:3]) == 1
    assert select_start([record(BUDGET_EXHAUSTED, 0.1), record(NUMERICAL_FAILURE, None)]) is None


def test_mr_f_never_reads_the_validation_part():
    """MR_F is fitted on F: whatever the readings of V say, its fit is the same bit for
    bit, while MR, which reads all the windows of the budget, sees them."""
    run = modeller_run(truth_rhs, CORNERS[:5], noise_seed=NOISE_SEEDS[1])
    split = split_budget(find_windows(run, NOMINAL, P3_LAYOUT), 5)
    fitting_rows = split.indices("fitting").rows_read
    measured = np.array(run.measured)
    outside = [tick for tick in range(run.n_samples) if tick not in fitting_rows]
    measured[outside] += np.array([40.0, 4.0])
    poisoned = observations(run.inputs, measured=measured, run=run.run)
    first = fit_mr_f(split, run, KNOWN, MODELLER, ONE_START)
    second = fit_mr_f(split, poisoned, KNOWN, MODELLER, ONE_START)
    assert first == second
    assert first.windows == tuple(window.key for window in split.fitting)
    assert first.name == "MR_F" and first.readings == 440
    comparator = fit_mr(split, run, KNOWN, MODELLER, ONE_START)
    assert comparator.windows == tuple(window.key for window in split.windows)
    assert fit_mr(split, poisoned, KNOWN, MODELLER, ONE_START) != comparator


def test_a_fit_repeats_exactly():
    data = windows_of(modeller_run(truth_rhs, CORNERS[:3], noise_seed=NOISE_SEEDS[2]))
    first = fit_mechanistic("MR", data, KNOWN, MODELLER)
    assert fit_mechanistic("MR", data, KNOWN, MODELLER) == first
    first_covariance = covariance(first.parameters, data, KNOWN)
    again = covariance(first.parameters, data, KNOWN)
    np.testing.assert_array_equal(first_covariance.sandwich, again.sandwich)


def test_what_a_fit_cannot_use_is_refused():
    data = windows_of(modeller_run(truth_rhs, CORNERS[:2], noise_seed=NOISE_SEEDS[0]))
    with pytest.raises(ValueError, match="at least one window"):
        fit_mechanistic("MR", (), KNOWN, MODELLER)
    with pytest.raises(ValueError, match="twice"):
        fit_mechanistic("MR", (data[0], data[0]), KNOWN, MODELLER)
    silent = WindowData(
        window=data[0].window,
        sample_period=6.0,
        noise_std=np.array([0.0, 0.5]),
        onset_time=data[0].onset_time,
        context=data[0].context,
        scored=data[0].scored,
        inputs=data[0].inputs,
    )
    with pytest.raises(ValueError, match="divides by the noise level"):
        fit_mechanistic("MR", (silent,), KNOWN, MODELLER)
    with pytest.raises(ValueError, match="share the noise levels"):
        fit_mechanistic("MR", (silent, data[1]), KNOWN, MODELLER)
    with pytest.raises(ValueError, match="at least one start"):
        FitSettings(starts=())
    with pytest.raises(ValueError, match="distinct"):
        FitSettings(starts=(Start("a"), Start("a")))
    with pytest.raises(ValueError, match="max_evaluations"):
        FitSettings(max_evaluations=0)
    with pytest.raises(ValueError, match="puts E/R"):
        Start("minus", activation_shift=-9000.0).point(TEXTBOOK)


def window_of_readings(value: float, sigma: tuple[float, float] = (5.0, 0.5)) -> WindowData:
    """A window of P3 at the nominal inputs whose every scored reading is ``value``."""
    return WindowData(
        window=Window("target.p3.e0.x1.n0", 1, 10, P3_LAYOUT),
        sample_period=6.0,
        noise_std=np.array(sigma),
        onset_time=60.0,
        context=np.tile([200.0, 355.0], (CONTEXT_READINGS, 1)),
        scored=np.full((110, 2), value),
        inputs=np.tile(NOMINAL, (110, 1)),
    )


def test_readings_that_cannot_be_normalised_are_refused_where_they_enter():
    """1e308 divided by a sigma of 0.1 is not a double: the loss is not defined on these
    data for any model, and the fit refuses them before any start."""
    with pytest.raises(ValueError, match="divided by the noise levels"):
        fit_mechanistic("MR", (window_of_readings(1e308, (0.1, 0.1)),), KNOWN, MODELLER)


def test_a_loss_that_is_not_representable_ends_every_start_as_a_numerical_failure():
    """Readings of 1e160 normalise to finite numbers, but the sum of the squares of the
    residuals overflows at every point: every start fails, none is selected, and the fit is
    a training failure, not a fit with an infinite objective."""
    settings = FitSettings(starts=DEFAULT_STARTS[:2])
    fit = fit_mechanistic("MR", (window_of_readings(1e160),), KNOWN, MODELLER, settings)
    assert [record.outcome for record in fit.starts] == [NUMERICAL_FAILURE] * 2
    assert all("not representable" in record.message for record in fit.starts)
    assert fit.selected is None and fit.parameters is None
    assert "no start of MR converged" in fit.training_failure.reason


def test_a_covariance_that_is_not_representable_says_so_instead_of_returning_it():
    """Noise levels of 1e-306 pass the boundary for readings of 1, but sensitivities of the
    order of a hundred divided by them are not doubles: there is no covariance to report."""
    spread = covariance(TRUE, (window_of_readings(1.0, (1e-306, 1e-306)),), KNOWN)
    assert spread.sandwich is None and spread.exact_initial_state is None
    assert "not representable" in spread.reason
    with pytest.raises(ValueError, match="no covariance"):
        spread.standard_errors()
    # noise levels of 1e200: the variance of a context mean overflows, and so the sandwich
    huge = covariance(TRUE, (window_of_readings(200.0, (1e200, 1e200)),), KNOWN)
    assert huge.sandwich is None and "not representable" in huge.reason


def window_the_start_reproduces(sigma: float) -> WindowData:
    """A window whose scored readings are the prediction of the textbook start itself, but
    for one reading moved by 1e-13, with noise levels ``sigma``: the residuals are finite
    and small in physical units, and divided by a tiny sigma they become huge."""
    base = window_of_readings(1.0, (sigma, sigma))
    theta = DEFAULT_STARTS[0].point(TEXTBOOK).coordinates(None)
    start = MechanisticModel("MR", KNOWN, MechanisticParameters.from_coordinates(theta, None))
    predicted = rollout(start, base.initial_state, base.inputs, 6.0, EVALUATION_SETTINGS).states
    scored = np.array(predicted)
    scored[-1, 0] += 1e-13
    return WindowData(
        window=base.window,
        sample_period=base.sample_period,
        noise_std=base.noise_std,
        onset_time=base.onset_time,
        context=base.context,
        scored=scored,
        inputs=base.inputs,
    )


def test_a_scale_of_the_optimiser_that_overflows_ends_the_start_instead_of_converging():
    """Codex's review of 9ea9a76. With sigmas of 1e-155 the residuals, the Jacobian, the loss
    and its gradient are all doubles, but x_scale="jac" makes least_squares form the norms of
    the columns of the Jacobian from their squares, which overflow: the scale became zero,
    the step vanished and the start was reported as converged by its tolerance on the step,
    or, with the warnings of this suite, the overflow escaped as an exception."""
    settings = FitSettings(starts=DEFAULT_STARTS[:2], max_evaluations=2)
    fit = fit_mechanistic("MR", (window_the_start_reproduces(1e-155),), KNOWN, MODELLER, settings)
    first, second = fit.starts
    assert first.outcome == NUMERICAL_FAILURE
    assert "norms of the columns of the Jacobian" in first.message
    # the second start begins elsewhere, where the loss itself is not a double: it is
    # recorded as well, so a failed start does not stop the others
    assert second.outcome == NUMERICAL_FAILURE and "loss is not representable" in second.message
    assert fit.selected is None and fit.training_failure is not None


def test_an_operation_of_the_optimiser_that_overflows_ends_the_start_instead_of_converging():
    """With sigmas of 3e-152 the norms of the columns are doubles as well, but the first
    radius of the trust region, the norm of the starting point times those norms, overflows
    inside least_squares: checking the quantities the loss hands over is not enough, so any
    operation of the optimiser that is not representable ends the start."""
    settings = FitSettings(starts=DEFAULT_STARTS[:1], max_evaluations=2)
    fit = fit_mechanistic("MR", (window_the_start_reproduces(3e-152),), KNOWN, MODELLER, settings)
    (record,) = fit.starts
    assert record.outcome == NUMERICAL_FAILURE
    assert "an operation of the optimiser" in record.message
    assert fit.selected is None and fit.training_failure is not None
