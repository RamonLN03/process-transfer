"""Metrics against readings (docs/m1_plan.md, section 9.1), checked against values computed
by hand, and the records of failures and the rules of paired comparison (section 9.3)."""

import math

import numpy as np
import pytest

from m1_support import NOMINAL, observations, p3_inputs, random_corners
from process_transfer.evaluation.metrics import Role, evaluate, score
from process_transfer.evaluation.outcomes import (
    IntegrationFailure,
    ReplicateOutcome,
    TrainingFailure,
    failure_table,
    paired_counts,
    paired_outcome,
    replicate_outcome,
)
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data

SIGMA = np.array([5.0, 0.5])


def two_windows():
    run = observations(p3_inputs(random_corners(2, 0)))
    return window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)


def test_a_score_computed_by_hand() -> None:
    errors = np.array([[1.0, 0.1], [-1.0, -0.1], [3.0, 0.3], [-3.0, 0.3]])
    result = score(errors, SIGMA, Role.HELD_OUT)
    # MSE: (1 + 1 + 9 + 9) / 4 = 5 and (0.01 + 0.01 + 0.09 + 0.09) / 4 = 0.05
    assert result.readings == 4
    assert result.mse == pytest.approx((5.0, 0.05), rel=1e-15)
    assert result.rmse == pytest.approx((math.sqrt(5.0), math.sqrt(0.05)), rel=1e-15)
    assert result.normalised_mse == pytest.approx((0.2, 0.2), rel=1e-15)
    assert result.j == pytest.approx(math.sqrt(0.2), rel=1e-15)
    assert result.excess_mse == pytest.approx((5.0 - 25.0, 0.05 - 0.25), rel=1e-14)
    assert result.excess_j_squared == pytest.approx(-0.8, rel=1e-14)


def test_scores_that_are_exact_in_binary() -> None:
    at_noise = score(np.array([[5.0, 0.5], [-5.0, -0.5]]), SIGMA, Role.HELD_OUT)
    assert (at_noise.j, at_noise.excess_mse, at_noise.excess_j_squared) == (1.0, (0.0, 0.0), 0.0)
    doubled = score(np.array([[10.0, 1.0], [10.0, -1.0]]), SIGMA, Role.HELD_OUT)
    assert doubled.mse == (100.0, 1.0) and doubled.normalised_mse == (4.0, 4.0)
    assert (doubled.j, doubled.excess_mse, doubled.excess_j_squared) == (2.0, (75.0, 0.75), 3.0)
    unequal = score(np.array([[10.0, 0.0]]), SIGMA, Role.HELD_OUT)  # (4 + 0) / 2 = 2
    assert unequal.j == math.sqrt(2.0)


def test_the_excess_over_the_noise_exists_only_for_held_out_readings() -> None:
    errors = np.array([[1.0, 0.1], [2.0, 0.2]])
    for role in (Role.FITTING, Role.SELECTION):
        result = score(errors, SIGMA, role)
        assert result.excess_mse is None and result.excess_j_squared is None
    assert score(errors, SIGMA, Role.HELD_OUT).excess_mse is not None
    with pytest.raises(ValueError, match="role"):
        score(errors, SIGMA, "held out")  # type: ignore[arg-type]


def test_a_negative_excess_is_reported_as_it_comes() -> None:
    result = score(np.zeros((3, 2)), SIGMA, Role.HELD_OUT)
    assert result.excess_mse == (-25.0, -0.25) and result.excess_j_squared == -1.0


def test_what_cannot_be_scored_is_refused() -> None:
    with pytest.raises(ValueError, match="exact sensor"):
        score(np.ones((2, 2)), np.array([0.0, 0.5]), Role.HELD_OUT)
    with pytest.raises(ValueError, match="integration failure"):
        score(np.array([[np.nan, 0.0]]), SIGMA, Role.HELD_OUT)
    with pytest.raises(ValueError, match="at least one row"):
        score(np.empty((0, 2)), SIGMA, Role.HELD_OUT)
    with pytest.raises(ValueError, match="not representable"):
        score(np.array([[1e200, 0.0]]), SIGMA, Role.HELD_OUT)  # an MSE of 1e400
    with pytest.raises(ValueError, match="not representable"):
        score(np.ones((1, 2)), np.array([1e-200, 0.5]), Role.HELD_OUT)  # errors of 1e200 sigmas


def test_an_evaluation_by_phase_and_by_window_computed_by_hand() -> None:
    windows = two_windows()
    predictions = []
    for data in windows:
        predicted = np.array(data.scored)
        predicted[:20, 0] += 10.0  # excursion: C_A off by 10
        predicted[20:70, 1] += 1.0  # return: T off by 1
        predictions.append(predicted)
    result = evaluate(windows, predictions, Role.HELD_OUT)
    phases = dict(result.by_phase)
    assert phases["excursion"].mse == (100.0, 0.0) and phases["excursion"].readings == 40
    assert phases["return"].mse == (0.0, 1.0) and phases["return"].readings == 100
    assert phases["settled"].mse == (0.0, 0.0) and phases["settled"].readings == 80
    # pooled: 20 of 110 readings off by 10 in C_A, 50 of 110 off by 1 in T
    assert result.pooled.mse == pytest.approx((100.0 * 20 / 110, 50 / 110), rel=1e-15)
    assert result.pooled.readings == 220
    assert [key for key, _ in result.by_window] == [data.key for data in windows]
    assert all(s.mse == result.pooled.mse for _, s in result.by_window)


def test_only_scored_readings_are_compared_never_the_context() -> None:
    windows = two_windows()
    predictions = [np.array(data.scored) + 1.0 for data in windows]
    before = evaluate(windows, predictions, Role.HELD_OUT)
    moved_context = [
        type(data)(
            window=data.window,
            sample_period=data.sample_period,
            noise_std=data.noise_std,
            onset_time=data.onset_time,
            context=data.context + 100.0,
            scored=data.scored,
            inputs=data.inputs,
        )
        for data in windows
    ]
    assert evaluate(moved_context, predictions, Role.HELD_OUT) == before


def test_an_evaluation_refuses_what_it_cannot_pool() -> None:
    windows = two_windows()
    predictions = [np.array(data.scored) for data in windows]
    with pytest.raises(ValueError, match="no window"):
        evaluate([], [], Role.HELD_OUT)
    with pytest.raises(ValueError, match="2 windows"):
        evaluate(windows, predictions[:1], Role.HELD_OUT)
    with pytest.raises(ValueError, match="twice"):
        evaluate([windows[0], windows[0]], predictions, Role.HELD_OUT)
    with pytest.raises(ValueError, match="shape"):
        evaluate(windows, [predictions[0][:-1], predictions[1]], Role.HELD_OUT)


def test_a_failure_loses_to_any_score_and_ties_with_another() -> None:
    assert paired_outcome(1.0, 2.0) == "first"
    assert paired_outcome(2.0, 1.0) == "second"
    assert paired_outcome(1.5, 1.5) == "tie"
    assert paired_outcome(None, 1e9) == "second"
    assert paired_outcome(1e9, None) == "first"
    assert paired_outcome(None, None) == "tie"
    with pytest.raises(ValueError, match="finite or absent"):
        paired_outcome(float("nan"), 1.0)


def outcome_with(results, model: str = "MR", replicate: str = "r1", training=None):
    windows = two_windows()
    return replicate_outcome(
        model, replicate, 2, windows, results, Role.HELD_OUT, training_failure=training
    )


def test_a_replicate_has_a_primary_score_only_when_every_window_completed() -> None:
    windows = two_windows()
    predicted = [np.array(data.scored) + 5.0 for data in windows]
    complete = outcome_with(predicted)
    assert complete.score == pytest.approx(math.sqrt((1.0 + 100.0) / 2.0), rel=1e-15)
    assert complete.conditional_on_success == complete.primary

    failure = IntegrationFailure(windows[1].key, "work limit", "reached 10 evaluations")
    partial = outcome_with([predicted[0], failure])
    assert partial.score is None and partial.primary is None
    assert partial.conditional_on_success.pooled.readings == 110  # labelled, kept apart
    assert partial.integration_failures == (failure,)

    untrained = outcome_with(None, training=TrainingFailure("no start converged"))
    assert untrained.score is None and untrained.conditional_on_success is None
    with pytest.raises(ValueError, match="no results"):
        outcome_with(predicted, training=TrainingFailure("no start converged"))


def test_the_record_of_a_replicate_is_consistent() -> None:
    complete = outcome_with([np.array(d.scored) for d in two_windows()])
    with pytest.raises(ValueError, match="exactly when"):
        ReplicateOutcome("MR", "r1", 2, 2, TrainingFailure("x"), (), complete.primary, None, ())


def test_paired_counts_keep_the_failed_replicates() -> None:
    windows = two_windows()
    good = [np.array(data.scored) + 1.0 for data in windows]
    bad = [np.array(data.scored) + 3.0 for data in windows]
    failed = [good[0], IntegrationFailure(windows[1].key, "integrator", "step size too small")]
    pairs = [
        (outcome_with(good, "HK", "r1"), outcome_with(bad, "MR", "r1")),  # HK wins
        (outcome_with(failed, "HK", "r2"), outcome_with(bad, "MR", "r2")),  # a failure loses
        (outcome_with(failed, "HK", "r3"), outcome_with(failed, "MR", "r3")),  # both fail
        (outcome_with(bad, "HK", "r4"), outcome_with(good, "MR", "r4")),  # MR wins
    ]
    counts = paired_counts(pairs)
    assert (counts.first_wins, counts.second_wins, counts.ties, counts.both_failed) == (1, 2, 1, 1)
    table = failure_table([first for first, _ in pairs])
    assert (table.replicates, table.scored, table.with_integration_failures) == (4, 2, 2)
    assert (table.failed_windows, table.training_failures) == (2, 0)
    with pytest.raises(ValueError, match="one replicate and budget"):
        paired_counts([(outcome_with(good, "HK", "r1"), outcome_with(good, "MR", "r2"))])


def test_scores_are_computed_without_overflow_whenever_they_are_representable() -> None:
    """Squares, means and normalisation are formed on values scaled by their largest
    magnitude, so that no intermediate overflows or underflows when the result does not."""
    large = score(np.array([[1e154, 1e154]]), np.array([1.0, 1.0]), Role.HELD_OUT)
    assert large.j == pytest.approx(1e154, rel=1e-15)
    assert large.mse == pytest.approx((1e308, 1e308), rel=1e-15)
    assert large.rmse == pytest.approx((1e154, 1e154), rel=1e-15)
    assert large.excess_mse == pytest.approx((1e308, 1e308), rel=1e-15)
    assert large.excess_j_squared == pytest.approx(1e308, rel=1e-15)
    small = score(np.array([[1.0, 1.0]]), np.array([1e200, 1e200]), Role.FITTING)
    # abs=0.0: pytest.approx allows 1e-12 in absolute terms by default, which zero would pass
    assert small.j > 0.0
    assert small.j == pytest.approx(1e-200, rel=1e-15, abs=0.0)
    assert small.mse == (1.0, 1.0)


def test_a_score_that_is_not_representable_is_refused_not_returned() -> None:
    # MSE - sigma^2 = 1 - 1e400: the excess over the noise of held-out readings
    with pytest.raises(ValueError, match="not representable"):
        score(np.array([[1.0, 1.0]]), np.array([1e200, 1e200]), Role.HELD_OUT)
    # errors of 1e400 sigmas
    with pytest.raises(ValueError, match="not representable"):
        score(np.array([[1e200, 1e200]]), np.array([1e-200, 1e-200]), Role.FITTING)
    # an MSE of 1e320, although its root would be representable
    with pytest.raises(ValueError, match="not representable"):
        score(np.array([[1e160, 0.0]]), np.array([1.0, 1.0]), Role.FITTING)


def test_the_normalised_rms_is_the_j_of_the_score_and_stays_representable() -> None:
    """J alone, shared with the training's criterion on V (Codex's audit of I3, F3): the J of
    ``score`` on ordinary errors, finite where the plain mean of squares overflows, and not
    lost to underflow where it is tiny."""
    from process_transfer.evaluation.metrics import normalised_rms

    rng = np.random.default_rng(7)
    errors = rng.normal(0.0, [5.0, 0.5], size=(110, 2))
    sigma = np.array([5.0, 0.5])
    assert normalised_rms(errors, sigma) == score(errors, sigma, Role.SELECTION).j
    for size in (1e153, 1e200):
        found = normalised_rms(np.full((110, 2), size), sigma)
        assert found == pytest.approx(size * math.sqrt((1 / 25 + 1 / 0.25) / 2), rel=1e-14)
    tiny = normalised_rms(np.full((110, 2), 1.0), np.array([1e170, 1e170]))
    assert tiny == pytest.approx(1e-170, rel=1e-14)
    with pytest.raises(ValueError, match="not representable"):
        normalised_rms(np.full((3, 2), 1e308), sigma)
    with pytest.raises(ValueError, match="must be finite"):
        normalised_rms(np.array([[np.inf, 0.0]]), sigma)
