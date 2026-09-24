"""Failures, the outcome of a replicate, and paired comparison (``docs/m1_plan.md``,
sections 8.7, 9.3 and 9.7).

Three kinds of failure are kept apart:

* a training failure: there is no model for that model, replicate and budget;
* an integration failure: a window whose rollout did not complete, because the integrator
  reported a failure, a state stopped being finite, the right-hand side could not be
  evaluated, or the limit on its evaluations was reached;
* a physical violation: a completed rollout whose prediction breaks a validity bound
  (``physics``). It keeps its score, and the violation is reported beside it.

The primary score of a model, replicate and budget on a set of windows exists only if the
model was trained and every window completed. Otherwise that replicate is a failure of the
model at that budget: no error is invented for it and it is not dropped. In a paired
comparison a failure loses to any score and ties with another failure; two scores are
compared as numbers, the lower winning and equal scores tying. The score over the windows
that did complete is kept apart and labelled as conditional on success.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from process_transfer.cstr_variables import FloatArray
from process_transfer.evaluation.metrics import Evaluation, Role, evaluate
from process_transfer.evaluation.windows import WindowData


@dataclass(frozen=True)
class TrainingFailure:
    """No model was obtained. ``reason`` says why, in words and with the numbers."""

    reason: str


@dataclass(frozen=True)
class IntegrationFailure:
    """A rollout of one window that did not complete."""

    window: tuple[str, int]  # (run, excursion)
    cause: str  # "integrator", "non-finite state", "right-hand side", "work limit", ...
    detail: str


@dataclass(frozen=True)
class PhysicalViolation:
    """A completed prediction outside a validity bound, at one scored instant."""

    window: tuple[str, int]
    tick: int
    bound: str
    value: float
    limit: float


@dataclass(frozen=True)
class ReplicateOutcome:
    """What one model, replicate and budget obtained on a set of windows."""

    model: str
    replicate: str
    budget: int
    windows: int  # windows in the set evaluated
    training_failure: TrainingFailure | None
    integration_failures: tuple[IntegrationFailure, ...]
    primary: Evaluation | None  # only when trained and every window completed
    conditional_on_success: Evaluation | None  # over the windows that completed, labelled
    violations: tuple[PhysicalViolation, ...]

    def __post_init__(self) -> None:
        trained = self.training_failure is None
        complete = trained and not self.integration_failures
        if (self.primary is not None) != complete:
            raise ValueError(
                "a primary score exists exactly when the model was trained and every window "
                "completed"
            )
        if not trained and (self.integration_failures or self.violations):
            raise ValueError("a model that was not trained has no rollouts to fail or to judge")
        if len(self.integration_failures) > self.windows:
            raise ValueError("more failed windows than windows")

    @property
    def score(self) -> float | None:
        """The primary score J, or None for a failure of this replicate."""
        return None if self.primary is None else self.primary.pooled.j


def replicate_outcome(
    model: str,
    replicate: str,
    budget: int,
    windows: Sequence[WindowData],
    results: Sequence[FloatArray | IntegrationFailure] | None,
    role: Role,
    violations: Sequence[PhysicalViolation] = (),
    training_failure: TrainingFailure | None = None,
) -> ReplicateOutcome:
    """The outcome of a replicate from the result of each window's rollout: the predicted
    states at its scored instants, or the record of its failure. A training failure has no
    results."""
    if len(windows) == 0:
        raise ValueError("an outcome is over at least one window")
    if training_failure is not None:
        if results is not None:
            raise ValueError("a model that was not trained has no results")
        return ReplicateOutcome(
            model, replicate, budget, len(windows), training_failure, (), None, None, ()
        )
    if results is None or len(results) != len(windows):
        raise ValueError("one result per window is needed")
    failures = tuple(result for result in results if isinstance(result, IntegrationFailure))
    completed = [
        (data, np.asarray(result))
        for data, result in zip(windows, results, strict=True)
        if not isinstance(result, IntegrationFailure)
    ]
    conditional = (
        evaluate([data for data, _ in completed], [p for _, p in completed], role)
        if completed
        else None
    )
    return ReplicateOutcome(
        model=model,
        replicate=replicate,
        budget=budget,
        windows=len(windows),
        training_failure=None,
        integration_failures=failures,
        primary=conditional if not failures else None,
        conditional_on_success=conditional,
        violations=tuple(violations),
    )


FIRST, SECOND, TIE = "first", "second", "tie"


def paired_outcome(first: float | None, second: float | None) -> str:
    """Which of two primary scores of one replicate wins: ``"first"``, ``"second"`` or
    ``"tie"``. None is a failure. A failure loses to any score; two failures tie."""
    for score in (first, second):
        if score is not None and not np.isfinite(score):
            raise ValueError(f"a primary score is finite or absent, got {score!r}")
    if first is None and second is None:
        return TIE
    if first is None:
        return SECOND
    if second is None:
        return FIRST
    if first < second:
        return FIRST
    if second < first:
        return SECOND
    return TIE


@dataclass(frozen=True)
class PairedCounts:
    first_wins: int
    second_wins: int
    ties: int
    both_failed: int  # among the ties


def paired_counts(pairs: Sequence[tuple[ReplicateOutcome, ReplicateOutcome]]) -> PairedCounts:
    """Paired outcomes over replicates, each pair being the same replicate and budget."""
    counts = {FIRST: 0, SECOND: 0, TIE: 0}
    both_failed = 0
    for first, second in pairs:
        if (first.replicate, first.budget) != (second.replicate, second.budget):
            raise ValueError(
                f"a pair compares one replicate and budget: {first.replicate!r} at "
                f"{first.budget} against {second.replicate!r} at {second.budget}"
            )
        counts[paired_outcome(first.score, second.score)] += 1
        both_failed += first.score is None and second.score is None
    return PairedCounts(counts[FIRST], counts[SECOND], counts[TIE], both_failed)


@dataclass(frozen=True)
class FailureTable:
    """For one model and budget: how many replicates scored and how the others failed."""

    replicates: int
    scored: int
    training_failures: int
    with_integration_failures: int
    failed_windows: int
    with_violations: int


def failure_table(outcomes: Sequence[ReplicateOutcome]) -> FailureTable:
    return FailureTable(
        replicates=len(outcomes),
        scored=sum(outcome.score is not None for outcome in outcomes),
        training_failures=sum(outcome.training_failure is not None for outcome in outcomes),
        with_integration_failures=sum(bool(outcome.integration_failures) for outcome in outcomes),
        failed_windows=sum(len(outcome.integration_failures) for outcome in outcomes),
        with_violations=sum(bool(outcome.violations) for outcome in outcomes),
    )
