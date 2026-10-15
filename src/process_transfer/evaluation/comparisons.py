"""Paired comparisons of the benchmark of M1, as the draft registration of M1-E03 proposes
(``docs/m1_e03_registration_draft.md``, section 7). Pure functions of scores: nothing is
read, simulated or fitted here, and the levels, margins and minimum counts are arguments,
since they are pending approval.

A score is E = J^2 - 1 of one model, replicate, budget and test set: a finite number,
negative values included, or None when the model has no score there (a training failure,
or a test window whose rollout failed; plan, section 9.3). A failure is not given a score.

Ordinal outcome and magnitude are kept apart. The ordinal outcome of a replicate is that of
``outcomes.paired_outcome``: a failure loses to a score and ties with a failure. The
magnitude, the difference of E, exists only where both models have a score, and is
summarised over those replicates alone, with their number.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from process_transfer.evaluation.outcomes import FIRST, SECOND, paired_outcome

CANDIDATE, COMPARATOR = "candidate", "comparator"
BOTH, ONLY_CANDIDATE, ONLY_COMPARATOR, NEITHER = (
    "both completed",
    "only the candidate completed",
    "only the comparator completed",
    "both failed",
)
RELEVANT = "relevant"
BELOW_MARGIN = "below the relevance margin"
NOT_ESTABLISHED = "magnitude not established"
NOT_COMPUTABLE = "magnitude not computable"


def _score(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or None for a failure, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Pair:
    """One replicate: which model is better in order, and by how much where both scored."""

    kind: str
    winner: str | None  # CANDIDATE, COMPARATOR, or None for a tie
    difference: float | None  # E of the comparator minus E of the candidate; None unless BOTH


def pair(candidate: object, comparator: object) -> Pair:
    a, b = _score("candidate", candidate), _score("comparator", comparator)
    order = paired_outcome(a, b)
    winner = CANDIDATE if order == FIRST else COMPARATOR if order == SECOND else None
    if a is None or b is None:
        kind = (
            NEITHER if a is None and b is None else ONLY_CANDIDATE if b is None else ONLY_COMPARATOR
        )
        return Pair(kind, winner, None)
    difference = b - a
    if not math.isfinite(difference):
        raise ValueError(f"the difference of the scores {b!r} and {a!r} is not representable")
    return Pair(BOTH, winner, difference)


def one_sided_p(successes: int, trials: int) -> float:
    """P(X >= successes) for X binomial with ``trials`` trials of probability 1/2, exactly.
    With no trials there is no evidence, and the p-value is 1."""
    if trials < 0 or not 0 <= successes <= max(trials, 0):
        raise ValueError(f"need 0 <= successes <= trials, got {successes} of {trials}")
    if trials == 0:
        return 1.0
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / 2**trials


@dataclass(frozen=True)
class Budget:
    """The ten (or n) replicates of one comparison at one budget, every one accounted for."""

    budget: int
    pairs: tuple[Pair, ...]

    def count(self, winner: str | None = None, kind: str | None = None) -> int:
        return sum(
            (winner is None or p.winner == winner) and (kind is None or p.kind == kind)
            for p in self.pairs
        )

    @property
    def wins(self) -> int:
        return self.count(winner=CANDIDATE)

    @property
    def losses(self) -> int:
        return self.count(winner=COMPARATOR)

    @property
    def ties(self) -> int:
        return len(self.pairs) - self.wins - self.losses

    @property
    def differences(self) -> list[float]:
        return [p.difference for p in self.pairs if p.difference is not None]

    def p_value(self, direction: str) -> float:
        """The exact one-sided sign test of the pairs that are not ties."""
        favoured = self.wins if direction == CANDIDATE else self.losses
        return one_sided_p(favoured, self.wins + self.losses)

    def magnitude(self, direction: str, delta: float, minimum_complete: int) -> str:
        """How large the advantage in ``direction`` is, from the replicates where both
        models scored. Fewer than ``minimum_complete`` of them do not establish it."""
        differences = self.differences
        if not differences:
            return NOT_COMPUTABLE
        if len(differences) < minimum_complete:
            return NOT_ESTABLISHED
        median = statistics.median(differences)
        oriented = median if direction == CANDIDATE else -median
        return RELEVANT if oriented >= delta else BELOW_MARGIN


def budget(number: int, candidate: Sequence[object], comparator: Sequence[object]) -> Budget:
    """The pairs of one budget, replicate by replicate. Nothing is dropped."""
    if len(candidate) != len(comparator) or not candidate:
        raise ValueError(
            f"the two models need a score or a failure for every replicate, got "
            f"{len(candidate)} and {len(comparator)}"
        )
    return Budget(number, tuple(pair(a, b) for a, b in zip(candidate, comparator, strict=True)))


@dataclass(frozen=True)
class Decision:
    """The verdict at one budget: tested or not, the direction confirmed if any, and the
    magnitude of that direction."""

    budget: int
    tested: bool
    direction: str | None
    magnitude: str | None

    @property
    def counts_for(self) -> str | None:
        """The direction this budget supports in the global verdict: a confirmed direction
        whose advantage is relevant, or rests on failures of the other model."""
        if self.direction and self.magnitude in (RELEVANT, NOT_ESTABLISHED, NOT_COMPUTABLE):
            return self.direction
        return None

    @property
    def label(self) -> str:
        if not self.tested:
            return "not tested"
        if self.direction is None:
            return "no decision"
        return f"{self.direction} better, {self.magnitude}"


@dataclass(frozen=True)
class Verdict:
    verdict: str  # supported, contradicted, budget-dependent, below the margin, no decision
    budgets: tuple[int, ...]  # the budgets the verdict rests on
    decisions: tuple[Decision, ...]


SUPPORTED = "supported"
CONTRADICTED = "contradicted"
BUDGET_DEPENDENT = "budget-dependent"
BELOW = "difference below the relevance margin"
NO_DECISION = "no confirmatory decision"


def _decide(
    result: Budget, rejected: dict[str, bool], tested: bool, delta: float, minimum: int
) -> Decision:
    directions = [d for d in (CANDIDATE, COMPARATOR) if rejected.get(d)]
    if len(directions) > 1:  # a sign test cannot reject both ways; checked, not assumed
        raise AssertionError(f"budget {result.budget}: both directions rejected")
    direction = directions[0] if directions else None
    magnitude = result.magnitude(direction, delta, minimum) if direction else None
    return Decision(result.budget, tested, direction, magnitude)


def ordered_family(
    results: Mapping[int, Budget],
    order: Sequence[int],
    alpha_family: float,
    delta: float,
    minimum_complete: int,
) -> Verdict:
    """A claim of H1: one fixed sequence over the budgets in ``order`` for each direction,
    each at alpha_family / 2, so the family never spends its level twice. The first budget
    of the order is the gate: the claim is supported or contradicted there or not at all,
    and the supported range runs from it while its budgets count for the candidate."""
    level = alpha_family / 2
    rejected = {b: {} for b in order}
    tested = {b: False for b in order}
    for direction in (CANDIDATE, COMPARATOR):
        for b in order:
            tested[b] = True
            if results[b].p_value(direction) <= level:
                rejected[b][direction] = True
            else:
                break
    decisions = tuple(
        _decide(results[b], rejected[b], tested[b], delta, minimum_complete) for b in order
    )
    gate = decisions[0]
    if gate.counts_for == CANDIDATE:
        run = []
        for decision in decisions:
            if decision.counts_for != CANDIDATE:
                break
            run.append(decision.budget)
        return Verdict(SUPPORTED, tuple(run), decisions)
    if gate.counts_for == COMPARATOR:
        return Verdict(CONTRADICTED, (gate.budget,), decisions)
    if gate.direction is not None:
        return Verdict(BELOW, (gate.budget,), decisions)
    return Verdict(NO_DECISION, (), decisions)


def unordered_family(
    results: Mapping[int, Budget],
    budgets: Sequence[int],
    alpha_family: float,
    delta: float,
    minimum_complete: int,
) -> Verdict:
    """A claim of H2: every budget tested in both directions, by Bonferroni at
    alpha_family / (2 x the number of budgets). Supported where some budget counts for the
    candidate and none for the comparator; contradicted the other way round; budget-dependent
    when budgets count both ways."""
    level = alpha_family / (2 * len(budgets))
    decisions = tuple(
        _decide(
            results[b],
            {d: results[b].p_value(d) <= level for d in (CANDIDATE, COMPARATOR)},
            True,
            delta,
            minimum_complete,
        )
        for b in budgets
    )
    for_candidate = tuple(d.budget for d in decisions if d.counts_for == CANDIDATE)
    for_comparator = tuple(d.budget for d in decisions if d.counts_for == COMPARATOR)
    if for_candidate and for_comparator:
        return Verdict(BUDGET_DEPENDENT, for_candidate + for_comparator, decisions)
    if for_candidate:
        return Verdict(SUPPORTED, for_candidate, decisions)
    if for_comparator:
        return Verdict(CONTRADICTED, for_comparator, decisions)
    detected = tuple(d.budget for d in decisions if d.direction is not None)
    if detected:
        return Verdict(BELOW, detected, decisions)
    return Verdict(NO_DECISION, (), decisions)


def mean_ranks(scores: Mapping[str, Sequence[object]]) -> dict[str, float]:
    """The mean over replicates of each model's rank, 1 the best. In every replicate a
    failure ranks below every score, failures tie among themselves, and ties take the mean
    of the ranks they span. Every replicate of every model enters."""
    names = list(scores)
    n = {len(v) for v in scores.values()}
    if len(n) != 1 or not names:
        raise ValueError("every model needs a score or a failure for every replicate")
    total = dict.fromkeys(names, 0.0)
    for r in range(n.pop()):
        keyed = {name: _score(name, scores[name][r]) for name in names}
        order = sorted(names, key=lambda m: (keyed[m] is None, keyed[m] or 0.0))
        position = 0
        while position < len(order):
            tied = [order[position]]
            while (
                position + len(tied) < len(order)
                and keyed[order[position + len(tied)]] == (keyed[order[position]])
            ):
                tied.append(order[position + len(tied)])
            rank = position + (len(tied) + 1) / 2
            for name in tied:
                total[name] += rank
            position += len(tied)
    replicates = len(next(iter(scores.values())))
    return {name: total[name] / replicates for name in names}


def rank_agreement(
    first: Mapping[str, float], second: Mapping[str, float]
) -> tuple[float | None, str]:
    """Kendall's tau-b between two orders of the same models, which allows ties, and why
    it is missing when it cannot be computed: an order in which every model ties has no
    direction to agree with."""
    from scipy.stats import kendalltau

    if set(first) != set(second) or len(first) < 2:
        raise ValueError("the two orders must rank the same two or more models")
    names = sorted(first)
    x, y = [first[n] for n in names], [second[n] for n in names]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, "not computable: every model ties in one of the two orders"
    return float(kendalltau(x, y, variant="b").statistic), "computed"
