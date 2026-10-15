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

The domain of every public argument is checked where it enters. An argument outside it
raises ``ValueError`` naming the argument and its value, so that a wrong configuration can
never read as a scientific result:

* a family level is a finite number strictly between 0 and 1;
* a margin is a finite number, zero or more;
* a count is an integer, never a boolean, at or above its minimum;
* a direction is ``CANDIDATE`` or ``COMPARATOR``;
* a budget is a positive integer, and an order of budgets is non-empty, without duplicates,
  and exactly the budgets whose results are given, each holding the same replicates;
* a series of replicates is non-empty, and paired series have the same length;
* a score or a rank is a finite number; a number too large for a float is refused, not
  rounded to infinity.

A valid input without evidence is not an error: no pair that is not a tie gives a p-value
of 1, and orders in which every model ties give an agreement that is not computable.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from process_transfer.evaluation.outcomes import FIRST, SECOND, paired_outcome

CANDIDATE, COMPARATOR = "candidate", "comparator"
DIRECTIONS = (CANDIDATE, COMPARATOR)
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
MAGNITUDES = (RELEVANT, BELOW_MARGIN, NOT_ESTABLISHED, NOT_COMPUTABLE)

# --------------------------------------------------------------------------- #
# Domains
# --------------------------------------------------------------------------- #


def _real(name: str, value: object) -> float:
    """A finite real number, as a float."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return converted


def _score(name: str, value: object) -> float | None:
    if value is None:
        return None
    try:
        return _real(name, value)
    except ValueError:
        raise ValueError(
            f"{name} must be a finite number or None for a failure, got {value!r}"
        ) from None


def _count(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return int(value)


def _level(name: str, value: object) -> float:
    level = _real(name, value)
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value!r}")
    return level


def _margin(name: str, value: object) -> float:
    margin = _real(name, value)
    if margin < 0.0:
        raise ValueError(f"{name} must be zero or more, got {value!r}")
    return margin


def _direction(value: object) -> str:
    if value not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {value!r}")
    return value  # type: ignore[return-value]


def _series(name: str, value: object) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty sequence of replicates, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# One replicate
# --------------------------------------------------------------------------- #


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


def one_sided_p(successes: object, trials: object) -> float:
    """P(X >= successes) for X binomial with ``trials`` trials of probability 1/2, exactly.
    With no trials there is no evidence, and the p-value is 1."""
    n = _count("trials", trials, 0)
    k = _count("successes", successes, 0)
    if k > n:
        raise ValueError(f"successes must not exceed trials, got {k} of {n}")
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def median(values: Sequence[float]) -> float:
    """The median of finite numbers. The midpoint of the two central values is taken so
    that it cannot overflow: from their sum when their signs differ, and as the smaller plus
    half their difference when they agree. Both are the same mathematical midpoint;
    ``statistics.median`` adds the two values first, which overflows near the largest
    float."""
    if not values:
        raise ValueError("the median of no values is not defined")
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    a, b = ordered[n // 2 - 1], ordered[n // 2]
    if a == b:
        return a
    middle = (a + b) / 2 if (a < 0.0) != (b < 0.0) else a + (b - a) / 2
    if not math.isfinite(middle):  # cannot happen for finite values; checked, not assumed
        raise ArithmeticError(f"the midpoint of {a!r} and {b!r} is not finite")
    return middle


# --------------------------------------------------------------------------- #
# One budget
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Budget:
    """The replicates of one comparison at one budget, every one accounted for."""

    budget: int
    pairs: tuple[Pair, ...]

    def __post_init__(self) -> None:
        _count("budget", self.budget, 1)
        if not self.pairs or not all(isinstance(p, Pair) for p in self.pairs):
            raise ValueError(f"budget {self.budget} needs one Pair for every replicate")

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
        return self._p_value(_direction(direction))

    def _p_value(self, direction: str) -> float:
        favoured = self.wins if direction == CANDIDATE else self.losses
        return one_sided_p(favoured, self.wins + self.losses)

    def magnitude(self, direction: str, delta: float, minimum_complete: int) -> str:
        """How large the advantage in ``direction`` is, from the replicates where both
        models scored. Fewer than ``minimum_complete`` of them do not establish it."""
        return self._magnitude(
            _direction(direction),
            _margin("delta", delta),
            _count("minimum_complete", minimum_complete, 1),
        )

    def _magnitude(self, direction: str, delta: float, minimum_complete: int) -> str:
        differences = self.differences
        if not differences:
            return NOT_COMPUTABLE
        if len(differences) < minimum_complete:
            return NOT_ESTABLISHED
        middle = median(differences)
        oriented = middle if direction == CANDIDATE else -middle
        return RELEVANT if oriented >= delta else BELOW_MARGIN


def budget(number: int, candidate: Sequence[object], comparator: Sequence[object]) -> Budget:
    """The pairs of one budget, replicate by replicate. Nothing is dropped."""
    first, second = _series("candidate", candidate), _series("comparator", comparator)
    if len(first) != len(second):
        raise ValueError(
            f"the two models need a score or a failure for every replicate, got "
            f"{len(first)} and {len(second)}"
        )
    return Budget(
        _count("budget", number, 1),
        tuple(pair(a, b) for a, b in zip(first, second, strict=True)),
    )


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decision:
    """The verdict at one budget: tested or not, the direction confirmed if any, and the
    magnitude of that direction."""

    budget: int
    tested: bool
    direction: str | None
    magnitude: str | None

    def __post_init__(self) -> None:
        if self.direction is not None:
            _direction(self.direction)
        if (self.direction is None) != (self.magnitude is None):
            raise ValueError("a magnitude is given exactly when a direction is confirmed")
        if self.magnitude is not None and self.magnitude not in MAGNITUDES:
            raise ValueError(f"magnitude must be one of {MAGNITUDES}, got {self.magnitude!r}")
        if self.direction is not None and not self.tested:
            raise ValueError("a direction can only be confirmed at a budget that was tested")

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


def _order(results: Mapping[int, Budget], order: Sequence[int]) -> tuple[int, ...]:
    """The budgets of a family: non-empty, without duplicates, exactly the budgets whose
    results are given, each result filed under its own budget, all over the same number of
    replicates."""
    if isinstance(order, str | bytes) or not isinstance(order, Sequence) or not order:
        raise ValueError(f"the budgets of a family must be a non-empty sequence, got {order!r}")
    budgets = tuple(_count("budget", b, 1) for b in order)
    if len(set(budgets)) != len(budgets):
        raise ValueError(f"the budgets of a family must be distinct, got {order!r}")
    if not isinstance(results, Mapping) or set(results) != set(budgets):
        given = sorted(results) if isinstance(results, Mapping) else results
        raise ValueError(
            f"the results must be exactly those of the budgets {budgets}, got {given!r}"
        )
    sizes = set()
    for b in budgets:
        result = results[b]
        if not isinstance(result, Budget) or result.budget != b:
            raise ValueError(f"the result filed under budget {b} is {result!r}")
        sizes.add(len(result.pairs))
    if len(sizes) != 1:
        raise ValueError(f"every budget must hold the same replicates, got {sorted(sizes)}")
    return budgets


def _decide(
    result: Budget, rejected: dict[str, bool], tested: bool, delta: float, minimum: int
) -> Decision:
    directions = [d for d in DIRECTIONS if rejected.get(d)]
    if len(directions) > 1:  # a sign test cannot reject both ways; checked, not assumed
        raise AssertionError(f"budget {result.budget}: both directions rejected")
    direction = directions[0] if directions else None
    magnitude = result._magnitude(direction, delta, minimum) if direction else None
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
    budgets = _order(results, order)
    level = _level("alpha_family", alpha_family) / 2
    margin = _margin("delta", delta)
    minimum = _count("minimum_complete", minimum_complete, 1)
    rejected: dict[int, dict[str, bool]] = {b: {} for b in budgets}
    tested = dict.fromkeys(budgets, False)
    for direction in DIRECTIONS:
        for b in budgets:
            tested[b] = True
            if results[b]._p_value(direction) <= level:
                rejected[b][direction] = True
            else:
                break
    decisions = tuple(_decide(results[b], rejected[b], tested[b], margin, minimum) for b in budgets)
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
    listed = _order(results, budgets)
    level = _level("alpha_family", alpha_family) / (2 * len(listed))
    margin = _margin("delta", delta)
    minimum = _count("minimum_complete", minimum_complete, 1)
    decisions = tuple(
        _decide(
            results[b],
            {d: results[b]._p_value(d) <= level for d in DIRECTIONS},
            True,
            margin,
            minimum,
        )
        for b in listed
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


# --------------------------------------------------------------------------- #
# H3: ranks and their agreement
# --------------------------------------------------------------------------- #


def mean_ranks(scores: Mapping[str, Sequence[object]]) -> dict[str, float]:
    """The mean over replicates of each model's rank, 1 the best. In every replicate a
    failure ranks below every score, failures tie among themselves, and ties take the mean
    of the ranks they span. Every replicate of every model enters."""
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError(f"scores must map one or more models to their replicates, got {scores!r}")
    series = {name: _series(f"the scores of {name!r}", value) for name, value in scores.items()}
    lengths = {len(value) for value in series.values()}
    if len(lengths) != 1:
        raise ValueError(f"every model needs the same replicates, got lengths {sorted(lengths)}")
    replicates = lengths.pop()
    names = list(series)
    total = dict.fromkeys(names, 0.0)
    for r in range(replicates):
        keyed = {name: _score(f"the score of {name!r}", series[name][r]) for name in names}
        order = sorted(names, key=lambda m: (keyed[m] is None, keyed[m] or 0.0))
        position = 0
        while position < len(order):
            tied = [order[position]]
            while (
                position + len(tied) < len(order)
                and keyed[order[position + len(tied)]] == keyed[order[position]]
            ):
                tied.append(order[position + len(tied)])
            rank = position + (len(tied) + 1) / 2
            for name in tied:
                total[name] += rank
            position += len(tied)
    return {name: total[name] / replicates for name in names}


def rank_agreement(
    first: Mapping[str, float], second: Mapping[str, float]
) -> tuple[float | None, str]:
    """Kendall's tau-b between two orders of the same models, which allows ties, and why
    it is missing when it cannot be computed: an order in which every model ties has no
    direction to agree with."""
    from scipy.stats import kendalltau

    for name, value in (("first", first), ("second", second)):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must map models to their ranks, got {value!r}")
    if set(first) != set(second) or len(first) < 2:
        raise ValueError(
            f"the two orders must rank the same two or more models, got {sorted(first)} and "
            f"{sorted(second)}"
        )
    names = sorted(first)
    x = [_real(f"the rank of {n!r} in first", first[n]) for n in names]
    y = [_real(f"the rank of {n!r} in second", second[n]) for n in names]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, "not computable: every model ties in one of the two orders"
    tau = float(kendalltau(x, y, variant="b").statistic)
    if not math.isfinite(tau):  # finite ranks with two or more values give a finite tau-b
        raise ArithmeticError(f"Kendall's tau-b of {x} and {y} is not finite: {tau!r}")
    return tau, "computed"
