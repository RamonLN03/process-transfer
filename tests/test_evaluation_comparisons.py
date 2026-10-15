"""The verdicts proposed in the draft registration of M1-E03, on synthetic scores.

Nothing here comes from the benchmark: the scores are numbers written for each case. The
levels, margin and minimum count are those the draft proposes, passed explicitly, since
they are pending approval: a family level of 0.0125, a margin of 0.05 on E, and five
replicates with a numeric difference before a magnitude is established.
"""

import math

import numpy as np
import pytest

from process_transfer.evaluation import comparisons as c

ALPHA, DELTA, MINIMUM = 0.0125, 0.05, 5
H1_MR_ORDER = (40, 20, 10, 5, 2)


def scores(pattern: list[tuple[object, object]]) -> tuple[list[object], list[object]]:
    return [a for a, _ in pattern], [b for _, b in pattern]


def budget(number: int, pattern: list[tuple[object, object]]) -> c.Budget:
    return c.budget(number, *scores(pattern))


WIN, LOSE, SMALL_WIN = (0.10, 0.20), (0.20, 0.10), (0.10, 0.11)  # (candidate, comparator)


def uniform(number: int, outcome: tuple[object, object], n: int = 10) -> c.Budget:
    return budget(number, [outcome] * n)


# --------------------------------------------------------------------------- #
# One replicate, and one budget
# --------------------------------------------------------------------------- #


def test_the_five_cases_of_a_pair() -> None:
    assert c.pair(0.1, 0.3) == c.Pair(c.BOTH, c.CANDIDATE, pytest.approx(0.2))
    assert c.pair(-0.02, -0.05) == c.Pair(c.BOTH, c.COMPARATOR, pytest.approx(-0.03))
    assert c.pair(0.4, None) == c.Pair(c.ONLY_CANDIDATE, c.CANDIDATE, None)
    assert c.pair(None, 0.4) == c.Pair(c.ONLY_COMPARATOR, c.COMPARATOR, None)
    assert c.pair(None, None) == c.Pair(c.NEITHER, None, None)
    assert c.pair(0.25, 0.25) == c.Pair(c.BOTH, None, 0.0)  # an exact tie keeps its difference


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0.1"])
def test_a_score_is_a_finite_number_or_a_failure(bad: object) -> None:
    with pytest.raises(ValueError, match="finite number or None"):
        c.pair(bad, 0.1)


def test_the_sign_test_is_exact() -> None:
    assert c.one_sided_p(10, 10) == 1 / 1024
    assert c.one_sided_p(9, 10) == 11 / 1024
    assert c.one_sided_p(8, 10) == 56 / 1024
    assert c.one_sided_p(8, 8) == 1 / 256
    assert c.one_sided_p(0, 0) == 1.0
    with pytest.raises(ValueError):
        c.one_sided_p(11, 10)


def test_every_replicate_is_accounted_for_and_only_scores_give_differences() -> None:
    result = budget(
        5,
        [WIN] * 4 + [(0.1, None)] * 2 + [(None, 0.1)] + [(None, None)] * 2 + [(0.3, 0.3)],
    )
    assert (result.wins, result.losses, result.ties) == (6, 1, 3)
    assert result.wins + result.losses + result.ties == 10
    assert result.count(kind=c.NEITHER) == 2 and result.count(kind=c.BOTH) == 5
    assert result.differences == pytest.approx([0.1] * 4 + [0.0])
    assert result.p_value(c.CANDIDATE) == c.one_sided_p(6, 7)


def test_magnitude_never_invents_a_difference_for_a_failure() -> None:
    every_comparator_failed = uniform(2, (0.1, None))
    assert every_comparator_failed.differences == []
    assert every_comparator_failed.magnitude(c.CANDIDATE, DELTA, MINIMUM) == c.NOT_COMPUTABLE
    four_scored = budget(2, [WIN] * 4 + [(0.1, None)] * 6)
    assert four_scored.magnitude(c.CANDIDATE, DELTA, MINIMUM) == c.NOT_ESTABLISHED
    five_scored = budget(2, [WIN] * 5 + [(0.1, None)] * 5)
    assert five_scored.magnitude(c.CANDIDATE, DELTA, MINIMUM) == c.RELEVANT


def test_the_limits_of_the_margin_are_inclusive() -> None:
    exact = uniform(10, (0.25, 0.5))  # a difference of exactly 0.25
    assert exact.magnitude(c.CANDIDATE, 0.25, MINIMUM) == c.RELEVANT
    assert exact.magnitude(c.CANDIDATE, math.nextafter(0.25, 1.0), MINIMUM) == c.BELOW_MARGIN
    assert exact.magnitude(c.COMPARATOR, 0.25, MINIMUM) == c.BELOW_MARGIN  # wrong direction


def test_the_limits_of_the_level_are_inclusive() -> None:
    results = {b: uniform(b, WIN) for b in H1_MR_ORDER}
    at = c.ordered_family(results, H1_MR_ORDER, 2 / 1024, DELTA, MINIMUM)  # level = p
    assert at.verdict == c.SUPPORTED
    below = c.ordered_family(results, H1_MR_ORDER, math.nextafter(2 / 1024, 0.0), DELTA, MINIMUM)
    assert below.verdict == c.NO_DECISION


# --------------------------------------------------------------------------- #
# H1: an ordered claim, one fixed sequence for each direction
# --------------------------------------------------------------------------- #


def test_losing_at_small_budgets_does_not_refute_from_some_budget_on() -> None:
    """HK better than MR at 40, 20 and 10, split at 5, worse at 2: supported from b = 10 on.
    The loss at b = 2 is recorded but not tested, so it is not a contradiction."""
    results = {
        40: uniform(40, WIN),
        20: uniform(20, WIN),
        10: uniform(10, WIN),
        5: budget(5, [WIN] * 5 + [LOSE] * 5),
        2: uniform(2, LOSE),
    }
    verdict = c.ordered_family(results, H1_MR_ORDER, ALPHA, DELTA, MINIMUM)
    assert verdict.verdict == c.SUPPORTED and verdict.budgets == (40, 20, 10)
    labels = {d.budget: d.label for d in verdict.decisions}
    assert labels == {
        40: "candidate better, relevant",
        20: "candidate better, relevant",
        10: "candidate better, relevant",
        5: "no decision",
        2: "not tested",
    }
    assert results[2].losses == 10  # still reported as an outcome of every replicate


def test_losing_at_the_largest_budget_contradicts_from_some_budget_on() -> None:
    results = {b: uniform(b, LOSE) for b in H1_MR_ORDER}
    verdict = c.ordered_family(results, H1_MR_ORDER, ALPHA, DELTA, MINIMUM)
    assert verdict.verdict == c.CONTRADICTED and verdict.budgets == (40,)


def test_h1_against_bn_starts_at_the_smallest_budget() -> None:
    order = (2, 5, 10, 20, 40)
    results = {
        2: uniform(2, WIN),
        5: uniform(5, WIN),
        10: uniform(10, (0.1, 0.1)),
        20: uniform(20, LOSE),
        40: uniform(40, LOSE),
    }
    verdict = c.ordered_family(results, order, ALPHA, DELTA, MINIMUM)
    assert verdict.verdict == c.SUPPORTED and verdict.budgets == (2, 5)
    # BN better at 20 and 40 is not tested in the confirmatory sequence: the gap may close
    assert {d.budget: d.label for d in verdict.decisions}[40] == "not tested"


def test_nine_of_ten_decides_at_one_level_and_not_at_the_other() -> None:
    """The count a claim needs follows from the level, not from a rule of its own."""
    results = {b: budget(b, [WIN] * 9 + [LOSE]) for b in H1_MR_ORDER}
    assert c.ordered_family(results, H1_MR_ORDER, ALPHA, DELTA, MINIMUM).verdict == c.NO_DECISION
    assert c.ordered_family(results, H1_MR_ORDER, 0.05, DELTA, MINIMUM).verdict == c.SUPPORTED


@pytest.mark.parametrize(
    ("pattern", "verdict"),
    [
        ([WIN] * 8 + [LOSE] * 2, c.NO_DECISION),  # inconclusive
        ([SMALL_WIN] * 10, c.BELOW),  # detected, below the margin
        ([(0.1, 0.1)] * 10, c.NO_DECISION),  # exact ties everywhere
        ([(None, None)] * 10, c.NO_DECISION),  # both failed everywhere
        ([WIN] * 8 + [(None, None)] * 2, c.SUPPORTED),  # 8 of 8, p = 1/256
        ([WIN] * 7 + [(None, None)] * 3, c.NO_DECISION),  # 7 of 7, p = 1/128
        ([(0.1, None)] * 10, c.SUPPORTED),  # every comparator failed: reliability
        ([WIN] * 3 + [(0.1, None)] * 7, c.SUPPORTED),  # three scored: reliability, below 5
    ],
)
def test_the_gate_decides_with_ties_failures_and_margins(
    pattern: list[tuple[object, object]], verdict: str
) -> None:
    results = {b: budget(b, pattern) for b in H1_MR_ORDER}
    assert c.ordered_family(results, H1_MR_ORDER, ALPHA, DELTA, MINIMUM).verdict == verdict


def test_a_reliability_win_is_labelled_as_one() -> None:
    results = {b: uniform(b, (0.1, None)) for b in H1_MR_ORDER}
    gate = c.ordered_family(results, H1_MR_ORDER, ALPHA, DELTA, MINIMUM).decisions[0]
    assert gate.label == "candidate better, magnitude not computable"


# --------------------------------------------------------------------------- #
# H2: an unordered claim over two budgets
# --------------------------------------------------------------------------- #


def test_winning_at_one_budget_and_losing_at_the_other_is_budget_dependent() -> None:
    """Codex's case: HK better in every replicate at b = 10 and worse in every one at
    b = 40. Before, H2 was both supported and contradicted; now it is one verdict."""
    results = {10: uniform(10, WIN), 40: uniform(40, LOSE)}
    verdict = c.unordered_family(results, (40, 10), ALPHA, DELTA, MINIMUM)
    assert verdict.verdict == c.BUDGET_DEPENDENT
    assert {d.budget: d.label for d in verdict.decisions} == {
        40: "comparator better, relevant",
        10: "candidate better, relevant",
    }


@pytest.mark.parametrize(
    ("at_40", "at_10", "verdict"),
    [
        (WIN, (0.1, 0.1), c.SUPPORTED),
        ((0.1, 0.1), LOSE, c.CONTRADICTED),
        (SMALL_WIN, (0.1, 0.1), c.BELOW),
        ((0.1, 0.1), (0.1, 0.1), c.NO_DECISION),
    ],
)
def test_the_unordered_verdict_is_unique(at_40: tuple, at_10: tuple, verdict: str) -> None:
    results = {40: uniform(40, at_40), 10: uniform(10, at_10)}
    assert c.unordered_family(results, (40, 10), ALPHA, DELTA, MINIMUM).verdict == verdict


def test_the_unordered_level_is_split_over_budgets_and_directions() -> None:
    """0.0125 / 4: 9 of 9 decides (p = 1/512), 8 of 8 does not (p = 1/256)."""
    nine = {b: budget(b, [WIN] * 9 + [(None, None)]) for b in (40, 10)}
    eight = {b: budget(b, [WIN] * 8 + [(None, None)] * 2) for b in (40, 10)}
    assert c.unordered_family(nine, (40, 10), ALPHA, DELTA, MINIMUM).verdict == c.SUPPORTED
    assert c.unordered_family(eight, (40, 10), ALPHA, DELTA, MINIMUM).verdict == c.NO_DECISION


# --------------------------------------------------------------------------- #
# Invariants over many synthetic patterns
# --------------------------------------------------------------------------- #


def random_budget(rng: np.random.Generator, number: int) -> c.Budget:
    pattern = []
    bias = rng.uniform(-0.2, 0.2)
    for _ in range(10):
        a = None if rng.random() < 0.1 else float(rng.normal(0.2, 0.05))
        b = None if rng.random() < 0.1 else float(rng.normal(0.2 + bias, 0.05))
        pattern.append((a, b))
    return budget(number, pattern)


def test_no_pattern_gives_a_contradictory_verdict_or_loses_a_replicate() -> None:
    rng = np.random.default_rng(20261015)
    for _ in range(400):
        results = {b: random_budget(rng, b) for b in H1_MR_ORDER}
        for result in results.values():
            assert result.wins + result.losses + result.ties == len(result.pairs) == 10
        ordered = c.ordered_family(results, H1_MR_ORDER, ALPHA, DELTA, MINIMUM)
        gate = ordered.decisions[0]
        assert gate.tested
        if ordered.verdict == c.SUPPORTED:
            assert gate.counts_for == c.CANDIDATE
        elif ordered.verdict == c.CONTRADICTED:
            assert gate.counts_for == c.COMPARATOR
        else:
            assert gate.counts_for is None
        unordered = c.unordered_family(results, (40, 10), ALPHA, DELTA, MINIMUM)
        directions = {d.counts_for for d in unordered.decisions} - {None}
        expected = {
            frozenset(): {c.BELOW, c.NO_DECISION},
            frozenset({c.CANDIDATE}): {c.SUPPORTED},
            frozenset({c.COMPARATOR}): {c.CONTRADICTED},
            frozenset({c.CANDIDATE, c.COMPARATOR}): {c.BUDGET_DEPENDENT},
        }[frozenset(directions)]
        assert unordered.verdict in expected


# --------------------------------------------------------------------------- #
# H3: ranks with failures and ties
# --------------------------------------------------------------------------- #


def test_a_failure_ranks_last_and_is_not_left_out() -> None:
    """A model that scores best in seven replicates and fails the other three is not
    ranked on its seven successes alone."""
    ranks = c.mean_ranks(
        {
            "risky": [0.1] * 7 + [None] * 3,
            "steady": [0.2] * 10,
            "slow": [0.3] * 7 + [None] * 3,
        }
    )
    # in the three hard replicates: steady 1, risky and slow tie at 2.5
    assert ranks == pytest.approx(
        {"risky": (7 + 7.5) / 10, "steady": (14 + 3) / 10, "slow": (21 + 7.5) / 10}
    )
    assert ranks["risky"] > 1.0  # its rank over successes alone would have been 1


def test_rank_agreement_allows_ties_and_says_when_it_cannot_be_computed() -> None:
    first = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert c.rank_agreement(first, {"a": 1.0, "b": 2.0, "c": 3.0}) == (1.0, "computed")
    assert c.rank_agreement(first, {"a": 3.0, "b": 2.0, "c": 1.0})[0] == pytest.approx(-1.0)
    tau, _ = c.rank_agreement(first, {"a": 1.0, "b": 1.0, "c": 3.0})
    assert 0.0 < tau < 1.0
    assert c.rank_agreement(first, {"a": 2.0, "b": 2.0, "c": 2.0}) == (
        None,
        "not computable: every model ties in one of the two orders",
    )
