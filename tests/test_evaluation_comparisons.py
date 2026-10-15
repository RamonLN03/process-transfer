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
        h2 = {b: results[b] for b in (40, 10)}  # a family is given exactly its budgets
        unordered = c.unordered_family(h2, (40, 10), ALPHA, DELTA, MINIMUM)
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


# --------------------------------------------------------------------------- #
# Codex's audit of 46489c8: a wrong configuration must never read as a result
# --------------------------------------------------------------------------- #

FIVE = {b: uniform(b, WIN) for b in H1_MR_ORDER}
TWO = {b: uniform(b, WIN) for b in (40, 10)}


@pytest.mark.parametrize("alpha", [math.nan, math.inf, -0.1, 0.0, 1.0, 2.0, True, "0.05"])
def test_a_family_level_outside_its_domain_is_refused(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha_family"):
        c.ordered_family(FIVE, H1_MR_ORDER, alpha, DELTA, MINIMUM)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha_family"):
        c.unordered_family(TWO, (40, 10), alpha, DELTA, MINIMUM)  # type: ignore[arg-type]


@pytest.mark.parametrize("delta", [math.nan, math.inf, -1.0, -1e-300, True, None])
def test_a_margin_outside_its_domain_is_refused(delta: object) -> None:
    with pytest.raises(ValueError, match="delta"):
        c.ordered_family(FIVE, H1_MR_ORDER, ALPHA, delta, MINIMUM)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="delta"):
        FIVE[40].magnitude(c.CANDIDATE, delta, MINIMUM)  # type: ignore[arg-type]


@pytest.mark.parametrize("minimum", [0, -1, True, 5.0, "5"])
def test_a_minimum_count_outside_its_domain_is_refused(minimum: object) -> None:
    with pytest.raises(ValueError, match="minimum_complete"):
        c.ordered_family(FIVE, H1_MR_ORDER, ALPHA, DELTA, minimum)  # type: ignore[arg-type]


@pytest.mark.parametrize("direction", ["candiate", "Candidate", "", None, "first"])
def test_a_misspelt_direction_is_refused_not_read_as_the_comparator(direction: object) -> None:
    with pytest.raises(ValueError, match="direction"):
        FIVE[40].p_value(direction)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="direction"):
        FIVE[40].magnitude(direction, DELTA, MINIMUM)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="direction"):
        c.Decision(40, True, direction, c.RELEVANT)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("results", "order", "message"),
    [
        ({}, (), "non-empty"),
        (FIVE, (40, 40, 20, 10, 5, 2), "distinct"),
        (FIVE, (40, 20), "exactly those of the budgets"),
        ({40: FIVE[40]}, (40, 20), "exactly those of the budgets"),
        ({40: FIVE[20]}, (40,), "filed under budget 40"),
        ({40: FIVE[40], 20: uniform(20, WIN, n=9)}, (40, 20), "same replicates"),
        (FIVE, "40", "non-empty sequence"),
        ({5: FIVE[5]}, (5.0,), "budget must be an integer"),
    ],
)
def test_an_order_of_budgets_must_match_its_results(
    results: dict, order: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        c.ordered_family(results, order, ALPHA, DELTA, MINIMUM)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        c.unordered_family(results, order, ALPHA, DELTA, MINIMUM)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("candidate", "comparator", "message"),
    [
        ([], [], "non-empty sequence"),
        ([0.1, 0.2], [0.1], "every replicate"),
        ("0.1", "0.2", "non-empty sequence"),
        ([10**400], [0.1], "candidate must be a finite number"),
        ([np.bool_(True)], [0.1], "candidate must be a finite number"),
    ],
)
def test_a_series_of_replicates_outside_its_domain_is_refused(
    candidate: object, comparator: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        c.budget(5, candidate, comparator)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("successes", "trials"), [(True, 1), (9.0, 10), (-1, 10), (11, 10), (1, -1)]
)
def test_the_sign_test_refuses_counts_outside_its_domain(successes: object, trials: object) -> None:
    with pytest.raises(ValueError):
        c.one_sided_p(successes, trials)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "scores",
    [{}, {"a": [], "b": []}, {"a": [0.1, 0.2], "b": [0.1]}, {"a": [math.nan], "b": [0.1]}],
)
def test_ranks_need_replicates_and_finite_scores(scores: dict) -> None:
    with pytest.raises(ValueError):
        c.mean_ranks(scores)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"a": 1.0, "b": math.nan, "c": 3.0}, {"a": 1.0, "b": 2.0, "c": 3.0}),
        ({"a": 1.0, "b": math.inf}, {"a": 1.0, "b": 2.0}),
        ({"a": True, "b": 2.0}, {"a": 1.0, "b": 2.0}),
        ({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0}),
        ({"a": 1.0}, {"a": 1.0}),
        ([1.0, 2.0], {"a": 1.0, "b": 2.0}),
    ],
)
def test_an_agreement_of_invalid_ranks_is_refused_not_computed(
    first: object, second: object
) -> None:
    with pytest.raises(ValueError):
        c.rank_agreement(first, second)  # type: ignore[arg-type]


def test_valid_inputs_without_evidence_are_not_errors() -> None:
    every_pair_a_tie = uniform(10, (None, None))
    assert every_pair_a_tie.p_value(c.CANDIDATE) == every_pair_a_tie.p_value(c.COMPARATOR) == 1.0
    assert c.rank_agreement({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 2.0})[0] is None
    # numpy numbers are numbers
    numpy_budget = c.budget(np.int64(10), [np.float64(0.1)] * 10, [np.float64(0.2)] * 10)
    assert numpy_budget.magnitude(c.CANDIDATE, np.float64(0.05), np.int64(5)) == c.RELEVANT


# --------------------------------------------------------------------------- #
# The median without overflow
# --------------------------------------------------------------------------- #


def test_codex_case_the_median_of_large_differences_does_not_overflow() -> None:
    """Every difference is 1e308. statistics.median added the two central values and gave
    infinity, so a margin of 1.1e308 read as reached. The median is 1e308, below it."""
    result = c.budget(10, [0.0] * 10, [1e308] * 10)
    assert c.median(result.differences) == 1e308
    assert result.magnitude(c.CANDIDATE, 1.1e308, 5) == c.BELOW_MARGIN
    assert result.magnitude(c.CANDIDATE, 1e308, 5) == c.RELEVANT


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1e308, 1.5e308], 1.25e308),  # large, same sign
        ([-1.5e308, -1e308], -1.25e308),
        ([1.7976931348623157e308] * 2, 1.7976931348623157e308),
        ([-1e308, 1e308], 0.0),  # opposite signs
        ([-1.0, 3.0], 1.0),
        ([2.0**-1000, 3 * 2.0**-1000], 2.0**-999),  # small magnitudes, exactly
        ([0.0, 1e-323], 5e-324),
        ([3.0, 1e308, -1.0], 3.0),  # an odd number of values
        ([0.25, 0.75, 0.25, 0.75], 0.5),
    ],
)
def test_the_median_is_the_midpoint_and_stays_finite(values: list[float], expected: float) -> None:
    found = c.median(values)
    assert found == pytest.approx(expected, rel=1e-15, abs=0.0) and math.isfinite(found)


def test_a_median_equal_to_the_margin_is_relevant() -> None:
    result = c.budget(10, [0.0] * 10, [0.25] * 5 + [0.75] * 5)  # median exactly 0.5
    assert result.magnitude(c.CANDIDATE, 0.5, 5) == c.RELEVANT
    assert result.magnitude(c.CANDIDATE, math.nextafter(0.5, 1.0), 5) == c.BELOW_MARGIN


def test_the_median_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError):
        c.median([])
