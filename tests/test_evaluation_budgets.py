"""Budgets and their division into F and V (docs/m1_plan.md, sections 5.4, 5.5 and 5.7):
the worked example, the counts of the table of budgets, disjointness, prefixes, and the
excitation of a part computed exactly from the known inputs."""

import pytest

from m1_support import NOMINAL, corner, observations, p3_inputs, random_corners
from process_transfer.evaluation.budgets import (
    BudgetSplit,
    excitation,
    integer_rank,
    split_budget,
    validation_windows,
)
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data


def run_windows(n_excursions: int, seed: int = 0):
    run = observations(p3_inputs(random_corners(n_excursions, seed)))
    return run, find_windows(run, NOMINAL, P3_LAYOUT)


def test_the_worked_example_of_the_plan_at_two_windows() -> None:
    _, found = run_windows(40)
    split = split_budget(found, 2)
    fitting, validation, every = (split.indices(p) for p in ("fitting", "validation", "all"))
    assert fitting.context == frozenset(range(1, 11))
    assert fitting.scored == frozenset(range(11, 121))
    assert validation.context == frozenset(range(121, 131))
    assert validation.scored == frozenset(range(131, 241))
    # a method that selects reads ticks 1 to 120 to fit and 121 to 240 to validate
    assert fitting.rows_read == frozenset(range(1, 121))
    assert validation.rows_read == frozenset(range(121, 241))
    # MR reads ticks 1 to 240
    assert every.rows_read == frozenset(range(1, 241))
    assert split.prefix_end == 250
    assert split.unused == frozenset({0, *range(241, 251)})


@pytest.mark.parametrize(
    ("budget", "fitting_rows", "validation_rows"),
    [(5, range(1, 481), range(481, 601)), (10, range(1, 961), range(961, 1201))],
)
def test_the_parts_at_five_and_ten_windows(
    budget: int, fitting_rows: range, validation_rows: range
) -> None:
    _, found = run_windows(40)
    split = split_budget(found, budget)
    assert split.indices("fitting").rows_read == frozenset(fitting_rows)
    assert split.indices("validation").rows_read == frozenset(validation_rows)


@pytest.mark.parametrize(
    ("budget", "n_fitting", "n_validation"),
    [(2, 1, 1), (5, 4, 1), (10, 8, 2), (20, 16, 4), (40, 32, 8)],
)
def test_the_counts_of_the_table_of_budgets(budget: int, n_fitting: int, n_validation: int):
    _, found = run_windows(40)
    split = split_budget(found, budget)
    assert (len(split.fitting), len(split.validation)) == (n_fitting, n_validation)
    assert split.prefix_end + 1 == 11 + 120 * budget  # rows of the prefix
    every = split.indices("all")
    assert (len(every.context), len(every.scored)) == (10 * budget, 110 * budget)
    assert len(split.unused) == 11
    # every row of the prefix is a context, a scored reading or unused, exactly one
    prefix = frozenset(range(split.prefix_end + 1))
    assert every.context | every.scored | split.unused == prefix
    assert not every.context & every.scored
    assert not split.indices("fitting").rows_read & split.indices("validation").rows_read
    # the exact record: the lead of 60 s and b cycles of 720 s
    assert 6.0 * split.prefix_end == 60.0 + 720.0 * budget


def test_the_data_of_a_budget_are_a_prefix_of_those_of_any_larger_budget() -> None:
    _, found = run_windows(40)
    splits = [split_budget(found, budget) for budget in (2, 5, 10, 20, 40)]
    for smaller, larger in zip(splits, splits[1:], strict=False):
        assert larger.windows[: smaller.budget] == smaller.windows
        assert smaller.indices("all").rows_read <= larger.indices("all").rows_read


def test_the_number_of_validation_windows() -> None:
    budgets = (2, 4, 5, 9, 10, 14, 15, 20, 40)
    assert [validation_windows(b) for b in budgets] == [1, 1, 1, 1, 2, 2, 3, 4, 8]
    for invalid in (1, 0, -3, True, 2.0, "5"):
        with pytest.raises(ValueError, match="2 or more"):
            validation_windows(invalid)  # type: ignore[arg-type]


def test_a_budget_larger_than_the_run_is_refused() -> None:
    _, found = run_windows(3)
    with pytest.raises(ValueError, match="has 3 windows; a budget of 5"):
        split_budget(found, 5)


def test_a_division_whose_parts_share_rows_or_leave_the_prefix_is_refused() -> None:
    _, found = run_windows(4)
    first, second, third = found.windows[:3]
    with pytest.raises(ValueError, match="order of time"):
        BudgetSplit(found.run_id, 2, (first,), (first,), 250)
    with pytest.raises(ValueError, match="beyond its prefix"):
        BudgetSplit(found.run_id, 2, (first,), (second,), 200)
    with pytest.raises(ValueError, match="validation windows"):
        BudgetSplit(found.run_id, 3, (first,), (second, third), 370)


def test_on_a_run_of_m0_a_budget_starts_at_the_second_excursion() -> None:
    run = observations(p3_inputs(random_corners(10, 1), lead=False))
    found = find_windows(run, NOMINAL, P3_LAYOUT)
    split = split_budget(found, 9)
    assert [w.excursion for w in split.fitting] == list(range(2, 10))
    assert [w.excursion for w in split.validation] == [10]
    assert split.prefix_end == 1200
    assert split.unused == frozenset({*range(0, 111), *range(1191, 1201)})


def test_the_excitation_of_a_part_is_read_from_the_inputs_at_its_onsets() -> None:
    corners = [[1, 1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, 1, -1]]
    run = observations(p3_inputs(corners))
    found = find_windows(run, NOMINAL, P3_LAYOUT)
    data = window_data(run, found.windows)
    seen = excitation(data, NOMINAL)
    assert seen.corners == tuple(tuple(c) for c in corners)
    assert (seen.distinct, seen.rank) == (3, 2)  # the last two corners are opposite
    one = excitation(data[:1], NOMINAL)
    assert (one.distinct, one.rank) == (1, 1)
    assert corner(corners[2])[0] < NOMINAL[0]


def test_the_rank_is_exact_on_integers() -> None:
    assert integer_rank([]) == 0
    assert integer_rank([[0, 0, 0, 0]]) == 0
    assert integer_rank([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) == 4
    assert integer_rank([[1, 1, 1, 1], [-1, -1, -1, -1], [2, 2, 2, 2]]) == 1
    assert integer_rank([[1, -1, 1, -1], [1, 1, -1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]]) == 4
    assert integer_rank([[1, 1, 0], [0, 1, 1], [1, 2, 1]]) == 2
