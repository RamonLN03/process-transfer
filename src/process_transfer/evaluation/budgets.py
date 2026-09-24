"""Budgets, and their division into a fitting part and a validation part
(``docs/m1_plan.md``, sections 5.3 to 5.7).

A budget of b is the first b windows of a run and the rows they span: the prefix of the
run up to the onset of the next excursion, or to the end of the run. On a P3 run of M1,
which starts with a lead of 60 s, that is the lead and the first b excursions with their
rests, 11 + 120 b rows; eleven of them are unused, tick 0 and the ten that would be the
context of window b + 1. On a P3 run of M0, which has no lead, the first excursion forms
no window, so b windows span b + 1 excursions.

A method that selects anything from data divides the windows of its budget in time: the
fitting part F is windows 1 to b - n_V and the validation part V the last n_V, with
n_V = max(1, floor(b / 5)). Each part is made of the contexts and the scored readings of its
own windows, so the context of the first validation window belongs to V and F does not
score it. The comparator MR, which selects nothing, fits on all b windows.

The division is made of explicit sets of ticks, checked when it is built: no reading is
both a context and scored, F and V share no row, and every row of the prefix is a context,
a scored reading or unused, exactly one of the three.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from process_transfer.cstr_variables import FloatArray
from process_transfer.evaluation.windows import RunWindows, Window, WindowData

FITTING = "fitting"
VALIDATION = "validation"


def validation_windows(budget: int) -> int:
    """n_V = max(1, floor(b / 5)), for a budget of b windows. A budget needs at least two
    windows, so that F is not empty."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 2:
        raise ValueError(f"a budget is a whole number of windows, 2 or more, got {budget!r}")
    return max(1, budget // 5)


@dataclass(frozen=True)
class PartIndices:
    """The ticks of one part of a budget, by role."""

    context: frozenset[int]
    scored: frozenset[int]
    inputs: frozenset[int]  # rows whose inputs drive the predictions of the part

    @property
    def rows_read(self) -> frozenset[int]:
        return self.context | self.scored | self.inputs


def part_indices(windows: Sequence[Window]) -> PartIndices:
    return PartIndices(
        context=frozenset(tick for window in windows for tick in window.context_ticks),
        scored=frozenset(tick for window in windows for tick in window.scored_ticks),
        inputs=frozenset(tick for window in windows for tick in window.input_ticks),
    )


@dataclass(frozen=True)
class BudgetSplit:
    """The first ``budget`` windows of a run, divided into F and V."""

    run_id: str
    budget: int
    fitting: tuple[Window, ...]
    validation: tuple[Window, ...]
    prefix_end: int  # the last tick of the prefix of the run that the budget spans

    def __post_init__(self) -> None:
        windows = self.windows
        if len(self.validation) != validation_windows(self.budget):
            raise ValueError(
                f"a budget of {self.budget} has {validation_windows(self.budget)} validation "
                f"windows, got {len(self.validation)}"
            )
        if len(windows) != self.budget:
            raise ValueError(f"a budget of {self.budget} has {len(windows)} windows")
        if any(window.run_id != self.run_id for window in windows):
            raise ValueError(f"every window of a budget belongs to run {self.run_id!r}")
        onsets = [window.onset for window in windows]
        if onsets != sorted(onsets) or len(set(onsets)) != len(onsets):
            raise ValueError(f"the windows of a budget are in the order of time, got {onsets}")
        fitting, validation = part_indices(self.fitting), part_indices(self.validation)
        every = part_indices(windows)
        if every.context & every.scored:
            raise ValueError(
                f"ticks {sorted(every.context & every.scored)} are both a context and scored"
            )
        if fitting.rows_read & validation.rows_read:
            raise ValueError(
                f"F and V share rows {sorted(fitting.rows_read & validation.rows_read)}"
            )
        prefix = set(range(self.prefix_end + 1))
        if not every.rows_read <= prefix:
            raise ValueError(
                f"the budget reads rows beyond its prefix, which ends at tick {self.prefix_end}"
            )

    @property
    def windows(self) -> tuple[Window, ...]:
        return self.fitting + self.validation

    def indices(self, part: str) -> PartIndices:
        """The ticks of F (``"fitting"``), of V (``"validation"``) or of all b windows
        (``"all"``), which is what the comparator MR reads."""
        chosen = {FITTING: self.fitting, VALIDATION: self.validation, "all": self.windows}
        if part not in chosen:
            raise ValueError(f"part must be one of {sorted(chosen)}, got {part!r}")
        return part_indices(chosen[part])

    @property
    def unused(self) -> frozenset[int]:
        """The ticks of the prefix that are neither a context nor scored."""
        every = part_indices(self.windows)
        return frozenset(range(self.prefix_end + 1)) - every.context - every.scored


def split_budget(run_windows: RunWindows, budget: int) -> BudgetSplit:
    """The budget of the first ``budget`` windows of a run, divided into F and V."""
    n_validation = validation_windows(budget)
    windows = run_windows.windows
    if len(windows) < budget:
        raise ValueError(
            f"run {run_windows.run_id!r} has {len(windows)} windows; a budget of {budget} "
            "needs that many"
        )
    chosen = windows[:budget]
    later = [onset for onset in run_windows.onsets if onset > chosen[-1].onset]
    prefix_end = later[0] if later else run_windows.n_ticks - 1
    return BudgetSplit(
        run_id=run_windows.run_id,
        budget=budget,
        fitting=chosen[: budget - n_validation],
        validation=chosen[budget - n_validation :],
        prefix_end=prefix_end,
    )


@dataclass(frozen=True)
class Excitation:
    """Which corners of the input box a set of windows visited, from the known inputs."""

    corners: tuple[tuple[int, ...], ...]  # the sign of each input at each onset, in order
    distinct: int
    rank: int  # of the sign vectors, computed exactly


def excitation(data: Sequence[WindowData], nominal_inputs: FloatArray) -> Excitation:
    """The signs of the inputs at the onset of each window, relative to the nominal inputs.
    The sign of a difference of two floating-point numbers is exact, so nothing is
    rounded; the rank is computed on the integers."""
    nominal = np.asarray(nominal_inputs, dtype=np.float64)
    corners = tuple(
        tuple(int(sign) for sign in np.sign(window.inputs[0] - nominal)) for window in data
    )
    return Excitation(corners=corners, distinct=len(set(corners)), rank=integer_rank(corners))


def integer_rank(rows: Sequence[Sequence[int]]) -> int:
    """The rank of a matrix of integers, by elimination in exact rational arithmetic."""
    matrix = [[Fraction(value) for value in row] for row in rows]
    if not matrix:
        return 0
    rank = 0
    for column in range(len(matrix[0])):
        pivot = next((r for r in range(rank, len(matrix)) if matrix[r][column] != 0), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        for r in range(len(matrix)):
            if r != rank and matrix[r][column] != 0:
                factor = matrix[r][column] / matrix[rank][column]
                matrix[r] = [a - factor * b for a, b in zip(matrix[r], matrix[rank], strict=True)]
        rank += 1
    return rank
