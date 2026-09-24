"""The index contract of M1 (docs/m1_plan.md, section 4.3): windows, phases and contexts
found from the known inputs, on runs of P3 with the lead of M1, on the P3 runs of M0 that
have no lead, and on single-input steps."""

import numpy as np
import pytest

from m1_support import NOMINAL, corner, observations, p3_inputs, random_corners, step_inputs
from process_transfer.evaluation.windows import (
    CONTEXT_READINGS,
    P3_LAYOUT,
    STEP_LAYOUT,
    Phase,
    WindowLayout,
    find_windows,
    layout_for,
    window_data,
)


def p3_run(n_excursions: int, lead: bool = True, seed: int = 0):
    run = observations(p3_inputs(random_corners(n_excursions, seed), lead=lead))
    return run, find_windows(run, NOMINAL, P3_LAYOUT)


def test_the_first_window_of_a_run_with_a_lead_follows_the_worked_example() -> None:
    _, found = p3_run(2)
    first, second = found.windows
    assert found.onsets == (10, 130) and found.skipped == ()
    assert list(first.context_ticks) == list(range(1, 11))
    assert list(first.scored_ticks) == list(range(11, 121))
    assert list(first.input_ticks) == list(range(10, 120))
    assert list(second.context_ticks) == list(range(121, 131))
    assert list(second.scored_ticks) == list(range(131, 241))
    assert found.n_ticks == 251


def test_the_phases_of_p3_are_20_50_and_40_readings() -> None:
    _, found = p3_run(1)
    scored = list(found.windows[0].scored_ticks)
    phases = {name: scored[positions] for name, positions in P3_LAYOUT.phase_slices()}
    assert phases["excursion"] == list(range(11, 31))
    assert phases["return"] == list(range(31, 81))
    assert phases["settled"] == list(range(81, 121))
    assert P3_LAYOUT.scored_readings == 110


def test_every_reading_of_a_run_is_a_context_scored_or_unused_never_two() -> None:
    _, found = p3_run(5)
    roles: dict[int, list[str]] = {tick: [] for tick in range(found.n_ticks)}
    for window in found.windows:
        for tick in window.context_ticks:
            roles[tick].append(f"context {window.excursion}")
        for tick in window.scored_ticks:
            roles[tick].append(f"scored {window.excursion}")
    assert all(len(found_roles) <= 1 for found_roles in roles.values())
    unused = [tick for tick, found_roles in roles.items() if not found_roles]
    assert unused == [0, *range(found.n_ticks - 10, found.n_ticks)]


def test_the_row_at_an_onset_carries_the_corner_and_its_reading_is_context() -> None:
    corners = [[1, -1, 1, -1]]
    run = observations(p3_inputs(corners))
    (data,) = window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)
    np.testing.assert_array_equal(run.inputs[10], corner(corners[0]))
    np.testing.assert_array_equal(run.inputs[9], NOMINAL)
    # tick-coded readings: C_A is the tick
    np.testing.assert_array_equal(data.context[:, 0], np.arange(1, 11))
    np.testing.assert_array_equal(data.scored[:, 0], np.arange(11, 121))
    np.testing.assert_array_equal(data.initial_state, [5.5, 1005.5])
    # the inputs that drive the prediction: the corner for 20 rows, then nominal; the input
    # of row 120 acts after the last scored instant and is not held
    np.testing.assert_array_equal(data.inputs[:20], np.tile(corner(corners[0]), (20, 1)))
    np.testing.assert_array_equal(data.inputs[20:], np.tile(NOMINAL, (90, 1)))
    assert data.inputs.shape == (110, 4)
    np.testing.assert_array_equal(data.scored_offsets, 6.0 * np.arange(1, 111))
    assert data.onset_time == 60.0


def test_window_data_holds_read_only_copies_and_no_row_outside_its_window() -> None:
    run, found = p3_run(3)
    for data in window_data(run, found.windows):
        window = data.window
        assert data.ticks_read == frozenset(range(window.first_tick, window.last_tick + 1))
        for values in (data.context, data.scored, data.inputs, data.initial_state):
            assert not values.flags.writeable


def test_the_runs_of_m0_without_a_lead_have_no_window_for_their_first_excursion() -> None:
    run, found = p3_run(10, lead=False)
    assert run.n_samples == 1201
    assert found.onsets == tuple(120 * j for j in range(10))
    assert [window.excursion for window in found.windows] == list(range(2, 11))
    (skipped,) = found.skipped
    assert (skipped.excursion, skipped.onset) == (1, 0)
    assert "no context" in skipped.reason
    second, last = found.windows[0], found.windows[-1]
    assert list(second.context_ticks) == list(range(111, 121))
    assert list(second.scored_ticks) == list(range(121, 231))
    assert list(last.scored_ticks) == list(range(1081, 1191))


def test_a_run_that_ends_before_a_window_does_records_the_excursion_without_it() -> None:
    inputs = p3_inputs(random_corners(2, 1))[:200]  # cut inside the second window
    found = find_windows(observations(inputs), NOMINAL, P3_LAYOUT)
    assert [window.onset for window in found.windows] == [10]
    (skipped,) = found.skipped
    assert skipped.onset == 130 and "ends at tick 199" in skipped.reason


def test_a_single_input_step_has_one_window_of_200_readings_in_four_phases() -> None:
    run = observations(step_inputs(3, +1), run="target.step.tc-up.l600.h600.r600.n0")
    found = find_windows(run, NOMINAL, STEP_LAYOUT)
    (window,) = found.windows
    assert window.onset == 100
    assert list(window.context_ticks) == list(range(91, 101))
    assert list(window.scored_ticks) == list(range(101, 301))
    scored = list(window.scored_ticks)
    bounds = [(scored[s][0], scored[s][-1]) for _, s in STEP_LAYOUT.phase_slices()]
    assert bounds == [(101, 150), (151, 200), (201, 250), (251, 300)]


def test_windows_are_found_whatever_the_readings_say() -> None:
    inputs = p3_inputs(random_corners(3, 2))
    quiet = find_windows(observations(inputs), NOMINAL, P3_LAYOUT)
    rng = np.random.default_rng(3)
    noisy = observations(inputs, measured=rng.normal(300.0, 50.0, size=(len(inputs), 2)))
    assert find_windows(noisy, NOMINAL, P3_LAYOUT).windows == quiet.windows


def test_the_layout_follows_the_operating_mode_and_a_steady_run_has_none() -> None:
    assert layout_for("p3") is P3_LAYOUT and layout_for("step") is STEP_LAYOUT
    with pytest.raises(ValueError, match="steady run has no excursion"):
        layout_for("steady")
    steady = observations(np.tile(NOMINAL, (1201, 1)))
    assert find_windows(steady, NOMINAL, P3_LAYOUT).windows == ()


@pytest.mark.parametrize("held", [19, 21])
def test_a_corner_held_for_other_than_20_rows_is_refused(held: int) -> None:
    inputs = p3_inputs(random_corners(2, 4))
    inputs[10:130] = NOMINAL
    inputs[10 : 10 + held] = corner([1, 1, 1, 1])
    with pytest.raises(ValueError, match="phases would be read wrongly|change again"):
        find_windows(observations(inputs), NOMINAL, P3_LAYOUT)


def test_a_corner_that_changes_while_held_is_refused() -> None:
    inputs = p3_inputs([[1, 1, 1, 1]])
    inputs[15] = corner([1, 1, 1, -1])
    with pytest.raises(ValueError, match="change again before tick 30"):
        find_windows(observations(inputs), NOMINAL, P3_LAYOUT)


def test_a_context_that_reaches_the_scored_readings_before_it_is_refused() -> None:
    inputs = np.tile(NOMINAL, (400, 1))
    inputs[10:30] = corner([1, 1, 1, 1])
    inputs[125:145] = corner([-1, -1, -1, -1])  # 115 rows after the first onset, not 120
    with pytest.raises(ValueError, match="no reading may be both"):
        find_windows(observations(inputs), NOMINAL, P3_LAYOUT)


def test_an_excursion_inside_the_window_of_the_one_before_is_refused() -> None:
    inputs = np.tile(NOMINAL, (400, 1))
    inputs[10:30] = corner([1, 1, 1, 1])
    inputs[35:55] = corner([-1, -1, -1, -1])
    with pytest.raises(ValueError, match="phases would be read wrongly"):
        find_windows(observations(inputs), NOMINAL, P3_LAYOUT)


def test_a_run_with_a_missing_tick_or_another_clock_is_refused() -> None:
    inputs = p3_inputs(random_corners(1, 5))
    run = observations(inputs)
    gap = type(run)(
        **{
            **run.__dict__,
            "times": np.delete(run.times, 50),
            "measured": np.delete(run.measured, 50, axis=0),
            "inputs": np.delete(run.inputs, 50, axis=0),
        }
    )
    with pytest.raises(ValueError, match="not the consecutive ticks"):
        find_windows(gap, NOMINAL, P3_LAYOUT)
    with pytest.raises(ValueError, match="every 5.0 s"):
        find_windows(observations(inputs, sample_period=5.0), NOMINAL, P3_LAYOUT)


def test_nominal_inputs_that_no_row_carries_are_refused_not_read_as_one_excursion() -> None:
    run = observations(p3_inputs(random_corners(2, 8)))
    off_by_one_bit = NOMINAL.copy()
    off_by_one_bit[0] = np.nextafter(NOMINAL[0], 1.0)
    with pytest.raises(ValueError, match="no row of run"):
        find_windows(run, off_by_one_bit, P3_LAYOUT)


def test_channels_in_another_order_are_refused() -> None:
    run = observations(p3_inputs(random_corners(1, 6)))
    swapped = type(run)(**{**run.__dict__, "measured_names": ("T", "C_A")})
    with pytest.raises(ValueError, match="evaluation contract is written for"):
        find_windows(swapped, NOMINAL, P3_LAYOUT)


def test_window_data_refuses_a_window_of_another_run() -> None:
    run, found = p3_run(1)
    other = observations(p3_inputs(random_corners(1, 7)), run="target.p3.e1.x1.n0")
    with pytest.raises(ValueError, match="belongs to run"):
        window_data(other, found.windows)


def test_a_layout_is_checked_when_it_is_made() -> None:
    with pytest.raises(ValueError, match="at least one reading"):
        WindowLayout("p3", 1, (Phase("a", 0),))
    with pytest.raises(ValueError, match="distinct"):
        WindowLayout("p3", 1, (Phase("a", 1), Phase("a", 1)))
    with pytest.raises(ValueError, match="must end inside the window"):
        WindowLayout("p3", 3, (Phase("a", 2),))
    assert CONTEXT_READINGS == 10
