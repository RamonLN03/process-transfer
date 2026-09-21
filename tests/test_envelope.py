"""Step responses from the nominal steady state at 24 input cases.

Two amplitude sets are kept apart on purpose:

* the ORIGINAL D-010 amplitudes (q and C_Af +-20 %), the conditions of M0-E01, kept
  as a historical regression of its finding that the target leaves the envelope;
* the A10 amplitudes chosen in D-018 (q and C_Af +-10 %), the current verification.

Scope of every statement below: single steps applied from the nominal steady state,
at 8 single-input excursions and the 16 corners of the input box. Nothing here speaks
for the interior of the box, for the rest of its boundary, or for chained input
changes. M0-E03 shows that A10 does NOT keep the target inside the envelope when
changes are chained (tests/test_checks.py pins that counterexample), so passing these
tests does not establish that an excitation sequence is safe.
"""

from pathlib import Path

import numpy as np
import pytest

from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.simulation import cstr_true
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.envelope import EnvelopeCase, input_cases, simulate_envelope
from process_transfer.simulation.steady_state import SteadyState, find_steady_states

TEMPERATURE_ENVELOPE = (335.0, 380.0)  # K, docs/assumptions.md
ALL_PLUS = "q+ C_Af+ T_f+ T_c+"

# (relative q, relative C_Af, kelvin T_f, kelvin T_c)
AMPLITUDE_SETS = {
    "original D-010": (0.20, 0.20, 5.0, 5.0),  # historical, M0-E01
    "A10": (0.10, 0.10, 5.0, 5.0),  # current, D-018
}


def test_input_cases_are_eight_singles_and_sixteen_corners() -> None:
    nominal = np.array([1.0, 10.0, 100.0, 1000.0])
    deviations = np.array([0.1, 1.0, 10.0, 100.0])
    cases = input_cases(nominal, deviations)
    singles, corners = cases[:8], cases[8:]

    assert len(singles) == 8 and len(corners) == 16
    assert len({label for label, _ in cases}) == 24
    for _, u in singles:
        assert np.count_nonzero(u != nominal) == 1
    for _, u in corners:
        np.testing.assert_allclose(np.abs(u - nominal), deviations)
    assert len({tuple(np.sign(u - nominal)) for _, u in corners}) == 16
    np.testing.assert_array_equal(nominal, [1.0, 10.0, 100.0, 1000.0])  # not mutated


Verification = tuple[np.ndarray, list[EnvelopeCase], dict[str, list[SteadyState]]]


@pytest.fixture(scope="module")
def verification(configs_dir: Path) -> dict[tuple[str, str], Verification]:
    result: dict[tuple[str, str], Verification] = {}
    for name in ("source", "target"):
        cfg = load_true_plant(configs_dir / f"{name}_cstr.yaml")
        p, u = TrueCSTRParameters.from_config(cfg), nominal_inputs(cfg.plant)

        def steady_states_at(inputs: np.ndarray, p: TrueCSTRParameters = p) -> list[SteadyState]:
            return find_steady_states(
                lambda x: cstr_true.rhs(0.0, x, inputs, p),
                c_a_upper=inputs[1],
                temperature_range=(300.0, 460.0),
                n_grid=321,
            )

        (nominal,) = steady_states_at(u)
        for label, (rel_q, rel_c, kelvin_f, kelvin_c) in AMPLITUDE_SETS.items():
            deviations = np.array([rel_q * u[0], rel_c * u[1], kelvin_f, kelvin_c])
            cases = input_cases(u, deviations)
            envelope = simulate_envelope(
                lambda x, inputs, p=p: cstr_true.rhs(0.0, x, inputs, p), nominal.state, cases
            )
            steady = {case_label: steady_states_at(inputs) for case_label, inputs in cases}
            result[(label, name)] = (u, envelope, steady)
    return result


ALL_KEYS = [(label, name) for label in AMPLITUDE_SETS for name in ("source", "target")]


def temperature_range(envelope: list[EnvelopeCase]) -> tuple[float, float]:
    return (
        min(case.temperature_range[0] for case in envelope),
        max(case.temperature_range[1] for case in envelope),
    )


@pytest.mark.parametrize("key", ALL_KEYS)
def test_each_tested_input_case_has_one_stable_steady_state_in_the_scanned_range(
    key: tuple[str, str], verification: dict[tuple[str, str], Verification]
) -> None:
    """At each of the 24 tested input cases, between 300 and 460 K on a 0.5 K grid. This
    is a statement about those 24 points, not about the whole input box."""
    for label, steady_states in verification[key][2].items():
        assert len(steady_states) == 1, label
        assert steady_states[0].is_stable(), label


@pytest.mark.parametrize("key", ALL_KEYS)
def test_every_step_response_converges_to_the_steady_state_of_its_inputs(
    key: tuple[str, str], verification: dict[tuple[str, str], Verification]
) -> None:
    _, envelope, steady = verification[key]
    for case in envelope:
        (state,) = steady[case.label]
        np.testing.assert_allclose(case.final_state, state.state, rtol=1e-6, err_msg=case.label)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_concentration_stays_physical(
    key: tuple[str, str], verification: dict[tuple[str, str], Verification]
) -> None:
    u, envelope, _ = verification[key]
    for case in envelope:
        assert case.c_a_range[0] > 0.0, case.label
        assert case.c_a_range[1] <= 1.2 * u[1], case.label


@pytest.mark.parametrize("key", ALL_KEYS)
def test_hottest_step_is_the_all_plus_corner(
    key: tuple[str, str], verification: dict[tuple[str, str], Verification]
) -> None:
    """More flow of a richer feed brings more reactant, hence more heat: among single
    steps from nominal the hottest corner has q up, not down. The first scratch
    calculation assumed the opposite and underestimated the peak."""
    hottest = max(verification[key][1], key=lambda case: case.temperature_range[1])
    assert hottest.label == ALL_PLUS


# --------------------------------------------------------------------------- #
# Historical: the original D-010 amplitudes, as run in M0-E01
# --------------------------------------------------------------------------- #


def test_historical_original_amplitudes_keep_the_source_inside_the_envelope(
    verification: dict[tuple[str, str], Verification],
) -> None:
    low, high = temperature_range(verification[("original D-010", "source")][1])
    assert (low, high) == pytest.approx((339.67, 372.44), abs=0.01)
    assert TEMPERATURE_ENVELOPE[0] <= low and high <= TEMPERATURE_ENVELOPE[1]


def test_historical_original_amplitudes_take_the_target_outside_the_envelope(
    verification: dict[tuple[str, str], Verification],
) -> None:
    """The finding of M0-E01 that led to D-018, kept reproducible: 385.29 K on the
    all-plus corner, 5.3 K above the limit."""
    envelope = verification[("original D-010", "target")][1]
    low, high = temperature_range(envelope)
    assert (low, high) == pytest.approx((341.19, 385.29), abs=0.01)
    assert high > TEMPERATURE_ENVELOPE[1]
    too_hot = [case.label for case in envelope if case.temperature_range[1] > 380.0]
    assert ALL_PLUS in too_hot


# --------------------------------------------------------------------------- #
# Current: the A10 amplitudes chosen in D-018
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"), [("source", (340.42, 365.75)), ("target", (342.37, 376.19))]
)
def test_a10_steps_from_the_nominal_steady_state_stay_inside_the_envelope(
    name: str, expected: tuple[float, float], verification: dict[tuple[str, str], Verification]
) -> None:
    """What D-018 established, and no more: single steps from nominal. Sequential
    excitation with the same amplitudes fails on the target (M0-E03)."""
    low, high = temperature_range(verification[("A10", name)][1])
    assert (low, high) == pytest.approx(expected, abs=0.01)
    assert TEMPERATURE_ENVELOPE[0] <= low and high <= TEMPERATURE_ENVELOPE[1]
