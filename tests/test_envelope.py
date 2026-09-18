"""D-009 across the input box: corner stability and the range the states visit
under the excitation amplitudes of D-010."""

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


def d010_deviations(u_nominal: np.ndarray) -> np.ndarray:
    """q and C_Af +-20 % of nominal, T_f and T_c +-5 K (docs/decisions.md D-010)."""
    return np.array([0.20 * u_nominal[0], 0.20 * u_nominal[1], 5.0, 5.0])


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
def verification(configs_dir: Path) -> dict[str, Verification]:
    result: dict[str, Verification] = {}
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
        cases = input_cases(u, d010_deviations(u))
        envelope = simulate_envelope(
            lambda x, inputs, p=p: cstr_true.rhs(0.0, x, inputs, p), nominal.state, cases
        )
        corners = {label: steady_states_at(inputs) for label, inputs in cases}
        result[name] = (u, envelope, corners)
    return result


@pytest.mark.parametrize("name", ["source", "target"])
def test_every_input_case_has_a_unique_stable_steady_state(
    name: str, verification: dict[str, Verification]
) -> None:
    """No fold and no Hopf bifurcation anywhere on the boundary of the input box."""
    for label, steady_states in verification[name][2].items():
        assert len(steady_states) == 1, label
        assert steady_states[0].is_stable(), label


@pytest.mark.parametrize("name", ["source", "target"])
def test_every_step_response_converges_to_the_steady_state_of_its_inputs(
    name: str, verification: dict[str, Verification]
) -> None:
    """After 40 min every case sits on the unique steady state of its inputs: no
    runaway and no sustained oscillation."""
    _, envelope, corners = verification[name]
    for case in envelope:
        (steady,) = corners[case.label]
        np.testing.assert_allclose(case.final_state, steady.state, rtol=1e-6, err_msg=case.label)


@pytest.mark.parametrize("name", ["source", "target"])
def test_concentration_stays_physical(name: str, verification: dict[str, Verification]) -> None:
    u, envelope, _ = verification[name]
    for case in envelope:
        assert case.c_a_range[0] > 0.0, case.label
        assert case.c_a_range[1] <= 1.2 * u[1], case.label


@pytest.mark.parametrize("name", ["source", "target"])
def test_hottest_case_is_the_all_plus_corner(
    name: str, verification: dict[str, Verification]
) -> None:
    """More flow of a richer feed brings more reactant, hence more heat: the hottest
    corner has q up, not down. The preliminary scratch calculation assumed the
    opposite and therefore underestimated the peak temperature."""
    hottest = max(verification[name][1], key=lambda case: case.temperature_range[1])
    assert hottest.label == ALL_PLUS


def test_source_stays_inside_the_temperature_envelope(
    verification: dict[str, Verification],
) -> None:
    for case in verification["source"][1]:
        assert case.temperature_range[0] >= TEMPERATURE_ENVELOPE[0], case.label
        assert case.temperature_range[1] <= TEMPERATURE_ENVELOPE[1], case.label


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M0-E01: under the D-010 amplitudes the target peaks at about 385 K on the "
        "all-plus corner, above the documented 380 K limit. Open decision D-018; "
        "this marker must be removed when the design or the amplitudes change."
    ),
)
def test_target_stays_inside_the_temperature_envelope(
    verification: dict[str, Verification],
) -> None:
    for case in verification["target"][1]:
        assert case.temperature_range[0] >= TEMPERATURE_ENVELOPE[0], case.label
        assert case.temperature_range[1] <= TEMPERATURE_ENVELOPE[1], case.label
