"""Acceptance checks of a trajectory, and the M0-E03 counterexample as a regression."""

import dataclasses

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.simulation import cstr_true
from process_transfer.simulation.checks import (
    TrajectoryCheck,
    check_trajectory,
    comparison_is_valid,
)
from process_transfer.simulation.integration import InputSegment, simulate_piecewise

L_PER_MIN = 1.0e-3 / 60.0
COLD = np.array([110.0 * L_PER_MIN, 550.0, 345.0, 332.5])  # q+, C_Af+, T_f-, T_c- at A10
HOT = np.array([110.0 * L_PER_MIN, 550.0, 355.0, 342.5])  # q+, C_Af+, T_f+, T_c+ at A10


def cold_then_hot() -> list[InputSegment]:
    return [InputSegment(120.0, COLD), InputSegment(120.0, HOT)]


def test_a_rest_at_the_nominal_inputs_is_accepted(true_plants: dict[str, PlantUnderTest]) -> None:
    for plant in true_plants.values():
        rest = [InputSegment(300.0, plant.nominal_inputs)]
        check = check_trajectory(
            simulate_piecewise(plant.f, plant.nominal_state, rest), plant.parameters
        )
        assert check.accepted
        assert check.seconds_above_limit == 0.0
        assert check.peak_temperature == pytest.approx(plant.nominal_state[1], abs=1e-6)


def test_a10_sequential_counterexample_takes_the_target_far_above_the_limit(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """M0-E03. The A10 amplitudes keep every step from the nominal steady state below
    380 K, but a cold stage followed by a hot one does not: the cold stage stores
    reactant, and heating then releases its heat at once. A10 is therefore NOT known to
    be safe for sequential excitation; this test pins the counterexample so that the
    requirement cannot be assumed closed."""
    plant = true_plants["target"]
    trajectory = simulate_piecewise(plant.f, plant.nominal_state, cold_then_hot(), 0.01)
    check = check_trajectory(trajectory, plant.parameters)

    end_of_cold_stage = trajectory.segments[0].states[-1]
    assert end_of_cold_stage[0] == pytest.approx(347.40, abs=0.05)  # mol/m^3, from 189.67
    assert end_of_cold_stage[1] == pytest.approx(345.14, abs=0.05)  # K

    assert check.peak_temperature == pytest.approx(395.63, abs=0.01)
    assert check.peak_time - 120.0 == pytest.approx(42.55, abs=0.1)
    assert not check.inside_envelope
    assert check.states_physical and check.balances_close  # a real trajectory, not an artefact
    assert not check.accepted
    # A real trajectory that leaves the envelope is not thereby physically invalid: this is
    # the diagnostic case M0-E08's variant B is designed to allow (comparison_is_valid below).
    assert check.physically_valid
    assert check.seconds_above_limit == pytest.approx(19.05, abs=0.1)  # s above 380 K


@pytest.mark.parametrize("method", ["LSODA", "DOP853", "Radau"])
def test_the_counterexample_does_not_depend_on_the_integrator(
    method: str, true_plants: dict[str, PlantUnderTest]
) -> None:
    plant = true_plants["target"]
    trajectory = simulate_piecewise(
        plant.f, plant.nominal_state, cold_then_hot(), 0.01, method=method, rtol=1e-11, atol=1e-11
    )
    assert trajectory.peak(1)[0] == pytest.approx(395.6297, abs=2e-4)


def test_sampling_can_only_underestimate_the_peak_and_the_refinement_recovers_it(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    plant = true_plants["target"]
    peaks = {}
    for period in (1.0, 0.1, 0.01):
        trajectory = simulate_piecewise(plant.f, plant.nominal_state, cold_then_hot(), period)
        peaks[period] = (trajectory.peak(1)[0], trajectory.refined_peak(1)[0])
    assert peaks[1.0][0] < peaks[0.1][0] < peaks[0.01][0]
    assert peaks[0.01][0] - peaks[0.1][0] < 2e-3  # 0.1 s sampling misses about 1 mK
    assert peaks[0.1][1] == pytest.approx(peaks[0.01][0], abs=2e-4)  # refined 0.1 s estimate


def test_the_same_sequence_keeps_the_source_inside_the_envelope(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    plant = true_plants["source"]
    check = check_trajectory(
        simulate_piecewise(plant.f, plant.nominal_state, cold_then_hot(), 0.1), plant.parameters
    )
    assert check.accepted
    assert check.peak_temperature == pytest.approx(375.40, abs=0.01)


def test_unphysical_states_are_reported_not_repaired(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    plant = true_plants["source"]
    trajectory = simulate_piecewise(
        plant.f, plant.nominal_state, [InputSegment(60.0, plant.nominal_inputs)]
    )
    (segment,) = trajectory.segments

    negative = segment.states.copy()
    negative[5, 0] = -1.0
    too_rich = segment.states.copy()
    too_rich[5, 0] = 2.0 * plant.nominal_inputs[1]
    too_cold = segment.states.copy()
    too_cold[5, 1] = 300.0
    for states in (negative, too_rich, too_cold):
        corrupted = dataclasses.replace(
            trajectory, segments=(dataclasses.replace(segment, states=states),)
        )
        assert not check_trajectory(corrupted, plant.parameters).states_physical


def test_non_positive_inputs_are_not_physical(true_plants: dict[str, PlantUnderTest]) -> None:
    """Regression: a negative feed flow integrates without complaint, and the trajectory
    used to be reported as physical."""
    plant = true_plants["source"]
    reversed_flow = plant.nominal_inputs.copy()
    reversed_flow[0] = -reversed_flow[0]
    trajectory = simulate_piecewise(
        plant.f, plant.nominal_state, [InputSegment(5.0, reversed_flow)]
    )
    check = check_trajectory(trajectory, plant.parameters, envelope=(0.0, 1000.0))
    assert not check.states_physical and not check.accepted


def test_a_temperature_outside_the_domain_of_the_conductance_law_is_not_physical(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """UA(T) = UA_ref [1 + alpha (T - T_ref)] turns negative below T_ref - 1/alpha, 150 K on
    the source. The enthalpy is set to zero so that no other rule is involved."""
    plant = true_plants["source"]
    athermal = dataclasses.replace(plant.parameters, reaction_enthalpy=0.0)
    trajectory = simulate_piecewise(
        lambda x, u: cstr_true.rhs(0.0, x, u, athermal),
        plant.nominal_state,
        [InputSegment(10.0, plant.nominal_inputs)],
    )
    (segment,) = trajectory.segments
    states = segment.states.copy()
    states[5, 1] = 100.0
    frozen = dataclasses.replace(
        trajectory, segments=(dataclasses.replace(segment, states=states),)
    )
    assert cstr_true.conductance(100.0, athermal) < 0.0
    assert not check_trajectory(frozen, athermal, envelope=(0.0, 1000.0)).states_physical

    constant_ua = dataclasses.replace(athermal, alpha=0.0)
    assert check_trajectory(frozen, constant_ua, envelope=(0.0, 1000.0)).states_physical


# =========================================================================== #
# M0-E08's mandatory validity gate (comparison_is_valid), and its regression:
# the verdict used to depend only on anchoring and numerical resolution, so a
# pair with a rejected primary or a secondary with open balances could still
# report H1 and H2 as holding. See docs/numerical_robustness.md.
# =========================================================================== #


def _check(**overrides: object) -> TrajectoryCheck:
    """A ``TrajectoryCheck`` that is accepted by every criterion, with the given
    fields overridden. The numeric extremes are arbitrary; only the boolean
    criteria matter to the tests that use this."""
    fields: dict[str, object] = {
        "peak_temperature": 350.0,
        "refined_peak_temperature": 350.0,
        "peak_time": 0.0,
        "min_temperature": 340.0,
        "c_a_range": (100.0, 250.0),
        "seconds_above_limit": 0.0,
        "relative_mass_residual": 1.0e-9,
        "relative_energy_residual": 1.0e-9,
        "values_finite": True,
        "states_physical": True,
        "inside_envelope": True,
        "balances_close": True,
    }
    fields.update(overrides)
    return TrajectoryCheck(**fields)  # type: ignore[arg-type]


def test_physically_valid_does_not_require_the_envelope() -> None:
    check = _check(inside_envelope=False)
    assert not check.accepted
    assert check.physically_valid


@pytest.mark.parametrize(
    "field", ["values_finite", "states_physical", "balances_close"]
)
def test_physically_valid_requires_every_other_criterion(field: str) -> None:
    check = _check(**{field: False})
    assert not check.physically_valid


def test_comparison_rejects_a_pair_whose_primary_is_not_accepted() -> None:
    """The primary (A) leaving the envelope must reject the pair, even though the
    secondary (B) is unaffected: this is the defect Codex reproduced by forcing
    ``balances_close=False`` and finding H1 and H2 still held."""
    primary = _check(inside_envelope=False)
    secondary = _check()
    assert not comparison_is_valid(primary, secondary)


def test_comparison_rejects_a_secondary_with_open_balances() -> None:
    primary = _check()
    secondary = _check(balances_close=False)
    assert not comparison_is_valid(primary, secondary)


def test_comparison_rejects_a_secondary_with_non_finite_values() -> None:
    primary = _check()
    secondary = _check(values_finite=False, states_physical=False, balances_close=False)
    assert not comparison_is_valid(primary, secondary)


def test_comparison_accepts_a_secondary_that_only_leaves_the_envelope() -> None:
    """A secondary built to probe how far the physics moves the plant, such as
    M0-E08's variant B, must not be rejected for leaving [335, 380] K alone."""
    primary = _check()
    secondary = _check(inside_envelope=False)
    assert comparison_is_valid(primary, secondary)


def test_an_invalid_case_fails_the_aggregate_verdict_and_the_exit_code() -> None:
    """Mirrors the aggregation in experiments/08_oracle_conductance.py: every case's
    ``comparison_is_valid`` feeds one ``all(...)`` that the exit code is drawn from
    (``0 if all(verdicts.values()) else 1``). One invalid case must flip both."""
    per_case_valid = [
        comparison_is_valid(_check(), _check()),
        comparison_is_valid(_check(), _check()),
        comparison_is_valid(_check(inside_envelope=False), _check()),  # A not accepted
    ]
    verdict = all(per_case_valid)
    exit_code = 0 if verdict else 1
    assert verdict is False
    assert exit_code == 1
