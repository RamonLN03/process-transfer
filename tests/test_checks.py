"""Acceptance checks of a trajectory, and the M0-E03 counterexample as a regression."""

import dataclasses

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.simulation.checks import check_trajectory
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
