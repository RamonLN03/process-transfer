"""Integrated mass and energy balances as an independent check of a trajectory."""

import dataclasses

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.simulation.balances import integrated_balances
from process_transfer.simulation.integration import InputSegment, simulate_piecewise

L_PER_MIN = 1.0e-3 / 60.0


def two_stage_sequence() -> list[InputSegment]:
    """A cold stage followed by a hot stage at the A10 levels (M0-E03 counterexample)."""
    return [
        InputSegment(120.0, np.array([110.0 * L_PER_MIN, 550.0, 345.0, 332.5])),
        InputSegment(120.0, np.array([110.0 * L_PER_MIN, 550.0, 355.0, 342.5])),
    ]


@pytest.mark.parametrize("name", ["source", "target"])
def test_balances_close_on_a_finely_sampled_trajectory(
    name: str, true_plants: dict[str, PlantUnderTest]
) -> None:
    plant = true_plants[name]
    trajectory = simulate_piecewise(plant.f, plant.nominal_state, two_stage_sequence(), 0.1)
    check = integrated_balances(trajectory, plant.parameters)
    assert check.reacted > 0.0
    assert check.adiabatic_heating > 0.0
    assert check.relative_mass_residual < 1.0e-6
    assert check.relative_energy_residual < 1.0e-6


def test_the_check_detects_inputs_attached_to_the_wrong_segment(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """Swap the recorded inputs of the two stages while keeping the states: the stored
    trajectory no longer satisfies the balances, and the check must say so."""
    plant = true_plants["target"]
    trajectory = simulate_piecewise(plant.f, plant.nominal_state, two_stage_sequence(), 0.1)
    first, second = trajectory.segments
    corrupted = dataclasses.replace(
        trajectory,
        segments=(
            dataclasses.replace(first, inputs=second.inputs),
            dataclasses.replace(second, inputs=first.inputs),
        ),
    )
    check = integrated_balances(corrupted, plant.parameters)
    assert check.relative_energy_residual > 1.0e-2


def test_coarse_sampling_shows_up_as_a_larger_residual(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    plant = true_plants["target"]
    fine = integrated_balances(
        simulate_piecewise(plant.f, plant.nominal_state, two_stage_sequence(), 0.1),
        plant.parameters,
    )
    coarse = integrated_balances(
        simulate_piecewise(plant.f, plant.nominal_state, two_stage_sequence(), 10.0),
        plant.parameters,
    )
    assert coarse.relative_energy_residual > 100.0 * fine.relative_energy_residual
