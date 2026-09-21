"""Protocol P3 (D-019): fast regression checks of what M0-E03b measures in full.

The experiment sweeps all 256 ordered pairs of excursions and takes about 15 s; it is
kept out of the test suite. Here: the recovery after every corner, which is cheap, and
a handful of pairs chosen for being the extreme ones in the sweep. The tolerances are
those of ``simulation/protocols.py`` and were fixed from the sensor noise and the
margin to the limit, not from these results.
"""

import dataclasses

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.simulation.checks import check_trajectory
from process_transfer.simulation.excitation import separated_excursions
from process_transfer.simulation.integration import Trajectory, simulate_piecewise
from process_transfer.simulation.protocols import (
    P3_HOLD,
    P3_PEAK_AGREEMENT,
    P3_RECOVERY_TOLERANCE_CA,
    P3_RECOVERY_TOLERANCE_T,
    P3_REST,
    a10_amplitudes,
    corner_label,
    corner_levels,
    p3_corners,
    p3_segments,
)

COLD, HOT = np.array([1, 1, -1, -1]), np.array([1, 1, 1, 1])
ALL_MINUS = np.array([-1, -1, -1, -1])


def run(plant: PlantUnderTest, corners: np.ndarray) -> Trajectory:
    return simulate_piecewise(
        plant.f, plant.nominal_state, p3_segments(plant.nominal_inputs, corners), 0.1
    )


def test_the_protocol_is_the_one_accepted_in_d019(true_plants: dict[str, PlantUnderTest]) -> None:
    u = true_plants["source"].nominal_inputs
    np.testing.assert_allclose(a10_amplitudes(u), [0.10 * u[0], 0.10 * u[1], 5.0, 5.0])
    assert (P3_HOLD, P3_REST) == (120.0, 600.0)
    corners = corner_levels()
    assert corners.shape == (16, 4) and len({corner_label(c) for c in corners}) == 16

    segments = p3_segments(u, np.array([COLD, HOT]))
    assert [segment.duration for segment in segments] == [120.0, 600.0, 120.0, 600.0]
    np.testing.assert_array_equal(segments[1].inputs, u)
    np.testing.assert_allclose(segments[2].inputs, u + a10_amplitudes(u))


@pytest.mark.parametrize(
    ("name", "expected_c_a", "expected_t"),
    [("source", 2.199e-5, 7.308e-7), ("target", 6.505e-3, 1.333e-3)],
)
def test_every_corner_excursion_recovers_within_tolerance(
    name: str, expected_c_a: float, expected_t: float, true_plants: dict[str, PlantUnderTest]
) -> None:
    """Residual distance to the nominal steady state after 120 s at a corner and 600 s
    at nominal. The largest values, on the ++-- corner of both plants, are those found
    independently by the reviewer."""
    plant = true_plants[name]
    residuals = {}
    for corner in corner_levels():
        trajectory = run(plant, corner[np.newaxis, :])
        assert check_trajectory(trajectory, plant.parameters).accepted
        residuals[corner_label(corner)] = np.abs(trajectory.states[-1] - plant.nominal_state)

    worst_c_a = max(residuals, key=lambda label: residuals[label][0])
    worst_t = max(residuals, key=lambda label: residuals[label][1])
    assert worst_c_a == worst_t == "++--"
    assert residuals[worst_c_a][0] == pytest.approx(expected_c_a, rel=0.05)
    assert residuals[worst_t][1] == pytest.approx(expected_t, rel=0.05)
    assert residuals[worst_c_a][0] <= P3_RECOVERY_TOLERANCE_CA
    assert residuals[worst_t][1] <= P3_RECOVERY_TOLERANCE_T


@pytest.mark.parametrize(("name", "expected_peak"), [("source", 365.7505), ("target", 376.1895)])
def test_the_state_carried_into_the_next_excursion_does_not_change_its_peak(
    name: str, expected_peak: float, true_plants: dict[str, PlantUnderTest]
) -> None:
    """Pairs that were extreme in the full sweep: the hottest excursion after the
    coldest, after itself and after the all-minus corner, and a cold one after a cold
    one. The state is not reset between the two excursions."""
    plant = true_plants[name]
    alone = {
        corner_label(corner): run(plant, corner[np.newaxis, :]).refined_peak(1)[0]
        for corner in (COLD, HOT)
    }
    largest = -np.inf
    for first, second in ((COLD, HOT), (HOT, HOT), (ALL_MINUS, HOT), (COLD, COLD)):
        trajectory = run(plant, np.array([first, second]))
        assert check_trajectory(trajectory, plant.parameters).accepted

        rest_end, next_start = trajectory.segments[1].states[-1], trajectory.segments[2].states[0]
        np.testing.assert_array_equal(next_start, rest_end)  # carried over, continuously
        assert not np.array_equal(next_start, plant.nominal_state)  # and not reset

        second_half = dataclasses.replace(trajectory, segments=trajectory.segments[2:])
        peak = second_half.refined_peak(1)[0]
        assert abs(peak - alone[corner_label(second)]) <= P3_PEAK_AGREEMENT
        largest = max(largest, trajectory.refined_peak(1)[0])
    assert largest == pytest.approx(expected_peak, abs=1e-3)
    assert largest < 380.0


@pytest.mark.parametrize("name", ["source", "target"])
def test_a_seeded_p3_sequence_is_accepted(
    name: str, true_plants: dict[str, PlantUnderTest]
) -> None:
    plant = true_plants[name]
    segments = separated_excursions(
        plant.nominal_inputs,
        a10_amplitudes(plant.nominal_inputs),
        n_excursions=4,
        hold=P3_HOLD,
        rest=P3_REST,
        rng=np.random.default_rng(1),
    )
    check = check_trajectory(
        simulate_piecewise(plant.f, plant.nominal_state, segments, 0.1), plant.parameters
    )
    assert check.accepted
    assert check.refined_peak_temperature < 380.0


@pytest.mark.parametrize("name", ["source", "target"])
def test_a_history_of_ten_excursions_behaves_like_ten_steps_from_nominal(
    name: str, true_plants: dict[str, PlantUnderTest]
) -> None:
    """M0-E03b examined pairs and argued that residuals do not build up over longer
    histories. Here is one such history, the sequence of excitation seed 0 that M0-E04
    observes: every excursion starts within the recovery tolerance of the nominal steady
    state and reproduces the peak of the same excursion started exactly there. The
    tolerances are those verified for pairs, unchanged."""
    plant = true_plants[name]
    corners = p3_corners(10, seed=0)
    whole = run(plant, corners)
    assert check_trajectory(whole, plant.parameters).accepted
    assert corner_label(corners[3]) == corner_label(corners[4]) == "++++"  # the hottest, twice

    for k, corner in enumerate(corners):
        start = np.abs(whole.segments[2 * k].states[0] - plant.nominal_state)
        assert start[0] <= P3_RECOVERY_TOLERANCE_CA and start[1] <= P3_RECOVERY_TOLERANCE_T
        piece = dataclasses.replace(whole, segments=whole.segments[2 * k : 2 * k + 2])
        alone = run(plant, corner[np.newaxis, :])
        assert abs(piece.refined_peak(1)[0] - alone.refined_peak(1)[0]) <= P3_PEAK_AGREEMENT
