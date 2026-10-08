"""The lead and the amplitude A5 of P3 (D-039; ``docs/m1_plan.md``, sections 4.2 and 6.1).

What A5 does to the plant is the question of M1-E02, which is an experiment and not a test.
The tests here therefore build the inputs of P3 at A5 and never simulate them: they check
amplitudes, durations, instants and the order of the corners. The continuity of the state
through the lead is checked by simulation at A10, the amplitude M0-E03b verified.
"""

import dataclasses

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.simulation.checks import check_trajectory
from process_transfer.simulation.excitation import excursions_with_rest
from process_transfer.simulation.integration import simulate_piecewise
from process_transfer.simulation.protocols import (
    P3_AMPLITUDES,
    P3_HOLD,
    P3_LEAD,
    P3_PEAK_AGREEMENT,
    P3_RECOVERY_TOLERANCE_CA,
    P3_RECOVERY_TOLERANCE_T,
    P3_REST,
    a5_amplitudes,
    a10_amplitudes,
    corner_levels,
    p3_amplitudes,
    p3_corners,
    p3_segments,
)


@pytest.fixture
def nominal(true_plants: dict[str, PlantUnderTest]) -> np.ndarray:
    return true_plants["target"].nominal_inputs


def test_a5_is_half_of_a10_to_the_last_bit(nominal: np.ndarray) -> None:
    a5 = a5_amplitudes(nominal)
    np.testing.assert_array_equal(2.0 * a5, a10_amplitudes(nominal))
    np.testing.assert_allclose(a5, [0.05 * nominal[0], 0.05 * nominal[1], 2.5, 2.5], rtol=1e-15)
    assert P3_AMPLITUDES == ("a10", "a5") and P3_LEAD == 60.0
    np.testing.assert_array_equal(p3_amplitudes(nominal, "a5"), a5)
    np.testing.assert_array_equal(p3_amplitudes(nominal, "a10"), a10_amplitudes(nominal))


@pytest.mark.parametrize("amplitude", ["A5", "a7", "", None, 5])
def test_an_amplitude_that_is_not_named_is_refused(nominal: np.ndarray, amplitude: object) -> None:
    with pytest.raises(ValueError, match="amplitude must be one of"):
        p3_amplitudes(nominal, amplitude)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="amplitude must be one of"):
        p3_segments(nominal, corner_levels()[:1], amplitude=amplitude)  # type: ignore[arg-type]


def test_without_a_lead_or_an_amplitude_p3_is_that_of_m0(nominal: np.ndarray) -> None:
    corners = p3_corners(10, 0)
    old = excursions_with_rest(nominal, a10_amplitudes(nominal), corners, P3_HOLD, P3_REST)
    for new in (
        p3_segments(nominal, corners),
        p3_segments(nominal, corners, "a10"),
        p3_segments(nominal, corners, "a10", 0.0),
        p3_segments(nominal, corners, lead=0),
    ):
        assert [s.duration for s in new] == [s.duration for s in old]
        for a, b in zip(new, old, strict=True):
            np.testing.assert_array_equal(a.inputs, b.inputs)


@pytest.mark.parametrize("amplitude", ["a10", "a5"])
def test_the_lead_comes_first_and_changes_nothing_after_it(
    nominal: np.ndarray, amplitude: str
) -> None:
    corners = p3_corners(40, 7)
    without = p3_segments(nominal, corners, amplitude)
    with_lead = p3_segments(nominal, corners, amplitude, P3_LEAD)
    assert len(with_lead) == len(without) + 1 == 81
    assert with_lead[0].duration == 60.0
    np.testing.assert_array_equal(with_lead[0].inputs, nominal)
    for a, b in zip(with_lead[1:], without, strict=True):
        assert a.duration == b.duration
        np.testing.assert_array_equal(a.inputs, b.inputs)
    # the corners are those of the seed, in order, at the declared amplitude
    amplitudes = p3_amplitudes(nominal, amplitude)
    for j, corner in enumerate(corners):
        np.testing.assert_array_equal(with_lead[1 + 2 * j].inputs, nominal + corner * amplitudes)
        np.testing.assert_array_equal(with_lead[2 + 2 * j].inputs, nominal)


def test_the_switching_instants_with_the_lead_are_those_of_the_plan(nominal: np.ndarray) -> None:
    segments = p3_segments(nominal, p3_corners(40, 7), "a5", P3_LEAD)
    instants = np.cumsum([0.0] + [s.duration for s in segments])
    # excursion j has its onset at 60 + 720 (j - 1) s, tick 10 + 120 (j - 1) of 6 s
    onsets = instants[1:-1:2]
    np.testing.assert_array_equal(onsets, 60.0 + 720.0 * np.arange(40))
    np.testing.assert_array_equal(onsets / 6.0, 10 + 120 * np.arange(40))
    assert instants[-1] == 28860.0  # 481 min, a run of 40 excursions (plan, section 5.2)
    assert np.all(np.mod(instants, 6.0) == 0.0)


@pytest.mark.parametrize("lead", [-6.0, -0.0001, np.nan, np.inf, True, "60", None])
def test_a_lead_outside_its_domain_is_refused(nominal: np.ndarray, lead: object) -> None:
    with pytest.raises(ValueError, match="lead must be a finite number of seconds"):
        p3_segments(nominal, corner_levels()[:1], "a10", lead)  # type: ignore[arg-type]


def test_the_state_is_carried_through_the_lead_and_never_reset(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """At A10, verified in M0-E03b: the lead leaves the steady state where it is, the state
    is continuous at every switch, and the excursion after the lead peaks where the same
    excursion from the exact steady state does."""
    plant = true_plants["target"]
    corner = np.array([[1, 1, 1, 1]])
    segments = p3_segments(plant.nominal_inputs, corner, "a10", P3_LEAD)
    trajectory = simulate_piecewise(plant.f, plant.nominal_state, segments, 0.1)
    assert check_trajectory(trajectory, plant.parameters).accepted
    np.testing.assert_array_equal(trajectory.segments[0].states[0], plant.nominal_state)
    for before, after in zip(trajectory.segments, trajectory.segments[1:], strict=False):
        np.testing.assert_array_equal(after.states[0], before.states[-1])
    end_of_lead = np.abs(trajectory.segments[0].states[-1] - plant.nominal_state)
    assert end_of_lead[0] <= P3_RECOVERY_TOLERANCE_CA and end_of_lead[1] <= P3_RECOVERY_TOLERANCE_T

    without = simulate_piecewise(
        plant.f, plant.nominal_state, p3_segments(plant.nominal_inputs, corner), 0.1
    )
    after_lead = dataclasses.replace(trajectory, segments=trajectory.segments[1:])
    gap = abs(after_lead.refined_peak(1)[0] - without.refined_peak(1)[0])
    assert gap <= P3_PEAK_AGREEMENT
