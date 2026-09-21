"""Integrated mass and energy balances as an independent check of a trajectory.

Includes regression tests for a confirmed defect: the energy residual was divided by
the signed heating of the reaction, so an endothermic reaction gave a negative
"relative error" that passed any tolerance, and a zero enthalpy divided by zero.

Several cases below use parameter values that the configuration schema does not
admit, such as k0 = 0. They are built directly, on purpose, to exercise the numerical
functions at their limits; they do not widen what a configuration file may contain.
"""

import dataclasses
import math

import numpy as np
import pytest

from conftest import PlantUnderTest
from process_transfer.simulation import cstr_true
from process_transfer.simulation.balances import BalanceCheck, integrated_balances
from process_transfer.simulation.checks import BALANCE_TOLERANCE, check_trajectory
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.integration import InputSegment, Trajectory, simulate_piecewise
from process_transfer.simulation.steady_state import find_steady_states

L_PER_MIN = 1.0e-3 / 60.0
WIDE_ENVELOPE = (0.0, 1000.0)  # so that only the balances decide in check_trajectory


def two_stage_sequence() -> list[InputSegment]:
    """A cold stage followed by a hot stage at the A10 levels (M0-E03 counterexample)."""
    return [
        InputSegment(120.0, np.array([110.0 * L_PER_MIN, 550.0, 345.0, 332.5])),
        InputSegment(120.0, np.array([110.0 * L_PER_MIN, 550.0, 355.0, 342.5])),
    ]


def at_rest(p: TrueCSTRParameters, u: np.ndarray, period: float = 0.01) -> Trajectory:
    """120 s at the nominal inputs, from the steady state of these parameters."""
    (steady,) = find_steady_states(lambda x: cstr_true.rhs(0.0, x, u, p), c_a_upper=u[1])
    return simulate_piecewise(
        lambda x, inputs: cstr_true.rhs(0.0, x, inputs, p),
        steady.state,
        [InputSegment(120.0, u)],
        period,
    )


def with_last_sample_shifted(trajectory: Trajectory, state: int, delta: float) -> Trajectory:
    """The same trajectory with one state of its last sample altered: an accumulation
    that the recorded inputs cannot explain."""
    *head, last = trajectory.segments
    states = last.states.copy()
    states[-1, state] += delta
    return dataclasses.replace(
        trajectory, segments=(*head, dataclasses.replace(last, states=states))
    )


# --------------------------------------------------------------------------- #
# Exothermic, as configured
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["source", "target"])
def test_balances_close_on_a_finely_sampled_trajectory(
    name: str, true_plants: dict[str, PlantUnderTest]
) -> None:
    plant = true_plants[name]
    trajectory = simulate_piecewise(plant.f, plant.nominal_state, two_stage_sequence(), 0.1)
    check = integrated_balances(trajectory, plant.parameters)
    assert check.reacted > 0.0
    assert check.reaction_heating > 0.0  # exothermic
    assert 0.0 <= check.relative_mass_residual < 1.0e-8
    assert 0.0 <= check.relative_energy_residual < 1.0e-8
    assert check.closes(BALANCE_TOLERANCE)


def test_the_check_detects_inputs_attached_to_the_wrong_segment(
    true_plants: dict[str, PlantUnderTest],
) -> None:
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
    assert not check.closes(BALANCE_TOLERANCE)


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
    assert not coarse.closes(BALANCE_TOLERANCE)


# --------------------------------------------------------------------------- #
# Endothermic and zero enthalpy (both admitted by the schema, D-015)
# --------------------------------------------------------------------------- #


def test_an_endothermic_trajectory_with_a_corrupted_sample_is_rejected(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """The reviewer's reproduction. With dH = +50 kJ/mol the old scale was -43.77 K and
    a last sample moved by 1 K gave a relative residual of -0.0228, which passed."""
    source = true_plants["source"]
    endothermic = dataclasses.replace(source.parameters, reaction_enthalpy=+5.0e4)
    trajectory = at_rest(endothermic, source.nominal_inputs, period=0.001)

    clean = integrated_balances(trajectory, endothermic)
    assert clean.reaction_heating == pytest.approx(-43.7665, abs=1e-3)  # K, signed
    assert clean.energy_scale > abs(clean.reaction_heating)  # flow and cooling count too
    assert clean.closes(BALANCE_TOLERANCE)
    assert check_trajectory(trajectory, endothermic, WIDE_ENVELOPE).accepted

    corrupted = with_last_sample_shifted(trajectory, state=1, delta=1.0)
    check = integrated_balances(corrupted, endothermic)
    # 1 K that no term explains. Not exactly 1: the altered sample also enters the
    # integrand, and shifts the integral by a few 1e-5 K.
    assert check.energy_residual == pytest.approx(1.0, abs=1e-3)
    assert check.relative_energy_residual > 1.0e-3  # positive, and far above the tolerance
    assert not check.closes(BALANCE_TOLERANCE)
    verdict = check_trajectory(corrupted, endothermic, WIDE_ENVELOPE)
    assert not verdict.balances_close and not verdict.accepted


def test_zero_enthalpy_neither_divides_by_zero_nor_hides_a_corrupted_sample(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    source = true_plants["source"]
    athermal = dataclasses.replace(source.parameters, reaction_enthalpy=0.0)
    trajectory = at_rest(athermal, source.nominal_inputs)

    clean = integrated_balances(trajectory, athermal)
    assert clean.reaction_heating == 0.0
    assert clean.energy_scale > 0.0  # the feed and the coolant still exchange heat
    assert clean.closes(BALANCE_TOLERANCE)

    corrupted = with_last_sample_shifted(trajectory, state=1, delta=1.0)
    assert not integrated_balances(corrupted, athermal).closes(BALANCE_TOLERANCE)


# --------------------------------------------------------------------------- #
# Limits of the numerical functions, outside what the schema admits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k0", [0.0, 1.0e-30])
def test_no_reaction_or_a_negligible_one_is_judged_on_the_flow_terms(
    k0: float, true_plants: dict[str, PlantUnderTest]
) -> None:
    source = true_plants["source"]
    inert = dataclasses.replace(source.parameters, k0=k0)
    start = np.array([300.0, 345.0])  # away from the inert steady state, so A accumulates
    trajectory = simulate_piecewise(
        lambda x, u: cstr_true.rhs(0.0, x, u, inert),
        start,
        [InputSegment(120.0, source.nominal_inputs)],
        0.01,
    )
    clean = integrated_balances(trajectory, inert)
    assert clean.reacted == pytest.approx(0.0, abs=1e-20)
    assert clean.mass_scale > 100.0  # mol/m^3 brought in by the feed alone
    assert clean.closes(BALANCE_TOLERANCE)

    corrupted = with_last_sample_shifted(trajectory, state=0, delta=1.0)
    check = integrated_balances(corrupted, inert)
    assert check.relative_mass_residual > 1.0e-3
    assert not check.closes(BALANCE_TOLERANCE)


def test_with_no_traffic_at_all_nothing_may_accumulate(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """Zero enthalpy with feed, coolant and reactor at one temperature: every term of
    the energy balance is exactly zero. The scale is zero, which is a valid physical
    limit and not an error; the only acceptable accumulation is none."""
    source = true_plants["source"]
    athermal = dataclasses.replace(source.parameters, reaction_enthalpy=0.0)
    inputs = source.nominal_inputs.copy()
    inputs[2] = inputs[3] = 350.0
    trajectory = simulate_piecewise(
        lambda x, u: cstr_true.rhs(0.0, x, u, athermal),
        np.array([250.0, 350.0]),
        [InputSegment(120.0, inputs)],
        0.01,
    )
    clean = integrated_balances(trajectory, athermal)
    assert clean.energy_scale == 0.0 and clean.energy_residual == 0.0
    assert clean.relative_energy_residual == 0.0
    assert clean.closes(BALANCE_TOLERANCE)

    # A last sample moved by a microkelvin: the altered sample gives the terms a traffic
    # of order 1e-10 K, against which a residual of 1e-6 K is enormous.
    corrupted = integrated_balances(
        with_last_sample_shifted(trajectory, state=1, delta=1.0e-6), athermal
    )
    assert corrupted.energy_scale < 1.0e-8
    assert corrupted.relative_energy_residual > 1.0e3
    assert not corrupted.closes(BALANCE_TOLERANCE)


def test_zero_traffic_is_resolved_explicitly_without_dividing() -> None:
    fields = dict(
        mass_residual=0.0,
        mass_scale=1.0,
        mass_resolution=0.0,
        energy_scale=0.0,
        energy_resolution=5.0e-14,
        reacted=1.0,
        reaction_heating=0.0,
    )
    nothing = BalanceCheck(energy_residual=0.0, **fields)
    assert nothing.relative_energy_residual == 0.0 and nothing.closes(BALANCE_TOLERANCE)
    rounding = BalanceCheck(energy_residual=4.0e-14, **fields)
    assert rounding.relative_energy_residual == 0.0 and rounding.closes(BALANCE_TOLERANCE)
    unexplained = BalanceCheck(energy_residual=1.0e-6, **fields)
    assert unexplained.relative_energy_residual == math.inf
    assert not unexplained.closes(BALANCE_TOLERANCE)


def test_a_heat_effect_below_the_resolution_of_the_states_is_not_a_false_alarm(
    true_plants: dict[str, PlantUnderTest],
) -> None:
    """With dH = 1e-30 J/mol the traffic is of order 1e-34 K, far below what a stored
    temperature can resolve. The residual is pure rounding; it is compared with the
    resolution of the accumulation, not with a made-up epsilon."""
    source = true_plants["source"]
    faint = dataclasses.replace(source.parameters, reaction_enthalpy=-1.0e-30)
    inputs = source.nominal_inputs.copy()
    inputs[2] = inputs[3] = 350.0
    trajectory = simulate_piecewise(
        lambda x, u: cstr_true.rhs(0.0, x, u, faint),
        np.array([250.0, 350.0]),
        [InputSegment(120.0, inputs)],
        0.01,
    )
    check = integrated_balances(trajectory, faint)
    assert 0.0 < check.energy_scale < 1.0e-25
    assert check.energy_residual <= check.energy_resolution
    assert check.closes(BALANCE_TOLERANCE)


# --------------------------------------------------------------------------- #
# Non-finite values and the sign of the reported ratios
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(
    bad: float, true_plants: dict[str, PlantUnderTest]
) -> None:
    plant = true_plants["source"]
    trajectory = simulate_piecewise(
        plant.f, plant.nominal_state, [InputSegment(60.0, plant.nominal_inputs)]
    )
    corrupted = with_last_sample_shifted(trajectory, state=1, delta=bad)
    assert not integrated_balances(corrupted, plant.parameters).closes(BALANCE_TOLERANCE)

    verdict = check_trajectory(corrupted, plant.parameters)
    assert not verdict.values_finite
    assert not verdict.accepted
    assert math.isnan(verdict.peak_temperature)  # reported as unknown, not invented


def test_a_balance_with_non_finite_numbers_never_closes() -> None:
    fine = dict(
        mass_residual=0.0,
        mass_scale=1.0,
        mass_resolution=0.0,
        energy_residual=0.0,
        energy_scale=1.0,
        energy_resolution=0.0,
        reacted=1.0,
        reaction_heating=1.0,
    )
    assert BalanceCheck(**fine).closes(BALANCE_TOLERANCE)
    for field in ("mass_residual", "mass_scale", "energy_residual", "energy_scale"):
        for bad in (math.nan, math.inf):
            assert not BalanceCheck(**{**fine, field: bad}).closes(BALANCE_TOLERANCE)


@pytest.mark.parametrize("enthalpy", [-5.0e4, 0.0, +5.0e4])
def test_reported_ratios_are_never_negative(
    enthalpy: float, true_plants: dict[str, PlantUnderTest]
) -> None:
    source = true_plants["source"]
    p = dataclasses.replace(source.parameters, reaction_enthalpy=enthalpy)
    check = integrated_balances(at_rest(p, source.nominal_inputs), p)
    assert check.relative_mass_residual >= 0.0
    assert check.relative_energy_residual >= 0.0
    assert check.mass_scale > 0.0 and check.energy_scale > 0.0
