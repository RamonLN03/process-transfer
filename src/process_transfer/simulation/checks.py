"""Acceptance checks for a simulated true-plant trajectory.

A trajectory is accepted when the integrator succeeded (``simulate_piecewise``
raises otherwise), every time, state and input is finite, the states are physical,
the temperature stays inside the documented envelope and the integrated balances
close. Nothing is clipped or repaired: a violation is reported as a violation.

These checks look at one segment at a time. That the segments form one continuous
trajectory, without a gap, an overlap or a jump of the state where the inputs change,
is a property of ``Trajectory`` itself, enforced when it is built and protected by
read-only arrays afterwards (``simulation/integration.py``). Pieces simulated apart
and put side by side are therefore rejected before they can be checked here.

Physical rules, for a reactor that only consumes A:

* the four inputs, feed flow and concentration, feed and coolant temperature, are
  positive;
* the conductance law stays in its valid domain: UA(T) is never negative (zero is
  the adiabatic limit);
* C_A stays positive;
* C_A never exceeds the richest feed applied so far (or its own initial value);
* for an exothermic reaction, T never falls below the coldest of the feed and the
  coolant applied so far (or its own initial value), because the reaction can only
  add heat. The rule is skipped when the reaction enthalpy is zero or positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from process_transfer.simulation.balances import integrated_balances
from process_transfer.simulation.cstr_true import TrueCSTRParameters, conductance
from process_transfer.simulation.integration import Trajectory

TEMPERATURE_ENVELOPE = (335.0, 380.0)  # K, docs/assumptions.md
# Fraction of the traffic through a balance that may be unaccounted for; see balances.py.
BALANCE_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class TrajectoryCheck:
    """Extremes of a trajectory and the verdict of each acceptance criterion."""

    peak_temperature: float  # K, largest sample
    refined_peak_temperature: float  # K, parabolic estimate of the maximum, not a bound
    peak_time: float  # s
    min_temperature: float  # K
    c_a_range: tuple[float, float]  # mol/m^3
    seconds_above_limit: float  # s spent above the upper temperature limit
    relative_mass_residual: float
    relative_energy_residual: float
    values_finite: bool
    states_physical: bool
    inside_envelope: bool
    balances_close: bool

    @property
    def accepted(self) -> bool:
        return (
            self.values_finite
            and self.states_physical
            and self.inside_envelope
            and self.balances_close
        )

    @property
    def physically_valid(self) -> bool:
        """Whether every check independent of the temperature envelope holds.

        Distinguishes a trajectory that leaves the documented operating envelope,
        a diagnostic result worth reporting on its own, from one that is invalid
        on finite values, physical states or the integrated mass and energy
        balances. ``accepted`` is the stricter criterion that also requires the
        envelope; this property is for a variant whose purpose is precisely to
        probe how far a trajectory moves, where leaving the envelope must stay a
        reported outcome rather than a rejection.
        """
        return self.values_finite and self.states_physical and self.balances_close


def _values_are_finite(trajectory: Trajectory) -> bool:
    return all(
        np.all(np.isfinite(segment.times))
        and np.all(np.isfinite(segment.states))
        and np.all(np.isfinite(segment.inputs))
        for segment in trajectory.segments
    )


def _states_are_physical(trajectory: Trajectory, p: TrueCSTRParameters) -> bool:
    exothermic = p.reaction_enthalpy < 0.0
    first = trajectory.segments[0].states[0]
    richest_feed, coldest_stream = float(first[0]), float(first[1])
    for segment in trajectory.segments:
        if np.any(segment.inputs <= 0.0):
            return False
        _, c_af, t_f, t_c = segment.inputs
        richest_feed = max(richest_feed, float(c_af))
        coldest_stream = min(coldest_stream, float(t_f), float(t_c))
        c_a, temperature = segment.states[:, 0], segment.states[:, 1]
        if np.any(c_a <= 0.0) or np.any(c_a > richest_feed * (1.0 + 1.0e-9)):
            return False
        if exothermic and np.any(temperature < coldest_stream * (1.0 - 1.0e-9)):
            return False
        if np.any(conductance(temperature, p) < 0.0):
            return False
    return True


def check_trajectory(
    trajectory: Trajectory,
    p: TrueCSTRParameters,
    envelope: tuple[float, float] = TEMPERATURE_ENVELOPE,
    balance_tolerance: float = BALANCE_TOLERANCE,
) -> TrajectoryCheck:
    """Evaluate every acceptance criterion on ``trajectory``."""
    if not _values_are_finite(trajectory):
        # Nothing can be measured on a trajectory with non-finite values; every verdict
        # is negative and the extremes are reported as not-a-number rather than invented.
        return TrajectoryCheck(
            peak_temperature=math.nan,
            refined_peak_temperature=math.nan,
            peak_time=math.nan,
            min_temperature=math.nan,
            c_a_range=(math.nan, math.nan),
            seconds_above_limit=math.nan,
            relative_mass_residual=math.nan,
            relative_energy_residual=math.nan,
            values_finite=False,
            states_physical=False,
            inside_envelope=False,
            balances_close=False,
        )

    times, temperature = trajectory.times, trajectory.states[:, 1]
    peak, peak_time = trajectory.peak(1)
    refined_peak, _ = trajectory.refined_peak(1)
    above = (temperature > envelope[1]).astype(np.float64)
    seconds_above = float(np.sum(0.5 * (above[1:] + above[:-1]) * np.diff(times)))

    balances = integrated_balances(trajectory, p)
    # The upper limit is judged on the refined maximum, which is never below the largest
    # sample. It is an estimate, not a bound: it reduces the error made by sampling but a
    # peak much narrower than the sampling period can still go unseen. Critical cases are
    # therefore also recomputed with finer sampling and a second integrator.
    inside = temperature.min() >= envelope[0] and max(peak, refined_peak) <= envelope[1]
    return TrajectoryCheck(
        peak_temperature=peak,
        refined_peak_temperature=refined_peak,
        peak_time=peak_time,
        min_temperature=float(temperature.min()),
        c_a_range=trajectory.state_range(0),
        seconds_above_limit=seconds_above,
        relative_mass_residual=balances.relative_mass_residual,
        relative_energy_residual=balances.relative_energy_residual,
        values_finite=True,
        states_physical=_states_are_physical(trajectory, p),
        inside_envelope=bool(inside),
        balances_close=balances.closes(balance_tolerance),
    )


def comparison_is_valid(primary: TrajectoryCheck, secondary: TrajectoryCheck) -> bool:
    """Whether a primary/secondary trajectory pair may be reported as valid.

    The primary trajectory must be fully accepted. The secondary trajectory must
    be ``physically_valid``, but is exempt from the envelope: for a diagnostic
    variant built to probe how far a change in the physics moves the plant,
    leaving the documented operating envelope is an admissible result to report,
    never a reason by itself to reject the comparison. A secondary trajectory
    that is not finite, not physical, or whose balances do not close is invalid
    regardless of the envelope, and its differences from the primary must not be
    presented as a scientifically acceptable result.
    """
    return primary.accepted and secondary.physically_valid
