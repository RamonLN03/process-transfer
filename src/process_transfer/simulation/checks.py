"""Acceptance checks for a simulated true-plant trajectory.

A trajectory is accepted when the integrator succeeded (``simulate_piecewise``
raises otherwise), the states are physical, the temperature stays inside the
documented envelope and the integrated balances close. Nothing is clipped or
repaired: a violation is reported as a violation.

Physical-state rules, for a reactor that only consumes A:

* C_A stays positive;
* C_A never exceeds the richest feed applied so far (or its own initial value);
* for an exothermic reaction, T never falls below the coldest of the feed and the
  coolant applied so far (or its own initial value), because the reaction can only
  add heat. The rule is skipped for an endothermic reaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from process_transfer.simulation.balances import integrated_balances
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.integration import Trajectory

TEMPERATURE_ENVELOPE = (335.0, 380.0)  # K, docs/assumptions.md
BALANCE_TOLERANCE = 1.0e-6  # relative, against the amount reacted / the adiabatic heating


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
    states_physical: bool
    inside_envelope: bool
    balances_close: bool

    @property
    def accepted(self) -> bool:
        return self.states_physical and self.inside_envelope and self.balances_close


def _states_are_physical(trajectory: Trajectory, p: TrueCSTRParameters) -> bool:
    exothermic = p.reaction_enthalpy < 0.0
    first = trajectory.segments[0].states[0]
    richest_feed, coldest_stream = float(first[0]), float(first[1])
    for segment in trajectory.segments:
        _, c_af, t_f, t_c = segment.inputs
        richest_feed = max(richest_feed, float(c_af))
        coldest_stream = min(coldest_stream, float(t_f), float(t_c))
        c_a, temperature = segment.states[:, 0], segment.states[:, 1]
        if np.any(c_a <= 0.0) or np.any(c_a > richest_feed * (1.0 + 1.0e-9)):
            return False
        if exothermic and np.any(temperature < coldest_stream * (1.0 - 1.0e-9)):
            return False
    return True


def check_trajectory(
    trajectory: Trajectory,
    p: TrueCSTRParameters,
    envelope: tuple[float, float] = TEMPERATURE_ENVELOPE,
    balance_tolerance: float = BALANCE_TOLERANCE,
) -> TrajectoryCheck:
    """Evaluate every acceptance criterion on ``trajectory``."""
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
        states_physical=_states_are_physical(trajectory, p),
        inside_envelope=bool(inside),
        balances_close=(
            balances.relative_mass_residual < balance_tolerance
            and balances.relative_energy_residual < balance_tolerance
        ),
    )
