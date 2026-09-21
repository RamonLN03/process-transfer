"""Integrated mass and energy balances of a simulated true-plant trajectory.

Over any interval the balances must close:

    C_A(end) - C_A(start) = integral of [ (q/V) (C_Af - C_A) - r ] dt
    T(end)   - T(start)   = integral of [ (q/V) (T_f - T) + beta r
                                          - UA(T) (T - T_c) / (V rho cp) ] dt

The integrals are evaluated here by Simpson quadrature of the physical terms on
the stored samples, segment by segment because the inputs are discontinuous. This
is independent of the integrator's own stepping: it checks that the trajectory as
stored, with the inputs as recorded, satisfies the balances. A wrong input
attached to a segment, a sample misaligned with a switching instant or a coarse
sampling period all show up as a residual.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.integrate import simpson

from process_transfer.simulation.cstr_true import TrueCSTRParameters, conductance, reaction_rate
from process_transfer.simulation.integration import Trajectory


@dataclass(frozen=True)
class BalanceCheck:
    """Accumulated closure errors and the natural scales to compare them with."""

    mass_residual: float  # mol/m^3, sum over segments of |accumulation - integral|
    reacted: float  # mol/m^3, integral of r dt: amount of A reacted per unit volume
    energy_residual: float  # K, sum over segments of |accumulation - integral|
    adiabatic_heating: float  # K, integral of beta r dt: heating by reaction

    @property
    def relative_mass_residual(self) -> float:
        return self.mass_residual / self.reacted

    @property
    def relative_energy_residual(self) -> float:
        return self.energy_residual / self.adiabatic_heating


def integrated_balances(trajectory: Trajectory, p: TrueCSTRParameters) -> BalanceCheck:
    """Closure of the integrated mass and energy balances along ``trajectory``."""
    beta = -p.reaction_enthalpy / (p.density * p.heat_capacity)  # K m^3 / mol
    thermal_mass = p.volume * p.density * p.heat_capacity  # J/K

    mass_residual = energy_residual = reacted = heating = 0.0
    for segment in trajectory.segments:
        q, c_af, t_f, t_c = segment.inputs
        c_a, temperature = segment.states[:, 0], segment.states[:, 1]
        dilution = q / p.volume

        rate = reaction_rate(c_a, temperature, p)
        ua = conductance(temperature, p)

        mass_terms = dilution * (c_af - c_a) - rate
        energy_terms = (
            dilution * (t_f - temperature) + beta * rate - ua / thermal_mass * (temperature - t_c)
        )

        mass_residual += abs((c_a[-1] - c_a[0]) - simpson(mass_terms, x=segment.times))
        energy_residual += abs(
            (temperature[-1] - temperature[0]) - simpson(energy_terms, x=segment.times)
        )
        reacted += float(simpson(rate, x=segment.times))
        heating += float(simpson(beta * rate, x=segment.times))

    return BalanceCheck(
        mass_residual=float(mass_residual),
        reacted=reacted,
        energy_residual=float(energy_residual),
        adiabatic_heating=heating,
    )
