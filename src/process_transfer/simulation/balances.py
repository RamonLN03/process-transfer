"""Integrated mass and energy balances of a simulated true-plant trajectory.

Over any interval the balances must close:

    C_A(end) - C_A(start) = integral of [ (q/V) (C_Af - C_A) - r ] dt
    T(end)   - T(start)   = integral of [ (q/V) (T_f - T) + beta r
                                          - UA(T) (T - T_c) / (V rho cp) ] dt

with beta = -dH / (rho cp). The integrals are evaluated by Simpson quadrature of
the physical terms on the stored samples, segment by segment because the inputs
are discontinuous. This is independent of the integrator's own stepping: it checks
that the trajectory as stored, with the inputs as recorded, satisfies the balances.
A wrong input attached to a segment, a sample misaligned with a switching instant
or a coarse sampling period all show up as a residual.

Units. Both balances are written per unit of reactor content. The mass residual is
a concentration, mol/m^3. The energy residual is a temperature, K: energy divided by
the thermal mass V rho cp.

Normalisation. A residual is compared with the *traffic* through its balance, the
integral of the magnitude of every physical term:

    mass scale   = integral of [ |(q/V) (C_Af - C_A)| + |r| ] dt                  mol/m^3
    energy scale = integral of [ |(q/V) (T_f - T)| + |beta r|
                                 + |UA(T) (T - T_c) / (V rho cp)| ] dt             K

so that residual / scale is the fraction of what went through the balance that is
unaccounted for. The scale is never negative, whatever the sign of the reaction
enthalpy, and it is zero only when nothing flowed, reacted or exchanged heat, in
which case nothing may have accumulated either.

Alternatives considered (docs/numerical_robustness.md):

* the magnitude of the reaction effect alone. Simple, but it vanishes with the
  enthalpy or with the rate while the flow and heat-exchange terms can still carry
  a residual, so the ratio blows up exactly where a trajectory may be perfectly good;
* the traffic through the balance, used here;
* an absolute tolerance added to a relative one. A fixed absolute tolerance does not
  suit an accumulated quantity, which grows with the length of the trajectory: it
  would be too loose for short runs or too tight for long ones. The only absolute
  term kept is one that is not a choice: the floating-point resolution of the
  accumulation x(end) - x(start), one unit in the last place of the states per
  segment. It matters only when the traffic is itself at rounding level.

Acceptance is ``residual <= tolerance * scale + resolution``, written as a product
so that no division is involved, and it requires every quantity to be finite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from process_transfer.simulation.cstr_true import TrueCSTRParameters, conductance, reaction_rate
from process_transfer.simulation.integration import Trajectory


def _relative(residual: float, scale: float, resolution: float) -> float:
    """``residual / scale`` for reporting. With no traffic at all the ratio is 0 when
    the residual is within the resolution of the accumulation and infinite otherwise."""
    if not (math.isfinite(residual) and math.isfinite(scale)):
        return math.nan
    if scale > 0.0:
        return residual / scale
    return 0.0 if residual <= resolution else math.inf


@dataclass(frozen=True)
class BalanceCheck:
    """Closure errors of both balances with the scales they are compared with."""

    mass_residual: float  # mol/m^3, sum over segments of |accumulation - integral|
    mass_scale: float  # mol/m^3, integral of |feed term| + |reaction term|
    mass_resolution: float  # mol/m^3, floating-point resolution of the accumulation
    energy_residual: float  # K, sum over segments of |accumulation - integral|
    energy_scale: float  # K, integral of |feed| + |reaction heat| + |heat exchange|
    energy_resolution: float  # K, floating-point resolution of the accumulation
    reacted: float  # mol/m^3, integral of r dt: amount of A reacted per unit volume
    reaction_heating: float  # K, integral of beta r dt: signed, negative if endothermic

    @property
    def relative_mass_residual(self) -> float:
        return _relative(self.mass_residual, self.mass_scale, self.mass_resolution)

    @property
    def relative_energy_residual(self) -> float:
        return _relative(self.energy_residual, self.energy_scale, self.energy_resolution)

    def closes(self, tolerance: float) -> bool:
        """True when both residuals are finite and within ``tolerance`` of the traffic
        through their balance, up to the resolution of the accumulation."""
        values = (
            self.mass_residual,
            self.mass_scale,
            self.mass_resolution,
            self.energy_residual,
            self.energy_scale,
            self.energy_resolution,
        )
        if not all(math.isfinite(value) for value in values):
            return False
        return (
            self.mass_residual <= tolerance * self.mass_scale + self.mass_resolution
            and self.energy_residual <= tolerance * self.energy_scale + self.energy_resolution
        )


def integrated_balances(trajectory: Trajectory, p: TrueCSTRParameters) -> BalanceCheck:
    """Closure of the integrated mass and energy balances along ``trajectory``."""
    if not all(
        np.all(np.isfinite(segment.times))
        and np.all(np.isfinite(segment.states))
        and np.all(np.isfinite(segment.inputs))
        for segment in trajectory.segments
    ):
        # Nothing can be integrated; report that, instead of arithmetic on infinities.
        return BalanceCheck(*([math.nan] * 8))

    beta = -p.reaction_enthalpy / (p.density * p.heat_capacity)  # K m^3 / mol, any sign
    thermal_mass = p.volume * p.density * p.heat_capacity  # J/K

    mass_residual = mass_scale = mass_resolution = 0.0
    energy_residual = energy_scale = energy_resolution = 0.0
    reacted = reaction_heating = 0.0
    for segment in trajectory.segments:
        q, c_af, t_f, t_c = segment.inputs
        times = segment.times
        c_a, temperature = segment.states[:, 0], segment.states[:, 1]
        dilution = q / p.volume

        rate = reaction_rate(c_a, temperature, p)
        feed_of_a = dilution * (c_af - c_a)
        feed_heat = dilution * (t_f - temperature)
        reaction_heat = beta * rate
        exchanged_heat = conductance(temperature, p) / thermal_mass * (temperature - t_c)

        mass_integral = simpson(feed_of_a - rate, x=times)
        energy_integral = simpson(feed_heat + reaction_heat - exchanged_heat, x=times)

        mass_residual += abs((c_a[-1] - c_a[0]) - mass_integral)
        energy_residual += abs((temperature[-1] - temperature[0]) - energy_integral)
        mass_scale += float(simpson(np.abs(feed_of_a) + np.abs(rate), x=times))
        energy_scale += float(
            simpson(np.abs(feed_heat) + np.abs(reaction_heat) + np.abs(exchanged_heat), x=times)
        )
        mass_resolution += float(np.spacing(max(abs(c_a[0]), abs(c_a[-1]))))
        energy_resolution += float(np.spacing(max(abs(temperature[0]), abs(temperature[-1]))))
        reacted += float(simpson(rate, x=times))
        reaction_heating += float(simpson(reaction_heat, x=times))

    return BalanceCheck(
        mass_residual=float(mass_residual),
        mass_scale=mass_scale,
        mass_resolution=mass_resolution,
        energy_residual=float(energy_residual),
        energy_scale=energy_scale,
        energy_resolution=energy_resolution,
        reacted=reacted,
        reaction_heating=reaction_heating,
    )
