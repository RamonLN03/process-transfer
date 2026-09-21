"""The true CSTR: simulation truth, including the hidden physics.

The saturating rate law and the temperature-dependent conductance below are
unknown to the modeller (docs/decisions.md D-004, D-006, D-007). All quantities
are plain SI floats; unit conversion happens once, in ``from_config``.

Balances (docs/assumptions.md), with state x = [C_A, T] and inputs
u = [q, C_Af, T_f, T_c]:

    dC_A/dt = (q/V) (C_Af - C_A) - r(C_A, T)
    dT/dt   = (q/V) (T_f - T) + (-dH / (rho cp)) r(C_A, T)
              - UA(T) / (V rho cp) (T - T_c)

    r(C_A, T) = k0 exp(-(E/R) / T) C_A / (1 + K_sat C_A)     saturating kinetics
    UA(T)     = UA_ref [1 + alpha (T - T_ref)]                 conductance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from process_transfer.config import TruePlantConfig
from process_transfer.cstr_variables import FloatArray
from process_transfer.validation import require_finite, require_non_negative, require_positive


@dataclass(frozen=True)
class TrueCSTRParameters:
    """Parameters of the true plant in SI. Hidden from the modeller."""

    volume: float  # V, m^3
    density: float  # rho, kg/m^3
    heat_capacity: float  # cp, J/(kg K)
    reaction_enthalpy: float  # dH, J/mol (negative: exothermic)
    k0: float  # pre-exponential factor, 1/s
    activation_temperature: float  # E/R, K
    saturation_constant: float  # K_sat, m^3/mol
    ua_ref: float  # UA at the reference temperature, W/K
    alpha: float  # relative slope of UA with temperature, 1/K
    t_ref: float  # reference temperature of UA(T), K

    def __post_init__(self) -> None:
        """Validate once, here, so that the right-hand side never has to.

        Zero is accepted for k0, E/R, K_sat and UA_ref: no reaction, no temperature
        dependence, no saturation and an adiabatic reactor are valid physical limits,
        used to test the numerical functions even though the configuration schema asks
        for positive values. Volume, density and heat capacity divide, and must be
        positive. The products formed from them are checked too, because finite
        factors do not guarantee a finite, non-zero product.
        """
        for name in ("volume", "density", "heat_capacity"):
            require_positive(name, getattr(self, name))
        for name in ("k0", "activation_temperature", "saturation_constant", "ua_ref"):
            require_non_negative(name, getattr(self, name))
        for name in ("reaction_enthalpy", "alpha", "t_ref"):
            require_finite(name, getattr(self, name))
        require_positive("volume * density * heat_capacity", self.thermal_mass)
        require_finite(
            "reaction_enthalpy / (density * heat_capacity)", self.adiabatic_coefficient
        )

    @property
    def thermal_mass(self) -> float:
        """V rho cp, J/K."""
        return self.volume * self.density * self.heat_capacity

    @property
    def adiabatic_coefficient(self) -> float:
        """beta = -dH / (rho cp), K m^3/mol: positive for an exothermic reaction."""
        return -self.reaction_enthalpy / (self.density * self.heat_capacity)

    @classmethod
    def from_config(cls, cfg: TruePlantConfig) -> TrueCSTRParameters:
        plant, physics = cfg.plant, cfg.true_physics
        return cls(
            volume=plant.design.volume.si,
            density=plant.properties.density.si,
            heat_capacity=plant.properties.heat_capacity.si,
            reaction_enthalpy=plant.properties.reaction_enthalpy.si,
            k0=physics.kinetics.k0.si,
            activation_temperature=physics.kinetics.activation_temperature.si,
            saturation_constant=physics.kinetics.saturation_constant.si,
            ua_ref=physics.heat_transfer.UA_ref.si,
            alpha=physics.heat_transfer.alpha.si,
            t_ref=physics.heat_transfer.T_ref.si,
        )


def reaction_rate(
    c_a: float | FloatArray, temperature: float | FloatArray, p: TrueCSTRParameters
) -> float | FloatArray:
    """True rate of A -> B in mol/(m^3 s): Arrhenius with saturation in C_A.

    Accepts scalars or arrays of samples, so that the same law serves the
    right-hand side and the integrated balances.

    Valid for T > 0 and 1 + K_sat C_A > 0, which every physical state satisfies.
    Outside that domain the result is infinite or undefined; the law is not guarded
    here, because it runs inside the integrator. A trajectory that leaves the domain
    ends up non-finite or non-physical and is rejected by ``simulate_piecewise`` and
    ``simulation/checks.py``."""
    k = p.k0 * np.exp(-p.activation_temperature / temperature)
    return k * c_a / (1.0 + p.saturation_constant * c_a)


def conductance(temperature: float | FloatArray, p: TrueCSTRParameters) -> float | FloatArray:
    """True heat-transfer conductance UA(T) in W/K, for scalars or arrays.

    The linear law is physically meaningful only where it is positive, that is for
    1 + alpha (T - T_ref) > 0 (T above 150 K on the source). It is not clipped here;
    ``simulation/checks.py`` rejects a trajectory that leaves that domain."""
    return p.ua_ref * (1.0 + p.alpha * (temperature - p.t_ref))


def rhs(t: float, x: FloatArray, u: FloatArray, p: TrueCSTRParameters) -> FloatArray:
    """Right-hand side dx/dt of the true plant. ``t`` is unused (autonomous system)."""
    c_a, temperature = x
    q, c_af, t_f, t_c = u

    dilution = q / p.volume  # 1/s
    rate = reaction_rate(c_a, temperature, p)  # mol/(m^3 s)
    thermal_mass = p.volume * p.density * p.heat_capacity  # J/K

    dc_a_dt = dilution * (c_af - c_a) - rate

    heating_by_reaction = (-p.reaction_enthalpy) / (p.density * p.heat_capacity) * rate  # K/s
    cooling = conductance(temperature, p) / thermal_mass * (temperature - t_c)  # K/s
    dt_dt = dilution * (t_f - temperature) + heating_by_reaction - cooling

    return np.array([dc_a_dt, dt_dt], dtype=np.float64)
