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

import math
from dataclasses import dataclass

import numpy as np

from process_transfer.config import TruePlantConfig
from process_transfer.cstr_variables import FloatArray


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


def reaction_rate(c_a: float, temperature: float, p: TrueCSTRParameters) -> float:
    """True rate of A -> B in mol/(m^3 s): Arrhenius with saturation in C_A."""
    k = p.k0 * math.exp(-p.activation_temperature / temperature)
    return k * c_a / (1.0 + p.saturation_constant * c_a)


def conductance(temperature: float, p: TrueCSTRParameters) -> float:
    """True heat-transfer conductance UA(T) in W/K."""
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
