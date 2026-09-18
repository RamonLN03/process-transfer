"""The modeller's CSTR: first-order Arrhenius kinetics and a constant conductance.

This is the simplified model an engineer would write from available knowledge
(docs/assumptions.md). It is built from the known plant specification and the
modeller's own parameter values only; it cannot see the true plant's hidden
physics. All quantities are plain SI floats.

Balances, with state x = [C_A, T] and inputs u = [q, C_Af, T_f, T_c]:

    dC_A/dt = (q/V) (C_Af - C_A) - k0 exp(-(E/R) / T) C_A
    dT/dt   = (q/V) (T_f - T) + (-dH / (rho cp)) k0 exp(-(E/R) / T) C_A
              - UA / (V rho cp) (T - T_c)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from process_transfer.config import ModellerConfig, PlantSpec
from process_transfer.cstr_variables import FloatArray


@dataclass(frozen=True)
class ModellerCSTRParameters:
    """Parameters of the modeller's model in SI."""

    # known from the plant specification
    volume: float  # V, m^3
    density: float  # rho, kg/m^3
    heat_capacity: float  # cp, J/(kg K)
    reaction_enthalpy: float  # dH, J/mol
    # the modeller's own values, to be re-estimated from data in later milestones
    k0: float  # pre-exponential factor, 1/s
    activation_temperature: float  # E/R, K
    ua: float  # constant heat-transfer conductance, W/K

    @classmethod
    def from_config(cls, plant: PlantSpec, modeller: ModellerConfig) -> ModellerCSTRParameters:
        return cls(
            volume=plant.design.volume.si,
            density=plant.properties.density.si,
            heat_capacity=plant.properties.heat_capacity.si,
            reaction_enthalpy=plant.properties.reaction_enthalpy.si,
            k0=modeller.kinetics.k0.si,
            activation_temperature=modeller.kinetics.activation_temperature.si,
            ua=modeller.heat_transfer.UA.si,
        )


def reaction_rate(c_a: float, temperature: float, p: ModellerCSTRParameters) -> float:
    """First-order Arrhenius rate of A -> B in mol/(m^3 s)."""
    return p.k0 * math.exp(-p.activation_temperature / temperature) * c_a


def rhs(t: float, x: FloatArray, u: FloatArray, p: ModellerCSTRParameters) -> FloatArray:
    """Right-hand side dx/dt of the modeller's model. ``t`` is unused."""
    c_a, temperature = x
    q, c_af, t_f, t_c = u

    dilution = q / p.volume  # 1/s
    rate = reaction_rate(c_a, temperature, p)  # mol/(m^3 s)
    thermal_mass = p.volume * p.density * p.heat_capacity  # J/K

    dc_a_dt = dilution * (c_af - c_a) - rate

    heating_by_reaction = (-p.reaction_enthalpy) / (p.density * p.heat_capacity) * rate  # K/s
    cooling = p.ua / thermal_mass * (temperature - t_c)  # K/s
    dt_dt = dilution * (t_f - temperature) + heating_by_reaction - cooling

    return np.array([dc_a_dt, dt_dt], dtype=np.float64)
