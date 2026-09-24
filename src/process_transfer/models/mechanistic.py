"""The modeller's model with the textbook values, MN, and with estimated values, MR
(``docs/m1_plan.md``, sections 7.1 and 8.1).

The equations are those of ``modeller/cstr_first_order.py``, which this module calls and
does not copy: first-order Arrhenius kinetics and a constant conductance in the balances of
a CSTR. What distinguishes MN from MR is three values:

    k(T) = k_350 exp(-(E/R) (1/T - 1/T_ref)),   T_ref = 350 K

the rate constant at the reference temperature, the activation temperature E/R and the
conductance UA. 350 K is the known nominal feed temperature of both plants. MN holds the
textbook values of ``configs/modeller_cstr.yaml``; MR holds values estimated on the target.
The known parameters V, rho, cp and dH are those of the export. The modeller's code takes
k0 = k_350 exp((E/R) / T_ref); MN is given the textbook k0 itself, so that its right-hand
side is the modeller's, bit for bit.

Coordinates of the estimation. The optimiser works on

    theta = (ln k_350, (E/R) / T_ref, ln UA)

ln k_350 and ln UA keep the rate constant and the conductance positive and put them on the
scale of relative changes; (E/R) / T_ref is the activation energy made dimensionless with
the reference temperature, the Arrhenius number, about 25, so that the three coordinates
are of comparable size and the tolerances of the optimiser act on them alike. It is a
change of units of E/R and nothing else: an estimate or a standard error in it is one in
E/R times T_ref. From theta, k0 = exp(theta_1 + theta_2). With E/R held fixed, the secondary
analysis of D-030, theta is (ln k_350, ln UA).

Domain. The rate law divides by the temperature, so an initial state at T <= 0 cannot start
the model; any finite C_A can, and a negative estimate of it is left to the validity bounds.
For non-negative k0, E/R and UA every trajectory from a physical state stays bounded, so
the model has no finite escape; overflow of exp for absurd parameters is an error, never a
clipped value.

Derivatives, from the equations above, with d = q/V, h = -dH / (rho cp),
c = UA / (V rho cp), k = k(T) and r = k C_A:

    df/dx = [[-d - k,  -r (E/R) / T^2               ],
             [ h k,    -d + h r (E/R) / T^2 - c      ]]

    df/d ln k_350         = [-r, h r]
    df/d ((E/R) / T_ref)  = [-r, h r] (1 - T_ref / T)
    df/d ln UA            = [0, -c (T - T_c)]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from process_transfer.config import ModellerConfig, load_modeller
from process_transfer.cstr_variables import FloatArray
from process_transfer.data.paths import repository_root
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.modeller import cstr_first_order
from process_transfer.modeller.cstr_first_order import ModellerCSTRParameters
from process_transfer.validation import require_finite, require_non_negative

REFERENCE_TEMPERATURE = 350.0  # K, the known nominal feed temperature of both plants
COORDINATES = ("ln_k_350", "arrhenius_number", "ln_ua")
COORDINATES_WITH_FIXED_ACTIVATION = ("ln_k_350", "ln_ua")


def modeller_values(path: Path | None = None) -> ModellerConfig:
    """The modeller's textbook values, from ``configs/modeller_cstr.yaml`` by default. That
    file holds nothing hidden; a plant configuration is never read here."""
    return load_modeller(path or repository_root() / "configs" / "modeller_cstr.yaml")


@dataclass(frozen=True)
class MechanisticParameters:
    """The three values the modeller holds, in SI: k0 as the modeller's code takes it."""

    k0: float  # 1/s
    activation_temperature: float  # E/R, K
    ua: float  # W/K

    def __post_init__(self) -> None:
        for name in ("k0", "activation_temperature", "ua"):
            object.__setattr__(self, name, require_non_negative(name, getattr(self, name)))

    @property
    def k_350(self) -> float:
        """The rate constant at T_ref, 1/s."""
        return self.k0 * math.exp(-self.activation_temperature / REFERENCE_TEMPERATURE)

    @classmethod
    def textbook(cls, modeller: ModellerConfig) -> MechanisticParameters:
        return cls(
            k0=modeller.kinetics.k0.si,
            activation_temperature=modeller.kinetics.activation_temperature.si,
            ua=modeller.heat_transfer.UA.si,
        )

    @classmethod
    def from_k_350(
        cls, k_350: float, activation_temperature: float, ua: float
    ) -> MechanisticParameters:
        """From the rate constant at T_ref. Raises ``OverflowError`` when k0 overflows."""
        k_350 = require_finite("k_350", k_350)
        activation_temperature = require_non_negative(
            "activation_temperature", activation_temperature
        )
        return cls(
            k_350 * math.exp(activation_temperature / REFERENCE_TEMPERATURE),
            activation_temperature,
            ua,
        )

    def coordinates(self, fixed_activation: float | None) -> FloatArray:
        """theta: (ln k_350, (E/R) / T_ref, ln UA), or (ln k_350, ln UA) with E/R fixed."""
        if self.k0 <= 0.0 or self.ua <= 0.0:
            raise ValueError("coordinates need k0 > 0 and UA > 0; they are logarithms")
        arrhenius = self.activation_temperature / REFERENCE_TEMPERATURE
        ln_k_350 = math.log(self.k0) - arrhenius
        if fixed_activation is None:
            return np.array([ln_k_350, arrhenius, math.log(self.ua)])
        if self.activation_temperature != fixed_activation:
            raise ValueError(
                f"E/R is fixed at {fixed_activation!r} K and these parameters hold "
                f"{self.activation_temperature!r} K"
            )
        return np.array([ln_k_350, math.log(self.ua)])

    @classmethod
    def from_coordinates(
        cls, theta: FloatArray, fixed_activation: float | None
    ) -> MechanisticParameters:
        """The parameters at theta. Raises ``OverflowError`` when exp overflows: that is a
        point outside the domain of the model, not a value to clip."""
        theta = np.asarray(theta, dtype=np.float64)
        if fixed_activation is None:
            ln_k_350, arrhenius, ln_ua = (float(value) for value in theta)
        else:
            ln_k_350, ln_ua = (float(value) for value in theta)
            arrhenius = fixed_activation / REFERENCE_TEMPERATURE
        return cls(
            k0=math.exp(ln_k_350 + arrhenius),
            activation_temperature=(
                REFERENCE_TEMPERATURE * arrhenius if fixed_activation is None else fixed_activation
            ),
            ua=math.exp(ln_ua),
        )


class MechanisticModel:
    """The modeller's equations with the given parameters: a continuous-time model with
    the derivatives the rollout needs for sensitivities."""

    def __init__(
        self,
        name: str,
        known: KnownPlant,
        parameters: MechanisticParameters,
        fixed_activation: float | None = None,
    ) -> None:
        self.name = name
        self.known = known
        self.parameters = parameters
        self.fixed_activation = fixed_activation
        self.parameter_names = (
            COORDINATES if fixed_activation is None else COORDINATES_WITH_FIXED_ACTIVATION
        )
        self.modeller_parameters = ModellerCSTRParameters(
            volume=known.volume,
            density=known.density,
            heat_capacity=known.heat_capacity,
            reaction_enthalpy=known.reaction_enthalpy,
            k0=parameters.k0,
            activation_temperature=parameters.activation_temperature,
            ua=parameters.ua,
        )
        p = self.modeller_parameters
        # the same operations as the modeller's right-hand side
        self._heat_per_mole = (-p.reaction_enthalpy) / (p.density * p.heat_capacity)
        self._exchange = p.ua / (p.volume * p.density * p.heat_capacity)

    def rhs(self, x: FloatArray, u: FloatArray) -> FloatArray:
        return cstr_first_order.rhs(0.0, x, u, self.modeller_parameters)

    def initial_state_problem(self, x0: FloatArray) -> str | None:
        if not x0[1] > 0.0:
            return (
                f"the rate law divides by the temperature, and the initial state has "
                f"T = {float(x0[1])!r} K"
            )
        return None

    def jacobian_state(self, x: FloatArray, u: FloatArray) -> FloatArray:
        c_a, temperature = float(x[0]), float(x[1])
        p = self.modeller_parameters
        dilution = float(u[0]) / p.volume
        k = cstr_first_order.reaction_rate(1.0, temperature, p)
        rate_slope = k * c_a * p.activation_temperature / temperature**2  # d r / dT
        h = self._heat_per_mole
        return np.array(
            [
                [-dilution - k, -rate_slope],
                [h * k, -dilution + h * rate_slope - self._exchange],
            ]
        )

    def jacobian_parameters(self, x: FloatArray, u: FloatArray) -> FloatArray:
        c_a, temperature = float(x[0]), float(x[1])
        p = self.modeller_parameters
        rate = cstr_first_order.reaction_rate(1.0, temperature, p) * c_a
        h = self._heat_per_mole
        columns = [[-rate, h * rate]]
        if self.fixed_activation is None:
            factor = 1.0 - REFERENCE_TEMPERATURE / temperature
            columns.append([-rate * factor, h * rate * factor])
        columns.append([0.0, -self._exchange * (temperature - float(u[3]))])
        return np.array(columns).T


def nominal_model(known: KnownPlant, modeller: ModellerConfig) -> MechanisticModel:
    """MN: the modeller's equations with the textbook values, nothing fitted."""
    return MechanisticModel("MN", known, MechanisticParameters.textbook(modeller))
