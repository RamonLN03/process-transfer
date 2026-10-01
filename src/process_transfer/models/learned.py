"""The learned models of M1: the neural black box BN and the hybrids HK, HU and HKU
(``docs/m1_plan.md``, sections 8.2, 8.3, 8.5 and 8.7).

The equations are written once, on an array module ``xp`` that is numpy or ``jax.numpy``,
so that the model trained with JAX (``models.training``) and the model evaluated by the
reference rollout of ``models.rollout`` are the same equation, computed by the same
operations. Nothing here imports JAX.

Networks. A network N is a sequence of layers (W, b); every hidden layer applies tanh, the
last is linear:

    N(z) = W_L tanh( ... tanh(W_1 z + b_1) ... ) + b_L

Every network starts with its last layer zero, W_L = 0 and b_L = 0, so that N = 0: each
hybrid starts as its mechanistic model and BN as a model whose state does not move. The
hidden layers start from Glorot's uniform law and zero biases, drawn from a declared seed;
they are not zero, so the gradient with respect to the last layer is not zero at the start
and the identity does not block learning.

Scales, from the fitting part F only (section 5.8): the mean m and the standard deviation s
of each state over the readings of F, contexts and scored readings, and of each input over
the rows whose inputs drive the predictions of F. A scale that is zero is refused, with the
name of the variable (``fitting_scales``); the refusal is a training failure.

BN, the neural black box, with z = ((x - m_x) / s_x, (u - m_u) / s_u):

    dx/dt = D N(z),    D = s_x / (60 s)

D makes an output of order one a change of one standard deviation of the state per minute,
the order of the plant's slowest time constant; it is a scale of the output, not knowledge
of the plant.

The hybrids keep the modeller's balances, with d = q / V, h = -dH / (rho cp) and
c = UA / (V rho cp):

    dC/dt = d (C_f - C) - r
    dT/dt = d (T_f - T) + h r - c (T - T_c)
    r     = k_350 exp(-(E/R) (1/T - 1/T_ref)) C g_K
    c     = UA g_U / (V rho cp)

HK: g_K = exp(N_K((C - m_C) / s_C, (T - m_T) / s_T)), g_U = 1. The arguments of the kinetic
factor are those a rate law can depend on, composition and temperature.
HU: g_K = 1, g_U = exp(N_U((T - m_T) / s_T, (T_c - m_Tc) / s_Tc)), the temperatures a film
coefficient depends on.
HKU: both factors, each with its own network.

The factors are positive whatever the weights, and equal to one when the last layer is zero.
The rate is then never negative for C >= 0, it vanishes without A, and the stoichiometry and
the heat of reaction are the known ones; the conductance is positive. The mechanistic
parameters are those of MR, in its coordinates theta = (ln k_350, (E/R) / T_ref, ln UA), or
(ln k_350, ln UA) with E/R held, and are trained with the networks. Only the networks are
regularised (``models.training``).

Domain. The rate law divides by the temperature, so a hybrid cannot start from T <= 0, as
MR cannot. BN has no such limit. An exponential that overflows, in the rate or in a factor,
is an arithmetic error that the rollout records as a failure; nothing is clipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from process_transfer.cstr_variables import FloatArray
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import WindowData
from process_transfer.models.mechanistic import REFERENCE_TEMPERATURE, MechanisticParameters

BN = "BN"
HK = "HK"
HU = "HU"
HKU = "HKU"
FAMILIES = (BN, HK, HU, HKU)
HYBRIDS = (HK, HU, HKU)
DERIVATIVE_TIME = 60.0  # s: D = s_x / DERIVATIVE_TIME

# The networks of each family, and the number of their inputs and outputs
_NETWORKS = {
    BN: {"dynamics": (6, 2)},
    HK: {"kinetic": (2, 1)},
    HU: {"thermal": (2, 1)},
    HKU: {"kinetic": (2, 1), "thermal": (2, 1)},
}

Layers = list[tuple[Any, Any]]


@dataclass(frozen=True)
class Scales:
    """The centring and scaling of the states and inputs, computed from F."""

    state_mean: FloatArray  # (2,)
    state_std: FloatArray
    input_mean: FloatArray  # (4,)
    input_std: FloatArray

    @property
    def derivative(self) -> FloatArray:
        """D of BN."""
        return self.state_std / DERIVATIVE_TIME


_STATE_NAMES = ("C_A", "T")
_INPUT_NAMES = ("q", "C_Af", "T_f", "T_c")


def fitting_scales(windows: Sequence[WindowData]) -> Scales:
    """The mean and standard deviation of each variable over the rows of ``windows``: the
    readings of their contexts and scored readings, and the inputs of the rows that drive
    their predictions. Raises ``ValueError`` naming a variable whose deviation is zero."""
    if not windows:
        raise ValueError("scales need at least one window of F")
    readings = np.concatenate([np.vstack([d.context, d.scored]) for d in windows])
    inputs = np.concatenate([d.inputs for d in windows])
    scales = Scales(
        state_mean=readings.mean(axis=0),
        state_std=readings.std(axis=0),
        input_mean=inputs.mean(axis=0),
        input_std=inputs.std(axis=0),
    )
    for names, values in ((_STATE_NAMES, scales.state_std), (_INPUT_NAMES, scales.input_std)):
        for name, value in zip(names, values, strict=True):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"the standard deviation of {name} over the fitting part is {value!r}; "
                    "no scale is invented for a variable that does not vary (section 5.8)"
                )
    return scales


def network(xp: Any, layers: Layers, z: Any) -> Any:
    """N(z): tanh on every hidden layer, the last layer linear."""
    a = z
    for w, b in layers[:-1]:
        a = xp.tanh(a @ w + b)
    w, b = layers[-1]
    return a @ w + b


def initial_layers(
    rng: np.random.Generator, inputs: int, hidden: Sequence[int], outputs: int
) -> Layers:
    """Glorot's uniform law for the hidden layers, zero biases, and a zero last layer."""
    sizes = [inputs, *hidden, outputs]
    layers = []
    for fan_in, fan_out in zip(sizes[:-2], sizes[1:-1], strict=True):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        layers.append((rng.uniform(-limit, limit, size=(fan_in, fan_out)), np.zeros(fan_out)))
    layers.append((np.zeros((sizes[-2], outputs)), np.zeros(outputs)))
    return layers


def initial_parameters(
    family: str,
    hidden: Sequence[int],
    rng: np.random.Generator,
    start: MechanisticParameters | None,
    fixed_activation: float | None = None,
) -> dict[str, Any]:
    """The parameters a training starts from: the networks of the family, and for a hybrid
    the coordinates of ``start``, which is MR_F of the same replicate and budget."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if not hidden or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in hidden):
        raise ValueError(f"hidden must be a non-empty list of positive integers, got {hidden!r}")
    parameters: dict[str, Any] = {}
    if family in HYBRIDS:
        if start is None:
            raise ValueError(f"{family} starts from MR_F, and none was given")
        parameters["theta"] = start.coordinates(fixed_activation)
    elif start is not None:
        raise ValueError("BN has no mechanistic parameters to start from")
    for name, (n_in, n_out) in _NETWORKS[family].items():
        parameters[name] = initial_layers(rng, n_in, hidden, n_out)
    return parameters


def network_size(parameters: dict[str, Any]) -> int:
    """The number of weights and biases of the networks."""
    return sum(
        int(np.size(w)) + int(np.size(b))
        for name, layers in parameters.items()
        if name != "theta"
        for w, b in layers
    )


def rhs(
    xp: Any,
    family: str,
    parameters: dict[str, Any],
    x: Any,
    u: Any,
    known: KnownPlant,
    scales: Scales,
    fixed_activation: float | None = None,
) -> Any:
    """dx/dt of the model, for states x (..., 2) and inputs u (..., 4)."""
    if family == BN:
        z = xp.concatenate(
            [
                (x - scales.state_mean) / scales.state_std,
                (u - scales.input_mean) / scales.input_std,
            ],
            axis=-1,
        )
        return scales.derivative * network(xp, parameters["dynamics"], z)
    c_a, temperature = x[..., 0], x[..., 1]
    q, c_af, t_f, t_c = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    theta = parameters["theta"]
    if fixed_activation is None:
        ln_k_350, arrhenius, ln_ua = theta[0], theta[1], theta[2]
    else:
        ln_k_350, ln_ua = theta[0], theta[1]
        arrhenius = fixed_activation / REFERENCE_TEMPERATURE
    # (E/R) (1/T - 1/T_ref) = ((E/R) / T_ref) (T_ref / T - 1)
    rate = xp.exp(ln_k_350 - arrhenius * (REFERENCE_TEMPERATURE / temperature - 1.0)) * c_a
    if "kinetic" in parameters:
        z = xp.stack(
            [
                (c_a - scales.state_mean[0]) / scales.state_std[0],
                (temperature - scales.state_mean[1]) / scales.state_std[1],
            ],
            axis=-1,
        )
        rate = rate * xp.exp(network(xp, parameters["kinetic"], z)[..., 0])
    exchange = xp.exp(ln_ua) / known.thermal_mass
    if "thermal" in parameters:
        z = xp.stack(
            [
                (temperature - scales.state_mean[1]) / scales.state_std[1],
                (t_c - scales.input_mean[3]) / scales.input_std[3],
            ],
            axis=-1,
        )
        exchange = exchange * xp.exp(network(xp, parameters["thermal"], z)[..., 0])
    dilution = q / known.volume
    d_c = dilution * (c_af - c_a) - rate
    d_t = (
        dilution * (t_f - temperature)
        + known.heat_release_per_mole * rate
        - exchange * (temperature - t_c)
    )
    return xp.stack([d_c, d_t], axis=-1)


class LearnedModel:
    """A trained learned model as a continuous-time model in numpy, for the reference
    rollout of the evaluation."""

    def __init__(
        self,
        name: str,
        family: str,
        parameters: dict[str, Any],
        known: KnownPlant,
        scales: Scales,
        fixed_activation: float | None = None,
    ) -> None:
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
        self.name, self.family, self.known, self.scales = name, family, known, scales
        self.fixed_activation = fixed_activation
        self.parameters = to_numpy(parameters)
        problem = domain_problem(family, self.parameters, fixed_activation)
        if problem is not None:
            raise ValueError(f"{name}: {problem}")
        self._mechanistic = (
            None
            if family == BN
            else MechanisticParameters.from_coordinates(self.parameters["theta"], fixed_activation)
        )

    def rhs(self, x: FloatArray, u: FloatArray) -> FloatArray:
        return rhs(
            np, self.family, self.parameters, x, u, self.known, self.scales, self.fixed_activation
        )

    def initial_state_problem(self, x0: FloatArray) -> str | None:
        if self.family in HYBRIDS and not x0[1] > 0.0:
            return (
                f"the rate law divides by the temperature, and the initial state has "
                f"T = {float(x0[1])!r} K"
            )
        return None

    def mechanistic_parameters(self) -> MechanisticParameters | None:
        """k_350, E/R and UA of a hybrid, or None for BN."""
        return self._mechanistic


def domain_problem(
    family: str, parameters: dict[str, Any], fixed_activation: float | None = None
) -> str | None:
    """Why ``parameters`` lie outside the domain of the model, or None when they do not.

    The domain is the one declared for every model of the family, the same in training,
    evaluation and storage: weights and parameters that are finite doubles, and for a hybrid
    mechanistic parameters that ``MechanisticParameters`` accepts, so an E/R that is not
    negative, as MR's fit keeps it with its bound, and a k0 and UA that are positive
    doubles. E/R = 0 is in the domain: a rate that does not depend on temperature."""
    for name, value in parameters.items():
        arrays = [value] if name == "theta" else [a for layer in value for a in layer]
        if not all(np.all(np.isfinite(np.asarray(a, dtype=np.float64))) for a in arrays):
            return f"the parameters {name!r} are not all finite"
    if family == BN:
        return None
    theta = np.asarray(parameters["theta"], dtype=np.float64)
    expected = 3 if fixed_activation is None else 2
    if theta.shape != (expected,):
        return f"theta must hold {expected} values, got shape {theta.shape}"
    try:
        MechanisticParameters.from_coordinates(theta, fixed_activation)
    except (OverflowError, ValueError) as error:
        return (
            f"the mechanistic parameters at theta = {theta.tolist()} are outside their "
            f"domain: {error}"
        )
    return None


def to_numpy(parameters: dict[str, Any]) -> dict[str, Any]:
    """A copy of the parameters as numpy arrays of doubles."""
    copy: dict[str, Any] = {}
    for name, value in parameters.items():
        if name == "theta":
            copy[name] = np.array(value, dtype=np.float64)
        else:
            copy[name] = [
                (np.array(w, dtype=np.float64), np.array(b, dtype=np.float64)) for w, b in value
            ]
    return copy
