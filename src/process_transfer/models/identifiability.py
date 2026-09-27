"""What a design of windows can tell about the parameters of the modeller's model, and the
profile of a fit against E/R: the reusable parts of M1-E01 (``docs/m1_plan.md``, section 7.4).

The point of evaluation (``steady_state_parameters``). Information is local: it is computed
at one point of the parameters. Section 7.4 puts that point where the model reproduces the
nominal steady state observed in the development data. At a steady state x = (C, T) under
the nominal inputs u = (q, C_f, T_f, T_c), with d = q / V and h = -dH / (rho cp), the two
balances of the modeller's model fix the rate and the conductance whatever E/R is:

    r    = d (C_f - C)                                 mass balance
    k(T) = r / C
    UA   = V rho cp [d (T_f - T) + h r] / (T - T_c)     energy balance

and k_350 = k(T) exp((E/R) (1/T - 1/T_ref)) for the E/R given. Domain: 0 < C < C_f, so that
the rate is positive; T > 0; T != T_c, since at T = T_c the energy balance leaves the
conductance undetermined; and the conductance so found must be positive and finite. A
state that the model cannot hold with such values is refused, never given the nearest one.

The information of one window (``window_information``). At parameters theta, for a window
started from x0 and driven by its known inputs, S is the sensitivity of the predictions at
the scored instants to theta and G their sensitivity to x0, each row divided by the noise
level of its channel, so that the weights 1 / sigma^2 of the loss become the identity. With
Sigma_0 = diag(sigma^2) / n, the covariance of the mean of n context readings of a constant
state, three p x p matrices:

    exact    S^T S                          information about theta with x0 known exactly
    context  S^T (I + G Sigma_0 G^T)^-1 S   information about theta when x0 is known only
                                            through its n context readings
    excess   C Sigma_0 C^T, C = S^T G       what the error of the context mean adds to the
                                            middle of the sandwich, S^T (I + G Sigma_0 G^T) S

``context`` is the Schur complement of theta in the Fisher information of (theta, x0) from
the scored readings and the context mean together; Woodbury's identity turns it into the
expression above, exact - C (Sigma_0^-1 + G^T G)^-1 C^T with C = S^T G. It is not computed
that way. The initial state is counted in standard deviations of its context mean, so that
G becomes G Sigma_0^(1/2) and Sigma_0 the identity, and the Schur complement is read from
the QR factorisation of the joint problem: the scored readings, rows [G Sigma_0^(1/2) S],
and the context mean, rows [I 0]. The block of theta in its triangular factor is a root of
the context information. Nothing is subtracted, and neither Sigma_0^-1 nor G^T G is formed:
with noise levels far from one, their sum can overflow while each is finite, and the solve
would then drop the correction and return the information with x0 exact (Codex's review of
``129063c``, ``docs/numerical_robustness.md``). Its inverse is the Cramer-Rao bound, at this
point and to first order, of an estimator that treats x0 as unknown and learns it from the
context. The estimator of M1 does not: it fixes x0 at the context mean and weights the
scored readings by 1 / sigma^2, and its covariance is the sandwich exact^-1 meat exact^-1,
which ``fitting.covariance`` computes on real windows and which is formed here the same way,
exact^-1 + exact^-1 excess exact^-1. In the order of positive semi-definite matrices

    exact^-1  <=  context^-1  <=  exact^-1 meat exact^-1,

the first gap being what ten context readings cannot tell about x0, the second what the
rule of the estimator adds to it. All three assume that the model is right, and all three
are local. The bound treats the initial state of each window as a free nuisance known only
through its context, not as the model's steady state, which depends on theta: it is the
bound for estimators that leave the initial state free.

Accuracy. The rank is judged without forming S^T S (below), but the context information of
a design is a sum of Gram matrices R^T R, inverted to give the bound, so the bound carries a
relative error of about the machine epsilon times the square of the condition number of S:
negligible for the condition numbers of the windows of P3, of the order of 10^2 to 10^3, and
the reason why a design whose S is close to rank deficient can be reported as having a
singular context information rather than as rank deficient.

A design (``design_covariance``). Windows are independent, since their contexts are
different readings and the noise is independent from sample to sample (D-020), so the
matrices of a design are the sums of those of its windows, each counted with a weight: a
number of windows, which may be fractional for an expected design. The singular values of
the stacked, normalised S of the design come from the triangular factors R of each window
(S = Q R), stacked with the square roots of their weights, so that its numerical rank is
judged by numpy's convention on those values, as ``fitting.covariance`` judges it, without
ever squaring S. A design that is rank deficient has no covariance, and says why.

The profile against E/R (``profile_settings``, ``profile_interval``). E/R is held at each
value of a grid and the other parameters are fitted by the procedure of MR, with its
declared starts that do not move E/R. The increase of the loss above its minimum,
2 N (J^2 - J_min^2) for N scored readings of two channels, is the increase of the sum of
the squared normalised residuals. If the model were right, a set where that increase stays
below 3.84 would be a 95 % profile interval for E/R with the initial states taken as exact;
the model of the target is not right, and the interval describes the sharpness of the
minimum, not an uncertainty.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from process_transfer.cstr_variables import FloatArray
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import CONTEXT_READINGS
from process_transfer.models.fitting import DEFAULT_FIT_SETTINGS, FitSettings
from process_transfer.models.mechanistic import REFERENCE_TEMPERATURE, MechanisticParameters
from process_transfer.models.rollout import (
    EVALUATION_SETTINGS,
    DifferentiableModel,
    RolloutFailure,
    RolloutSettings,
    rollout,
)
from process_transfer.validation import require_finite, require_non_negative

# 95 % point of a chi-square with one degree of freedom
CHI2_ONE_95 = 3.841458820694124


def steady_state_parameters(
    known: KnownPlant, state: FloatArray, activation_temperature: float
) -> MechanisticParameters:
    """The k_350 and UA with which the modeller's model, at the given E/R, has ``state`` as a
    steady state under the known nominal inputs."""
    x = np.asarray(state, dtype=np.float64)
    if x.shape != (2,) or not np.all(np.isfinite(x)):
        raise ValueError(f"the state must be two finite values, C_A and T; got {state!r}")
    c_a, temperature = float(x[0]), float(x[1])
    activation = require_non_negative("activation_temperature", activation_temperature)
    q, c_af, t_f, t_c = known.nominal_inputs
    if not 0.0 < c_a < c_af:
        raise ValueError(
            f"C_A = {c_a!r} mol/m^3 must lie between 0 and the nominal feed concentration "
            f"{c_af!r} mol/m^3 for a positive first-order rate to hold it steady"
        )
    if not temperature > 0.0:
        raise ValueError(f"the rate law divides by the temperature, and T = {temperature!r} K")
    if temperature == t_c:
        raise ValueError(
            f"at T = T_c = {t_c!r} K the energy balance does not determine the conductance"
        )
    dilution = q / known.volume
    rate = dilution * (c_af - c_a)
    k_at_t = rate / c_a
    heat_removed = dilution * (t_f - temperature) + known.heat_release_per_mole * rate  # K/s
    ua = known.thermal_mass * heat_removed / (temperature - t_c)
    if not (math.isfinite(ua) and ua > 0.0):
        raise ValueError(
            f"no positive conductance holds C_A = {c_a!r} mol/m^3 and T = {temperature!r} K "
            f"steady under the nominal inputs: the energy balance gives UA = {ua!r} W/K"
        )
    exponent = activation * (1.0 / temperature - 1.0 / REFERENCE_TEMPERATURE)
    try:
        k_350 = k_at_t * math.exp(exponent)
        return MechanisticParameters.from_k_350(k_350, activation, ua)
    except OverflowError:
        raise ValueError(
            f"with E/R = {activation!r} K the rate constant of this steady state overflows"
        ) from None


@dataclass(frozen=True)
class WindowInformation:
    """What one window tells about theta, in the coordinates of the model's parameters."""

    rows: int  # of S: one per scored reading and channel
    root: FloatArray  # R, with R^T R = S^T S; p x p when S has at least p rows
    exact: FloatArray  # S^T S
    context: FloatArray  # S^T (I + G Sigma_0 G^T)^-1 S
    excess: FloatArray  # C Sigma_0 C^T, C = S^T G
    end_state: FloatArray  # the predicted state at the last scored instant

    @property
    def meat(self) -> FloatArray:
        """S^T (I + G Sigma_0 G^T) S, the middle of the sandwich."""
        return self.exact + self.excess


@dataclass(frozen=True)
class NoInformation:
    """A window whose information could not be computed, and why."""

    reason: str


def window_information(
    model: DifferentiableModel,
    initial_state: FloatArray,
    inputs: FloatArray,
    sample_period: float,
    noise_std: FloatArray,
    context_readings: int = CONTEXT_READINGS,
    settings: RolloutSettings = EVALUATION_SETTINGS,
) -> WindowInformation | NoInformation:
    """The information about the parameters of ``model`` from one window started at
    ``initial_state`` and driven by ``inputs``, one row per sampling period."""
    sigma = np.asarray(noise_std, dtype=np.float64)
    if sigma.shape != (2,) or not np.all(np.isfinite(sigma)) or not np.all(sigma > 0.0):
        raise ValueError(f"the noise levels must be two positive finite values, got {noise_std!r}")
    if isinstance(context_readings, bool) or not isinstance(context_readings, int):
        raise ValueError(f"context_readings must be a positive integer, got {context_readings!r}")
    if context_readings < 1:
        raise ValueError(f"context_readings must be a positive integer, got {context_readings!r}")
    try:
        readings = float(context_readings)
    except OverflowError:
        raise ValueError(
            f"context_readings = {context_readings!r} is beyond the largest double"
        ) from None
    # the noise model: the variance of a reading and that of the context mean must be
    # positive doubles, and so must the factors that carry the initial state into units of
    # the standard deviation of its context mean, ratios of two noise levels
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        variance = sigma**2
        context_variance = variance / readings  # the diagonal of Sigma_0, in state units
        # (sigma_j / sqrt(n)) / sigma_o: row o, the channel read; column j, the initial state
        to_context_units = (sigma[None, :] / sigma[:, None]) / math.sqrt(readings)
    for name, values in (
        ("sigma^2", variance),
        ("sigma^2 / context_readings", context_variance),
        ("(sigma_j / sqrt(context_readings)) / sigma_o", to_context_units),
    ):
        if not (np.all(np.isfinite(values)) and np.all(values > 0.0)):
            raise ValueError(
                f"with noise levels {sigma.tolist()} and {context_readings} context readings, "
                f"{name} is {values.tolist()}, not positive doubles"
            )
    result = rollout(
        model,
        initial_state,
        inputs,
        sample_period,
        settings,
        parameter_sensitivities=True,
        initial_state_sensitivities=True,
    )
    if isinstance(result, RolloutFailure):
        return NoInformation(f"the rollout failed: {result.cause}: {result.detail}")
    p = len(model.parameter_names)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        s = (result.parameter_sensitivities / sigma[None, :, None]).reshape(-1, p)
        # G Sigma_0^(1/2): the sensitivity to the initial state counted in standard
        # deviations of its context mean, so that Sigma_0 becomes the identity
        g = (result.initial_state_sensitivities * to_context_units[None, :, :]).reshape(-1, 2)
        exact = s.T @ s
        coupling = s.T @ g  # C Sigma_0^(1/2), C = S^T G
        excess = coupling @ coupling.T
    for name, values in (
        ("S", s),
        ("G Sigma_0^(1/2)", g),
        ("S^T S", exact),
        ("S^T G Sigma_0^(1/2)", coupling),
        ("C Sigma_0 C^T", excess),
    ):
        if not np.all(np.isfinite(values)):
            return NoInformation(f"{name} of the normalised sensitivities is not representable")
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        # the joint least-squares problem of (x0, theta): the scored readings, rows [G S],
        # and the context mean, rows [I 0]; the block of theta in the triangular factor of
        # its QR factorisation is the root of the Schur complement, the context information
        joint = np.block([[g, s], [np.eye(2), np.zeros((2, p))]])
        context_root = np.linalg.qr(joint, mode="r")[2:, 2:]
        context = context_root.T @ context_root
        root = np.linalg.qr(s, mode="r")
    if not np.all(np.isfinite(context)):
        return NoInformation("the context information is not representable")
    return WindowInformation(
        rows=s.shape[0],
        root=root,
        exact=_symmetric(exact),
        context=_symmetric(context),
        excess=_symmetric(excess),
        end_state=np.array(result.states[-1]),
    )


def _symmetric(matrix: FloatArray) -> FloatArray:
    return 0.5 * (matrix + matrix.T)


@dataclass(frozen=True)
class DesignCovariance:
    """The covariances of a design of windows, in (ln k_350, E/R in K, ln UA), or without E/R
    when it is held fixed."""

    names: tuple[str, ...]
    singular_values: tuple[float, ...]  # of the stacked normalised S of the design
    exact_initial_state: FloatArray | None  # (S^T S)^-1
    context_bound: FloatArray | None  # (S^T Sigma^-1 S)^-1
    sandwich: FloatArray | None  # (S^T S)^-1 S^T Sigma S (S^T S)^-1
    reason: str | None  # why the covariances could not be computed

    def standard_errors(self, kind: str) -> dict[str, float]:
        """``kind`` is "exact", "context" or "sandwich"."""
        matrix = self._matrix(kind)
        return {name: math.sqrt(matrix[i, i]) for i, name in enumerate(self.names)}

    def correlations(self, kind: str) -> FloatArray:
        matrix = self._matrix(kind)
        sd = np.sqrt(np.diag(matrix))
        return matrix / np.outer(sd, sd)

    def _matrix(self, kind: str) -> FloatArray:
        chosen = {
            "exact": self.exact_initial_state,
            "context": self.context_bound,
            "sandwich": self.sandwich,
        }
        if kind not in chosen:
            raise ValueError(f"kind must be one of {sorted(chosen)}, got {kind!r}")
        if chosen[kind] is None:
            raise ValueError(f"no covariance: {self.reason}")
        return chosen[kind]


def design_covariance(
    blocks: Sequence[WindowInformation],
    weights: Sequence[float] | None = None,
    fixed_activation: float | None = None,
) -> DesignCovariance:
    """The covariances of a design made of ``blocks``, each counted ``weights`` times (once
    each when no weights are given)."""
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("a design needs at least one window")
    counts = np.ones(len(blocks)) if weights is None else np.asarray(weights, dtype=np.float64)
    if counts.shape != (len(blocks),):
        raise ValueError(f"{len(counts)} weights for {len(blocks)} windows")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0) or not np.any(counts > 0.0):
        raise ValueError(f"weights must be finite, not negative and not all zero, got {weights!r}")
    p = blocks[0].exact.shape[0]
    if any(block.exact.shape != (p, p) for block in blocks):
        raise ValueError("the windows of a design must have the same parameters")
    estimated = fixed_activation is None
    names = ("ln k_350", "E/R", "ln UA") if estimated else ("ln k_350", "ln UA")
    if len(names) != p:
        raise ValueError(f"{p} parameters, and E/R is {'estimated' if estimated else 'fixed'}")
    units = np.array([1.0, REFERENCE_TEMPERATURE, 1.0] if estimated else [1.0, 1.0])
    used = [(w, b) for w, b in zip(counts, blocks, strict=True) if w > 0.0]
    rows = math.fsum(float(w) * b.rows for w, b in used)  # Python floats: inf, no warning
    if not math.isfinite(rows):
        raise ValueError(f"the weights {weights!r} give a number of rows beyond the largest double")
    stacked = np.vstack([math.sqrt(w) * b.root for w, b in used])
    _, singular, right = np.linalg.svd(stacked, full_matrices=False)
    values = tuple(float(v) for v in singular)
    # numpy's convention for the numerical rank, with the rows of the design's S; fewer
    # singular values than parameters means fewer rows than parameters
    if len(singular) < p or singular[-1] <= singular[0] * max(rows, p) * np.finfo(float).eps:
        return DesignCovariance(names, values, None, None, None, "the design is rank deficient")
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        bread = right.T @ np.diag(1.0 / singular**2) @ right
        context = sum(w * b.context for w, b in used)
        excess = sum(w * b.excess for w, b in used)
        sandwich = bread + bread @ excess @ bread  # as fitting.covariance forms it
    eigenvalues, vectors = np.linalg.eigh(_symmetric(context))
    if not eigenvalues[-1] > 0.0 or eigenvalues[0] <= eigenvalues[-1] * p * np.finfo(float).eps:
        reason = "the information with the initial state from its context is singular"
        return DesignCovariance(names, values, None, None, None, reason)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        bound = vectors @ np.diag(1.0 / eigenvalues) @ vectors.T
        scale = np.outer(units, units)
        matrices = [_symmetric(m) * scale for m in (bread, bound, sandwich)]
    if not all(np.all(np.isfinite(m)) for m in matrices):
        reason = "a covariance is not representable in double precision"
        return DesignCovariance(names, values, None, None, None, reason)
    return DesignCovariance(names, values, *matrices, None)


def profile_settings(
    activation: float, settings: FitSettings = DEFAULT_FIT_SETTINGS
) -> FitSettings:
    """The settings of a fit with E/R held at ``activation``: those of ``settings``, with its
    declared starts that do not move E/R."""
    activation = require_finite("activation", activation)
    starts = tuple(start for start in settings.starts if start.activation_shift == 0.0)
    if not starts:
        raise ValueError("every declared start moves E/R; none can start a fit with E/R held")
    return dataclasses.replace(settings, fixed_activation_temperature=activation, starts=starts)


@dataclass(frozen=True)
class ProfileInterval:
    """Where the increase of the loss crosses a threshold on each side of the lowest point of
    a profile, by linear interpolation between the points of the grid."""

    threshold: float
    lowest: float  # the E/R of the lowest point of the grid
    low: float | None
    high: float | None
    low_reason: str | None  # why no crossing was found below, if none was
    high_reason: str | None


def profile_interval(
    activations: Sequence[float], increases: Sequence[float | None], threshold: float
) -> ProfileInterval:
    """The crossings of ``threshold`` by the increase of the loss, walking out from the
    lowest point of the grid. A point with no value, where no start converged, stops the
    walk: nothing is interpolated across it."""
    grid = np.asarray(activations, dtype=np.float64)
    if grid.ndim != 1 or len(grid) != len(increases) or len(grid) == 0:
        raise ValueError("one increase for each point of the grid, and at least one point")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("the grid must be finite and strictly increasing")
    if not (math.isfinite(threshold) and threshold > 0.0):
        raise ValueError(f"the threshold must be positive and finite, got {threshold!r}")
    values = [None if v is None else require_finite("increase", v) for v in increases]
    known = [i for i, v in enumerate(values) if v is not None]
    if not known:
        raise ValueError("no point of the grid has a value")
    lowest = min(known, key=lambda i: (values[i], i))
    if values[lowest] > threshold:
        raise ValueError(
            f"the lowest increase, {values[lowest]!r}, is above the threshold; the minimum of "
            "the profile is not where the increases were measured from"
        )

    def walk(step: int) -> tuple[float | None, str | None]:
        i = lowest
        while True:
            j = i + step
            if j < 0 or j >= len(grid):
                return None, "the increase stays below the threshold to the end of the grid"
            if values[j] is None:
                return None, f"no start converged at E/R = {float(grid[j])!r} K, before a crossing"
            if values[j] > threshold:
                a, b = values[i], values[j]
                return float(grid[i] + (threshold - a) / (b - a) * (grid[j] - grid[i])), None
            i = j

    low, low_reason = walk(-1)
    high, high_reason = walk(+1)
    return ProfileInterval(threshold, float(grid[lowest]), low, high, low_reason, high_reason)
