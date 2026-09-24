"""Estimation of MR and MR_F (``docs/m1_plan.md``, sections 5.5, 7.4, 8.1 and 8.7).

MR, the comparator, is the modeller's model with k_350, E/R and UA estimated on all the
windows of a budget. MR_F is the same procedure on the fitting part F alone; it starts the
hybrids. The procedure is one function, ``fit_mechanistic``, and the two differ only in the
windows it is handed. It is handed ``WindowData`` and nothing else: no observations, no
split, no path. A fit of MR_F has no way to read V.

Loss. The score J of section 9.1 on the scored readings of the windows, each window started
from the mean of its own context:

    J^2 = 1 / (2 N)  sum over the N scored readings and both channels of
          ((prediction - reading) / sigma)^2

``scipy.optimize.least_squares`` minimises half the sum of the squares of the residuals
((prediction - reading) / sigma) / sqrt(2 N), whose minimiser is that of J; the objective
of a start is J at its endpoint. The sigmas are those of the data sheet. The Jacobian comes
from the sensitivities integrated with the states, not from finite differences, which with
an adaptive integrator would difference its error along with the model.

Coordinates and scales. theta = (ln k_350, (E/R) / T_ref, ln UA) (``mechanistic``), with
(E/R) / T_ref >= 0, since the model is defined for a non-negative activation temperature,
and the others free. The trust region is scaled by the norms of the columns of the
Jacobian (``x_scale="jac"``), which makes each step independent of the units of each
coordinate; the three coordinates are of comparable size, so the tolerances on the step
act on them alike. Every option of ``least_squares`` is given explicitly, so that a change
of its defaults between versions of SciPy changes nothing here.

Starts and selection. The fit starts from the textbook values and from a declared list of
further points, in that order (``DEFAULT_STARTS``). The outcome of each start is recorded:

    converged           least_squares stopped on one of its tolerances (status 1 to 4)
    budget exhausted    it reached its limit of evaluations (status 0)
    numerical failure   the rollout of a window failed, or the parameters overflowed

A failed rollout ends its start. The TRF method of least_squares happens to treat a trial
point with residuals that are not finite as a rejected step, but that is not documented,
and a failure the procedure does not treat must end the fit (section 8.7). Nothing replaces
a residual. The fit keeps, among the starts that converged, the endpoint with the lowest
objective, the first in the declared order on a tie. A start that exhausted its budget is
recorded with its endpoint and is not selectable: its endpoint satisfies none of the
declared criteria. If no start converged, the fit is a training failure, with the outcome
of every start in its reason. Every endpoint is kept, with the spread of each parameter over
the converged ones and the singular values of the Jacobian at each: several solutions, or
directions the data do not determine, show up there without a threshold deciding for them.

Covariance (``covariance``). At an estimate, with S the sensitivity of the predictions to
theta, W = 1 / sigma^2, G the sensitivity of the predictions of a window to its initial
state and Sigma_0 = diag(sigma^2) / 10 the covariance of a context mean, the covariance of
the estimator is the sandwich

    (S^T W S)^-1  S^T W Sigma W S  (S^T W S)^-1,     Sigma = D_sigma + G Sigma_0 G^T

block-diagonal over windows, whose contexts are different readings. (S^T W S)^-1 alone is
the covariance with the initial state taken as exact, which is optimistic. Both assume the
model is right, and neither says anything about a wrong structure.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from process_transfer.config import ModellerConfig
from process_transfer.cstr_variables import FloatArray
from process_transfer.evaluation.budgets import BudgetSplit
from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import CONTEXT_READINGS, WindowData, window_data
from process_transfer.measurement.observations import Observations
from process_transfer.models.mechanistic import (
    REFERENCE_TEMPERATURE,
    MechanisticModel,
    MechanisticParameters,
)
from process_transfer.models.rollout import (
    EVALUATION_SETTINGS,
    RolloutFailure,
    RolloutSettings,
    rollout,
)
from process_transfer.validation import require_finite, require_positive

CONVERGED = "converged"
BUDGET_EXHAUSTED = "budget exhausted"
NUMERICAL_FAILURE = "numerical failure"


@dataclass(frozen=True)
class Start:
    """A starting point, declared relative to the textbook values."""

    label: str
    k_350_factor: float = 1.0
    activation_shift: float = 0.0  # K, added to E/R
    ua_factor: float = 1.0

    def __post_init__(self) -> None:
        require_positive("k_350_factor", self.k_350_factor)
        require_finite("activation_shift", self.activation_shift)
        require_positive("ua_factor", self.ua_factor)

    def point(self, textbook: MechanisticParameters) -> MechanisticParameters:
        activation = textbook.activation_temperature + self.activation_shift
        if activation <= 0.0:
            raise ValueError(f"start {self.label!r} puts E/R at {activation!r} K")
        return MechanisticParameters.from_k_350(
            textbook.k_350 * self.k_350_factor, activation, textbook.ua * self.ua_factor
        )


# The textbook values, then a point on each side of them for the two parameters that the
# energy balance couples, k and UA, and for E/R. The factors of two and the 2000 K are wide
# against the standard errors of section 7.5 of the plan and narrow against the physics:
# every start is a model a modeller could have written down.
DEFAULT_STARTS = (
    Start("textbook"),
    Start("k_350 x 2, UA / 2", k_350_factor=2.0, ua_factor=0.5),
    Start("k_350 / 2, UA x 2", k_350_factor=0.5, ua_factor=2.0),
    Start("E/R - 2000 K", activation_shift=-2000.0),
    Start("E/R + 2000 K", activation_shift=2000.0),
)
STARTS_WITH_FIXED_ACTIVATION = DEFAULT_STARTS[:3]


@dataclass(frozen=True)
class FitSettings:
    """Everything that decides a fit besides its data."""

    fixed_activation_temperature: float | None = None  # K; None: E/R is estimated
    starts: tuple[Start, ...] = DEFAULT_STARTS
    max_evaluations: int = 100  # of the residuals, per start: least_squares' max_nfev
    ftol: float = 1.0e-10
    xtol: float = 1.0e-10
    gtol: float = 1.0e-10
    rollout: RolloutSettings = EVALUATION_SETTINGS

    def __post_init__(self) -> None:
        if not self.starts:
            raise ValueError("a fit needs at least one start")
        labels = [start.label for start in self.starts]
        if len(set(labels)) != len(labels):
            raise ValueError(f"the labels of the starts must be distinct, got {labels}")
        if self.fixed_activation_temperature is not None:
            require_positive("fixed_activation_temperature", self.fixed_activation_temperature)
            moved = [start.label for start in self.starts if start.activation_shift != 0.0]
            if moved:
                raise ValueError(f"E/R is fixed, and the starts {moved} move it")
        limit = self.max_evaluations
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"max_evaluations must be a positive integer, got {limit!r}")
        for name in ("ftol", "xtol", "gtol"):
            require_positive(name, getattr(self, name))


DEFAULT_FIT_SETTINGS = FitSettings()


@dataclass(frozen=True)
class StartRecord:
    """What one start of a fit did."""

    start: Start
    initial: MechanisticParameters
    outcome: str  # CONVERGED, BUDGET_EXHAUSTED or NUMERICAL_FAILURE
    message: str
    final: MechanisticParameters | None  # None after a numerical failure
    objective: float | None  # J on the windows of the fit, at the endpoint
    singular_values: tuple[float, ...] | None  # of the Jacobian of (prediction - reading) / sigma
    evaluations: int  # of the residuals: each one a rollout of every window
    jacobian_evaluations: int
    seconds: float = field(compare=False)


@dataclass(frozen=True)
class FitResult:
    """A fit of MR or MR_F: every start, the one selected, or the training failure."""

    name: str
    windows: tuple[tuple[str, int], ...]  # the windows the fit was handed, and no other
    readings: int
    settings: FitSettings
    textbook: MechanisticParameters
    starts: tuple[StartRecord, ...]
    selected: int | None
    training_failure: TrainingFailure | None

    @property
    def parameters(self) -> MechanisticParameters | None:
        return None if self.selected is None else self.starts[self.selected].final

    @property
    def objective(self) -> float | None:
        return None if self.selected is None else self.starts[self.selected].objective

    def model(self, known: KnownPlant) -> MechanisticModel:
        if self.parameters is None:
            raise ValueError(f"{self.name} has no model: {self.training_failure}")
        return MechanisticModel(
            self.name, known, self.parameters, self.settings.fixed_activation_temperature
        )

    def spread(self) -> dict[str, tuple[float, float]]:
        """The smallest and largest value of each parameter over the converged endpoints."""
        finals = [s.final for s in self.starts if s.outcome == CONVERGED and s.final is not None]
        if not finals:
            return {}
        values = {
            "k_350": [p.k_350 for p in finals],
            "activation_temperature": [p.activation_temperature for p in finals],
            "ua": [p.ua for p in finals],
        }
        return {name: (min(found), max(found)) for name, found in values.items()}


class _FitStop(Exception):
    """A point at which the loss cannot be evaluated; it ends the start."""


class _Loss:
    """Residuals and their Jacobian at theta, computed together and kept for the last theta,
    since least_squares asks for both at every point it accepts."""

    def __init__(
        self,
        name: str,
        windows: Sequence[WindowData],
        known: KnownPlant,
        fixed_activation: float | None,
        settings: RolloutSettings,
    ) -> None:
        self.name, self.windows, self.known = name, windows, known
        self.fixed_activation, self.settings = fixed_activation, settings
        self.sigma = windows[0].noise_std
        self.scale = 1.0 / math.sqrt(2.0 * sum(len(data.scored) for data in windows))
        self.evaluations = 0
        self._key: bytes | None = None
        self._value: tuple[FloatArray, FloatArray] | None = None

    def _evaluate(self, theta: FloatArray) -> tuple[FloatArray, FloatArray]:
        key = np.asarray(theta, dtype=np.float64).tobytes()
        if key == self._key and self._value is not None:
            return self._value
        try:
            parameters = MechanisticParameters.from_coordinates(theta, self.fixed_activation)
        except OverflowError:
            raise _FitStop(f"the parameters overflow at theta = {list(theta)}") from None
        model = MechanisticModel(self.name, self.known, parameters, self.fixed_activation)
        residuals, jacobian = [], []
        for data in self.windows:
            result = rollout(
                model,
                data.initial_state,
                data.inputs,
                data.sample_period,
                self.settings,
                parameter_sensitivities=True,
            )
            if isinstance(result, RolloutFailure):
                raise _FitStop(
                    f"the rollout of window {data.key} failed at theta = {list(theta)}: "
                    f"{result.cause}: {result.detail}"
                )
            residuals.append(((result.states - data.scored) / self.sigma).ravel())
            normalised = result.parameter_sensitivities / self.sigma[None, :, None]
            jacobian.append(normalised.reshape(-1, len(model.parameter_names)))
        self.evaluations += 1
        self._key = key
        self._value = (np.concatenate(residuals) * self.scale, np.vstack(jacobian) * self.scale)
        return self._value

    def residuals(self, theta: FloatArray) -> FloatArray:
        return self._evaluate(theta)[0]

    def jacobian(self, theta: FloatArray) -> FloatArray:
        return self._evaluate(theta)[1]


def _check_windows(windows: Sequence[WindowData]) -> None:
    if len(windows) == 0:
        raise ValueError("a fit needs at least one window")
    keys = [data.key for data in windows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"a window is given twice: {keys}")
    sigma = windows[0].noise_std
    if any(not np.array_equal(data.noise_std, sigma) for data in windows):
        raise ValueError("the windows of a fit must share the noise levels of their sensors")
    if not np.all(sigma > 0.0):
        raise ValueError(
            f"the loss divides by the noise level of each sensor, and they are {sigma.tolist()}"
        )


def select_start(records: Sequence[StartRecord]) -> int | None:
    """The position of the selected start: among those that converged, the lowest
    objective, the first in the declared order on a tie. None when none converged."""
    converged = [i for i, record in enumerate(records) if record.outcome == CONVERGED]
    if not converged:
        return None
    return min(converged, key=lambda i: (records[i].objective, i))


def fit_mechanistic(
    name: str,
    windows: Sequence[WindowData],
    known: KnownPlant,
    modeller: ModellerConfig,
    settings: FitSettings = DEFAULT_FIT_SETTINGS,
) -> FitResult:
    """Estimate the modeller's parameters on ``windows``, from every declared start."""
    windows = tuple(windows)
    _check_windows(windows)
    fixed = settings.fixed_activation_temperature
    textbook = MechanisticParameters.textbook(modeller)
    if fixed is None:
        bounds = (np.array([-np.inf, 0.0, -np.inf]), np.full(3, np.inf))
    else:
        bounds = (np.full(2, -np.inf), np.full(2, np.inf))
    records = []
    for start in settings.starts:
        initial = start.point(textbook)
        if fixed is not None:
            initial = MechanisticParameters.from_k_350(initial.k_350, fixed, initial.ua)
        loss = _Loss(name, windows, known, fixed, settings.rollout)
        began = time.perf_counter()
        try:
            found = least_squares(
                loss.residuals,
                initial.coordinates(fixed),
                jac=loss.jacobian,
                bounds=bounds,
                method="trf",
                ftol=settings.ftol,
                xtol=settings.xtol,
                gtol=settings.gtol,
                x_scale="jac",
                loss="linear",
                max_nfev=settings.max_evaluations,
                tr_solver="exact",
                verbose=0,
            )
        except _FitStop as stop:
            records.append(
                StartRecord(
                    start,
                    initial,
                    NUMERICAL_FAILURE,
                    str(stop),
                    None,
                    None,
                    None,
                    loss.evaluations,
                    0,
                    time.perf_counter() - began,
                )
            )
            continue
        outcome = CONVERGED if found.status > 0 else BUDGET_EXHAUSTED
        normalised_jacobian = found.jac / loss.scale
        records.append(
            StartRecord(
                start=start,
                initial=initial,
                outcome=outcome,
                message=f"status {found.status}: {found.message}",
                final=MechanisticParameters.from_coordinates(found.x, fixed),
                objective=math.sqrt(2.0 * found.cost),
                singular_values=tuple(
                    float(s) for s in np.linalg.svd(normalised_jacobian, compute_uv=False)
                ),
                evaluations=int(found.nfev),
                jacobian_evaluations=int(found.njev),
                seconds=time.perf_counter() - began,
            )
        )
    selected = select_start(records)
    failure = None
    if selected is None:
        outcomes = "; ".join(f"{r.start.label}: {r.outcome} ({r.message})" for r in records)
        failure = TrainingFailure(f"no start of {name} converged: {outcomes}")
    return FitResult(
        name=name,
        windows=tuple(data.key for data in windows),
        readings=sum(len(data.scored) for data in windows),
        settings=settings,
        textbook=textbook,
        starts=tuple(records),
        selected=selected,
        training_failure=failure,
    )


def fit_mr(
    split: BudgetSplit,
    observations: Observations,
    known: KnownPlant,
    modeller: ModellerConfig,
    settings: FitSettings = DEFAULT_FIT_SETTINGS,
) -> FitResult:
    """MR, the comparator: fitted on all the windows of the budget."""
    return fit_mechanistic(
        "MR", window_data(observations, split.windows), known, modeller, settings
    )


def fit_mr_f(
    split: BudgetSplit,
    observations: Observations,
    known: KnownPlant,
    modeller: ModellerConfig,
    settings: FitSettings = DEFAULT_FIT_SETTINGS,
) -> FitResult:
    """MR_F: the same procedure on the fitting part F alone. Only the windows of F are
    taken out of the observations; V is never read."""
    return fit_mechanistic(
        "MR_F", window_data(observations, split.fitting), known, modeller, settings
    )


@dataclass(frozen=True)
class Covariance:
    """The covariance of an estimate, in (ln k_350, E/R in K, ln UA), or without E/R when it
    is fixed."""

    names: tuple[str, ...]
    singular_values: tuple[float, ...]  # of the Jacobian of (prediction - reading) / sigma
    exact_initial_state: FloatArray | None  # (S^T W S)^-1: the initial state taken as exact
    sandwich: FloatArray | None  # with the error of the context mean
    reason: str | None  # why the covariance could not be computed

    def standard_errors(self, sandwich: bool = True) -> dict[str, float]:
        matrix = self.sandwich if sandwich else self.exact_initial_state
        if matrix is None:
            raise ValueError(f"no covariance: {self.reason}")
        return {name: math.sqrt(matrix[i, i]) for i, name in enumerate(self.names)}


def covariance(
    parameters: MechanisticParameters,
    windows: Sequence[WindowData],
    known: KnownPlant,
    fixed_activation: float | None = None,
    settings: RolloutSettings = EVALUATION_SETTINGS,
) -> Covariance:
    """The covariance of the weighted least-squares estimate at ``parameters``."""
    windows = tuple(windows)
    _check_windows(windows)
    model = MechanisticModel("covariance", known, parameters, fixed_activation)
    estimated = fixed_activation is None
    names = ("ln k_350", "E/R", "ln UA") if estimated else ("ln k_350", "ln UA")
    units = np.array([1.0, REFERENCE_TEMPERATURE, 1.0] if estimated else [1.0, 1.0])
    sigma = windows[0].noise_std
    context_covariance = np.diag(sigma**2) / CONTEXT_READINGS
    blocks, couplings = [], []
    for data in windows:
        result = rollout(
            model,
            data.initial_state,
            data.inputs,
            data.sample_period,
            settings,
            parameter_sensitivities=True,
            initial_state_sensitivities=True,
        )
        if isinstance(result, RolloutFailure):
            reason = f"the rollout of window {data.key} failed: {result.cause}: {result.detail}"
            return Covariance(names, (), None, None, reason)
        s = (result.parameter_sensitivities / sigma[None, :, None]).reshape(-1, len(names))
        g = (result.initial_state_sensitivities / sigma[None, :, None]).reshape(-1, 2)
        blocks.append(s)
        couplings.append(s.T @ g)  # S_w^T W G_w
    stacked = np.vstack(blocks)
    _, singular, right = np.linalg.svd(stacked, full_matrices=False)
    values = tuple(float(v) for v in singular)
    # numpy's convention for the numerical rank of a matrix, not a threshold chosen here
    if singular[-1] <= singular[0] * max(stacked.shape) * np.finfo(np.float64).eps:
        return Covariance(names, values, None, None, "the Jacobian is rank deficient")
    bread = right.T @ np.diag(1.0 / singular**2) @ right  # (S^T W S)^-1
    initial_error = sum(m @ context_covariance @ m.T for m in couplings)
    sandwich = bread + bread @ initial_error @ bread
    scale = np.outer(units, units)
    return Covariance(names, values, bread * scale, sandwich * scale, None)
