"""The interface of a continuous-time model, and its rollout under piecewise-constant inputs.

A model of M1 is an equation dx/dt = f(x, u) for the states x = (C_A, T) under the inputs
u = (q, C_Af, T_f, T_c), in SI; time does not enter it. A rollout starts from an initial
state and is driven by the inputs of the rows of a window, each applied from its own
instant until the next row (zero-order hold, right-continuous): row i acts over
[i h, (i + 1) h) after the onset, h the sampling period. It returns the states at the
instants h, 2 h, ..., L h, the scored instants of the window, and nothing else. It is given
no reading: the initial state is computed by the evaluation contract from the context, and
no later measurement is within its reach.

The integration restarts at every change of the inputs, one ``solve_ivp`` call per run of
equal rows, so that no step of the integrator crosses a discontinuity; each piece starts
from the state at which the previous one ended. ``simulation.integration.simulate_piecewise``
does the same for the truth, and the models may not import it. Its loop is written again
here rather than moved to a neutral module (D-032): a rollout must return failures as
records instead of raising, limit the evaluations of the right-hand side, sample only the
sensor instants and integrate sensitivities, none of which the simulation needs, and the
code of M0 stays as it was audited.

Failures. A rollout that does not complete is returned as a ``RolloutFailure`` with its
cause, never as numbers: the initial state is outside the domain of the model; the
right-hand side raised an arithmetic error or returned a derivative that is not finite; the
integrator reported a failure; a state stopped being finite; or the limit on evaluations of
the right-hand side was reached. The last two are not redundant: LSODA given a right-hand
side that returns infinity, or a solution that grows without bound, does not stop by itself
(reproduced while writing this module: more than 100 000 evaluations on either). Nothing is
clipped, and no state is replaced by another.

Sensitivities. For a model that states its Jacobians, the rollout can also integrate the
sensitivities of the states to the parameters, dS/dt = A S + B with S(0) = 0, and to the
initial state, dG/dt = A G with G(0) = I, where A = df/dx and B = df/dtheta. They are
integrated with the states, under the same error control, and serve the Jacobian of a fit
and the covariance of its estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from process_transfer.cstr_variables import INPUT_NAMES, STATE_NAMES, FloatArray
from process_transfer.evaluation.outcomes import IntegrationFailure
from process_transfer.evaluation.windows import WindowData
from process_transfer.validation import require_positive

N_STATES = len(STATE_NAMES)


class ContinuousTimeModel(Protocol):
    """dx/dt = f(x, u), in SI."""

    name: str

    def rhs(self, x: FloatArray, u: FloatArray) -> FloatArray: ...

    def initial_state_problem(self, x0: FloatArray) -> str | None:
        """Why the model cannot start from ``x0``, or None when it can."""
        ...


class DifferentiableModel(ContinuousTimeModel, Protocol):
    """A model that states the derivatives of its right-hand side."""

    parameter_names: tuple[str, ...]

    def jacobian_state(self, x: FloatArray, u: FloatArray) -> FloatArray:
        """df/dx, shape (2, 2)."""
        ...

    def jacobian_parameters(self, x: FloatArray, u: FloatArray) -> FloatArray:
        """df/dtheta, shape (2, number of parameters)."""
        ...


@dataclass(frozen=True)
class RolloutSettings:
    """How a rollout is integrated.

    The defaults are the reference integration of the evaluation. rtol and atol control the
    local error of every component, states and sensitivities alike, atol in the SI unit of
    each; with them the error of the states against a far tighter integration is checked to
    stay below 1 % of the sigmas of the sensors (``tests/test_models_rollout.py``, and on a
    sample of real windows in the smoke run of I1). ``max_rhs_evaluations`` is a guard for a
    window, not a tolerance: a window of 660 s takes a few hundred evaluations, and the
    guard only stops a rollout that no longer makes progress.
    """

    method: str = "LSODA"
    rtol: float = 1.0e-8
    atol: float = 1.0e-6
    max_rhs_evaluations: int = 100_000

    def __post_init__(self) -> None:
        # scipy replaces a tolerance it finds too small and only warns; refuse it here
        require_positive("rtol", self.rtol)
        require_positive("atol", self.atol)
        limit = self.max_rhs_evaluations
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"max_rhs_evaluations must be a positive integer, got {limit!r}")


EVALUATION_SETTINGS = RolloutSettings()


@dataclass(frozen=True)
class RolloutFailure:
    """A rollout that did not complete, and why."""

    cause: str  # "initial state", "right-hand side", "integrator", "non-finite state", "work limit"
    detail: str

    def for_window(self, window: tuple[str, int]) -> IntegrationFailure:
        return IntegrationFailure(window=window, cause=self.cause, detail=self.detail)


@dataclass(frozen=True)
class Rollout:
    """The states at the scored instants, with their sensitivities when asked for."""

    states: FloatArray  # (L, 2)
    rhs_evaluations: int
    parameter_sensitivities: FloatArray | None  # (L, 2, number of parameters)
    initial_state_sensitivities: FloatArray | None  # (L, 2, 2)


class _Stop(Exception):
    """Raised inside the right-hand side to end a rollout; carries its record."""

    def __init__(self, failure: RolloutFailure) -> None:
        super().__init__(failure.detail)
        self.failure = failure


def _segments(inputs: FloatArray) -> list[tuple[int, int]]:
    """Runs of equal consecutive rows, as (first row, row after the last)."""
    changes = np.flatnonzero(np.any(inputs[1:] != inputs[:-1], axis=1)) + 1
    bounds = [0, *changes.tolist(), len(inputs)]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def rollout(
    model: ContinuousTimeModel,
    initial_state: FloatArray,
    inputs: FloatArray,
    sample_period: float,
    settings: RolloutSettings = EVALUATION_SETTINGS,
    parameter_sensitivities: bool = False,
    initial_state_sensitivities: bool = False,
) -> Rollout | RolloutFailure:
    """Integrate ``model`` from ``initial_state`` under ``inputs``, one row per sampling
    period, and return the states at the ends of the periods, or the failure."""
    x0 = np.array(initial_state, dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.float64)
    if x0.shape != (N_STATES,) or not np.all(np.isfinite(x0)):
        raise ValueError(f"the initial state must be {N_STATES} finite values, got {x0!r}")
    if inputs.ndim != 2 or inputs.shape[1] != len(INPUT_NAMES) or len(inputs) == 0:
        raise ValueError(f"inputs must have one row per period and {len(INPUT_NAMES)} columns")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("inputs must be finite")
    h = require_positive("sample_period", sample_period)
    if not math.isfinite(h * len(inputs)):
        raise ValueError(
            f"with sample_period = {h!r} s the {len(inputs)} periods of the inputs end beyond "
            "the largest double; the instants of the rollout are not representable"
        )
    problem = model.initial_state_problem(x0)
    if problem is not None:
        return RolloutFailure("initial state", problem)

    n_parameters = 0
    if parameter_sensitivities or initial_state_sensitivities:
        if not all(hasattr(model, name) for name in ("jacobian_state", "jacobian_parameters")):
            raise ValueError(f"model {model.name!r} states no Jacobians; no sensitivities")
    if parameter_sensitivities:
        n_parameters = len(model.parameter_names)  # type: ignore[attr-defined]
    size = N_STATES * (1 + n_parameters + (N_STATES if initial_state_sensitivities else 0))
    state = np.zeros(size)
    state[:N_STATES] = x0
    if initial_state_sensitivities:
        state[size - N_STATES * N_STATES :] = np.eye(N_STATES).ravel()

    evaluations = 0
    rows = len(inputs)
    samples = np.empty((rows, size))
    for index, (first, stop) in enumerate(_segments(inputs)):
        u = inputs[first]

        def derivative(t: float, y: FloatArray, u: FloatArray = u) -> FloatArray:
            nonlocal evaluations
            evaluations += 1
            if evaluations > settings.max_rhs_evaluations:
                raise _Stop(
                    RolloutFailure(
                        "work limit",
                        f"{settings.max_rhs_evaluations} evaluations of the right-hand side "
                        f"reached at t = {t!r} s after the onset",
                    )
                )
            return _evaluate(model, y, u, t, n_parameters, initial_state_sensitivities)

        instants = h * np.arange(first + 1, stop + 1, dtype=np.float64)
        try:
            solution = solve_ivp(
                derivative,
                (h * first, h * stop),
                state,
                method=settings.method,
                t_eval=instants,
                rtol=settings.rtol,
                atol=settings.atol,
            )
        except _Stop as stopped:
            return stopped.failure
        if not solution.success:
            return RolloutFailure(
                "integrator", f"segment {index} ({settings.method}): {solution.message}"
            )
        values = solution.y.T
        if values.shape != (stop - first, size) or not np.all(np.isfinite(values)):
            return RolloutFailure(
                "non-finite state", f"segment {index}: a state or sensitivity is not finite"
            )
        samples[first:stop] = values
        state = values[-1].copy()

    states = samples[:, :N_STATES].copy()
    sensitivities = None
    if parameter_sensitivities:
        block = samples[:, N_STATES : N_STATES * (1 + n_parameters)]
        sensitivities = block.reshape(rows, N_STATES, n_parameters).copy()
    initial = None
    if initial_state_sensitivities:
        block = samples[:, size - N_STATES * N_STATES :]
        initial = block.reshape(rows, N_STATES, N_STATES).copy()
    return Rollout(states, evaluations, sensitivities, initial)


def _evaluate(
    model: ContinuousTimeModel,
    y: FloatArray,
    u: FloatArray,
    t: float,
    n_parameters: int,
    initial_state_sensitivities: bool,
) -> FloatArray:
    """The derivative of the augmented state, or a stop with the reason it has none."""
    x = y[:N_STATES]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            parts = [np.asarray(model.rhs(x, u), dtype=np.float64)]
            if n_parameters or initial_state_sensitivities:
                a = model.jacobian_state(x, u)  # type: ignore[attr-defined]
            if n_parameters:
                s = y[N_STATES : N_STATES * (1 + n_parameters)].reshape(N_STATES, n_parameters)
                b = model.jacobian_parameters(x, u)  # type: ignore[attr-defined]
                parts.append((a @ s + b).ravel())
            if initial_state_sensitivities:
                g = y[-N_STATES * N_STATES :].reshape(N_STATES, N_STATES)
                parts.append((a @ g).ravel())
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise _Stop(
            RolloutFailure(
                "right-hand side",
                f"{type(error).__name__} ({error}) at t = {t!r} s after the onset, "
                f"x = {x.tolist()}",
            )
        ) from None
    derivative = np.concatenate(parts)
    if not np.all(np.isfinite(derivative)):
        raise _Stop(
            RolloutFailure(
                "right-hand side",
                f"a derivative that is not finite at t = {t!r} s after the onset, x = {x.tolist()}",
            )
        )
    return derivative


def predict_window(
    model: ContinuousTimeModel, data: WindowData, settings: RolloutSettings = EVALUATION_SETTINGS
) -> FloatArray | IntegrationFailure:
    """The prediction of a window: its initial state and its inputs, and nothing else of it."""
    result = rollout(model, data.initial_state, data.inputs, data.sample_period, settings)
    if isinstance(result, RolloutFailure):
        return result.for_window(data.key)
    return result.states
