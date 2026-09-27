"""Training of the learned models of M1 with JAX, and their selection on V
(``docs/m1_plan.md``, sections 5.5, 5.8, 8.5 and 8.7; the choice of JAX is D-035).

Importing this module sets JAX to double precision for the whole process, before any array
is made: every model of M1 is computed in doubles.

What a training reads. A training is handed the ``WindowData`` of F and of V and nothing
else, as ``fitting.fit_mechanistic`` is. F gives the loss, the scales and, through MR_F, the
start of a hybrid; V gives the criterion that chooses a checkpoint, and nothing else.

The loss is the score J^2 of section 9.1 on the scored readings of F, each window started
from the mean of its own context, plus the penalty of the configuration on the networks:

    loss = J_F^2 + lambda * (sum of the squares of every weight and bias of the networks)

The mechanistic parameters of a hybrid are not penalised. The penalty is zero only where
every weight is zero, where the factors of a hybrid are one and BN does not move. For a
hybrid it also bounds the departure from the identity everywhere, not only where the data
are: |ln g| <= sum |W_L| + |b_L|, since tanh lies in [-1, 1].

The rollout of the training. A fixed-step fourth-order Runge-Kutta scheme, ``substeps``
steps per row of the inputs, each row integrated with its own inputs. No step crosses a
change of the inputs, which happen only at the start of a row, and the scheme is the same
function for every window, so the gradient of the loss is the exact derivative of the
computed loss, by automatic differentiation. What is selected and evaluated is the
continuous-time equation, rolled out by the reference integration of ``models.rollout``: the
criterion on V at every checkpoint uses it. At each checkpoint the largest difference on V
between the two rollouts is recorded, in sigmas, as a measure of the training scheme.

The optimiser is Adam, written out below, with a constant rate. Checkpoints are the step 0
and every ``validation_every`` steps up to ``max_steps``; at each, the criterion is J on V by
the reference rollout. Nothing stops a training early.

Failures (section 8.7). A loss or a gradient that is not finite at a step ends the training
as a training failure of that configuration, with the step and the reason; its checkpoints
are not used, since the procedure does not treat the failure. A checkpoint whose rollout
of a window of V fails is recorded and cannot be selected. A scale of F that is zero is a
refusal before any step. A training that ends without failure selects the checkpoint with
the lowest criterion among those whose rollouts of V all completed, the earliest on a tie;
if there is none, it is a training failure. The first checkpoint of a hybrid is its start,
MR_F with the factors at one, so a hybrid whose training does not improve on V ends as MR_F.

Across a grid (``select_configuration``): the configuration whose selected checkpoint has
the lowest criterion on V, the first in the declared order on a tie; configurations that
failed are recorded and not selectable; if all failed, the model has a training failure at
that replicate and budget.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import WindowData
from process_transfer.models import learned
from process_transfer.models.mechanistic import MechanisticParameters
from process_transfer.models.rollout import EVALUATION_SETTINGS, RolloutSettings, predict_window
from process_transfer.validation import require_positive

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Configuration:
    """One point of the grid of a family: the hidden layers of each of its networks and the
    weight of the penalty."""

    family: str
    hidden: tuple[int, ...]
    penalty: float

    def __post_init__(self) -> None:
        if self.family not in learned.FAMILIES:
            raise ValueError(f"family must be one of {learned.FAMILIES}, got {self.family!r}")
        if not self.hidden or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.hidden
        ):
            raise ValueError(f"hidden must be positive integers, got {self.hidden!r}")
        if not (math.isfinite(self.penalty) and self.penalty >= 0.0):
            raise ValueError(f"the penalty must be finite and not negative, got {self.penalty!r}")

    @property
    def label(self) -> str:
        return f"{self.family} {'x'.join(map(str, self.hidden))} lambda={self.penalty:g}"


@dataclass(frozen=True)
class TrainingSettings:
    """Everything that decides a training besides its data, configuration and seed."""

    learning_rate: float
    max_steps: int
    validation_every: int
    substeps: int = 2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    reference: RolloutSettings = EVALUATION_SETTINGS

    def __post_init__(self) -> None:
        require_positive("learning_rate", self.learning_rate)
        for name in ("max_steps", "validation_every", "substeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.max_steps % self.validation_every:
            raise ValueError("max_steps must be a multiple of validation_every")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {value!r}")
        require_positive("epsilon", self.epsilon)


@dataclass(frozen=True)
class Checkpoint:
    step: int
    fitting_loss: float  # J_F^2 by the training rollout, without the penalty
    penalty: float
    validation: float | None  # J on V by the reference rollout; None when a rollout failed
    failures: tuple[str, ...]  # the rollouts of V that failed
    schemes_differ_in_sigmas: float | None  # largest difference of the two rollouts on V
    seconds: float = field(compare=False)


@dataclass(frozen=True)
class TrainingRecord:
    """What one training of one configuration did."""

    configuration: Configuration
    settings: TrainingSettings
    seed: int
    fixed_activation: float | None
    fitting_windows: tuple[tuple[str, int], ...]
    validation_windows: tuple[tuple[str, int], ...]
    scales: learned.Scales | None
    network_size: int
    checkpoints: tuple[Checkpoint, ...]
    selected: int | None  # position in checkpoints
    parameters: dict[str, Any] | None  # of the selected checkpoint, numpy
    failure: TrainingFailure | None
    steps_run: int
    seconds: dict[str, float] = field(compare=False)

    @property
    def criterion(self) -> float | None:
        return None if self.selected is None else self.checkpoints[self.selected].validation

    def model(self, known: KnownPlant, name: str | None = None) -> learned.LearnedModel:
        if self.parameters is None or self.scales is None:
            raise ValueError(f"{self.configuration.label} has no model: {self.failure}")
        return learned.LearnedModel(
            name or self.configuration.family,
            self.configuration.family,
            self.parameters,
            known,
            self.scales,
            self.fixed_activation,
        )


def training_seed(base: int, replicate: int, configuration: int) -> int:
    """The seed of the initial weights of one configuration on one replicate: the same at
    every budget, so that the budgets of a replicate start from the same weights."""
    for name, value in (("base", base), ("replicate", replicate), ("configuration", configuration)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(np.random.SeedSequence([base, replicate, configuration]).generate_state(1)[0])


def _stack(windows: Sequence[WindowData]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = {len(d.scored) for d in windows}
    periods = {d.sample_period for d in windows}
    sigmas = {tuple(d.noise_std) for d in windows}
    if len(lengths) != 1 or len(periods) != 1 or len(sigmas) != 1:
        raise ValueError(
            "the windows of a training must share their length, sampling period and noise "
            f"levels; got lengths {sorted(lengths)}, periods {sorted(periods)}, noise {sigmas}"
        )
    sigma = np.array(windows[0].noise_std, dtype=np.float64)
    if not np.all(sigma > 0.0):
        raise ValueError(f"the loss divides by the noise levels, and they are {sigma.tolist()}")
    return (
        np.array([d.initial_state for d in windows]),
        np.array([d.inputs for d in windows]),
        np.array([d.scored for d in windows]),
        sigma,
    )


def rk4_rollout(
    family: str,
    parameters: dict[str, Any],
    x0: Any,
    inputs: Any,
    period: float,
    substeps: int,
    known: KnownPlant,
    scales: learned.Scales,
    fixed_activation: float | None,
) -> Any:
    """The states at the ends of the rows, for windows x0 (W, 2) and inputs (W, L, 4), by
    ``substeps`` steps of the classical Runge-Kutta scheme per row."""
    h = period / substeps

    def f(x: Any, u: Any) -> Any:
        return learned.rhs(jnp, family, parameters, x, u, known, scales, fixed_activation)

    def row(x: Any, u: Any) -> tuple[Any, Any]:
        for _ in range(substeps):
            k1 = f(x, u)
            k2 = f(x + 0.5 * h * k1, u)
            k3 = f(x + 0.5 * h * k2, u)
            k4 = f(x + h * k3, u)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x, x

    _, states = jax.lax.scan(row, x0, jnp.swapaxes(inputs, 0, 1))
    return jnp.swapaxes(states, 0, 1)


def _penalty_of(parameters: dict[str, Any]) -> Any:
    return sum(
        jnp.sum(w**2) + jnp.sum(b**2)
        for name, layers in parameters.items()
        if name != "theta"
        for w, b in layers
    )


def validation_score(
    model: learned.LearnedModel,
    windows: Sequence[WindowData],
    settings: RolloutSettings = EVALUATION_SETTINGS,
) -> tuple[float | None, tuple[str, ...], np.ndarray | None]:
    """J on ``windows`` by the reference rollout, the failures, and the predictions."""
    predictions, failures = [], []
    for data in windows:
        found = predict_window(model, data, settings)
        if isinstance(found, np.ndarray):
            predictions.append(found)
        else:
            failures.append(f"{data.key}: {found.cause}: {found.detail}")
    if failures:
        return None, tuple(failures), None
    sigma = windows[0].noise_std
    residuals = np.array(
        [(p - d.scored) / sigma for p, d in zip(predictions, windows, strict=True)]
    )
    return math.sqrt(float(np.mean(residuals**2))), (), np.array(predictions)


def train(
    configuration: Configuration,
    settings: TrainingSettings,
    fitting: Sequence[WindowData],
    validation: Sequence[WindowData],
    known: KnownPlant,
    seed: int,
    start: MechanisticParameters | None = None,
    fixed_activation: float | None = None,
    hold_mechanistic: bool = False,
) -> TrainingRecord:
    """Train one configuration on F and choose its checkpoint on V. ``hold_mechanistic``
    keeps the mechanistic parameters of a hybrid at ``start`` and trains its networks alone,
    as the test of recovery of a planted correction does; the procedure of M1 trains them
    together."""
    began = time.perf_counter()
    family = configuration.family
    fitting, validation = tuple(fitting), tuple(validation)
    if not fitting or not validation:
        raise ValueError("a training needs windows of F and of V")
    keys = [d.key for d in fitting + validation]
    if len(set(keys)) != len(keys):
        raise ValueError(f"F and V must be disjoint sets of windows, got {keys}")
    x0, inputs, scored, sigma = _stack(fitting)
    if tuple(validation[0].noise_std) != tuple(sigma):
        raise ValueError("F and V must share the noise levels of their sensors")
    period = fitting[0].sample_period
    rng = np.random.default_rng(seed)
    initial = learned.initial_parameters(family, configuration.hidden, rng, start, fixed_activation)
    size = learned.network_size(initial)

    def record(
        checkpoints: Sequence[Checkpoint],
        selected: int | None,
        parameters: dict[str, Any] | None,
        failure: TrainingFailure | None,
        steps: int,
        scales: learned.Scales | None,
        timing: dict[str, float],
    ) -> TrainingRecord:
        timing["total"] = time.perf_counter() - began
        return TrainingRecord(
            configuration=configuration,
            settings=settings,
            seed=seed,
            fixed_activation=fixed_activation,
            fitting_windows=tuple(d.key for d in fitting),
            validation_windows=tuple(d.key for d in validation),
            scales=scales,
            network_size=size,
            checkpoints=tuple(checkpoints),
            selected=selected,
            parameters=parameters,
            failure=failure,
            steps_run=steps,
            seconds=timing,
        )

    timing = {"compile": 0.0, "steps": 0.0, "validation": 0.0}
    try:
        scales = learned.fitting_scales(fitting)
    except ValueError as error:
        return record((), None, None, TrainingFailure(f"refused scale: {error}"), 0, None, timing)

    flat, unravel = ravel_pytree(jax.tree_util.tree_map(jnp.asarray, initial))
    if hold_mechanistic and family == learned.BN:
        raise ValueError("BN has no mechanistic parameters to hold")
    # 1 where a parameter is trained, 0 where it is held
    trained, _ = ravel_pytree(
        {
            name: (
                np.zeros_like(value)
                if name == "theta" and hold_mechanistic
                else jax.tree_util.tree_map(np.ones_like, value)
            )
            for name, value in initial.items()
        }
    )
    vx0, vinputs, _, _ = _stack(validation)
    arguments = (period, settings.substeps, known, scales, fixed_activation)

    def fitting_loss(vector: Any) -> Any:
        parameters = unravel(vector)
        predicted = rk4_rollout(family, parameters, x0, inputs, *arguments)
        return jnp.mean(((predicted - scored) / sigma) ** 2)

    def objective(vector: Any) -> tuple[Any, Any]:
        loss = fitting_loss(vector)
        penalty = configuration.penalty * _penalty_of(unravel(vector))
        return loss + penalty, (loss, penalty)

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    training_rollout = jax.jit(
        lambda vector: rk4_rollout(family, unravel(vector), vx0, vinputs, *arguments)
    )

    tick = time.perf_counter()
    (total, (loss, penalty)), gradient = value_and_grad(flat)
    jax.block_until_ready(gradient)
    training_rollout(flat).block_until_ready()
    timing["compile"] = time.perf_counter() - tick

    checkpoints: list[Checkpoint] = []
    best: tuple[float, int, dict[str, Any]] | None = None

    def checkpoint(step: int, vector: Any, loss: Any, penalty: Any) -> None:
        nonlocal best
        tick = time.perf_counter()
        parameters = learned.to_numpy(unravel(vector))
        model = learned.LearnedModel(family, family, parameters, known, scales, fixed_activation)
        score, failures, predicted = validation_score(model, validation, settings.reference)
        differ = None
        if predicted is not None:
            scheme = np.asarray(training_rollout(vector))
            differ = (
                float(np.max(np.abs(scheme - predicted) / sigma))
                if np.all(np.isfinite(scheme))
                else math.inf
            )
        checkpoints.append(
            Checkpoint(
                step,
                float(loss),
                float(penalty),
                score,
                failures,
                differ,
                time.perf_counter() - tick,
            )
        )
        timing["validation"] += time.perf_counter() - tick
        if score is not None and (best is None or score < best[0]):
            best = (score, len(checkpoints) - 1, parameters)

    m = jnp.zeros_like(flat)
    v = jnp.zeros_like(flat)
    step = 0
    failure = None
    tick = time.perf_counter()
    while True:
        if not (np.isfinite(float(total)) and bool(jnp.all(jnp.isfinite(gradient)))):
            failure = TrainingFailure(
                f"{configuration.label}: the loss or its gradient is not finite at step {step} "
                f"(loss {float(total)!r})"
            )
            break
        if step % settings.validation_every == 0:
            timing["steps"] += time.perf_counter() - tick
            checkpoint(step, flat, loss, penalty)
            tick = time.perf_counter()
        if step == settings.max_steps:
            break
        step += 1
        gradient = gradient * trained
        m = settings.beta1 * m + (1.0 - settings.beta1) * gradient
        v = settings.beta2 * v + (1.0 - settings.beta2) * gradient**2
        m_hat = m / (1.0 - settings.beta1**step)
        v_hat = v / (1.0 - settings.beta2**step)
        flat = flat - settings.learning_rate * m_hat / (jnp.sqrt(v_hat) + settings.epsilon)
        (total, (loss, penalty)), gradient = value_and_grad(flat)
    timing["steps"] += time.perf_counter() - tick
    if failure is not None:
        return record(checkpoints, None, None, failure, step, scales, timing)
    if best is None:
        failure = TrainingFailure(
            f"{configuration.label}: every checkpoint has a rollout of V that failed"
        )
        return record(checkpoints, None, None, failure, step, scales, timing)
    return record(checkpoints, best[1], best[2], None, step, scales, timing)


def select_configuration(records: Sequence[TrainingRecord]) -> int | None:
    """The position of the selected configuration: the lowest criterion on V among those
    that did not fail, the first in the declared order on a tie. None when all failed."""
    usable = [i for i, r in enumerate(records) if r.failure is None and r.criterion is not None]
    if not usable:
        return None
    return min(usable, key=lambda i: (records[i].criterion, i))
