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
hybrid it bounds the departure of each factor from one everywhere, not only where the data
are: |ln g| <= sum |W_L| + |b_L|, since tanh lies in [-1, 1]. It does not hold the hybrid
near MR_F: the mechanistic parameters, trained with the networks and not penalised, can move
away from MR_F's values as far as the data pull them.

The rollout of the training. A fixed-step fourth-order Runge-Kutta scheme, ``substeps``
steps per row of the inputs, each row integrated with its own inputs. No step crosses a
change of the inputs, which happen only at the start of a row, and the scheme is the same
function for every window, so the gradient of the loss is the exact derivative of the
computed loss, by automatic differentiation. What is selected and evaluated is the
continuous-time equation, rolled out by the reference integration of ``models.rollout``: the
criterion on V at every checkpoint uses it. At each checkpoint the largest difference on V
between the two rollouts is recorded, in sigmas, as a measure of the training scheme.

The optimiser is Adam, written out below, with a constant rate. The coordinates of a hybrid
are MR's, whose fit keeps E/R >= 0 with a bound; each step of Adam is projected onto that
domain, so E/R reaches its valid limit of zero and stays there rather than leaving it. Where a
step leaves E/R positive, which is every step of the pilot of I3, the projection changes
nothing, bit for bit. The steps at which it acted are counted in the record. Checkpoints are
the step 0 and every ``validation_every`` steps up to ``max_steps``; at each, the criterion is
J on V by the reference rollout, computed as the metrics of the evaluation compute it.
Nothing stops a training early.

Failures (section 8.7). A loss or a gradient that is not finite at a step ends the training
as a training failure of that configuration, with the step and the reason; its checkpoints
are not used, since the procedure does not treat the failure. So does a state of Adam that is
not finite, its moments, their corrections, the update or the parameters it gives: a finite
gradient can overflow when it is squared, and an infinite second moment would stop every
update while the loss stayed finite. A checkpoint cannot be selected, and records why, when a
rollout of a window of V fails, when its criterion is not representable, or when its
parameters lie outside the domain of the family (``learned.domain_problem``). A scale of F
that is zero is a refusal before any step. A training that ends without failure selects the
checkpoint with the lowest finite criterion, the earliest on a tie; if there is none, it is a
training failure. The first checkpoint of a hybrid is its start, MR_F with the factors at
one, so a hybrid whose training does not improve on V ends as MR_F. At the selected
checkpoint the difference between the training scheme and the reference is also recorded on
F, where fast dynamics may lie that V does not show.

The windows of F and V together must share their length, sampling period and noise levels;
a training refuses them otherwise, before any step.

Provenance. A run that trains records ``learning_environment()``: the environment of every
run, and the versions of jax and jaxlib, the backend, the devices and the precision that
decide its numbers. Only the paths that train import this module, so JAX stays optional for
the others.

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
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from process_transfer.data.provenance import environment
from process_transfer.evaluation.metrics import normalised_rms
from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import WindowData
from process_transfer.models import learned
from process_transfer.models.mechanistic import MechanisticParameters
from process_transfer.models.rollout import EVALUATION_SETTINGS, RolloutSettings, predict_window
from process_transfer.validation import require_positive

jax.config.update("jax_enable_x64", True)


def learning_environment() -> dict[str, object]:
    """The environment of a run that trains: ``data.provenance.environment()`` and, under
    ``learning``, what decides the numbers of JAX. A version that cannot be read is recorded
    as unknown, not guessed."""
    versions: dict[str, str | None] = {}
    for name in ("jax", "jaxlib"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        **environment(),
        "learning": {
            **versions,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "x64": bool(jax.config.jax_enable_x64),
            "float_dtype": str(jnp.asarray(0.0).dtype),
        },
    }


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
    # the largest difference of the training scheme from the reference on F at the selected
    # checkpoint, in sigmas; None when there is none, or a reference rollout of F failed
    fitting_schemes_differ_in_sigmas: float | None = None
    # the steps at which the projection onto E/R >= 0 moved the parameters
    bound_steps: int = 0

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
    """J on ``windows`` by the reference rollout, the failures, and the predictions.

    J is computed as the metrics of the evaluation compute it (``metrics.normalised_rms``),
    by a scaled root mean square that neither overflows nor underflows on the way to a value
    that is representable. A J that is not representable, because the normalised errors are
    not, is recorded as a failure of the checkpoint, never returned as inf; the predictions
    are still returned."""
    predictions, failures = [], []
    for data in windows:
        found = predict_window(model, data, settings)
        if isinstance(found, np.ndarray):
            predictions.append(found)
        else:
            failures.append(f"{data.key}: {found.cause}: {found.detail}")
    if failures:
        return None, tuple(failures), None
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.concatenate([p - d.scored for p, d in zip(predictions, windows, strict=True)])
    try:
        criterion = normalised_rms(errors, windows[0].noise_std)
    except ValueError as error:
        return None, (f"the criterion on V is not representable: {error}",), np.array(predictions)
    return criterion, (), np.array(predictions)


def _largest_difference(scheme: np.ndarray, reference: np.ndarray, sigma: np.ndarray) -> float:
    """The largest difference of two rollouts in sigmas, inf when it is not representable."""
    with np.errstate(over="ignore", invalid="ignore"):
        found = np.abs(np.asarray(scheme) - reference) / sigma
    return float(np.max(found)) if np.all(np.isfinite(found)) else math.inf


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
    # the windows of F and V together share their length, sampling period and noise levels:
    # one period serves the rollouts of training on both
    _stack(fitting + validation)
    x0, inputs, scored, sigma = _stack(fitting)
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
        on_fitting: float | None = None,
        bound_steps: int = 0,
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
            fitting_schemes_differ_in_sigmas=on_fitting,
            bound_steps=bound_steps,
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
    # the lower bound of each coordinate: E/R >= 0 for a hybrid that estimates it, nothing
    # elsewhere. Each step of Adam is projected onto it, the domain MR's fit keeps with its
    # bound; where a step leaves every coordinate inside, the projection changes nothing
    lower, _ = ravel_pytree(
        {
            name: (
                np.array([-np.inf, 0.0, -np.inf])
                if name == "theta" and fixed_activation is None
                else jax.tree_util.tree_map(lambda a: np.full(np.shape(a), -np.inf), value)
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
    fitting_rollout = jax.jit(
        lambda vector: rk4_rollout(family, unravel(vector), x0, inputs, *arguments)
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
        differ = None
        try:
            model = learned.LearnedModel(
                family, family, parameters, known, scales, fixed_activation
            )
        except ValueError as error:
            # outside the declared domain: recorded, and not selectable
            score, failures, predicted = None, (f"the model is not in its domain: {error}",), None
        else:
            score, failures, predicted = validation_score(model, validation, settings.reference)
        if predicted is not None:
            differ = _largest_difference(training_rollout(vector), predicted, sigma)
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
        if score is not None and math.isfinite(score) and (best is None or score < best[0]):
            best = (score, len(checkpoints) - 1, parameters)

    m = jnp.zeros_like(flat)
    v = jnp.zeros_like(flat)
    step = 0
    failure = None
    bound_steps = 0
    quantities = (
        "the first moment",
        "the second moment",
        "the corrected first moment",
        "the corrected second moment",
        "the update",
        "the parameters",
    )
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
        update = settings.learning_rate * m_hat / (jnp.sqrt(v_hat) + settings.epsilon)
        moved = flat - update
        # a finite loss and gradient do not make the state of the optimiser finite: the
        # square of a large gradient overflows the second moment, whose infinite root then
        # stops every update while the loss stays finite
        flags = np.asarray(
            jnp.stack(
                [jnp.all(jnp.isfinite(a)) for a in (m, v, m_hat, v_hat, update, moved)]
                + [jnp.any(moved < lower)]
            )
        )
        if not np.all(flags[:-1]):
            name = quantities[int(np.argmin(flags[:-1]))]
            failure = TrainingFailure(
                f"{configuration.label}: {name} of Adam is not representable at step {step}"
            )
            break
        if flags[-1]:
            bound_steps += 1
        flat = jnp.maximum(moved, lower)
        (total, (loss, penalty)), gradient = value_and_grad(flat)
    timing["steps"] += time.perf_counter() - tick
    if failure is not None:
        return record(checkpoints, None, None, failure, step, scales, timing, None, bound_steps)
    if best is None:
        failure = TrainingFailure(
            f"{configuration.label}: no checkpoint has a criterion on V: every one has a "
            "failed rollout, a criterion that is not representable or parameters outside "
            "the domain"
        )
        return record(checkpoints, None, None, failure, step, scales, timing, None, bound_steps)
    # the scheme against the reference on F, at the selected checkpoint
    tick = time.perf_counter()
    selected = learned.LearnedModel(family, family, best[2], known, scales, fixed_activation)
    references = [predict_window(selected, d, settings.reference) for d in fitting]
    on_fitting = None
    if all(isinstance(r, np.ndarray) for r in references):
        flat_selected, _ = ravel_pytree(jax.tree_util.tree_map(jnp.asarray, best[2]))
        on_fitting = _largest_difference(
            fitting_rollout(flat_selected), np.array(references), sigma
        )
    timing["validation"] += time.perf_counter() - tick
    return record(
        checkpoints, best[1], best[2], None, step, scales, timing, on_fitting, bound_steps
    )


def select_configuration(records: Sequence[TrainingRecord]) -> int | None:
    """The position of the selected configuration: the lowest criterion on V among those
    that did not fail, the first in the declared order on a tie. None when all failed."""
    usable = [i for i, r in enumerate(records) if r.failure is None and r.criterion is not None]
    if not usable:
        return None
    return min(usable, key=lambda i: (records[i].criterion, i))
