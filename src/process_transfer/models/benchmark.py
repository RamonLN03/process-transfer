"""The development runs of M1-E03, task by task (``docs/experiment_log.md``, M1-E03,
sections 2 to 5, 8 and 9; D-042).

What is here is the registered procedure on windows already read:

* the specification of the development runs: sets, budgets, grids, rates, steps, the
  steps of the scheme per row, seeds and the analysis with E/R held, with a digest that
  identifies it;
* the tasks that a condition calls for, every one with its key, data and seed;
* one task run on its windows: MR on all b windows, MR_F on F, BL on all b windows, or the
  training of one configuration on F with its checkpoints chosen on V;
* the selection of a configuration within a family, on V only, and its effort;
* the accounting that makes every expected task an outcome, a failure included;
* the records from which a fitted MR, MR_F or BL is rebuilt; a learned model is written by
  ``persistence``.

It reads no file, no configuration of a plant, no test set and nothing of the truth. The
script of the development runs reads the exports of the development sets and hands their
windows in (``experiments/m1_e03_development.py``). The values are those approved in
D-042; a smaller specification serves the tests only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from process_transfer.data.identifiers import run_identifier
from process_transfer.evaluation.budgets import split_budget
from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import RunWindows, WindowData
from process_transfer.models.fitting import (
    DEFAULT_STARTS,
    STARTS_WITH_FIXED_ACTIVATION,
    FitResult,
    FitSettings,
    fit_mechanistic,
)
from process_transfer.models.linear import (
    InputDirections,
    LinearCoordinates,
    LinearFit,
    LinearModel,
    LinearParameters,
    fit_linear,
)
from process_transfer.models.mechanistic import MechanisticModel, MechanisticParameters
from process_transfer.models.training import (
    Configuration,
    TrainingRecord,
    TrainingSettings,
    select_configuration,
    train,
    training_seed,
)

# the two analyses; "held" is the analysis with E/R held, a word that is safe in a key
PRIMARY, HELD = "primary", "held"
MECHANISTIC = ("MR_F", "MR", "BL")
FAMILIES = ("BN", "HK", "HU", "HKU")


@dataclass(frozen=True)
class DevelopmentSet:
    """A development set as its registered definition makes it, without its private seed."""

    dataset_id: str
    excitation_seeds: tuple[int, ...]
    amplitude: str
    n_excursions: int
    lead_s: int


@dataclass(frozen=True)
class Specification:
    """Everything that decides the development runs besides their data."""

    plant: str
    sets: Mapping[str, DevelopmentSet]  # condition -> set
    budgets: Mapping[str, tuple[int, ...]]  # condition -> budgets
    grid: Mapping[str, tuple[Configuration, ...]]  # family -> configurations, declared order
    rates: Mapping[str, float]  # family -> rate of Adam
    steps: int
    validation_every: int
    substeps: int
    seed_bases: Mapping[str, int]  # condition -> base of the initial weights
    held_activation: float  # E/R in K for the analysis with E/R held
    held_condition: str
    held_budgets: tuple[int, ...]
    held_families: tuple[str, ...]  # the learned families trained with E/R held

    def __post_init__(self) -> None:
        if set(self.sets) != set(self.budgets) or set(self.sets) != set(self.seed_bases):
            raise ValueError("every condition needs a set, budgets and a base of seeds")
        if set(self.grid) != set(self.rates):
            raise ValueError("every family of the grid needs a rate, and only those")
        if self.held_condition not in self.sets:
            raise ValueError(f"the held condition {self.held_condition!r} is not a condition")
        if not set(self.held_budgets) <= set(self.budgets[self.held_condition]):
            raise ValueError("the budgets with E/R held must be budgets of their condition")
        if not set(self.held_families) <= set(self.grid):
            raise ValueError("the families with E/R held must be families of the grid")
        for family, configurations in self.grid.items():
            if not configurations or any(c.family != family for c in configurations):
                raise ValueError(f"the grid of {family} must list configurations of {family}")
        self.training_settings(next(iter(self.grid)))  # checks steps, every and substeps

    def training_settings(self, family: str) -> TrainingSettings:
        """The settings of a training, every one given explicitly: the default number of
        steps of the scheme per row is not used."""
        return TrainingSettings(
            learning_rate=self.rates[family],
            max_steps=self.steps,
            validation_every=self.validation_every,
            substeps=self.substeps,
        )

    def run_ids(self, condition: str) -> tuple[str, ...]:
        """The runs of a condition's set, in the order of their excitation seeds: replicate r
        is the run of the r-th seed."""
        s = self.sets[condition]
        return tuple(
            run_identifier(
                self.plant, "p3", seed, s.n_excursions, 0, lead_s=s.lead_s, amplitude=s.amplitude
            )
            for seed in s.excitation_seeds
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "plant": self.plant,
            "sets": {k: dataclasses.asdict(v) for k, v in sorted(self.sets.items())},
            "budgets": {k: list(v) for k, v in sorted(self.budgets.items())},
            "grid": {k: [dataclasses.asdict(c) for c in v] for k, v in sorted(self.grid.items())},
            "rates": dict(sorted(self.rates.items())),
            "steps": self.steps,
            "validation_every": self.validation_every,
            "substeps": self.substeps,
            "seed_bases": dict(sorted(self.seed_bases.items())),
            "held_activation": self.held_activation,
            "held_condition": self.held_condition,
            "held_budgets": list(self.held_budgets),
            "held_families": list(self.held_families),
        }

    def digest(self) -> str:
        """The SHA-256 of the canonical JSON of the specification: runs made under another
        specification are never combined with these."""
        text = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


def _grid(family: str, sizes: Sequence[tuple[int, ...]]) -> tuple[Configuration, ...]:
    return tuple(Configuration(family, size, penalty) for size in sizes for penalty in (1e-4, 1e-2))


REGISTERED = Specification(
    plant="target",
    sets={
        "A10": DevelopmentSet("m1-e03-train-a10", tuple(range(31001, 31011)), "a10", 40, 60),
        "A5": DevelopmentSet("m1-e03-train-a5", tuple(range(32001, 32011)), "a5", 40, 60),
    },
    budgets={"A10": (2, 5, 10, 20, 40), "A5": (10, 40)},
    grid={
        "BN": _grid("BN", ((16, 16), (32, 32))),
        "HK": _grid("HK", ((8,), (16,))),
        "HU": _grid("HU", ((8,), (16,))),
        "HKU": _grid("HKU", ((8,), (16,))),
    },
    rates={"BN": 1e-3, "HK": 1e-2, "HU": 1e-3, "HKU": 1e-2},
    steps=3000,
    validation_every=100,
    substeps=4,
    seed_bases={"A10": 20261111, "A5": 20261112},
    held_activation=8750.0,
    held_condition="A10",
    held_budgets=(5, 40),
    held_families=("HK",),
)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Task:
    """One fit or training of the development runs."""

    kind: str  # "MR_F", "MR", "BL" or "training"
    condition: str
    replicate: int
    run_id: str
    budget: int
    analysis: str  # PRIMARY or HELD
    family: str | None = None
    index: int | None = None  # position in the declared grid of the family
    configuration: Configuration | None = None
    seed: int | None = None

    @property
    def key(self) -> str:
        head = f"{self.condition}/r{self.replicate:02d}/b{self.budget:02d}/{self.analysis}"
        if self.kind == "training":
            return f"{head}/{self.family}/{self.index}"
        return f"{head}/{self.kind}"

    @property
    def start_key(self) -> str:
        """The key of the MR_F a hybrid starts from: of its own replicate, budget and
        analysis."""
        return f"{self.condition}/r{self.replicate:02d}/b{self.budget:02d}/{self.analysis}/MR_F"


def tasks(spec: Specification, condition: str) -> tuple[list[Task], list[Task]]:
    """The fits, then the trainings, that a condition calls for. The trainings of hybrids
    need the MR_F of their replicate and budget, so the fits come first."""
    if condition not in spec.sets:
        raise ValueError(f"condition must be one of {sorted(spec.sets)}, got {condition!r}")
    fits: list[Task] = []
    trainings: list[Task] = []
    base = spec.seed_bases[condition]
    for replicate, run_id in enumerate(spec.run_ids(condition)):
        for budget in spec.budgets[condition]:
            analyses = [PRIMARY]
            if condition == spec.held_condition and budget in spec.held_budgets:
                analyses.append(HELD)
            for analysis in analyses:
                kinds = MECHANISTIC if analysis == PRIMARY else ("MR_F", "MR")
                for kind in kinds:
                    fits.append(Task(kind, condition, replicate, run_id, budget, analysis))
                families = spec.grid if analysis == PRIMARY else spec.held_families
                for family in families:
                    for index, configuration in enumerate(spec.grid[family]):
                        trainings.append(
                            Task(
                                "training",
                                condition,
                                replicate,
                                run_id,
                                budget,
                                analysis,
                                family,
                                index,
                                configuration,
                                training_seed(base, replicate, index),
                            )
                        )
    return fits, trainings


def parts(
    run_windows: RunWindows, data: Sequence[WindowData], budget: int
) -> tuple[tuple[WindowData, ...], tuple[WindowData, ...]]:
    """F and V of a budget, as ``evaluation.budgets.split_budget`` divides it."""
    split = split_budget(run_windows, budget)
    by_key = {d.key: d for d in data}
    return (
        tuple(by_key[w.key] for w in split.fitting),
        tuple(by_key[w.key] for w in split.validation),
    )


# --------------------------------------------------------------------------- #
# Records of fitted models
# --------------------------------------------------------------------------- #


def mechanistic_record(parameters: MechanisticParameters | None) -> dict[str, float] | None:
    return None if parameters is None else dataclasses.asdict(parameters)


def mechanistic_model(name: str, known: KnownPlant, record: dict[str, float]) -> MechanisticModel:
    return MechanisticModel(name, known, MechanisticParameters(**record))


def linear_record(fit: LinearFit) -> dict[str, Any] | None:
    """What rebuilds BL: its excited directions, coordinates and parameters, exactly."""
    if fit.parameters is None:
        return None
    d, c, p = fit.directions, fit.coordinates, fit.parameters
    return {
        "directions": {
            "nominal": d.nominal.tolist(),
            "amplitudes": d.amplitudes.tolist(),
            "basis": d.basis.tolist(),
            "rank": d.rank,
            "levels": [[str(v) for v in row] for row in d.levels],
        },
        "coordinates": {
            "rank": c.rank,
            "sigma": c.sigma.tolist(),
            "origin": c.origin.tolist(),
            "time": c.time,
        },
        "parameters": {
            "a": p.a.tolist(),
            "equilibrium": p.equilibrium.tolist(),
            "c": p.c.tolist(),
        },
    }


def linear_model(record: dict[str, Any]) -> LinearModel:
    d, c, p = record["directions"], record["coordinates"], record["parameters"]
    directions = InputDirections(
        np.array(d["nominal"], dtype=np.float64),
        np.array(d["amplitudes"], dtype=np.float64),
        np.array(d["basis"], dtype=np.float64).reshape(4, d["rank"]),
        d["rank"],
        tuple(tuple(Fraction(v) for v in row) for row in d["levels"]),
    )
    coordinates = LinearCoordinates(
        c["rank"], np.array(c["sigma"], dtype=np.float64), np.array(c["origin"]), c["time"]
    )
    parameters = LinearParameters(
        np.array(p["a"], dtype=np.float64),
        np.array(p["equilibrium"], dtype=np.float64),
        np.array(p["c"], dtype=np.float64).reshape(2, d["rank"]),
    )
    return LinearModel("BL", directions, parameters, coordinates)


# --------------------------------------------------------------------------- #
# Running a task
# --------------------------------------------------------------------------- #


def _failure(found: TrainingFailure | None) -> str | None:
    return None if found is None else found.reason


def run_fit(
    task: Task,
    fitting: Sequence[WindowData],
    validation: Sequence[WindowData],
    known: KnownPlant,
    modeller: Any,
    spec: Specification,
) -> dict[str, Any]:
    """MR_F on F alone, MR and BL on all b windows, with E/R held where the task says. MR
    and BL select nothing; MR_F reads no window of V."""
    held = task.analysis == HELD
    settings = (
        FitSettings(
            fixed_activation_temperature=spec.held_activation, starts=STARTS_WITH_FIXED_ACTIVATION
        )
        if held
        else FitSettings(starts=DEFAULT_STARTS)
    )
    windows = tuple(fitting) if task.kind == "MR_F" else tuple(fitting) + tuple(validation)
    began = time.perf_counter()
    if task.kind == "BL":
        found: FitResult | LinearFit = fit_linear("BL", windows, known.nominal_inputs)
        model = linear_record(found)
        extra: dict[str, Any] = {"rank": found.directions.rank}
    else:
        found = fit_mechanistic(task.kind, windows, known, modeller, settings)
        model = mechanistic_record(found.parameters)
        extra = {"fixed_activation": settings.fixed_activation_temperature}
    return {
        "key": task.key,
        "kind": task.kind,
        "condition": task.condition,
        "replicate": task.replicate,
        "run": task.run_id,
        "budget": task.budget,
        "analysis": task.analysis,
        "windows": [list(k) for k in found.windows],
        "model": model,
        "objective": found.objective,
        "failure": _failure(found.training_failure),
        "starts": [
            {"outcome": s.outcome, "objective": s.objective, "evaluations": s.evaluations}
            for s in found.starts
        ],
        "seconds": time.perf_counter() - began,
        **extra,
    }


def blocked(task: Task, reason: str) -> dict[str, Any]:
    """A training that cannot run because its start, MR_F, failed: a training failure,
    recorded like any other and not retried."""
    return {
        **_training_head(task),
        "failure": f"{task.configuration.label}: not trained, since its start failed: {reason}",
        "trained": False,
        "selected_step": None,
        "criterion": None,
        "checkpoints": [],
        "model_file": None,
        "model_sha256": None,
    }


def _training_head(task: Task) -> dict[str, Any]:
    return {
        "key": task.key,
        "kind": "training",
        "condition": task.condition,
        "replicate": task.replicate,
        "run": task.run_id,
        "budget": task.budget,
        "analysis": task.analysis,
        "family": task.family,
        "index": task.index,
        "configuration": dataclasses.asdict(task.configuration),
        "seed": task.seed,
    }


def run_training(
    task: Task,
    fitting: Sequence[WindowData],
    validation: Sequence[WindowData],
    known: KnownPlant,
    start: MechanisticParameters | None,
    spec: Specification,
) -> tuple[dict[str, Any], TrainingRecord]:
    """One configuration trained on F, its checkpoint chosen on V by the reference rollout,
    with no refit afterwards. A hybrid starts from ``start``, MR_F of its replicate and
    budget; BN has none. A failure is recorded as it happened and not retried."""
    held = task.analysis == HELD
    began = time.perf_counter()
    record = train(
        task.configuration,
        spec.training_settings(task.family),
        fitting,
        validation,
        known,
        task.seed,
        start=start,
        fixed_activation=spec.held_activation if held else None,
    )
    found = {
        **_training_head(task),
        "failure": _failure(record.failure),
        "trained": True,
        "settings": dataclasses.asdict(record.settings),
        "fitting_windows": [list(k) for k in record.fitting_windows],
        "validation_windows": [list(k) for k in record.validation_windows],
        "start": mechanistic_record(start),
        "network_size": record.network_size,
        "selected_step": None
        if record.selected is None
        else record.checkpoints[record.selected].step,
        "criterion": record.criterion,
        "checkpoints": [
            {
                "step": c.step,
                "fitting": c.fitting_loss,
                "penalty": c.penalty,
                "validation": c.validation,
                "failures": list(c.failures),
                "schemes_differ_in_sigmas": c.schemes_differ_in_sigmas,
            }
            for c in record.checkpoints
        ],
        "fitting_schemes_differ_in_sigmas": record.fitting_schemes_differ_in_sigmas,
        "bound_steps": record.bound_steps,
        "steps_run": record.steps_run,
        "seconds": dict(record.seconds),
        "wall_seconds": time.perf_counter() - began,
        "model_file": None,
        "model_sha256": None,
    }
    return found, record


# --------------------------------------------------------------------------- #
# Selection and accounting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Selection:
    """The configuration a family keeps at one replicate, budget and analysis, chosen on V
    alone, and what the choice cost."""

    key: str
    tried: tuple[str, ...]  # keys of the trainings, in declared order
    criteria: tuple[float | None, ...]
    failures: tuple[str | None, ...]
    selected: str | None  # key of the training kept
    failure: str | None  # why there is no model, when every configuration failed
    seconds: float
    validation_windows: tuple[tuple[str, int], ...] = field(default=())


def select(
    family_key: str, results: Sequence[tuple[dict[str, Any], TrainingRecord | None]]
) -> Selection:
    """The configuration kept, by ``training.select_configuration``: the lowest criterion on
    V among the configurations that did not fail, the first in declared order on a tie. A
    training that could not run counts as failed. A family whose every configuration failed
    has no model there: a training failure, not retried."""
    ordered = sorted(results, key=lambda pair: pair[0]["index"])
    if [found["index"] for found, _ in ordered] != list(range(len(ordered))):
        raise ValueError(f"{family_key}: the trainings of the grid are not all present")
    trained = [(i, record) for i, (_, record) in enumerate(ordered) if record is not None]
    position = select_configuration([record for _, record in trained])
    chosen = None if position is None else trained[position][0]
    records = [found for found, _ in ordered]
    windows = next(
        (tuple(tuple(k) for k in r["validation_windows"]) for r in records if r["trained"]), ()
    )
    return Selection(
        key=family_key,
        tried=tuple(r["key"] for r in records),
        criteria=tuple(r["criterion"] for r in records),
        failures=tuple(r["failure"] for r in records),
        selected=None if chosen is None else records[chosen]["key"],
        failure=None
        if chosen is not None
        else f"{family_key}: every configuration failed: "
        + "; ".join(str(r["failure"]) for r in records),
        seconds=sum(r.get("wall_seconds", 0.0) for r in records),
        validation_windows=windows,
    )


def family_key(task: Task) -> str:
    return task.key.rsplit("/", 1)[0]


def expected_keys(spec: Specification, condition: str, replicates: Sequence[int]) -> set[str]:
    fits, trainings = tasks(spec, condition)
    return {t.key for t in fits + trainings if t.replicate in replicates}


def compatible(provenances: Sequence[Mapping[str, Any]]) -> list[str]:
    """Why runs of the development set could not be combined, or nothing: they must share
    the specification, the commit, a working tree that git called clean, and the data."""
    problems = []
    for name in ("specification_sha256", "commit", "data"):
        values = {json.dumps(p.get(name), sort_keys=True) for p in provenances}
        if len(values) > 1:
            problems.append(f"the runs differ in {name}")
    if not all(p.get("code_identified") is True for p in provenances):
        problems.append("a run was not made from a clean commit")
    return problems
