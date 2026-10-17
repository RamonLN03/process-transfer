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
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np

from process_transfer.data.export import EXPORT_MANIFEST
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
# how a run of the development runs ended, as its summary says
COMPLETE, INCOMPLETE, INTERRUPTED = "complete", "incomplete", "interrupted"
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


def select(family_key: str, records: Sequence[Mapping[str, Any]]) -> Selection:
    """The configuration kept, by ``training.select_configuration`` applied to the records
    of the family's trainings: the lowest criterion on V among the configurations that did
    not fail, the first in declared order on a tie. A training that could not run counts as
    failed. A family whose every configuration failed has no model there: a training
    failure, not retried."""
    ordered = sorted(records, key=lambda r: r["index"])
    if [r["index"] for r in ordered] != list(range(len(ordered))):
        raise ValueError(f"{family_key}: the trainings of the grid are not all present")
    outcomes = [SimpleNamespace(failure=r["failure"], criterion=r["criterion"]) for r in ordered]
    chosen = select_configuration(outcomes)
    windows = next(
        (tuple(tuple(k) for k in r["validation_windows"]) for r in ordered if r["trained"]), ()
    )
    return Selection(
        key=family_key,
        tried=tuple(r["key"] for r in ordered),
        criteria=tuple(r["criterion"] for r in ordered),
        failures=tuple(r["failure"] for r in ordered),
        selected=None if chosen is None else ordered[chosen]["key"],
        failure=None
        if chosen is not None
        else f"{family_key}: every configuration failed: "
        + "; ".join(str(r["failure"]) for r in ordered),
        seconds=sum(r.get("wall_seconds", 0.0) for r in ordered),
        validation_windows=windows,
    )


def family_key(key: str) -> str:
    """The key of the family a training belongs to: its key without the configuration."""
    return key.rsplit("/", 1)[0]


def expected_keys(spec: Specification, condition: str, replicates: Sequence[int]) -> set[str]:
    fits, trainings = tasks(spec, condition)
    return {t.key for t in fits + trainings if t.replicate in replicates}


def families(spec: Specification, condition: str, replicates: Sequence[int]) -> dict[str, int]:
    """Every family of trainings at the replicates, with the number of its configurations."""
    found: dict[str, int] = {}
    for task in tasks(spec, condition)[1]:
        if task.replicate in replicates:
            found[family_key(task.key)] = found.get(family_key(task.key), 0) + 1
    return found


# --------------------------------------------------------------------------- #
# Tasks in worker processes
# --------------------------------------------------------------------------- #

_WORKER: dict[str, Any] = {}


def initialise_worker(context: Mapping[str, Any]) -> None:
    """The data a worker process needs, handed once: the specification, the known plant,
    the windows of every run and the modeller's values."""
    _WORKER.clear()
    _WORKER.update(context)


def _parts_of(task: Task) -> tuple[tuple[WindowData, ...], tuple[WindowData, ...]]:
    run_windows, data = _WORKER["runs"][task.run_id]
    return parts(run_windows, data, task.budget)


def fit_task(task: Task) -> dict[str, Any]:
    fitting, validation = _parts_of(task)
    return run_fit(
        task, fitting, validation, _WORKER["known"], _WORKER["modeller"], _WORKER["spec"]
    )


def train_task(item: tuple[Task, Mapping[str, float] | None]) -> tuple[dict[str, Any], Any]:
    task, start = item
    fitting, validation = _parts_of(task)
    parameters = None if start is None else MechanisticParameters(**start)
    return run_training(task, fitting, validation, _WORKER["known"], parameters, _WORKER["spec"])


# --------------------------------------------------------------------------- #
# The identity of the observable data, and the provenance of a run
# --------------------------------------------------------------------------- #


def canonical_sha256(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def data_identity(
    dataset_id: str,
    known: KnownPlant,
    run_content: Mapping[str, str],
    export_files: Mapping[str, str],
    generation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """What identifies the observable data a run used, with its SHA-256:

    * the data set;
    * the known parameters of the plant as they were read (``KnownPlant``), which the
      content hash of an observation set does not cover;
    * the content hash of every run, which covers the readings, the inputs and the
      specification of the instruments;
    * the SHA-256 of every file of the export, its manifest included;
    * the attempt of generation the manifest records, commit and time, which links to the
      private provenance without reading it.

    Nothing private enters: no master seed, no plant configuration, nothing of the truth."""
    identity = {
        "dataset_id": dataset_id,
        "known_plant": dataclasses.asdict(known),
        "run_content_sha256": dict(sorted(run_content.items())),
        "export_files_sha256": dict(sorted(export_files.items())),
        "generation": None if generation is None else dict(generation),
    }
    return {**identity, "sha256": canonical_sha256(identity)}


ENVIRONMENT = (
    ("python",),
    ("implementation",),
    ("machine",),
    ("packages", "numpy"),
    ("packages", "scipy"),
    ("learning", "jax"),
    ("learning", "jaxlib"),
    ("learning", "backend"),
    ("learning", "x64"),
    ("learning", "float_dtype"),
)


def relevant_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the environment that decides the numbers. Runs combined must agree on
    every item of it; the full platform string may differ, and is reported."""
    found = {}
    for path in ENVIRONMENT:
        node: Any = environment
        for part in path:
            node = node.get(part) if isinstance(node, Mapping) else None
        found[".".join(path)] = node
    return found


DATA_FIELDS = (
    "dataset_id",
    "known_plant",
    "run_content_sha256",
    "export_files_sha256",
    "generation",
    "sha256",
)
GENERATION_FIELDS = ("generated_at_utc", "commit", "code_identified", "packages")
KNOWN_FIELDS = tuple(f.name for f in dataclasses.fields(KnownPlant))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _hex(value: object, pattern: re.Pattern[str] = _SHA256) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _hashes(name: str, found: object, expected: set[str]) -> list[str]:
    """A mapping from exactly ``expected`` to SHA-256 digests."""
    if not isinstance(found, Mapping):
        return [f"{name} is not a mapping"]
    problems = []
    if set(found) != expected:
        problems.append(
            f"{name} covers {len(found)} names, not the {len(expected)} expected: missing "
            f"{sorted(expected - set(found))}, not expected {sorted(set(found) - expected)}"
        )
    bad = sorted(str(k) for k, v in found.items() if not _hex(v))
    if bad:
        problems.append(f"{name} holds values that are not SHA-256 digests, for {bad}")
    return problems


def _known(found: object, spec: Specification) -> list[str]:
    """The known parameters of the plant of the specification, valid as ``KnownPlant``."""
    if not isinstance(found, Mapping) or set(found) != set(KNOWN_FIELDS):
        names = sorted(found) if isinstance(found, Mapping) else found
        return [f"the known parameters are {names!r}, not the fields {list(KNOWN_FIELDS)}"]
    if found["plant_id"] != spec.plant:
        return [f"the known parameters are of {found['plant_id']!r}, not of {spec.plant!r}"]
    nominal = found["nominal_inputs"]
    scalars = [found[name] for name in KNOWN_FIELDS if name not in ("plant_id", "nominal_inputs")]
    if not isinstance(nominal, list | tuple) or not all(map(_number, [*scalars, *nominal])):
        return ["the known parameters are not all finite numbers"]
    try:
        KnownPlant(**{**found, "nominal_inputs": tuple(nominal)})
    except ValueError as error:
        return [f"the known parameters are not valid: {error}"]
    return []


def _generation(found: object) -> list[str]:
    """The attempt of generation as the generator writes it into the manifest: when, from
    which commit, from a clean tree, with which packages."""
    if not isinstance(found, Mapping) or set(found) != set(GENERATION_FIELDS):
        names = sorted(found) if isinstance(found, Mapping) else found
        return [
            f"the attempt of generation has {names!r}, not the fields {list(GENERATION_FIELDS)}"
        ]
    problems = []
    if not _hex(found["commit"], _COMMIT):
        problems.append(f"the data were generated at {found['commit']!r}, not at a full commit")
    if found["code_identified"] is not True:
        problems.append("the data were not generated from a clean commit")
    when = found["generated_at_utc"]
    try:
        stamp = datetime.fromisoformat(when) if isinstance(when, str) else None
    except ValueError:
        stamp = None
    if stamp is None or stamp.utcoffset() != timedelta(0):
        problems.append(f"the time of generation {when!r} is not a time in UTC")
    packages = found["packages"]
    if (
        not isinstance(packages, Mapping)
        or not packages
        or not all(
            isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in packages.items()
        )
    ):
        problems.append("the packages of the generation are not a mapping of names to versions")
    return problems


def check_data_identity(data: object, spec: Specification, condition: str) -> list[str]:
    """Why ``data`` is not the identity of the development set of ``condition``, or nothing.
    Its structure is checked before its hash: every field and its type, valid known
    parameters, the content hash of exactly the runs of the condition, the hash of exactly
    the files of its export, an attempt of generation as the generator writes it, and a
    SHA-256 that is that of the rest. A hash that matches an incomplete content proves
    nothing about the data."""
    if not isinstance(data, Mapping):
        return ["the identity of its data is not a mapping"]
    if set(data) != set(DATA_FIELDS):
        return [
            f"the identity of its data has the fields {sorted(data)}, not {sorted(DATA_FIELDS)}"
        ]
    problems = []
    if data["dataset_id"] != spec.sets[condition].dataset_id:
        problems.append(f"its data are not the development set of {condition}")
    problems += _known(data["known_plant"], spec)
    runs = set(spec.run_ids(condition))
    problems += _hashes("the content hashes of its runs", data["run_content_sha256"], runs)
    files = {EXPORT_MANIFEST} | {f"{run_id}.parquet" for run_id in runs}
    problems += _hashes("the hashes of its export", data["export_files_sha256"], files)
    problems += _generation(data["generation"])
    try:  # a value that JSON cannot hold, such as not a number, has no canonical form
        recomputed = canonical_sha256({k: v for k, v in data.items() if k != "sha256"})
    except (TypeError, ValueError):
        recomputed = None
    if not _hex(data["sha256"]) or data["sha256"] != recomputed:
        problems.append("the identity of its data does not match its content")
    return problems


REQUIRED = (
    "run_id",
    "commit",
    "code_identified",
    "environment",
    "specification_sha256",
    "condition",
    "replicates",
    "data",
)


def check_provenance(provenance: Mapping[str, Any], spec: Specification) -> list[str]:
    """Why a provenance cannot be used, or nothing."""
    name = provenance.get("run_id", "a run")
    missing = [k for k in REQUIRED if k not in provenance]
    if missing:
        return [f"{name}: the provenance lacks {missing}"]
    problems = []
    commit = provenance["commit"]
    if (
        not isinstance(commit, str)
        or len(commit) not in (40, 64)
        or not all(c in "0123456789abcdef" for c in commit)
    ):
        problems.append(f"{name}: the commit {commit!r} is not a full object name")
    if provenance["code_identified"] is not True:
        problems.append(f"{name}: not made from a clean commit")
    if provenance["specification_sha256"] != spec.digest():
        problems.append(f"{name}: made under another specification")
    condition = provenance["condition"]
    if condition not in spec.sets:
        problems.append(f"{name}: unknown condition {condition!r}")
        return problems
    replicates = provenance["replicates"]
    allowed = range(len(spec.run_ids(condition)))
    if (
        not isinstance(replicates, list)
        or not replicates
        or len(set(replicates)) != len(replicates)
        or not all(
            isinstance(r, int) and not isinstance(r, bool) and r in allowed for r in replicates
        )
    ):
        problems.append(f"{name}: the replicates {replicates!r} are not distinct replicates")
    problems += [f"{name}: {p}" for p in check_data_identity(provenance["data"], spec, condition)]
    environment = relevant_environment(provenance["environment"])
    absent = sorted(k for k, v in environment.items() if v is None)
    if absent:
        problems.append(f"{name}: the environment lacks {absent}")
    return problems


def compatible(provenances: Sequence[Mapping[str, Any]], spec: Specification) -> list[str]:
    """Why runs of the development sets could not be combined, or nothing. Each must have a
    complete provenance; together they must share the commit, the specification and the
    relevant environment; and the runs of each condition must share the identity of that
    condition's data. A10 and A5 are different sets and are not compared with each other."""
    problems = [p for provenance in provenances for p in check_provenance(provenance, spec)]
    if problems:
        return problems
    for name in ("commit", "specification_sha256"):
        if len({p[name] for p in provenances}) > 1:
            problems.append(f"the runs differ in {name}")
    environments = {canonical_sha256(relevant_environment(p["environment"])) for p in provenances}
    if len(environments) > 1:
        problems.append("the runs differ in their numerical environment")
    by_condition: dict[str, set[str]] = {}
    for p in provenances:
        by_condition.setdefault(p["condition"], set()).add(p["data"]["sha256"])
    for condition, identities in sorted(by_condition.items()):
        if len(identities) > 1:
            problems.append(f"the runs of {condition} differ in the identity of their data")
    return problems


ENDINGS = {
    "missing": "has no summary.json: it did not end, and a run that did not end is not combined",
    "malformed": "has a summary.json that does not say how the run ended",
    "incomplete": (
        "ended incomplete, with tasks pending after errors of infrastructure; it is not "
        "combined until its recovery has been decided and checked, as a separate step"
    ),
    "interrupted": (
        "was interrupted; it is not combined until its recovery has been decided and "
        "checked, as a separate step"
    ),
}
EXPORT_CHECKS = {
    "missing": (
        "has no exports_after.json: the final check of its exports was not made, and a "
        "missing check is not taken for a pass"
    ),
    "malformed": (
        'has an exports_after.json that is not {"unchanged": true} or {"unchanged": false}'
    ),
    "changed": "says its exports changed during the run",
}


def check_ending(run: Mapping[str, Any]) -> list[str]:
    """Why a run cannot be combined by the way it ended, or nothing. ``run["ending"]`` holds
    the status its summary gives and the result of the final check of its exports, as the
    run's own files state them; nothing is inferred from an absence. A run must have ended
    complete, with its exports unchanged. ``combine`` also requires that it hold a record of
    every task it was given."""
    where = run["directory"]
    ending = run.get("ending")
    if not isinstance(ending, Mapping):
        return [f"{where}: how the run ended was not read"]
    problems = []
    status, exports = ending.get("summary"), ending.get("exports")
    if status != COMPLETE:
        problems.append(f"{where} {ENDINGS.get(status, ENDINGS['malformed'])}")
    if exports != "unchanged":
        problems.append(f"{where} {EXPORT_CHECKS.get(exports, EXPORT_CHECKS['malformed'])}")
    return problems


def combine(runs: Sequence[Mapping[str, Any]], spec: Specification) -> dict[str, Any]:
    """The records of runs made apart, put together. Each item of ``runs`` holds the
    ``provenance`` of a run, its ``records`` by key, its ``directory`` and its ``ending``.
    The runs must be compatible, and each must have ended complete with its exports
    unchanged (``check_ending``); a task recorded twice is refused, whatever its outcome,
    so no repetition is chosen by its result; a record whose key the specification does not
    call for is refused. Failures are outcomes and stay in every count. The originals are
    not changed: the result refers to them."""
    if not runs:
        raise ValueError("no run to combine")
    problems = compatible([r["provenance"] for r in runs], spec)
    if problems:
        raise ValueError("; ".join(problems))
    names = [r["provenance"]["run_id"] for r in runs]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"the runs {repeated} are given more than once")
    problems = [p for run in runs for p in check_ending(run)]
    if problems:
        raise ValueError("; ".join(problems))
    owners: dict[str, str] = {}
    records: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        condition = run["provenance"]["condition"]
        allowed = expected_keys(spec, condition, run["provenance"]["replicates"])
        for key, record in run["records"].items():
            if key not in allowed:
                raise ValueError(f"{key}, in {run['directory']}, is not a task of that run")
            if record.get("key") != key:
                raise ValueError(
                    f"{run['directory']}: the record filed as {key} says {record.get('key')}"
                )
            if key in owners:
                raise ValueError(
                    f"{key} is recorded twice, in {owners[key]} and {run['directory']}; "
                    "a repetition is not chosen by its result"
                )
            owners[key] = str(run["directory"])
            records[key] = record
        if set(run["records"]) != allowed:
            raise ValueError(
                f"{run['directory']} holds {len(set(run['records']) & allowed)} of the "
                f"{len(allowed)} records of its tasks; a run is combined whole"
            )
    coverage = {}
    for condition in sorted({r["provenance"]["condition"] for r in runs}):
        every = list(range(len(spec.run_ids(condition))))
        expected = expected_keys(spec, condition, every)
        found = {k for k in records if k.startswith(f"{condition}/")}
        coverage[condition] = {
            "expected": len(expected),
            "recorded": len(found),
            "missing": len(expected - found),
            "complete": found == expected,
        }
    selections = {}
    for condition in coverage:
        every = list(range(len(spec.run_ids(condition))))
        for key, size in sorted(families(spec, condition, every).items()):
            members = [
                r for k, r in records.items() if family_key(k) == key and r["kind"] == "training"
            ]
            if len(members) == size:
                selections[key] = select(key, members)
    return {
        "runs": [
            {
                "directory": str(r["directory"]),
                "run_id": r["provenance"]["run_id"],
                "condition": r["provenance"]["condition"],
                "replicates": r["provenance"]["replicates"],
                "provenance_sha256": canonical_sha256(r["provenance"]),
                "ledger_sha256": r.get("ledger_sha256"),
                "records": len(r["records"]),
                "ending": dict(r["ending"]),
            }
            for r in runs
        ],
        "commit": runs[0]["provenance"]["commit"],
        "specification_sha256": spec.digest(),
        "coverage": coverage,
        "complete": all(c["complete"] for c in coverage.values())
        and set(coverage) == set(spec.sets),
        "failures": {
            "fits": sorted(
                k for k, r in records.items() if r["kind"] != "training" and r["failure"]
            ),
            "trainings": sorted(
                k for k, r in records.items() if r["kind"] == "training" and r["failure"]
            ),
        },
        "selections": selections,
    }
