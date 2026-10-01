"""I3 of M1: development runs of the learned models (``docs/m1_plan.md``, section 13, I3).
Exploratory: nothing here is a result of M1, and no choice made from it touches benchmark
data, which do not exist yet.

Three phases, each run on its own:

* ``rates``: the rate of Adam for each family, from a short declared list, on the three target
  runs of `m0-e05` at the largest budget they hold, nine windows (F eight, V one), with one
  configuration per family.
* ``grid``: the candidate configurations of each family, two sizes and two weights of the
  penalty, on the same runs at two, five and nine windows, with the rates chosen from
  ``rates``; MR, MR_F and BL beside them.
* ``cost``: the cost of a fit at the budgets of the benchmark that the data of M0 do not
  reach, 20 and 40 windows, on synthetic P3 runs of 40 excursions with the lead of M1. They are
  simulated by the model side's rollout, from a hybrid of the modeller's model with a planted
  kinetic factor and the noise of D-020; nothing of the simulation of M0 is read. The fits
  are the same as in ``grid``.

Every training is a separate task in a pool of processes, one CPU thread each. The summary
of each phase records, for every fit, its configuration, seed, checkpoints, selection,
failure and seconds, and for the phase the wall time, the number of processes and the sum of
the seconds of its fits, so that the three are not confused.

What it reads. The export `m0-e05`, opened and verified and read only; the modeller's
values. It imports nothing of the simulation, of the generator or of the private branch.

Usage, with PT_DATA_DIR outside the repository:

    python experiments/m1_i3_pilot.py --phase rates --exports <exports> --workers 14
    python experiments/m1_i3_pilot.py --phase grid --exports <exports> --workers 14 \
        --rates BN=... HK=... HU=... HKU=...
    python experiments/m1_i3_pilot.py --phase cost --exports <exports> --workers 14 \
        --rates BN=... HK=... HU=... HKU=...
"""

from __future__ import annotations

import os

# one thread per process, before JAX is imported
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import enum  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from process_transfer.cstr_variables import (  # noqa: E402
    INPUT_NAMES,
    INPUT_UNITS,
    STATE_NAMES,
    STATE_UNITS,
)
from process_transfer.data.export import open_export_directory  # noqa: E402
from process_transfer.data.paths import data_dir, repository_root  # noqa: E402
from process_transfer.data.provenance import environment, git_state, new_run_directory  # noqa: E402
from process_transfer.evaluation.budgets import excitation, split_budget  # noqa: E402
from process_transfer.evaluation.outcomes import TrainingFailure  # noqa: E402
from process_transfer.evaluation.plant import KnownPlant, read_known_plant  # noqa: E402
from process_transfer.evaluation.windows import (  # noqa: E402
    WindowData,
    find_windows,
    layout_for,
    window_data,
)
from process_transfer.measurement.observations import Observations  # noqa: E402
from process_transfer.models.fitting import fit_mechanistic  # noqa: E402
from process_transfer.models.linear import fit_linear  # noqa: E402
from process_transfer.models.mechanistic import (  # noqa: E402
    REFERENCE_TEMPERATURE,
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
)
from process_transfer.models.rollout import rollout  # noqa: E402
from process_transfer.models.training import (  # noqa: E402
    Configuration,
    TrainingSettings,
    train,
    training_seed,
    validation_score,
)

# =========================================================================== #
# Fixed before the first run
# =========================================================================== #

PLANT = "target"
P3_EXPORT = "m0-e05"
SEED_BASE = 20260929  # the initial weights: training_seed(SEED_BASE, replicate, configuration)
RATES = (1e-3, 3e-3, 1e-2)  # the rates of Adam tried in ``rates``
RATE_STEPS, RATE_EVERY = 3000, 100
RATE_CONFIGURATIONS = {
    "BN": Configuration("BN", (16, 16), 1e-4),
    "HK": Configuration("HK", (8,), 1e-4),
    "HU": Configuration("HU", (8,), 1e-4),
    "HKU": Configuration("HKU", (8,), 1e-4),
}
# the owner's proposal for the grids: BN two hidden layers of 16 or 32; the hybrids one of 8 or
# 16, applied to both factors of HKU; two weights of the penalty
PENALTIES = (1e-4, 1e-2)
GRID = {
    "BN": [Configuration("BN", (w, w), lam) for w in (16, 32) for lam in PENALTIES],
    "HK": [Configuration("HK", (w,), lam) for w in (8, 16) for lam in PENALTIES],
    "HU": [Configuration("HU", (w,), lam) for w in (8, 16) for lam in PENALTIES],
    "HKU": [Configuration("HKU", (w,), lam) for w in (8, 16) for lam in PENALTIES],
}
GRID_BUDGETS = (2, 5, 9)
COST_BUDGETS = (20, 40)
GRID_STEPS, GRID_EVERY = 3000, 100

# the synthetic runs of ``cost``: a hybrid of the modeller's model, E/R 9000 K, k_350 and UA
# near those MR finds on the target, with a kinetic factor of the size of the target's
# mismatch, the noise of D-020, and P3 with the lead of M1
SYNTHETIC = MechanisticParameters.from_k_350(0.0177, 9000.0, 1330.0)
SYNTHETIC_RUNS = 2
SYNTHETIC_EXCURSIONS = 40
SYNTHETIC_CORNER_SEED = 20260930
SYNTHETIC_NOISE_SEED = 20260931
A10_RELATIVE, A10_ABSOLUTE = (0.10, 0.10), (5.0, 5.0)


def plain(value: object) -> object:
    """What json can write."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | range):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def fingerprints(directory: Path) -> dict[str, str]:
    found = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def m0_runs(exports: Path) -> tuple[KnownPlant, dict[str, tuple[object, tuple[WindowData, ...]]]]:
    export = open_export_directory(exports / P3_EXPORT)
    known = read_known_plant(export, PLANT)
    runs = {}
    for run_id in export.run_ids:
        if export.run(run_id)["plant_id"] != PLANT:
            continue
        observations = export.observations(run_id)
        found = find_windows(observations, known.nominal_inputs, layout_for("p3"))
        runs[run_id] = (found, window_data(observations, found.windows))
    return known, runs


class _Planted:
    """The hybrid that simulates the synthetic runs: the modeller's balances with the rate
    multiplied by exp(-0.25 (C_A - 190) / 100 + 0.01 (T - 355))."""

    name = "planted"

    def __init__(self, known: KnownPlant) -> None:
        self.known = known

    def rhs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        c_a, temperature = float(x[0]), float(x[1])
        k = SYNTHETIC.k_350 * math.exp(
            -SYNTHETIC.activation_temperature * (1.0 / temperature - 1.0 / REFERENCE_TEMPERATURE)
        )
        rate = k * c_a * math.exp(-0.25 * (c_a - 190.0) / 100.0 + 0.01 * (temperature - 355.0))
        d = u[0] / self.known.volume
        return np.array(
            [
                d * (u[1] - c_a) - rate,
                d * (u[2] - temperature)
                + self.known.heat_release_per_mole * rate
                - SYNTHETIC.ua / self.known.thermal_mass * (temperature - u[3]),
            ]
        )

    def initial_state_problem(self, x0: np.ndarray) -> str | None:
        return None if x0[1] > 0.0 else "T <= 0"


def synthetic_runs(known: KnownPlant) -> dict[str, tuple[object, tuple[WindowData, ...]]]:
    """P3 runs of 40 excursions with the lead of M1, from the planted hybrid."""
    model = _Planted(known)
    nominal = np.array(known.nominal_inputs)
    settle = rollout(model, np.array([190.0, 355.0]), np.tile(nominal, (1200, 1)), 6.0)
    steady = settle.states[-1]
    if np.max(np.abs(model.rhs(steady, nominal))) > 1e-9:
        raise RuntimeError("the planted hybrid did not settle in two hours")
    amplitudes = np.array(
        [A10_RELATIVE[0] * nominal[0], A10_RELATIVE[1] * nominal[1], *A10_ABSOLUTE]
    )
    corners = np.random.default_rng(SYNTHETIC_CORNER_SEED).choice(
        [-1, 1], size=(SYNTHETIC_RUNS, SYNTHETIC_EXCURSIONS, 4)
    )
    noise = np.random.default_rng(SYNTHETIC_NOISE_SEED)
    runs = {}
    for r in range(SYNTHETIC_RUNS):
        inputs = np.tile(nominal, (11 + 120 * SYNTHETIC_EXCURSIONS, 1))
        for j, signs in enumerate(corners[r]):
            onset = 10 + 120 * j
            inputs[onset : onset + 20] = nominal + signs * amplitudes
        states = rollout(model, steady, inputs[:-1], 6.0).states
        exact = np.vstack([steady, states])
        readings = exact + noise.normal(0.0, 1.0, exact.shape) * np.array([5.0, 0.5])
        run_id = f"target.p3.e{9000 + r}.x{SYNTHETIC_EXCURSIONS}.n0"
        observations = Observations(
            plant="target",
            run=run_id,
            times=6.0 * np.arange(len(inputs), dtype=np.float64),
            measured=readings,
            inputs=inputs,
            measured_names=STATE_NAMES,
            measured_units=STATE_UNITS,
            input_names=INPUT_NAMES,
            input_units=INPUT_UNITS,
            sample_period=6.0,
            noise_std=(5.0, 0.5),
        )
        found = find_windows(observations, nominal, layout_for("p3"))
        runs[run_id] = (found, window_data(observations, found.windows))
    return runs


# --------------------------------------------------------------------------- #
# Tasks, one per process
# --------------------------------------------------------------------------- #

_CONTEXT: dict[str, object] = {}


def _initialise(context: dict[str, object]) -> None:
    _CONTEXT.update(context)


def _part(run_id: str, budget: int) -> tuple[tuple[WindowData, ...], tuple[WindowData, ...]]:
    found, data = _CONTEXT["runs"][run_id]
    split = split_budget(found, budget)
    by_key = {d.key: d for d in data}
    return (
        tuple(by_key[w.key] for w in split.fitting),
        tuple(by_key[w.key] for w in split.validation),
    )


def mechanistic_task(task: tuple[str, str, int]) -> dict[str, object]:
    """MR on all b windows, MR_F on F, or BL on all b windows."""
    kind, run_id, budget = task
    known, modeller = _CONTEXT["known"], _CONTEXT["modeller"]
    fitting, validation = _part(run_id, budget)
    began = time.perf_counter()
    if kind == "MR_F":
        fit = fit_mechanistic("MR_F", fitting, known, modeller)
        found = {
            "parameters": fit.parameters,
            "objective": fit.objective,
            "failure": fit.training_failure,
        }
        if fit.parameters is not None:
            model = MechanisticModel("MR_F", known, fit.parameters)
            found["validation"] = validation_score(model, validation)[0]
    elif kind == "MR":
        fit = fit_mechanistic("MR", fitting + validation, known, modeller)
        found = {
            "parameters": fit.parameters,
            "objective": fit.objective,
            "failure": fit.training_failure,
        }
    else:
        fit = fit_linear("BL", fitting + validation, known.nominal_inputs)
        found = {
            "rank": fit.directions.rank,
            "objective": fit.objective,
            "failure": fit.training_failure,
            "starts": [(s.outcome, s.objective, s.evaluations) for s in fit.starts],
        }
    found.update(kind=kind, run=run_id, budget=budget, seconds=time.perf_counter() - began)
    return found


def training_task(task: dict[str, object]) -> dict[str, object]:
    known = _CONTEXT["known"]
    fitting, validation = _part(task["run"], task["budget"])
    settings = TrainingSettings(
        learning_rate=task["rate"], max_steps=task["steps"], validation_every=task["every"]
    )
    start = task.get("start")
    began = time.perf_counter()
    record = train(
        task["configuration"], settings, fitting, validation, known, task["seed"], start=start
    )
    seconds = time.perf_counter() - began
    return {
        "run": task["run"],
        "budget": task["budget"],
        "configuration": record.configuration.label,
        "family": record.configuration.family,
        "rate": task["rate"],
        "seed": task["seed"],
        "network_size": record.network_size,
        "failure": record.failure,
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
        "mechanistic": None
        if record.parameters is None or "theta" not in record.parameters
        else record.model(known).mechanistic_parameters(),
        "fitting_schemes_differ_in_sigmas": record.fitting_schemes_differ_in_sigmas,
        "bound_steps": record.bound_steps,
        "seconds": record.seconds,
        "wall_seconds": seconds,
    }


def blocked_outcome(task: dict[str, object], reason: str) -> dict[str, object]:
    """The outcome of a hybrid that could not be trained because its start, MR_F of the
    same run and budget, failed: a training failure, recorded like any other."""
    configuration = task["configuration"]
    return {
        "run": task["run"],
        "budget": task["budget"],
        "configuration": configuration.label,
        "family": configuration.family,
        "rate": task["rate"],
        "seed": task["seed"],
        "network_size": None,
        "failure": TrainingFailure(
            f"{configuration.label}: not trained, since its start failed: {reason}"
        ),
        "selected_step": None,
        "criterion": None,
        "checkpoints": [],
        "mechanistic": None,
        "fitting_schemes_differ_in_sigmas": None,
        "bound_steps": 0,
        "seconds": {},
        "wall_seconds": 0.0,
    }


def outcome_key(outcome: dict[str, object]) -> tuple[str, int, str, float]:
    return (outcome["run"], outcome["budget"], outcome["configuration"], outcome["rate"])


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def run_tasks(function, tasks: Sequence, workers: int, context: dict) -> tuple[list, float]:  # noqa: ANN001
    began = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_initialise, initargs=(context,)) as pool:
            results = list(pool.map(function, tasks))
    else:
        _initialise(context)
        results = [function(task) for task in tasks]
    return results, time.perf_counter() - began


def starts_of(
    mechanistic: list[dict[str, object]],
) -> dict[tuple[str, int], MechanisticParameters | str]:
    """MR_F of each run and budget, or why there is none."""
    found: dict[tuple[str, int], MechanisticParameters | str] = {}
    for m in mechanistic:
        if m["kind"] != "MR_F":
            continue
        key = (m["run"], m["budget"])
        if m["parameters"] is not None:
            found[key] = m["parameters"]
        else:
            failure = m["failure"]
            found[key] = f"MR_F of {key[0]} at {key[1]} windows: " + (
                failure.reason if failure is not None else "no parameters"
            )
    return found


def trainings(
    configurations: dict[str, list[Configuration]],
    rates: dict[str, float],
    runs: Sequence[str],
    budgets: Sequence[int],
    starts: dict[tuple[str, int], MechanisticParameters | str],
    steps: int,
    every: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The trainings to run, and the outcomes of those that cannot run: a hybrid whose
    MR_F failed, or was never fitted, is a training failure with that reason, never left
    out of the count."""
    tasks, blocked = [], []
    for replicate, run_id in enumerate(runs):
        for budget in budgets:
            for family, listed in configurations.items():
                for index, configuration in enumerate(listed):
                    task = {
                        "run": run_id,
                        "budget": budget,
                        "configuration": configuration,
                        "rate": rates[family],
                        "steps": steps,
                        "every": every,
                        "start": None,
                        "seed": training_seed(SEED_BASE, replicate, index),
                    }
                    if family != "BN":
                        start = starts.get(
                            (run_id, budget), f"MR_F of {run_id} at {budget} windows: not fitted"
                        )
                        if isinstance(start, str):
                            blocked.append(blocked_outcome(task, start))
                            continue
                        task["start"] = start
                    tasks.append(task)
    return tasks, blocked


def expected_outcomes(
    configurations: dict[str, list[Configuration]],
    rates: Sequence[dict[str, float]],
    runs: Sequence[str],
    budgets: Sequence[int],
) -> set[tuple[str, int, str, float]]:
    """Every run, budget, configuration and rate that must have an outcome."""
    return {
        (run_id, budget, configuration.label, chosen[family])
        for chosen in rates
        for run_id in runs
        for budget in budgets
        for family, listed in configurations.items()
        for configuration in listed
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--phase", choices=("rates", "grid", "cost"), required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rates", nargs="*", default=[], help="FAMILY=RATE for grid and cost")
    arguments = parser.parse_args()
    root = repository_root().resolve()
    target = data_dir().resolve()
    if target == root or root in target.parents:
        print(f"PT_DATA_DIR resolves to {target}, inside the repository; refused", file=sys.stderr)
        return 1
    state = git_state()
    began = time.perf_counter()
    exports_before = fingerprints(arguments.exports / P3_EXPORT)
    known, runs = m0_runs(arguments.exports)
    modeller = modeller_values()
    summary: dict[str, object] = {
        "phase": arguments.phase,
        "provenance": {"git": state, "environment": environment(), "data_dir": str(target)},
    }
    rates = {}
    for item in arguments.rates:
        family, value = item.split("=")
        rates[family] = float(value)
    if arguments.phase == "rates":
        budgets, run_ids = (9,), list(runs)
        configurations = {f: [c] for f, c in RATE_CONFIGURATIONS.items()}
        context = {"known": known, "modeller": modeller, "runs": runs}
        mechanistic, mech_wall = run_tasks(
            mechanistic_task,
            [("MR_F", r, b) for r in run_ids for b in budgets],
            arguments.workers,
            context,
        )
        starts = starts_of(mechanistic)
        tasks, blocked = [], []
        rate_lists = [{f: rate for f in configurations} for rate in RATES]
        for chosen in rate_lists:
            found, held = trainings(
                configurations, chosen, run_ids, budgets, starts, RATE_STEPS, RATE_EVERY
            )
            tasks += found
            blocked += held
        steps, every = RATE_STEPS, RATE_EVERY
    else:
        if set(rates) != set(GRID):
            print(f"--rates must give one rate for each of {sorted(GRID)}", file=sys.stderr)
            return 2
        if arguments.phase == "grid":
            budgets, run_ids = GRID_BUDGETS, list(runs)
        else:
            runs = synthetic_runs(known)
            budgets, run_ids = COST_BUDGETS, list(runs)
            summary["synthetic"] = {
                "parameters": SYNTHETIC,
                "runs": run_ids,
                "excursions": SYNTHETIC_EXCURSIONS,
                "corner_seed": SYNTHETIC_CORNER_SEED,
                "noise_seed": SYNTHETIC_NOISE_SEED,
                "factor": "exp(-0.25 (C_A - 190) / 100 + 0.01 (T - 355))",
            }
        context = {"known": known, "modeller": modeller, "runs": runs}
        kinds = [(k, r, b) for r in run_ids for b in budgets for k in ("MR_F", "MR", "BL")]
        mechanistic, mech_wall = run_tasks(mechanistic_task, kinds, arguments.workers, context)
        starts = starts_of(mechanistic)
        configurations, rate_lists = GRID, [rates]
        tasks, blocked = trainings(GRID, rates, run_ids, budgets, starts, GRID_STEPS, GRID_EVERY)
        steps, every = GRID_STEPS, GRID_EVERY
    results, train_wall = run_tasks(training_task, tasks, arguments.workers, context)
    results += blocked
    # every expected outcome is recorded once: a failure does not leave the denominator
    expected = expected_outcomes(configurations, rate_lists, run_ids, budgets)
    recorded = [outcome_key(r) for r in results]
    summary["accounting"] = {
        "expected": len(expected),
        "recorded": len(recorded),
        "blocked": len(blocked),
        "complete": set(recorded) == expected and len(recorded) == len(expected),
    }
    summary["excitation"] = {
        f"{r} b={b}": plain(excitation(_part_of(runs, r, b), known.nominal_inputs))
        for r in run_ids
        for b in budgets
    }
    summary["mechanistic"] = mechanistic
    summary["trainings"] = results
    summary["fixed_before_the_run"] = {
        "seed_base": SEED_BASE,
        "rates_tried": RATES,
        "rates": rates,
        "steps": steps,
        "validation_every": every,
        "grid": GRID,
        "rate_configurations": RATE_CONFIGURATIONS,
        "budgets": budgets,
    }
    summary["cost"] = {
        "workers": arguments.workers,
        "wall_seconds_mechanistic": mech_wall,
        "wall_seconds_trainings": train_wall,
        "sum_of_fit_seconds_mechanistic": sum(m["seconds"] for m in mechanistic),
        "sum_of_fit_seconds_trainings": sum(r["wall_seconds"] for r in results),
        "fits": {"mechanistic": len(mechanistic), "trainings": len(results)},
    }
    summary["exports_unchanged"] = fingerprints(arguments.exports / P3_EXPORT) == exports_before
    summary["seconds"] = time.perf_counter() - began
    directory = new_run_directory("m1_i3_pilot", state)
    (directory / f"summary_{arguments.phase}.json").write_text(
        json.dumps(plain(summary), indent=1), encoding="utf-8"
    )
    print(directory)
    print(json.dumps(plain(summary["cost"]), indent=1))
    return 0 if summary["exports_unchanged"] and summary["accounting"]["complete"] else 1


def _part_of(runs, run_id: str, budget: int) -> tuple[WindowData, ...]:  # noqa: ANN001
    found, data = runs[run_id]
    split = split_budget(found, budget)
    keys = {w.key for w in split.windows}
    return tuple(d for d in data if d.key in keys)


if __name__ == "__main__":
    sys.exit(main())
