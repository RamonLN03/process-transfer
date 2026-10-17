"""M1-E03, the development runs of I6: every fit and training that the registration calls
for on the development sets, with each configuration chosen on V, and nothing else.

Registered in ``docs/experiment_log.md`` (M1-E03); the procedure is that of
``process_transfer.models.benchmark``. Two gates come before an official run (D-043):

* Codex's review of the registration;
* the checks before the full run of section 10 of the registration: ingestion, disk
  space, and the cost of the first replicate, which this script runs with
  ``--replicates 0``.

What the script reads: the exports of ``m1-e03-train-a10`` and ``m1-e03-train-a5``, read
only, under ``PT_DATA_DIR/available/exports`` or ``--exports``. Each is checked to be the
registered development set, with exactly its runs. Nothing else is read: no test set, no
plant configuration, no definition of a data set, nothing of the private branch or of the
truth.

What it writes, in a new directory ``PT_DATA_DIR/experiments/m1_e03_development/<run id>``
that no other run uses:

* ``records/<key>.json``, one for every task, a failure included, each written once;
* ``models/<key>.json``, the selected checkpoint of every configuration that has one,
  written by ``models.persistence`` with the SHA-256 of its content;
* ``selections.json``, the configuration each family keeps at each replicate, budget and
  analysis, chosen on V alone, and the effort the choice took;
* ``summary.json``, with the provenance (commit, clean tree, environment with JAX, the
  specification and its SHA-256, the content hashes of the runs read, the command), the
  accounting of every expected task, and the cost.

An official run refuses a working tree that git does not call clean, and a directory of
outputs inside the repository. Runs made apart, such as the first replicate and then the
others, are combined only if ``benchmark.compatible`` finds the same specification, commit,
clean tree and data.

    python experiments/m1_e03_development.py --condition A10 --replicates 0 --workers 14
    python experiments/m1_e03_development.py --condition A10 --workers 14

Exit codes: 0 when every expected task has an outcome, a failure included, and the exports
were not changed; 1 when not; 2 when the run is refused before it starts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# one thread per process, before JAX is imported
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from process_transfer.data.export import open_export_directory  # noqa: E402
from process_transfer.data.paths import data_dir, repository_root  # noqa: E402
from process_transfer.data.provenance import git_state, new_run_directory  # noqa: E402
from process_transfer.evaluation.budgets import excitation  # noqa: E402
from process_transfer.evaluation.plant import KnownPlant, read_known_plant  # noqa: E402
from process_transfer.evaluation.windows import (  # noqa: E402
    RunWindows,
    WindowData,
    find_windows,
    layout_for,
    window_data,
)
from process_transfer.models import benchmark  # noqa: E402
from process_transfer.models.mechanistic import MechanisticParameters, modeller_values  # noqa: E402
from process_transfer.models.persistence import save_selected  # noqa: E402
from process_transfer.models.training import learning_environment  # noqa: E402

EXPERIMENT = "m1_e03_development"
Runs = dict[str, tuple[RunWindows, tuple[WindowData, ...]]]


class Refused(Exception):
    """A run that must not start, with the reason."""


def plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    return value


def write_once(path: Path, content: object) -> None:
    """A record is written once: an existing file is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(plain(content), handle, indent=1, allow_nan=False)


def file_name(key: str) -> str:
    return key.replace("/", "_") + ".json"


# --------------------------------------------------------------------------- #
# Reading the development set
# --------------------------------------------------------------------------- #


def check_export(
    dataset_id: str,
    run_ids: Sequence[str],
    plants: Mapping[str, str],
    expected: Sequence[str],
    registered_id: str,
) -> None:
    """The export must be the registered development set: its identifier, exactly its
    runs, and only the target. A test set or any other data set is refused."""
    if dataset_id != registered_id:
        raise Refused(f"the export is {dataset_id!r}, not the development set {registered_id!r}")
    if sorted(run_ids) != sorted(expected):
        missing = sorted(set(expected) - set(run_ids))
        extra = sorted(set(run_ids) - set(expected))
        raise Refused(f"{dataset_id}: runs missing {missing}, runs not registered {extra}")
    other = sorted({plant for plant in plants.values() if plant != "target"})
    if other:
        raise Refused(f"{dataset_id}: runs of plants other than the target: {other}")


def load_condition(
    exports: Path, spec: benchmark.Specification, condition: str
) -> tuple[KnownPlant, Runs, dict[str, str]]:
    """The known plant, the windows of every run, and the content hash of every run, from
    the export of the condition's development set."""
    registered = spec.sets[condition].dataset_id
    export = open_export_directory(exports / registered)
    plants = {run_id: str(export.run(run_id)["plant_id"]) for run_id in export.run_ids}
    check_export(export.dataset_id, export.run_ids, plants, spec.run_ids(condition), registered)
    known = read_known_plant(export, spec.plant)
    runs: Runs = {}
    hashes = {}
    for run_id in spec.run_ids(condition):
        observations = export.observations(run_id)
        found = find_windows(observations, known.nominal_inputs, layout_for("p3"))
        runs[run_id] = (found, window_data(observations, found.windows))
        hashes[run_id] = str(export.run(run_id)["content_sha256"])
    return known, runs, hashes


def fingerprints(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------- #
# Tasks in processes
# --------------------------------------------------------------------------- #

_CONTEXT: dict[str, Any] = {}


def _initialise(context: dict[str, Any]) -> None:
    _CONTEXT.update(context)


def _parts(task: benchmark.Task) -> tuple[tuple[WindowData, ...], tuple[WindowData, ...]]:
    found, data = _CONTEXT["runs"][task.run_id]
    return benchmark.parts(found, data, task.budget)


def _fit(task: benchmark.Task) -> dict[str, Any]:
    fitting, validation = _parts(task)
    return benchmark.run_fit(
        task, fitting, validation, _CONTEXT["known"], _CONTEXT["modeller"], _CONTEXT["spec"]
    )


def _train(item: tuple[benchmark.Task, dict[str, float] | None]) -> tuple[dict, Any]:
    task, start = item
    fitting, validation = _parts(task)
    parameters = None if start is None else MechanisticParameters(**start)
    return benchmark.run_training(
        task, fitting, validation, _CONTEXT["known"], parameters, _CONTEXT["spec"]
    )


def _map(function, items: Sequence, workers: int, context: dict[str, Any]) -> list:  # noqa: ANN001
    if workers > 1 and len(items) > 1:
        with ProcessPoolExecutor(workers, initializer=_initialise, initargs=(context,)) as pool:
            return list(pool.map(function, items))
    _initialise(context)
    return [function(item) for item in items]


# --------------------------------------------------------------------------- #
# The development runs of one condition
# --------------------------------------------------------------------------- #


def execute(
    spec: benchmark.Specification,
    condition: str,
    replicates: Sequence[int],
    known: KnownPlant,
    runs: Runs,
    modeller: Any,
    directory: Path,
    workers: int = 1,
) -> dict[str, Any]:
    """Every task of ``condition`` at ``replicates``, its record written once; the hybrids
    started from MR_F of their replicate, budget and analysis; the selected checkpoint of
    every configuration written; the selection of every family; the accounting."""
    fits, trainings = benchmark.tasks(spec, condition)
    fits = [t for t in fits if t.replicate in replicates]
    trainings = [t for t in trainings if t.replicate in replicates]
    context = {"spec": spec, "known": known, "runs": runs, "modeller": modeller}

    began = time.perf_counter()
    fitted = _map(_fit, fits, workers, context)
    fit_seconds = time.perf_counter() - began
    for record in fitted:
        write_once(directory / "records" / file_name(record["key"]), record)
    starts = {r["key"]: r for r in fitted if r["kind"] == "MR_F"}

    to_train, outcomes = [], []
    for task in trainings:
        if task.family == "BN":
            to_train.append((task, None))
            continue
        start = starts.get(task.start_key)
        if start is None or start["model"] is None:
            reason = "no MR_F was fitted" if start is None else start["failure"]
            outcomes.append((benchmark.blocked(task, f"{task.start_key}: {reason}"), None))
        else:
            to_train.append((task, start["model"]))
    began = time.perf_counter()
    outcomes += _map(_train, to_train, workers, context)
    train_seconds = time.perf_counter() - began

    by_family: dict[str, list] = {}
    (directory / "models").mkdir(parents=True, exist_ok=True)
    for found, record in outcomes:
        if record is not None and record.selected is not None:
            path = directory / "models" / file_name(found["key"])
            found["model_file"] = str(path.relative_to(directory))
            found["model_sha256"] = save_selected(record, known, path)
        write_once(directory / "records" / file_name(found["key"]), found)
        by_family.setdefault(found["key"].rsplit("/", 1)[0], []).append((found, record))
    selections = {key: benchmark.select(key, pairs) for key, pairs in sorted(by_family.items())}
    write_once(directory / "selections.json", selections)

    # the excitation of every budget, from its known inputs (plan, section 5.3)
    excited = {}
    for replicate in replicates:
        run_windows, data = runs[spec.run_ids(condition)[replicate]]
        for budget in spec.budgets[condition]:
            fitting, validation = benchmark.parts(run_windows, data, budget)
            excited[f"{condition}/r{replicate:02d}/b{budget:02d}"] = {
                "fitting": excitation(fitting, known.nominal_inputs),
                "budget": excitation(fitting + validation, known.nominal_inputs),
            }

    expected = benchmark.expected_keys(spec, condition, replicates)
    recorded = [r["key"] for r in fitted] + [found["key"] for found, _ in outcomes]
    return {
        "excitation": excited,
        "accounting": {
            "expected": len(expected),
            "recorded": len(recorded),
            "complete": sorted(recorded) == sorted(expected),
            "training_failures": sum(1 for found, _ in outcomes if found["failure"] is not None),
            "fit_failures": sum(1 for r in fitted if r["failure"] is not None),
            "families_without_model": sorted(
                k for k, s in selections.items() if s.selected is None
            ),
        },
        "cost": {
            "workers": workers,
            "fits": len(fitted),
            "trainings": len(outcomes),
            "wall_seconds_fits": fit_seconds,
            "wall_seconds_trainings": train_seconds,
            "sum_of_fit_seconds": sum(r["seconds"] for r in fitted),
            "sum_of_training_seconds": sum(f.get("wall_seconds", 0.0) for f, _ in outcomes),
        },
    }


def refuse_a_data_directory_inside_the_repository() -> Path:
    found, repository = data_dir().resolve(), repository_root().resolve()
    if found == repository or repository in found.parents:
        raise Refused(
            f"PT_DATA_DIR resolves to {found}, inside the repository {repository}; set it to a "
            "new directory outside the repository and outside OneDrive"
        )
    return found


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", choices=sorted(benchmark.REGISTERED.sets), required=True)
    parser.add_argument("--replicates", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exports", type=Path, default=None)
    options = parser.parse_args(arguments)
    spec = benchmark.REGISTERED
    try:
        target = refuse_a_data_directory_inside_the_repository()
        state = git_state()
        if not state["code_identified"]:
            raise Refused(f"the code is not identified by a clean commit: {state['reason']}")
        all_replicates = list(range(len(spec.run_ids(options.condition))))
        replicates = all_replicates if options.replicates is None else options.replicates
        if not replicates or not set(replicates) <= set(all_replicates):
            raise Refused(f"replicates must be among {all_replicates}, got {replicates}")
        exports = options.exports or target / "available" / "exports"
        export_directory = exports / spec.sets[options.condition].dataset_id
        if not export_directory.is_dir():
            raise Refused(f"no export of the development set at {export_directory}")
        before = fingerprints(export_directory)
        known, runs, hashes = load_condition(exports, spec, options.condition)
    except Refused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    directory = new_run_directory(EXPERIMENT, state)
    began = time.perf_counter()
    found = execute(
        spec,
        options.condition,
        replicates,
        known,
        runs,
        modeller_values(),
        directory,
        options.workers,
    )
    unchanged = fingerprints(export_directory) == before
    summary = {
        "experiment": "M1-E03, development runs (I6)",
        "provenance": {
            "run_id": directory.name,
            "command": " ".join(["python", *sys.argv]),
            "git": state,
            "commit": state["commit"],
            "code_identified": state["code_identified"],
            "environment": learning_environment(),
            "specification": spec.as_dict(),
            "specification_sha256": spec.digest(),
            "condition": options.condition,
            "replicates": replicates,
            "data": {"dataset_id": spec.sets[options.condition].dataset_id, "runs": hashes},
            "data_dir": str(target),
        },
        **found,
        "exports_unchanged": unchanged,
        "seconds": time.perf_counter() - began,
    }
    write_once(directory / "summary.json", summary)
    print(f"{directory}: complete {found['accounting']['complete']}, exports unchanged {unchanged}")
    return 0 if found["accounting"]["complete"] and unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
