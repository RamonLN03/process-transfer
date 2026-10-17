"""M1-E03, the development runs of I6: every fit and training that the registration calls
for on the development sets, with each configuration chosen on V, and nothing else.

Registered in ``docs/experiment_log.md`` (M1-E03); the procedure is that of
``process_transfer.models.benchmark``, and D-044 and D-045 record how it is executed. The
checks before a full run are those of section 10 of the registration: ingestion, disk
space, and the cost of the first replicate, which this script runs with ``--replicates 0``.

What the script reads: the exports of ``m1-e03-train-a10`` and ``m1-e03-train-a5``, read
only, under ``PT_DATA_DIR/available/exports`` or ``--exports``. Each is checked to be the
registered development set, with exactly its runs. Nothing else is read: no test set, no
plant configuration, no definition of a data set, nothing of the private branch or of the
truth.

What it writes, in a new directory ``PT_DATA_DIR/experiments/m1_e03_development/<run id>``
that no other run uses, as the run goes:

* ``provenance.json``, before the first task: commit and clean tree, environment with JAX,
  the specification and its SHA-256, the identity of the observable data (the content of
  every run, the known parameters of the plant, every file of the export), the command;
* ``records/<key>.json`` for every task as soon as it ends, a failure included, and
  ``models/<key>.json`` for the selected checkpoint of every configuration that has one,
  each written once and entered with its SHA-256 in ``ledger.jsonl``;
* ``selections.json``, the configuration each family keeps, chosen on V alone, for the
  families whose every configuration has an outcome;
* ``summary.json`` at the end, whatever happened: complete or not, the tasks still
  pending, the errors of infrastructure, the accounting, the excitation and the cost.

A training that fails, by a loss that is not finite for instance, is a scientific outcome
and is recorded. An exception of the software, a dead worker or an interruption is an
error of infrastructure: the task stays pending, nothing is retried, and the run is
incomplete. Records already written stay as they are.

``--combine`` puts runs made apart together, such as the first replicate and then the
others, after checking their provenance, the integrity of every record and model against
its ledger, and that no task is recorded twice. It writes a new directory under
``experiments/m1_e03_combined`` and changes nothing in the runs it reads.

    python experiments/m1_e03_development.py --condition A10 --replicates 0 --workers 8
    python experiments/m1_e03_development.py --combine <run directory> <run directory> ...

Exit codes: 0 when every expected task has an outcome and the exports were not changed;
1 when the exports changed during the run; 2 when the run is refused before it starts, or
runs cannot be combined; 3 when the run is incomplete, by errors of infrastructure or an
interruption. ``summary.json`` says which: ``complete``, ``incomplete`` (tasks pending
after errors of infrastructure) or ``interrupted``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# one thread per process, before JAX is imported
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from process_transfer.data.export import open_export_directory  # noqa: E402
from process_transfer.data.paths import data_dir, output_dir, repository_root  # noqa: E402
from process_transfer.data.provenance import (  # noqa: E402
    git_state,
    new_directory,
    new_run_directory,
)
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
from process_transfer.models.mechanistic import modeller_values  # noqa: E402
from process_transfer.models.persistence import save_selected  # noqa: E402
from process_transfer.models.training import learning_environment  # noqa: E402

EXPERIMENT = "m1_e03_development"
Runs = dict[str, tuple[RunWindows, tuple[WindowData, ...]]]
COMPLETE, INCOMPLETE, INTERRUPTED = "complete", "incomplete", "interrupted"


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


def write_once(path: Path, content: object) -> str:
    """A file is written once, never replaced; the SHA-256 of what was written is returned."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plain(content), indent=1, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_name(key: str) -> str:
    return key.replace("/", "_") + ".json"


class Ledger:
    """``ledger.jsonl``: one line for every record and model written, with its SHA-256,
    appended as each is written and never rewritten."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "ledger.jsonl"

    def enter(self, key: str, kind: str, path: Path, sha256: str) -> None:
        line = {
            "key": key,
            "kind": kind,
            "file": path.relative_to(self.directory).as_posix(),
            "sha256": sha256,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")

    def read(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]


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


def fingerprints(directory: Path) -> dict[str, str]:
    return {
        p.relative_to(directory).as_posix(): file_sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def load_condition(
    exports: Path, spec: benchmark.Specification, condition: str
) -> tuple[KnownPlant, Runs, dict[str, Any]]:
    """The known plant, the windows of every run, and the identity of the observable data,
    from the export of the condition's development set."""
    registered = spec.sets[condition].dataset_id
    directory = exports / registered
    files = fingerprints(directory)
    export = open_export_directory(directory)
    plants = {run_id: str(export.run(run_id)["plant_id"]) for run_id in export.run_ids}
    check_export(export.dataset_id, export.run_ids, plants, spec.run_ids(condition), registered)
    known = read_known_plant(export, spec.plant)
    runs: Runs = {}
    content = {}
    for run_id in spec.run_ids(condition):
        observations = export.observations(run_id)
        found = find_windows(observations, known.nominal_inputs, layout_for("p3"))
        runs[run_id] = (found, window_data(observations, found.windows))
        content[run_id] = str(export.run(run_id)["content_sha256"])
    identity = benchmark.data_identity(
        registered, known, content, files, export.manifest.get("attempt")
    )
    return known, runs, identity


# --------------------------------------------------------------------------- #
# The development runs of one condition
# --------------------------------------------------------------------------- #


def _each(function: Callable, items: Sequence, workers: int, context: Mapping[str, Any]):  # noqa: ANN202
    """Run ``function`` on every item and yield (item, result, error) as each ends, so that
    a result can be written before the next one. An exception is yielded as an error of
    infrastructure, and the other items go on. Processes are started by spawn on every
    platform: a fork of a process that has started JAX can deadlock."""
    if workers > 1 and len(items) > 1:
        with ProcessPoolExecutor(
            workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=benchmark.initialise_worker,
            initargs=(context,),
        ) as pool:
            futures = {pool.submit(function, item): item for item in items}
            try:
                for future in as_completed(futures):
                    try:
                        yield futures[future], future.result(), None
                    except Exception as error:  # noqa: BLE001 - recorded, never swallowed
                        yield futures[future], None, error
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        return
    benchmark.initialise_worker(context)
    for item in items:
        try:
            result = function(item)
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            yield item, None, error
            continue
        yield item, result, None


def _error(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


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
    """Every task of ``condition`` at ``replicates``; each record and model written as its
    task ends; the hybrids started from MR_F of their replicate, budget and analysis; the
    selection of every family whose every configuration has an outcome; the accounting.
    ``summary.json`` is written whatever happens, an interruption included. The provenance
    must have been written first."""
    if not (directory / "provenance.json").is_file():
        raise ValueError(f"{directory}: provenance.json must be written before any task")
    fits, trainings = benchmark.tasks(spec, condition)
    fits = [t for t in fits if t.replicate in replicates]
    trainings = [t for t in trainings if t.replicate in replicates]
    context = {"spec": spec, "known": known, "runs": runs, "modeller": modeller}
    ledger = Ledger(directory)
    records: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, str]] = {}
    pending: dict[str, str] = {t.key: "not started" for t in fits + trainings}
    cost = {"workers": workers, "wall_seconds_fits": 0.0, "wall_seconds_trainings": 0.0}
    state = {"interrupted": None}

    def keep(record: dict[str, Any]) -> None:
        path = directory / "records" / file_name(record["key"])
        ledger.enter(record["key"], "record", path, write_once(path, record))
        records[record["key"]] = record
        pending.pop(record["key"], None)

    try:
        began = time.perf_counter()
        for task, record, error in _each(benchmark.fit_task, fits, workers, context):
            if error is not None:
                errors[task.key] = _error(error)
                pending[task.key] = "error of infrastructure"
            else:
                keep(record)
        cost["wall_seconds_fits"] = time.perf_counter() - began

        to_train = []
        for task in trainings:
            if task.family == "BN":
                to_train.append((task, None))
                continue
            start = records.get(task.start_key)
            if start is None:  # the fit did not end: the training waits, it is not a failure
                pending[task.key] = f"its start {task.start_key} has no record"
            elif start["model"] is None:  # the fit failed: a scientific outcome
                keep(benchmark.blocked(task, f"{task.start_key}: {start['failure']}"))
            else:
                to_train.append((task, start["model"]))
        began = time.perf_counter()
        (directory / "models").mkdir(parents=True, exist_ok=True)
        for (task, _), result, error in _each(benchmark.train_task, to_train, workers, context):
            if error is not None:
                errors[task.key] = _error(error)
                pending[task.key] = "error of infrastructure"
                continue
            found, trained = result
            if trained.selected is not None:
                path = directory / "models" / file_name(found["key"])
                found["model_file"] = path.relative_to(directory).as_posix()
                found["model_sha256"] = save_selected(trained, known, path)
                ledger.enter(found["key"], "model", path, file_sha256(path))
            keep(found)
        cost["wall_seconds_trainings"] = time.perf_counter() - began
    except BaseException as interruption:
        state["interrupted"] = _error(interruption)
        raise
    finally:
        summary = _close(
            spec,
            condition,
            replicates,
            known,
            runs,
            directory,
            records,
            errors,
            pending,
            cost,
            state,
        )
    return summary


def _close(
    spec, condition, replicates, known, runs, directory, records, errors, pending, cost, state
):  # noqa: ANN001, ANN202, PLR0913
    """The selections and the summary, from what was recorded."""
    sizes = benchmark.families(spec, condition, replicates)
    selections = {}
    for key, size in sorted(sizes.items()):
        members = [
            r
            for k, r in records.items()
            if benchmark.family_key(k) == key and r["kind"] == "training"
        ]
        if len(members) == size:
            selections[key] = benchmark.select(key, members)
    write_once(directory / "selections.json", selections)

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
    complete = set(records) == expected and not errors and state["interrupted"] is None
    summary = {
        "status": COMPLETE
        if complete
        else INTERRUPTED
        if state["interrupted"] is not None
        else INCOMPLETE,
        "interrupted": state["interrupted"],
        "errors_of_infrastructure": errors,
        "pending": dict(sorted(pending.items())),
        "accounting": {
            "expected": len(expected),
            "recorded": len(records),
            "complete": complete,
            "unexpected": sorted(set(records) - expected),
            "training_failures": sorted(
                k for k, r in records.items() if r["kind"] == "training" and r["failure"]
            ),
            "fit_failures": sorted(
                k for k, r in records.items() if r["kind"] != "training" and r["failure"]
            ),
            "families_selected": len(selections),
            "families_expected": len(sizes),
            "families_without_model": sorted(
                k for k, s in selections.items() if s.selected is None
            ),
        },
        "excitation": excited,
        "cost": {
            **cost,
            "fits": sum(r["kind"] != "training" for r in records.values()),
            "trainings": sum(r["kind"] == "training" for r in records.values()),
            "sum_of_fit_seconds": sum(
                r["seconds"] for r in records.values() if r["kind"] != "training"
            ),
            "sum_of_training_seconds": sum(
                r.get("wall_seconds", 0.0) for r in records.values() if r["kind"] == "training"
            ),
        },
    }
    write_once(directory / "summary.json", summary)
    return summary


def refuse_a_data_directory_inside_the_repository() -> Path:
    found, repository = data_dir().resolve(), repository_root().resolve()
    if found == repository or repository in found.parents:
        raise Refused(
            f"PT_DATA_DIR resolves to {found}, inside the repository {repository}; set it to a "
            "new directory outside the repository and outside OneDrive"
        )
    return found


# --------------------------------------------------------------------------- #
# Combining runs made apart
# --------------------------------------------------------------------------- #


def read_run(directory: Path) -> dict[str, Any]:
    """The provenance and records of a run, each record and model checked against the
    ledger: a file that is missing, altered, or not in the ledger is refused."""
    directory = Path(directory)
    if not (directory / "provenance.json").is_file():
        raise ValueError(f"{directory} is not a run of this script: it has no provenance.json")
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    entries = Ledger(directory).read()
    listed = {e["file"]: e for e in entries}
    pairs = [(e["kind"], e["key"]) for e in entries]
    if len(listed) != len(entries) or len(set(pairs)) != len(pairs):
        raise ValueError(f"{directory}: the ledger enters a file or a key twice")
    on_disk = {
        p.relative_to(directory).as_posix()
        for sub in ("records", "models")
        if (directory / sub).is_dir()
        for p in (directory / sub).glob("*.json")
    }
    if on_disk != set(listed):
        raise ValueError(
            f"{directory}: files and ledger disagree: not in the ledger "
            f"{sorted(on_disk - set(listed))}, missing {sorted(set(listed) - on_disk)}"
        )
    for name, entry in listed.items():
        if file_sha256(directory / name) != entry["sha256"]:
            raise ValueError(f"{directory}: {name} does not match its ledger")
    records = {}
    for entry in entries:
        if entry["kind"] == "record":
            record = json.loads((directory / entry["file"]).read_text(encoding="utf-8"))
            model = record.get("model_file")
            if model is not None:
                if listed.get(model, {}).get("key") != entry["key"]:
                    raise ValueError(
                        f"{directory}: the model of {entry['key']} is not in the ledger"
                    )
                saved = json.loads((directory / model).read_text(encoding="utf-8"))
                if saved.get("sha256") != record.get("model_sha256"):
                    raise ValueError(
                        f"{directory}: the model of {entry['key']} is not the one recorded"
                    )
            records[entry["key"]] = record
    ledger = Ledger(directory).path
    return {
        "directory": directory,
        "provenance": provenance,
        "records": records,
        "ledger_sha256": file_sha256(ledger) if ledger.exists() else None,
    }


def combine_runs(
    directories: Sequence[Path], spec: benchmark.Specification | None = None
) -> tuple[Path, dict[str, Any]]:
    """The runs in ``directories``, read and checked, combined by ``benchmark.combine`` into
    a new directory; the runs read are not changed."""
    spec = benchmark.REGISTERED if spec is None else spec
    runs = [read_run(Path(d)) for d in directories]
    combined = benchmark.combine(runs, spec)
    platforms = sorted({r["provenance"]["environment"].get("platform") for r in runs})
    combined["platforms"] = platforms  # reported; not part of the numerical environment
    target = new_directory(output_dir("experiments", "m1_e03_combined"))
    write_once(target / "combined.json", combined)
    return target, combined


def provenance_of(
    directory: Path,
    state: Mapping[str, Any],
    spec: benchmark.Specification,
    condition: str,
    replicates: Sequence[int],
    workers: int,
    identity: Mapping[str, Any],
    target: Path,
) -> dict[str, Any]:
    """What identifies a run, written before its first task."""
    return {
        "experiment": "M1-E03, development runs (I6)",
        "run_id": directory.name,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(["python", *sys.argv]),
        "git": dict(state),
        "commit": state["commit"],
        "code_identified": state["code_identified"],
        "environment": learning_environment(),
        "specification": spec.as_dict(),
        "specification_sha256": spec.digest(),
        "condition": condition,
        "replicates": list(replicates),
        "workers": workers,
        "data": dict(identity),
        "data_dir": str(target),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", choices=sorted(benchmark.REGISTERED.sets))
    parser.add_argument("--replicates", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exports", type=Path, default=None)
    parser.add_argument("--combine", type=Path, nargs="+", metavar="RUN_DIRECTORY")
    options = parser.parse_args(arguments)
    spec = benchmark.REGISTERED
    try:
        target = refuse_a_data_directory_inside_the_repository()
        if options.combine:
            directory, combined = combine_runs(options.combine)
            print(f"{directory}: complete {combined['complete']}")
            return 0
        if options.condition is None:
            raise Refused("--condition is required")
        if options.workers < 1:
            raise Refused(f"--workers must be at least 1, got {options.workers}")
        state = git_state()
        if not state["code_identified"]:
            raise Refused(f"the code is not identified by a clean commit: {state['reason']}")
        all_replicates = list(range(len(spec.run_ids(options.condition))))
        replicates = all_replicates if options.replicates is None else options.replicates
        if (
            not replicates
            or len(set(replicates)) != len(replicates)
            or not set(replicates) <= set(all_replicates)
        ):
            raise Refused(f"replicates must be distinct among {all_replicates}, got {replicates}")
        exports = options.exports or target / "available" / "exports"
        export_directory = exports / spec.sets[options.condition].dataset_id
        if not export_directory.is_dir():
            raise Refused(f"no export of the development set at {export_directory}")
        known, runs, identity = load_condition(exports, spec, options.condition)
    except Refused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    except ValueError as refusal:  # runs that cannot be combined
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    directory = new_run_directory(EXPERIMENT, state)
    provenance = provenance_of(
        directory, state, spec, options.condition, replicates, options.workers, identity, target
    )
    write_once(directory / "provenance.json", provenance)
    try:
        summary = execute(
            spec,
            options.condition,
            replicates,
            known,
            runs,
            modeller_values(),
            directory,
            options.workers,
        )
    except (KeyboardInterrupt, Exception):  # recorded in summary.json by execute
        traceback.print_exc()
        print(f"{directory}: interrupted; the records written are kept", file=sys.stderr)
        return 3
    unchanged = fingerprints(export_directory) == identity["export_files_sha256"]
    write_once(directory / "exports_after.json", {"unchanged": unchanged})
    print(f"{directory}: {summary['status']}, exports unchanged {unchanged}")
    if summary["status"] != COMPLETE:
        return 3
    return 0 if unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
