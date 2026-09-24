"""Development smoke run of I1 of M1: the evaluation contract and MN, MR and MR_F on the
exports of M0.

What this is. A bounded run that shows the code of I1 working end to end on real exports:
windows found from the known inputs, budgets divided into F and V, MR and MR_F fitted from
every declared start, MN, MR_F and MR rolled out and scored, failures, validity bounds and
implied terms recorded, and the reference integration checked against far tighter ones on
real windows. It is development, not an experiment of M1: nothing is registered, no
hypothesis is tested, and no number it prints is a result of M1 or of a benchmark. The
data sets of M0 were built to verify the data path and are not the benchmark of M1 (section
5.1 of the plan); every use of them is exploratory, and their P3 runs have no lead, so the
first excursion of each forms no window.

What it reads. The exports ``m0-e05`` (P3, three target runs of ten excursions) and
``m0-e07`` (single-input steps, eight target runs), opened and verified by
``open_export_directory``, read only; the known parameters of the target from them; the
modeller's values of ``configs/modeller_cstr.yaml``. It imports nothing of the simulation or
the generator. What it writes goes to a new directory under
``PT_DATA_DIR/development/m1-i1-smoke/``.

What it does, for each target run of ``m0-e05`` taken as a replicate: its nine windows
form a budget of nine, F the first eight and V the last; MR is fitted on the nine and MR_F
on F; MN, MR_F and MR are evaluated on the windows of the replicate, with the role those
windows played for each model, on the windows of the two other runs and on the eight steps
of ``m0-e07``, both held out from every model. The accuracy of the reference integration is
checked on every window of the first run and on the eight steps, for MN and for the MR of
the first run, against LSODA and DOP853 at rtol = 1e-12.

Usage, with PT_DATA_DIR set to a directory outside the repository and the exports of M0
given explicitly:

    python experiments/m1_i1_smoke_run.py --exports <PT_DATA_DIR of M0>/available/exports
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from process_transfer.data.export import Export, open_export_directory
from process_transfer.data.paths import output_dir
from process_transfer.data.provenance import environment, git_state, new_directory
from process_transfer.evaluation.budgets import excitation, split_budget
from process_transfer.evaluation.metrics import Role
from process_transfer.evaluation.outcomes import failure_table, replicate_outcome
from process_transfer.evaluation.physics import (
    check_implied_terms,
    implied_terms_along,
    validity_violations,
)
from process_transfer.evaluation.plant import KnownPlant, read_known_plant
from process_transfer.evaluation.windows import WindowData, find_windows, layout_for, window_data
from process_transfer.measurement.observations import Observations
from process_transfer.models.fitting import FitResult, covariance, fit_mechanistic
from process_transfer.models.mechanistic import MechanisticModel, modeller_values, nominal_model
from process_transfer.models.rollout import EVALUATION_SETTINGS, RolloutSettings, predict_window

PLANT = "target"
BUDGET = 9  # every window of a P3 run of M0: excursions 2 to 10
TIGHT = {
    "LSODA rtol 1e-12": RolloutSettings("LSODA", 1e-12, 1e-10, 10_000_000),
    "DOP853 rtol 1e-12": RolloutSettings("DOP853", 1e-12, 1e-10, 10_000_000),
}


def plain(value: object) -> object:
    """What json can write: dataclasses as dictionaries, arrays as lists, enums as values."""
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
    return value


def target_runs(
    export: Export, operating_mode: str, known: KnownPlant
) -> dict[str, tuple[Observations, object, tuple[WindowData, ...]]]:
    """The observations, windows and window data of every target run of an export."""
    layout = layout_for(operating_mode)
    found = {}
    for run_id in export.run_ids:
        if export.run(run_id)["plant_id"] != PLANT:
            continue
        observations = export.observations(run_id)
        run_windows = find_windows(observations, known.nominal_inputs, layout)
        found[run_id] = (observations, run_windows, window_data(observations, run_windows.windows))
    return found


def evaluated(
    model: MechanisticModel,
    replicate: str,
    data: Sequence[WindowData],
    role: Role,
    known: KnownPlant,
) -> dict[str, object]:
    """Roll a model out on windows; its outcome, violations and implied terms."""
    results = [predict_window(model, window) for window in data]
    violations, incompatible, negative = [], 0, 0
    for window, result in zip(data, results, strict=True):
        if isinstance(result, np.ndarray):
            violations.extend(validity_violations(window, result, known))
            check = check_implied_terms(implied_terms_along(model.rhs, window, result, known))
            incompatible += check.incompatible_heat_flow
            negative += check.negative_rate
    outcome = replicate_outcome(model.name, replicate, BUDGET, data, results, role, violations)
    return {
        "outcome": outcome,
        "implied_terms": {"negative_rate": negative, "incompatible_heat_flow": incompatible},
    }


def integration_accuracy(
    models: Sequence[MechanisticModel], data: Sequence[WindowData]
) -> dict[str, list[float]]:
    """The largest error of the reference integration against tighter ones, in sigmas."""
    worst = {}
    for model in models:
        for label, settings in TIGHT.items():
            largest = np.zeros(2)
            for window in data:
                reference = predict_window(model, window, settings)
                evaluated_states = predict_window(model, window, EVALUATION_SETTINGS)
                error = np.max(np.abs(evaluated_states - reference), axis=0) / window.noise_std
                largest = np.maximum(largest, error)
            worst[f"{model.name} against {label}"] = largest.tolist()
    return worst


def describe(fit: FitResult) -> str:
    if fit.parameters is None:
        return f"{fit.name}: training failure: {fit.training_failure.reason}"
    p = fit.parameters
    return (
        f"{fit.name}: k_350 {p.k_350:.5g} 1/s, E/R {p.activation_temperature:.5g} K, "
        f"UA {p.ua:.5g} W/K, J {fit.objective:.4f} on {fit.readings} readings; starts "
        f"{[record.outcome for record in fit.starts]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exports", type=Path, required=True, help="the exports of M0")
    arguments = parser.parse_args()
    started = time.perf_counter()
    state = git_state()
    directory = new_directory(output_dir("development", "m1-i1-smoke"), state)

    p3 = open_export_directory(arguments.exports / "m0-e05")
    steps = open_export_directory(arguments.exports / "m0-e07")
    known = read_known_plant(p3, PLANT)
    if read_known_plant(steps, PLANT) != known:
        raise ValueError("m0-e05 and m0-e07 state different known parameters of the target")
    modeller = modeller_values()
    mn = nominal_model(known, modeller)
    p3_runs = target_runs(p3, "p3", known)
    step_data = [
        data for _, _, run_data in target_runs(steps, "step", known).values() for data in run_data
    ]

    replicates = {}
    for replicate, (observations, run_windows, _) in p3_runs.items():
        split = split_budget(run_windows, BUDGET)
        every = window_data(observations, split.windows)
        fitting = window_data(observations, split.fitting)
        validation = window_data(observations, split.validation)
        fits = {
            "MR": fit_mechanistic("MR", every, known, modeller),
            "MR_F": fit_mechanistic("MR_F", fitting, known, modeller),
        }
        others = [
            data
            for run, (_, _, run_data) in p3_runs.items()
            if run != replicate
            for data in run_data
        ]
        models = [mn] + [fit.model(known) for fit in fits.values() if fit.parameters]
        scores = {}
        for model in models:
            own = Role.HELD_OUT if model.name == "MN" else Role.FITTING
            scores[model.name] = {
                "own windows": evaluated(model, replicate, every, own, known),
                "other P3 runs": evaluated(model, replicate, others, Role.HELD_OUT, known),
                "steps": evaluated(model, replicate, step_data, Role.HELD_OUT, known),
            }
            if model.name == "MR_F":
                scores["MR_F"]["V, held out from MR_F"] = evaluated(
                    model, replicate, validation, Role.HELD_OUT, known
                )
        mr = fits["MR"]
        replicates[replicate] = {
            "budget": split,
            "skipped": run_windows.skipped,
            "excitation": {
                "fitting": excitation(fitting, known.nominal_inputs),
                "all": excitation(every, known.nominal_inputs),
            },
            "fits": fits,
            "spread": {name: fit.spread() for name, fit in fits.items()},
            "covariance of MR, as if its structure were right": (
                covariance(mr.parameters, every, known) if mr.parameters else None
            ),
            "scores": scores,
        }

    first_run = next(iter(p3_runs))
    first_mr = replicates[first_run]["fits"]["MR"]
    accuracy_models = [mn] + ([first_mr.model(known)] if first_mr.parameters else [])
    accuracy = integration_accuracy(accuracy_models, list(p3_runs[first_run][2]) + step_data)
    tables = {}
    for name in ("MN", "MR_F", "MR"):
        for view in ("other P3 runs", "steps"):
            found = [
                r["scores"][name][view]["outcome"]
                for r in replicates.values()
                if name in r["scores"]
            ]
            tables[f"{name}, {view}"] = failure_table(found) if found else None
    summary = {
        "what": (
            "Development smoke run of I1 of M1 on the exports of M0. Not an experiment, not "
            "registered, not a result of M1 and not a benchmark result."
        ),
        "attempt": {"git": state, "environment": environment()},
        "exports": {
            export.dataset_id: {run: export.run(run)["content_sha256"] for run in export.run_ids}
            for export in (p3, steps)
        },
        "known_plant": known,
        "modeller": {
            "k0": modeller.kinetics.k0.si,
            "activation_temperature": modeller.kinetics.activation_temperature.si,
            "ua": modeller.heat_transfer.UA.si,
        },
        "evaluation_rollout": EVALUATION_SETTINGS,
        "integration_accuracy_in_sigmas": accuracy,
        "replicates": replicates,
        "failure_tables": tables,
        "seconds": time.perf_counter() - started,
    }
    text = json.dumps(plain(summary), indent=2)
    (directory / "summary.json").write_text(text, encoding="utf-8")

    print(f"development smoke run of I1, not a result of M1; written to {directory}")
    for replicate, record in replicates.items():
        print(replicate)
        for fit in record["fits"].values():
            print(f"  {describe(fit)}")
        for name, views in record["scores"].items():
            parts = []
            for view, found in views.items():
                score = found["outcome"].score
                parts.append(f"{view} J {score:.3f}" if score is not None else f"{view} failed")
            print(f"  {name}: " + ", ".join(parts))
    print(f"integration accuracy, in sigmas: {accuracy}")
    print(f"{summary['seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
