"""I3 of M1: the checks that follow Codex's audit of ``1a9fad4``. Exploratory, on the
development data of M0 and the synthetic runs of the pilot; nothing here is a result of M1.

Two parts, both bounded:

* ``reproduce``: five trainings of the pilot run again with the corrected code: four of the
  phase ``grid`` on the runs of M0 and the synthetic case of the phase ``cost`` whose training
  scheme departed from its equation. Each is compared, checkpoint by checkpoint, with the
  summary the pilot wrote: the selected step, the loss on F, the criterion on V and the
  difference of the two rollouts. The corrections were made so that they act only outside
  the ordinary scale; this shows whether they changed an ordinary result.
* ``substeps``: a complete training with two and with four steps of the scheme per row, for a
  representative case of the pilot on the runs of M0 and for the synthetic case above. For
  each, the selected checkpoint, its criterion, and the largest difference between the
  training scheme and the reference rollout on F and on V at that checkpoint. A training with
  four steps is a new training, not the old weights rolled out more finely.

The pilot's summaries are read from ``--pilot``, the directory of its runs. The data are
those of the pilot: the export ``m0-e05``, read only, and the synthetic runs that
``experiments/m1_i3_pilot.py`` simulates with the model side's rollout. It imports nothing of
the simulation, the generator or the private branch.

Usage, with PT_DATA_DIR outside the repository:

    python experiments/m1_i3_audit_checks.py --exports <exports> --pilot <pilot runs> --workers 8
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from process_transfer.data.paths import data_dir, repository_root  # noqa: E402
from process_transfer.data.provenance import git_state, new_run_directory  # noqa: E402
from process_transfer.models.fitting import fit_mechanistic  # noqa: E402
from process_transfer.models.mechanistic import modeller_values  # noqa: E402
from process_transfer.models.training import (  # noqa: E402
    Configuration,
    TrainingSettings,
    learning_environment,
    train,
    training_seed,
)

_PILOT_SCRIPT = Path(__file__).resolve().parent / "m1_i3_pilot.py"
_spec = importlib.util.spec_from_file_location("m1_i3_pilot", _PILOT_SCRIPT)
pilot = importlib.util.module_from_spec(_spec)
sys.modules["m1_i3_pilot"] = pilot
_spec.loader.exec_module(pilot)

GRID_RUN = "20260927T221514Z_fc563c8"
COST_RUN = "20260927T222728Z_fc563c8"
RATES = {"BN": 1e-3, "HK": 1e-2, "HU": 1e-3, "HKU": 1e-2}  # chosen by the pilot's rule
# (phase, family index in its grid list, run, budget): the grid's index fixes the seed
REPRODUCE = (
    ("grid", ("HK", 0), "target.p3.e0.x10.n0", 5),
    ("grid", ("BN", 0), "target.p3.e0.x10.n0", 9),
    ("grid", ("HKU", 3), "target.p3.e1.x10.n0", 2),
    ("grid", ("HU", 0), "target.p3.e2.x10.n0", 9),
    ("cost", ("HK", 0), "target.p3.e9001.x40.n0", 20),
)
SUBSTEPS = (
    ("grid", ("HK", 0), "target.p3.e0.x10.n0", 9),
    ("cost", ("HK", 0), "target.p3.e9001.x40.n0", 20),
)

_CONTEXT: dict[str, object] = {}


def _initialise(exports: str) -> None:
    known, runs = pilot.m0_runs(Path(exports))
    runs.update(pilot.synthetic_runs(known))
    pilot._initialise({"known": known, "modeller": modeller_values(), "runs": runs})
    _CONTEXT.update(known=known)


def task(item: tuple[str, tuple[str, int], str, int, int]) -> dict[str, object]:
    phase, (family, index), run_id, budget, substeps = item
    known = _CONTEXT["known"]
    fitting, validation = pilot._part(run_id, budget)
    configuration: Configuration = pilot.GRID[family][index]
    replicate = sorted(r for r in pilot._CONTEXT["runs"] if (".x40." in r) == (phase == "cost"))
    seed = training_seed(pilot.SEED_BASE, replicate.index(run_id), index)
    start = None
    if family != "BN":
        start = fit_mechanistic("MR_F", fitting, known, modeller_values()).parameters
    settings = TrainingSettings(
        learning_rate=RATES[family], max_steps=3000, validation_every=100, substeps=substeps
    )
    began = time.perf_counter()
    record = train(configuration, settings, fitting, validation, known, seed, start=start)
    selected = None if record.selected is None else record.checkpoints[record.selected]
    return {
        "phase": phase,
        "configuration": configuration.label,
        "run": run_id,
        "budget": budget,
        "substeps": substeps,
        "seed": seed,
        "failure": None if record.failure is None else record.failure.reason,
        "selected_step": None if selected is None else selected.step,
        "criterion": record.criterion,
        "schemes_differ_on_v": None if selected is None else selected.schemes_differ_in_sigmas,
        "schemes_differ_on_f": record.fitting_schemes_differ_in_sigmas,
        "bound_steps": record.bound_steps,
        "checkpoints": [
            {
                "step": c.step,
                "fitting": c.fitting_loss,
                "validation": c.validation,
                "schemes_differ_in_sigmas": c.schemes_differ_in_sigmas,
            }
            for c in record.checkpoints
        ],
        "seconds": time.perf_counter() - began,
    }


def compare(found: dict[str, object], stored: dict[str, object]) -> dict[str, object]:
    """Checkpoint by checkpoint, the new training against the pilot's record of it."""

    def relative(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None if a is b else math.inf
        return 0.0 if a == b else abs(a - b) / max(abs(a), abs(b))

    pairs = list(zip(found["checkpoints"], stored["checkpoints"], strict=True))
    worst = {
        key: max((relative(a[key], b[key]) or 0.0) for a, b in pairs)
        for key in ("fitting", "validation", "schemes_differ_in_sigmas")
    }
    return {
        "same_selected_step": found["selected_step"] == stored["selected_step"],
        "same_failure": (found["failure"] is None) == (stored["failure"] is None),
        "fitting_losses_equal_bit_for_bit": all(a["fitting"] == b["fitting"] for a, b in pairs),
        "largest_relative_difference": worst,
        "criterion": [found["criterion"], stored["criterion"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    root = repository_root().resolve()
    target = data_dir().resolve()
    if target == root or root in target.parents:
        print(f"PT_DATA_DIR resolves to {target}, inside the repository; refused", file=sys.stderr)
        return 1
    state = git_state()
    began = time.perf_counter()
    stored = {
        phase: json.loads((arguments.pilot / run / f"summary_{phase}.json").read_text())
        for phase, run in (("grid", GRID_RUN), ("cost", COST_RUN))
    }
    # two steps per row for every case, the reproduced ones and those of SUBSTEPS that are
    # not among them; then four steps for the cases of SUBSTEPS
    extra = [item for item in SUBSTEPS if item not in REPRODUCE]
    items = (
        [(*item, 2) for item in REPRODUCE]
        + [(*item, 2) for item in extra]
        + [(*item, 4) for item in SUBSTEPS]
    )
    with ProcessPoolExecutor(
        arguments.workers, initializer=_initialise, initargs=(str(arguments.exports),)
    ) as pool:
        results = list(pool.map(task, items))
    reproduced = []
    for result in results[: len(REPRODUCE)]:
        match = [
            t
            for t in stored[result["phase"]]["trainings"]
            if t["configuration"] == result["configuration"]
            and t["run"] == result["run"]
            and t["budget"] == result["budget"]
        ]
        (original,) = match
        reproduced.append(
            {
                **{k: result[k] for k in ("configuration", "run", "budget")},
                **compare(result, original),
            }
        )
    # two steps per row: the reproduced trainings of the two cases of SUBSTEPS
    two = {(r["configuration"], r["run"], r["budget"]): r for r in results if r["substeps"] == 2}
    substeps = []
    for result in (r for r in results if r["substeps"] == 4):
        key = (result["configuration"], result["run"], result["budget"])
        for found in (two[key], result):
            substeps.append(
                {
                    k: found[k]
                    for k in (
                        "configuration",
                        "run",
                        "budget",
                        "substeps",
                        "failure",
                        "selected_step",
                        "criterion",
                        "schemes_differ_on_f",
                        "schemes_differ_on_v",
                        "bound_steps",
                        "seconds",
                    )
                }
            )
    summary = {
        "provenance": {
            "git": state,
            "environment": learning_environment(),
            "data_dir": str(target),
        },
        "pilot_runs": {"grid": GRID_RUN, "cost": COST_RUN},
        "reproduced": reproduced,
        "substeps": substeps,
        "trainings": results,
        "workers": arguments.workers,
        "wall_seconds": time.perf_counter() - began,
        "sum_of_fit_seconds": sum(r["seconds"] for r in results),
    }
    directory = new_run_directory("m1_i3_audit_checks", state)
    (directory / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(directory)
    print(json.dumps({k: summary[k] for k in ("reproduced", "substeps")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
