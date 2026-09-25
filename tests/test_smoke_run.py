"""The tables of the smoke run of I1 keep every declared replicate: a model that was not
trained is counted as a training failure of that replicate, never left out (docs/m1_plan.md,
section 9.3). The script itself is imported and run on synthetic replicates; its logic is
not copied here."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from m1_support import NOMINAL, modeller_run, observations, random_corners, step_inputs
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, STEP_LAYOUT, find_windows, window_data
from process_transfer.models.fitting import DEFAULT_STARTS, FitSettings
from process_transfer.models.mechanistic import (
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
)

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "m1_i1_smoke_run.py"
KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
MODELLER = modeller_values()
TRAINED = FitSettings(starts=DEFAULT_STARTS[:1])
NOT_TRAINED = FitSettings(starts=DEFAULT_STARTS[:1], max_evaluations=1)
REPLICATES = ("target.p3.e0.x2.n0", "target.p3.e1.x2.n0", "target.p3.e2.x2.n0")


@pytest.fixture(scope="module")
def smoke():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("m1_i1_smoke_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def replicates_data():  # noqa: ANN201
    """Three P3 runs of two excursions each, with the lead of M1, simulated with the
    modeller's equations, and one run of a single-input step."""
    textbook = MechanisticParameters.textbook(MODELLER)
    truth = MechanisticModel(
        "truth", KNOWN, MechanisticParameters.from_k_350(1.3 * textbook.k_350, 9200.0, 1400.0)
    )
    p3_runs = {}
    for seed, run in enumerate(REPLICATES):
        observed = modeller_run(truth.rhs, random_corners(2, seed), noise_seed=100 + seed, run=run)
        found = find_windows(observed, NOMINAL, P3_LAYOUT)
        p3_runs[run] = (observed, found, window_data(observed, found.windows))
    step = observations(
        step_inputs(3, +1),
        measured=np.tile([200.0, 355.0], (301, 1)),
        run="target.step.tc-up.l600.h600.r600.n0",
    )
    step_data = window_data(step, find_windows(step, NOMINAL, STEP_LAYOUT).windows)
    return p3_runs, list(step_data)


def run(smoke, replicates_data, settings_for):  # noqa: ANN001, ANN201
    p3_runs, step_data = replicates_data
    replicates = smoke.run_replicates(p3_runs, step_data, KNOWN, MODELLER, 2, settings_for)
    return replicates, smoke.failure_tables(replicates)


def counts(table) -> tuple[int, int, int]:  # noqa: ANN001
    return table.replicates, table.scored, table.training_failures


def test_when_every_fit_is_trained_every_replicate_is_scored(smoke, replicates_data) -> None:  # noqa: ANN001
    replicates, tables = run(smoke, replicates_data, lambda replicate: TRAINED)
    assert set(tables) == {
        "MN, own windows",
        "MN, other P3 runs",
        "MN, steps",
        "MR_F, own windows",
        "MR_F, other P3 runs",
        "MR_F, steps",
        "MR_F, V, held out from MR_F",
        "MR, own windows",
        "MR, other P3 runs",
        "MR, steps",
    }
    for label, table in tables.items():
        assert counts(table) == (3, 3, 0), label
    json.dumps(smoke.plain(replicates))  # the summary can be written


def test_a_replicate_whose_fits_fail_stays_in_the_tables(smoke, replicates_data) -> None:  # noqa: ANN001
    failing = REPLICATES[2]
    replicates, tables = run(
        smoke, replicates_data, lambda replicate: NOT_TRAINED if replicate == failing else TRAINED
    )
    for label, table in tables.items():
        expected = (3, 3, 0) if label.startswith("MN") else (3, 2, 1)
        assert counts(table) == expected, label
    for name in ("MR", "MR_F"):
        assert replicates[failing]["fits"][name].training_failure is not None
        for view, found in replicates[failing]["scores"][name].items():
            outcome = found["outcome"]
            assert outcome.score is None and outcome.training_failure is not None, view
            assert outcome.windows > 0 and found["implied_terms"] is None
    json.dumps(smoke.plain(replicates))


def test_when_no_fit_is_trained_the_tables_still_count_every_replicate(
    smoke,  # noqa: ANN001
    replicates_data,  # noqa: ANN001
) -> None:
    replicates, tables = run(smoke, replicates_data, lambda replicate: NOT_TRAINED)
    for name in ("MR_F", "MR"):
        views = [label for label in tables if label.startswith(f"{name}, ")]
        assert len(views) == (4 if name == "MR_F" else 3)
        for label in views:
            assert counts(tables[label]) == (3, 0, 3), label
    for label in ("MN, own windows", "MN, other P3 runs", "MN, steps"):
        assert counts(tables[label]) == (3, 3, 0), label  # MN is never trained
    json.dumps(smoke.plain(replicates))
