"""The development runs of M1-E03 (I6): the registered specification, and its executor on
synthetic data.

The specification is checked against the registration and its definitions. The executor
is run on small specifications of the tests, on runs simulated with the modeller's
equations on the truth side of the tests, with excitation seeds of the tests (41 and 42).
No benchmark data are read or generated: a test that reached the generator would fail.

Regression tests for the three findings of Codex's audit of I6 (D-045):

* records and models are written as each task ends, and an interruption or an error of
  infrastructure leaves what was written and a run marked as not complete;
* the identity of the data a run used covers the known parameters of the plant and every
  file of the export, so that equal observations with other known parameters are told
  apart;
* runs made apart are combined only with a complete and valid provenance, the same
  numerical environment and the same data, every record and model intact, and no task
  recorded twice.
"""

import dataclasses
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from m1_support import NOMINAL, modeller_run, random_corners
from process_transfer.config import load_dataset_definition
from process_transfer.data import database
from process_transfer.data.export import export_dataset
from process_transfer.data.parquet_store import DatasetWriter, open_dataset
from process_transfer.data.records import KnownParameter, PlantRecord, RunRecord
from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.plant import NOMINAL_INPUT_UNITS, PROPERTY_UNITS, KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.models import benchmark
from process_transfer.models.linear import fit_linear
from process_transfer.models.mechanistic import (
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
)
from process_transfer.models.persistence import load_selected
from process_transfer.models.rollout import predict_window
from process_transfer.models.training import (
    Configuration,
    TrainingSettings,
    training_seed,
    validation_score,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "m1_e03_development.py"


def load_script():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("m1_e03_development", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m1_e03_development"] = module
    spec.loader.exec_module(module)
    return module


DEV = load_script()
R = benchmark.REGISTERED

# --------------------------------------------------------------------------- #
# The registered specification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("condition", "file"), [("A10", "m1_e03_train_a10.yaml"), ("A5", "m1_e03_train_a5.yaml")]
)
def test_the_development_sets_are_those_registered(condition: str, file: str) -> None:
    definition = load_dataset_definition(ROOT / "configs" / "datasets" / file)
    s = R.sets[condition]
    assert s.dataset_id == definition.dataset_id
    assert s.excitation_seeds == tuple(definition.excitation_seeds)
    assert (s.amplitude, s.n_excursions, s.lead_s) == (
        definition.amplitude,
        definition.n_excursions,
        definition.lead_s,
    )
    assert R.plant == "target" and all(
        d.dataset_id.startswith("m1-e03-train") for d in R.sets.values()
    )


def test_the_specification_is_the_approved_one() -> None:
    assert R.budgets == {"A10": (2, 5, 10, 20, 40), "A5": (10, 40)}
    assert {f: [(c.hidden, c.penalty) for c in g] for f, g in R.grid.items()} == {
        "BN": [((16, 16), 1e-4), ((16, 16), 1e-2), ((32, 32), 1e-4), ((32, 32), 1e-2)],
        "HK": [((8,), 1e-4), ((8,), 1e-2), ((16,), 1e-4), ((16,), 1e-2)],
        "HU": [((8,), 1e-4), ((8,), 1e-2), ((16,), 1e-4), ((16,), 1e-2)],
        "HKU": [((8,), 1e-4), ((8,), 1e-2), ((16,), 1e-4), ((16,), 1e-2)],
    }
    assert R.rates == {"BN": 1e-3, "HK": 1e-2, "HU": 1e-3, "HKU": 1e-2}
    assert (R.steps, R.validation_every, R.substeps) == (3000, 100, 4)
    assert R.seed_bases == {"A10": 20261111, "A5": 20261112}
    assert (R.held_activation, R.held_condition, R.held_budgets, R.held_families) == (
        8750.0,
        "A10",
        (5, 40),
        ("HK",),
    )


def test_four_steps_per_row_are_given_explicitly_not_by_the_default() -> None:
    settings = R.training_settings("HK")
    assert settings.substeps == 4
    assert TrainingSettings(learning_rate=1e-2, max_steps=3000, validation_every=100).substeps == 2
    assert settings == TrainingSettings(1e-2, 3000, 100, substeps=4)


def test_the_counts_of_the_registration() -> None:
    fits10, trainings10 = benchmark.tasks(R, "A10")
    fits5, trainings5 = benchmark.tasks(R, "A5")
    every = fits10 + trainings10 + fits5 + trainings5
    primary = [t for t in every if t.analysis == benchmark.PRIMARY]
    held = [t for t in every if t.analysis == benchmark.HELD]
    assert sum(t.kind == "training" for t in primary) == 1120
    assert sum(t.kind != "training" for t in primary) == 210
    assert sum(t.kind == "training" for t in held) == 80
    assert sorted(t.kind for t in held if t.kind != "training") == ["MR"] * 20 + ["MR_F"] * 20
    keys = [t.key for t in every]
    assert len(set(keys)) == len(keys) == 1450
    assert len({DEV.file_name(k) for k in keys}) == 1450


def test_the_seeds_follow_the_registration() -> None:
    _, trainings = benchmark.tasks(R, "A10")
    _, trainings5 = benchmark.tasks(R, "A5")
    for task in trainings + trainings5:
        assert task.seed == training_seed(R.seed_bases[task.condition], task.replicate, task.index)
    # the same weights at every budget, and for E/R held as for E/R estimated
    by = {(t.replicate, t.family, t.index): set() for t in trainings}
    for t in trainings:
        by[(t.replicate, t.family, t.index)].add(t.seed)
    assert all(len(seeds) == 1 for seeds in by.values())
    assert {t.seed for t in trainings if t.replicate == 0} != {
        t.seed for t in trainings if t.replicate == 1
    }


def test_the_digest_identifies_the_specification() -> None:
    assert R.digest() == R.digest()
    assert dataclasses.replace(R, substeps=2).digest() != R.digest()
    assert dataclasses.replace(R, seed_bases={"A10": 1, "A5": 20261112}).digest() != R.digest()


# --------------------------------------------------------------------------- #
# The executor on synthetic data
# --------------------------------------------------------------------------- #

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
TRUE = MechanisticParameters.from_k_350(0.0177, 9000.0, 1330.0)
SMALL = benchmark.Specification(
    plant="target",
    sets={"A10": benchmark.DevelopmentSet("synthetic-a10", (41, 42), "a10", 3, 60)},
    budgets={"A10": (2, 3)},
    grid={
        "BN": (Configuration("BN", (4,), 1e-4),),
        "HK": (Configuration("HK", (4,), 1e-4), Configuration("HK", (4,), 1e-2)),
    },
    rates={"BN": 1e-3, "HK": 1e-2},
    steps=4,
    validation_every=2,
    substeps=4,
    seed_bases={"A10": 7},
    held_activation=8750.0,
    held_condition="A10",
    held_budgets=(3,),
    held_families=("HK",),
)
# three fits and three trainings a replicate, for the tests of interruptions and errors
TINY = dataclasses.replace(SMALL, budgets={"A10": (2,)}, held_budgets=())
STATE = {"commit": "a" * 40, "code_identified": True, "dirty": False}
ATTEMPT = {"generated_at_utc": "2026-10-17T00:00:00+00:00", "commit": "0" * 40}
TIMES = ("seconds", "wall_seconds")


@pytest.fixture(scope="module")
def observed() -> dict:
    return {
        run_id: modeller_run(
            MechanisticModel("m", KNOWN, TRUE).rhs, random_corners(3, seed), seed, run=run_id
        )
        for seed, run_id in zip((41, 42), SMALL.run_ids("A10"), strict=True)
    }


@pytest.fixture(scope="module")
def runs(observed: dict) -> dict:
    found = {}
    for run_id, observations in observed.items():
        windows = find_windows(observations, NOMINAL, P3_LAYOUT)
        found[run_id] = (windows, window_data(observations, windows.windows))
    return found


def identity_of(observed: dict, known: KnownPlant = KNOWN) -> dict:
    content = {run_id: o.content_digest() for run_id, o in observed.items()}
    return benchmark.data_identity("synthetic-a10", known, content, {}, None)


def begin(directory: Path, spec: benchmark.Specification, replicates: list, observed: dict) -> Path:
    """The provenance of a run of the tests, written before its first task as ``main``
    writes it."""
    provenance = DEV.provenance_of(
        directory, STATE, spec, "A10", replicates, 1, identity_of(observed), directory
    )
    DEV.write_once(directory / "provenance.json", provenance)
    return directory


@pytest.fixture(scope="module")
def executed(
    runs: dict, observed: dict, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict]:
    directory = begin(tmp_path_factory.mktemp("development"), SMALL, [0], observed)
    out = DEV.execute(SMALL, "A10", [0], KNOWN, runs, modeller_values(), directory)
    return directory, out


@pytest.fixture(scope="module")
def blocked_run(
    runs: dict, observed: dict, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict, list]:
    """Replicate 1, with every MR_F failing: its hybrids are blocked, and BN is trained."""
    real_fit, real_train = benchmark.fit_mechanistic, benchmark.train
    calls: list[str] = []

    def failing_mr_f(name, windows, known, modeller, settings):  # noqa: ANN001, ANN202
        found = real_fit(name, windows, known, modeller, settings)
        if name == "MR_F":
            return dataclasses.replace(
                found, selected=None, training_failure=TrainingFailure("forced")
            )
        return found

    def counted(configuration, *arguments, **options):  # noqa: ANN001, ANN202
        calls.append(configuration.label)
        return real_train(configuration, *arguments, **options)

    directory = begin(tmp_path_factory.mktemp("blocked"), SMALL, [1], observed)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(benchmark, "fit_mechanistic", failing_mr_f)
        patch.setattr(benchmark, "train", counted)
        out = DEV.execute(SMALL, "A10", [1], KNOWN, runs, modeller_values(), directory)
    return directory, out, calls


def records(directory: Path) -> dict[str, dict]:
    found = {}
    for path in (directory / "records").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        found[record["key"]] = record
    return found


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_expected_task_has_one_record(executed: tuple[Path, dict]) -> None:
    directory, out = executed
    found = records(directory)
    assert out["status"] == "complete" and out["accounting"]["complete"] is True
    assert out["pending"] == {} and out["errors_of_infrastructure"] == {}
    assert set(found) == benchmark.expected_keys(SMALL, "A10", [0]) and len(found) == 16
    assert not any(k.startswith("A10/r01") for k in found)  # replicate 1 was not asked for
    assert set(out["excitation"]) == {"A10/r00/b02", "A10/r00/b03"}
    assert len(out["excitation"]["A10/r00/b02"]["fitting"].corners) == 1  # F of b = 2
    assert read_json(directory / "summary.json")["status"] == "complete"


def test_every_record_and_model_is_in_the_ledger_with_its_hash(
    executed: tuple[Path, dict],
) -> None:
    directory, _ = executed
    entries = DEV.Ledger(directory).read()
    files = {e["file"]: e for e in entries}
    assert len(files) == len(entries)
    on_disk = [p for sub in ("records", "models") for p in (directory / sub).glob("*.json")]
    assert sorted(p.relative_to(directory).as_posix() for p in on_disk) == sorted(files)
    for path in on_disk:
        assert DEV.file_sha256(path) == files[path.relative_to(directory).as_posix()]["sha256"]
    models = [e for e in entries if e["kind"] == "model"]
    assert len(models) == sum(r.get("model_file") is not None for r in records(directory).values())
    assert set(DEV.read_run(directory)["records"]) == benchmark.expected_keys(SMALL, "A10", [0])


def test_f_and_v_are_kept_apart(executed: tuple[Path, dict], runs: dict) -> None:
    directory, _ = executed
    found = records(directory)
    for budget in SMALL.budgets["A10"]:
        windows, data = runs[SMALL.run_ids("A10")[0]]
        fitting, validation = benchmark.parts(windows, data, budget)
        f = sorted(list(d.key) for d in fitting)
        v = sorted(list(d.key) for d in validation)
        head = f"A10/r00/b{budget:02d}/primary"
        assert sorted(found[f"{head}/MR_F"]["windows"]) == f  # MR_F reads F alone
        assert sorted(found[f"{head}/MR"]["windows"]) == sorted(f + v)  # MR all b windows
        assert sorted(found[f"{head}/BL"]["windows"]) == sorted(f + v)
        for key in (f"{head}/BN/0", f"{head}/HK/0", f"{head}/HK/1"):
            assert sorted(found[key]["fitting_windows"]) == f
            assert sorted(found[key]["validation_windows"]) == v
            assert not set(map(tuple, f)) & set(map(tuple, v))


def test_the_hybrids_start_from_mr_f_and_the_settings_are_explicit(
    executed: tuple[Path, dict],
) -> None:
    directory, _ = executed
    found = records(directory)
    for key, record in found.items():
        if record["kind"] != "training":
            continue
        assert record["settings"]["substeps"] == 4 and record["settings"]["max_steps"] == 4
        assert record["seed"] == training_seed(7, record["replicate"], record["index"])
        if record["family"] == "BN":
            assert record["start"] is None
        else:
            start = found[key.rsplit("/", 2)[0] + "/MR_F"]["model"]
            assert record["start"] == start
    held = found["A10/r00/b03/held/HK/0"]["start"]
    assert held["activation_temperature"] == 8750.0
    assert found["A10/r00/b03/held/MR"]["fixed_activation"] == 8750.0


def test_the_selection_is_made_on_v_and_its_effort_recorded(executed: tuple[Path, dict]) -> None:
    directory, _ = executed
    found = records(directory)
    selections = read_json(directory / "selections.json")
    hk = selections["A10/r00/b03/primary/HK"]
    assert hk["tried"] == ["A10/r00/b03/primary/HK/0", "A10/r00/b03/primary/HK/1"]
    criteria = [found[k]["criterion"] for k in hk["tried"]]
    assert hk["criteria"] == criteria
    assert hk["selected"] == hk["tried"][min(range(2), key=lambda i: (criteria[i], i))]
    assert hk["seconds"] > 0 and hk["validation_windows"]
    assert set(selections) == {
        f"A10/r00/b{b:02d}/{a}/{f}"
        for b, a, f in [
            (2, "primary", "BN"),
            (2, "primary", "HK"),
            (3, "primary", "BN"),
            (3, "primary", "HK"),
            (3, "held", "HK"),
        ]
    }


def test_a_selected_model_is_written_and_reproduces_its_criterion(
    executed: tuple[Path, dict], runs: dict
) -> None:
    directory, _ = executed
    found = records(directory)
    record = found["A10/r00/b03/primary/HK/0"]
    saved = load_selected(directory / record["model_file"])
    assert saved.sha256 == record["model_sha256"]
    windows, data = runs[SMALL.run_ids("A10")[0]]
    _, validation = benchmark.parts(windows, data, 3)
    assert validation_score(saved.model, validation)[0] == record["criterion"]


def test_mr_and_bl_are_rebuilt_from_their_records(executed: tuple[Path, dict], runs: dict) -> None:
    """The fits are deterministic: fitted again, MR and BL give the records written, to the
    last bit, and the models rebuilt from those records predict what the fitted ones do.

    The objective that MR records is not compared with the error of a rollout of its
    parameters: the fit computes its objective by an integration that carries the
    sensitivities, the rollout of the metric integrates the states alone, and the two
    differ by about 1e-10 relative here (D-045)."""
    directory, _ = executed
    found = records(directory)
    windows, data = runs[SMALL.run_ids("A10")[0]]
    fitting, validation = benchmark.parts(windows, data, 3)
    tasks = {t.key: t for t in benchmark.tasks(SMALL, "A10")[0]}
    for kind in ("MR", "BL"):
        key = f"A10/r00/b03/primary/{kind}"
        again = benchmark.run_fit(tasks[key], fitting, validation, KNOWN, modeller_values(), SMALL)
        assert json.loads(json.dumps(again["model"])) == found[key]["model"]
    fit = fit_linear("BL", fitting + validation, KNOWN.nominal_inputs)
    rebuilt = benchmark.linear_model(found["A10/r00/b03/primary/BL"]["model"])
    mr = benchmark.mechanistic_model("MR", KNOWN, found["A10/r00/b03/primary/MR"]["model"])
    direct = MechanisticModel(
        "MR", KNOWN, MechanisticParameters(**found["A10/r00/b03/primary/MR"]["model"])
    )
    for window in fitting + validation:
        np.testing.assert_array_equal(
            predict_window(rebuilt, window), predict_window(fit.model(), window)
        )
        np.testing.assert_array_equal(predict_window(mr, window), predict_window(direct, window))


def test_a_failed_start_blocks_its_hybrids_without_a_retry(
    blocked_run: tuple[Path, dict, list],
) -> None:
    directory, out, calls = blocked_run
    found = records(directory)
    assert out["status"] == "complete" and out["accounting"]["complete"] is True
    blocked = [r for r in found.values() if r["kind"] == "training" and not r["trained"]]
    assert len(blocked) == 6  # the hybrids of both budgets and of E/R held
    assert all("not trained, since its start failed" in r["failure"] for r in blocked)
    assert len(calls) == 2  # BN at each budget, and nothing tried again
    assert len(out["accounting"]["fit_failures"]) == 3  # every MR_F, a scientific outcome
    selections = read_json(directory / "selections.json")
    assert selections["A10/r01/b02/primary/HK"]["selected"] is None
    assert "every configuration failed" in selections["A10/r01/b02/primary/HK"]["failure"]
    assert selections["A10/r01/b02/primary/BN"]["selected"] == "A10/r01/b02/primary/BN/0"


# --------------------------------------------------------------------------- #
# Finding 1: what is written as the run goes, and a run that does not end
# --------------------------------------------------------------------------- #


def interrupted_after(real, count: int) -> tuple:  # noqa: ANN001
    """``real`` for the first ``count`` tasks, then an interruption, as Ctrl+C gives."""
    calls: list = []

    def wrapper(item):  # noqa: ANN001, ANN202
        calls.append(item)
        if len(calls) > count:
            raise KeyboardInterrupt
        return real(item)

    return wrapper, calls


def test_an_execution_needs_its_provenance_written_first(runs: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance.json must be written before any task"):
        DEV.execute(TINY, "A10", [0], KNOWN, runs, modeller_values(), tmp_path)
    assert not (tmp_path / "records").exists()


def test_an_interruption_after_a_fit_keeps_it_and_the_run_is_not_complete(
    runs: dict, observed: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper, calls = interrupted_after(benchmark.fit_task, 1)
    monkeypatch.setattr(benchmark, "fit_task", wrapper)
    directory = begin(tmp_path, TINY, [0], observed)
    with pytest.raises(KeyboardInterrupt):
        DEV.execute(TINY, "A10", [0], KNOWN, runs, modeller_values(), directory)
    found = records(directory)
    assert list(found) == [calls[0].key] == ["A10/r00/b02/primary/MR_F"]
    assert found[calls[0].key]["model"] is not None
    assert [e["key"] for e in DEV.Ledger(directory).read()] == [calls[0].key]
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "interrupted" and summary["accounting"]["complete"] is False
    assert summary["interrupted"]["type"] == "KeyboardInterrupt"
    assert set(summary["pending"]) == benchmark.expected_keys(TINY, "A10", [0]) - set(found)
    assert read_json(directory / "selections.json") == {}
    assert set(DEV.read_run(directory)["records"]) == set(found)  # intact, and readable


def test_an_interruption_after_a_training_keeps_it_and_its_model(
    runs: dict, observed: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper, calls = interrupted_after(benchmark.train_task, 1)
    monkeypatch.setattr(benchmark, "train_task", wrapper)
    directory = begin(tmp_path, TINY, [0], observed)
    with pytest.raises(KeyboardInterrupt):
        DEV.execute(TINY, "A10", [0], KNOWN, runs, modeller_values(), directory)
    found = records(directory)
    head = "A10/r00/b02/primary"
    trained = calls[0][0].key
    assert trained == f"{head}/BN/0"
    assert set(found) == {f"{head}/MR_F", f"{head}/MR", f"{head}/BL", trained}
    saved = load_selected(directory / found[trained]["model_file"])
    assert saved.sha256 == found[trained]["model_sha256"]
    ledger = {(e["key"], e["kind"]) for e in DEV.Ledger(directory).read()}
    assert (trained, "model") in ledger and (trained, "record") in ledger
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "interrupted" and summary["accounting"]["complete"] is False
    assert set(summary["pending"]) == {f"{head}/HK/0", f"{head}/HK/1"}
    # HK has configurations without an outcome, so it is not selected
    assert list(read_json(directory / "selections.json")) == [f"{head}/BN"]
    assert set(DEV.read_run(directory)["records"]) == set(found)


def test_an_error_of_infrastructure_leaves_its_task_pending_not_failed(
    runs: dict, observed: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = benchmark.fit_task

    def lost(task):  # noqa: ANN001, ANN202
        if task.kind == "MR_F":
            raise RuntimeError("the worker was lost")
        return real(task)

    monkeypatch.setattr(benchmark, "fit_task", lost)
    directory = begin(tmp_path, TINY, [0], observed)
    out = DEV.execute(TINY, "A10", [0], KNOWN, runs, modeller_values(), directory)
    head = "A10/r00/b02/primary"
    assert out["status"] == "incomplete" and out["accounting"]["complete"] is False
    assert out["errors_of_infrastructure"][f"{head}/MR_F"]["type"] == "RuntimeError"
    assert set(records(directory)) == {f"{head}/MR", f"{head}/BL", f"{head}/BN/0"}
    # the hybrids wait for their start; they are neither blocked nor counted as failed
    assert out["pending"][f"{head}/HK/0"] == f"its start {head}/MR_F has no record"
    assert out["pending"][f"{head}/MR_F"] == "error of infrastructure"
    assert out["accounting"]["training_failures"] == [] == out["accounting"]["fit_failures"]


def test_errors_in_worker_processes_are_recorded_and_the_others_go_on(
    runs: dict, observed: dict, tmp_path: Path
) -> None:
    directory = begin(tmp_path, TINY, [0], observed)
    # without the modeller's values every mechanistic fit raises, in its worker
    out = DEV.execute(TINY, "A10", [0], KNOWN, runs, None, directory, workers=2)
    head = "A10/r00/b02/primary"
    assert out["status"] == "incomplete"
    assert set(out["errors_of_infrastructure"]) == {f"{head}/MR_F", f"{head}/MR"}
    assert set(records(directory)) == {f"{head}/BL", f"{head}/BN/0"}
    assert set(out["pending"]) == {f"{head}/MR_F", f"{head}/MR", f"{head}/HK/0", f"{head}/HK/1"}


def stable(found: dict) -> dict:
    """A record without the times it took, the only thing allowed to differ."""
    return {k: v for k, v in found.items() if k not in TIMES}


def test_two_processes_give_what_one_gives(
    executed: tuple[Path, dict], runs: dict, observed: dict, tmp_path: Path
) -> None:
    """The same tasks, records, selections, models and predictions with two processes as
    with one. Equality is exact: each task is a deterministic computation on the CPU, with
    the same program, inputs and seeds in whichever process runs it, and the order in which
    tasks end decides nothing."""
    sequential, _ = executed
    directory = begin(tmp_path, SMALL, [0], observed)
    out = DEV.execute(SMALL, "A10", [0], KNOWN, runs, modeller_values(), directory, workers=2)
    assert out["status"] == "complete"
    one, two = records(sequential), records(directory)
    assert set(one) == set(two)
    for key in one:
        assert stable(one[key]) == stable(two[key]), key
    selections = [
        {k: stable(v) for k, v in read_json(d / "selections.json").items()}
        for d in (sequential, directory)
    ]
    assert selections[0] == selections[1]
    windows, data = runs[SMALL.run_ids("A10")[0]]
    for key, record in one.items():
        if record.get("model_file") is None:
            continue
        a = load_selected(sequential / record["model_file"])
        b = load_selected(directory / two[key]["model_file"])
        assert a.sha256 == b.sha256 == record["model_sha256"]
        _, validation = benchmark.parts(windows, data, record["budget"])
        for window in validation:
            np.testing.assert_array_equal(
                predict_window(a.model, window), predict_window(b.model, window)
            )


# --------------------------------------------------------------------------- #
# Finding 2: the identity of the observable data
# --------------------------------------------------------------------------- #


def plant_record(known: KnownPlant) -> PlantRecord:
    values = {
        "reactor_volume": known.volume,
        "density": known.density,
        "heat_capacity": known.heat_capacity,
        "reaction_enthalpy": known.reaction_enthalpy,
        **dict(zip(NOMINAL_INPUT_UNITS, known.nominal_inputs, strict=True)),
    }
    units = {**PROPERTY_UNITS, **NOMINAL_INPUT_UNITS}
    parameters = tuple(KnownParameter(n, v, units[n]) for n, v in values.items())
    return PlantRecord("target", "target", "cstr", parameters)


def write_export(
    root: Path, observed: dict, known: KnownPlant, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """The runs of the tests as a data set under ``root``, ingested and exported."""
    monkeypatch.setenv("PT_DATA_DIR", str(root))
    dataset = SMALL.sets["A10"].dataset_id
    with DatasetWriter(dataset, [plant_record(known)], ATTEMPT) as writer:
        for observations in observed.values():
            writer.add_run(observations, RunRecord(dataset, "p3", "synthetic run for a test"))
        writer.publish()
    connection = database.connect()
    try:
        database.ingest_dataset(connection, open_dataset(dataset))
        assert export_dataset(connection, dataset, ATTEMPT).status == "created"
    finally:
        connection.close()
    return root / "available" / "exports"


def test_equal_observations_with_other_known_parameters_are_other_data(
    observed: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = dataclasses.replace(KNOWN, volume=0.11)
    one = write_export(tmp_path / "one", observed, KNOWN, monkeypatch)
    two = write_export(tmp_path / "two", observed, other, monkeypatch)
    known_one, _, first = DEV.load_condition(one, SMALL, "A10")
    known_two, _, second = DEV.load_condition(two, SMALL, "A10")
    assert (known_one, known_two) == (KNOWN, other)
    assert first["run_content_sha256"] == second["run_content_sha256"]  # the same readings
    assert first["known_plant"] != second["known_plant"]
    assert first["export_files_sha256"] != second["export_files_sha256"]
    assert first["sha256"] != second["sha256"]
    assert first["generation"] == ATTEMPT
    a = DEV.provenance_of(Path("r0"), STATE, SMALL, "A10", [0], 1, first, tmp_path)
    b = DEV.provenance_of(Path("r1"), STATE, SMALL, "A10", [1], 1, second, tmp_path)
    a, b = json.loads(json.dumps(a)), json.loads(json.dumps(b))
    assert any("identity of their data" in p for p in benchmark.compatible([a, b], SMALL))
    assert benchmark.compatible([a, {**b, "data": a["data"]}], SMALL) == []
    # the identity alone, without an export: the known parameters are in it
    assert identity_of(observed)["sha256"] != identity_of(observed, other)["sha256"]


def test_a_run_writes_its_provenance_before_its_first_task_and_exits_with_three_when_stopped(
    observed: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exports = write_export(tmp_path / "pt", observed, KNOWN, monkeypatch)
    monkeypatch.setattr(benchmark, "REGISTERED", TINY)
    monkeypatch.setattr(DEV, "git_state", lambda: dict(STATE))
    real = benchmark.fit_task
    seen: list = []

    def first_then_stop(task):  # noqa: ANN001, ANN202
        seen.append(list(tmp_path.glob("pt/experiments/m1_e03_development/*/provenance.json")))
        if len(seen) > 1:
            raise KeyboardInterrupt
        return real(task)

    monkeypatch.setattr(benchmark, "fit_task", first_then_stop)
    before = DEV.fingerprints(exports / "synthetic-a10")
    assert DEV.main(["--condition", "A10", "--replicates", "0"]) == 3
    (path,) = seen[0]  # the provenance was there when the first fit began
    directory = path.parent
    provenance = read_json(path)
    _, _, identity = DEV.load_condition(exports, TINY, "A10")
    assert provenance["data"] == json.loads(json.dumps(identity))
    assert benchmark.check_provenance(provenance, TINY) == []
    assert read_json(directory / "summary.json")["status"] == "interrupted"
    assert list(records(directory)) == ["A10/r00/b02/primary/MR_F"]
    assert DEV.fingerprints(exports / "synthetic-a10") == before


def test_a_run_that_ends_exits_with_zero_and_can_be_combined(
    observed: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_export(tmp_path / "pt", observed, KNOWN, monkeypatch)
    monkeypatch.setattr(benchmark, "REGISTERED", TINY)
    monkeypatch.setattr(DEV, "git_state", lambda: dict(STATE))
    assert DEV.main(["--condition", "A10", "--replicates", "0"]) == 0
    (directory,) = (tmp_path / "pt" / "experiments" / "m1_e03_development").iterdir()
    assert read_json(directory / "summary.json")["status"] == "complete"
    assert read_json(directory / "exports_after.json") == {"unchanged": True}
    assert DEV.main(["--combine", str(directory)]) == 0
    (combined,) = (tmp_path / "pt" / "experiments" / "m1_e03_combined").iterdir()
    found = read_json(combined / "combined.json")
    assert found["coverage"]["A10"] == {
        "expected": 12,
        "recorded": 6,
        "missing": 6,
        "complete": False,
    }
    assert found["complete"] is False  # replicate 1 was not run: valid, and partial
    assert DEV.main(["--combine", str(directory), str(directory)]) == 2


# --------------------------------------------------------------------------- #
# Finding 3: provenance, compatibility and the combination of runs made apart
# --------------------------------------------------------------------------- #

BOTH = dataclasses.replace(
    SMALL,
    sets={**SMALL.sets, "A5": benchmark.DevelopmentSet("synthetic-a5", (43, 44), "a5", 3, 60)},
    budgets={**SMALL.budgets, "A5": (3,)},
    seed_bases={"A10": 7, "A5": 8},
)


def provenance(spec: benchmark.Specification = SMALL, condition: str = "A10", /, **change) -> dict:  # noqa: ANN003
    identity = benchmark.data_identity(
        spec.sets[condition].dataset_id, KNOWN, {"run": "0" * 64}, {}, None
    )
    found = DEV.provenance_of(
        Path(f"run-{condition}"), STATE, spec, condition, [0], 1, identity, Path("x")
    )
    return json.loads(json.dumps({**found, **change}))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"commit": "abc"}, "not a full object name"),
        ({"commit": None}, "not a full object name"),
        ({"code_identified": False}, "not made from a clean commit"),
        ({"specification_sha256": "0" * 64}, "another specification"),
        ({"condition": "A7"}, "unknown condition"),
        ({"replicates": [0, 0]}, "not distinct replicates"),
        ({"replicates": [2]}, "not distinct replicates"),
        ({"replicates": [True]}, "not distinct replicates"),
        ({"replicates": []}, "not distinct replicates"),
    ],
)
def test_an_invalid_provenance_is_refused(change: dict, message: str) -> None:
    assert benchmark.check_provenance(provenance(), SMALL) == []
    problems = benchmark.check_provenance(provenance(**change), SMALL)
    assert any(message in p for p in problems), problems
    assert benchmark.compatible([provenance(), provenance(**change)], SMALL)


@pytest.mark.parametrize("field", benchmark.REQUIRED)
def test_an_incomplete_provenance_is_refused(field: str) -> None:
    found = provenance()
    del found[field]
    assert any("lacks" in p for p in benchmark.check_provenance(found, SMALL))
    assert benchmark.compatible([provenance(), found], SMALL)


def test_the_identity_of_the_data_is_checked_against_its_content() -> None:
    found = provenance()
    found["data"]["known_plant"]["volume"] = 0.2  # the hash left as it was
    assert any("does not match its content" in p for p in benchmark.check_provenance(found, SMALL))
    found["data"]["dataset_id"] = "synthetic-a5"
    assert any("not the development set" in p for p in benchmark.check_provenance(found, SMALL))
    missing = provenance()
    del missing["environment"]["learning"]["x64"]
    assert any("environment lacks" in p for p in benchmark.check_provenance(missing, SMALL))


def test_runs_made_apart_must_share_the_numerical_environment() -> None:
    one = provenance()
    numpy = provenance()
    numpy["environment"]["packages"]["numpy"] = "0.0.1"
    x64 = provenance()
    x64["environment"]["learning"]["x64"] = False
    for other in (numpy, x64):
        assert any("numerical environment" in p for p in benchmark.compatible([one, other], SMALL))
    platform = provenance()
    platform["environment"]["platform"] = "another-platform-string"
    assert benchmark.compatible([one, platform], SMALL) == []  # reported, not refused
    assert any(
        "commit" in p for p in benchmark.compatible([one, provenance(commit="b" * 40)], SMALL)
    )


def test_a10_and_a5_are_other_sets_and_their_identities_are_not_compared() -> None:
    a10, a5 = provenance(BOTH, "A10"), provenance(BOTH, "A5")
    assert a10["data"]["sha256"] != a5["data"]["sha256"]
    assert benchmark.compatible([a10, a5], BOTH) == []
    other = provenance(BOTH, "A10")
    other["data"] = benchmark.data_identity("synthetic-a10", KNOWN, {"run": "1" * 64}, {}, None)
    assert any("A10 differ" in p for p in benchmark.compatible([a10, a5, other], BOTH))


def copy_of(directory: Path, target: Path, **change) -> Path:  # noqa: ANN003
    shutil.copytree(directory, target)
    found = read_json(target / "provenance.json")
    found.update(change)
    (target / "provenance.json").write_text(json.dumps(found), encoding="utf-8")
    return target


def test_disjoint_replicates_combine_into_a_new_directory(
    executed: tuple[Path, dict],
    blocked_run: tuple[Path, dict, list],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "pt"))
    r0, r1 = executed[0], blocked_run[0]
    before = [DEV.fingerprints(r0), DEV.fingerprints(r1)]
    target, found = DEV.combine_runs([r0, r1], SMALL)
    assert [DEV.fingerprints(r0), DEV.fingerprints(r1)] == before  # the runs are not changed
    assert target.parent == tmp_path / "pt" / "experiments" / "m1_e03_combined"
    assert read_json(target / "combined.json")["complete"] is True
    assert found["coverage"]["A10"] == {
        "expected": 32,
        "recorded": 32,
        "missing": 0,
        "complete": True,
    }
    # failures are outcomes, and are counted
    assert len(found["failures"]["fits"]) == 3 and len(found["failures"]["trainings"]) == 6
    assert len(found["selections"]) == 10
    assert [r["directory"] for r in found["runs"]] == [str(r0), str(r1)]
    assert all(r["provenance_sha256"] and r["ledger_sha256"] for r in found["runs"])
    partial = benchmark.combine([DEV.read_run(r0)], SMALL)
    assert partial["coverage"]["A10"]["missing"] == 16 and partial["complete"] is False


def test_a_task_recorded_twice_is_refused_whatever_its_outcome(
    executed: tuple[Path, dict], tmp_path: Path
) -> None:
    r0 = executed[0]
    with pytest.raises(ValueError, match="given more than once"):
        benchmark.combine([DEV.read_run(r0), DEV.read_run(r0)], SMALL)
    again = copy_of(r0, tmp_path / "again", run_id="another-run")
    with pytest.raises(ValueError, match="recorded twice"):
        benchmark.combine([DEV.read_run(r0), DEV.read_run(again)], SMALL)


def test_a_record_of_a_task_the_run_was_not_given_is_refused(
    executed: tuple[Path, dict], tmp_path: Path
) -> None:
    foreign = copy_of(executed[0], tmp_path / "foreign", replicates=[1])
    with pytest.raises(ValueError, match="is not a task of that run"):
        benchmark.combine([DEV.read_run(foreign)], SMALL)


def test_runs_with_other_data_or_environment_are_not_combined(
    executed: tuple[Path, dict], blocked_run: tuple[Path, dict, list], tmp_path: Path
) -> None:
    r0, r1 = executed[0], blocked_run[0]
    original = read_json(r1 / "provenance.json")
    other = benchmark.data_identity(
        "synthetic-a10",
        dataclasses.replace(KNOWN, volume=0.11),
        original["data"]["run_content_sha256"],
        {},
        None,
    )
    environment = json.loads(json.dumps(original["environment"]))
    environment["learning"]["jax"] = "0.0.1"
    for name, change, message in (
        ("known", {"data": other}, "identity of their data"),
        ("environment", {"environment": environment}, "numerical environment"),
        ("incomplete", {"commit": None}, "not a full object name"),
    ):
        copy = copy_of(r1, tmp_path / name, **change)
        with pytest.raises(ValueError, match=message):
            benchmark.combine([DEV.read_run(r0), DEV.read_run(copy)], SMALL)


@pytest.mark.parametrize("damage", ["record", "model", "unlisted", "ledger"])
def test_a_record_or_model_that_does_not_match_its_ledger_is_refused(
    executed: tuple[Path, dict], tmp_path: Path, damage: str
) -> None:
    copy = copy_of(executed[0], tmp_path / "copy")
    if damage == "record":
        path = copy / "records" / DEV.file_name("A10/r00/b02/primary/BL")
        path.write_bytes(path.read_bytes().replace(b'"BL"', b'"BN"', 1))
        message = "does not match its ledger"
    elif damage == "model":
        path = next((copy / "models").glob("*.json"))
        path.write_bytes(path.read_bytes() + b" ")
        message = "does not match its ledger"
    elif damage == "unlisted":
        (copy / "records" / "extra.json").write_text("{}", encoding="utf-8")
        message = "files and ledger disagree"
    else:
        lines = (copy / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        (copy / "ledger.jsonl").write_text("\n".join(lines + lines[:1]) + "\n", encoding="utf-8")
        message = "enters a file or a key twice"
    with pytest.raises(ValueError, match=message):
        DEV.read_run(copy)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_a_record_is_never_overwritten(tmp_path: Path) -> None:
    DEV.write_once(tmp_path / "records" / "a.json", {"x": 1})
    with pytest.raises(FileExistsError):
        DEV.write_once(tmp_path / "records" / "a.json", {"x": 2})


@pytest.mark.parametrize(
    ("dataset_id", "run_ids", "plants", "message"),
    [
        ("m1-e03-test-a10", None, None, "not the development set"),
        ("m1-e03-train-a10", "missing", None, "runs missing"),
        ("m1-e03-train-a10", "extra", None, "runs not registered"),
        ("m1-e03-train-a10", None, "source", "plants other than the target"),
    ],
)
def test_only_the_registered_development_set_is_read(
    dataset_id: str, run_ids: object, plants: object, message: str
) -> None:
    expected = list(R.run_ids("A10"))
    ids = {"missing": expected[:-1], "extra": expected + ["target.p3.e1.x40.l60.a10.n0"]}.get(
        run_ids, expected
    )
    owners = {r: (plants or "target") for r in ids}
    with pytest.raises(DEV.Refused, match=message):
        DEV.check_export(dataset_id, ids, owners, expected, R.sets["A10"].dataset_id)


def test_an_official_run_is_refused_without_a_clean_commit_or_outside_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", "data")
    assert DEV.main(["--condition", "A10"]) == 2  # inside the repository
    assert DEV.main(["--combine", "anything"]) == 2
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    assert DEV.main(["--combine", str(tmp_path)]) == 2  # not a run: no provenance
    monkeypatch.setattr(DEV, "git_state", lambda: {"code_identified": False, "reason": "dirty"})
    assert DEV.main(["--condition", "A10"]) == 2
    monkeypatch.setattr(DEV, "git_state", lambda: {"code_identified": True, "commit": "abc"})
    assert DEV.main(["--condition", "A10"]) == 2  # no export of the development set
    assert DEV.main(["--condition", "A10", "--replicates", "10"]) == 2
    assert DEV.main(["--condition", "A10", "--replicates", "0", "0"]) == 2
    assert DEV.main([]) == 2  # no condition
    assert DEV.main(["--condition", "A10", "--workers", "0"]) == 2
    assert not (tmp_path / "data" / "experiments").exists()
