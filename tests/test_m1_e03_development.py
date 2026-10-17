"""The development runs of M1-E03 (I6): the registered specification, and its executor on
synthetic data.

The specification is checked against the registration and its definitions. The executor
is run on a small specification of the tests, on runs simulated with the modeller's
equations on the truth side of the tests, with excitation seeds of the tests (41 and 42).
No benchmark data are read or generated: a test that reached the generator would fail.
"""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from m1_support import NOMINAL, modeller_run, random_corners
from process_transfer.config import load_dataset_definition
from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.plant import KnownPlant
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


@pytest.fixture(scope="module")
def runs() -> dict:
    found = {}
    for seed, run_id in zip((41, 42), SMALL.run_ids("A10"), strict=True):
        observations = modeller_run(
            MechanisticModel("m", KNOWN, TRUE).rhs, random_corners(3, seed), seed, run=run_id
        )
        windows = find_windows(observations, NOMINAL, P3_LAYOUT)
        found[run_id] = (windows, window_data(observations, windows.windows))
    return found


@pytest.fixture(scope="module")
def executed(runs: dict, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    directory = tmp_path_factory.mktemp("development")
    out = DEV.execute(SMALL, "A10", [0], KNOWN, runs, modeller_values(), directory)
    return directory, out


def records(directory: Path) -> dict[str, dict]:
    found = {}
    for path in (directory / "records").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        found[record["key"]] = record
    return found


def test_every_expected_task_has_one_record(executed: tuple[Path, dict]) -> None:
    directory, out = executed
    found = records(directory)
    assert out["accounting"]["complete"] is True
    assert set(found) == benchmark.expected_keys(SMALL, "A10", [0]) and len(found) == 16
    assert not any(k.startswith("A10/r01") for k in found)  # replicate 1 was not asked for
    assert set(out["excitation"]) == {"A10/r00/b02", "A10/r00/b03"}
    assert len(out["excitation"]["A10/r00/b02"]["fitting"].corners) == 1  # F of b = 2


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
    selections = json.loads((directory / "selections.json").read_text(encoding="utf-8"))
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
    last bit, and the models rebuilt from those records predict what the fitted ones do."""
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
    runs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    monkeypatch.setattr(benchmark, "fit_mechanistic", failing_mr_f)
    monkeypatch.setattr(benchmark, "train", counted)
    out = DEV.execute(SMALL, "A10", [1], KNOWN, runs, modeller_values(), tmp_path)
    found = records(tmp_path)
    assert out["accounting"]["complete"] is True
    blocked = [r for r in found.values() if r["kind"] == "training" and not r["trained"]]
    assert len(blocked) == 6  # the hybrids of both budgets and of E/R held
    assert all("not trained, since its start failed" in r["failure"] for r in blocked)
    assert len(calls) == 2  # BN at each budget, and nothing tried again
    selections = json.loads((tmp_path / "selections.json").read_text(encoding="utf-8"))
    assert selections["A10/r01/b02/primary/HK"]["selected"] is None
    assert "every configuration failed" in selections["A10/r01/b02/primary/HK"]["failure"]
    assert selections["A10/r01/b02/primary/BN"]["selected"] == "A10/r01/b02/primary/BN/0"


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


def test_runs_made_apart_combine_only_when_compatible() -> None:
    one = {
        "specification_sha256": R.digest(),
        "commit": "abc",
        "code_identified": True,
        "data": {"runs": {"r": "h"}},
    }
    assert benchmark.compatible([one, dict(one)]) == []
    for change, problem in (
        ({"specification_sha256": "other"}, "specification_sha256"),
        ({"commit": "abd"}, "commit"),
        ({"data": {"runs": {"r": "g"}}}, "data"),
        ({"code_identified": False}, "clean commit"),
    ):
        problems = benchmark.compatible([one, {**one, **change}])
        assert any(problem in p for p in problems)


def test_an_official_run_is_refused_without_a_clean_commit_or_outside_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", "data")
    assert DEV.main(["--condition", "A10"]) == 2  # inside the repository
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(DEV, "git_state", lambda: {"code_identified": False, "reason": "dirty"})
    assert DEV.main(["--condition", "A10"]) == 2
    monkeypatch.setattr(DEV, "git_state", lambda: {"code_identified": True, "commit": "abc"})
    assert DEV.main(["--condition", "A10"]) == 2  # no export of the development set
    assert DEV.main(["--condition", "A10", "--replicates", "10"]) == 2
    assert not (tmp_path / "data" / "experiments").exists()
