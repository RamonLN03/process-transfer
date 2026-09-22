"""The whole data path on a small data set: generation, Parquet, DuckDB, SQL checks,
export and the scan for hidden information. This is the storage and SQL round trip that
runs in CI.

The seeds are fixed, arbitrary and those of no experiment: excitation seeds 41 and 42, a
master seed of 987654321, one excursion of twelve minutes per run.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from process_transfer.config import load_dataset_definition
from process_transfer.data import database
from process_transfer.data.export import ExportConflictError, ExportError, export_dataset
from process_transfer.data.parquet_store import open_dataset
from process_transfer.generation import pipeline
from process_transfer.generation.__main__ import main
from process_transfer.generation.leak_scan import HiddenValues, scan_available
from process_transfer.generation.plants import StartingPointError, load_virtual_plant

MASTER_SEED = 987654321
RUNS = [f"{plant}.p3.e{seed}.x1.n0" for seed in (41, 42) for plant in ("source", "target")]


def write_definition(directory: Path, configs_dir: Path, **changes: object) -> Path:
    definition = {
        "dataset_id": "pipeline-test",
        "description": "A small data set for the tests of the data path.",
        "plants": [str(configs_dir / "source_cstr.yaml"), str(configs_dir / "target_cstr.yaml")],
        "sensors": str(configs_dir / "sensors_cstr.yaml"),
        "protocol": "p3",
        "n_excursions": 1,
        "excitation_seeds": [41, 42],
        "noise_realisation": 0,
        "sensor_master_seed": MASTER_SEED,
        "simulation_period": {"value": 0.1, "unit": "s"},
        **changes,
    }
    path = directory / "definition.json"  # JSON is YAML; this avoids quoting Windows paths
    path.write_text(json.dumps(definition), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory, configs_dir: Path):
    """One run of the pipeline, shared by the tests that only read its results."""
    root = tmp_path_factory.mktemp("pipeline")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PT_DATA_DIR", str(root / "data"))
    definition = write_definition(root, configs_dir)
    report = pipeline.run_pipeline(definition, figures=False)
    yield root / "data", definition, report
    monkeypatch.undo()


def test_the_whole_path_passes_every_mandatory_check(generated) -> None:  # noqa: ANN001
    data, _, report = generated
    assert report["ok"] is True
    assert len(report["checks"]) == pipeline.EXPECTED_CHECKS == 10
    assert all(report["checks"].values()), report["checks"]
    assert report["dataset"]["status"] == "created" and report["export"]["status"] == "created"
    assert report["ingestion"] == dict.fromkeys(sorted(RUNS), "ingested")
    assert report["quality"] == dict.fromkeys(report["quality"], 0) and len(report["quality"]) == 7
    assert report["rows"] == {
        "plants": 2,
        "process_parameters": 16,
        "sensors": 12,
        "operating_runs": 4,
        "measurements": 4 * 121 * 6,
    }
    assert sorted(path.name for path in (data / "available").iterdir()) == [
        "databases",
        "datasets",
        "exports",
    ]
    assert [p.name for p in (data / "private" / "datasets" / "pipeline-test").iterdir()]


def test_the_export_is_what_a_model_needs_and_no_more(generated) -> None:  # noqa: ANN001
    data, _, _ = generated
    directory = data / "available" / "exports" / "pipeline-test"
    manifest = json.loads((directory / "export.json").read_text(encoding="utf-8"))
    assert sorted(run["run_id"] for run in manifest["runs"]) == sorted(RUNS)
    assert "zero-order hold, right-continuous" in manifest["time_convention"]
    target = manifest["plants"]["target"]
    assert [(c["column"], c["channel_kind"], c["unit"]) for c in target["channels"]] == [
        ("C_A", "measured", "mol/m^3"),
        ("T", "measured", "K"),
        ("q", "input", "m^3/s"),
        ("C_Af", "input", "mol/m^3"),
        ("T_f", "input", "K"),
        ("T_c", "input", "K"),
    ]
    assert [c["noise_std"] for c in target["channels"]] == [5.0, 0.5, None, None, None, None]
    assert {p["parameter"] for p in target["known_parameters"]} == {
        "reactor_volume",
        "density",
        "heat_capacity",
        "reaction_enthalpy",
        "nominal_feed_flow",
        "nominal_feed_concentration",
        "nominal_feed_temperature",
        "nominal_coolant_temperature",
    }

    table = pq.read_table(directory / "target.p3.e41.x1.n0.parquet")
    assert table.schema.names == [
        "plant_id",
        "run_id",
        "sample_index",
        "time_s",
        "C_A",
        "T",
        "q",
        "C_Af",
        "T_f",
        "T_c",
    ]
    assert table.num_rows == 121 and table["time_s"].to_pylist()[:3] == [0.0, 6.0, 12.0]
    assert table.schema.field("C_A").metadata == {b"unit": b"mol/m^3", b"channel_kind": b"measured"}
    assert all(table.schema.field(name).type == pa.float64() for name in ("time_s", "C_A", "T_c"))
    # the inputs step at 120 s, and the row of the step carries the new inputs
    coolant = table["T_c"].to_numpy()
    assert coolant[19] != coolant[20] and coolant[20] == 337.5 and np.all(coolant[20:] == 337.5)


def test_a_second_generation_reproduces_the_content(generated, configs_dir: Path) -> None:  # noqa: ANN001
    data, definition, first = generated
    before = {
        path: path.stat().st_mtime_ns
        for path in (data / "available" / "datasets").rglob("*")
        if path.is_file()
    }
    second = pipeline.run_pipeline(definition, figures=False)
    assert second["ok"] is True and second["runs"] == first["runs"]  # the same content hashes
    assert second["dataset"]["status"] == "already_present"
    assert second["export"]["status"] == "already_present"
    assert set(second["ingestion"].values()) == {"already_present"}
    assert second["rows"] == first["rows"]
    assert before == {path: path.stat().st_mtime_ns for path in before}  # not a file was touched

    attempts = sorted((data / "private" / "datasets" / "pipeline-test").iterdir())
    assert len(attempts) >= 2  # one private record per attempt, none reused
    records = [json.loads((a / "provenance.json").read_text(encoding="utf-8")) for a in attempts]
    assert {record["sensor_master_seed"] for record in records} == {MASTER_SEED}
    assert records[0]["runs"] == records[1]["runs"]  # seeds, streams, truth checks, hashes
    assert records[0]["report"]["dataset"]["status"] == "created"
    assert records[1]["publication"] == "already_present"


def test_reading_and_exporting_work_without_the_private_branch(
    generated,  # noqa: ANN001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the data set is handed over: no private branch, no database, no export. It is
    opened, ingested into a new database and exported again, and gives the same content."""
    data, _, first = generated
    handed_over = tmp_path / "handed-over"
    shutil.copytree(
        data / "available" / "datasets" / "pipeline-test",
        handed_over / "available" / "datasets" / "pipeline-test",
    )
    monkeypatch.setenv("PT_DATA_DIR", str(handed_over))
    dataset = open_dataset("pipeline-test")
    connection = database.connect(database.database_path("rebuilt"))
    assert set(database.ingest_dataset(connection, dataset).values()) == {"ingested"}
    result = export_dataset(connection, "pipeline-test", {"by": "a reader"})
    connection.close()

    assert result.status == "created"
    assert {run["run_id"]: run["content_sha256"] for run in result.manifest["runs"]} == first[
        "runs"
    ]
    original = data / "available" / "exports" / "pipeline-test" / "source.p3.e42.x1.n0.parquet"
    rebuilt = result.directory / "source.p3.e42.x1.n0.parquet"
    assert pq.read_table(rebuilt).equals(pq.read_table(original))  # by content, not by bytes
    assert not (handed_over / "private").exists()


def test_nothing_hidden_is_in_the_available_branch_and_the_private_record_has_it(
    generated,  # noqa: ANN001
) -> None:
    data, _, report = generated
    assert report["hidden_information_scan"]["findings"] == []
    assert report["hidden_information_scan"]["files"] == 9 + 1 + 5  # data set, database, export

    def keys(node: object) -> list[str]:
        if isinstance(node, dict):
            return [*node, *(key for value in node.values() for key in keys(value))]
        return [key for value in node for key in keys(value)] if isinstance(node, list) else []

    documents = sorted((data / "available").rglob("*.json"))
    assert [path.name for path in documents] == ["manifest.json", "export.json"]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        assert str(MASTER_SEED) not in text and "true_physics" not in text
        assert not [key for key in keys(json.loads(text)) if "seed" in key.lower()]

    (attempt,) = sorted((data / "private" / "datasets" / "pipeline-test").iterdir())[:1]
    record = json.loads((attempt / "provenance.json").read_text(encoding="utf-8"))
    assert record["sensor_master_seed"] == MASTER_SEED
    assert len(record["runs"]["target.p3.e41.x1.n0"]["noise_stream"]) == 4
    assert record["runs"]["target.p3.e41.x1.n0"]["truth_accepted"] is True
    assert "true_physics" in (attempt / "configs" / "target_cstr.yaml").read_text(encoding="utf-8")
    assert {"git", "environment", "definition", "integration", "manifest"} <= set(record)
    assert {"pyarrow", "duckdb", "numpy", "scipy"} <= set(record["environment"]["packages"])


# --------------------------------------------------------------------------- #
# The scan finds what is planted
# --------------------------------------------------------------------------- #


def test_the_scan_finds_a_leak_of_each_kind(tmp_path: Path) -> None:
    hidden = HiddenValues(
        parameters={"k0 of target": 2.4e9, "alpha of target": 0.002},
        states=np.array([189.67276279476130, 355.16866636550080]),
        integers={"the master seed of the noise": MASTER_SEED},
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    pq.write_table(pa.table({"time_s": [0.0, 6.0], "T": [355.4, 354.9]}), clean / "run.parquet")
    (clean / "export.json").write_text(
        '{"dataset_id": "x", "rows": 2, "sigma": 0.5}', encoding="utf-8"
    )
    assert scan_available([clean], hidden, ())["findings"] == []

    planted = {
        "a name": ("a.parquet", pa.table({"time_s": [0.0], "T_exact": [355.4]})),
        "an exact state": ("b.parquet", pa.table({"time_s": [0.0], "T": [355.16866636550080]})),
        "a hidden parameter": ("c.parquet", pa.table({"value": [0.002 * (1 + 1e-14)]})),
        "the seed as a number": ("d.parquet", pa.table({"n": pa.array([MASTER_SEED], pa.int64())})),
    }
    for label, (name, table) in planted.items():
        directory = tmp_path / name.split(".")[0]
        directory.mkdir()
        pq.write_table(table, directory / name)
        assert scan_available([directory], hidden, ())["findings"], label

    for label, document in {
        "a key": {"sensor_seed": 1},
        "a nested key": {"runs": [{"noise_stream": [1, 2, 3, 4]}]},
        "the seed as a value": {"entropy": MASTER_SEED},
        "a hidden value": {"rate": 2.4e9},
        "a mention": {"note": "copied from true_physics"},
    }.items():
        directory = tmp_path / label.replace(" ", "_")
        directory.mkdir()
        (directory / "x.json").write_text(json.dumps(document), encoding="utf-8")
        assert scan_available([directory], hidden, ())["findings"], label

    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "source_cstr.yaml").write_text("true_physics: {}", encoding="utf-8")
    assert (
        "of a kind that the available branch does not hold"
        in scan_available([stray], hidden, ())["findings"][0]
    )
    with pytest.raises(ValueError, match="nothing to scan"):
        scan_available([tmp_path / "nowhere"], hidden, ())
    with pytest.raises(ValueError, match="cannot be searched for"):
        scan_available([clean], HiddenValues({"alpha": 0.0}, np.array([])), ())


def test_a_small_seed_or_a_value_that_equals_an_instant_is_not_a_leak(tmp_path: Path) -> None:
    """Regression. The scan compared every number with the hidden ones, so a master seed
    of 0 was "found" in every tick, position and flag, and a data set generated with it
    could never pass. Fields whose values the contract fixes are not searched; the same
    values in a free field still are."""
    clean = tmp_path / "clean"
    clean.mkdir()
    pq.write_table(
        pa.table(
            {
                "sample_index": pa.array([0, 1, 2], pa.int64()),
                "channel_index": pa.array([0, 1, 1], pa.int32()),
                "quality_flag": pa.array([0, 0, 0], pa.int16()),
                "time_s": [0.0, 6.0, 12.0],
                "T": [355.4, 354.9, 355.1],
            }
        ),
        clean / "run.parquet",
    )
    (clean / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": 1,
                "runs": [{"rows": 3, "n_samples": 3, "start_time_s": 0.0, "end_time_s": 12.0}],
                "tables": {"plants": {"rows": 2}},
            }
        ),
        encoding="utf-8",
    )
    for seed in (0, 1, 2, 3):
        hidden = HiddenValues({"k of x": 6.0}, np.array([12.0]), {"the master seed": seed})
        assert scan_available([clean], hidden, ())["findings"] == [], seed

    planted = tmp_path / "planted"
    planted.mkdir()
    pq.write_table(
        pa.table({"n": pa.array([0], pa.int64()), "value": [6.0]}), planted / "run.parquet"
    )
    (planted / "x.json").write_text(
        json.dumps({"entropy": 0, "words": [[0, 5, 6, 7]], "rate": 6.0}), encoding="utf-8"
    )
    hidden = HiddenValues({"k of x": 6.0}, np.array([]), {"the master seed": 0})
    assert sorted(scan_available([planted], hidden, ())["findings"]) == [
        "run.parquet.n: holds the master seed, 0",
        "run.parquet.value: holds the hidden value of k of x, 6.0",
        "x.json: holds the hidden value of k of x, 6.0",
        "x.json: holds the master seed, 0",
    ]


def test_a_master_seed_of_zero_passes_the_whole_path(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression, end to end: the definition is valid and the data set is clean, so the
    scan must find nothing and the command must succeed."""
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    definition = write_definition(
        tmp_path,
        configs_dir,
        dataset_id="seed-zero-test",
        excitation_seeds=[46],
        sensor_master_seed=0,
    )
    report = pipeline.run_pipeline(definition, figures=False)
    assert report["hidden_information_scan"]["findings"] == []
    assert report["ok"] is True and all(report["checks"].values()), report["checks"]


# --------------------------------------------------------------------------- #
# Failures are reported and are not a pass
# --------------------------------------------------------------------------- #


def test_the_command_returns_zero_only_when_every_check_passes(
    tmp_path: Path,
    configs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    definition = write_definition(
        tmp_path, configs_dir, dataset_id="command-test", excitation_seeds=[43]
    )
    assert main([str(definition), "--no-figures"]) == 0
    assert "pass  no_hidden_information_in_the_available_branch" in capsys.readouterr().out

    found = {"files_scanned": ["x"], "findings": ["planted for the test"]}
    monkeypatch.setattr(pipeline, "scan_available", lambda *arguments: found)
    assert main([str(definition), "--no-figures"]) == 1
    printed = capsys.readouterr()
    assert "FAIL  no_hidden_information_in_the_available_branch" in printed.out
    assert "at least one mandatory check failed" in printed.err
    attempts = sorted((tmp_path / "data" / "private" / "datasets" / "command-test").iterdir())
    report = json.loads((attempts[-1] / "pipeline_report.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["hidden_information_scan"]["findings"] == ["planted for the test"]

    assert main([str(tmp_path / "missing.yaml")]) == 2


def test_a_failure_half_way_leaves_a_report_and_no_data_set(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    definition = write_definition(tmp_path, configs_dir, dataset_id="failing-test")
    real, calls = pipeline.generate_run, {"n": 0}

    def fails_on_the_third_run(*arguments: object, **options: object):  # noqa: ANN202
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("the true trajectory was not accepted (planted for the test)")
        return real(*arguments, **options)

    monkeypatch.setattr(pipeline, "generate_run", fails_on_the_third_run)
    with pytest.raises(ValueError, match="was not accepted"):
        pipeline.run_pipeline(definition, figures=False)

    datasets = tmp_path / "data" / "available" / "datasets"
    assert not (datasets / "failing-test").exists()
    assert list(datasets.iterdir()) == []  # not even the staging directory is left
    (attempt,) = (tmp_path / "data" / "private" / "datasets" / "failing-test").iterdir()
    report = json.loads((attempt / "pipeline_report.json").read_text(encoding="utf-8"))
    assert report["ok"] is False and "true_trajectories_accepted" not in report["checks"]


def test_an_export_is_refused_rather_than_trimmed_or_overwritten(generated) -> None:  # noqa: ANN001
    data, _, _ = generated
    connection = database.connect()
    database.ingest_dataset(connection, open_dataset("pipeline-test"))
    with pytest.raises(ExportError, match="holds no run of the data set 'elsewhere'"):
        export_dataset(connection, "elsewhere", {})
    connection.execute(
        "DELETE FROM measurements WHERE sensor_id = 'target.T_c' AND sample_index = 7"
    )
    with pytest.raises(ExportError, match="is not complete"):
        export_dataset(connection, "pipeline-test", {})
    exports = data / "available" / "exports"
    assert sorted(path.name for path in exports.iterdir()) == ["pipeline-test"]  # no staging left
    connection.close()


def test_an_export_with_other_content_under_the_same_name_is_a_conflict(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "first"))
    definition = write_definition(
        tmp_path, configs_dir, dataset_id="conflict-test", excitation_seeds=[44]
    )
    assert pipeline.run_pipeline(definition, figures=False)["ok"]
    shutil.copytree(
        tmp_path / "first" / "available" / "exports", tmp_path / "second" / "available" / "exports"
    )

    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "second"))
    other = write_definition(
        tmp_path, configs_dir, dataset_id="conflict-test", excitation_seeds=[45]
    )
    with pytest.raises(ExportConflictError, match="Nothing was overwritten"):
        pipeline.run_pipeline(other, figures=False)


# --------------------------------------------------------------------------- #
# Definitions and starting points
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"excitation_seeds": [1, 1]}, "distinct non-negative integers"),
        ({"excitation_seeds": []}, "distinct non-negative integers"),
        ({"excitation_seeds": [-1]}, "distinct non-negative integers"),
        ({"plants": []}, "one or more distinct files"),
        ({"n_excursions": 0}, "greater than or equal to 1"),
        ({"sensor_master_seed": -5}, "greater than or equal to 0"),
        ({"protocol": "p0"}, "Input should be 'p3'"),
        ({"simulation_period": {"value": 0.1, "unit": "K"}}, "requires time"),
        ({"amplitude": 0.2}, "Extra inputs are not permitted"),
    ],
)
def test_an_invalid_definition_is_refused(
    changes: dict, message: str, tmp_path: Path, configs_dir: Path
) -> None:
    with pytest.raises(ValidationError, match=message):
        load_dataset_definition(write_definition(tmp_path, configs_dir, **changes))


def test_the_starting_point_of_a_plant_is_verified_not_assumed(
    configs_dir: Path, tmp_path: Path
) -> None:
    for name, state in (("source", (250.02, 350.00)), ("target", (189.67, 355.17))):
        plant = load_virtual_plant(configs_dir / f"{name}_cstr.yaml")
        assert plant.nominal_state == pytest.approx(state, abs=0.01)
        assert plant.max_real_part * 60.0 < -0.5
        assert not hasattr(plant.spec, "true_physics")

    text = (configs_dir / "source_cstr.yaml").read_text(encoding="utf-8")

    def variant(name: str, **values: str) -> Path:
        changed = text
        for field, value in values.items():
            before = changed
            old = {"feed": "0.5, unit: mol/L", "coolant": "337.5, unit: K", "ua": "1.0e+5, unit"}[
                field
            ]
            changed = changed.replace(old, value, 1)
            assert changed != before, field
        path = tmp_path / f"{name}_cstr.yaml"
        path.write_text(changed, encoding="utf-8")
        return path

    # a richer feed, 0.8 mol/L: one steady state, at -0.17 1/min, short of the margin of D-017
    with pytest.raises(StartingPointError, match="not stable with the margin of D-017"):
        load_virtual_plant(variant("rich", feed="0.8, unit: mol/L"))
    # a rich feed with weak, cold cooling: three steady states under the true rate law
    with pytest.raises(StartingPointError, match="3 steady states were found"):
        load_virtual_plant(
            variant(
                "multiple", feed="1.0, unit: mol/L", coolant="310.0, unit: K", ua="5.0e+4, unit"
            )
        )
