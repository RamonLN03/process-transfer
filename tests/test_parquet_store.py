"""Immutable Parquet data sets: round trip by content, idempotent repetition, conflict,
interrupted writing, integrity, and reading without the private branch.

Round trips are compared by content and metadata. No test depends on the bytes of a
Parquet file (D-012).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import synthetic_observations
from process_transfer.config import load_true_plant
from process_transfer.data import parquet_store, schema
from process_transfer.data.parquet_store import (
    DatasetConflictError,
    DatasetError,
    DatasetIntegrityError,
    DatasetWriter,
    open_dataset,
    open_dataset_directory,
)
from process_transfer.data.records import PlantRecord, RunRecord, known_plant
from process_transfer.measurement.observations import Observations

DATASET = "unit-test"
ATTEMPT = {"generated_at_utc": "2026-09-21T00:00:00+00:00", "commit": "0" * 40}
RECORD = RunRecord(DATASET, "p3", "synthetic run for a test")


@pytest.fixture(autouse=True)
def data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "pt-data"))
    return tmp_path / "pt-data"


@pytest.fixture(scope="module")
def plants(configs_dir: Path) -> list[PlantRecord]:
    return [
        known_plant(load_true_plant(configs_dir / f"{name}_cstr.yaml").plant)
        for name in ("source", "target")
    ]


def four_runs() -> list[Observations]:
    return [
        synthetic_observations("source", "source.p3.e0.x1.n0", offset=60.0),
        synthetic_observations("source", "source.p3.e1.x1.n0", offset=61.0),
        synthetic_observations("target", "target.p3.e0.x1.n0"),
        synthetic_observations("target", "target.p3.e1.x1.n0", offset=1.0),
    ]


def publish(plants: list[PlantRecord], runs: list[Observations], dataset_id: str = DATASET):
    record = RunRecord(dataset_id, "p3", "synthetic run for a test")
    with DatasetWriter(dataset_id, plants, ATTEMPT) as writer:
        for run in runs:
            writer.add_run(run, record)
        return writer.publish()


def staging_directories(root: Path) -> list[Path]:
    datasets = root / "available" / "datasets"
    return sorted(datasets.glob(".staging-*")) if datasets.is_dir() else []


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_observations_come_back_bit_for_bit_with_their_metadata(
    plants: list[PlantRecord], data_root: Path
) -> None:
    runs = four_runs()
    result = publish(plants, runs)
    assert result.status == "created"
    assert result.directory == data_root / "available" / "datasets" / DATASET

    dataset = open_dataset(DATASET)
    assert dataset.run_ids == tuple(sorted(run.run for run in runs))
    for run in runs:
        back = dataset.observations(run.run)
        for field in ("times", "measured", "inputs"):
            np.testing.assert_array_equal(getattr(back, field), getattr(run, field))
        for field in ("plant", "run", "measured_names", "measured_units", "input_names"):
            assert getattr(back, field) == getattr(run, field)
        assert back.input_units == run.input_units and back.noise_std == run.noise_std
        assert back.sample_period == run.sample_period
        assert back.content_digest() == run.content_digest()


def test_the_tables_say_what_the_contract_says(plants: list[PlantRecord]) -> None:
    publish(plants, four_runs())
    dataset = open_dataset(DATASET)

    sensors = dataset.table("sensors").to_pylist()
    assert len(sensors) == 12  # six channels per plant
    by_id = {row["sensor_id"]: row for row in sensors}
    reading, known_input = by_id["target.C_A"], by_id["target.T_c"]
    assert (reading["channel_kind"], reading["noise_model"], reading["noise_std"]) == (
        "measured",
        "additive_gaussian",
        5.0,
    )
    assert reading["unit"] == "mol/m^3" and reading["sampling_period_s"] == 6.0
    # an input is known exactly: it has no noise model, which is not a noise of zero
    assert (known_input["channel_kind"], known_input["noise_model"], known_input["noise_std"]) == (
        "input",
        None,
        None,
    )
    assert [by_id[f"target.{name}"]["channel_index"] for name in ("q", "C_Af", "T_f", "T_c")] == [
        0,
        1,
        2,
        3,
    ]

    (run,) = [
        row
        for row in dataset.table("operating_runs").to_pylist()
        if row["run_id"] == "target.p3.e0.x1.n0"
    ]
    assert (run["start_time_s"], run["end_time_s"], run["n_samples"]) == (0.0, 120.0, 21)
    assert run["dataset_id"] == DATASET and run["plant_id"] == "target"
    assert run["content_sha256"] == four_runs()[2].content_digest()

    long = dataset.measurements("target.p3.e0.x1.n0")
    assert long.num_rows == 21 * 6
    assert set(long["quality_flag"].to_pylist()) == {schema.QUALITY_GOOD}
    assert long.schema.field("time_s").metadata == {b"unit": b"s"}
    assert long.schema.field("value").type == pa.float64()
    assert not long.schema.field("value").nullable


def test_readings_and_instants_are_stored_as_they_are(plants: list[PlantRecord]) -> None:
    """No rounding, no clipping: awkward doubles, a negative concentration reading and a
    reading far above any true state all come back identical."""
    run = synthetic_observations()
    measured = run.measured.copy()
    measured[:6, 0] = [-2.5, 5e-324, float(np.nextafter(190.0, 191.0)), -0.0, 1.0e305, 0.1 * 3]
    awkward = Observations(**{**vars(run), "measured": measured})
    publish(plants[1:], [awkward])

    back = open_dataset(DATASET).observations(run.run)
    assert np.array_equal(back.measured.view(np.uint64), measured.view(np.uint64))  # bit for bit
    assert back.measured[0, 0] == -2.5 and np.signbit(back.measured[3, 0])


def test_switching_instants_and_the_new_inputs_are_preserved(plants: list[PlantRecord]) -> None:
    run = synthetic_observations()
    publish(plants[1:], [run])
    long = open_dataset(DATASET).measurements(run.run).to_pandas()
    coolant = long[long["sensor_id"] == "target.T_c"].sort_values("sample_index")
    changed = coolant[coolant["value"].diff().fillna(0.0) != 0.0]
    assert changed["sample_index"].tolist() == [10] and changed["time_s"].tolist() == [60.0]
    assert changed["value"].tolist() == [342.5]  # the row of the switch carries the new input
    assert coolant["value"].iloc[9] == 337.5


def test_a_run_that_does_not_start_at_zero_keeps_its_instants(plants: list[PlantRecord]) -> None:
    late = synthetic_observations()
    late = Observations(**{**vars(late), "times": late.times + 720.0})
    publish(plants[1:], [late])
    dataset = open_dataset(DATASET)
    (row,) = dataset.table("operating_runs").to_pylist()
    assert (row["start_time_s"], row["end_time_s"]) == (720.0, 840.0)
    long = dataset.measurements(late.run)
    assert long["sample_index"].to_pylist()[:3] == [0, 1, 2]
    assert long["time_s"].to_pylist()[:3] == [720.0, 726.0, 732.0]


# --------------------------------------------------------------------------- #
# Repetition, conflict, interruption
# --------------------------------------------------------------------------- #


def snapshot(directory: Path) -> dict[str, tuple[int, bytes]]:
    return {
        path.relative_to(directory).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_writing_the_same_data_set_again_changes_nothing(
    plants: list[PlantRecord], data_root: Path
) -> None:
    first = publish(plants, four_runs())
    before = snapshot(first.directory)
    again = publish(plants, list(reversed(four_runs())))  # the order of writing is not content
    assert again.status == "already_present"
    assert snapshot(first.directory) == before  # not a byte, not a timestamp
    assert staging_directories(data_root) == []


def test_the_same_identity_with_other_content_is_a_conflict(
    plants: list[PlantRecord], data_root: Path
) -> None:
    first = publish(plants, four_runs())
    before = snapshot(first.directory)
    altered = four_runs()
    altered[2] = synthetic_observations("target", "target.p3.e0.x1.n0", offset=1.0e-9)
    with pytest.raises(DatasetConflictError, match="Nothing was overwritten"):
        publish(plants, altered)
    assert snapshot(first.directory) == before
    assert staging_directories(data_root) == []
    with pytest.raises(DatasetConflictError):  # fewer runs is other content too
        publish(plants, four_runs()[:3] + [synthetic_observations("target", "target.p3.e9.x1.n0")])


def test_an_interrupted_writing_is_not_a_data_set(
    plants: list[PlantRecord], data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = data_root / "available" / "datasets" / DATASET

    # the process dies after two runs: nothing removes the staging directory
    writer = DatasetWriter(DATASET, plants, ATTEMPT)
    for run in four_runs()[:2]:
        writer.add_run(run, RECORD)
    assert not final.exists()
    (left_behind,) = staging_directories(data_root)
    with pytest.raises(
        DatasetIntegrityError, match="not a data set, or its writing was interrupted"
    ):
        open_dataset(DATASET)
    with pytest.raises(DatasetIntegrityError, match="interrupted"):
        open_dataset_directory(left_behind)  # and what was left is not taken for one

    # a failure while the tables are being written, inside publish
    calls = {"n": 0}
    real = pq.write_table

    def fails_on_the_third_table(table: pa.Table, where: Path, **options: object) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        real(table, where, **options)

    with DatasetWriter(DATASET, plants, ATTEMPT) as second:
        for run in four_runs():
            second.add_run(run, RECORD)
        monkeypatch.setattr(parquet_store.pq, "write_table", fails_on_the_third_table)
        with pytest.raises(OSError, match="disk full"):
            second.publish()
    monkeypatch.setattr(parquet_store.pq, "write_table", real)
    assert not final.exists()
    assert staging_directories(data_root) == [left_behind]  # the second cleaned up after itself

    # the data set can still be written afterwards, and the leftover is not in the way
    assert publish(plants, four_runs()).status == "created"
    assert open_dataset(DATASET).run_ids == tuple(sorted(run.run for run in four_runs()))


def test_a_writer_is_used_once(plants: list[PlantRecord]) -> None:
    with DatasetWriter(DATASET, plants[1:], ATTEMPT) as writer:
        writer.add_run(synthetic_observations(), RECORD)
        writer.publish()
        with pytest.raises(DatasetError, match="already published"):
            writer.add_run(synthetic_observations(run="target.p3.e5.x1.n0"), RECORD)
        with pytest.raises(DatasetError, match="already published"):
            writer.publish()


# --------------------------------------------------------------------------- #
# Integrity of what is on disk
# --------------------------------------------------------------------------- #


def test_a_data_set_that_was_altered_is_not_opened(
    plants: list[PlantRecord], data_root: Path
) -> None:
    directory = publish(plants, four_runs()).directory
    target = directory / "measurements" / "target.p3.e0.x1.n0.parquet"
    original = target.read_bytes()

    table = pq.read_table(target)
    values = table["value"].to_numpy().copy()
    values[5] += 1.0e-9
    pq.write_table(table.set_column(5, table.schema.field("value"), pa.array(values)), target)
    with pytest.raises(DatasetIntegrityError, match="does not have the content recorded"):
        open_dataset(DATASET)

    narrowed = table.set_column(
        5, pa.field("value", pa.float32(), nullable=False), pa.array(values, pa.float32())
    )
    pq.write_table(narrowed, target)
    with pytest.raises(DatasetIntegrityError, match="does not have the schema of measurements"):
        open_dataset(DATASET)

    pq.write_table(table.slice(0, table.num_rows - 1), target)
    with pytest.raises(DatasetIntegrityError, match="rows"):
        open_dataset(DATASET)

    target.write_bytes(original)
    open_dataset(DATASET)  # restored, it opens again

    (directory / "notes.txt").write_text("left here by hand", encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="unexpected \\['notes.txt'\\]"):
        open_dataset(DATASET)
    (directory / "notes.txt").unlink()

    target.unlink()
    with pytest.raises(
        DatasetIntegrityError, match="missing \\['measurements/target.p3.e0.x1.n0.parquet'\\]"
    ):
        open_dataset(DATASET)


def test_a_small_table_that_was_altered_is_not_opened(plants: list[PlantRecord]) -> None:
    directory = publish(plants, four_runs()).directory
    path = directory / "sensors.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["noise_std"] = 3.8  # someone edits the instrument specification by hand
    pq.write_table(pa.Table.from_pylist(rows, schema=schema.SENSORS), path)
    with pytest.raises(
        DatasetIntegrityError, match="sensors in .* is not the table of the manifest"
    ):
        open_dataset(DATASET)


def test_a_manifest_of_another_contract_is_refused(plants: list[PlantRecord]) -> None:
    directory = publish(plants, four_runs()).directory
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for key, value, message in (
        ("schema_version", 99, "version 99"),
        ("contract", "something/else", "something/else"),
        ("encodings", {"runs": "observations/v0", "tables": "table/v1"}, "observations/v0"),
    ):
        (directory / "manifest.json").write_text(
            json.dumps({**manifest, key: value}), encoding="utf-8"
        )
        with pytest.raises(DatasetIntegrityError, match=message):
            open_dataset(DATASET)
    assert "seed" not in json.dumps(manifest).lower()
    assert (
        manifest["attempt"] == ATTEMPT and "time_s is process time" in manifest["time_convention"]
    )


# --------------------------------------------------------------------------- #
# What the writer refuses
# --------------------------------------------------------------------------- #


def with_changes(**changes: object) -> Observations:
    return Observations(**{**vars(synthetic_observations()), **changes})


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        (with_changes(plant="Target"), "plant must be lower-case ASCII"),
        (with_changes(run="target/../escape"), "run must be lower-case ASCII"),
        (with_changes(run="nul.p3"), "Windows device"),
        (with_changes(plant="elsewhere"), "not among the plants of this data set"),
        (with_changes(measured_units=("mol/L", "K")), "stored data are SI"),
        (with_changes(input_units=("m^3/s", "mol/m^3", "K", "furlong")), "is not known"),
        (with_changes(input_names=("q", "C_A", "T_f", "T_c")), "every channel needs its own name"),
        (with_changes(measured_names=("C_A", " ")), "must be a non-empty string"),
        (
            with_changes(times=6.0 * np.arange(21) + 0.25 * (np.arange(21) == 7)),
            "not on the sampling clock",
        ),
        (
            with_changes(times=6.0 * (np.arange(21) + (np.arange(21) > 7))),
            "the run is not complete",
        ),
        (with_changes(sample_period=5.0), "not on the sampling clock"),
    ],
)
def test_observations_that_cannot_be_stored_are_refused(
    observations: Observations, message: str, plants: list[PlantRecord], data_root: Path
) -> None:
    with (
        DatasetWriter(DATASET, plants, ATTEMPT) as writer,
        pytest.raises(ValueError, match=message),
    ):
        writer.add_run(observations, RECORD)
    assert staging_directories(data_root) == []
    assert not (data_root / "available" / "datasets" / DATASET).exists()


def test_what_does_not_belong_to_the_data_set_is_refused(plants: list[PlantRecord]) -> None:
    with DatasetWriter(DATASET, plants, ATTEMPT) as writer:
        writer.add_run(synthetic_observations(), RECORD)
        with pytest.raises(DatasetConflictError, match="already added"):
            writer.add_run(synthetic_observations(), RECORD)
        with pytest.raises(ValueError, match="belongs to 'another'"):
            writer.add_run(
                synthetic_observations(run="target.p3.e1.x1.n0"), RunRecord("another", "p3", "x")
            )
        with pytest.raises(DatasetConflictError, match="channels of plant 'target'"):
            writer.add_run(with_changes(run="target.p3.e2.x1.n0", noise_std=(3.8, 0.5)), RECORD)
        with pytest.raises(TypeError, match="expected Observations"):
            writer.add_run({"times": [0.0]}, RECORD)  # type: ignore[arg-type]
        with pytest.raises(DatasetError, match="no run was added for the plants \\['source'\\]"):
            writer.publish()

    with (
        DatasetWriter(DATASET, plants[1:], ATTEMPT) as empty,
        pytest.raises(DatasetError, match="without runs"),
    ):
        empty.publish()
    with pytest.raises(ValueError, match="dataset_id"):
        DatasetWriter("Data Set", plants, ATTEMPT)
    with pytest.raises(ValueError, match="at least one plant"):
        DatasetWriter(DATASET, [], ATTEMPT)
    with pytest.raises(ValueError, match="each given once"):
        DatasetWriter(DATASET, [plants[0], plants[0]], ATTEMPT)
    with pytest.raises(ValueError, match="serialisable as JSON"):
        DatasetWriter(DATASET, plants, {"when": object()})


# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #


def test_the_location_follows_pt_data_dir_and_not_the_working_directory(
    plants: list[PlantRecord], data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "some" / "working" / "directory"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)  # as an IDE may do
    directory = publish(plants[1:], [synthetic_observations()]).directory
    assert directory == data_root / "available" / "datasets" / DATASET
    assert list(elsewhere.iterdir()) == []
    assert open_dataset(DATASET).directory == directory


def test_reading_needs_nothing_but_the_data_set_itself(
    plants: list[PlantRecord], data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data set is copied alone to a place that has no private branch and no other
    data set, and is read from there."""
    directory = publish(plants, four_runs()).directory
    (data_root / "private" / "datasets" / DATASET).mkdir(parents=True)
    (data_root / "private" / "datasets" / DATASET / "provenance.json").write_text(
        '{"sensor_seed": 5}', encoding="utf-8"
    )

    alone = tmp_path / "handed-over" / "available" / "datasets" / DATASET
    shutil.copytree(directory, alone)
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "handed-over"))
    assert not (tmp_path / "handed-over" / "private").exists()
    dataset = open_dataset(DATASET)
    assert dataset.directory == alone
    assert (
        dataset.observations("target.p3.e0.x1.n0").content_digest()
        == four_runs()[2].content_digest()
    )
    assert not (tmp_path / "handed-over" / "private").exists()  # and reading did not create it
