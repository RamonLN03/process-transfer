"""Exports of aligned series: an export that is already on disk is verified, not taken on
the word of its manifest; an export is verified before it is published; and what is not
an export is refused when opened.

Regression tests for a defect reported in review: ``export_dataset`` compared only the
manifest of an existing export, so an export whose files had been altered, lost or added
to was reported as already present, and a directory without a manifest raised a bare
``FileNotFoundError``.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import synthetic_observations
from process_transfer.config import load_true_plant
from process_transfer.data import database, export
from process_transfer.data.export import (
    Export,
    ExportConflictError,
    ExportIntegrityError,
    export_dataset,
    open_export_directory,
)
from process_transfer.data.parquet_store import DatasetWriter, open_dataset
from process_transfer.data.records import PlantRecord, RunRecord, known_plant

DATASET = "export-test"
RUN = "target.p3.e0.x1.n0"
ATTEMPT = {"generated_at_utc": "2026-09-22T00:00:00+00:00", "commit": "0" * 40}


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


@pytest.fixture
def exported(plants: list[PlantRecord]):
    """A data set of two runs, ingested and exported once; the open connection and the
    export directory."""
    with DatasetWriter(DATASET, plants, ATTEMPT) as writer:
        for run in (
            synthetic_observations("source", "source.p3.e0.x1.n0", offset=60.0),
            synthetic_observations("target", RUN),
        ):
            writer.add_run(run, RunRecord(DATASET, "p3", "synthetic run for a test"))
        writer.publish()
    connection = database.connect()
    database.ingest_dataset(connection, open_dataset(DATASET))
    result = export_dataset(connection, DATASET, ATTEMPT)
    assert result.status == "created"
    yield connection, result.directory
    connection.close()


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def staging_left(directory: Path) -> list[Path]:
    return sorted(directory.parent.glob(".staging-*"))


def with_manifest(directory: Path, edit) -> None:  # noqa: ANN001
    manifest = json.loads((directory / "export.json").read_text(encoding="utf-8"))
    edit(manifest)
    (directory / "export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# An export that is already there is examined, not believed
# --------------------------------------------------------------------------- #


def test_an_export_that_was_altered_is_not_taken_for_the_same(exported) -> None:  # noqa: ANN001
    connection, directory = exported
    intact = snapshot(directory)
    run_file = directory / f"{RUN}.parquet"
    table = pq.read_table(run_file)

    def export_again() -> None:
        with pytest.raises(ExportIntegrityError, match="Nothing was overwritten") as caught:
            export_dataset(connection, DATASET, {"commit": "1" * 40})
        assert snapshot(directory) == damaged  # the damaged export is left as it is
        assert staging_left(directory) == []
        return str(caught.value)

    readings = table["T"].to_numpy().copy()
    readings[5] += 1.0e-9
    pq.write_table(table.set_column(5, table.schema.field("T"), pa.array(readings)), run_file)
    damaged = snapshot(directory)
    assert "does not have the content recorded" in export_again()

    run_file.unlink()
    damaged = snapshot(directory)
    assert f"missing ['{RUN}.parquet']" in export_again()

    pq.write_table(table, run_file)
    (directory / "notes.parquet").write_bytes(b"")
    damaged = snapshot(directory)
    assert "unexpected ['notes.parquet']" in export_again()
    (directory / "notes.parquet").unlink()

    manifest = (directory / "export.json").read_bytes()
    (directory / "export.json").unlink()
    damaged = snapshot(directory)
    assert "it is not an export, or its writing was interrupted" in export_again()
    (directory / "export.json").write_bytes(manifest)

    assert snapshot(directory) == intact
    result = export_dataset(connection, DATASET, {"commit": "1" * 40})
    assert result.status == "already_present"  # restored, it is the same export again
    assert snapshot(directory) == intact  # and another attempt does not touch it


def test_other_content_under_the_same_name_is_a_conflict_when_intact(exported) -> None:  # noqa: ANN001
    connection, directory = exported
    with_manifest(directory, lambda m: m["runs"][0].__setitem__("description", "edited by hand"))
    with pytest.raises(ExportConflictError, match="with other content. Nothing was overwritten"):
        export_dataset(connection, DATASET, ATTEMPT)
    with_manifest(directory, lambda m: m.__setitem__("version", 2))
    with pytest.raises(ExportIntegrityError, match="version 2; this code reads"):
        export_dataset(connection, DATASET, ATTEMPT)


def test_an_export_is_verified_before_it_is_published(
    exported,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same runs are exported under another name, through a writer that drops the
    last row of every file: the read-back must refuse it and nothing must be published."""
    connection, directory = exported
    written = pq.write_table

    def one_row_short(table: pa.Table, where: Path, **options: object) -> None:
        written(table.slice(0, table.num_rows - 1), where, **options)

    monkeypatch.setattr(export.pq, "write_table", one_row_short)
    connection.execute("UPDATE operating_runs SET dataset_id = 'second'")
    with pytest.raises(ExportIntegrityError, match="has 20 rows where 21 are listed"):
        export_dataset(connection, "second", ATTEMPT)
    assert not (directory.parent / "second").exists()
    assert staging_left(directory) == []


# --------------------------------------------------------------------------- #
# Opening an export
# --------------------------------------------------------------------------- #


def test_a_verified_export_gives_back_the_observations(exported) -> None:  # noqa: ANN001
    _, directory = exported
    opened = open_export_directory(directory)
    assert isinstance(opened, Export)
    assert opened.dataset_id == DATASET and opened.run_ids == ("source.p3.e0.x1.n0", RUN)
    back = opened.observations(RUN)
    original = synthetic_observations("target", RUN)
    assert back.content_digest() == original.content_digest()
    assert np.array_equal(back.measured, original.measured)
    assert opened.table(RUN).schema.names == [
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
    with pytest.raises(KeyError, match="has no run 'nowhere'"):
        opened.observations("nowhere")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m.__setitem__("dataset_id", "Export Test"), "dataset_id must be lower-case"),
        (lambda m: m.__setitem__("content_encoding", "observations/v0"), "hashes its content"),
        (lambda m: m["runs"][1].__setitem__("plant_id", "elsewhere"), "does not describe"),
        (lambda m: m["runs"][1].__setitem__("file", "source.p3.e0.x1.n0.parquet"), "listed with"),
        (lambda m: m["runs"][1].__setitem__("rows", 20), "has 21 rows where 20 are listed"),
        (lambda m: m["runs"][1].__setitem__("content_sha256", "abc"), "no SHA-256"),
        (lambda m: m["runs"][1].__setitem__("end_time_s", 121.0), "the manifest says"),
        (lambda m: m["runs"].append(dict(m["runs"][1])), "listed twice"),
        (lambda m: m["plants"].pop("source"), "does not describe"),
        (
            lambda m: m["plants"]["target"]["channels"][5].__setitem__("noise_std", 0.0),
            "presented with noise",
        ),
        (
            lambda m: m["plants"]["target"]["channels"][0].__setitem__("noise_model", None),
            "needs a noise model",
        ),
        (
            lambda m: m["plants"]["target"]["channels"][0].__setitem__("channel_kind", "state"),
            "not one of",
        ),
        (
            lambda m: m["plants"]["target"]["channels"][1].__setitem__("unit", "degC"),
            "does not have the schema of its run",
        ),
    ],
)
def test_what_is_not_an_export_is_refused(exported, edit, message: str) -> None:  # noqa: ANN001
    _, directory = exported
    with_manifest(directory, edit)
    with pytest.raises(ExportIntegrityError, match=message):
        open_export_directory(directory)


def test_rows_that_do_not_match_their_manifest_are_refused(exported) -> None:  # noqa: ANN001
    _, directory = exported
    run_file = directory / f"{RUN}.parquet"
    table = pq.read_table(run_file)

    def refused(changed: pa.Table, message: str) -> None:
        pq.write_table(changed, run_file)
        with pytest.raises(ExportIntegrityError, match=message):
            open_export_directory(directory)

    ticks = table["sample_index"].to_numpy().copy()
    ticks[3], ticks[4] = ticks[4], ticks[3]
    refused(table.set_column(2, table.schema.field("sample_index"), pa.array(ticks)), "in order")

    other = pa.array(["source"] + ["target"] * (table.num_rows - 1))
    refused(table.set_column(0, table.schema.field("plant_id"), other), "not only 'target'")

    refused(table.drop_columns(["T_c"]), "does not have the schema of its run")
    pq.write_table(table, run_file)
    open_export_directory(directory)  # restored, it opens again


def test_the_figures_are_drawn_from_a_verified_export(exported, tmp_path: Path) -> None:  # noqa: ANN001
    pytest.importorskip("matplotlib")
    from process_transfer.generation.figures import figure_readings

    _, directory = exported
    (directory / "run.parquet").write_bytes(b"")
    with pytest.raises(ExportIntegrityError, match="unexpected \\['run.parquet'\\]"):
        figure_readings(directory, tmp_path)
    assert list(tmp_path.glob("*.png")) == []
