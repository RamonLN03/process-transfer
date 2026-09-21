"""DuckDB: constraints, atomic and idempotent ingestion, conflicts, and reading back.

The SQL is in the files under ``sql/``; these tests run it on small examples whose
results are known. The quality queries and the analysis queries are in
``test_sql_queries.py``.
"""

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import synthetic_observations
from process_transfer.config import load_true_plant
from process_transfer.data import database
from process_transfer.data.database import (
    IngestionConflictError,
    IngestionQualityError,
    connect,
    database_path,
    ingest_dataset,
    ingest_run,
    observations_from_database,
    quality_report,
)
from process_transfer.data.parquet_store import Dataset, DatasetWriter, open_dataset
from process_transfer.data.records import KnownParameter, PlantRecord, RunRecord, known_plant
from process_transfer.measurement.observations import Observations

ATTEMPT = {"generated_at_utc": "2026-09-21T00:00:00+00:00"}


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


def runs() -> list[Observations]:
    return [
        synthetic_observations("source", "source.p3.e0.x1.n0", offset=60.0),
        synthetic_observations("source", "source.p3.e1.x1.n0", offset=61.0),
        synthetic_observations("target", "target.p3.e0.x1.n0"),
        synthetic_observations("target", "target.p3.e1.x1.n0", offset=1.0),
    ]


def published(
    plants: list[PlantRecord], observations: list[Observations], name: str = "db-test"
) -> Dataset:
    with DatasetWriter(name, plants, ATTEMPT) as writer:
        for run in observations:
            writer.add_run(run, RunRecord(name, "p3", "synthetic run for a test"))
        writer.publish()
    return open_dataset(name)


def counts(connection: duckdb.DuckDBPyConnection, schema_name: str = "main") -> dict[str, int]:
    return {
        table: connection.execute(f"SELECT count(*) FROM {schema_name}.{table}").fetchone()[0]
        for table in database.TABLES
    }


# --------------------------------------------------------------------------- #
# Ingestion and reading back
# --------------------------------------------------------------------------- #


def test_a_data_set_goes_in_and_comes_back_bit_for_bit(plants: list[PlantRecord]) -> None:
    dataset = published(plants, runs())
    connection = connect()
    assert ingest_dataset(connection, dataset) == {
        run.run: "ingested" for run in sorted(runs(), key=lambda r: r.run)
    }
    assert counts(connection) == {
        "plants": 2,
        "process_parameters": 16,
        "sensors": 12,
        "operating_runs": 4,
        "measurements": 4 * 21 * 6,
    }
    assert counts(connection, "staging") == dict.fromkeys(
        database.TABLES, 0
    )  # nothing is left there
    assert database.failed_checks(quality_report(connection)) == {}

    for run in runs():
        back = observations_from_database(connection, run.run)
        for field in ("times", "measured", "inputs"):
            np.testing.assert_array_equal(getattr(back, field), getattr(run, field))
        assert back.content_digest() == run.content_digest()
        assert back.content_digest() == dataset.observations(run.run).content_digest()


def test_ingesting_again_changes_nothing(plants: list[PlantRecord]) -> None:
    dataset = published(plants, runs())
    connection = connect()
    ingest_dataset(connection, dataset)
    before = counts(connection)
    assert set(ingest_dataset(connection, dataset).values()) == {"already_present"}
    assert counts(connection) == before
    assert counts(connection, "staging") == dict.fromkeys(database.TABLES, 0)


def test_a_database_file_keeps_what_was_ingested(
    plants: list[PlantRecord], data_root: Path
) -> None:
    dataset = published(plants, runs())
    path = database_path("db-test")
    assert path == data_root / "available" / "databases" / "db-test.duckdb"
    connection = connect(path)
    ingest_dataset(connection, dataset)
    connection.close()

    reader = connect(path, read_only=True)
    assert counts(reader)["measurements"] == 4 * 21 * 6
    assert (
        observations_from_database(reader, "target.p3.e0.x1.n0").content_digest()
        == runs()[2].content_digest()
    )
    with pytest.raises(duckdb.Error):
        reader.execute("DELETE FROM measurements")
    reader.close()
    with pytest.raises(ValueError, match="database name"):
        database_path("../outside")


# --------------------------------------------------------------------------- #
# Conflicts
# --------------------------------------------------------------------------- #


def test_the_same_run_with_other_content_is_a_conflict(plants: list[PlantRecord]) -> None:
    connection = connect()
    ingest_dataset(connection, published(plants, runs()))
    before = counts(connection)

    # the files of the run are replaced on disk after the data set was published: the
    # same data set, the same run, other content
    dataset = open_dataset("db-test")
    altered = synthetic_observations("target", "target.p3.e0.x1.n0", offset=1.0e-9)
    other = published(plants[1:], [altered], name="db-test-scratch")
    path = dataset.measurement_path("target.p3.e0.x1.n0")
    path.write_bytes(other.measurement_path("target.p3.e0.x1.n0").read_bytes())
    with pytest.raises(IngestionConflictError, match="files changed after the data set"):
        ingest_run(connection, dataset, "target.p3.e0.x1.n0")

    # the same run offered by another data set, even with the same content
    twin = published(plants, runs(), name="db-test-twin")
    with pytest.raises(IngestionConflictError, match="as part of the data set 'db-test'"):
        ingest_run(connection, twin, "source.p3.e0.x1.n0")
    assert counts(connection) == before
    assert counts(connection, "staging") == dict.fromkeys(database.TABLES, 0)
    np.testing.assert_array_equal(
        observations_from_database(connection, "target.p3.e0.x1.n0").measured, runs()[2].measured
    )  # the first content is still there


def test_a_data_set_regenerated_with_other_content_does_not_replace_what_is_stored(
    plants: list[PlantRecord], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The database outlives its data sets. If the data directory is wiped and the same
    data set is generated again with other seeds, the same identities arrive with other
    content: that is a conflict, and what the database holds stays as it was."""
    connection = connect()
    ingest_dataset(connection, published(plants, runs()))
    before = counts(connection)

    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "another-data-directory"))
    regenerated = runs()
    regenerated[2] = synthetic_observations("target", "target.p3.e0.x1.n0", offset=0.5)
    again = published(plants, regenerated)  # the same dataset_id, "db-test"
    assert ingest_run(connection, again, "source.p3.e0.x1.n0") == "already_present"
    with pytest.raises(IngestionConflictError, match="with different content"):
        ingest_run(connection, again, "target.p3.e0.x1.n0")
    assert counts(connection) == before
    np.testing.assert_array_equal(
        observations_from_database(connection, "target.p3.e0.x1.n0").measured, runs()[2].measured
    )


def test_a_plant_described_differently_is_a_conflict(plants: list[PlantRecord]) -> None:
    connection = connect()
    ingest_dataset(connection, published(plants, runs()))
    before = counts(connection)
    source, target = plants

    bigger = PlantRecord(
        "target",
        "target",
        "cstr",
        tuple(
            KnownParameter(p.parameter, 0.2 if p.parameter == "reactor_volume" else p.value, p.unit)
            for p in target.parameters
        ),
    )
    new_run = synthetic_observations("target", "target.p3.e7.x1.n0")
    with pytest.raises(IngestionConflictError, match="process_parameters of plant 'target'"):
        ingest_dataset(connection, published([bigger], [new_run], name="db-test-volume"))

    quieter = Observations(**{**vars(new_run), "noise_std": (3.8, 0.5)})
    with pytest.raises(IngestionConflictError, match="sensors of plant 'target'"):
        ingest_dataset(connection, published([target], [quieter], name="db-test-noise"))
    assert counts(connection) == before

    # the same plant, described the same way, takes new runs
    assert ingest_dataset(connection, published([target], [new_run], name="db-test-more")) == {
        "target.p3.e7.x1.n0": "ingested"
    }
    assert source.plant_id == "source"


# --------------------------------------------------------------------------- #
# Atomicity
# --------------------------------------------------------------------------- #


def test_a_failure_half_way_leaves_nothing_of_the_run(
    plants: list[PlantRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = published(plants, runs())
    connection = connect()
    ingest_run(connection, dataset, "source.p3.e0.x1.n0")
    before = counts(connection)

    def dies_after_the_run_row(connection: duckdb.DuckDBPyConnection, plant_is_new: bool) -> None:
        for table in ("plants", "process_parameters", "sensors", "operating_runs"):
            if plant_is_new or table == "operating_runs":
                connection.execute(f"INSERT INTO main.{table} SELECT * FROM staging.{table}")
        assert connection.execute("SELECT count(*) FROM main.operating_runs").fetchone()[0] == 2
        raise OSError("the disk went away")

    monkeypatch.setattr(database, "_insert_run", dies_after_the_run_row)
    with pytest.raises(OSError, match="the disk went away"):
        ingest_run(connection, dataset, "target.p3.e0.x1.n0")  # a new plant and a new run
    monkeypatch.undo()

    assert counts(connection) == before  # no plant, no channel, no run, no measurement of it
    assert counts(connection, "staging") == dict.fromkeys(database.TABLES, 0)
    assert connection.execute("SELECT current_schema()").fetchone()[0] == "main"
    assert (
        ingest_run(connection, dataset, "target.p3.e0.x1.n0") == "ingested"
    )  # and it can be retried
    assert database.failed_checks(quality_report(connection)) == {}


def tampered(dataset: Dataset, run_id: str, change) -> None:  # noqa: ANN001
    """Alter the file of a run after the data set was opened and verified."""
    path = dataset.measurement_path(run_id)
    pq.write_table(change(pq.read_table(path)), path)


def test_files_that_changed_after_verification_do_not_get_in(plants: list[PlantRecord]) -> None:
    dataset = published(plants, runs())
    connection = connect()
    run_id = "target.p3.e0.x1.n0"

    def with_value(table: pa.Table, row: int, value: float) -> pa.Table:
        values = table["value"].to_numpy().copy()
        values[row] = value
        return table.set_column(5, table.schema.field("value"), pa.array(values))

    original = pq.read_table(dataset.measurement_path(run_id))
    cases = {
        "q01_duplicates": lambda t: pa.concat_tables([t, t.slice(3, 1)]),
        "q04_non_finite": lambda t: with_value(t, 4, float("nan")),
        "q06_cadence": lambda t: t.slice(0, t.num_rows - 1),
    }
    for check, change in cases.items():
        tampered(dataset, run_id, change)
        with pytest.raises(IngestionQualityError) as caught:
            ingest_run(connection, dataset, run_id)
        assert check in caught.value.findings, (check, caught.value.findings)
        assert counts(connection) == dict.fromkeys(database.TABLES, 0)
        pq.write_table(original, dataset.measurement_path(run_id))

    # a changed reading passes every record-level check, and is caught by the content hash
    tampered(dataset, run_id, lambda t: with_value(t, 4, 123.456))
    with pytest.raises(
        IngestionConflictError, match="files changed after the data set was verified"
    ):
        ingest_run(connection, dataset, run_id)
    assert counts(connection) == dict.fromkeys(database.TABLES, 0)


# --------------------------------------------------------------------------- #
# The constraints of the main schema, the second line of defence
# --------------------------------------------------------------------------- #


RUN = "target.p3.e0.x1.n0"


def row(
    plant: str, run: str, sensor: str, tick: int, time: object, value: object, flag: int
) -> str:
    """One row of ``measurements`` as SQL values; ``time`` and ``value`` may be SQL text."""
    return f"('{plant}', '{run}', '{sensor}', {tick}, {time}, {value}, {flag})"


def test_the_main_schema_refuses_what_the_contract_forbids(plants: list[PlantRecord]) -> None:
    connection = connect()
    ingest_dataset(connection, published(plants, runs()))
    good = row("target", RUN, "target.T", 500, 3000.0, 355.0, 0)
    connection.execute(f"INSERT INTO measurements VALUES {good}")  # a new tick of an existing run

    refused = {
        "the same row again": good,
        "the sensor of another plant": row("target", RUN, "source.T", 501, 3006.0, 355.0, 0),
        "the run of another plant": row("source", RUN, "source.T", 501, 3006.0, 355.0, 0),
        "a run that does not exist": row(
            "target", "target.p3.e9.x1.n0", "target.T", 0, 0.0, 355.0, 0
        ),
        "a reading that is not a number": row("target", RUN, "target.T", 501, 3006.0, "'NaN'", 0),
        "an infinite instant": row("target", RUN, "target.T", 501, "'Infinity'", 355.0, 0),
        "a missing reading": row("target", RUN, "target.T", 501, 3006.0, "NULL", 0),
        "a negative tick": row("target", RUN, "target.T", -1, -6.0, 355.0, 0),
        "an undefined quality flag": row("target", RUN, "target.T", 501, 3006.0, 355.0, 7),
    }
    for label, values in refused.items():
        with pytest.raises(duckdb.ConstraintException):
            connection.execute(f"INSERT INTO measurements VALUES {values}")
        assert label

    # a negative reading is not forbidden: it is what a noisy sensor can give
    negative = row("target", RUN, "target.C_A", 500, 3000.0, -7.5, 0)
    connection.execute(f"INSERT INTO measurements VALUES {negative}")

    channel = (
        "INSERT INTO sensors VALUES ('target.{0}', 'target', '{0}', '{1}', 9, 'K', 6.0, {2}, {3})"
    )
    with pytest.raises(duckdb.ConstraintException):  # a known input is not a noisy sensor
        connection.execute(channel.format("x", "input", "'additive_gaussian'", "0.0"))
    with pytest.raises(duckdb.ConstraintException):  # and a sensor has a noise level
        connection.execute(channel.format("y", "measured", "'additive_gaussian'", "NULL"))
    connection.execute(channel.format("z", "input", "NULL", "NULL"))  # a valid known input


def test_the_database_holds_the_five_tables_and_nothing_else() -> None:
    connection = connect()
    catalogue = "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = "
    tables = connection.execute(catalogue + "'BASE TABLE' ORDER BY ALL").fetchall()
    assert tables == [
        (schema_name, table)
        for schema_name in ("main", "staging")
        for table in sorted(database.TABLES)
    ]
    views = connection.execute(
        catalogue + "'VIEW' AND table_schema = 'main' ORDER BY ALL"
    ).fetchall()
    assert views == [("main", "aligned_series"), ("main", "lagged_measurements")]
