"""DuckDB as a derived analytical store of published data sets. No ORM: the schema and
the queries are the readable files under ``sql/``, and this module only runs them.

    sql/schema/     the five tables with their constraints, and a staging schema without
    sql/quality/    checks that return the defective rows; no row means a pass
    sql/views/      alignment and lags
    sql/analysis/   comparison of plants and aggregation in time, with parameters

Ingestion of one run is one transaction. The rows of the run are loaded from the files
named in the manifest, never from a wildcard, into the staging schema; the quality
queries must return nothing; the content rebuilt from the staged rows must have the hash
recorded for the run; and only then are the rows inserted into the main schema. Anything
that fails rolls the whole transaction back, so a run is in the database entirely or not
at all.

The policy on repetition is that of the data sets. A run that is already there with the
same content is accepted and changes nothing. The same identity with other content, for
a run, a plant, its parameters or its channels, is a conflict and an error.

A database can be rebuilt from its data sets at any time. It lives under
``PT_DATA_DIR/available/databases``, reads only the available branch, and holds nothing
that the data sets do not hold.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from process_transfer.data import schema
from process_transfer.data.identifiers import path_identifier
from process_transfer.data.parquet_store import Dataset, observations_from_long
from process_transfer.data.paths import data_dir, repository_root
from process_transfer.measurement.observations import Observations

TABLES = ("plants", "process_parameters", "sensors", "operating_runs", "measurements")


class DatabaseError(Exception):
    """Something is wrong with a request to the database or with what it holds."""


class IngestionConflictError(DatabaseError):
    """The same identity with different content. Nothing was changed."""


class IngestionQualityError(DatabaseError):
    """The staged rows of a run failed the quality queries. Nothing was ingested."""

    def __init__(self, run_id: str, findings: Mapping[str, list[dict[str, object]]]) -> None:
        self.run_id = run_id
        self.findings = dict(findings)
        lines = [
            f"  {name}: {len(rows)} finding(s), first: {rows[0]}" for name, rows in findings.items()
        ]
        super().__init__(
            f"the run {run_id!r} failed the quality checks and was not ingested:\n"
            + "\n".join(lines)
        )


def sql_root() -> Path:
    root = repository_root() / "sql"
    if not root.is_dir():
        raise DatabaseError(
            f"the SQL files were expected under {root}. They live in the repository, next to "
            "pyproject.toml, so the database needs a checkout and not only an installed package"
        )
    return root


def read_sql(relative: str) -> str:
    return (sql_root() / relative).read_text(encoding="utf-8")


def sql_files(directory: str) -> list[str]:
    """The ``.sql`` files of one directory of ``sql/``, in name order, as relative paths."""
    return [f"{directory}/{path.name}" for path in sorted((sql_root() / directory).glob("*.sql"))]


def database_path(name: str) -> Path:
    """``PT_DATA_DIR/available/databases/<name>.duckdb``; the directory is created."""
    directory = data_dir() / "available" / "databases"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{path_identifier('database name', name)}.duckdb"


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """A connection with the schema and the views in place. ``None`` is a database in
    memory. A read-only connection expects them to be there already."""
    connection = duckdb.connect(":memory:" if path is None else str(path), read_only=read_only)
    if not read_only:
        for relative in (*sql_files("schema"), *sql_files("views")):
            connection.execute(read_sql(relative))
    return connection


def _rows(connection: duckdb.DuckDBPyConnection, sql: str, parameters: object = None) -> list[dict]:
    result = connection.execute(sql) if parameters is None else connection.execute(sql, parameters)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def quality_report(
    connection: duckdb.DuckDBPyConnection, schema_name: str = "main"
) -> dict[str, list[dict[str, object]]]:
    """Run every query of ``sql/quality`` against ``main`` or ``staging`` and return what
    each found. An empty list means that the check passed."""
    if schema_name not in ("main", "staging"):
        raise ValueError(f"schema_name must be 'main' or 'staging', got {schema_name!r}")
    report: dict[str, list[dict[str, object]]] = {}
    connection.execute(f"USE {schema_name}")
    try:
        for relative in sql_files("quality"):
            report[Path(relative).stem] = _rows(connection, read_sql(relative))
    finally:
        connection.execute("USE main")
    return report


def failed_checks(
    report: Mapping[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    return {name: rows for name, rows in report.items() if rows}


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def _clear_staging(connection: duckdb.DuckDBPyConnection) -> None:
    for table in TABLES:
        connection.execute(f"DELETE FROM staging.{table}")


def _stage(connection: duckdb.DuckDBPyConnection, dataset: Dataset, run_id: str) -> str:
    """Load the rows of one run, and of its plant, into the empty staging schema. Every
    file is named explicitly, from the manifest; columns are listed, never ``*``."""
    _clear_staging(connection)
    runs = [run for run in dataset.manifest["runs"] if run["run_id"] == run_id]  # type: ignore[union-attr]
    if len(runs) != 1:
        raise DatabaseError(f"the data set {dataset.dataset_id!r} has no run {run_id!r}")
    plant_id = str(runs[0]["plant_id"])

    def columns(name: str) -> str:
        return ", ".join(schema.SCHEMAS[name].names)

    def path(name: str) -> str:
        return str(dataset.directory / dataset.manifest["tables"][name]["file"])  # type: ignore[index]

    for name, key, value in (
        ("plants", "plant_id", plant_id),
        ("process_parameters", "plant_id", plant_id),
        ("sensors", "plant_id", plant_id),
        ("operating_runs", "run_id", run_id),
    ):
        connection.execute(
            f"INSERT INTO staging.{name} ({columns(name)}) "
            f"SELECT {columns(name)} FROM read_parquet(?) WHERE {key} = ?",
            [path(name), value],
        )
    connection.execute(
        f"INSERT INTO staging.measurements ({columns('measurements')}) "
        f"SELECT {columns('measurements')} FROM read_parquet(?)",
        [str(dataset.measurement_path(run_id))],
    )
    return plant_id


def _differs(connection: duckdb.DuckDBPyConnection, table: str, where: str, value: str) -> bool:
    """Whether the staged rows and the rows already in ``main`` disagree, in either
    direction, for the rows selected by ``where``. NULLs compare as equal, as they should
    for a null noise level."""
    for first, second in (("staging", "main"), ("main", "staging")):
        found = connection.execute(
            f"SELECT count(*) FROM (SELECT * FROM {first}.{table} WHERE {where} "
            f"EXCEPT SELECT * FROM {second}.{table} WHERE {where})",
            [value, value],
        ).fetchone()
        if found[0]:
            return True
    return False


def _insert_run(connection: duckdb.DuckDBPyConnection, plant_is_new: bool) -> None:
    """From staging to main, parents before children. A separate function so that a test
    can make it fail half way and see that nothing is left behind."""
    if plant_is_new:
        for table in ("plants", "process_parameters", "sensors"):
            connection.execute(f"INSERT INTO main.{table} SELECT * FROM staging.{table}")
    connection.execute("INSERT INTO main.operating_runs SELECT * FROM staging.operating_runs")
    connection.execute("INSERT INTO main.measurements SELECT * FROM staging.measurements")


def ingest_run(connection: duckdb.DuckDBPyConnection, dataset: Dataset, run_id: str) -> str:
    """Ingest one run of a published data set, atomically. Returns ``"ingested"``, or
    ``"already_present"`` when the same run with the same content was there."""
    connection.execute("BEGIN TRANSACTION")
    try:
        status = _ingest_within_transaction(connection, dataset, run_id)
    except BaseException:
        # not to hide the failure, which is raised again, but to leave nothing of the run
        connection.execute("ROLLBACK")
        connection.execute("USE main")
        raise
    if status == "ingested":
        connection.execute("COMMIT")
    else:
        connection.execute("ROLLBACK")  # nothing to keep: staging goes back to empty
    return status


def _ingest_within_transaction(
    connection: duckdb.DuckDBPyConnection, dataset: Dataset, run_id: str
) -> str:
    plant_id = _stage(connection, dataset, run_id)

    findings = failed_checks(quality_report(connection, "staging"))
    if findings:
        raise IngestionQualityError(run_id, findings)

    recorded = connection.execute("SELECT content_sha256 FROM staging.operating_runs").fetchall()
    rebuilt = observations_from_database(connection, run_id, "staging")
    if len(recorded) != 1 or rebuilt.content_digest() != recorded[0][0]:
        raise IngestionConflictError(
            f"the staged rows of {run_id!r} do not have the content recorded for the run; the "
            "files changed after the data set was verified"
        )

    known_plant = connection.execute(
        "SELECT count(*) FROM main.plants WHERE plant_id = ?", [plant_id]
    ).fetchone()[0]
    if known_plant:
        for table in ("plants", "process_parameters", "sensors"):
            if _differs(connection, table, "plant_id = ?", plant_id):
                raise IngestionConflictError(
                    f"{table} of plant {plant_id!r} in the database differ from those of the data "
                    f"set {dataset.dataset_id!r}. Nothing was changed"
                )

    known_run = connection.execute(
        "SELECT count(*) FROM main.operating_runs WHERE run_id = ?", [run_id]
    ).fetchone()[0]
    expected = connection.execute("SELECT count(*) FROM staging.measurements").fetchone()[0]
    if known_run:
        owner = connection.execute(
            "SELECT dataset_id FROM main.operating_runs WHERE run_id = ?", [run_id]
        ).fetchone()[0]
        if owner != dataset.dataset_id:
            # a run belongs to one data set; its record says which, and would lie otherwise
            raise IngestionConflictError(
                f"the run {run_id!r} is already in the database as part of the data set "
                f"{owner!r}, and cannot also come from {dataset.dataset_id!r}. Nothing was changed"
            )
        if _differs(connection, "operating_runs", "run_id = ?", run_id) or _differs(
            connection, "measurements", "run_id = ?", run_id
        ):
            raise IngestionConflictError(
                f"the run {run_id!r} is already in the database with different content. "
                "Nothing was changed"
            )
        return "already_present"

    _insert_run(connection, plant_is_new=not known_plant)
    stored = connection.execute(
        "SELECT count(*) FROM main.measurements WHERE run_id = ?", [run_id]
    ).fetchone()[0]
    if stored != expected:
        raise DatabaseError(f"{stored} rows of {run_id!r} were stored where {expected} were staged")
    _clear_staging(connection)  # within the same transaction: staging is empty between runs
    return "ingested"


def ingest_dataset(connection: duckdb.DuckDBPyConnection, dataset: Dataset) -> dict[str, str]:
    """Ingest every run of a data set, each in its own transaction, in the order of the
    manifest. The status of each run is returned; the first failure stops the rest."""
    return {run_id: ingest_run(connection, dataset, run_id) for run_id in dataset.run_ids}


# --------------------------------------------------------------------------- #
# Reading back
# --------------------------------------------------------------------------- #


def _long_table(connection: duckdb.DuckDBPyConnection, run_id: str, schema_name: str) -> pa.Table:
    names = schema.MEASUREMENTS.names
    found = connection.execute(
        f"SELECT {', '.join(names)} FROM {schema_name}.measurements WHERE run_id = ?", [run_id]
    ).fetchnumpy()
    arrays = [
        pa.array(
            np.asarray(found[name]).tolist() if field.type == pa.string() else found[name],
            field.type,
        )
        for name, field in zip(names, schema.MEASUREMENTS, strict=True)
    ]
    return pa.Table.from_arrays(arrays, schema=schema.MEASUREMENTS)


def observations_from_database(
    connection: duckdb.DuckDBPyConnection, run_id: str, schema_name: str = "main"
) -> Observations:
    """The observations of a run rebuilt from the database, by the same function that
    rebuilds them from Parquet. A database gives no order of its own: the order of the
    channels comes from ``channel_index`` and that of the rows from ``sample_index``."""
    if schema_name not in ("main", "staging"):
        raise ValueError(f"schema_name must be 'main' or 'staging', got {schema_name!r}")
    runs = _rows(
        connection, f"SELECT * FROM {schema_name}.operating_runs WHERE run_id = ?", [run_id]
    )
    if len(runs) != 1:
        raise DatabaseError(f"the database has {len(runs)} runs called {run_id!r}")
    sensors = _rows(
        connection,
        f"SELECT * FROM {schema_name}.sensors WHERE plant_id = ?",
        [runs[0]["plant_id"]],
    )
    return observations_from_long(runs[0], sensors, _long_table(connection, run_id, schema_name))


def aligned_series(connection: duckdb.DuckDBPyConnection, run_id: str) -> list[dict[str, object]]:
    return _rows(
        connection, "SELECT * FROM aligned_series WHERE run_id = ? ORDER BY sample_index", [run_id]
    )


def compare_plants(
    connection: duckdb.DuckDBPyConnection, source: str, target: str
) -> list[dict[str, object]]:
    return _rows(
        connection,
        read_sql("analysis/source_target_comparison.sql"),
        {"source": source, "target": target},
    )


def block_average(
    connection: duckdb.DuckDBPyConnection, block_ticks: int
) -> list[dict[str, object]]:
    """The six-second series averaged over blocks of ``block_ticks`` ticks, for every run.
    Edges, label, coverage and the treatment of inputs are described in the SQL file."""
    if isinstance(block_ticks, (bool, np.bool_)) or not isinstance(block_ticks, (int, np.integer)):
        raise ValueError(f"block_ticks must be a whole number of ticks, got {block_ticks!r}")
    if block_ticks < 1:
        raise ValueError(f"block_ticks must be 1 or more, got {block_ticks!r}")
    return _rows(
        connection, read_sql("analysis/block_average.sql"), {"block_ticks": int(block_ticks)}
    )
