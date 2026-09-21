"""Aligned series for future models, exported from the database by SQL.

    PT_DATA_DIR/available/exports/<dataset_id>/
        export.json               columns, units, the time convention, the instruments, the
                                  known parameters of each plant, and the runs with their hashes
        <run_id>.parquet          one row per tick: identifiers, sample_index, time_s, the two
                                  readings and the four known inputs, as 64-bit floats

The rows come from the view ``aligned_series`` (``sql/views/aligned_series.sql``), at the
original sampling of six seconds. Nothing is resampled, filled in or filtered: a run whose
aligned rows are not all complete is refused, not trimmed. Every exported run is rebuilt
into an ``Observations`` and must have the content hash recorded when it was written, so
what a model reads is, bit for bit, what the sensors gave.

Only what is in the database is exported, and the database holds only the available
branch: no initial state, no exact state, no parameter of the hidden physics and no seed
of the noise. An export is written once, like a data set: the same content again changes
nothing, other content under the same name is a conflict.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from process_transfer.data import schema
from process_transfer.data.identifiers import path_identifier
from process_transfer.data.paths import data_dir
from process_transfer.measurement.observations import CONTENT_ENCODING, Observations

EXPORT_FORMAT = "process-transfer/aligned-series"
EXPORT_VERSION = 1
EXPORT_MANIFEST = "export.json"
_STAGING_PREFIX = ".staging-"
_IDENTIFIERS = ("plant_id", "run_id")


class ExportError(Exception):
    """An export could not be made, or what is on disk is not the export requested."""


class ExportConflictError(ExportError):
    """An export of that name exists with other content. Nothing was overwritten."""


@dataclass(frozen=True)
class ExportResult:
    directory: Path
    status: str  # "created" or "already_present"
    manifest: dict[str, object]


def exports_root() -> Path:
    return data_dir() / "available" / "exports"


def _records(connection: duckdb.DuckDBPyConnection, sql: str, parameters: list) -> list[dict]:
    result = connection.execute(sql, parameters)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def aligned_table(
    connection: duckdb.DuckDBPyConnection, run: Mapping[str, object], channels: list[dict]
) -> pa.Table:
    """The aligned series of one run as an Arrow table with units on its columns. Every
    tick of the run must be there, once, with its six channels."""
    found = connection.execute(
        "SELECT * FROM aligned_series WHERE run_id = ? ORDER BY sample_index", [run["run_id"]]
    ).fetchnumpy()
    n = int(run["n_samples"])  # type: ignore[arg-type]
    ticks = np.asarray(found["sample_index"])
    if len(ticks) != n or not np.array_equal(ticks, np.arange(n)) or not np.all(found["complete"]):
        raise ExportError(
            f"the aligned series of {run['run_id']!r} is not complete: {len(ticks)} rows for "
            f"{n} samples, {int(np.sum(~np.asarray(found['complete'], dtype=bool)))} of them "
            "incomplete. Nothing is filled in or dropped to make an export"
        )
    fields = [pa.field(name, pa.string(), nullable=False) for name in _IDENTIFIERS]
    fields.append(pa.field("sample_index", pa.int64(), nullable=False))
    fields.append(pa.field("time_s", pa.float64(), nullable=False, metadata={"unit": "s"}))
    arrays: list[pa.Array] = [
        pa.array(np.asarray(found[name]).tolist(), pa.string()) for name in _IDENTIFIERS
    ]
    arrays += [pa.array(ticks, pa.int64()), pa.array(found["time_s"], pa.float64())]
    for channel in channels:
        name = str(channel["variable_name"])
        values = np.asarray(found[name], dtype=np.float64)
        if np.ma.isMaskedArray(found[name]) and np.ma.count_masked(found[name]):
            raise ExportError(f"{run['run_id']!r}: channel {name!r} has no value at some tick")
        metadata = {"unit": str(channel["unit"]), "channel_kind": str(channel["channel_kind"])}
        fields.append(pa.field(name, pa.float64(), nullable=False, metadata=metadata))
        arrays.append(pa.array(values, pa.float64()))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def observations_from_aligned(
    table: pa.Table, run: Mapping[str, object], channels: list[dict]
) -> Observations:
    """The observations that an aligned table holds, to compare with what was stored."""

    def of_kind(kind: str) -> list[dict]:
        return [channel for channel in channels if channel["channel_kind"] == kind]

    def matrix(kind: str) -> np.ndarray:
        columns = [table[str(c["variable_name"])].to_numpy() for c in of_kind(kind)]
        return np.column_stack(columns) if columns else np.empty((table.num_rows, 0))

    measured, inputs = of_kind(schema.CHANNEL_MEASURED), of_kind(schema.CHANNEL_INPUT)
    return Observations(
        plant=str(run["plant_id"]),
        run=str(run["run_id"]),
        times=table["time_s"].to_numpy(),
        measured=matrix(schema.CHANNEL_MEASURED),
        inputs=matrix(schema.CHANNEL_INPUT),
        measured_names=tuple(str(c["variable_name"]) for c in measured),
        measured_units=tuple(str(c["unit"]) for c in measured),
        input_names=tuple(str(c["variable_name"]) for c in inputs),
        input_units=tuple(str(c["unit"]) for c in inputs),
        sample_period=float(run["sampling_period_s"]),  # type: ignore[arg-type]
        noise_std=tuple(float(c["noise_std"]) for c in measured),
    )


def _identity(manifest: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in manifest.items() if key != "attempt"}


def export_dataset(
    connection: duckdb.DuckDBPyConnection, dataset_id: str, attempt: Mapping[str, object]
) -> ExportResult:
    """Export every run of ``dataset_id`` that the database holds."""
    path_identifier("dataset_id", dataset_id)
    runs = _records(
        connection,
        "SELECT * FROM operating_runs WHERE dataset_id = ? ORDER BY run_id",
        [dataset_id],
    )
    if not runs:
        raise ExportError(f"the database holds no run of the data set {dataset_id!r}")

    root = exports_root()
    root.mkdir(parents=True, exist_ok=True)
    final = root / dataset_id
    staging = root / f"{_STAGING_PREFIX}{dataset_id}-{secrets.token_hex(4)}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        manifest = _write(connection, dataset_id, runs, attempt, staging)
        if not final.exists():
            try:
                os.rename(staging, final)
            except OSError:
                if not final.exists():
                    raise
            else:
                return ExportResult(final, "created", manifest)
        present = json.loads((final / EXPORT_MANIFEST).read_text(encoding="utf-8"))
        if _identity(present) != _identity(manifest):
            raise ExportConflictError(
                f"an export of {dataset_id!r} exists at {final} with other content. Nothing "
                "was overwritten"
            )
        return ExportResult(final, "already_present", present)
    finally:
        if staging.is_dir() and staging.name.startswith(_STAGING_PREFIX):
            shutil.rmtree(staging)


def _write(
    connection: duckdb.DuckDBPyConnection,
    dataset_id: str,
    runs: list[dict],
    attempt: Mapping[str, object],
    staging: Path,
) -> dict[str, object]:
    plants: dict[str, dict[str, object]] = {}
    exported: list[dict[str, object]] = []
    for run in runs:
        plant_id = str(run["plant_id"])
        channels = _records(
            connection,
            "SELECT * FROM sensors WHERE plant_id = ? ORDER BY channel_kind DESC, channel_index",
            [plant_id],
        )  # 'measured' before 'input', each in the order of the plant
        table = aligned_table(connection, run, channels)
        rebuilt = observations_from_aligned(table, run, channels)
        if rebuilt.content_digest() != run["content_sha256"]:
            raise ExportError(
                f"the aligned series of {run['run_id']!r} does not have the content recorded "
                "for the run; it is not exported"
            )
        pq.write_table(table, staging / f"{run['run_id']}.parquet")
        exported.append(
            {
                "run_id": run["run_id"],
                "plant_id": plant_id,
                "file": f"{run['run_id']}.parquet",
                "rows": table.num_rows,
                "operating_mode": run["operating_mode"],
                "description": run["description"],
                "start_time_s": run["start_time_s"],
                "end_time_s": run["end_time_s"],
                "sampling_period_s": run["sampling_period_s"],
                "content_sha256": run["content_sha256"],
            }
        )
        if plant_id not in plants:
            (plant,) = _records(connection, "SELECT * FROM plants WHERE plant_id = ?", [plant_id])
            parameters = _records(
                connection,
                "SELECT parameter, value, unit FROM process_parameters WHERE plant_id = ? "
                "ORDER BY parameter",
                [plant_id],
            )
            plants[plant_id] = {
                "name": plant["name"],
                "process_type": plant["process_type"],
                "known_parameters": parameters,
                "channels": [
                    {
                        "column": c["variable_name"],
                        "channel_kind": c["channel_kind"],
                        "unit": c["unit"],
                        "sampling_period_s": c["sampling_period_s"],
                        "noise_model": c["noise_model"],
                        "noise_std": c["noise_std"],
                    }
                    for c in channels
                ],
            }

    manifest: dict[str, object] = {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "dataset_id": dataset_id,
        "content_encoding": CONTENT_ENCODING,
        "time_convention": schema.TIME_CONVENTION,
        "columns": ["plant_id", "run_id", "sample_index", "time_s", "<one per channel>"],
        "what_is_not_here": (
            "no initial or exact state, no parameter of the hidden physics, no measurement "
            "error and no seed of the noise; readings are as the sensors gave them, never "
            "clipped or corrected"
        ),
        "plants": plants,
        "runs": exported,
        "attempt": json.loads(json.dumps(dict(attempt))),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    (staging / EXPORT_MANIFEST).write_text(text, encoding="utf-8")
    return json.loads(text)
