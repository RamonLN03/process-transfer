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
of the noise. An export is written once, like a data set: it is built in a staging
directory, read back and verified, and renamed into place. The same content again changes
nothing; other content under the same name is a conflict.

Reading. ``open_export_directory`` verifies an export the way ``parquet_store`` verifies a
data set: manifest, format, identifiers, the exact list of files, the schema of every run,
its rows, its ticks and its content hash. An export that is already on disk is examined
this way before it is reported as present: its manifest is not taken at its word.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
from collections.abc import Mapping, Sequence
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
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExportError(Exception):
    """An export could not be made, or what is on disk is not the export requested."""


class ExportConflictError(ExportError):
    """An export of that name exists with other content. Nothing was overwritten."""


class ExportIntegrityError(ExportError):
    """An export on disk does not match its manifest or the format."""


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


def run_schema(channels: Sequence[Mapping[str, object]]) -> pa.Schema:
    """The columns of an exported run: the identifiers, the tick, the instant and one
    column per channel of its plant in the order given, each with its unit and kind on
    the field. Written and verified with the same schema."""
    fields = [pa.field(name, pa.string(), nullable=False) for name in _IDENTIFIERS]
    fields.append(pa.field("sample_index", pa.int64(), nullable=False))
    fields.append(pa.field("time_s", pa.float64(), nullable=False, metadata={"unit": "s"}))
    for channel in channels:
        metadata = {"unit": str(channel["unit"]), "channel_kind": str(channel["channel_kind"])}
        name = str(channel["variable_name"])
        fields.append(pa.field(name, pa.float64(), nullable=False, metadata=metadata))
    return pa.schema(fields)


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
    arrays: list[pa.Array] = [
        pa.array(np.asarray(found[name]).tolist(), pa.string()) for name in _IDENTIFIERS
    ]
    arrays += [pa.array(ticks, pa.int64()), pa.array(found["time_s"], pa.float64())]
    for channel in channels:
        name = str(channel["variable_name"])
        values = np.asarray(found[name], dtype=np.float64)
        if np.ma.isMaskedArray(found[name]) and np.ma.count_masked(found[name]):
            raise ExportError(f"{run['run_id']!r}: channel {name!r} has no value at some tick")
        arrays.append(pa.array(values, pa.float64()))
    return pa.Table.from_arrays(arrays, schema=run_schema(channels))


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
        open_export_directory(staging)  # read back and verify before anyone can see it
        if not final.exists():
            try:
                os.rename(staging, final)
            except OSError:
                # another writer published the same name between the test and the rename;
                # what is there now is examined below, never replaced
                if not final.exists():
                    raise
            else:
                return ExportResult(final, "created", manifest)
        try:
            present = open_export_directory(final)
        except ExportIntegrityError as error:
            raise ExportIntegrityError(
                f"an export of {dataset_id!r} exists at {final} and is not intact. Nothing was "
                f"overwritten. {error}"
            ) from None
        if _identity(present.manifest) != _identity(manifest):
            raise ExportConflictError(
                f"an export of {dataset_id!r} exists at {final} with other content. Nothing "
                "was overwritten"
            )
        return ExportResult(final, "already_present", dict(present.manifest))
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


# --------------------------------------------------------------------------- #
# Reading an export back, verified
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Export:
    """A published export, verified when it was opened."""

    directory: Path
    manifest: Mapping[str, object]

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(str(run["run_id"]) for run in self.manifest["runs"])  # type: ignore[union-attr]

    def run(self, run_id: str) -> dict[str, object]:
        """The entry of a run in the manifest, with ``n_samples`` for the readers that
        expect a row of ``operating_runs``."""
        for run in self.manifest["runs"]:  # type: ignore[union-attr]
            if run["run_id"] == run_id:
                return {**run, "n_samples": run["rows"]}
        raise KeyError(f"the export {self.dataset_id!r} has no run {run_id!r}")

    def channels(self, plant_id: str) -> list[dict[str, object]]:
        """The channels of a plant, named as ``aligned_table`` names them."""
        plants = self.manifest["plants"]  # type: ignore[index]
        if plant_id not in plants:  # type: ignore[operator]
            raise KeyError(f"the export {self.dataset_id!r} has no plant {plant_id!r}")
        return [
            {**channel, "variable_name": channel["column"]}
            for channel in plants[plant_id]["channels"]  # type: ignore[index]
        ]

    def table(self, run_id: str) -> pa.Table:
        """The exported rows of a run, checked against the manifest: schema, no nulls,
        row count, identifiers, ticks and instants."""
        run = self.run(run_id)
        return _read_run(
            self.directory / str(run["file"]), run, self.channels(str(run["plant_id"]))
        )

    def observations(self, run_id: str) -> Observations:
        """The observations of a run rebuilt from its exported rows and checked against
        the content hash recorded for it."""
        run = self.run(run_id)
        rebuilt = observations_from_aligned(
            self.table(run_id), run, self.channels(str(run["plant_id"]))
        )
        if rebuilt.content_digest() != run["content_sha256"]:
            raise ExportIntegrityError(
                f"the run {run_id!r} does not have the content recorded for it: "
                f"{rebuilt.content_digest()} against {run['content_sha256']}"
            )
        return rebuilt


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportIntegrityError(message)


def _identifier(name: str, value: object) -> str:
    try:
        return path_identifier(name, value)
    except ValueError as error:
        raise ExportIntegrityError(str(error)) from None


def _finite_float(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _verify_manifest(directory: Path, manifest: object) -> None:
    """What the manifest must say before any file it names is opened."""
    _require(isinstance(manifest, dict), f"{directory}: the manifest is not a JSON object")
    assert isinstance(manifest, dict)
    _require(
        manifest.get("format") == EXPORT_FORMAT and manifest.get("version") == EXPORT_VERSION,
        f"{directory} follows {manifest.get('format')!r} version {manifest.get('version')!r}; "
        f"this code reads {EXPORT_FORMAT!r} version {EXPORT_VERSION}",
    )
    _require(
        manifest.get("content_encoding") == CONTENT_ENCODING,
        f"{directory} hashes its content with {manifest.get('content_encoding')!r}, not with "
        f"{CONTENT_ENCODING!r}",
    )
    _identifier("dataset_id", manifest.get("dataset_id"))

    plants = manifest.get("plants")
    _require(isinstance(plants, dict) and bool(plants), f"{directory}: the manifest names no plant")
    assert isinstance(plants, dict)
    for plant_id, plant in plants.items():
        _identifier("plant_id", plant_id)
        channels = plant.get("channels") if isinstance(plant, dict) else None
        _require(
            isinstance(channels, list) and bool(channels),
            f"{directory}: the plant {plant_id!r} has no channels",
        )
        names = [channel.get("column") for channel in channels]
        _require(
            all(isinstance(name, str) and name.strip() for name in names)
            and len(set(names)) == len(names),
            f"{directory}: the channels of {plant_id!r} need distinct, non-empty names: {names}",
        )
        for channel in channels:
            kind, name = channel.get("channel_kind"), channel["column"]
            _require(
                kind in schema.CHANNEL_KINDS,
                f"{directory}: channel {name!r} of {plant_id!r} is of kind {kind!r}, which is "
                f"not one of {schema.CHANNEL_KINDS}",
            )
            _require(
                isinstance(channel.get("unit"), str) and bool(channel["unit"].strip()),
                f"{directory}: channel {name!r} of {plant_id!r} has no unit",
            )
            noise_model, noise_std = channel.get("noise_model"), channel.get("noise_std")
            if kind == schema.CHANNEL_MEASURED:
                _require(
                    noise_model == schema.NOISE_ADDITIVE_GAUSSIAN
                    and _finite_float(noise_std)
                    and noise_std >= 0,
                    f"{directory}: measured channel {name!r} of {plant_id!r} needs a noise model "
                    f"and a noise level of zero or more, got {noise_model!r} and {noise_std!r}",
                )
            else:
                _require(
                    noise_model is None and noise_std is None,
                    f"{directory}: known input {name!r} of {plant_id!r} is presented with noise",
                )

    runs = manifest.get("runs")
    _require(isinstance(runs, list) and bool(runs), f"{directory}: the manifest lists no run")
    assert isinstance(runs, list)
    for run in runs:
        _require(isinstance(run, dict), f"{directory}: a run entry is not a JSON object")
        run_id = _identifier("run_id", run.get("run_id"))
        plant_id = _identifier("plant_id", run.get("plant_id"))
        _require(
            plant_id in plants,
            f"{directory}: run {run_id!r} is of plant {plant_id!r}, which the manifest does "
            "not describe",
        )
        _require(
            run.get("file") == f"{run_id}.parquet",
            f"{directory}: run {run_id!r} is listed with the file {run.get('file')!r}, not "
            f"with {run_id + '.parquet'!r}",
        )
        _require(
            isinstance(run.get("rows"), int)
            and not isinstance(run["rows"], bool)
            and run["rows"] >= 1,
            f"{directory}: run {run_id!r} is listed with {run.get('rows')!r} rows",
        )
        _require(
            isinstance(run.get("content_sha256"), str)
            and _SHA256.fullmatch(run["content_sha256"]) is not None,
            f"{directory}: run {run_id!r} has no SHA-256 content hash",
        )
        _require(
            all(
                _finite_float(run.get(key))
                for key in ("start_time_s", "end_time_s", "sampling_period_s")
            )
            and run["sampling_period_s"] > 0
            and run["end_time_s"] >= run["start_time_s"],
            f"{directory}: run {run_id!r} has an impossible extent or sampling period",
        )
    run_ids = [run["run_id"] for run in runs]
    _require(len(set(run_ids)) == len(run_ids), f"{directory}: a run is listed twice: {run_ids}")


def _read_run(path: Path, run: Mapping[str, object], channels: list[dict[str, object]]) -> pa.Table:
    _require(path.is_file(), f"{path} is listed in the manifest and is not there")
    table = pq.read_table(path)
    expected = run_schema(channels)
    _require(
        table.schema.equals(expected, check_metadata=True),
        f"{path.name} does not have the schema of its run:\n{table.schema}\nagainst\n{expected}",
    )
    for name in table.schema.names:
        _require(
            table[name].null_count == 0,
            f"{path.name}: {table[name].null_count} of {table.num_rows} rows have no {name}",
        )
    n = int(run["rows"])  # type: ignore[arg-type]
    _require(table.num_rows == n, f"{path.name} has {table.num_rows} rows where {n} are listed")
    for column in _IDENTIFIERS:
        found = set(table[column].to_pylist())
        _require(
            found == {run[column]},
            f"{path.name}: the column {column} holds {sorted(found)}, not only {run[column]!r}",
        )
    _require(
        np.array_equal(table["sample_index"].to_numpy(), np.arange(n)),
        f"{path.name}: sample_index does not run from 0 to {n - 1} in order",
    )
    times = table["time_s"].to_numpy()
    _require(
        times[0] == run["start_time_s"] and times[-1] == run["end_time_s"],
        f"{path.name}: the rows run from {times[0]!r} to {times[-1]!r} s, and the manifest says "
        f"from {run['start_time_s']!r} to {run['end_time_s']!r} s",
    )
    return table


def open_export_directory(directory: Path) -> Export:
    """Open and verify the export in ``directory``: manifest, format, identifiers, the
    exact list of files, and every run against its schema, rows and content hash. Only
    that directory is read."""
    directory = Path(directory)
    manifest_path = directory / EXPORT_MANIFEST
    _require(
        manifest_path.is_file(),
        f"{directory} has no {EXPORT_MANIFEST}: it is not an export, or its writing was "
        "interrupted",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_manifest(directory, manifest)

    listed = {EXPORT_MANIFEST} | {run["file"] for run in manifest["runs"]}
    # The directory is listed only to notice a file that the manifest does not know; data
    # are always read from the list above. The listing cannot leave this directory.
    found = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    _require(
        found == listed,
        f"{directory} does not hold exactly the files of its manifest: missing "
        f"{sorted(listed - found)}, unexpected {sorted(found - listed)}",
    )
    export = Export(directory, manifest)
    for run_id in export.run_ids:
        export.observations(run_id)  # reads the run, checks it and its content hash
    return export
