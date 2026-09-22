"""Immutable Parquet data sets under ``PT_DATA_DIR/available/datasets``.

Layout of a data set, all of it available to a model:

    <dataset_id>/
        manifest.json                    written last; without it there is no data set
        plants.parquet
        process_parameters.parquet
        sensors.parquet
        operating_runs.parquet
        measurements/<run_id>.parquet    long format, one file per run

Writing. A data set is built in a staging directory next to its final place, read back
and verified, and only then renamed into place. A rename of a directory on one volume is
atomic, so an interrupted write leaves a ``.staging-...`` directory, which no reader
takes for a data set, and never a half-written one under the final name. This is not a
transaction manager: it protects against an interrupted process, not against a power
cut in the middle of a rename, and it does not coordinate several machines.

A published data set is never modified. Writing the same identity again is accepted when
the content is the same, and changes nothing; with different content it is a conflict.

Reading. Files are taken from the list in the manifest, never from a wildcard, so
nothing outside the data set can be picked up, the private branch least of all. Reading
needs nothing but this directory. Schemas, row counts and content hashes are verified,
and so are the identifiers, the relations between the tables and the quality flag: what
the writer refuses is refused on reading too, whatever wrote the files.
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

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from process_transfer.data import schema
from process_transfer.data.identifiers import path_identifier
from process_transfer.data.paths import data_dir
from process_transfer.data.records import (
    PlantRecord,
    RunRecord,
    measurement_table,
    require_si_unit,
    run_row,
    sensor_id,
    sensor_rows,
    table_from_rows,
    validate_observations,
)
from process_transfer.measurement.observations import CONTENT_ENCODING, Observations

MANIFEST = "manifest.json"
CONTRACT = "process-transfer/dataset"
_STAGING_PREFIX = ".staging-"
SMALL_TABLES = ("plants", "process_parameters", "sensors", "operating_runs")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DatasetError(Exception):
    """Something is wrong with a data set on disk or with a request to write one."""


class DatasetConflictError(DatasetError):
    """The same identity with different content. Nothing was overwritten."""


class DatasetIntegrityError(DatasetError):
    """A data set on disk does not match its manifest or the contract."""


def datasets_root() -> Path:
    """``PT_DATA_DIR/available/datasets``. It is not created here."""
    return data_dir() / "available" / "datasets"


def dataset_directory(dataset_id: str) -> Path:
    return datasets_root() / path_identifier("dataset_id", dataset_id)


def _measurement_file(run_id: str) -> str:
    return f"measurements/{run_id}.parquet"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetIntegrityError(message)


def _identifier(name: str, value: object) -> str:
    try:
        return path_identifier(name, value)
    except ValueError as error:
        raise DatasetIntegrityError(str(error)) from None


def _identity(manifest: Mapping[str, object]) -> dict[str, object]:
    """The part of a manifest that says what the data are. The attempt, which says when
    and with which code they were written, is left out: two attempts at the same data
    set are expected to agree on everything else."""
    return {
        key: manifest[key]
        for key in ("contract", "schema_version", "encodings", "dataset_id", "tables", "runs")
    }


@dataclass(frozen=True)
class PublishResult:
    directory: Path
    status: str  # "created", or "already_present" when an identical data set was there
    manifest: dict[str, object]


class DatasetWriter:
    """Collects the runs of a data set, one at a time, and publishes them together.

    Runs are written to the staging directory as they are added, so only one run needs
    to be in memory at a time. Use as a context manager: leaving it without ``publish``
    removes the staging directory.
    """

    def __init__(
        self, dataset_id: str, plants: Sequence[PlantRecord], attempt: Mapping[str, object]
    ) -> None:
        self.dataset_id = path_identifier("dataset_id", dataset_id)
        self.plants = {plant.plant_id: plant for plant in plants}
        if len(self.plants) != len(plants) or len(plants) == 0:
            raise ValueError("a data set needs at least one plant, each given once")
        try:
            self.attempt = json.loads(json.dumps(dict(attempt)))  # plain and a copy
        except TypeError as error:
            raise ValueError(f"attempt must be serialisable as JSON: {error}") from None
        self._sensors: dict[str, list[dict[str, object]]] = {}
        self._runs: list[dict[str, object]] = []
        self._manifest_runs: list[dict[str, object]] = []
        self._closed = False

        root = datasets_root()
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self.final = root / self.dataset_id
        self.staging = root / f"{_STAGING_PREFIX}{self.dataset_id}-{secrets.token_hex(4)}"
        (self.staging / "measurements").mkdir(parents=True, exist_ok=False)

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, *exception: object) -> None:
        self.abandon()

    def add_run(self, observations: Observations, record: RunRecord) -> None:
        """Write one run to the staging directory. It needs the observations and a few
        words about the run, and nothing of the truth."""
        if self._closed:
            raise DatasetError("this writer has already published or abandoned its data set")
        validate_observations(observations)
        if not isinstance(record, RunRecord):
            raise TypeError(f"expected a RunRecord, got {type(record).__name__}")
        if record.dataset_id != self.dataset_id:
            raise ValueError(
                f"the run says it belongs to {record.dataset_id!r}, not to {self.dataset_id!r}"
            )
        if observations.plant not in self.plants:
            raise ValueError(
                f"the run {observations.run!r} is of plant {observations.plant!r}, which is not "
                f"among the plants of this data set: {sorted(self.plants)}"
            )
        if any(row["run_id"] == observations.run for row in self._runs):
            raise DatasetConflictError(f"the run {observations.run!r} was already added")

        channels = sensor_rows(observations)
        known = self._sensors.setdefault(observations.plant, channels)
        if known != channels:
            raise DatasetConflictError(
                f"the channels of plant {observations.plant!r} in run {observations.run!r} "
                "differ from those of an earlier run: names, order, units, sampling period "
                "and noise level belong to the plant and must agree across its runs"
            )

        table = measurement_table(observations)
        pq.write_table(table, self.staging / _measurement_file(observations.run))
        row = run_row(observations, record)
        self._runs.append(row)
        self._manifest_runs.append(
            {
                "run_id": observations.run,
                "plant_id": observations.plant,
                "file": _measurement_file(observations.run),
                "rows": table.num_rows,
                "n_samples": observations.n_samples,
                "content_sha256": row["content_sha256"],
            }
        )

    def publish(self) -> PublishResult:
        """Write the small tables and the manifest, verify the whole by reading it back,
        and move it into place. Returns whether it was created or was already there."""
        if self._closed:
            raise DatasetError("this writer has already published or abandoned its data set")
        if not self._runs:
            raise DatasetError("a data set without runs is not published")
        idle = sorted(set(self.plants) - set(self._sensors))
        if idle:
            raise DatasetError(f"no run was added for the plants {idle}")
        units: dict[str, set[str]] = {}
        for rows in self._sensors.values():
            for row in rows:
                units.setdefault(str(row["variable_name"]), set()).add(str(row["unit"]))
        mixed = {name: sorted(found) for name, found in units.items() if len(found) > 1}
        if mixed:
            # one variable, one unit: otherwise the plants of a data set cannot be compared
            raise DatasetError(f"a variable has different units on different plants: {mixed}")

        tables = {
            "plants": table_from_rows(
                "plants",
                [
                    {"plant_id": p.plant_id, "name": p.name, "process_type": p.process_type}
                    for p in self.plants.values()
                ],
            ),
            "process_parameters": table_from_rows(
                "process_parameters",
                [
                    {
                        "plant_id": plant.plant_id,
                        "parameter": parameter.parameter,
                        "value": parameter.value,
                        "unit": parameter.unit,
                    }
                    for plant in self.plants.values()
                    for parameter in plant.parameters
                ],
            ),
            "sensors": table_from_rows(
                "sensors", [row for rows in self._sensors.values() for row in rows]
            ),
            "operating_runs": table_from_rows("operating_runs", self._runs),
        }
        manifest: dict[str, object] = {
            "contract": CONTRACT,
            "schema_version": schema.SCHEMA_VERSION,
            "encodings": {"runs": CONTENT_ENCODING, "tables": schema.TABLE_ENCODING},
            "dataset_id": self.dataset_id,
            "time_convention": schema.TIME_CONVENTION,
            "tables": {},
            "runs": sorted(self._manifest_runs, key=lambda run: run["run_id"]),
            "attempt": self.attempt,
        }
        for name, table in tables.items():
            pq.write_table(table, self.staging / f"{name}.parquet")
            manifest["tables"][name] = {  # type: ignore[index]
                "file": f"{name}.parquet",
                "rows": table.num_rows,
                "content_sha256": schema.table_digest(name, table),
            }
        text = json.dumps(manifest, indent=2, sort_keys=True)
        (self.staging / MANIFEST).write_text(text, encoding="utf-8")

        open_dataset_directory(self.staging)  # read back and verify before anyone can see it

        if not self.final.exists():
            try:
                os.rename(self.staging, self.final)
            except OSError:
                # another writer published the same identity between the test and the
                # rename; what is there now is examined below, never replaced
                if not self.final.exists():
                    raise
            else:
                self._closed = True
                return PublishResult(self.final, "created", manifest)

        present = open_dataset_directory(self.final)
        self.abandon()
        if _identity(present.manifest) != _identity(manifest):
            raise DatasetConflictError(
                f"the data set {self.dataset_id!r} already exists at {self.final} with different "
                "content. Nothing was overwritten. A published data set is never modified: "
                "give the new data another dataset_id"
            )
        return PublishResult(self.final, "already_present", dict(present.manifest))

    def abandon(self) -> None:
        """Remove the staging directory of this writer, if it is still there."""
        self._closed = True
        staging = self.staging
        # only ever a directory that this writer created, under the root it was given
        ours = staging.name.startswith(_STAGING_PREFIX) and staging.parent == self._root
        if ours and staging.is_dir():
            shutil.rmtree(staging)


@dataclass(frozen=True)
class Dataset:
    """A published data set, verified when it was opened."""

    directory: Path
    manifest: Mapping[str, object]

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(str(run["run_id"]) for run in self.manifest["runs"])  # type: ignore[union-attr]

    def table(self, name: str) -> pa.Table:
        if name not in SMALL_TABLES:
            raise KeyError(
                f"{name!r} is not one of {SMALL_TABLES}; runs are read with measurements()"
            )
        entry = self.manifest["tables"][name]  # type: ignore[index]
        return _read(self.directory / entry["file"], name)

    def measurement_path(self, run_id: str) -> Path:
        for run in self.manifest["runs"]:  # type: ignore[union-attr]
            if run["run_id"] == run_id:
                return self.directory / run["file"]
        raise KeyError(f"the data set {self.dataset_id!r} has no run {run_id!r}")

    def measurements(self, run_id: str) -> pa.Table:
        return _read(self.measurement_path(run_id), "measurements")

    def observations(self, run_id: str) -> Observations:
        """The observations of a run rebuilt from the long table, with the order of its
        channels, and checked against the content hash recorded when it was written."""
        matching = [
            row for row in self.table("operating_runs").to_pylist() if row["run_id"] == run_id
        ]
        if len(matching) != 1:
            raise KeyError(f"the data set {self.dataset_id!r} has no run {run_id!r}")
        run = matching[0]
        sensors = [
            row for row in self.table("sensors").to_pylist() if row["plant_id"] == run["plant_id"]
        ]
        rebuilt = observations_from_long(run, sensors, self.measurements(run_id))
        if rebuilt.content_digest() != run["content_sha256"]:
            raise DatasetIntegrityError(
                f"the run {run_id!r} does not have the content recorded for it: "
                f"{rebuilt.content_digest()} against {run['content_sha256']}"
            )
        return rebuilt


def observations_from_long(
    run: Mapping[str, object], sensors: Sequence[Mapping[str, object]], measurements: pa.Table
) -> Observations:
    """An ``Observations`` from one row of ``operating_runs``, the channels of its plant
    and its rows of ``measurements``. Used for Parquet and for DuckDB alike. Every row
    must name the run and its plant, use a channel of that plant and carry the quality
    flag of the contract; every channel must have the same ticks and the same instants;
    nothing is filled in. What is rebuilt passes the validation of the writer."""
    _verify_rows(run, sensors, measurements)
    n = int(run["n_samples"])  # type: ignore[arg-type]
    columns: dict[str, list[np.ndarray]] = {schema.CHANNEL_MEASURED: [], schema.CHANNEL_INPUT: []}
    reference: tuple[np.ndarray, np.ndarray] | None = None
    ordered = sorted(sensors, key=lambda row: (row["channel_kind"], row["channel_index"]))
    for channel in ordered:
        rows = measurements.filter(pc.equal(measurements["sensor_id"], channel["sensor_id"]))
        order = pc.sort_indices(rows, sort_keys=[("sample_index", "ascending")])
        rows = rows.take(order)
        ticks = rows["sample_index"].to_numpy()
        times = rows["time_s"].to_numpy()
        if len(ticks) != n or not np.array_equal(ticks, np.arange(n)):
            raise DatasetIntegrityError(
                f"channel {channel['sensor_id']!r} of run {run['run_id']!r} has {len(ticks)} "
                f"rows where {n} consecutive ticks were recorded"
            )
        if reference is None:
            reference = (ticks, times)
        elif not np.array_equal(times, reference[1]):
            raise DatasetIntegrityError(
                f"channel {channel['sensor_id']!r} of run {run['run_id']!r} does not have the "
                "instants of the other channels"
            )
        columns[str(channel["channel_kind"])].append(rows["value"].to_numpy())
    if reference is None or len(measurements) != n * len(ordered):
        raise DatasetIntegrityError(
            f"run {run['run_id']!r} has {len(measurements)} rows, not {n} for each of its "
            f"{len(ordered)} channels"
        )

    def of_kind(kind: str, field: str) -> tuple:
        return tuple(row[field] for row in ordered if row["channel_kind"] == kind)

    def matrix(kind: str) -> np.ndarray:
        return np.column_stack(columns[kind]) if columns[kind] else np.empty((n, 0))

    rebuilt = Observations(
        plant=str(run["plant_id"]),
        run=str(run["run_id"]),
        times=reference[1],
        measured=matrix(schema.CHANNEL_MEASURED),
        inputs=matrix(schema.CHANNEL_INPUT),
        measured_names=of_kind(schema.CHANNEL_MEASURED, "variable_name"),
        measured_units=of_kind(schema.CHANNEL_MEASURED, "unit"),
        input_names=of_kind(schema.CHANNEL_INPUT, "variable_name"),
        input_units=of_kind(schema.CHANNEL_INPUT, "unit"),
        sample_period=float(run["sampling_period_s"]),  # type: ignore[arg-type]
        noise_std=of_kind(schema.CHANNEL_MEASURED, "noise_std"),
    )
    try:
        validate_observations(rebuilt)  # identifiers, names, SI units, the sensor clock
    except ValueError as error:
        raise DatasetIntegrityError(f"run {run['run_id']!r}: {error}") from None
    _require(
        float(run["start_time_s"]) == rebuilt.times[0]  # type: ignore[arg-type]
        and float(run["end_time_s"]) == rebuilt.times[-1],  # type: ignore[arg-type]
        f"run {run['run_id']!r} runs from {rebuilt.times[0]!r} to {rebuilt.times[-1]!r} s, and "
        f"operating_runs says from {run['start_time_s']!r} to {run['end_time_s']!r} s",
    )
    return rebuilt


def _verify_rows(
    run: Mapping[str, object], sensors: Sequence[Mapping[str, object]], measurements: pa.Table
) -> None:
    """What every row of a run must say, checked on reading whatever was written: the run
    and its plant, a channel of that plant, and the quality flag of the contract."""
    run_id = run["run_id"]
    _require(bool(sensors), f"the plant {run['plant_id']!r} of run {run_id!r} has no channels")
    for column, label in (("run_id", "run"), ("plant_id", "plant")):
        found = set(pc.unique(measurements[column]).to_pylist())
        _require(
            found <= {run[column]},
            f"rows of run {run_id!r} name the {label} {sorted(found - {run[column]})}, not "
            f"{run[column]!r}",
        )
    known = {channel["sensor_id"] for channel in sensors}
    found = set(pc.unique(measurements["sensor_id"]).to_pylist())
    _require(
        found <= known,
        f"rows of run {run_id!r} use the channels {sorted(found - known)}, which its plant "
        f"{run['plant_id']!r} does not have",
    )
    flags = measurements["quality_flag"].to_numpy()
    other = flags[flags != schema.QUALITY_GOOD]
    _require(
        other.size == 0,
        f"{other.size} rows of run {run_id!r} carry the quality flags "
        f"{np.unique(other).tolist()}; contract version {schema.SCHEMA_VERSION} defines only "
        f"{schema.QUALITY_GOOD}, and the writer stores nothing else",
    )
    periods = {float(channel["sampling_period_s"]) for channel in sensors}  # type: ignore[arg-type]
    _require(
        periods == {float(run["sampling_period_s"])},  # type: ignore[arg-type]
        f"run {run_id!r} is sampled every {run['sampling_period_s']!r} s and its channels "
        f"every {sorted(periods)} s",
    )


def _read(path: Path, name: str) -> pa.Table:
    if not path.is_file():
        raise DatasetIntegrityError(f"{path} is listed in the manifest and is not there")
    table = pq.read_table(path)
    expected = schema.SCHEMAS[name]
    if not table.schema.equals(expected, check_metadata=False):
        raise DatasetIntegrityError(
            f"{path.name} does not have the schema of {name}:\n{table.schema}\nagainst\n{expected}"
        )
    try:
        schema.require_no_nulls(name, table)  # Arrow does not enforce it; see schema.py
    except ValueError as error:
        raise DatasetIntegrityError(f"{path.name}: {error}") from None
    return table


def open_dataset(dataset_id: str) -> Dataset:
    """The published data set ``dataset_id`` under ``PT_DATA_DIR``, verified, and the one
    its manifest says it is: a data set renamed or copied under another name is refused."""
    dataset = open_dataset_directory(dataset_directory(dataset_id))
    _require(
        dataset.dataset_id == dataset_id,
        f"{dataset.directory} holds the data set {dataset.dataset_id!r}, not {dataset_id!r}",
    )
    return dataset


def open_dataset_directory(directory: Path) -> Dataset:
    """Open and verify the data set in ``directory``: manifest, identifiers, list of
    files, schemas, row counts, the relations between the tables, the quality flag and
    the content hashes. Only that directory is read."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise DatasetIntegrityError(
            f"{directory} has no {MANIFEST}: it is not a data set, or its writing was interrupted"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != CONTRACT
        or manifest.get("schema_version") != schema.SCHEMA_VERSION
    ):
        raise DatasetIntegrityError(
            f"{directory} follows {manifest.get('contract')!r} version "
            f"{manifest.get('schema_version')!r}; this code reads {CONTRACT!r} version "
            f"{schema.SCHEMA_VERSION}"
        )
    expected_encodings = {"runs": CONTENT_ENCODING, "tables": schema.TABLE_ENCODING}
    if manifest.get("encodings") != expected_encodings:
        raise DatasetIntegrityError(
            f"{directory} hashes its content with {manifest.get('encodings')!r}, not with "
            f"{expected_encodings!r}"
        )

    dataset_id = _identifier("dataset_id", manifest.get("dataset_id"))
    _require(
        isinstance(manifest.get("tables"), dict) and set(manifest["tables"]) == set(SMALL_TABLES),
        f"{directory}: the manifest lists the tables {sorted(manifest.get('tables') or [])}, "
        f"not {sorted(SMALL_TABLES)}",
    )
    _require(
        isinstance(manifest.get("runs"), list) and bool(manifest["runs"]),
        f"{directory}: the manifest lists no run",
    )
    for run in manifest["runs"]:
        run_id = _identifier("run_id", run.get("run_id"))
        _identifier("plant_id", run.get("plant_id"))
        _require(
            run.get("file") == _measurement_file(run_id),
            f"{directory}: run {run_id!r} is listed with the file {run.get('file')!r}, not "
            f"with {_measurement_file(run_id)!r}",
        )
    run_ids = [run["run_id"] for run in manifest["runs"]]
    _require(len(set(run_ids)) == len(run_ids), f"{directory}: a run is listed twice: {run_ids}")

    listed = {MANIFEST} | {entry["file"] for entry in manifest["tables"].values()}
    listed |= {run["file"] for run in manifest["runs"]}
    # The directory is listed only to notice a file that the manifest does not know; data
    # are always read from the list above. The listing cannot leave this directory.
    found = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    if found != listed:
        raise DatasetIntegrityError(
            f"{directory} does not hold exactly the files of its manifest: missing "
            f"{sorted(listed - found)}, unexpected {sorted(found - listed)}"
        )

    dataset = Dataset(directory, manifest)
    for name in SMALL_TABLES:
        entry, table = manifest["tables"][name], dataset.table(name)
        if (
            table.num_rows != entry["rows"]
            or schema.table_digest(name, table) != entry["content_sha256"]
        ):
            raise DatasetIntegrityError(f"{name} in {directory} is not the table of the manifest")
    recorded = {row["run_id"]: row for row in dataset.table("operating_runs").to_pylist()}
    if sorted(recorded) != sorted(dataset.run_ids):
        raise DatasetIntegrityError(
            f"operating_runs lists {sorted(recorded)} and the manifest {sorted(dataset.run_ids)}"
        )
    _verify_small_tables(dataset, dataset_id)
    for run in manifest["runs"]:
        row = recorded[run["run_id"]]
        if any(row[key] != run[key] for key in ("plant_id", "content_sha256", "n_samples")):
            raise DatasetIntegrityError(
                f"run {run['run_id']!r}: manifest and operating_runs differ"
            )
        if dataset.measurements(run["run_id"]).num_rows != run["rows"]:
            raise DatasetIntegrityError(
                f"run {run['run_id']!r} does not have its {run['rows']} rows"
            )
        dataset.observations(run["run_id"])  # rebuilds the run and checks its content hash
    return dataset


def _verify_small_tables(dataset: Dataset, dataset_id: str) -> None:
    """The identifiers and the relations of the four small tables, as the contract
    states them and as the writer enforces them: what was refused on writing is refused
    on reading too, whatever wrote the files."""
    where = dataset.directory
    plants = dataset.table("plants").to_pylist()
    plant_ids = [_identifier("plants.plant_id", row["plant_id"]) for row in plants]
    _require(len(set(plant_ids)) == len(plant_ids), f"{where}: a plant is defined twice")
    known = set(plant_ids)

    def of_plant(table: str, row: Mapping[str, object]) -> str:
        _require(
            row["plant_id"] in known,
            f"{where}: {table} names the plant {row['plant_id']!r}, which plants does not have",
        )
        return str(row["plant_id"])

    def si(what: str, unit: object) -> None:
        try:
            require_si_unit(what, unit)
        except ValueError as error:
            raise DatasetIntegrityError(f"{where}: {error}") from None

    def positive(row: Mapping[str, object], *keys: str) -> bool:
        return all(math.isfinite(row[key]) and row[key] > 0 for key in keys)  # type: ignore[arg-type]

    parameters = dataset.table("process_parameters").to_pylist()
    keys = [(of_plant("process_parameters", row), row["parameter"]) for row in parameters]
    _require(len(set(keys)) == len(keys), f"{where}: a parameter is given twice")
    for row in parameters:
        si(f"process_parameters.{row['parameter']}", row["unit"])
        _require(
            math.isfinite(row["value"]),
            f"{where}: {row['plant_id']}.{row['parameter']} is {row['value']!r}",
        )

    sensors = dataset.table("sensors").to_pylist()
    ids = [row["sensor_id"] for row in sensors]
    _require(len(set(ids)) == len(ids), f"{where}: a channel is defined twice")
    units: dict[str, set[str]] = {}
    positions: dict[tuple[str, str], list[int]] = {}
    for row in sensors:
        plant_id = of_plant("sensors", row)
        name, kind, channel = row["variable_name"], row["channel_kind"], row["sensor_id"]
        _require(bool(name.strip()), f"{where}: the channel {channel!r} has no variable name")
        _require(
            channel == sensor_id(plant_id, name),
            f"{where}: the channel {channel!r} is not named after its plant and variable, "
            f"{sensor_id(plant_id, name)!r}",
        )
        _require(
            kind in schema.CHANNEL_KINDS,
            f"{where}: channel {channel!r} is of kind {kind!r}, not one of {schema.CHANNEL_KINDS}",
        )
        si(f"sensors.{channel}", row["unit"])
        _require(
            positive(row, "sampling_period_s"),
            f"{where}: channel {channel!r} is sampled every {row['sampling_period_s']!r} s",
        )
        noise_model, noise_std = row["noise_model"], row["noise_std"]
        if kind == schema.CHANNEL_MEASURED:
            _require(
                noise_model == schema.NOISE_ADDITIVE_GAUSSIAN
                and noise_std is not None
                and math.isfinite(noise_std)
                and noise_std >= 0,
                f"{where}: measured channel {channel!r} needs a noise model and a noise level "
                f"of zero or more, got {noise_model!r} and {noise_std!r}",
            )
        else:
            _require(
                noise_model is None and noise_std is None,
                f"{where}: known input {channel!r} is presented with noise",
            )
        units.setdefault(name, set()).add(row["unit"])
        positions.setdefault((plant_id, kind), []).append(int(row["channel_index"]))
    mixed = {name: sorted(found) for name, found in units.items() if len(found) > 1}
    _require(not mixed, f"{where}: a variable has different units on different plants: {mixed}")
    for (plant_id, kind), found in positions.items():
        _require(
            sorted(found) == list(range(len(found))),
            f"{where}: the {kind} channels of {plant_id!r} are at positions {sorted(found)}, "
            f"not 0 to {len(found) - 1}",
        )
    _require(
        known <= {plant_id for plant_id, _ in positions},
        f"{where}: the plants {sorted(known - {p for p, _ in positions})} have no channel",
    )

    runs = dataset.table("operating_runs").to_pylist()
    run_ids = [_identifier("operating_runs.run_id", row["run_id"]) for row in runs]
    _require(len(set(run_ids)) == len(run_ids), f"{where}: a run is defined twice")
    for row in runs:
        of_plant("operating_runs", row)
        _require(
            row["dataset_id"] == dataset_id,
            f"{where}: the run {row['run_id']!r} says it belongs to {row['dataset_id']!r}, not "
            f"to {dataset_id!r}",
        )
        _require(
            _SHA256.fullmatch(row["content_sha256"]) is not None,
            f"{where}: the run {row['run_id']!r} has no SHA-256 content hash",
        )
        _require(
            positive(row, "sampling_period_s")
            and math.isfinite(row["start_time_s"])
            and math.isfinite(row["end_time_s"])
            and row["end_time_s"] >= row["start_time_s"]
            and row["n_samples"] >= 1,
            f"{where}: the run {row['run_id']!r} has an impossible extent, sampling period or "
            "number of samples",
        )
        _require(
            bool(row["operating_mode"].strip()) and bool(row["description"].strip()),
            f"{where}: the run {row['run_id']!r} has no operating mode or no description",
        )
    _require(
        known <= {row["plant_id"] for row in runs},
        f"{where}: the plants {sorted(known - {row['plant_id'] for row in runs})} have no run",
    )
