"""The five tables of M0 as explicit Arrow schemas (``docs/data_contract.md``).

Types, nullability and column order are part of the contract and are verified when a
file is read, not only intended when it is written. Floats are 64-bit throughout:
readings and instants are stored as they are, never rounded. Units are SI; the unit of
a fixed column is carried as field metadata, the unit of a value is that of its channel
in ``sensors``.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa

from process_transfer.canonical import encode_list, encode_scalar, encode_string

SCHEMA_VERSION = 1
TABLE_ENCODING = "table/v1"

CHANNEL_MEASURED = "measured"
CHANNEL_INPUT = "input"
CHANNEL_KINDS = (CHANNEL_MEASURED, CHANNEL_INPUT)
NOISE_ADDITIVE_GAUSSIAN = "additive_gaussian"

# The only value of contract version 1: the value is present and finite, the instant is
# finite and on the sampling clock of its channel, and the channel is declared with a
# known unit. The writer refuses anything else, so it never writes another value.
QUALITY_GOOD = 0

TIME_CONVENTION = (
    "time_s is process time in seconds on the clock of the run; it is not a date and has "
    "no time zone. sample_index is the integer tick of the sensor clock: time_s = "
    "start_time_s + sample_index * sampling_period_s. A row holds the readings of the "
    "state at its instant and the inputs applied from that instant on, until the next row "
    "at which they differ (zero-order hold, right-continuous)."
)


def _field(
    name: str, kind: pa.DataType, nullable: bool = False, unit: str | None = None
) -> pa.Field:
    metadata = None if unit is None else {"unit": unit}
    return pa.field(name, kind, nullable=nullable, metadata=metadata)


PLANTS = pa.schema(
    [
        _field("plant_id", pa.string()),
        _field("name", pa.string()),
        _field("process_type", pa.string()),
    ]
)

PROCESS_PARAMETERS = pa.schema(
    [
        _field("plant_id", pa.string()),
        _field("parameter", pa.string()),
        _field("value", pa.float64(), unit="given by the column unit"),
        _field("unit", pa.string()),
    ]
)

SENSORS = pa.schema(
    [
        _field("sensor_id", pa.string()),
        _field("plant_id", pa.string()),
        _field("variable_name", pa.string()),
        _field("channel_kind", pa.string()),
        _field("channel_index", pa.int32()),
        _field("unit", pa.string()),
        _field("sampling_period_s", pa.float64(), unit="s"),
        _field("noise_model", pa.string(), nullable=True),
        _field("noise_std", pa.float64(), nullable=True, unit="given by the column unit"),
    ]
)

OPERATING_RUNS = pa.schema(
    [
        _field("run_id", pa.string()),
        _field("plant_id", pa.string()),
        _field("dataset_id", pa.string()),
        _field("operating_mode", pa.string()),
        _field("description", pa.string()),
        _field("start_time_s", pa.float64(), unit="s"),
        _field("end_time_s", pa.float64(), unit="s"),
        _field("sampling_period_s", pa.float64(), unit="s"),
        _field("n_samples", pa.int64()),
        _field("content_sha256", pa.string()),
    ]
)

MEASUREMENTS = pa.schema(
    [
        _field("plant_id", pa.string()),
        _field("run_id", pa.string()),
        _field("sensor_id", pa.string()),
        _field("sample_index", pa.int64()),
        _field("time_s", pa.float64(), unit="s"),
        _field("value", pa.float64(), unit="given by sensors.unit"),
        _field("quality_flag", pa.int16()),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {
    "plants": PLANTS,
    "process_parameters": PROCESS_PARAMETERS,
    "sensors": SENSORS,
    "operating_runs": OPERATING_RUNS,
    "measurements": MEASUREMENTS,
}

# The columns that identify a row, and by which rows are ordered before a table is hashed.
KEYS: dict[str, tuple[str, ...]] = {
    "plants": ("plant_id",),
    "process_parameters": ("plant_id", "parameter"),
    "sensors": ("sensor_id",),
    "operating_runs": ("run_id",),
    "measurements": ("run_id", "sensor_id", "sample_index"),
}


def require_no_nulls(name: str, table: pa.Table) -> None:
    """``ValueError`` if a column that the schema of ``name`` declares not nullable holds a
    null. Arrow records nullability and does not enforce it: a table with a null in such
    a column is built without complaint, so it is checked here, on writing and on reading.
    """
    for field in SCHEMAS[name]:
        if not field.nullable and table.column(field.name).null_count:
            raise ValueError(
                f"{name}.{field.name} must not be null, and "
                f"{table.column(field.name).null_count} of {table.num_rows} rows are"
            )


def table_digest(name: str, table: pa.Table) -> str:
    """SHA-256 of the canonical encoding ``table/v1`` of a small table: a version header,
    the table name, the column names, then the rows ordered by their key, every value
    with its kind and length. It does not depend on the order of the rows in a file or
    on how Parquet wrote them. Meant for the four small tables; the content of a run is
    identified by the digest of its observations."""
    schema = SCHEMAS[name]
    if table.schema.names != schema.names:
        raise ValueError(f"{name}: columns {table.schema.names} are not {schema.names}")
    rows = table.to_pylist()
    rows.sort(key=lambda row: tuple(row[column] for column in KEYS[name]))
    digest = hashlib.sha256()
    digest.update(f"process-transfer/{TABLE_ENCODING}\n".encode("ascii"))
    digest.update(encode_string(name))
    digest.update(encode_list([encode_string(column) for column in schema.names]))
    digest.update(
        encode_list(
            [encode_list([encode_scalar(row[column]) for column in schema.names]) for row in rows]
        )
    )
    return digest.hexdigest()
