"""From what is known and observed to the rows of the five tables.

Everything here is on the available side. The only inputs are an ``Observations``, the
known specification of a plant (``PlantSpec``) and a few words about a run. There is no
``OperatingRun``, no ``RunTruth`` and no ``TruePlantConfig`` in any signature, and this
package does not import the simulation (``tests/test_boundaries.py``).

Known parameters pass through an explicit list of fields. A plant configuration is never
copied, so a field added to it later is not exported until someone adds it here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from process_transfer.config import PlantSpec, Quantity
from process_transfer.data import schema
from process_transfer.data.identifiers import path_identifier
from process_transfer.measurement.observations import Observations
from process_transfer.sampling_clock import nearest_ticks
from process_transfer.units import UnknownUnitError, si_unit_of
from process_transfer.validation import require_finite

# What an engineer knows about a plant, by name, and where it is in a PlantSpec. The
# nominal values of the modeller's simplified model (k0, E/R, UA) are not here: they are
# a starting point of a model, not known parameters of a plant.
KNOWN_PARAMETERS: tuple[tuple[str, Callable[[PlantSpec], Quantity]], ...] = (
    ("reactor_volume", lambda spec: spec.design.volume),
    ("density", lambda spec: spec.properties.density),
    ("heat_capacity", lambda spec: spec.properties.heat_capacity),
    ("reaction_enthalpy", lambda spec: spec.properties.reaction_enthalpy),
    ("nominal_feed_flow", lambda spec: spec.nominal_inputs.feed_flow),
    ("nominal_feed_concentration", lambda spec: spec.nominal_inputs.feed_concentration),
    ("nominal_feed_temperature", lambda spec: spec.nominal_inputs.feed_temperature),
    ("nominal_coolant_temperature", lambda spec: spec.nominal_inputs.coolant_temperature),
)


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _si_unit(name: str, unit: object) -> str:
    try:
        si_unit = si_unit_of(unit)  # type: ignore[arg-type]
    except (UnknownUnitError, TypeError) as error:
        raise ValueError(f"the unit of {name} is not known: {error}") from None
    if si_unit != unit:
        raise ValueError(f"{name} is given in {unit!r}; stored data are SI, here {si_unit!r}")
    return si_unit


@dataclass(frozen=True)
class KnownParameter:
    parameter: str
    value: float  # SI
    unit: str  # SI

    def __post_init__(self) -> None:
        _text("parameter", self.parameter)
        object.__setattr__(self, "value", require_finite(self.parameter, self.value))
        _si_unit(self.parameter, self.unit)


@dataclass(frozen=True)
class PlantRecord:
    """A plant as the available side knows it."""

    plant_id: str
    name: str
    process_type: str
    parameters: tuple[KnownParameter, ...]

    def __post_init__(self) -> None:
        path_identifier("plant_id", self.plant_id)
        _text("name", self.name)
        _text("process_type", self.process_type)
        object.__setattr__(self, "parameters", tuple(self.parameters))
        names = [parameter.parameter for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"a parameter is given twice for {self.plant_id}: {names}")


def known_plant(spec: PlantSpec) -> PlantRecord:
    """The record of a plant from the known part of its configuration, in SI. The argument
    is a ``PlantSpec``: the hidden physics of the file is not reachable from it."""
    if not isinstance(spec, PlantSpec):
        raise TypeError(
            "known_plant takes the known specification of a plant (PlantSpec), not "
            f"{type(spec).__name__}"
        )
    parameters = tuple(
        KnownParameter(name, quantity.si, quantity.si_unit)
        for name, quantity in ((name, get(spec)) for name, get in KNOWN_PARAMETERS)
    )
    return PlantRecord(spec.name, spec.name, spec.process_type, parameters)


@dataclass(frozen=True)
class RunRecord:
    """The few words about a run that its observations do not hold. All of it is what
    whoever ran the test on the plant would know."""

    dataset_id: str
    operating_mode: str
    description: str

    def __post_init__(self) -> None:
        path_identifier("dataset_id", self.dataset_id)
        _text("operating_mode", self.operating_mode)
        _text("description", self.description)


def sensor_id(plant_id: str, variable_name: str) -> str:
    return f"{plant_id}.{variable_name}"


def validate_observations(observations: Observations) -> np.ndarray:
    """The integer ticks of the rows, after checking what ``Observations`` itself does not:
    identifiers usable in a path, non-empty and distinct channel names, SI units, and a
    complete run, every row on the sensor clock and no tick missing.

    Contract version 1 stores complete runs only. A missing reading would be a missing
    tick, and the quality queries look for one, but nothing in M0 produces it.
    """
    if not isinstance(observations, Observations):
        raise TypeError(f"expected Observations, got {type(observations).__name__}")
    path_identifier("plant", observations.plant)
    path_identifier("run", observations.run)
    names = (*observations.measured_names, *observations.input_names)
    for name in names:
        _text("a channel name", name)
    if len(set(names)) != len(names):
        raise ValueError(f"every channel needs its own name, measured or input: {list(names)}")
    if len(observations.measured_names) == 0:
        raise ValueError("a run without any measured variable is not stored")
    for name, unit in zip(
        names, (*observations.measured_units, *observations.input_units), strict=True
    ):
        _si_unit(name, unit)

    times = observations.times
    ticks, on_clock = nearest_ticks(times, float(times[0]), observations.sample_period)
    if not np.all(on_clock):
        first = int(np.flatnonzero(~on_clock)[0])
        raise ValueError(
            f"row {first}, at t = {float(times[first])!r} s, is not on the sampling clock of "
            f"{observations.sample_period!r} s that starts at {float(times[0])!r} s"
        )
    if not np.array_equal(ticks, np.arange(len(times), dtype=np.int64)):
        first = int(np.flatnonzero(ticks != np.arange(len(times)))[0])
        raise ValueError(
            f"the run is not complete: row {first} is tick {int(ticks[first])} of the sensor "
            "clock. Contract version 1 stores complete runs only"
        )
    return ticks


def sensor_rows(observations: Observations) -> list[dict[str, object]]:
    """The channels of a run: its measured variables, then its known inputs."""
    rows: list[dict[str, object]] = []
    measured = zip(
        observations.measured_names,
        observations.measured_units,
        observations.noise_std,
        strict=True,
    )
    for index, (name, unit, noise_std) in enumerate(measured):
        rows.append(
            {
                "sensor_id": sensor_id(observations.plant, name),
                "plant_id": observations.plant,
                "variable_name": name,
                "channel_kind": schema.CHANNEL_MEASURED,
                "channel_index": index,
                "unit": unit,
                "sampling_period_s": observations.sample_period,
                "noise_model": schema.NOISE_ADDITIVE_GAUSSIAN,
                "noise_std": float(noise_std),
            }
        )
    inputs = zip(observations.input_names, observations.input_units, strict=True)
    for index, (name, unit) in enumerate(inputs):
        rows.append(
            {
                "sensor_id": sensor_id(observations.plant, name),
                "plant_id": observations.plant,
                "variable_name": name,
                "channel_kind": schema.CHANNEL_INPUT,
                "channel_index": index,
                "unit": unit,
                "sampling_period_s": observations.sample_period,
                "noise_model": None,  # known exactly (D-020): no noise model, not a zero one
                "noise_std": None,
            }
        )
    return rows


def run_row(observations: Observations, record: RunRecord) -> dict[str, object]:
    return {
        "run_id": observations.run,
        "plant_id": observations.plant,
        "dataset_id": record.dataset_id,
        "operating_mode": record.operating_mode,
        "description": record.description,
        "start_time_s": float(observations.times[0]),
        "end_time_s": float(observations.times[-1]),
        "sampling_period_s": observations.sample_period,
        "n_samples": observations.n_samples,
        "content_sha256": observations.content_digest(),
    }


def measurement_table(observations: Observations) -> pa.Table:
    """The readings and the inputs of a run in long format, one row per channel and tick,
    channel by channel. Values and instants are stored as they are, as 64-bit floats."""
    ticks = validate_observations(observations)
    n = observations.n_samples
    columns = [
        (name, observations.measured[:, index])
        for index, name in enumerate(observations.measured_names)
    ] + [
        (name, observations.inputs[:, index]) for index, name in enumerate(observations.input_names)
    ]
    sensor_ids = np.repeat([sensor_id(observations.plant, name) for name, _ in columns], n)
    return pa.table(
        {
            "plant_id": pa.array([observations.plant] * (n * len(columns)), pa.string()),
            "run_id": pa.array([observations.run] * (n * len(columns)), pa.string()),
            "sensor_id": pa.array(sensor_ids.tolist(), pa.string()),
            "sample_index": pa.array(np.tile(ticks, len(columns)), pa.int64()),
            "time_s": pa.array(np.tile(observations.times, len(columns)), pa.float64()),
            "value": pa.array(np.concatenate([values for _, values in columns]), pa.float64()),
            "quality_flag": pa.array(
                np.full(n * len(columns), schema.QUALITY_GOOD, dtype=np.int16), pa.int16()
            ),
        },
        schema=schema.MEASUREMENTS,
    )


def table_from_rows(name: str, rows: Sequence[dict[str, object]]) -> pa.Table:
    """A table with the schema of ``name``. Arrow checks the types; it does not enforce
    nullability, which is checked here."""
    table = pa.Table.from_pylist(list(rows), schema=schema.SCHEMAS[name])
    schema.require_no_nulls(name, table)
    return table
