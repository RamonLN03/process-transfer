"""What the available side may hold: known parameters through an explicit list, table
digests that depend on content only, and the private record of an attempt."""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import BaseModel

from conftest import synthetic_observations
from process_transfer.canonical import encode_list, encode_numbers, encode_scalar, encode_string
from process_transfer.config import PlantSpec, Quantity, load_modeller, load_true_plant
from process_transfer.data import schema
from process_transfer.data.private_store import private_root, write_private_attempt
from process_transfer.data.records import (
    KNOWN_PARAMETERS,
    KnownParameter,
    PlantRecord,
    RunRecord,
    known_plant,
    measurement_table,
    sensor_rows,
    table_from_rows,
)


def quantities(model: BaseModel, prefix: str = "") -> dict[str, Quantity]:
    found: dict[str, Quantity] = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, Quantity):
            found[f"{prefix}{name}"] = value
        elif isinstance(value, BaseModel):
            found.update(quantities(value, f"{prefix}{name}."))
    return found


# --------------------------------------------------------------------------- #
# Known parameters
# --------------------------------------------------------------------------- #


def test_every_known_quantity_is_listed_and_nothing_else_is(configs_dir: Path) -> None:
    """The list is explicit. A quantity added to ``PlantSpec`` is not exported until it is
    added to the list, and this test says so when that happens."""
    spec = load_true_plant(configs_dir / "target_cstr.yaml").plant
    in_the_specification = quantities(spec)
    assert len(in_the_specification) == len(KNOWN_PARAMETERS) == 8
    listed = [get(spec) for _, get in KNOWN_PARAMETERS]
    assert all(any(q is known for known in in_the_specification.values()) for q in listed)
    assert len({id(q) for q in listed}) == 8  # each one once


@pytest.mark.parametrize("name", ["source", "target"])
def test_the_record_of_a_plant_is_in_si_and_holds_nothing_hidden(
    name: str, configs_dir: Path
) -> None:
    cfg = load_true_plant(configs_dir / f"{name}_cstr.yaml")
    record = known_plant(cfg.plant)
    assert (record.plant_id, record.name, record.process_type) == (name, name, "cstr")
    values = {p.parameter: (p.value, p.unit) for p in record.parameters}
    assert values["reactor_volume"] == (pytest.approx(0.1), "m^3")  # 100 L
    assert values["density"] == (1000.0, "kg/m^3")
    assert values["heat_capacity"] == (pytest.approx(239.0), "J/(kg*K)")
    assert values["reaction_enthalpy"] == (-5.0e4, "J/mol")  # signed, as configured (D-015)
    assert values["nominal_feed_flow"] == (pytest.approx(100.0e-3 / 60.0), "m^3/s")
    assert values["nominal_feed_concentration"] == (pytest.approx(500.0), "mol/m^3")
    assert values["nominal_feed_temperature"] == (350.0, "K")
    assert values["nominal_coolant_temperature"] == (337.5, "K")

    hidden = quantities(cfg.true_physics)
    assert len(hidden) == 6
    exported = [value for value, _ in values.values()]
    for label, quantity in hidden.items():
        if label == "heat_transfer.T_ref":
            continue  # 350 K, which is also the known feed temperature; checked by name below
        assert not any(math.isclose(quantity.si, value, rel_tol=1e-12) for value in exported), label
    assert not {
        "k0",
        "activation_temperature",
        "saturation_constant",
        "UA_ref",
        "alpha",
        "T_ref",
    } & set(values)


def test_the_nominal_values_of_the_modeller_are_not_known_plant_parameters(
    configs_dir: Path,
) -> None:
    """k0, E/R and UA of the simplified model are where a model starts, not facts about a
    plant. They are not exported, and the exporter has no way to be handed them."""
    modeller = load_modeller(configs_dir / "modeller_cstr.yaml")
    record = known_plant(load_true_plant(configs_dir / "source_cstr.yaml").plant)
    exported = [p.value for p in record.parameters]
    for quantity in quantities(modeller).values():
        assert not any(math.isclose(quantity.si, value, rel_tol=1e-12) for value in exported)
    with pytest.raises(TypeError, match="PlantSpec"):
        known_plant(modeller)  # type: ignore[arg-type]


def test_the_whole_configuration_cannot_be_handed_over(configs_dir: Path) -> None:
    cfg = load_true_plant(configs_dir / "source_cstr.yaml")
    with pytest.raises(TypeError, match="not TruePlantConfig"):
        known_plant(cfg)  # type: ignore[arg-type]
    assert isinstance(cfg.plant, PlantSpec) and not hasattr(cfg.plant, "true_physics")


def test_invalid_records_are_refused() -> None:
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="density"):
            KnownParameter("density", bad, "kg/m^3")
    with pytest.raises(ValueError, match="stored data are SI"):
        KnownParameter("reactor_volume", 100.0, "L")
    with pytest.raises(ValueError, match="is not known"):
        KnownParameter("reactor_volume", 0.1, "barrels")
    with pytest.raises(ValueError, match="non-empty"):
        KnownParameter("", 0.1, "m^3")
    volume = KnownParameter("reactor_volume", 0.1, "m^3")
    with pytest.raises(ValueError, match="given twice"):
        PlantRecord("target", "target", "cstr", (volume, volume))
    with pytest.raises(ValueError, match="plant_id"):
        PlantRecord("Target Plant", "target", "cstr", ())
    with pytest.raises(ValueError, match="non-empty"):
        RunRecord("m0", "p3", "   ")
    with pytest.raises(ValueError, match="dataset_id"):
        RunRecord("M0 data", "p3", "x")


# --------------------------------------------------------------------------- #
# Rows and digests
# --------------------------------------------------------------------------- #


def test_the_long_table_of_a_run() -> None:
    observations = synthetic_observations(n=5)
    table = measurement_table(observations)
    assert table.schema.equals(schema.MEASUREMENTS)
    assert table.num_rows == 5 * 6
    rows = table.to_pylist()
    assert [row["sensor_id"] for row in rows[::5]] == [
        f"target.{name}" for name in ("C_A", "T", "q", "C_Af", "T_f", "T_c")
    ]
    assert [row["sample_index"] for row in rows[:5]] == [0, 1, 2, 3, 4]
    assert [row["time_s"] for row in rows[:5]] == [0.0, 6.0, 12.0, 18.0, 24.0]
    assert rows[3]["value"] == -2.5  # the negative reading, as it was
    assert {row["quality_flag"] for row in rows} == {0}
    assert [row["channel_kind"] for row in sensor_rows(observations)] == ["measured"] * 2 + [
        "input"
    ] * 4


def test_a_table_digest_depends_on_content_and_not_on_the_order_of_rows() -> None:
    rows = sensor_rows(synthetic_observations())
    digest = schema.table_digest("sensors", table_from_rows("sensors", rows))
    assert digest == schema.table_digest("sensors", table_from_rows("sensors", rows[::-1]))
    altered = [dict(row) for row in rows]
    altered[0]["noise_std"] = float(np.nextafter(5.0, 6.0))  # one unit in the last place
    assert digest != schema.table_digest("sensors", table_from_rows("sensors", altered))
    nulled = [dict(row) for row in rows]
    nulled[0]["noise_std"] = None  # a null is not a zero
    zeroed = [dict(row) for row in rows]
    zeroed[0]["noise_std"] = 0.0
    assert (
        len(
            {
                schema.table_digest("sensors", table_from_rows("sensors", r))
                for r in (rows, nulled, zeroed)
            }
        )
        == 3
    )
    with pytest.raises(ValueError, match="columns"):
        schema.table_digest("sensors", pa.table({"sensor_id": ["x"]}))


def test_arrow_refuses_rows_that_do_not_fit_the_schema() -> None:
    rows = sensor_rows(synthetic_observations())
    with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError)):
        table_from_rows("sensors", [{**rows[0], "sampling_period_s": "six"}])
    # Arrow records that a column is not nullable and does not enforce it; this code does
    assert pa.Table.from_pylist([{**rows[0], "sensor_id": None}], schema=schema.SENSORS).num_rows
    with pytest.raises(ValueError, match="sensors.sensor_id must not be null"):
        table_from_rows("sensors", [{**rows[0], "sensor_id": None}])
    table_from_rows("sensors", [{**rows[0], "noise_std": None, "noise_model": None}])  # nullable


def test_the_canonical_encodings_are_unambiguous() -> None:
    assert encode_string("ab") == b"S" + (2).to_bytes(8, "big") + b"ab"
    assert encode_list([encode_string("a"), encode_string("b")]) != encode_list(
        [encode_string("ab")]
    )
    assert encode_numbers(np.zeros((2, 3))) != encode_numbers(np.zeros((3, 2)))
    assert encode_numbers(np.zeros((0, 4))) != encode_numbers(
        np.zeros((0, 2))
    )  # empty, still shaped
    assert encode_scalar(None) == b"N" and encode_scalar(0) != encode_scalar(0.0)
    assert encode_scalar(-0.0) != encode_scalar(0.0)  # bit for bit
    assert encode_scalar(np.int64(7)) == encode_scalar(7)
    assert encode_scalar(np.float64(0.5)) == encode_scalar(0.5)
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            encode_scalar(bad)
    for bad in (True, np.bool_(False), b"bytes", [1], {"a": 1}):
        with pytest.raises(TypeError):
            encode_scalar(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        encode_string(7)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The private record of an attempt
# --------------------------------------------------------------------------- #


def test_every_attempt_has_its_own_private_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configs_dir: Path
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path))
    state = {"commit": "0123456789abcdef" * 2 + "01234567", "dirty": False}
    provenance = {"sensor_seed": 5, "streams": {"target.p3.e0.x10.n0": [1, 2, 3, 4]}}
    files = [configs_dir / "target_cstr.yaml", configs_dir / "sensors_cstr.yaml"]
    first = write_private_attempt("m0-test", provenance, files, state)
    second = write_private_attempt("m0-test", provenance, files, state)  # the same second

    assert first != second and first.parent == second.parent == private_root() / "m0-test"
    assert private_root() == tmp_path / "private" / "datasets"
    record = json.loads((first / "provenance.json").read_text(encoding="utf-8"))
    assert record["sensor_seed"] == 5 and record["attempt_directory"] == first.name
    assert [entry["name"] for entry in record["configurations"]] == [
        "target_cstr.yaml",
        "sensors_cstr.yaml",
    ]
    # the private copy is the whole file, hidden physics included; that is what it is for
    assert "true_physics" in (first / "configs" / "target_cstr.yaml").read_text(encoding="utf-8")
    assert not (tmp_path / "available").exists()  # nothing of this goes to the available branch


def test_a_private_record_that_is_not_strict_json_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path))
    for bad in ({"peak": math.nan}, {"when": object()}):
        with pytest.raises(ValueError, match="strict JSON"):
            write_private_attempt("m0-test", bad, [], {"commit": None, "dirty": None})
    with pytest.raises(ValueError, match="dataset_id"):
        write_private_attempt("../outside", {}, [], {"commit": None, "dirty": None})
    assert not (tmp_path / "private").exists()  # nothing was created for a refused record
