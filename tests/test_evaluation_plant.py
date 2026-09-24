"""The known specification of a plant is read from a verified export, never from a plant
configuration file, and whatever does not match the contract is refused, not converted."""

import copy
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_observations
from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.data import database
from process_transfer.data.export import Export, export_dataset, open_export_directory
from process_transfer.data.parquet_store import DatasetWriter, open_dataset
from process_transfer.data.records import RunRecord, known_plant
from process_transfer.evaluation.plant import KnownPlant, read_known_plant

DATASET = "plant-test"
ATTEMPT = {"generated_at_utc": "2026-09-25T00:00:00+00:00", "commit": "0" * 40}


@pytest.fixture
def export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configs_dir: Path) -> Export:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "pt-data"))
    plants = [known_plant(load_true_plant(configs_dir / "target_cstr.yaml").plant)]
    with DatasetWriter(DATASET, plants, ATTEMPT) as writer:
        writer.add_run(
            synthetic_observations("target", "target.p3.e0.x1.n0"),
            RunRecord(DATASET, "p3", "synthetic run for a test"),
        )
        writer.publish()
    connection = database.connect()
    database.ingest_dataset(connection, open_dataset(DATASET))
    result = export_dataset(connection, DATASET, ATTEMPT)
    connection.close()
    return open_export_directory(result.directory)


def test_the_known_parameters_are_those_of_the_plant(export: Export, configs_dir: Path) -> None:
    plant = read_known_plant(export, "target")
    spec = load_true_plant(configs_dir / "target_cstr.yaml").plant
    assert plant.volume == spec.design.volume.si == 0.1
    assert plant.density == spec.properties.density.si
    assert plant.heat_capacity == spec.properties.heat_capacity.si
    assert plant.reaction_enthalpy == spec.properties.reaction_enthalpy.si
    np.testing.assert_array_equal(plant.nominal_inputs, nominal_inputs(spec))
    assert not plant.nominal_inputs.flags.writeable
    assert plant.thermal_mass == 0.1 * 1000.0 * 239.0
    assert plant.heat_release_per_mole == 50000.0 / (1000.0 * 239.0)


Edit = Callable[[list[dict]], object]


def edited(export: Export, edit: Edit) -> Export:
    manifest = copy.deepcopy(dict(export.manifest))
    edit(manifest["plants"]["target"]["known_parameters"])
    return Export(export.directory, manifest)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda entries: entries.pop(), "expected exactly"),
        (lambda entries: entries.append(dict(entries[0])), "expected exactly"),
        (lambda entries: entries[0].update(parameter="alpha"), "expected exactly"),
        (lambda entries: entries[0].update(unit="g/cm^3"), "not converted"),
        (lambda entries: entries[0].update(value="1000"), "must be a number"),
        (lambda entries: entries[0].update(value=True), "must be a number"),
        (lambda entries: entries[0].update(value=float("nan")), "finite"),
    ],
)
def test_known_parameters_that_break_the_contract_are_refused(
    export: Export, edit: Edit, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_known_plant(edited(export, edit), "target")


def test_a_plant_that_the_export_does_not_describe_is_refused(export: Export) -> None:
    with pytest.raises(KeyError, match="no plant 'source'"):
        read_known_plant(export, "source")


def test_known_parameters_are_checked_where_they_enter() -> None:
    nominal = np.array([1e-3, 500.0, 350.0, 337.5])
    with pytest.raises(ValueError, match="volume"):
        KnownPlant("p", 0.0, 1000.0, 239.0, -5e4, nominal)
    with pytest.raises(ValueError, match="nominal T_c"):
        KnownPlant("p", 0.1, 1000.0, 239.0, -5e4, np.array([1e-3, 500.0, 350.0, -1.0]))
    with pytest.raises(ValueError, match="4 values"):
        KnownPlant("p", 0.1, 1000.0, 239.0, -5e4, nominal[:3])
    with pytest.raises(ValueError, match="volume \\* density \\* heat_capacity"):
        KnownPlant("p", 1e200, 1e200, 239.0, -5e4, nominal)
