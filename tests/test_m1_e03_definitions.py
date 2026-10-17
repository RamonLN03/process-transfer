"""The registered definitions of the data sets of M1-E03 (docs/experiment_log.md, M1-E03).

They are checked as the generator would read them, and nothing is simulated or generated:
the definitions load; their runs, identities, inputs, durations and sizes are those of the
plan (sections 5.2 and 5.4); the sets are disjoint; and their seeds and noise streams are
new. The simulator is replaced by a function that fails, so that a test that reached it
would fail rather than generate benchmark data.
"""

from pathlib import Path

import numpy as np
import pytest

from process_transfer.config import load_dataset_definition, load_sensors
from process_transfer.data.identifiers import require_distinct_runs
from process_transfer.generation import pipeline
from process_transfer.measurement.sensors import MeasurementSpec

DATASETS = Path(__file__).resolve().parents[1] / "configs" / "datasets"
NOMINAL = np.array([0.1 / 60.0, 500.0, 350.0, 337.5])

# set: file, runs, excursions per run, amplitude, rows per run
SETS = {
    "train-A10": ("m1_e03_train_a10.yaml", 10, 40, "a10", 4811),
    "train-A5": ("m1_e03_train_a5.yaml", 10, 40, "a5", 4811),
    "test-A10": ("m1_e03_test_a10.yaml", 8, 10, "a10", 1211),
    "test-A5": ("m1_e03_test_a5.yaml", 4, 10, "a5", 1211),
    "test-steps": ("m1_e03_test_steps.yaml", 8, None, None, 301),
}
CHANNELS = 6  # two readings and four inputs a row
# Seeds already used: the excitation seeds of M0-E03 (0 to 19), M0-E04/E05/E08 (0 to 2),
# the calibration of I1 (2026), the synthetic runs of the pilot of I3 (from 9000) and the
# tests of the data path (41 to 45); and every master, sensor, draw or weight seed of the
# experiments and of the data sets of M0.
USED_EXCITATION_SEEDS = set(range(20)) | {2026} | set(range(9000, 9010)) | set(range(41, 46))
USED_OTHER_SEEDS = {
    20260922,
    20260923,
    20260924,
    2026,
    2027,
    20260927,
    20260928,
    20260929,
    20260930,
    20260931,
    987654321,
}
M0_DEFINITIONS = ("m0_e05.yaml", "m0_e06.yaml", "m0_e07.yaml")


@pytest.fixture(autouse=True)
def nothing_is_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*arguments: object, **options: object) -> None:
        raise AssertionError("a check of the definitions must not simulate anything")

    monkeypatch.setattr(pipeline, "simulate_piecewise", refuse)


def definitions() -> dict[str, object]:
    return {name: load_dataset_definition(DATASETS / spec[0]) for name, spec in SETS.items()}


def runs_of(definition: object) -> list[pipeline.RunDefinition]:
    return pipeline.define_runs(definition, ["target"])


def test_every_registered_definition_loads_on_the_target_alone() -> None:
    assert sorted(p.name for p in DATASETS.glob("m1_e03_*.yaml")) == sorted(
        s[0] for s in SETS.values()
    )
    assert not (DATASETS / "m1_e03_draft").exists()  # moved, not copied
    for name, definition in definitions().items():
        assert [Path(p).name for p in definition.plants] == ["target_cstr.yaml"], name
        assert Path(definition.sensors).name == "sensors_cstr.yaml"
        assert definition.simulation_period.si == 0.1


@pytest.mark.parametrize("name", sorted(SETS))
def test_each_set_has_the_runs_inputs_and_size_of_the_plan(name: str) -> None:
    _, n_runs, excursions, amplitude, rows = SETS[name]
    definition = definitions()[name]
    runs = runs_of(definition)
    assert len(runs) == n_runs
    sensors = load_sensors(DATASETS / definition.sensors)
    period = MeasurementSpec.from_config(sensors).sample_period
    assert period == 6.0
    for run in runs:
        segments, _ = pipeline.run_segments(run, NOMINAL)
        pipeline.require_switches_on_the_sensor_clock(run.run_id, segments, period)
        assert sum(s.duration for s in segments) / period + 1 == rows
        if excursions is None:  # the step tests of D-026
            assert [s.duration for s in segments] == [600.0, 600.0, 600.0]
            assert run.run_id.endswith(".l600.h600.r600.n1")
        else:
            seed = run.settings["excitation_seed"]
            assert run.run_id == f"target.p3.e{seed}.x{excursions}.l60.{amplitude}.n0"
            assert segments[0].duration == 60.0 and len(segments) == 1 + 2 * excursions
            np.testing.assert_array_equal(segments[0].inputs, NOMINAL)


def test_the_sizes_are_those_of_the_plan() -> None:
    measurements = {
        name: SETS[name][1] * SETS[name][4] * CHANNELS for name in SETS
    }  # runs x rows x channels
    assert measurements["train-A10"] + measurements["train-A5"] == 577_320
    assert sum(measurements[n] for n in ("test-A10", "test-A5", "test-steps")) == 101_640


def test_the_sets_are_disjoint_and_their_noise_streams_distinct() -> None:
    found = definitions()
    ids = [run.run_id for definition in found.values() for run in runs_of(definition)]
    assert len(ids) == sum(spec[1] for spec in SETS.values()) == 40
    streams = require_distinct_runs(ids)  # raises on a shared identity or stream
    assert len(set(streams.values())) == 40
    assert len({d.dataset_id for d in found.values()}) == 5
    masters = [d.sensor_master_seed for d in found.values()]
    assert len(set(masters)) == 5


def test_no_seed_or_run_of_an_earlier_experiment_is_reused() -> None:
    found = definitions()
    excitation = [s for d in found.values() if d.protocol == "p3" for s in d.excitation_seeds]
    assert len(excitation) == len(set(excitation)) == 32
    assert not set(excitation) & USED_EXCITATION_SEEDS
    masters = {d.sensor_master_seed for d in found.values()}
    assert not masters & USED_OTHER_SEEDS
    m0 = [load_dataset_definition(DATASETS / name) for name in M0_DEFINITIONS]
    assert not masters & {d.sensor_master_seed for d in m0}
    m0_ids = {run.run_id for d in m0 for run in pipeline.define_runs(d, ["source", "target"])}
    new_ids = {run.run_id for d in found.values() for run in runs_of(d)}
    assert not m0_ids & new_ids
    assert not {d.dataset_id for d in m0} & {d.dataset_id for d in found.values()}
