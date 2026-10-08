"""P3 with the lead and the amplitude of M1 through the data path (D-039).

* The definitions of M0 give the runs, inputs and words they gave before. What is pinned
  here is what does not depend on the platform: identities, settings, input segments and
  descriptions. The content hashes of simulated data are not pinned; the exact
  reproduction of the published data sets of M0 is a recorded check on the reference
  environment (experiment log, I4), not a test.
* A definition of M1 gives the new identities and segments. At A5 only the inputs are
  built: what A5 does to the plant is M1-E02's question, and no A5 data are generated here.
* The whole path runs on one short run with the lead at A10, the amplitude M0-E03b
  verified, and its windows are read back from the export.

The seeds are those of the tests of the data path, of no experiment: excitation seeds 41
to 43 and the master seed 987654321.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from m1_support import observations, p3_inputs
from process_transfer.config import load_dataset_definition
from process_transfer.data import database
from process_transfer.data.export import open_export_directory
from process_transfer.evaluation.windows import find_windows, layout_for
from process_transfer.generation import pipeline
from process_transfer.simulation.integration import InputSegment
from process_transfer.simulation.protocols import (
    a5_amplitudes,
    a10_amplitudes,
    p3_corners,
    p3_segments,
)

MASTER_SEED = 987654321
NOMINAL = np.array([0.1 / 60.0, 500.0, 350.0, 337.5])

M0_RUNS = {
    "m0_e05.yaml": [
        f"{plant}.p3.e{seed}.x10.n0" for seed in (0, 1, 2) for plant in ("source", "target")
    ],
    "m0_e06.yaml": ["source.steady.d7200.n0", "target.steady.d7200.n0"],
    "m0_e07.yaml": [
        f"{plant}.step.{name}-{direction}.l600.h600.r600.n0"
        for name in ("q", "caf", "tf", "tc")
        for direction in ("up", "down")
        for plant in ("source", "target")
    ],
}
# the description of target.p3.e0.x10.n0 in the published export of m0-e05
M0_DESCRIPTION = (
    "M0-E05, the full data path on three P3 sequences per plant. Protocol P3 (D-019): A10 "
    "amplitudes, 120 s at a corner of the input box, 600 s at the nominal inputs, 10 "
    "excursions, excitation seed 0, noise realisation 0."
)


def write_definition(directory: Path, configs_dir: Path, **changes: object) -> Path:
    definition = {
        "dataset_id": "lead-test",
        "description": "A small data set with the lead of M1, for the tests of the data path.",
        "plants": [str(configs_dir / "target_cstr.yaml")],
        "sensors": str(configs_dir / "sensors_cstr.yaml"),
        "protocol": "p3",
        "n_excursions": 1,
        "excitation_seeds": [41],
        "noise_realisation": 0,
        "sensor_master_seed": MASTER_SEED,
        "simulation_period": {"value": 0.1, "unit": "s"},
        "lead": {"value": 60, "unit": "s"},
        "amplitude": "a10",
        **changes,
    }
    definition = {key: value for key, value in definition.items() if value is not None}
    path = directory / "definition.json"  # JSON is YAML
    path.write_text(json.dumps(definition), encoding="utf-8")
    return path


def inputs_on_the_sensor_clock(segments: list[InputSegment], period: float = 6.0) -> np.ndarray:
    """The inputs of each row of a run: those applied from its instant on (right-continuous),
    and at the last instant those of the last segment."""
    ends = np.cumsum([segment.duration for segment in segments])
    ticks = period * np.arange(round(ends[-1] / period) + 1)
    index = np.minimum(np.searchsorted(ends, ticks, side="right"), len(segments) - 1)
    return np.array([segments[i].inputs for i in index])


@pytest.mark.parametrize("name", sorted(M0_RUNS))
def test_the_definitions_of_m0_give_the_runs_they_gave(name: str, configs_dir: Path) -> None:
    definition = load_dataset_definition(configs_dir / "datasets" / name)
    runs = pipeline.define_runs(definition, ["source", "target"])
    assert sorted(run.run_id for run in runs) == sorted(M0_RUNS[name])
    if definition.protocol == "p3":
        assert definition.lead is None and definition.amplitude is None
        for run in runs:
            assert set(run.settings) == {"excitation_seed", "n_excursions"}
            segments, about = pipeline.run_segments(run, NOMINAL)
            corners = p3_corners(10, int(run.settings["excitation_seed"]))
            assert len(segments) == 20 and len(about["corners"]) == 10  # type: ignore[arg-type]
            for got, expected in zip(segments, p3_segments(NOMINAL, corners), strict=True):
                assert got.duration == expected.duration
                np.testing.assert_array_equal(got.inputs, expected.inputs)
        target = next(run for run in runs if run.run_id == "target.p3.e0.x10.n0")
        assert pipeline.describe_run(definition.description, target) == M0_DESCRIPTION
    # the record of a definition of M0 holds the fields it sets, as it did
    recorded = definition.model_dump(mode="json", exclude_unset=True)
    if definition.protocol == "p3":
        assert "lead" not in recorded and "amplitude" not in recorded


@pytest.mark.parametrize("amplitude", ["a10", "a5"])
def test_a_definition_of_m1_gives_its_identities_and_inputs(
    tmp_path: Path, configs_dir: Path, amplitude: str
) -> None:
    definition = load_dataset_definition(
        write_definition(tmp_path, configs_dir, amplitude=amplitude, excitation_seeds=[42, 43])
    )
    assert definition.lead_s == 60 and definition.amplitude == amplitude
    runs = pipeline.define_runs(definition, ["target"])
    assert [run.run_id for run in runs] == [
        f"target.p3.e42.x1.l60.{amplitude}.n0",
        f"target.p3.e43.x1.l60.{amplitude}.n0",
    ]
    run = runs[0]
    assert run.settings == {
        "excitation_seed": 42,
        "n_excursions": 1,
        "lead_s": 60,
        "amplitude": amplitude,
    }
    segments, _ = pipeline.run_segments(run, NOMINAL)
    assert [s.duration for s in segments] == [60.0, 120.0, 600.0]
    np.testing.assert_array_equal(segments[0].inputs, NOMINAL)
    amplitudes = a5_amplitudes(NOMINAL) if amplitude == "a5" else a10_amplitudes(NOMINAL)
    np.testing.assert_array_equal(segments[1].inputs, NOMINAL + p3_corners(1, 42)[0] * amplitudes)
    description = pipeline.describe_run(definition.description, run)
    assert "60 s at the nominal inputs" in description
    assert f"{amplitude.upper()} amplitudes" in description and "seed 42" in description


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"amplitude": None}, "given together or not at all"),
        ({"lead": None}, "given together or not at all"),
        ({"lead": {"value": 0, "unit": "s"}}, "greater than 0"),
        ({"lead": {"value": -60, "unit": "s"}}, "greater than 0"),
        ({"lead": {"value": 60.5, "unit": "s"}}, "whole number of seconds"),
        ({"lead": {"value": 60, "unit": "K"}}, "requires time"),
        ({"amplitude": "a7"}, "Input should be 'a10' or 'a5'"),
        ({"amplitude": "A5"}, "Input should be 'a10' or 'a5'"),
    ],
)
def test_a_definition_with_a_lead_or_amplitude_outside_its_domain_is_refused(
    tmp_path: Path, configs_dir: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        load_dataset_definition(write_definition(tmp_path, configs_dir, **changes))


def test_a_lead_off_the_clock_of_the_sensors_is_refused_before_simulating(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segments = p3_segments(NOMINAL, p3_corners(2, 41), "a10", 61.0)
    with pytest.raises(ValueError, match=r"not ticks of the sensors, every 6.0 s"):
        pipeline.require_switches_on_the_sensor_clock("r", segments, 6.0)
    pipeline.require_switches_on_the_sensor_clock("r", p3_segments(NOMINAL, p3_corners(2, 41)), 6.0)

    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    simulated: list[object] = []
    monkeypatch.setattr(pipeline, "simulate_piecewise", lambda *a, **k: simulated.append(a))
    definition = write_definition(tmp_path, configs_dir, lead={"value": 61, "unit": "s"})
    with pytest.raises(ValueError, match="target.p3.e41.x1.l61.a10.n0"):
        pipeline.run_pipeline(definition, figures=False)
    assert simulated == []
    assert not (tmp_path / "data" / "available" / "datasets" / "lead-test").exists()


@pytest.mark.parametrize("amplitude", ["a10", "a5"])
def test_the_windows_of_a_run_built_by_the_new_protocol_are_found_from_its_inputs(
    amplitude: str,
) -> None:
    """Inputs only, on the clock of the sensors; the readings are synthetic."""
    corners = p3_corners(5, 7)
    with_lead = inputs_on_the_sensor_clock(p3_segments(NOMINAL, corners, amplitude, 60.0))
    if amplitude == "a10":  # the layout the tests of I1 were written for
        np.testing.assert_array_equal(with_lead, p3_inputs(corners.tolist(), lead=True))
    found = find_windows(
        observations(with_lead, run=f"target.p3.e7.x5.l60.{amplitude}.n0"),
        NOMINAL,
        layout_for("p3"),
    )
    assert found.onsets == tuple(10 + 120 * j for j in range(5)) and not found.skipped
    assert [w.excursion for w in found.windows] == [1, 2, 3, 4, 5]
    for window, following in zip(found.windows, found.windows[1:], strict=False):
        assert not set(window.context_ticks) & set(window.scored_ticks)
        assert not set(following.context_ticks) & set(window.scored_ticks)
    # the ten rows before each onset carry the nominal inputs: the lead, then each rest;
    # the row at the onset carries the corner (right-continuous)
    for window in found.windows:
        before = with_lead[window.onset - 10 : window.onset]
        np.testing.assert_array_equal(before, np.tile(NOMINAL, (10, 1)))
        assert not np.array_equal(with_lead[window.onset], NOMINAL)

    without = inputs_on_the_sensor_clock(p3_segments(NOMINAL, corners, amplitude))
    m0 = find_windows(observations(without), NOMINAL, layout_for("p3"))
    assert m0.onsets[0] == 0 and [s.excursion for s in m0.skipped] == [1]


def test_the_whole_path_runs_with_the_lead(
    tmp_path: Path, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    report = pipeline.run_pipeline(write_definition(tmp_path, configs_dir), figures=False)
    assert report["ok"] is True, report["checks"]
    run_id = "target.p3.e41.x1.l60.a10.n0"
    assert list(report["runs"]) == [run_id]
    assert report["rows"]["measurements"] == 131 * 6  # (60 + 720) s / 6 s + 1 rows

    export = open_export_directory(Path(report["export"]["directory"]))
    observed = export.observations(run_id)
    assert observed.times[0] == 0.0 and observed.times[-1] == 780.0
    found = find_windows(observed, NOMINAL, layout_for("p3"))
    assert found.onsets == (10,) and len(found.windows) == 1 and not found.skipped

    connection = database.connect(database.database_path("lead-test"))
    (mode, description) = connection.execute(
        "SELECT operating_mode, description FROM operating_runs"
    ).fetchone()
    connection.close()
    assert mode == "p3" and "60 s at the nominal inputs" in description

    private = json.loads(
        (Path(report["private_record"]) / "provenance.json").read_text(encoding="utf-8")
    )
    assert private["p3_extensions_of_m1"]["lead_s"] == 60
    assert private["p3_extensions_of_m1"]["amplitude"] == "a10"
    assert private["definition"]["lead"] == {"value": 60.0, "unit": "s"}
    assert private["runs"][run_id]["truth_accepted"] is True
