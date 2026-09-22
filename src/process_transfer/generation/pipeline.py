"""From configuration files to a verified data set, a database and an export.

    1  load the definition of the data set, the plants and the instruments
    2  build the plants and verify their starting points
    3  generate protocol P3 for every excitation seed
    4  simulate and validate the true trajectories
    5  observe them with the sensors
    6  write the Parquet data set, and the private record of the attempt
    7  ingest it into DuckDB
    8  run the SQL quality checks
    9  export the aligned series
    10 check the whole, and write a report and basic figures

Runs are processed one after another, so that only one dense trajectory, 72 001 samples
for two hours, is in memory at a time. What is kept of a run is small: its observations,
the exact values at the sensor instants for the scan of hidden information, and a summary
of the checks of its truth.

The report is written whatever happens, in the private record of the attempt, and says
which mandatory checks passed. ``ok`` is true only if all of them did. Nothing is
retried, no seed is replaced and no check is skipped to obtain a pass.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from process_transfer.config import (
    DatasetDefinitionConfig,
    load_dataset_definition,
    load_sensors,
)
from process_transfer.data import database
from process_transfer.data.export import export_dataset, open_export_directory
from process_transfer.data.identifiers import require_distinct_runs, run_identifier
from process_transfer.data.parquet_store import DatasetWriter, open_dataset
from process_transfer.data.private_store import write_private_attempt
from process_transfer.data.provenance import environment, git_state
from process_transfer.data.records import RunRecord, known_plant
from process_transfer.generation.leak_scan import HiddenValues, scan_available
from process_transfer.generation.plants import VirtualPlant, load_virtual_plant
from process_transfer.measurement.observations import Observations
from process_transfer.measurement.sensors import MeasurementSpec
from process_transfer.simulation.integration import simulate_piecewise
from process_transfer.simulation.operating_run import observe_trajectory
from process_transfer.simulation.protocols import (
    P3_HOLD,
    P3_REST,
    a10_amplitudes,
    corner_label,
    p3_corners,
    p3_segments,
)


@dataclass(frozen=True)
class RunDefinition:
    run_id: str
    plant_id: str
    excitation_seed: int
    n_excursions: int
    noise_realisation: int
    stream: tuple[int, int, int, int]


@dataclass
class GeneratedRun:
    definition: RunDefinition
    observations: Observations
    exact: np.ndarray  # exact values at the sensor instants: for the scan only, never written
    expected_switches: list[tuple[float, list[float]]]  # instant, inputs applied from it on
    private: dict[str, object] = field(default_factory=dict)


def define_runs(definition: DatasetDefinitionConfig, plant_ids: list[str]) -> list[RunDefinition]:
    """Every plant under every excitation seed, with the noise stream that follows from
    each identity. Two runs with one identity or one stream are refused here."""
    identities = [
        (
            plant_id,
            seed,
            run_identifier(
                plant_id,
                definition.protocol,
                seed,
                definition.n_excursions,
                definition.noise_realisation,
            ),
        )
        for seed in definition.excitation_seeds
        for plant_id in plant_ids
    ]
    streams = require_distinct_runs(run_id for _, _, run_id in identities)
    return [
        RunDefinition(
            run_id,
            plant_id,
            seed,
            definition.n_excursions,
            definition.noise_realisation,
            streams[run_id],
        )
        for plant_id, seed, run_id in identities
    ]


def generate_run(
    plant: VirtualPlant,
    run: RunDefinition,
    measurement: MeasurementSpec,
    master_seed: int,
    simulation_period: float,
) -> GeneratedRun:
    """Simulate, validate and observe one run of protocol P3. The state is carried from
    segment to segment and never reset. A truth that is not accepted raises."""
    corners = p3_corners(run.n_excursions, run.excitation_seed)
    segments = p3_segments(plant.nominal_inputs, corners)
    trajectory = simulate_piecewise(plant.f, plant.nominal_state, segments, simulation_period)
    observed = observe_trajectory(
        trajectory,
        plant.parameters,
        measurement,
        plant=plant.plant_id,
        run=run.run_id,
        sensor_seed=master_seed,
        sensor_stream=run.stream,
    )
    switches, instant = [], 0.0
    for segment in segments:
        switches.append((instant, [float(value) for value in segment.inputs]))
        instant += segment.duration
    check = observed.truth.check
    private = {
        "plant_id": run.plant_id,
        "excitation_seed": run.excitation_seed,
        "n_excursions": run.n_excursions,
        "noise_realisation": run.noise_realisation,
        "noise_stream": list(run.stream),
        "corners": [corner_label(corner) for corner in corners],
        "true_samples": int(len(trajectory.times)),
        "truth_accepted": bool(check.accepted),
        "refined_peak_temperature_K": check.refined_peak_temperature,
        "min_temperature_K": check.min_temperature,
        "c_a_range_mol_m3": list(check.c_a_range),
        "relative_mass_residual": check.relative_mass_residual,
        "relative_energy_residual": check.relative_energy_residual,
        "content_sha256": observed.observations.content_digest(),
    }
    return GeneratedRun(run, observed.observations, observed.truth.exact, switches, private)


def _same(a: Observations, b: Observations) -> bool:
    return (
        a.content_digest() == b.content_digest()
        and all(
            np.array_equal(getattr(a, f), getattr(b, f)) for f in ("times", "measured", "inputs")
        )
        and (a.measured_names, a.measured_units, a.input_names, a.input_units)
        == (b.measured_names, b.measured_units, b.input_names, b.input_units)
        and (a.sample_period, a.noise_std) == (b.sample_period, b.noise_std)
    )


def _stored_switches(
    rows: list[dict[str, object]], names: tuple[str, ...]
) -> list[tuple[float, list[float]]]:
    """The instants at which the stored inputs change, and the inputs from then on, read
    from the aligned series of the database. The first row counts as the first switch."""
    found, previous = [], None
    for row in rows:
        inputs = [float(row[name]) for name in names]  # type: ignore[arg-type]
        if inputs != previous:
            found.append((float(row["time_s"]), inputs))  # type: ignore[arg-type]
        previous = inputs
    return found


def _size(path: Path) -> int:
    paths = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in paths)


def run_pipeline(definition_path: Path, figures: bool = True) -> dict[str, object]:
    """Run the whole path for one data set definition and return the report, which is
    also written to the private record of the attempt."""
    definition_path = Path(definition_path).resolve()
    report: dict[str, object] = {
        "ok": False,
        "checks": {},
        "stages_s": {},
        "definition": str(definition_path),
    }
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
    stages: dict[str, float] = report["stages_s"]  # type: ignore[assignment]
    started = time.perf_counter()
    state = git_state()
    private_record: dict[str, object] = {"git": state, "environment": environment()}
    configuration_files: list[Path] = [definition_path]
    attempt_directory: Path | None = None
    connection = None

    def stage(name: str, since: float) -> float:
        stages[name] = round(time.perf_counter() - since, 3)
        return time.perf_counter()

    try:
        # 1. configurations
        tick = time.perf_counter()
        definition = load_dataset_definition(definition_path)
        # paths in a definition are relative to the file that holds them
        plant_files = [(definition_path.parent / name).resolve() for name in definition.plants]
        sensors_file = (definition_path.parent / definition.sensors).resolve()
        configuration_files += [*plant_files, sensors_file]
        measurement = MeasurementSpec.from_config(load_sensors(sensors_file))
        simulation_period = definition.simulation_period.si
        report["dataset_id"] = definition.dataset_id
        tick = stage("1_load_configurations", tick)

        # 2. plants and their starting points
        plants = {plant.plant_id: plant for plant in (load_virtual_plant(f) for f in plant_files)}
        checks["starting_points_verified"] = len(plants) == len(plant_files)
        runs = define_runs(definition, list(plants))
        checks["run_identities_and_noise_streams_distinct"] = True  # define_runs raises otherwise
        tick = stage("2_build_plants_and_check_starting_points", tick)

        # 3 to 6. generate, validate, observe and write, one run at a time
        attempt = {
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "commit": state["commit"],
            "code_identified": state["code_identified"],
            "packages": environment()["packages"],
        }
        generated: list[GeneratedRun] = []
        with DatasetWriter(
            definition.dataset_id, [known_plant(plant.spec) for plant in plants.values()], attempt
        ) as writer:
            for run in runs:
                made = generate_run(
                    plants[run.plant_id],
                    run,
                    measurement,
                    definition.sensor_master_seed,
                    simulation_period,
                )
                description = (
                    f"{definition.description} Protocol P3 (D-019): A10 amplitudes, "
                    f"{P3_HOLD:g} s at a corner of the input box, {P3_REST:g} s at the nominal "
                    f"inputs, {run.n_excursions} excursions, excitation seed "
                    f"{run.excitation_seed}, noise realisation {run.noise_realisation}."
                )
                writer.add_run(
                    made.observations, RunRecord(definition.dataset_id, "p3", description)
                )
                generated.append(made)
            published = writer.publish()
        checks["true_trajectories_accepted"] = all(g.private["truth_accepted"] for g in generated)
        report["dataset"] = {"directory": str(published.directory), "status": published.status}
        tick = stage("3_to_6_generate_validate_observe_write_parquet", tick)

        private_record.update(
            {
                "dataset_id": definition.dataset_id,
                "definition": definition.model_dump(mode="json"),
                "sensor_master_seed": definition.sensor_master_seed,
                "noise_generator": "numpy SeedSequence(entropy=master seed, spawn_key=(four words "
                "of the SHA-256 of run_id, index of the state)); no global generator",
                "integration": {
                    "method": "LSODA",
                    "rtol": 1.0e-9,
                    "atol": 1.0e-9,
                    "sample_period_s": simulation_period,
                },
                "sensors": {
                    "sample_period_s": measurement.sample_period,
                    "noise_std_SI": list(measurement.noise_std),
                },
                "amplitudes_A10_SI": {
                    p.plant_id: a10_amplitudes(p.nominal_inputs).tolist() for p in plants.values()
                },
                "plants": {
                    p.plant_id: {
                        "nominal_state_SI": p.nominal_state.tolist(),
                        "eigenvalues_per_s": [[v.real, v.imag] for v in p.eigenvalues],
                        "steady_state_residual": p.residual.tolist(),
                    }
                    for p in plants.values()
                },
                "runs": {g.definition.run_id: g.private for g in generated},
                "publication": published.status,
                "manifest": published.manifest,
            }
        )

        # the Parquet data set, read back
        dataset = open_dataset(definition.dataset_id)
        checks["parquet_round_trip_equal"] = all(
            _same(dataset.observations(g.definition.run_id), g.observations) for g in generated
        )
        tick = stage("6b_read_parquet_back", tick)

        # 7. DuckDB
        database_file = database.database_path(definition.dataset_id)
        connection = database.connect(database_file)
        report["ingestion"] = database.ingest_dataset(connection, dataset)
        checks["duckdb_reconstruction_equal"] = all(
            _same(
                database.observations_from_database(connection, g.definition.run_id), g.observations
            )
            for g in generated
        )
        tick = stage("7_ingest_duckdb", tick)

        # 8. SQL quality checks, on what the database now holds
        quality = database.quality_report(connection)
        report["quality"] = {name: len(rows) for name, rows in quality.items()}
        report["quality_findings"] = database.failed_checks(quality)
        checks["sql_quality_checks_pass"] = not report["quality_findings"]
        input_names = generated[0].observations.input_names
        checks["times_and_input_changes_preserved"] = all(
            _stored_switches(database.aligned_series(connection, g.definition.run_id), input_names)
            == g.expected_switches
            for g in generated
        )
        tick = stage("8_sql_quality_checks", tick)

        before = {
            t: connection.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in database.TABLES
        }
        again = database.ingest_dataset(connection, dataset)
        after = {
            t: connection.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in database.TABLES
        }
        checks["reingestion_changes_nothing"] = (
            set(again.values()) == {"already_present"} and before == after
        )
        report["rows"] = after

        # 9. export
        exported = export_dataset(connection, definition.dataset_id, attempt)
        report["export"] = {"directory": str(exported.directory), "status": exported.status}
        connection.close()
        connection = None
        checks["export_equals_stored_content"] = _export_matches(exported.directory, generated)
        tick = stage("9_export", tick)

        # 10. nothing hidden in what was written
        hidden = HiddenValues(
            parameters=_hidden_parameters(plants),
            states=np.concatenate(
                [g.exact.ravel() for g in generated] + [p.nominal_state for p in plants.values()]
            ),
            integers={"the master seed of the noise": definition.sensor_master_seed}
            | {
                f"a word of the noise stream of {g.definition.run_id}": w
                for g in generated
                for w in g.definition.stream
            },
        )
        scan = scan_available(
            [published.directory, database_file, exported.directory],
            hidden,
            (*database.TABLES, "aligned_series", "lagged_measurements"),
        )
        report["hidden_information_scan"] = {
            "files": len(scan["files_scanned"]),
            "findings": scan["findings"],
        }  # type: ignore[arg-type]
        checks["no_hidden_information_in_the_available_branch"] = not scan["findings"]
        report["bytes"] = {
            "dataset": _size(published.directory),
            "database": _size(database_file),
            "export": _size(exported.directory),
        }
        tick = stage("10_scan_for_hidden_information", tick)
        report["ok"] = (
            all(bool(value) for value in checks.values()) and len(checks) == EXPECTED_CHECKS
        )
        report["runs"] = {g.definition.run_id: g.private["content_sha256"] for g in generated}
        if figures:
            report["figures_from"] = str(exported.directory)
    finally:
        if connection is not None:
            connection.close()
        report["total_s"] = round(time.perf_counter() - started, 3)
        private_record["report"] = report
        attempt_directory = write_private_attempt(
            str(report.get("dataset_id", "unidentified")),
            private_record,
            configuration_files,
            state,
        )
        report["private_record"] = str(attempt_directory)
        (attempt_directory / "pipeline_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    if figures and report["ok"]:
        from process_transfer.generation.figures import figure_readings  # matplotlib only if asked

        figure_readings(Path(str(report["figures_from"])), attempt_directory)
    return report


EXPECTED_CHECKS = 10


def _hidden_parameters(plants: Mapping[str, VirtualPlant]) -> dict[str, float]:
    """The hidden parameters in SI, leaving out a value that a known quantity also has:
    T_ref is 350 K, which is the known feed temperature as well."""
    hidden: dict[str, float] = {}
    for plant in plants.values():
        p, known = plant.parameters, set(plant.nominal_inputs.tolist())
        for name in (
            "k0",
            "activation_temperature",
            "saturation_constant",
            "ua_ref",
            "alpha",
            "t_ref",
        ):
            value = float(getattr(p, name))
            if value != 0.0 and value not in known:
                hidden[f"{name} of {plant.plant_id}"] = value
    return hidden


def _export_matches(directory: Path, generated: list[GeneratedRun]) -> bool:
    """Open the export from the disk, verified, as a model would, and compare its runs with
    what was generated. Only the export directory is read."""
    export = open_export_directory(directory)
    if sorted(export.run_ids) != sorted(g.definition.run_id for g in generated):
        return False
    return all(_same(export.observations(g.definition.run_id), g.observations) for g in generated)
