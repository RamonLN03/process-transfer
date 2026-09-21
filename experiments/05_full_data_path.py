"""M0-E05: the full data path, from configuration files to an export a model could read.

Six runs of protocol P3 (D-019) read by the sensors of D-020: three excitation seeds on
each of the two plants, two hours each. They are generated, validated, observed, written
to Parquet, ingested into DuckDB, checked by SQL and exported, and every step is compared
with the one before it. This is a test of software, of whether the path keeps the data
intact and the truth out. It does not say that these data are enough to train a model,
and it is not a benchmark for M1.

Registration
------------
The definition of the data set, ``configs/datasets/m0_e05.yaml``, the block "Fixed before
the first run" below, and the hypotheses, criteria, expected costs and expected artefacts
of the M0-E05 entry of ``docs/experiment_log.md`` were committed before this script was
run on that definition. Known beforehand: the true P3 sequences of excitation seeds 0, 1
and 2 were accepted on both plants in M0-E03, and seed 0 was observed in M0-E04 under
other noise streams. The noise streams of this data set follow from its run identities
and had not been drawn. The script had been exercised outside the repository on a smaller
definition with other seeds, to debug it. No seed or tolerance is changed after a result,
and a criterion that fails is reported as failing.

Hypotheses
----------
H1  The six true trajectories are accepted, from verified starting points.
H2  The observations are identical before and after Parquet, after reconstruction from
    DuckDB, and in the exported aligned series: arrays bit for bit, and content hashes.
H3  Instants and input changes are preserved: every run has the 20 input settings of its
    protocol at the instants of its protocol, and the row of a change carries the new inputs.
H4  The SQL quality queries find nothing in the valid data set, find every defect put on
    purpose into a separate copy of a run, and do not report valid oddities.
H5  A second generation reproduces the content: the same hashes, and not a file of the
    available branch is touched.
H6  Ingesting again changes nothing, and is reported as such.
H7  The data set can be read, ingested and exported from a place that holds nothing but
    the data set itself: no private branch, no database, no export.
H8  The available branch holds no hidden information in files, tables, metadata or
    exports; and the scan that says so does find a leak when one is planted in a copy.
H9  The noise of the six runs is as specified, and independent between plants and between
    runs: every z-score of the diagnostics of M0-E04 is within +-Z_LIMIT.

Run from the repository root (expected well under a minute):

    python experiments/05_full_data_path.py

The data set, database and export go under PT_DATA_DIR/available, the private record of
each generation under PT_DATA_DIR/private, and the summary and figures of this experiment
under PT_DATA_DIR/experiments/m0_e05/<run id>. The exit code is 0 only if every
hypothesis holds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from process_transfer import __version__
from process_transfer.config import load_dataset_definition, load_sensors
from process_transfer.data import database
from process_transfer.data.parquet_store import datasets_root, open_dataset
from process_transfer.data.paths import data_dir, repository_root
from process_transfer.data.provenance import (
    copy_with_fingerprints,
    environment,
    git_state,
    new_run_directory,
)
from process_transfer.generation.figures import figure_readings
from process_transfer.generation.leak_scan import HiddenValues, scan_available
from process_transfer.generation.pipeline import define_runs, generate_run, run_pipeline
from process_transfer.generation.plants import load_virtual_plant
from process_transfer.measurement.noise_statistics import correlation, noise_statistics
from process_transfer.measurement.sensors import MeasurementSpec

# =========================================================================== #
# Fixed before the first run
# =========================================================================== #

DEFINITION = repository_root() / "configs" / "datasets" / "m0_e05.yaml"
# That file fixes: dataset m0-e05; source and target; protocol P3, 10 excursions, 2 h per
# run; excitation seeds 0, 1 and 2, the same on both plants; noise realisation 0; master
# seed 20260922; the truth stored every 0.1 s and read by the sensors every 6 s.

Z_LIMIT = 4.0
# H9 uses the z-scores of measurement/noise_statistics.py, as M0-E04 did. Twelve series of
# 1201 errors, one per run and variable: mean, spread, lag-one and two tail fractions, 60
# scores. Correlations, 42 scores: each error series with the exact value it was added to
# (12); C_A with T within a run (6); for each excitation seed, each variable of the source
# with each of the target (12); within a plant, the same variable between two of its runs
# (12). 102 scores in all. For correct noise |z| > 4 has probability 6.3e-5, so a correct
# sensor fails this criterion with a probability of about 0.6 %.

# Defects put on purpose into a copy of one run in the staging schema, each with the
# quality query that must report it. ONE stands for one reading of the temperature.
ONE = "WHERE sensor_id = 'target.T' AND sample_index = 100"
DEFECTS = (
    (
        "a reading stored twice",
        "INSERT INTO staging.measurements SELECT * FROM staging.measurements " + ONE,
        "q01_duplicates",
    ),
    (
        "a reading of a channel that does not exist",
        f"UPDATE staging.measurements SET sensor_id = 'target.pH' {ONE}",
        "q02_references",
    ),
    (
        "a reading that uses the channel of the other plant",
        f"UPDATE staging.measurements SET sensor_id = 'source.T' {ONE}",
        "q02_references",
    ),
    (
        "a missing value",
        f"UPDATE staging.measurements SET value = NULL {ONE}",
        "q03_required_fields",
    ),
    (
        "a reading that is not a number",
        f"UPDATE staging.measurements SET value = 'NaN' {ONE}",
        "q04_non_finite",
    ),
    (
        "a known input presented with noise",
        "UPDATE staging.sensors SET noise_model = 'additive_gaussian', noise_std = 0.0 "
        "WHERE sensor_id = 'target.T_c'",
        "q05_metadata",
    ),
    (
        "a channel sampled every minute in a run sampled every six seconds",
        "UPDATE staging.sensors SET sampling_period_s = 60.0 WHERE sensor_id = 'target.T'",
        "q05_metadata",
    ),
    (
        "ten minutes of readings missing",
        "DELETE FROM staging.measurements WHERE sample_index BETWEEN 200 AND 299",
        "q06_cadence",
    ),
    (
        "an instant a quarter of a second late",
        f"UPDATE staging.measurements SET time_s = time_s + 0.25 {ONE}",
        "q06_cadence",
    ),
    (
        "one channel absent at one instant",
        "DELETE FROM staging.measurements WHERE sensor_id = 'target.T_c' AND sample_index = 100",
        "q07_missing_channels",
    ),
)
# Valid records that must not be reported: a negative concentration reading, a reading far
# above any true state, and an instant that differs from its clock in the last bit.
NOT_DEFECTS = (
    "UPDATE staging.measurements SET value = -12.5 "
    "WHERE sensor_id = 'target.C_A' AND sample_index = 100",
    "UPDATE staging.measurements SET value = 1.0e6 " + ONE,
    "UPDATE staging.measurements SET time_s = nextafter(time_s, 'Infinity'::DOUBLE) "
    "WHERE sample_index = 100",
)

# What a reader does when handed nothing but the data set. Run in another process, with
# PT_DATA_DIR pointing at a directory that holds only a copy of the data set.
READER = """
import json, sys
from process_transfer.data import database
from process_transfer.data.export import export_dataset
from process_transfer.data.parquet_store import open_dataset
from process_transfer.data.paths import data_dir

dataset = open_dataset(sys.argv[1])
connection = database.connect(database.database_path("rebuilt"))
statuses = database.ingest_dataset(connection, dataset)
findings = database.failed_checks(database.quality_report(connection))
exported = export_dataset(connection, sys.argv[1], {"by": "a reader without the private branch"})
connection.close()
print(json.dumps({
    "statuses": statuses,
    "quality_findings": {name: len(rows) for name, rows in findings.items()},
    "hashes": {run["run_id"]: run["content_sha256"] for run in exported.manifest["runs"]},
    "private_exists": (data_dir() / "private").exists(),
}))
"""


# =========================================================================== #
# Parts
# =========================================================================== #


def available_snapshot() -> dict[str, int]:
    """Modification times of every file of the available data sets and exports."""
    roots = (datasets_root(), data_dir() / "available" / "exports")
    return {
        str(path): path.stat().st_mtime_ns
        for root in roots
        if root.is_dir()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def part_reader_without_private(dataset_id: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temporary:
        handed_over = Path(temporary) / "available" / "datasets" / dataset_id
        shutil.copytree(datasets_root() / dataset_id, handed_over)
        completed = subprocess.run(
            [sys.executable, "-c", READER, dataset_id],
            capture_output=True,
            text=True,
            check=False,
            env={**__import__("os").environ, "PT_DATA_DIR": temporary},
        )
        if completed.returncode != 0:
            return {"ran": False, "error": completed.stderr.strip().splitlines()[-1:]}
        return {"ran": True, **json.loads(completed.stdout.strip().splitlines()[-1])}


def part_planted_defects(dataset_id: str, run_id: str) -> dict[str, object]:
    dataset = open_dataset(dataset_id)
    connection = database.connect()
    results: dict[str, object] = {}

    def staged_report(statements: tuple[str, ...]) -> dict[str, list]:
        connection.execute("BEGIN TRANSACTION")
        try:
            database.stage_run(connection, dataset, run_id)
            for statement in statements:
                connection.execute(statement)
            return database.failed_checks(database.quality_report(connection, "staging"))
        finally:
            connection.execute("ROLLBACK")  # every defect goes into its own copy

    results["valid_copy_findings"] = {k: len(v) for k, v in staged_report(()).items()}
    found = {}
    for label, statement, query in DEFECTS:
        failed = staged_report((statement,))
        found[label] = {"expected": query, "reported_by": sorted(failed), "found": query in failed}
    results["defects"] = found
    results["valid_oddities_findings"] = {k: len(v) for k, v in staged_report(NOT_DEFECTS).items()}
    connection.close()
    return results


def part_noise(dataset_id: str) -> tuple[dict[str, object], dict[str, float]]:
    """Truth side. The runs are generated again to obtain the exact values, and the stored
    readings are compared with them."""
    definition = load_dataset_definition(DEFINITION)
    measurement = MeasurementSpec.from_config(
        load_sensors((DEFINITION.parent / definition.sensors).resolve())
    )
    plants = {
        plant.plant_id: plant
        for plant in (
            load_virtual_plant((DEFINITION.parent / n).resolve()) for n in definition.plants
        )
    }
    dataset = open_dataset(dataset_id)
    errors: dict[str, np.ndarray] = {}
    exact: dict[str, np.ndarray] = {}
    for run in define_runs(definition, list(plants)):
        made = generate_run(
            plants[run.plant_id],
            run,
            measurement,
            definition.sensor_master_seed,
            definition.simulation_period.si,
        )
        errors[run.run_id] = dataset.observations(run.run_id).measured - made.exact
        exact[run.run_id] = made.exact

    scores: dict[str, float] = {}
    series: dict[str, object] = {}
    variables = measurement.variables
    for run_id, error in errors.items():
        for column, (variable, sigma) in enumerate(
            zip(variables, measurement.noise_std, strict=True)
        ):
            found = noise_statistics(error[:, column], sigma)
            series[f"{run_id} {variable}"] = {
                "n": found.n,
                "mean": found.mean,
                "rms": found.rms,
                "sigma": sigma,
                "beyond_one_sigma": found.beyond_one_sigma,
                "largest_in_sigmas": found.largest_in_sigmas,
                **{f"z_{name}": z for name, z in found.z_scores.items()},
            }
            for name, z in found.z_scores.items():
                scores[f"{run_id} {variable}: {name}"] = z
            scores[f"{run_id} {variable}: error with its exact value"] = correlation(
                error[:, column], exact[run_id][:, column]
            )[1]
        scores[f"{run_id}: {variables[0]} with {variables[1]}"] = correlation(
            error[:, 0], error[:, 1]
        )[1]
    for seed in definition.excitation_seeds:
        source, target = (
            errors[f"{plant}.p3.e{seed}.x{definition.n_excursions}.n{definition.noise_realisation}"]
            for plant in ("source", "target")
        )
        for i, first in enumerate(variables):
            for j, second in enumerate(variables):
                scores[f"e{seed}: source {first} with target {second}"] = correlation(
                    source[:, i], target[:, j]
                )[1]
    for plant in plants:
        ids = sorted(run_id for run_id in errors if run_id.startswith(f"{plant}."))
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                for column, variable in enumerate(variables):
                    scores[f"{variable}: {ids[a]} with {ids[b]}"] = correlation(
                        errors[ids[a]][:, column], errors[ids[b]][:, column]
                    )[1]
    worst = max(scores, key=lambda label: abs(scores[label]))
    outside = {label: z for label, z in scores.items() if abs(z) > Z_LIMIT}
    return (
        {
            "series": series,
            "n_scores": len(scores),
            "largest_abs_z": abs(scores[worst]),
            "largest_abs_z_label": worst,
            "outside_limit": outside,
            "z_limit": Z_LIMIT,
            "mean_of_scores": float(np.mean(list(scores.values()))),
            "rms_of_scores": float(np.sqrt(np.mean(np.square(list(scores.values()))))),
        },
        scores,
    )


def part_planted_leak(dataset_id: str, master_seed: int) -> dict[str, object]:
    """The scan of the pipeline found nothing. That is only worth something if the scan
    finds a leak when there is one, so one of each kind is planted in a copy."""
    hidden = HiddenValues(
        {"alpha of target": 0.002}, np.array([355.16866636550080]), {"the master seed": master_seed}
    )
    source = data_dir() / "available" / "exports" / dataset_id
    found = {}
    with tempfile.TemporaryDirectory() as temporary:
        copy = Path(temporary) / "export"
        shutil.copytree(source, copy)
        found["untouched copy"] = len(scan_available([copy], hidden, ())["findings"])
        manifest = json.loads((copy / "export.json").read_text(encoding="utf-8"))
        (copy / "export.json").write_text(
            json.dumps({**manifest, "sensor_seed": master_seed}), encoding="utf-8"
        )
        found["a seed in the manifest"] = len(scan_available([copy], hidden, ())["findings"])
        (copy / "export.json").write_text(json.dumps(manifest), encoding="utf-8")
        name = next(p for p in sorted(copy.glob("target.*.parquet")))
        table = pq.read_table(name)
        pq.write_table(
            table.append_column("T_exact", pa.array(np.full(table.num_rows, 355.16866636550080))),
            name,
        )
        found["an exact state in a column"] = len(scan_available([copy], hidden, ())["findings"])
    return found


# =========================================================================== #
# Main
# =========================================================================== #


def main() -> int:
    started = time.perf_counter()
    print(f"process_transfer {__version__}; M0-E05 the full data path")
    state = git_state()
    directory = new_run_directory("m0_e05", state)
    if not state["code_identified"]:
        print(f"  NOTE: the code of this run is not fully identified: {state['reason']}")
    definition = load_dataset_definition(DEFINITION)
    dataset_id = definition.dataset_id
    timings: dict[str, float] = {}

    tick = time.perf_counter()
    first = run_pipeline(DEFINITION, figures=False)
    timings["first_generation"] = round(time.perf_counter() - tick, 2)
    before = available_snapshot()
    tick = time.perf_counter()
    second = run_pipeline(DEFINITION, figures=False)
    timings["second_generation"] = round(time.perf_counter() - tick, 2)
    untouched = before == available_snapshot()

    tick = time.perf_counter()
    reader = part_reader_without_private(dataset_id)
    timings["reader_without_private"] = round(time.perf_counter() - tick, 2)
    tick = time.perf_counter()
    target_run = sorted(run_id for run_id in first["runs"] if run_id.startswith("target."))[0]
    defects = part_planted_defects(dataset_id, target_run)
    timings["planted_defects"] = round(time.perf_counter() - tick, 2)
    tick = time.perf_counter()
    noise, _ = part_noise(dataset_id)
    timings["noise_diagnostics"] = round(time.perf_counter() - tick, 2)
    leak = part_planted_leak(dataset_id, definition.sensor_master_seed)

    checks = first["checks"]
    verdicts = {
        "H1_truth_accepted_from_verified_starting_points": bool(
            checks.get("starting_points_verified") and checks.get("true_trajectories_accepted")
        ),
        "H2_observations_identical_through_parquet_duckdb_and_export": bool(
            checks.get("parquet_round_trip_equal")
            and checks.get("duckdb_reconstruction_equal")
            and checks.get("export_equals_stored_content")
        ),
        "H3_instants_and_input_changes_preserved": bool(
            checks.get("times_and_input_changes_preserved")
        ),
        "H4_quality_queries": bool(
            checks.get("sql_quality_checks_pass")
            and not defects["valid_copy_findings"]
            and all(item["found"] for item in defects["defects"].values())
            and not defects["valid_oddities_findings"]
        ),
        "H5_second_generation_reproduces_the_content": bool(
            second["ok"]
            and second["runs"] == first["runs"]
            and second["dataset"]["status"] == second["export"]["status"] == "already_present"
            and untouched
        ),
        "H6_reingestion_changes_nothing": bool(
            checks.get("reingestion_changes_nothing")
            and set(second["ingestion"].values()) == {"already_present"}
            and second["rows"] == first["rows"]
        ),
        "H7_readable_without_the_private_branch": bool(
            reader.get("ran")
            and reader.get("hashes") == first["runs"]
            and not reader.get("quality_findings")
            and reader.get("private_exists") is False
        ),
        "H8_no_hidden_information_and_the_scan_can_find_it": bool(
            checks.get("no_hidden_information_in_the_available_branch")
            and leak["untouched copy"] == 0
            and leak["a seed in the manifest"] > 0
            and leak["an exact state in a column"] > 0
        ),
        "H9_noise_as_specified_and_independent": not noise["outside_limit"],
    }

    figures = figure_readings(data_dir() / "available" / "exports" / dataset_id, directory)
    configurations = [DEFINITION] + [(DEFINITION.parent / n).resolve() for n in definition.plants]
    configurations.append((DEFINITION.parent / definition.sensors).resolve())
    timings["total"] = round(time.perf_counter() - started, 2)
    summary = {
        "provenance": {
            "experiment": "M0-E05",
            "run_id": directory.name,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git": state,
            "configurations": copy_with_fingerprints(configurations, directory / "configs"),
            "environment": environment(),
            "z_limit": Z_LIMIT,
            "contains_hidden_parameters": True,
        },
        "verdicts": verdicts,
        "pipeline_first": first,
        "pipeline_second": {
            k: second[k]
            for k in (
                "ok",
                "checks",
                "dataset",
                "export",
                "ingestion",
                "rows",
                "stages_s",
                "total_s",
            )
        },
        "reader_without_private": reader,
        "planted_defects": defects,
        "planted_leaks": leak,
        "noise": noise,
        "figures": [path.name for path in figures],
        "timings_s": timings,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  rows {first['rows']}")
    print(f"  bytes on disk {first['bytes']}")
    print(f"  pipeline stages, s: {first['stages_s']}")
    print(
        f"  noise: {noise['n_scores']} z-scores, largest |z| = {noise['largest_abs_z']:.2f} "
        f"({noise['largest_abs_z_label']})"
    )
    print("\n== Verdicts ==")
    for label, verdict in verdicts.items():
        print(f"  {label}: {'holds' if verdict else 'FAILS'}")
    print(f"\nrun times, s: {timings}")
    print(f"summary and figures written to {directory}")
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
