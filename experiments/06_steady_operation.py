"""M0-E06: steady operation with noise only (D-010), from the plants to an export.

One run of two hours per plant at the nominal inputs, started at the verified nominal
steady state and read by the sensors of D-020, the noise of the two plants independent
because the noise stream of a run follows from its identity. Nothing is drawn at random
for the inputs: the identity of a run names no seed. The run is generated, validated,
observed, written to Parquet, ingested into DuckDB, checked by SQL and exported, by the
same path as M0-E05, and its readings are compared with the constant states they were
added to.

What this experiment is and is not. It completes the first of the two operating runs
that D-010 promised and that had never been generated. It is a test of the simulator and
of the data path under no excitation: the states must stay where they started, and the
readings must be that constant plus the specified noise. The noise is a hypothesis of the
simulator, D-020; nothing here says that a real sensor behaves like it. Nor does the run
say anything about the plants' dynamics, which it does not excite.

Registration
------------
Fixed before the first run: ``configs/datasets/m0_e06.yaml`` (data set m0-e06; source and
target; protocol steady, 7200 s; noise realisation 0; master seed 20260923; the truth
stored every 0.1 s, the sensors every 6 s) and the criteria below, committed before this
script was run on that definition. Known beforehand: the nominal steady states and their
stability (M0-E01, verified again on every generation); the sensors under P3 (M0-E04,
M0-E05); the data path on short steady runs in the unit tests, with another master seed
and 120 s. The two noise streams of this data set had not been drawn.

Hypotheses
----------
H1  Drift. Started at the nominal steady state, the true states stay there: over the two
    hours, max |x(t) - x(0)| is within DRIFT_FRACTION of the scale of each state on each
    plant, the balances close and the trajectory is accepted.
H2  Path. The ten mandatory checks of the generator pass: acceptance, Parquet round trip,
    DuckDB reconstruction, SQL quality queries, instants and inputs, idempotent
    reingestion, export equal to the stored content, and no hidden information.
H3  Cadence and constancy. Each stored run has the 1201 rows that the definition implies,
    its instants are on the sensor clock of 6 s by the rule of the contract, and every
    stored input row equals the nominal inputs of its plant exactly.
H4  Noise. Every z-score of the diagnostics is within +-Z_LIMIT: for each of the four
    error series, reading minus exact state, of 1201 errors, mean, spread, lag-one and
    two tail fractions (20 scores); the correlation of the C_A errors with the T errors
    within a run (2); and the correlation of each variable of the source with each of the
    target (4). 26 scores; for correct noise at least one exceeds 4 with a probability of
    about 0.16 %. The correlation of an error series with the exact state it was added to
    is not computed here: the exact state of a steady run is constant by design, H1 bounds
    its variation to a millionth of its scale, and the correlation with a constant series
    is undefined. ``noise_statistics.correlation`` refuses it; it is not set to zero.
H5  Separation. The scan of the available branch finds nothing; the export of a run holds
    exactly the ten columns of the contract; the description of a run names no seed; the
    private record holds the master seed and both streams.

Drift tolerance, derived from the scales and the numerical precision and not from a
result. The integrator (LSODA, rtol = atol = 1e-9) keeps the local error of one step below
1e-9 of the state, 2.5e-7 mol/m^3 and 3.5e-7 K, atol being negligible at these values. At
a steady state that is stable with the margin of D-017 the error of one step is damped by
the following ones, with time constants of about a minute, so the global error is of the
order of a few local errors and not of their sum over the 72 000 steps. A margin of one
thousand over the local tolerance gives DRIFT_FRACTION = 1e-6 of the scale of each state:
2.5e-4 mol/m^3 and 3.5e-4 K. That is 1/20 000 of sigma_CA, 1/1400 of sigma_T, and about
1/150 and 1/14 of the recovery tolerances of P3. The residual of the polished root, below
1e-9 of the feed terms (generation/plants.py) and in practice near 1e-16 of them, could
move the state by at most residual / |lambda|, about 5e-7 mol/m^3 even at the allowed
limit, which the tolerance covers. Exact equality of every integrated state with the root
is not asked for: the integrator has no reason to reproduce the root bit for bit.

Run from the repository root (expected under a minute):

    python experiments/06_steady_operation.py

The data set, database and export go under PT_DATA_DIR/available, the private record of
the generation under PT_DATA_DIR/private, and the summary and figures of this experiment
under PT_DATA_DIR/experiments/m0_e06/<run id>, which hold exact states and errors and are
diagnostic artefacts. The exit code is 0 only if every hypothesis holds.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from process_transfer import __version__  # noqa: E402
from process_transfer.config import load_dataset_definition, load_sensors  # noqa: E402
from process_transfer.data.export import open_export_directory  # noqa: E402
from process_transfer.data.parquet_store import open_dataset  # noqa: E402
from process_transfer.data.paths import data_dir, repository_root  # noqa: E402
from process_transfer.data.provenance import (  # noqa: E402
    copy_with_fingerprints,
    environment,
    git_state,
    new_run_directory,
)
from process_transfer.generation.figures import figure_readings  # noqa: E402
from process_transfer.generation.pipeline import (  # noqa: E402
    define_runs,
    generate_run,
    run_pipeline,
)
from process_transfer.generation.plants import VirtualPlant, load_virtual_plant  # noqa: E402
from process_transfer.measurement.noise_statistics import (  # noqa: E402
    correlation,
    noise_statistics,
)
from process_transfer.measurement.sensors import MeasurementSpec  # noqa: E402
from process_transfer.sampling_clock import nearest_ticks  # noqa: E402
from process_transfer.simulation.checks import check_trajectory  # noqa: E402
from process_transfer.simulation.integration import simulate_piecewise  # noqa: E402
from process_transfer.simulation.protocols import steady_segments  # noqa: E402

# =========================================================================== #
# Fixed before the first run
# =========================================================================== #

DEFINITION = repository_root() / "configs" / "datasets" / "m0_e06.yaml"
DRIFT_FRACTION = 1.0e-6  # of the scale of each state; derivation in the docstring
Z_LIMIT = 4.0
STATE_NAMES = ("C_A", "T")
EXPORT_COLUMNS = [
    "plant_id",
    "run_id",
    "sample_index",
    "time_s",
    "C_A",
    "T",
    "q",
    "C_Af",
    "T_f",
    "T_c",
]
CORRELATION_WITH_EXACT_STATE = (
    "not computed: the exact state of a steady run is constant by design and the "
    "correlation with a constant series is undefined; it is neither set to zero nor "
    "computed with an epsilon"
)

INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PLANT_COLOR = {"source": "#2a78d6", "target": "#eb6834"}


# =========================================================================== #
# Parts
# =========================================================================== #


def load_plants(definition) -> dict[str, VirtualPlant]:  # noqa: ANN001
    return {
        plant.plant_id: plant
        for plant in (
            load_virtual_plant((DEFINITION.parent / name).resolve()) for name in definition.plants
        )
    }


def part_truth(
    plants: dict[str, VirtualPlant], duration: float, period: float
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Truth side: the two hours at the nominal inputs, from the nominal steady state."""
    results: dict[str, object] = {}
    deviations: dict[str, np.ndarray] = {}
    for plant_id, plant in plants.items():
        segments = steady_segments(plant.nominal_inputs, duration)
        trajectory = simulate_piecewise(plant.f, plant.nominal_state, segments, period)
        check = check_trajectory(trajectory, plant.parameters)
        deviation = trajectory.states - plant.nominal_state
        drift = np.max(np.abs(deviation), axis=0)
        tolerance = DRIFT_FRACTION * np.abs(plant.nominal_state)
        deviations[plant_id] = np.column_stack([trajectory.times, deviation])
        results[plant_id] = {
            "n_samples": int(len(trajectory.times)),
            "rhs_evaluations": int(trajectory.n_rhs_evaluations),
            "nominal_state": plant.nominal_state.tolist(),
            "residual_at_start": plant.residual.tolist(),
            "drift": dict(zip(STATE_NAMES, drift.tolist(), strict=True)),
            "final_deviation": dict(zip(STATE_NAMES, deviation[-1].tolist(), strict=True)),
            "tolerance": dict(zip(STATE_NAMES, tolerance.tolist(), strict=True)),
            "within_tolerance": bool(np.all(drift <= tolerance)),
            "accepted": bool(check.accepted),
            "refined_peak_temperature_K": check.refined_peak_temperature,
            "min_temperature_K": check.min_temperature,
            "relative_mass_residual": check.relative_mass_residual,
            "relative_energy_residual": check.relative_energy_residual,
        }
    return results, deviations


def part_stored(
    dataset_id: str, plants: dict[str, VirtualPlant], duration_s: int, sample_period: float
) -> dict[str, object]:
    """Available side: cadence, constancy of the inputs and the columns of the export.
    The number of rows follows from the definition: one reading per sensor period over
    the duration, both ends included; the instants must be on the sensor clock by the
    rule of the contract (``sampling_clock``), not equal to a product bit for bit."""
    dataset = open_dataset(dataset_id)
    export = open_export_directory(data_dir() / "available" / "exports" / dataset_id)
    periods = duration_s / sample_period
    expected_rows = int(periods) + 1 if float(periods).is_integer() else None
    descriptions = {
        row["run_id"]: str(row["description"])
        for row in dataset.table("operating_runs").to_pylist()
    }
    findings: dict[str, object] = {"runs": {}, "expected_rows": expected_rows}
    for run_id in dataset.run_ids:
        observations = dataset.observations(run_id)
        plant = plants[observations.plant]
        n = observations.n_samples
        ticks, on_clock = nearest_ticks(observations.times, 0.0, sample_period)
        findings["runs"][run_id] = {  # type: ignore[index]
            "n_samples": n,
            "instants_on_the_sensor_clock": bool(
                np.all(on_clock) and np.array_equal(ticks, np.arange(n))
            ),
            "inputs_equal_nominal": bool(np.all(observations.inputs == plant.nominal_inputs)),
            "export_columns": export.table(run_id).schema.names,
            "description_names_no_seed": "seed" not in descriptions[run_id].lower(),
        }
    findings["ok"] = all(
        r["n_samples"] == expected_rows
        and r["instants_on_the_sensor_clock"]
        and r["inputs_equal_nominal"]
        and r["export_columns"] == EXPORT_COLUMNS
        and r["description_names_no_seed"]
        for r in findings["runs"].values()  # type: ignore[union-attr]
    )
    return findings


def part_noise(
    definition,  # noqa: ANN001
    measurement: MeasurementSpec,
    plants: dict[str, VirtualPlant],
) -> tuple[dict[str, object], dict[str, float]]:
    """Truth side: the runs generated again for their exact values, and the stored
    readings compared with them. The exact state is constant here, see H4."""
    dataset = open_dataset(definition.dataset_id)
    errors: dict[str, np.ndarray] = {}
    constancy: dict[str, dict[str, float]] = {}
    for run in define_runs(definition, list(plants)):
        made = generate_run(
            plants[run.plant_id],
            run,
            measurement,
            definition.sensor_master_seed,
            definition.simulation_period.si,
        )
        errors[run.run_id] = dataset.observations(run.run_id).measured - made.exact
        spread = np.max(np.abs(made.exact - plants[run.plant_id].nominal_state), axis=0)
        constancy[run.run_id] = dict(zip(STATE_NAMES, spread.tolist(), strict=True))

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
        scores[f"{run_id}: {variables[0]} with {variables[1]}"] = correlation(
            error[:, 0], error[:, 1]
        )[1]
    source, target = (errors[run_id] for run_id in sorted(errors))  # source before target
    for i, first in enumerate(variables):
        for j, second in enumerate(variables):
            scores[f"source {first} with target {second}"] = correlation(
                source[:, i], target[:, j]
            )[1]
    worst = max(scores, key=lambda label: abs(scores[label]))
    outside = {label: z for label, z in scores.items() if abs(z) > Z_LIMIT}
    return (
        {
            "series": series,
            "exact_state_spread_on_the_sensor_grid": constancy,
            "correlation_with_the_exact_state": CORRELATION_WITH_EXACT_STATE,
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


# =========================================================================== #
# Figures
# =========================================================================== #


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_drift(
    deviations: dict[str, np.ndarray], truth: dict[str, object], directory: Path
) -> Path:
    """|x(t) - x(0)| of the true states on a logarithmic axis, against the tolerance.
    Deviations that are exactly zero cannot be drawn on that axis and are left out."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), facecolor=SURFACE)
    units = ("mol/m^3", "K")
    for column, ax in enumerate(axes):
        style(ax)
        for plant_id, table in deviations.items():
            minutes, values = table[:, 0] / 60.0, np.abs(table[:, 1 + column])
            drawn = values > 0.0
            ax.plot(
                minutes[drawn],
                values[drawn],
                color=PLANT_COLOR[plant_id],
                linewidth=0.8,
                label=plant_id,
            )
            tolerance = truth[plant_id]["tolerance"][STATE_NAMES[column]]  # type: ignore[index]
            ax.axhline(tolerance, color=PLANT_COLOR[plant_id], linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlabel("process time, min", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel(
            f"|{STATE_NAMES[column]}(t) - {STATE_NAMES[column]}(0)|, {units[column]}",
            fontsize=9,
            color=INK_SECONDARY,
        )
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "M0-E06: drift of the true states over two hours at the nominal inputs; dotted lines "
        "are the tolerances, 1e-6 of each state; exact zeros are not drawn",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = directory / "fig1_drift_of_the_true_states.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# =========================================================================== #
# Main
# =========================================================================== #


def main() -> int:
    started = time.perf_counter()
    print(f"process_transfer {__version__}; M0-E06 steady operation with noise only")
    state = git_state()
    directory = new_run_directory("m0_e06", state)
    if not state["code_identified"]:
        print(f"  NOTE: the code of this run is not fully identified: {state['reason']}")
    definition = load_dataset_definition(DEFINITION)
    measurement = MeasurementSpec.from_config(
        load_sensors((DEFINITION.parent / definition.sensors).resolve())
    )
    plants = load_plants(definition)
    timings: dict[str, float] = {}

    tick = time.perf_counter()
    truth, deviations = part_truth(
        plants, float(definition.duration_s), definition.simulation_period.si
    )
    timings["truth"] = round(time.perf_counter() - tick, 2)
    tick = time.perf_counter()
    report = run_pipeline(DEFINITION, figures=False)
    timings["pipeline"] = round(time.perf_counter() - tick, 2)
    tick = time.perf_counter()
    stored = part_stored(
        definition.dataset_id, plants, definition.duration_s, measurement.sample_period
    )
    noise, _ = part_noise(definition, measurement, plants)
    timings["stored_and_noise"] = round(time.perf_counter() - tick, 2)

    checks = report["checks"]
    verdicts = {
        "H1_states_stay_at_the_steady_state_and_are_accepted": all(
            r["within_tolerance"] and r["accepted"]  # type: ignore[index]
            for r in truth.values()
        ),
        "H2_the_ten_checks_of_the_path_pass": bool(report["ok"]),
        "H3_cadence_and_constant_inputs": bool(stored["ok"]),
        "H4_noise_as_specified_and_independent_between_plants": not noise["outside_limit"],
        "H5_separation_of_truth_and_observations": bool(
            checks.get("no_hidden_information_in_the_available_branch")
            and stored["ok"]
            and report["hidden_information_scan"]["findings"] == []
        ),
    }

    figures = [figure_drift(deviations, truth, directory)]
    figures += figure_readings(
        data_dir() / "available" / "exports" / definition.dataset_id, directory
    )
    configurations = [DEFINITION] + [(DEFINITION.parent / n).resolve() for n in definition.plants]
    configurations.append((DEFINITION.parent / definition.sensors).resolve())
    timings["total"] = round(time.perf_counter() - started, 2)
    summary = {
        "provenance": {
            "experiment": "M0-E06",
            "run_id": directory.name,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git": state,
            "configurations": copy_with_fingerprints(configurations, directory / "configs"),
            "environment": environment(),
            "drift_fraction": DRIFT_FRACTION,
            "z_limit": Z_LIMIT,
            "contains_hidden_parameters": True,
        },
        "verdicts": verdicts,
        "truth": truth,
        "pipeline": report,
        "stored": stored,
        "noise": noise,
        "figures": [path.name for path in figures],
        "timings_s": timings,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    for plant_id, result in truth.items():
        print(
            f"  {plant_id}: drift {result['drift']}, tolerance {result['tolerance']}, "  # type: ignore[index]
            f"accepted {result['accepted']}"  # type: ignore[index]
        )
    print(f"  rows {report['rows']}, bytes on disk {report['bytes']}")
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
