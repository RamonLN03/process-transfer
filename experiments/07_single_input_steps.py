"""M0-E07: single-input step tests with ten-minute holds (D-010), verified on the truth
side first, then generated, stored and exported.

Eight independent tests per plant: for each input, q, C_Af, T_f and T_c, and each
direction, 600 s at the nominal inputs, 600 s with that input moved by its A10 amplitude,
600 s at the nominal inputs. Each test starts at the verified nominal steady state, carries
its state across its three segments, and is never chained to another test: the transitions
between tests have not been studied, and the safety of P3 is not extrapolated to these
holds. What a hold of 600 s and the return from it do to the plants is measured here.

Part 1, truth. The sixteen true trajectories are simulated and checked for acceptance,
finite, physical, inside [335, 380] K over lead, hold and recovery, balances closed, with
their thermal extremes, the response to the step, the state reached at the end of the hold
against the steady state of the stepped inputs, the extreme of the return, and the recovery
after the final 600 s. If any test is not accepted, the diagnostics are written, nothing is
observed or published, and the exit code is 1.

Part 2, only after the acceptance of all sixteen: the generator on the definition, the
stored data read back, and the noise diagnostics.

Registration
------------
Fixed before the first run: ``configs/datasets/m0_e07.yaml`` (data set m0-e07; source and
target; protocol step; lead, hold and recovery of 600 s; the four inputs in both
directions; noise realisation 0; master seed 20260924; the truth stored every 0.1 s, the
sensors every 6 s) and the criteria below, committed before this script was run on that
definition. Known beforehand: single-input steps at the A10 amplitudes from the nominal
steady state stayed inside the envelope for 40 min in M0-E02, which is how A10 was chosen
(D-018); the return from the stepped state to the nominal inputs had not been simulated;
the data path on short step tests in the unit tests, 60 s segments with another master
seed; and one smoke run of this script outside the repository, with segments of 120 s and
another master seed, to debug it. The sixteen noise streams of ``m0-e07`` follow from the
identities under master seed 20260924 and had not been drawn.

Hypotheses
----------
H1  Acceptance. All sixteen true trajectories are accepted: finite, physical, inside the
    envelope over lead, hold and recovery, balances closed. Expected from M0-E02 for the
    step itself; the return step is the part not examined before.
H2  Path. The ten mandatory checks of the generator pass.
H3  Switching instants and cadence. In every stored run the inputs change exactly at
    600 s and at 1200 s, to the settings of the test and back to nominal, and nowhere
    else; each run has 301 rows on the 6 s clock.
H4  Noise. Every z-score is within +-Z_LIMIT: for each of the 32 error series of 301
    errors, mean, spread, lag-one and two tails (160 scores); each error series with the
    exact state it was added to (32); C_A with T within a run (16); for each test, each
    variable of the source with each of the target (32). 240 scores; for correct noise at
    least one exceeds 4 with a probability of about 1.5 %.
H5  Separation. The scan of the available branch finds nothing and no description names
    a seed.

Reported and not hypotheses, because no rule of sign or size is imposed on them: the
refined peak and the minimum temperature of every test with their times and margins to the
envelope; the initial response, the sign of dC_A/dt and dT/dt at the step from the
right-hand side at the state reached under the stepped inputs; the extreme of each state
during the hold, its time, and whether it lies inside the hold, a non-monotone response,
or at its end; the state at the end of the hold against the steady state of the stepped
inputs; the extreme of the return; and the recovery, |x(1800 s) - x_nominal|, set against
the recovery tolerances of P3 for reference only, not as a criterion.

Run from the repository root (expected under two minutes):

    python experiments/07_single_input_steps.py

The data set, database and export go under PT_DATA_DIR/available, the private record of
the generation under PT_DATA_DIR/private, and the summary and figures of this experiment
under PT_DATA_DIR/experiments/m0_e07/<run id>, which hold exact states and errors and are
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
from process_transfer.data.identifiers import run_definition  # noqa: E402
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
from process_transfer.simulation import cstr_true  # noqa: E402
from process_transfer.simulation.checks import TEMPERATURE_ENVELOPE, check_trajectory  # noqa: E402
from process_transfer.simulation.integration import simulate_piecewise  # noqa: E402
from process_transfer.simulation.protocols import (  # noqa: E402
    P3_RECOVERY_TOLERANCE_CA,
    P3_RECOVERY_TOLERANCE_T,
    STEP_DIRECTIONS,
    STEP_INPUTS,
    single_step_segments,
)
from process_transfer.simulation.steady_state import find_steady_states  # noqa: E402

# =========================================================================== #
# Fixed before the first run
# =========================================================================== #

DEFINITION = repository_root() / "configs" / "datasets" / "m0_e07.yaml"
Z_LIMIT = 4.0
STATE_NAMES = ("C_A", "T")
INPUT_LABELS = {"q": "q", "caf": "C_Af", "tf": "T_f", "tc": "T_c"}

INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PLANT_COLOR = {"source": "#2a78d6", "target": "#eb6834"}
LIMIT_COLOR = "#d03b3b"


# =========================================================================== #
# Part 1: the truth of the sixteen tests
# =========================================================================== #


def load_plants(definition) -> dict[str, VirtualPlant]:  # noqa: ANN001
    return {
        plant.plant_id: plant
        for plant in (
            load_virtual_plant((DEFINITION.parent / name).resolve()) for name in definition.plants
        )
    }


def _first_move(values: np.ndarray) -> float:
    """The direction in which a state first moves after the step: the sign of its first
    sampled change, 0.1 s after the switching instant. The derivative at the instant
    itself is reported separately and is not used here: for a state on which the stepped
    input has no direct effect it is the residual of the steady state, of the order of
    1e-15, whose sign says nothing. Should the first sample not move at all, the direction
    is that of the largest deviation from the starting value over the segment."""
    first = float(np.sign(values[1] - values[0]))
    if first != 0.0:
        return first
    deviation = values - values[0]
    return float(np.sign(deviation[int(np.argmax(np.abs(deviation)))]))


def _extreme(times: np.ndarray, values: np.ndarray, direction: float) -> dict[str, object]:
    """The extreme of ``values`` in the direction of the first move, its time, and whether
    it lies strictly inside the segment, a non-monotone response, or at its end."""
    index = int(np.argmax(values)) if direction > 0.0 else int(np.argmin(values))
    return {
        "direction_of_the_first_move": direction,
        "value": float(values[index]),
        "time_s": float(times[index]),
        "inside_the_segment": bool(0 < index < len(values) - 1),
        "value_at_the_end": float(values[-1]),
        "beyond_the_end": float(abs(values[index] - values[-1])),
    }


def test_truth(plant: VirtualPlant, input_name: str, direction: str, durations, period: float):  # noqa: ANN001
    lead, hold, recovery = durations
    segments = single_step_segments(
        plant.nominal_inputs, input_name, direction, lead, hold, recovery
    )
    trajectory = simulate_piecewise(plant.f, plant.nominal_state, segments, period)
    check = check_trajectory(trajectory, plant.parameters)
    stepped_inputs = segments[1].inputs
    at_step = trajectory.segments[1].states[0]  # the state at the switching instant
    slope = cstr_true.rhs(0.0, at_step, stepped_inputs, plant.parameters)
    hold_segment, back_segment = trajectory.segments[1], trajectory.segments[2]
    peak, peak_time = trajectory.refined_peak(1)
    lowest, lowest_time = (
        float(np.min(trajectory.states[:, 1])),
        float(trajectory.times[int(np.argmin(trajectory.states[:, 1]))]),
    )
    stepped = find_steady_states(
        lambda x: cstr_true.rhs(0.0, x, stepped_inputs, plant.parameters),
        c_a_upper=float(max(stepped_inputs[1], plant.nominal_inputs[1])),
    )
    end_of_hold = hold_segment.states[-1]
    stepped_state = None
    if len(stepped) == 1:
        (steady,) = stepped
        stepped_state = {
            "state": steady.state.tolist(),
            "max_real_part_per_min": steady.max_real_part * 60.0,
            "slowest_time_constant_s": (
                -1.0 / steady.max_real_part if steady.max_real_part < 0.0 else None
            ),
            "distance_at_the_end_of_the_hold": dict(
                zip(STATE_NAMES, np.abs(end_of_hold - steady.state).tolist(), strict=True)
            ),
        }
    final = trajectory.states[-1]
    residual = np.abs(final - plant.nominal_state)
    return {
        "input": input_name,
        "direction": direction,
        "stepped_inputs": stepped_inputs.tolist(),
        "accepted": bool(check.accepted),
        "values_finite": check.values_finite,
        "states_physical": check.states_physical,
        "inside_envelope": check.inside_envelope,
        "balances_close": check.balances_close,
        "relative_mass_residual": check.relative_mass_residual,
        "relative_energy_residual": check.relative_energy_residual,
        "seconds_above_limit": check.seconds_above_limit,
        "refined_peak_temperature_K": peak,
        "peak_time_s": peak_time,
        "margin_to_upper_limit_K": TEMPERATURE_ENVELOPE[1] - peak,
        "min_temperature_K": lowest,
        "min_time_s": lowest_time,
        "margin_to_lower_limit_K": lowest - TEMPERATURE_ENVELOPE[0],
        "c_a_range_mol_m3": list(check.c_a_range),
        "initial_response": {
            "dC_A_dt_mol_m3_s": float(slope[0]),
            "dT_dt_K_s": float(slope[1]),
        },
        "during_the_hold": {
            name: _extreme(
                hold_segment.times,
                hold_segment.states[:, k],
                _first_move(hold_segment.states[:, k]),
            )
            for k, name in enumerate(STATE_NAMES)
        },
        "steady_states_of_the_stepped_inputs": len(stepped),
        "stepped_steady_state": stepped_state,
        "return": {
            name: {
                "max": float(np.max(back_segment.states[:, k])),
                "min": float(np.min(back_segment.states[:, k])),
            }
            for k, name in enumerate(STATE_NAMES)
        },
        "recovery_after_the_final_segment": {
            "C_A_mol_m3": float(residual[0]),
            "T_K": float(residual[1]),
            "within_p3_tolerances_for_reference": bool(
                residual[0] <= P3_RECOVERY_TOLERANCE_CA and residual[1] <= P3_RECOVERY_TOLERANCE_T
            ),
        },
        "n_samples": int(len(trajectory.times)),
        "rhs_evaluations": int(trajectory.n_rhs_evaluations),
    }, trajectory


def part_truth(plants, durations, period):  # noqa: ANN001
    results: dict[str, dict[str, object]] = {}
    curves: dict[str, dict[str, np.ndarray]] = {}
    for plant_id, plant in plants.items():
        results[plant_id] = {}
        curves[plant_id] = {}
        for input_name in STEP_INPUTS:
            for direction in STEP_DIRECTIONS:
                found, trajectory = test_truth(plant, input_name, direction, durations, period)
                results[plant_id][f"{input_name}-{direction}"] = found
                curves[plant_id][f"{input_name}-{direction}"] = np.column_stack(
                    [trajectory.times, trajectory.states]
                )[::10]  # every second is enough for a figure
    return results, curves


# =========================================================================== #
# Part 2: the stored data and the noise
# =========================================================================== #


def part_stored(definition, plants, sample_period: float) -> dict[str, object]:  # noqa: ANN001
    dataset = open_dataset(definition.dataset_id)
    lead, hold, recovery = definition.durations_s
    total = lead + hold + recovery
    periods = total / sample_period
    expected_rows = int(periods) + 1 if float(periods).is_integer() else None
    descriptions = {
        row["run_id"]: str(row["description"])
        for row in dataset.table("operating_runs").to_pylist()
    }
    findings: dict[str, object] = {"runs": {}, "expected_rows": expected_rows}
    for run in define_runs(definition, list(plants)):
        observations = dataset.observations(run.run_id)
        plant = plants[run.plant_id]
        n = observations.n_samples
        ticks, on_clock = nearest_ticks(observations.times, 0.0, sample_period)
        expected = single_step_segments(
            plant.nominal_inputs,
            str(run.settings["input"]),
            str(run.settings["direction"]),
            lead,
            hold,
            recovery,
        )
        changes = [
            int(k)
            for k in range(1, n)
            if not np.array_equal(observations.inputs[k], observations.inputs[k - 1])
        ]
        stepped = observations.inputs[int(lead / sample_period)] if changes else None
        findings["runs"][run.run_id] = {  # type: ignore[index]
            "n_samples": n,
            "instants_on_the_sensor_clock": bool(
                np.all(on_clock) and np.array_equal(ticks, np.arange(n))
            ),
            "changes_at_s": [float(observations.times[k]) for k in changes],
            "changes_as_defined": bool(
                changes == [int(lead / sample_period), int((lead + hold) / sample_period)]
                and stepped is not None
                and np.array_equal(stepped, expected[1].inputs)
                and np.array_equal(observations.inputs[0], plant.nominal_inputs)
                and np.array_equal(observations.inputs[-1], plant.nominal_inputs)
            ),
            "description_names_no_seed": "seed" not in descriptions[run.run_id].lower(),
        }
    findings["ok"] = all(
        r["n_samples"] == expected_rows
        and r["instants_on_the_sensor_clock"]
        and r["changes_as_defined"]
        and r["description_names_no_seed"]
        for r in findings["runs"].values()  # type: ignore[union-attr]
    )
    return findings


def part_noise(definition, measurement, plants):  # noqa: ANN001
    dataset = open_dataset(definition.dataset_id)
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
    by_definition: dict[str, dict[str, np.ndarray]] = {}
    for run_id, error in errors.items():
        by_definition.setdefault(run_definition(run_id), {})[run_id.split(".", 1)[0]] = error
    for definition_part, plants_errors in by_definition.items():
        source, target = plants_errors["source"], plants_errors["target"]
        for i, first in enumerate(variables):
            for j, second in enumerate(variables):
                scores[f"{definition_part}: source {first} with target {second}"] = correlation(
                    source[:, i], target[:, j]
                )[1]
    worst = max(scores, key=lambda label: abs(scores[label]))
    outside = {label: z for label, z in scores.items() if abs(z) > Z_LIMIT}
    return {
        "series": series,
        "n_scores": len(scores),
        "largest_abs_z": abs(scores[worst]),
        "largest_abs_z_label": worst,
        "outside_limit": outside,
        "z_limit": Z_LIMIT,
        "mean_of_scores": float(np.mean(list(scores.values()))),
        "rms_of_scores": float(np.sqrt(np.mean(np.square(list(scores.values()))))),
    }


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


def figure_responses(curves, plants, durations, directory: Path) -> list[Path]:  # noqa: ANN001
    """One figure per plant: a row per input, C_A on the left and T on the right, the true
    response to the step up (solid) and down (dashed) against process time, the three
    phases separated by vertical lines. Truth side: exact states, no readings."""
    lead, hold, _ = durations
    written = []
    for plant_id, tests in curves.items():
        fig, axes = plt.subplots(4, 2, figsize=(11.0, 11.0), sharex=True, facecolor=SURFACE)
        for row, input_name in enumerate(STEP_INPUTS):
            for column, (state, unit) in enumerate(zip(STATE_NAMES, ("mol/m^3", "K"), strict=True)):
                ax = axes[row, column]
                style(ax)
                for direction, linestyle in (("up", "-"), ("down", "--")):
                    table = tests[f"{input_name}-{direction}"]
                    ax.plot(
                        table[:, 0] / 60.0,
                        table[:, 1 + column],
                        color=PLANT_COLOR[plant_id],
                        linestyle=linestyle,
                        linewidth=1.1,
                        label=f"{INPUT_LABELS[input_name]} {direction}",
                    )
                for instant in (lead, lead + hold):
                    ax.axvline(instant / 60.0, color=AXIS, linewidth=0.8)
                nominal = plants[plant_id].nominal_state[column]
                ax.axhline(nominal, color=INK_SECONDARY, linewidth=0.6, linestyle=":")
                if column == 1:
                    ax.axhline(TEMPERATURE_ENVELOPE[1], color=LIMIT_COLOR, linewidth=0.8)
                    ax.text(
                        0.2,
                        TEMPERATURE_ENVELOPE[1],
                        "380 K limit",
                        color=LIMIT_COLOR,
                        fontsize=7,
                        va="bottom",
                    )
                ax.set_ylabel(f"{state}, {unit}", fontsize=9, color=INK_SECONDARY)
                ax.legend(frameon=False, fontsize=8, loc="best")
                if row == 3:
                    ax.set_xlabel("process time, min", fontsize=9, color=INK_SECONDARY)
        fig.suptitle(
            f"M0-E07, {plant_id}: true response to single-input steps of A10 amplitude; lead, "
            "hold and recovery of 10 min; dotted: nominal steady state; exact states, truth side",
            fontsize=10,
            color=INK,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = directory / f"fig1_true_responses_{plant_id}.png"
        fig.savefig(path, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)
    return written


# =========================================================================== #
# Main
# =========================================================================== #


def main() -> int:
    started = time.perf_counter()
    print(f"process_transfer {__version__}; M0-E07 single-input step tests")
    state = git_state()
    directory = new_run_directory("m0_e07", state)
    if not state["code_identified"]:
        print(f"  NOTE: the code of this run is not fully identified: {state['reason']}")
    definition = load_dataset_definition(DEFINITION)
    measurement = MeasurementSpec.from_config(
        load_sensors((DEFINITION.parent / definition.sensors).resolve())
    )
    plants = load_plants(definition)
    durations = tuple(float(value) for value in definition.durations_s)
    period = definition.simulation_period.si
    timings: dict[str, float] = {}
    configurations = [DEFINITION] + [(DEFINITION.parent / n).resolve() for n in definition.plants]
    configurations.append((DEFINITION.parent / definition.sensors).resolve())
    summary: dict[str, object] = {
        "provenance": {
            "experiment": "M0-E07",
            "run_id": directory.name,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git": state,
            "configurations": copy_with_fingerprints(configurations, directory / "configs"),
            "environment": environment(),
            "z_limit": Z_LIMIT,
            "p3_recovery_tolerances_for_reference": {
                "C_A_mol_m3": P3_RECOVERY_TOLERANCE_CA,
                "T_K": P3_RECOVERY_TOLERANCE_T,
            },
            "contains_hidden_parameters": True,
        }
    }

    tick = time.perf_counter()
    truth, curves = part_truth(plants, durations, period)
    timings["truth"] = round(time.perf_counter() - tick, 2)
    summary["truth"] = truth
    figures = figure_responses(curves, plants, durations, directory)
    accepted = all(test["accepted"] for tests in truth.values() for test in tests.values())
    for plant_id, tests in truth.items():
        hottest = max(tests.values(), key=lambda t: t["refined_peak_temperature_K"])
        coldest = min(tests.values(), key=lambda t: t["min_temperature_K"])
        n_accepted = sum(t["accepted"] for t in tests.values())
        print(
            f"  {plant_id}: {n_accepted} of {len(tests)} accepted; hottest "
            f"{hottest['input']}-{hottest['direction']} at "
            f"{hottest['refined_peak_temperature_K']:.2f} K, coldest "
            f"{coldest['input']}-{coldest['direction']} at {coldest['min_temperature_K']:.2f} K"
        )
    verdicts: dict[str, bool] = {"H1_all_sixteen_true_trajectories_accepted": accepted}
    # The diagnostics of the truth are written now, so that a failure later cannot lose them.
    summary.update({"verdicts": verdicts, "figures": [path.name for path in figures]})
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    if accepted:
        tick = time.perf_counter()
        report = run_pipeline(DEFINITION, figures=False)
        timings["pipeline"] = round(time.perf_counter() - tick, 2)
        tick = time.perf_counter()
        stored = part_stored(definition, plants, measurement.sample_period)
        noise = part_noise(definition, measurement, plants)
        timings["stored_and_noise"] = round(time.perf_counter() - tick, 2)
        figures += figure_readings(
            data_dir() / "available" / "exports" / definition.dataset_id, directory
        )
        verdicts.update(
            {
                "H2_the_ten_checks_of_the_path_pass": bool(report["ok"]),
                "H3_switching_instants_and_cadence": bool(stored["ok"]),
                "H4_noise_as_specified_and_independent": not noise["outside_limit"],
                "H5_separation_of_truth_and_observations": bool(
                    report["hidden_information_scan"]["findings"] == [] and stored["ok"]
                ),
            }
        )
        summary.update({"pipeline": report, "stored": stored, "noise": noise})
        print(f"  rows {report['rows']}, bytes on disk {report['bytes']}")
        print(
            f"  noise: {noise['n_scores']} z-scores, largest |z| = {noise['largest_abs_z']:.2f} "
            f"({noise['largest_abs_z_label']})"
        )
    else:
        summary["not_published"] = (
            "at least one true trajectory was not accepted; the diagnostics above are kept "
            "and no data set, database or export was written"
        )
        print("  NOT PUBLISHED: a true trajectory was not accepted; see the summary")

    timings["total"] = round(time.perf_counter() - started, 2)
    summary["verdicts"] = verdicts
    summary["figures"] = [path.name for path in figures]
    summary["timings_s"] = timings
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("\n== Verdicts ==")
    for label, verdict in verdicts.items():
        print(f"  {label}: {'holds' if verdict else 'FAILS'}")
    print(f"\nrun times, s: {timings}")
    print(f"summary and figures written to {directory}")
    return 0 if accepted and all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
