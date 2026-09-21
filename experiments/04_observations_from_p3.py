"""M0-E04: reproducible observations of protocol P3 on source and target.

What is measured on the two virtual plants when they are excited with P3 (D-019) and
read by the sensors of D-020: C_A with sigma = 5 mol/m^3 and T with sigma = 0.5 K,
every 6 s, the four inputs known without error.

Registration
------------
Everything in the block "Fixed before the first run" below, and the hypotheses, method
and acceptance criteria of the M0-E04 entry of docs/experiment_log.md, were committed
before this script was run with these seeds. Known beforehand, and stated in that
entry: the P3 sequence of excitation seed 0 was accepted on both plants in M0-E03; the
unit tests of the measurement code had been run, with other seeds; and the plotting
code had been exercised on a shorter run with other seeds, outside the repository. No
seed is dropped or replaced after seeing a result, no reading is clipped, and a
criterion that fails is reported as failing.

Hypotheses
----------
H1  The true trajectories are accepted by simulation/checks.py at 0.1 s resolution.
H2  The observation grid is one row every 6 s from 0 to 7200 s, 1201 rows, strictly
    increasing; every row is a stored sample of the truth; every input change falls
    on a row, which carries the new inputs.
H3  Reading minus exact state behaves as independent zero-mean Gaussian noise with
    the specified sigma, the same on both plants: each of the 64 z-scores listed
    under "Acceptance of H3" below is within +-Z_LIMIT.
H4  The same configuration and seeds reproduce the content exactly; another sensor
    seed leaves corners, true trajectory, instants and inputs exactly as they were
    and changes every reading.

Run from the repository root (well under a minute):

    python experiments/04_observations_from_p3.py

Outputs go to PT_DATA_DIR/experiments/m0_e04/<run id> (git-ignored): three figures,
summary.json with a provenance block, and copies of the configurations. They are
diagnostic artefacts of the simulator. The figures and the "diagnostics" block use
exact states and measurement errors, which no model may be given. No array is written
to disk: the storage layer does not exist yet, and the content of every observation
set is identified by a digest instead.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from process_transfer import __version__  # noqa: E402
from process_transfer.config import load_sensors, load_true_plant  # noqa: E402
from process_transfer.cstr_variables import INPUT_NAMES, nominal_inputs  # noqa: E402
from process_transfer.data.paths import repository_root  # noqa: E402
from process_transfer.data.provenance import (  # noqa: E402
    copy_with_fingerprints,
    environment,
    git_state,
    new_run_directory,
)
from process_transfer.measurement.noise_statistics import (  # noqa: E402
    correlation,
    noise_statistics,
)
from process_transfer.measurement.observations import Observations  # noqa: E402
from process_transfer.measurement.sensors import MeasurementSpec  # noqa: E402
from process_transfer.simulation import cstr_true  # noqa: E402
from process_transfer.simulation.checks import (  # noqa: E402
    BALANCE_TOLERANCE,
    TEMPERATURE_ENVELOPE,
)
from process_transfer.simulation.cstr_true import TrueCSTRParameters  # noqa: E402
from process_transfer.simulation.integration import simulate_piecewise  # noqa: E402
from process_transfer.simulation.operating_run import (  # noqa: E402
    OperatingRun,
    observe_trajectory,
)
from process_transfer.simulation.protocols import (  # noqa: E402
    A10_COOLANT_TEMPERATURE,
    A10_FEED_TEMPERATURE,
    A10_RELATIVE_FEED_CONCENTRATION,
    A10_RELATIVE_FLOW,
    P3_HOLD,
    P3_REST,
    corner_label,
    p3_corners,
    p3_segments,
)
from process_transfer.simulation.steady_state import find_steady_states  # noqa: E402

# =========================================================================== #
# Fixed before the first run
# =========================================================================== #

PLANTS = ("source", "target")
N_EXCURSIONS = 10  # 10 x (120 s + 600 s) = 2 h per run, the first adaptation budget of D-011
SIMULATION_PERIOD = 0.1  # s, resolution of the truth, on which peaks and balances are judged
# The sensor period and noise levels come from configs/sensors_cstr.yaml: 6 s, 5 mol/m^3, 0.5 K.

# Two kinds of randomness, two seeds, two generators that never meet.
EXCITATION_SEED = 0  # corners of P3; the same sequence on both plants (D-008). Seed 0 of M0-E03.
SENSOR_SEED = 2026  # scenario A
OTHER_SENSOR_SEED = 2027  # scenario B: the same excitation read by another realisation of noise
# Noise stream of a run: (plant index, run index). Both plants have one run here.
PLANT_INDEX = {"source": 0, "target": 1}
RUN_INDEX = 0


@dataclass(frozen=True)
class Scenario:
    key: str
    excitation_seed: int
    sensor_seed: int
    purpose: str


SCENARIOS = (
    Scenario("A", EXCITATION_SEED, SENSOR_SEED, "the reference observations"),
    Scenario("A-again", EXCITATION_SEED, SENSOR_SEED, "scenario A generated a second time"),
    Scenario("B", EXCITATION_SEED, OTHER_SENSOR_SEED, "the same excitation, another sensor seed"),
)
STATISTICAL_SCENARIOS = ("A", "B")  # A-again repeats A and adds no independent sample

# Acceptance of H3. Every statistic below is a z-score: its distance from what correct
# noise would give, in standard deviations of what correct noise gives over n = 1201
# readings (measurement/noise_statistics.py). Per error series, 2 scenarios x 2 plants x
# 2 variables = 8 series:
Z_STATISTICS = ("mean", "spread", "lag_one", "beyond_one_sigma", "beyond_two_sigma")  # 40 scores
# and correlations, 24 scores: C_A against T within a plant (4); each variable of the
# source against each variable of the target (8); the same series under the two sensor
# seeds (4); each error series against the exact state it was added to (8).
Z_LIMIT = 4.0
# 64 scores in all. For correct noise |z| > 4 has probability 6.3e-5, so at least one of
# 64 exceeds it with probability of about 0.4 %: that is the chance of rejecting a correct
# sensor with seeds chosen blindly, as these were. What the limit resolves over 1201
# readings: a standard deviation wrong by 8 % or more, a bias of 0.12 sigma or more, a
# correlation of 0.12 or more. The reading of D-010 that was not chosen, 3.8 instead of
# 5 mol/m^3 on the target, would score about -12.
# Expected variation of one series of 1201 readings: the mean within +-sigma/sqrt(n) =
# +-0.029 sigma, the deviation within +-sigma/sqrt(2n) = +-0.020 sigma (one sigma each).

DIAGNOSTIC_WINDOW = 360.0  # s of scenario A shown in figure 2: the first excursion and after

INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PLANT_COLOR = {"source": "#2a78d6", "target": "#eb6834"}  # as in every figure of M0
LIMIT_COLOR = "#d03b3b"


# =========================================================================== #
# Plants and generation
# =========================================================================== #


@dataclass(frozen=True)
class Plant:
    name: str
    parameters: TrueCSTRParameters
    nominal_inputs: np.ndarray
    nominal_state: np.ndarray

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return cstr_true.rhs(0.0, x, u, self.parameters)


def load_plant(name: str) -> Plant:
    cfg = load_true_plant(repository_root() / "configs" / f"{name}_cstr.yaml")
    p, u = TrueCSTRParameters.from_config(cfg), nominal_inputs(cfg.plant)
    (steady,) = find_steady_states(lambda x: cstr_true.rhs(0.0, x, u, p), c_a_upper=u[1])
    return Plant(name, p, u, steady.state)


def generate(plant: Plant, scenario: Scenario, measurement: MeasurementSpec) -> OperatingRun:
    """One run, from the configuration and the two seeds and nothing else. The state is
    carried from segment to segment and never reset."""
    corners = p3_corners(N_EXCURSIONS, scenario.excitation_seed)
    trajectory = simulate_piecewise(
        plant.f,
        plant.nominal_state,
        p3_segments(plant.nominal_inputs, corners),
        SIMULATION_PERIOD,
    )
    return observe_trajectory(
        trajectory,
        plant.parameters,
        measurement,
        plant=plant.name,
        run=f"p3-e{scenario.excitation_seed}",
        sensor_seed=scenario.sensor_seed,
        sensor_stream=(PLANT_INDEX[plant.name], RUN_INDEX),
    )


# =========================================================================== #
# Checks
# =========================================================================== #


def check_truth(runs: dict[str, dict[str, OperatingRun]]) -> dict[str, object]:
    print("\n== H1. The true trajectories, judged every 0.1 s ==")
    results: dict[str, object] = {}
    for name in PLANTS:
        check = runs["A"][name].truth.check
        results[name] = {
            "accepted": bool(check.accepted),
            "refined_peak_K": check.refined_peak_temperature,
            "min_temperature_K": check.min_temperature,
            "c_a_range_mol_m3": list(check.c_a_range),
            "relative_mass_residual": check.relative_mass_residual,
            "relative_energy_residual": check.relative_energy_residual,
            "true_samples": int(len(runs["A"][name].truth.trajectory.times)),
        }
        print(
            f"  {name:6s} accepted={check.accepted}  T {check.min_temperature:.2f} to "
            f"{check.refined_peak_temperature:.4f} K  C_A {check.c_a_range[0]:.1f} to "
            f"{check.c_a_range[1]:.1f} mol/m^3  balances {check.relative_mass_residual:.1e}, "
            f"{check.relative_energy_residual:.1e}"
        )
    results["all_accepted"] = all(results[name]["accepted"] for name in PLANTS)
    return results


def check_grid(
    runs: dict[str, dict[str, OperatingRun]], measurement: MeasurementSpec
) -> dict[str, object]:
    print("\n== H2. The observation grid ==")
    period = measurement.sample_period
    duration = N_EXCURSIONS * (P3_HOLD + P3_REST)
    expected_times = period * np.arange(int(round(duration / period)) + 1)
    corners = p3_corners(N_EXCURSIONS, EXCITATION_SEED)
    results: dict[str, object] = {}
    for name in PLANTS:
        run = runs["A"][name]
        observed, truth = run.observations, run.truth
        changed = np.flatnonzero(np.any(np.diff(observed.inputs, axis=0) != 0.0, axis=1)) + 1
        switching = np.sort(
            np.concatenate(
                [
                    (P3_HOLD + P3_REST) * np.arange(1, N_EXCURSIONS),
                    (P3_HOLD + P3_REST) * np.arange(N_EXCURSIONS) + P3_HOLD,
                ]
            )
        )
        results[name] = {
            "rows": int(observed.n_samples),
            "instants_as_expected": bool(np.array_equal(observed.times, expected_times)),
            "strictly_increasing": bool(np.all(np.diff(observed.times) > 0.0)),
            "rows_are_stored_true_samples": bool(
                np.array_equal(truth.trajectory.times[truth.sample_indices], observed.times)
            ),
            "true_samples_between_rows": sorted({int(v) for v in np.diff(truth.sample_indices)}),
            "input_changes": int(len(changed)),
            "input_changes_at_the_switching_instants": bool(
                np.array_equal(observed.times[changed], switching)
            ),
            "first_row_carries_the_first_corner": bool(
                np.array_equal(observed.inputs[0], truth.trajectory.segments[0].inputs)
            ),
        }
        ok = all(v for k, v in results[name].items() if isinstance(v, bool))
        results[name]["as_hypothesised"] = bool(ok and observed.n_samples == len(expected_times))
        print(
            f"  {name:6s} {observed.n_samples} rows every {period:g} s, "
            f"{len(changed)} input changes, all on rows: "
            f"{results[name]['input_changes_at_the_switching_instants']}; one row every "
            f"{results[name]['true_samples_between_rows']} true samples"
        )
    results["corners"] = [corner_label(corner) for corner in corners]
    results["same_inputs_on_both_plants"] = bool(
        np.array_equal(
            runs["A"]["source"].observations.inputs, runs["A"]["target"].observations.inputs
        )
    )
    print(f"  corners of seed {EXCITATION_SEED}: {' '.join(results['corners'])}")
    return results


def check_noise(
    runs: dict[str, dict[str, OperatingRun]], measurement: MeasurementSpec
) -> tuple[dict[str, object], dict[str, float]]:
    print("\n== H3. The noise: reading minus exact state ==")
    series: dict[str, object] = {}
    scores: dict[str, float] = {}
    for key in STATISTICAL_SCENARIOS:
        for name in PLANTS:
            errors = runs[key][name].truth.errors
            for column, (variable, sigma) in enumerate(
                zip(measurement.variables, measurement.noise_std, strict=True)
            ):
                found = noise_statistics(errors[:, column], sigma)
                label = f"{key} {name} {variable}"
                series[label] = dataclasses.asdict(found)
                for statistic in Z_STATISTICS:
                    scores[f"{label}: {statistic}"] = found.z_scores[statistic]
                print(
                    f"  {label:14s} n={found.n}  mean {found.mean:+.4f}  rms {found.rms:.4f} "
                    f"(sigma {sigma:g})  beyond 1 sigma {found.beyond_one_sigma:.3f}  beyond 2 "
                    f"{found.beyond_two_sigma:.3f}  largest {found.largest_in_sigmas:.2f} sigma"
                )

    correlations: dict[str, object] = {}

    def add(label: str, a: np.ndarray, b: np.ndarray) -> None:
        r, z = correlation(a, b)
        correlations[label] = {"r": r, "z": z}
        scores[f"correlation, {label}"] = z

    variables = measurement.variables
    for key in STATISTICAL_SCENARIOS:
        source, target = (runs[key][name].truth for name in PLANTS)
        for name, truth in zip(PLANTS, (source, target), strict=True):
            add(
                f"{key} {name}: {variables[0]} with {variables[1]}",
                truth.errors[:, 0],
                truth.errors[:, 1],
            )
            for column, variable in enumerate(variables):
                add(
                    f"{key} {name}: error of {variable} with its exact value",
                    truth.errors[:, column],
                    truth.exact[:, column],
                )
        for i, first in enumerate(variables):
            for j, second in enumerate(variables):
                add(
                    f"{key}: source {first} with target {second}",
                    source.errors[:, i],
                    target.errors[:, j],
                )
    for name in PLANTS:
        for column, variable in enumerate(variables):
            add(
                f"{name} {variable}: sensor seed {SENSOR_SEED} with {OTHER_SENSOR_SEED}",
                runs["A"][name].truth.errors[:, column],
                runs["B"][name].truth.errors[:, column],
            )

    worst = max(scores, key=lambda label: abs(scores[label]))
    outside = {label: z for label, z in scores.items() if abs(z) > Z_LIMIT}
    print(
        f"  {len(scores)} z-scores; largest |z| = {abs(scores[worst]):.2f} ({worst}); "
        f"outside +-{Z_LIMIT:g}: {len(outside)}"
    )
    results = {
        "series": series,
        "correlations": correlations,
        "n_scores": len(scores),
        "largest_abs_z": abs(scores[worst]),
        "largest_abs_z_label": worst,
        "outside_limit": outside,
        "z_limit": Z_LIMIT,
        "all_within_limit": len(outside) == 0,
        "rms_of_all_scores": float(np.sqrt(np.mean(np.square(list(scores.values()))))),
    }
    return results, scores


def check_reproducibility(runs: dict[str, dict[str, OperatingRun]]) -> dict[str, object]:
    print("\n== H4. Reproduction, and the two kinds of randomness ==")
    results: dict[str, object] = {}
    for name in PLANTS:
        a, again, b = (runs[key][name] for key in ("A", "A-again", "B"))
        same = {
            field: bool(
                np.array_equal(getattr(a.observations, field), getattr(again.observations, field))
            )
            for field in ("times", "measured", "inputs")
        }
        results[name] = {
            "digest_A": a.observations.content_digest(),
            "digest_A_again": again.observations.content_digest(),
            "digest_B": b.observations.content_digest(),
            "A_again_equals_A": same,
            "A_again_true_states_equal_A": bool(
                np.array_equal(a.truth.trajectory.states, again.truth.trajectory.states)
            ),
            "B_true_states_equal_A": bool(
                np.array_equal(a.truth.trajectory.states, b.truth.trajectory.states)
            ),
            "B_instants_and_inputs_equal_A": bool(
                np.array_equal(a.observations.times, b.observations.times)
                and np.array_equal(a.observations.inputs, b.observations.inputs)
            ),
            "B_readings_that_differ_from_A": int(
                np.sum(np.any(a.observations.measured != b.observations.measured, axis=1))
            ),
            "rows": int(a.observations.n_samples),
        }
        reproduced = all(same.values()) and results[name]["A_again_true_states_equal_A"]
        independent = (
            results[name]["B_true_states_equal_A"]
            and results[name]["B_instants_and_inputs_equal_A"]
            and results[name]["B_readings_that_differ_from_A"] == a.observations.n_samples
        )
        results[name]["as_hypothesised"] = bool(
            reproduced
            and independent
            and results[name]["digest_A"] == results[name]["digest_A_again"]
            and results[name]["digest_A"] != results[name]["digest_B"]
        )
        print(
            f"  {name:6s} A again equals A: {reproduced}; another sensor seed leaves truth, "
            f"instants and inputs unchanged: {independent} "
            f"({results[name]['B_readings_that_differ_from_A']} of {a.observations.n_samples} "
            "rows of readings differ)"
        )
        print(f"         digest of A {results[name]['digest_A'][:16]}...")
    return results


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


def figure_inputs_and_response(runs: dict[str, OperatingRun], directory: Path) -> None:
    """The experiment as applied, and the true response of both plants."""
    fig, axes = plt.subplots(6, 1, figsize=(11.0, 10.5), sharex=True, facecolor=SURFACE)
    observed = runs["source"].observations  # the inputs are the same on both plants
    minutes = observed.times / 60.0
    scale = {"q": 60000.0, "C_Af": 1.0, "T_f": 1.0, "T_c": 1.0}
    unit = {"q": "L/min", "C_Af": "mol/m^3", "T_f": "K", "T_c": "K"}
    for ax, (column, name) in zip(axes[:4], enumerate(INPUT_NAMES), strict=True):
        style(ax)
        ax.step(
            minutes,
            scale[name] * observed.inputs[:, column],
            where="post",
            color=INK,
            linewidth=1.1,
        )
        ax.set_ylabel(f"{name}, {unit[name]}", fontsize=9, color=INK_SECONDARY)
    for ax, (column, label) in zip(axes[4:], enumerate(("C_A, mol/m^3", "T, K")), strict=True):
        style(ax)
        for name in PLANTS:
            trajectory = runs[name].truth.trajectory
            ax.plot(
                trajectory.times / 60.0,
                trajectory.states[:, column],
                color=PLANT_COLOR[name],
                linewidth=1.1,
                label=name,
            )
        ax.set_ylabel(label, fontsize=9, color=INK_SECONDARY)
    axes[5].axhline(TEMPERATURE_ENVELOPE[1], color=LIMIT_COLOR, linewidth=1.0)
    axes[5].annotate(
        f"limit {TEMPERATURE_ENVELOPE[1]:g} K",
        (0.0, TEMPERATURE_ENVELOPE[1]),
        xytext=(4, -11),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
    axes[4].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", ncols=2)
    axes[5].set_xlabel("time, min", fontsize=9, color=INK_SECONDARY)
    fig.suptitle(
        f"P3 with excitation seed {EXCITATION_SEED}: the same inputs on both plants, and the "
        "true response of each (exact states, diagnostic)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(directory / "fig1_inputs_and_response.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_truth_against_readings(
    runs: dict[str, OperatingRun], measurement: MeasurementSpec, directory: Path
) -> None:
    """Diagnostic only: exact states, which no model may see, against the readings."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), sharex=True, facecolor=SURFACE)
    labels = ("C_A, mol/m^3", "T, K")
    for column_index, name in enumerate(PLANTS):
        run = runs[name]
        trajectory, observed = run.truth.trajectory, run.observations
        fine = trajectory.times <= DIAGNOSTIC_WINDOW
        rows = observed.times <= DIAGNOSTIC_WINDOW
        for row_index, sigma in enumerate(measurement.noise_std):
            ax = axes[row_index, column_index]
            style(ax)
            exact = trajectory.states[fine, row_index]
            ax.fill_between(
                trajectory.times[fine],
                exact - sigma,
                exact + sigma,
                color=PLANT_COLOR[name],
                alpha=0.15,
                linewidth=0,
                label="exact +- 1 sigma (not a bound)",
            )
            ax.plot(
                trajectory.times[fine],
                exact,
                color=PLANT_COLOR[name],
                linewidth=1.3,
                label="exact state",
            )
            ax.plot(
                observed.times[rows],
                observed.measured[rows, row_index],
                linestyle="none",
                marker="o",
                markersize=3.0,
                color=INK,
                label="reading, every 6 s",
            )
            ax.set_ylabel(labels[row_index], fontsize=9, color=INK_SECONDARY)
            if row_index == 0:
                ax.set_title(name, fontsize=10, color=INK, loc="left")
            else:
                ax.set_xlabel("time, s", fontsize=9, color=INK_SECONDARY)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")
    fig.suptitle(
        f"Diagnostic, not data for modelling: exact states against readings, first "
        f"{DIAGNOSTIC_WINDOW:g} s of scenario A",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(directory / "fig2_truth_against_readings.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_noise(
    runs: dict[str, OperatingRun],
    measurement: MeasurementSpec,
    scores: dict[str, float],
    directory: Path,
) -> None:
    fig = plt.figure(figsize=(12.0, 6.6), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, width_ratios=(1.0, 1.0, 0.12, 1.5))
    z = np.linspace(-4.5, 4.5, 300)
    for row_index, name in enumerate(PLANTS):
        for column, (variable, sigma) in enumerate(
            zip(measurement.variables, measurement.noise_std, strict=True)
        ):
            ax = fig.add_subplot(grid[row_index, column])
            style(ax)
            standardised = runs[name].truth.errors[:, column] / sigma
            ax.hist(
                standardised,
                bins=np.linspace(-4.5, 4.5, 37),
                density=True,
                color=PLANT_COLOR[name],
                alpha=0.75,
            )
            ax.plot(z, norm.pdf(z), color=INK, linewidth=1.1)
            ax.set_title(
                f"{name}, {variable}: rms {np.sqrt(np.mean(standardised**2)):.3f} sigma, "
                f"n = {len(standardised)}",
                fontsize=9,
                color=INK,
                loc="left",
            )
            ax.set_xlabel("error / sigma", fontsize=9, color=INK_SECONDARY)

    ax = fig.add_subplot(grid[:, 3])
    style(ax)
    values = np.array(list(scores.values()))
    order = np.argsort(values)
    ax.plot(
        values[order],
        np.arange(len(values)),
        linestyle="none",
        marker="o",
        markersize=3.5,
        color=INK,
    )
    for limit in (-Z_LIMIT, Z_LIMIT):
        ax.axvline(limit, color=LIMIT_COLOR, linewidth=1.0)
    ax.annotate(
        f"limit +-{Z_LIMIT:g}",
        (Z_LIMIT, len(values) - 1),  # top of the line, clear of the legend at the bottom
        xytext=(-4, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=INK,
    )
    expected = norm.ppf((np.arange(len(values)) + 0.5) / len(values))
    ax.plot(
        expected,
        np.arange(len(values)),
        color=INK_SECONDARY,
        linewidth=1.0,
        label="standard normal",
    )
    ax.set_xlim(-5.0, 5.0)
    ax.set_yticks([])
    ax.set_xlabel("z-score", fontsize=9, color=INK_SECONDARY)
    ax.set_title(f"all {len(values)} z-scores, sorted", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.suptitle(
        "Measurement errors of scenario A against the specified Gaussian noise (diagnostic)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(directory / "fig3_noise.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# =========================================================================== #
# Main
# =========================================================================== #


def main() -> None:
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    started = time.perf_counter()
    print(f"process_transfer {__version__}; M0-E04 observations of P3 on source and target")
    state = git_state()
    directory = new_run_directory("m0_e04", state)
    if not state["code_identified"]:
        print(f"  NOTE: the code of this run is not fully identified: {state['reason']}")

    configs = repository_root() / "configs"
    measurement = MeasurementSpec.from_config(load_sensors(configs / "sensors_cstr.yaml"))
    plants = {name: load_plant(name) for name in PLANTS}
    defaults = inspect.signature(simulate_piecewise).parameters
    configurations = [configs / f"{name}_cstr.yaml" for name in PLANTS] + [
        configs / "sensors_cstr.yaml"
    ]

    summary: dict[str, object] = {
        "provenance": {
            "experiment": "M0-E04",
            "run_id": directory.name,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git": state,
            "configurations": copy_with_fingerprints(configurations, directory / "configs"),
            "environment": environment(),
            "protocol": {
                "name": "P3",
                "amplitudes_A10": [
                    A10_RELATIVE_FLOW,
                    A10_RELATIVE_FEED_CONCENTRATION,
                    A10_FEED_TEMPERATURE,
                    A10_COOLANT_TEMPERATURE,
                ],
                "hold_s": P3_HOLD,
                "rest_s": P3_REST,
                "n_excursions": N_EXCURSIONS,
                "state_reset_between_segments": False,
            },
            "seeds": {
                "excitation": EXCITATION_SEED,
                "sensor": {"A": SENSOR_SEED, "A-again": SENSOR_SEED, "B": OTHER_SENSOR_SEED},
                "sensor_stream": {name: [PLANT_INDEX[name], RUN_INDEX] for name in PLANTS},
                "generators": "excitation: numpy default_rng(seed); sensors: SeedSequence("
                "entropy=seed, spawn_key=(plant index, run index, channel)); no global generator",
            },
            "scenarios": [dataclasses.asdict(scenario) for scenario in SCENARIOS],
            "sensors": {
                "sample_period_s": measurement.sample_period,
                "variables": list(measurement.variables),
                "noise_std_SI": list(measurement.noise_std),
            },
            "criteria": {
                "temperature_envelope_K": list(TEMPERATURE_ENVELOPE),
                "balance_tolerance": BALANCE_TOLERANCE,
                "truth": "process_transfer.simulation.checks.check_trajectory",
                "z_limit": Z_LIMIT,
                "z_statistics": list(Z_STATISTICS),
            },
            "integration": {
                "method": defaults["method"].default,
                "rtol": defaults["rtol"].default,
                "atol": defaults["atol"].default,
                "sample_period_s": SIMULATION_PERIOD,
                "restart": "one solver call per input segment; the state is never reset",
            },
            "contains_hidden_parameters": True,
            "contains_exact_states_or_errors": "in the diagnostics block and the figures only",
        }
    }

    timings: dict[str, float] = {}
    tick = time.perf_counter()
    runs = {
        scenario.key: {name: generate(plants[name], scenario, measurement) for name in PLANTS}
        for scenario in SCENARIOS
    }
    timings["generation"] = round(time.perf_counter() - tick, 1)

    tick = time.perf_counter()
    truth = check_truth(runs)
    grid = check_grid(runs, measurement)
    noise, scores = check_noise(runs, measurement)
    reproduction = check_reproducibility(runs)
    timings["checks"] = round(time.perf_counter() - tick, 1)

    example: Observations = runs["A"]["target"].observations
    rows = example.n_samples
    summary["observations"] = {  # built from the Observations objects and nothing else
        "fields": [field.name for field in dataclasses.fields(Observations)],
        "rows_per_run": rows,
        "numbers_per_run": rows * (1 + len(example.measured_names) + len(example.input_names)),
        "runs_generated": len(SCENARIOS) * len(PLANTS),
        "measured": dict(zip(example.measured_names, example.measured_units, strict=True)),
        "inputs": dict(zip(example.input_names, example.input_units, strict=True)),
        "digests": {
            key: {name: runs[key][name].observations.content_digest() for name in PLANTS}
            for key in runs
        },
    }
    summary["diagnostics"] = {  # uses exact states and errors: never an input to a model
        "H1_truth": truth,
        "H2_grid": grid,
        "H3_noise": noise,
        "H4_reproducibility": reproduction,
    }
    verdicts = {
        "H1_true_trajectories_accepted": bool(truth["all_accepted"]),
        "H2_grid_and_alignment": bool(
            all(grid[name]["as_hypothesised"] for name in PLANTS)
            and grid["same_inputs_on_both_plants"]
        ),
        "H3_noise_within_limit": bool(noise["all_within_limit"]),
        "H4_reproducible_and_independent": bool(
            all(reproduction[name]["as_hypothesised"] for name in PLANTS)
        ),
    }
    summary["verdicts"] = verdicts
    print("\n== Verdicts ==")
    for label, verdict in verdicts.items():
        print(f"  {label}: {'holds' if verdict else 'FAILS'}")

    # The summary is written before the figures, so that a plotting error cannot lose it.
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tick = time.perf_counter()
    figure_inputs_and_response(runs["A"], directory)
    figure_truth_against_readings(runs["A"], measurement, directory)
    figure_noise(runs["A"], measurement, scores, directory)
    timings["figures"] = round(time.perf_counter() - tick, 1)
    timings["total"] = round(time.perf_counter() - started, 1)
    summary["timings_s"] = timings
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nrun times, s: {timings}")
    print(f"figures and summary.json written to {directory}")


if __name__ == "__main__":
    main()
