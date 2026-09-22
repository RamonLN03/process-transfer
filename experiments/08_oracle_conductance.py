"""M0-E08: the effect of the temperature-dependent conductance on its own. An oracle
diagnostic, stored among the diagnostic artefacts only.

For each plant and each of the three P3 sequences of M0-E05, excitation seeds 0, 1 and 2
with ten excursions, two simulations from the same nominal steady state:

    A  the true plant as it is;
    B  the same plant with the same true kinetics, and UA(T) replaced by a constant
       conductance equal to UA_true(T_nominal) of that plant.

The anchoring at T_nominal is deliberate. It keeps the energy balance closed at the
starting point, so the two variants start from one steady state and differ only by the
temperature dependence of the conductance, not by a different conductance at the start.
Setting alpha to zero while keeping UA_ref would not do: on the target T_nominal differs
from T_ref, so the conductance at the nominal point would change and the nominal point
with it. The official YAML files are not touched; variant B is built here, from the true
parameters, by replacing two fields.

This is an oracle. It uses the hidden physics, and its results and figures go under
PT_DATA_DIR/experiments only, never under the available branch. It answers how much UA(T)
contributes to the trajectories under these experiments when the rest of the physics is
known. Whether that contribution can be separated from re-estimated parameters and from
noise is a question for M1 and is not answered here; the comparison with the sigmas of
D-020 is a diagnostic of scale, not a detection threshold.

Registration
------------
Fixed before the first run and committed: the six cases, source and target under the P3
sequences of seeds 0, 1 and 2 with ten excursions, 2 h each; variant B as defined above;
the integration of the project (LSODA, rtol = atol = 1e-9, stored every 0.1 s) for both
variants, and a tighter one (rtol = atol = 1e-11) of both, whose distance from the first
estimates the integration error; the sensor grid of 6 s and the sigmas of D-020, 5 mol/m^3
and 0.5 K, for the normalised differences; the phases of P3, 120 s excursions and 600 s
rests, for the metrics per phase. alpha_source = 0.005 1/K and alpha_target = 0.002 1/K
are those of the plants and are not changed. One smoke run outside the repository, with
one excursion of seed 900 on both plants, was used to debug the script.

Hypotheses and criteria
-----------------------
H1  Anchoring. For each plant the conductance of B equals UA_A(T_nominal) exactly, and the
    right-hand side of B at the nominal state equals that of A bit for bit: the starting
    point is a steady state of both variants, closed to the residual of the plant.
H2  Numerical resolution. For every state, plant and sequence, the integration error
    estimate of each variant, the largest |x at 1e-9 - x at 1e-11| over the dense grid, is
    at most RESOLUTION_FRACTION of the largest |B - A| of that state on that sequence. A
    case that fails is reported as not resolved, and its differences are not interpreted.
H3  Validity of the pair (mandatory). A is fully accepted (finite, physical, inside the
    envelope, balances closed). B is physically valid independent of its own envelope:
    finite, physical, balances closed; B leaving [335, 380] K is a reported diagnostic, not
    a validity failure. A case that fails H3 is reported as invalid and its differences are
    not a scientifically acceptable result, whatever H1 and H2 say about it. See
    ``comparison_is_valid`` in ``simulation/checks.py``.

Reported, without being a validity criterion: whether B stays inside [335, 380] K, its peak
and the seconds above the limit if any, a variant that leaves the envelope being kept and
reported as a diagnostic, never mixed with the mandatory validity of H3; the range of
UA_A(T) / UA_B over each trajectory; the differences B - A of C_A and T on the dense grid,
largest absolute value and root mean square, over the whole run and over the excursions and
the rests separately; the same on the sensor grid in units of the sigmas of D-020; and the
fraction of the time at which B is hotter than A.

Expected before the run, from M0-E03, where UA/UA_ref ranged from 0.972 to 1.032 on the
source and from 0.997 to 1.023 on the target under P3: temperature differences of a few
tenths of a kelvin at most, below one sigma_T; concentration differences of a few mol/m^3,
of the order of one sigma_CA; much smaller differences during the rests, which take 83 %
of the time; an integration error near 1e-7, so H2 by a wide margin.

Run from the repository root (expected under a minute):

    python experiments/08_oracle_conductance.py

Outputs go to PT_DATA_DIR/experiments/m0_e08/<run id>: summary.json and one figure per
plant. They contain hidden parameters and exact states. The exit code is 0 only if H1, H2
and H3 hold for every case; a validity failure (H3) fails the run even if H1 and H2 hold.
"""

from __future__ import annotations

import dataclasses
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
from process_transfer.data.paths import repository_root  # noqa: E402
from process_transfer.data.provenance import (  # noqa: E402
    copy_with_fingerprints,
    environment,
    git_state,
    new_run_directory,
)
from process_transfer.generation.plants import VirtualPlant, load_virtual_plant  # noqa: E402
from process_transfer.simulation import cstr_true  # noqa: E402
from process_transfer.simulation.checks import check_trajectory, comparison_is_valid  # noqa: E402
from process_transfer.simulation.cstr_true import TrueCSTRParameters, conductance  # noqa: E402
from process_transfer.simulation.integration import Trajectory, simulate_piecewise  # noqa: E402
from process_transfer.simulation.operating_run import sensor_sample_indices  # noqa: E402
from process_transfer.simulation.protocols import (  # noqa: E402
    corner_label,
    p3_corners,
    p3_segments,
)

# =========================================================================== #
# Fixed before the first run
# =========================================================================== #

CONFIGS = repository_root() / "configs"
PLANT_FILES = {"source": CONFIGS / "source_cstr.yaml", "target": CONFIGS / "target_cstr.yaml"}
SEEDS = (0, 1, 2)  # the P3 sequences of M0-E05
N_EXCURSIONS = 10
SIMULATION_PERIOD = 0.1  # s
TOLERANCE = 1.0e-9  # rtol = atol of the project's integration
TIGHT_TOLERANCE = 1.0e-11  # the second integration, to estimate the error of the first
RESOLUTION_FRACTION = 0.01  # the integration error must be at most this fraction of the effect
SENSOR_PERIOD = 6.0  # s, D-020
SIGMA = {"C_A": 5.0, "T": 0.5}  # mol/m^3 and K, D-020
STATE_NAMES = ("C_A", "T")

INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PLANT_COLOR = {"source": "#2a78d6", "target": "#eb6834"}
EXCURSION_SHADE = "#e9e8e1"


# =========================================================================== #
# Variant B, and the checks of its anchoring
# =========================================================================== #


def constant_conductance_variant(plant: VirtualPlant) -> TrueCSTRParameters:
    """The parameters of variant B: UA constant and equal to UA_true(T_nominal)."""
    anchored = float(conductance(float(plant.nominal_state[1]), plant.parameters))
    return dataclasses.replace(plant.parameters, ua_ref=anchored, alpha=0.0)


def anchoring(plant: VirtualPlant, variant: TrueCSTRParameters) -> dict[str, object]:
    t_nominal = float(plant.nominal_state[1])
    ua_a = float(conductance(t_nominal, plant.parameters))
    ua_b = float(conductance(t_nominal, variant))
    rhs_a = cstr_true.rhs(0.0, plant.nominal_state, plant.nominal_inputs, plant.parameters)
    rhs_b = cstr_true.rhs(0.0, plant.nominal_state, plant.nominal_inputs, variant)
    dilution = plant.nominal_inputs[0] / plant.parameters.volume
    scale = np.array([dilution * plant.nominal_inputs[1], dilution * plant.nominal_inputs[2]])
    return {
        "T_nominal_K": t_nominal,
        "T_ref_K": plant.parameters.t_ref,
        "alpha_per_K": plant.parameters.alpha,
        "UA_A_at_T_nominal_W_per_K": ua_a,
        "UA_B_W_per_K": ua_b,
        "UA_ref_of_the_plant_W_per_K": plant.parameters.ua_ref,
        "conductances_equal": ua_a == ua_b,
        "rhs_A_at_nominal": rhs_a.tolist(),
        "rhs_B_at_nominal": rhs_b.tolist(),
        "rhs_equal_bit_for_bit": bool(np.array_equal(rhs_a, rhs_b)),
        "closed_to_the_residual_of_the_plant": bool(np.all(np.abs(rhs_b) <= 1.0e-9 * scale)),
    }


# =========================================================================== #
# One case: a plant under one sequence
# =========================================================================== #


def phases(trajectory: Trajectory) -> np.ndarray:
    """True at the samples that belong to an excursion of P3, taken from the segment
    structure: segments alternate, an excursion of P3_HOLD then a rest of P3_REST, and
    the sample at a switching instant belongs to the segment that starts there."""
    excursion = np.zeros(len(trajectory.times), dtype=bool)
    position = 0
    for index, segment in enumerate(trajectory.segments):
        n = len(segment.times) - (1 if index < len(trajectory.segments) - 1 else 0)
        if index % 2 == 0:
            excursion[position : position + n] = True
        position += n
    return excursion


def _metrics(delta: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = delta[mask]
    if values.size == 0:
        raise ValueError("a phase without samples has no metrics")
    return {
        "max_abs": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "n": int(values.size),
    }


def one_case(plant: VirtualPlant, variant: TrueCSTRParameters, seed: int) -> dict[str, object]:
    corners = p3_corners(N_EXCURSIONS, seed)
    segments = p3_segments(plant.nominal_inputs, corners)

    def f_b(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return cstr_true.rhs(0.0, x, u, variant)

    runs = {}
    for label, f in (("A", plant.f), ("B", f_b)):
        runs[label] = simulate_piecewise(f, plant.nominal_state, segments, SIMULATION_PERIOD)
        runs[label + "_tight"] = simulate_piecewise(
            f,
            plant.nominal_state,
            segments,
            SIMULATION_PERIOD,
            rtol=TIGHT_TOLERANCE,
            atol=TIGHT_TOLERANCE,
        )
    a, b = runs["A"], runs["B"]
    if not np.array_equal(a.times, b.times):
        raise ValueError("the two variants were not sampled at the same instants")
    delta = b.states - a.states  # B - A on the dense grid
    error = {
        "A": np.max(np.abs(a.states - runs["A_tight"].states), axis=0),
        "B": np.max(np.abs(b.states - runs["B_tight"].states), axis=0),
    }
    excursion = phases(a)
    sensors = sensor_sample_indices(a, SENSOR_PERIOD)
    check_a, check_b = check_trajectory(a, plant.parameters), check_trajectory(b, variant)
    valid = comparison_is_valid(check_a, check_b)
    ratio = conductance(a.states[:, 1], plant.parameters) / variant.ua_ref

    per_state: dict[str, object] = {}
    resolved = True
    for k, name in enumerate(STATE_NAMES):
        largest = float(np.max(np.abs(delta[:, k])))
        worst_error = float(max(error["A"][k], error["B"][k]))
        state_resolved = worst_error <= RESOLUTION_FRACTION * largest
        resolved = resolved and state_resolved
        normalised = delta[sensors, k] / SIGMA[name]
        per_state[name] = {
            "dense_grid": {
                "whole_run": _metrics(delta[:, k], np.ones(len(delta), dtype=bool)),
                "excursions": _metrics(delta[:, k], excursion),
                "rests": _metrics(delta[:, k], ~excursion),
            },
            "sensor_grid_in_sigmas": {
                "whole_run": _metrics(normalised, np.ones(len(normalised), dtype=bool)),
                "excursions": _metrics(normalised, excursion[sensors]),
                "rests": _metrics(normalised, ~excursion[sensors]),
                "sigma": SIGMA[name],
            },
            "integration_error_estimate": {
                "A": float(error["A"][k]),
                "B": float(error["B"][k]),
                "fraction_of_the_largest_difference": (
                    worst_error / largest if largest > 0.0 else None
                ),
                "resolved": bool(state_resolved),
            },
            "fraction_of_time_B_above_A": float(np.mean(delta[:, k] > 0.0)),
        }
    return {
        "seed": seed,
        "corners": [corner_label(corner) for corner in corners],
        "n_samples": int(len(a.times)),
        "rhs_evaluations": {"A": a.n_rhs_evaluations, "B": b.n_rhs_evaluations},
        "UA_A_over_UA_B_range": [float(np.min(ratio)), float(np.max(ratio))],
        "A": {
            "accepted": check_a.accepted,
            "values_finite": check_a.values_finite,
            "states_physical": check_a.states_physical,
            "inside_envelope": check_a.inside_envelope,
            "balances_close": check_a.balances_close,
            "refined_peak_temperature_K": check_a.refined_peak_temperature,
            "min_temperature_K": check_a.min_temperature,
        },
        "B": {
            "accepted": check_b.accepted,
            "physically_valid": check_b.physically_valid,
            "values_finite": check_b.values_finite,
            "states_physical": check_b.states_physical,
            "inside_envelope": check_b.inside_envelope,
            "balances_close": check_b.balances_close,
            "refined_peak_temperature_K": check_b.refined_peak_temperature,
            "min_temperature_K": check_b.min_temperature,
            "seconds_above_limit": check_b.seconds_above_limit,
            "relative_mass_residual": check_b.relative_mass_residual,
            "relative_energy_residual": check_b.relative_energy_residual,
        },
        "valid": bool(valid),
        "differences_B_minus_A": per_state,
        "resolved": bool(resolved),
        "figure_series": {
            "times_s": a.times[::10].tolist(),
            "delta_C_A": delta[::10, 0].tolist(),
            "delta_T": delta[::10, 1].tolist(),
            "excursion": excursion[::10].tolist(),
        },
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


def figure_differences(plant_id: str, cases: list[dict[str, object]], directory: Path) -> Path:
    """B - A over time for every sequence of one plant, excursions shaded, with the
    sigmas of D-020 as dotted lines for scale."""
    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(11.0, 2.6 * len(cases) + 0.8),
        sharex=True,
        squeeze=False,
        facecolor=SURFACE,
    )
    for row, case in enumerate(cases):
        series = case["figure_series"]  # type: ignore[index]
        minutes = np.asarray(series["times_s"]) / 60.0  # type: ignore[index]
        excursion = np.asarray(series["excursion"], dtype=bool)  # type: ignore[index]
        for column, (name, key, unit) in enumerate(
            (("C_A", "delta_C_A", "mol/m^3"), ("T", "delta_T", "K"))
        ):
            ax = axes[row, column]
            style(ax)
            ax.fill_between(
                minutes,
                0.0,
                1.0,
                where=excursion,
                transform=ax.get_xaxis_transform(),
                color=EXCURSION_SHADE,
                linewidth=0,
            )
            ax.plot(minutes, series[key], color=PLANT_COLOR[plant_id], linewidth=0.9)  # type: ignore[index]
            for sign in (1.0, -1.0):
                ax.axhline(sign * SIGMA[name], color=INK_SECONDARY, linewidth=0.6, linestyle=":")
            ax.axhline(0.0, color=AXIS, linewidth=0.6)
            ax.set_ylabel(f"{name}(B) - {name}(A), {unit}", fontsize=9, color=INK_SECONDARY)
            ax.set_title(f"P3 seed {case['seed']}", fontsize=9, color=INK, loc="left")
            if row == len(cases) - 1:
                ax.set_xlabel("process time, min", fontsize=9, color=INK_SECONDARY)
    fig.suptitle(
        f"M0-E08, {plant_id}: constant conductance UA(T_nominal) (B) against UA(T) (A), the "
        "same true kinetics and inputs; excursions shaded; dotted: the sigmas of D-020, for "
        "scale only. Oracle diagnostic",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = directory / f"fig1_differences_{plant_id}.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# =========================================================================== #
# Main
# =========================================================================== #


def main() -> int:
    started = time.perf_counter()
    print(f"process_transfer {__version__}; M0-E08 the effect of UA(T) alone, oracle diagnostic")
    state = git_state()
    directory = new_run_directory("m0_e08", state)
    if not state["code_identified"]:
        print(f"  NOTE: the code of this run is not fully identified: {state['reason']}")
    plants = {plant_id: load_virtual_plant(path) for plant_id, path in PLANT_FILES.items()}
    summary: dict[str, object] = {
        "provenance": {
            "experiment": "M0-E08",
            "run_id": directory.name,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git": state,
            "configurations": copy_with_fingerprints(
                list(PLANT_FILES.values()), directory / "configs"
            ),
            "environment": environment(),
            "seeds": list(SEEDS),
            "n_excursions": N_EXCURSIONS,
            "tolerances": {"project": TOLERANCE, "tight": TIGHT_TOLERANCE},
            "resolution_fraction": RESOLUTION_FRACTION,
            "sigmas": SIGMA,
            "oracle": True,
            "contains_hidden_parameters": True,
        },
        "plants": {},
    }
    verdicts = {"H1_anchoring": True, "H2_resolved": True, "H3_valid": True}
    figures = []
    timings: dict[str, float] = {}
    for plant_id, plant in plants.items():
        tick = time.perf_counter()
        variant = constant_conductance_variant(plant)
        anchor = anchoring(plant, variant)
        verdicts["H1_anchoring"] = verdicts["H1_anchoring"] and bool(
            anchor["conductances_equal"]
            and anchor["rhs_equal_bit_for_bit"]
            and anchor["closed_to_the_residual_of_the_plant"]
        )
        cases = [one_case(plant, variant, seed) for seed in SEEDS]
        verdicts["H2_resolved"] = verdicts["H2_resolved"] and all(
            case["resolved"] for case in cases
        )
        verdicts["H3_valid"] = verdicts["H3_valid"] and all(case["valid"] for case in cases)
        figures.append(figure_differences(plant_id, cases, directory))
        summary["plants"][plant_id] = {  # type: ignore[index]
            "anchoring": anchor,
            "cases": [{k: v for k, v in case.items() if k != "figure_series"} for case in cases],
        }
        timings[plant_id] = round(time.perf_counter() - tick, 2)
        print(
            f"  {plant_id}: UA_B = {anchor['UA_B_W_per_K']:.3f} W/K, alpha "
            f"{anchor['alpha_per_K']}; anchored {anchor['rhs_equal_bit_for_bit']}"
        )
        for case in cases:
            d = case["differences_B_minus_A"]
            t, c = d["T"]["dense_grid"], d["C_A"]["dense_grid"]
            tag = "" if case["valid"] else " -- INVALID PAIR, differences not interpretable"
            print(
                f"    seed {case['seed']}: max |dT| {t['whole_run']['max_abs']:.3f} K (excursions "
                f"{t['excursions']['max_abs']:.3f}, rests {t['rests']['max_abs']:.3f}), rms "
                f"{t['whole_run']['rms']:.3f} K; max |dC_A| {c['whole_run']['max_abs']:.2f} "
                f"mol/m^3, rms {c['whole_run']['rms']:.2f}; on the sensor grid max "
                f"{d['T']['sensor_grid_in_sigmas']['whole_run']['max_abs']:.2f} sigma_T and "
                f"{d['C_A']['sensor_grid_in_sigmas']['whole_run']['max_abs']:.2f} sigma_CA; "
                f"B peak {case['B']['refined_peak_temperature_K']:.2f} K, inside envelope "
                f"{case['B']['inside_envelope']}; resolved {case['resolved']}{tag}"
            )
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
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
