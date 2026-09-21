"""M0-E03b: does protocol P3 start every excursion at the nominal steady state?

P3, accepted in D-019 as the initial excitation: A10 amplitudes, excursions of 120 s
to the corners of the input box, 600 s at the nominal inputs between them. Its case
rests on each excursion behaving like a step from the nominal steady state, which
M0-E02 and M0-E03 enumerated. That holds only if the rest is long enough. Here it is
measured, without ever resetting the state to the steady state:

1. recovery: after each of the 16 excursions from nominal and 600 s of rest, how far
   the plant is from its nominal steady state;
2. carried state: all 256 ordered pairs of excursions, repeats included. Excursion a,
   rest, excursion b, rest, the residual of a being carried into b. For each pair: the
   acceptance checks on the whole trajectory, its peak, and the difference between
   the peak of b inside the pair and the peak of b started from the exact steady state.

Tolerances. They were fixed before this sweep was run, from quantities that do not
depend on its results. The reviewer's numbers for both checks were known beforehand;
the tolerances were not derived from them.

* recovery, temperature: 0.005 K, one hundredth of the planned sensor noise sigma_T
  = 0.5 K. A residual a hundred times below what the sensor resolves cannot be told
  from an exact restart in any data generated later;
* recovery, concentration: 0.038 mol/m^3, one hundredth of the smaller of the two
  readings of sigma_CA still open in D-020 (2 % of the target's nominal C_A, 3.8
  mol/m^3; the other reading is 5 mol/m^3). The stricter one is used so that the
  check does not depend on that decision;
* agreement of peaks: 0.05 K, a tenth of sigma_T and about 1 % of the 3.8 K that
  separate the worst step from nominal (376.19 K) from the 380 K limit. Below it the
  carried state can neither be seen by the sensor nor erode the margin materially.

Scope. The present source and target, A10 amplitudes, corner excursions of 120 s, a
rest of 600 s, pairs of excursions. It is evidence about these conditions, not a
guarantee for other plants, amplitudes, holds, rests or longer histories, and the
rest time is measured here, not optimised.

Run from the repository root (about 15 s):

    python experiments/03_p3_recovery_and_pairs.py

Outputs go to PT_DATA_DIR/experiments/m0_e03b/<run id> (git-ignored), one directory
per run. They are diagnostic artefacts of the simulator and may contain hidden
parameters.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from process_transfer import __version__  # noqa: E402
from process_transfer.config import load_true_plant  # noqa: E402
from process_transfer.cstr_variables import nominal_inputs  # noqa: E402
from process_transfer.data.paths import repository_root  # noqa: E402
from process_transfer.data.provenance import (  # noqa: E402
    copy_with_fingerprints,
    environment,
    git_state,
    new_run_directory,
)
from process_transfer.simulation import cstr_true  # noqa: E402
from process_transfer.simulation.checks import (  # noqa: E402
    BALANCE_TOLERANCE,
    TEMPERATURE_ENVELOPE,
    check_trajectory,
)
from process_transfer.simulation.cstr_true import TrueCSTRParameters  # noqa: E402
from process_transfer.simulation.excitation import excursions_with_rest  # noqa: E402
from process_transfer.simulation.integration import simulate_piecewise  # noqa: E402
from process_transfer.simulation.steady_state import find_steady_states  # noqa: E402

# =========================================================================== #
# Fixed before the sweep was run
# =========================================================================== #

A10 = (0.10, 0.10, 5.0, 5.0)  # relative q, relative C_Af, kelvin T_f, kelvin T_c (D-018)
HOLD = 120.0  # s, excursion (D-019)
REST = 600.0  # s at the nominal inputs (D-019)
SAMPLE_PERIOD = 0.1  # s

RECOVERY_TOLERANCE_T = 0.005  # K
RECOVERY_TOLERANCE_CA = 0.038  # mol/m^3
PEAK_AGREEMENT = 0.05  # K

PLANTS = ("source", "target")
INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PLANT_COLOR = {"source": "#2a78d6", "target": "#eb6834"}  # as in every figure of M0
LIMIT_COLOR = "#d03b3b"


@dataclass(frozen=True)
class Plant:
    name: str
    parameters: TrueCSTRParameters
    nominal_inputs: np.ndarray
    nominal_state: np.ndarray
    slowest_decay: float  # 1/s, largest real part of the eigenvalues at nominal (negative)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return cstr_true.rhs(0.0, x, u, self.parameters)

    @property
    def amplitudes(self) -> np.ndarray:
        u = self.nominal_inputs
        return np.array([A10[0] * u[0], A10[1] * u[1], A10[2], A10[3]])


def load_plant(name: str) -> Plant:
    cfg = load_true_plant(repository_root() / "configs" / f"{name}_cstr.yaml")
    p, u = TrueCSTRParameters.from_config(cfg), nominal_inputs(cfg.plant)
    (steady,) = find_steady_states(lambda x: cstr_true.rhs(0.0, x, u, p), c_a_upper=u[1])
    return Plant(name, p, u, steady.state, steady.max_real_part)


def corners() -> list[np.ndarray]:
    return [np.array(signs, dtype=np.int64) for signs in itertools.product((1, -1), repeat=4)]


def sign_string(levels: np.ndarray) -> str:
    return "".join("+" if value > 0 else "-" for value in levels)


def part_1_recovery(plants: dict[str, Plant]) -> tuple[dict, dict]:
    print(
        f"\n== 1. Recovery: an excursion of {HOLD:.0f} s from nominal, "
        f"then {REST:.0f} s at nominal =="
    )
    results: dict[str, object] = {}
    decays: dict[str, list] = {}
    for plant in plants.values():
        worst_c = worst_t = 0.0
        worst_c_corner = worst_t_corner = ""
        peaks: dict[str, float] = {}
        decays[plant.name] = []
        all_accepted = True
        for corner in corners():
            segments = excursions_with_rest(
                plant.nominal_inputs, plant.amplitudes, corner[np.newaxis, :], HOLD, REST
            )
            trajectory = simulate_piecewise(plant.f, plant.nominal_state, segments, SAMPLE_PERIOD)
            check = check_trajectory(trajectory, plant.parameters)
            all_accepted &= check.accepted
            peaks[sign_string(corner)] = check.refined_peak_temperature

            rest = trajectory.segments[1]
            deviation = np.abs(rest.states - plant.nominal_state)
            decays[plant.name].append((rest.times - rest.times[0], deviation))
            if deviation[-1, 0] > worst_c:
                worst_c, worst_c_corner = float(deviation[-1, 0]), sign_string(corner)
            if deviation[-1, 1] > worst_t:
                worst_t, worst_t_corner = float(deviation[-1, 1]), sign_string(corner)

        recovered = worst_c <= RECOVERY_TOLERANCE_CA and worst_t <= RECOVERY_TOLERANCE_T
        results[plant.name] = {
            "max_residual_c_a_mol_m3": worst_c,
            "corner_of_max_c_a": worst_c_corner,
            "max_residual_t_K": worst_t,
            "corner_of_max_t": worst_t_corner,
            "linear_decay_factor_over_rest": float(np.exp(plant.slowest_decay * REST)),
            "all_trajectories_accepted": bool(all_accepted),
            "within_tolerance": bool(recovered),
            "peak_K_from_exact_nominal": peaks,
        }
        print(
            f"  {plant.name:6s} largest residual after the rest: {worst_c:.3e} mol/m^3 "
            f"({worst_c_corner}), {worst_t:.3e} K ({worst_t_corner}); tolerance "
            f"{RECOVERY_TOLERANCE_CA} mol/m^3 and {RECOVERY_TOLERANCE_T} K -> "
            f"{'within' if recovered else 'OUTSIDE'}"
        )
        print(
            f"         slowest mode at nominal decays by exp({plant.slowest_decay * 60:.3f}/min x "
            f"{REST / 60:.0f} min) = {np.exp(plant.slowest_decay * REST):.2e} over the rest"
        )
    return results, decays


def part_2_pairs(plants: dict[str, Plant], recovery: dict) -> dict:
    print("\n== 2. Carried state: 256 ordered pairs of excursions, repeats included ==")
    results: dict[str, object] = {}
    for plant in plants.values():
        from_nominal = recovery[plant.name]["peak_K_from_exact_nominal"]
        worst_peak, worst_pair = -np.inf, ""
        worst_gap, worst_gap_pair = 0.0, ""
        rejected: list[str] = []
        for first, second in itertools.product(corners(), repeat=2):
            segments = excursions_with_rest(
                plant.nominal_inputs, plant.amplitudes, np.array([first, second]), HOLD, REST
            )
            trajectory = simulate_piecewise(plant.f, plant.nominal_state, segments, SAMPLE_PERIOD)
            label = f"{sign_string(first)} -> {sign_string(second)}"
            check = check_trajectory(trajectory, plant.parameters)
            if not check.accepted:
                rejected.append(label)
            if check.refined_peak_temperature > worst_peak:
                worst_peak, worst_pair = check.refined_peak_temperature, label

            # the second excursion and its rest, with the state carried over from the first
            second_half = dataclasses.replace(trajectory, segments=trajectory.segments[2:])
            gap = abs(second_half.refined_peak(1)[0] - from_nominal[sign_string(second)])
            if gap > worst_gap:
                worst_gap, worst_gap_pair = float(gap), label

        agrees = worst_gap <= PEAK_AGREEMENT
        results[plant.name] = {
            "pairs": 256,
            "rejected": rejected,
            "max_peak_K": float(worst_peak),
            "pair_of_max_peak": worst_pair,
            "max_peak_difference_from_exact_nominal_K": worst_gap,
            "pair_of_max_difference": worst_gap_pair,
            "within_tolerance": bool(agrees),
        }
        print(
            f"  {plant.name:6s} rejected {len(rejected)} of 256; largest peak {worst_peak:.4f} K "
            f"on {worst_pair}"
        )
        print(
            f"         second excursion against the same excursion from exact nominal: largest "
            f"difference {worst_gap:.2e} K on {worst_gap_pair}; tolerance {PEAK_AGREEMENT} K -> "
            f"{'within' if agrees else 'OUTSIDE'}"
        )
    return results


def figure_recovery(decays: dict, directory) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor=SURFACE)
    panels = (
        (0, "|C_A - nominal|, mol/m^3", RECOVERY_TOLERANCE_CA),
        (1, "|T - nominal|, K", RECOVERY_TOLERANCE_T),
    )
    for ax, (column, label, tolerance) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for plant_name in PLANTS:
            for index, (times, deviation) in enumerate(decays[plant_name]):
                positive = deviation[:, column] > 0.0
                ax.semilogy(
                    times[positive],
                    deviation[positive, column],
                    color=PLANT_COLOR[plant_name],
                    linewidth=0.9,
                    alpha=0.55,
                    label=plant_name if index == 0 else None,
                )
        ax.axhline(tolerance, color=LIMIT_COLOR, linewidth=1.0)
        ax.annotate(
            f"tolerance {tolerance:g}",
            (0.0, tolerance),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
        ax.set_xlabel("time at the nominal inputs, s", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel(label, fontsize=9, color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.suptitle(
        "Return to the nominal steady state after each of the 16 corner excursions (P3)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(directory / "fig1_recovery.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    started = time.perf_counter()
    print(f"process_transfer {__version__}; M0-E03b recovery and carried state under P3")
    state = git_state()
    directory = new_run_directory("m0_e03b", state)
    if not state["code_identified"]:
        print(f"  NOTE: the code of this run is not fully identified: {state['reason']}")
    plants = {name: load_plant(name) for name in PLANTS}
    defaults = inspect.signature(simulate_piecewise).parameters
    configurations = [repository_root() / "configs" / f"{name}_cstr.yaml" for name in PLANTS]

    summary: dict[str, object] = {
        "provenance": {
            "experiment": "M0-E03b",
            "run_id": directory.name,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git": state,
            "configurations": copy_with_fingerprints(configurations, directory / "configs"),
            "environment": environment(),
            "protocol": {"amplitudes_A10": A10, "hold_s": HOLD, "rest_s": REST, "corners": 16},
            "tolerances": {
                "recovery_t_K": RECOVERY_TOLERANCE_T,
                "recovery_c_a_mol_m3": RECOVERY_TOLERANCE_CA,
                "peak_agreement_K": PEAK_AGREEMENT,
            },
            "criteria": {
                "temperature_envelope_K": list(TEMPERATURE_ENVELOPE),
                "balance_tolerance": BALANCE_TOLERANCE,
                "definition": "process_transfer.simulation.checks.check_trajectory",
            },
            "integration": {
                "method": defaults["method"].default,
                "rtol": defaults["rtol"].default,
                "atol": defaults["atol"].default,
                "sample_period_s": SAMPLE_PERIOD,
                "restart": "one solver call per input segment; the state is never reset",
            },
            "contains_hidden_parameters": True,
        }
    }
    timings: dict[str, float] = {}
    tick = time.perf_counter()
    summary["recovery"], decays = part_1_recovery(plants)
    timings["recovery"] = round(time.perf_counter() - tick, 1)
    tick = time.perf_counter()
    summary["pairs"] = part_2_pairs(plants, summary["recovery"])
    timings["pairs"] = round(time.perf_counter() - tick, 1)
    figure_recovery(decays, directory)
    timings["total"] = round(time.perf_counter() - started, 1)
    summary["timings_s"] = timings

    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nrun times, s: {timings}")
    print(f"figure and summary.json written to {directory}")


if __name__ == "__main__":
    main()
