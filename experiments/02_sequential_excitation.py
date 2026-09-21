"""M0-E03: does the A10 excitation keep the plants inside the envelope when input
changes are chained on a 120 s clock?

Pre-registration
----------------
Everything in the block "Fixed before any comparison was run" below (amplitude
sets, protocols, seeds, clock, durations, acceptance criteria and cross-checks) was
written and committed before any protocol was compared. The only result known at
that point was the two-stage counterexample reported by the reviewer, reproduced in
part A. No seed is dropped, no state is clipped, and a protocol that fails is
reported as failing.

What the results can and cannot show: every statement produced here is evidence
about the sequences that were simulated. None of it is a guarantee over all input
sequences that a protocol can generate.

Run from the repository root (about five minutes):

    python experiments/02_sequential_excitation.py

Figures and a JSON summary are written under PT_DATA_DIR/m0_e03 (git-ignored).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from process_transfer import __version__  # noqa: E402
from process_transfer.config import load_true_plant  # noqa: E402
from process_transfer.cstr_variables import nominal_inputs  # noqa: E402
from process_transfer.data.paths import output_dir, repository_root  # noqa: E402
from process_transfer.simulation import cstr_true  # noqa: E402
from process_transfer.simulation.checks import (  # noqa: E402
    TEMPERATURE_ENVELOPE,
    TrajectoryCheck,
    check_trajectory,
)
from process_transfer.simulation.cstr_true import TrueCSTRParameters  # noqa: E402
from process_transfer.simulation.envelope import input_cases  # noqa: E402
from process_transfer.simulation.excitation import (  # noqa: E402
    binary_levels,
    levels_to_segments,
    limited_move_levels,
    separated_excursions,
)
from process_transfer.simulation.integration import (  # noqa: E402
    InputSegment,
    Trajectory,
    simulate_piecewise,
)
from process_transfer.simulation.steady_state import find_steady_states  # noqa: E402

# =========================================================================== #
# Fixed before any comparison was run
# =========================================================================== #

CLOCK = 120.0  # s between input changes (D-010: 2 min)
SEQUENCE_DURATION = 7200.0  # s per seeded sequence
SEEDS = tuple(range(20))  # every seed is reported; none is selected or dropped
SAMPLE_PERIOD = 0.1  # s
STEP_DURATION = 2400.0  # s, steps from the nominal steady state (as in M0-E01/E02)
SECOND_STAGE = 600.0  # s held after a transition, long enough for the whole transient
SETTLING_TIME = 3600.0  # s used to settle a plant at a corner before a transition
COLD_DWELLS = (120.0, 240.0, 600.0)  # s spent in the cold stage of an adversarial ramp
REST = 600.0  # s at nominal between the excursions of protocol P3

# Amplitudes as (relative q, relative C_Af, kelvin T_f, kelvin T_c)
A10 = (0.10, 0.10, 5.0, 5.0)  # the choice recorded in D-018
REDUCED_THERMAL = (0.10, 0.10, 2.5, 2.5)


@dataclass(frozen=True)
class Protocol:
    key: str
    description: str
    amplitudes: tuple[float, float, float, float]
    kind: str  # "binary", "limited" or "separated"
    rationale: str


PROTOCOLS = (
    Protocol(
        "P0",
        "A10, binary levels",
        A10,
        "binary",
        "The baseline: D-010 read as every input at -1 or +1 on each tick, so an input "
        "may cross its whole range at once.",
    ),
    Protocol(
        "P1",
        "thermal inputs +-2.5 K, binary levels",
        REDUCED_THERMAL,
        "binary",
        "The counterexample is driven by a 10 K swing of both thermal inputs; halving "
        "their amplitude halves that swing and keeps q and C_Af at +-10 %.",
    ),
    Protocol(
        "P2",
        "A10, three levels, at most one level per tick",
        A10,
        "limited",
        "The counterexample needs a jump from the coldest to the hottest level in one "
        "tick; forbidding it forces a pass through the nominal level.",
    ),
    Protocol(
        "P3",
        "A10, 120 s excursions separated by 600 s at nominal",
        A10,
        "separated",
        "Restores the premise of M0-E02, where every step started from the nominal "
        "steady state, at the price of much less excitation per hour.",
    ),
    Protocol(
        "P4",
        "thermal inputs +-2.5 K, three levels, at most one level per tick",
        REDUCED_THERMAL,
        "limited",
        "Both mitigations together, in case neither is enough alone.",
    ),
)

# Acceptance criteria, applied by process_transfer.simulation.checks.check_trajectory:
#   - the integrator succeeds on every segment (it raises otherwise);
#   - states are physical (C_A > 0, C_A never above the richest feed so far, T never
#     below the coldest stream so far);
#   - T stays inside TEMPERATURE_ENVELOPE = [335, 380] K, judged on the refined maximum;
#   - integrated mass and energy balances close to 1e-6 relative.
# A protocol "passes on the tested cases" for a plant only if every seeded sequence,
# every adversarial ramp and, for binary protocols, every corner-to-corner transition
# of its amplitude set is accepted.
#
# Cross-checks: the worst seeded sequence of every protocol and plant is recomputed
# with DOP853 at rtol = atol = 1e-12, and with LSODA sampled every 0.01 s.

PLANTS = ("source", "target")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
PLANT_COLOR = {"source": "#2a78d6", "target": "#eb6834"}  # categorical slots 1 and 2
LIMIT_COLOR = "#d03b3b"  # status "critical", always with a text label


# =========================================================================== #
# Plants and helpers
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


def absolute_amplitudes(plant: Plant, amplitudes: tuple[float, float, float, float]) -> np.ndarray:
    u = plant.nominal_inputs
    return np.array([amplitudes[0] * u[0], amplitudes[1] * u[1], amplitudes[2], amplitudes[3]])


def run(plant: Plant, x0: np.ndarray, segments: list[InputSegment], **kwargs: object) -> Trajectory:
    return simulate_piecewise(
        plant.f, x0, segments, kwargs.pop("sample_period", SAMPLE_PERIOD), **kwargs
    )


def corner_levels() -> list[np.ndarray]:
    return [np.array(signs, dtype=np.int64) for signs in itertools.product((1, -1), repeat=4)]


def sign_string(levels: np.ndarray) -> str:
    return "".join("+" if value > 0 else "-" if value < 0 else "0" for value in levels)


def segments_for(protocol: Protocol, plant: Plant, seed: int) -> list[InputSegment]:
    rng = np.random.default_rng(seed)
    amplitudes = absolute_amplitudes(plant, protocol.amplitudes)
    if protocol.kind == "separated":
        n_excursions = int(round(SEQUENCE_DURATION / (CLOCK + REST)))
        return separated_excursions(
            plant.nominal_inputs, amplitudes, n_excursions, CLOCK, REST, rng
        )
    n_ticks = int(round(SEQUENCE_DURATION / CLOCK))
    generator = binary_levels if protocol.kind == "binary" else limited_move_levels
    return levels_to_segments(plant.nominal_inputs, amplitudes, generator(n_ticks, 4, rng), CLOCK)


COLD = np.array([1, 1, -1, -1])  # q+, C_Af+, T_f-, T_c-
MIDDLE = np.array([1, 1, 0, 0])
HOT = np.array([1, 1, 1, 1])


def adversarial_ramp(protocol: Protocol, plant: Plant, cold_dwell: float) -> list[InputSegment]:
    """The fastest route from the coldest to the hottest thermal condition that the
    protocol allows, with q and C_Af high throughout."""
    u, amplitudes = plant.nominal_inputs, absolute_amplitudes(plant, protocol.amplitudes)

    def stage(levels: np.ndarray, duration: float) -> InputSegment:
        return InputSegment(duration, u + levels * amplitudes)

    if protocol.kind == "binary":
        return [stage(COLD, cold_dwell), stage(HOT, SECOND_STAGE)]
    if protocol.kind == "limited":
        return [stage(COLD, cold_dwell), stage(MIDDLE, CLOCK), stage(HOT, SECOND_STAGE)]
    zero = np.zeros(4)
    return [stage(COLD, CLOCK), stage(zero, REST), stage(HOT, CLOCK), stage(zero, REST)]


def summarise(check: TrajectoryCheck) -> dict[str, object]:
    return {
        "peak_K": round(check.refined_peak_temperature, 4),
        "min_K": round(check.min_temperature, 4),
        "c_a_range": [round(value, 2) for value in check.c_a_range],
        "seconds_above_limit": round(check.seconds_above_limit, 2),
        "relative_mass_residual": check.relative_mass_residual,
        "relative_energy_residual": check.relative_energy_residual,
        "states_physical": check.states_physical,
        "inside_envelope": check.inside_envelope,
        "balances_close": check.balances_close,
        "accepted": check.accepted,
    }


# =========================================================================== #
# Parts of the experiment
# =========================================================================== #


def part_a_counterexample(plants: dict[str, Plant]) -> tuple[dict, dict[str, Trajectory]]:
    print("\n== A. Two-stage counterexample at the A10 levels ==")
    results: dict[str, object] = {}
    trajectories: dict[str, Trajectory] = {}
    for plant in plants.values():
        segments = adversarial_ramp(PROTOCOLS[0], plant, CLOCK)
        segments = [segments[0], InputSegment(CLOCK, segments[1].inputs)]  # 120 s + 120 s
        by_sampling = {}
        for period in (1.0, 0.1, 0.01):
            trajectory = run(plant, plant.nominal_state, segments, sample_period=period)
            by_sampling[period] = (trajectory.peak(1)[0], trajectory.refined_peak(1)[0])
        trajectories[plant.name] = trajectory
        check = check_trajectory(trajectory, plant.parameters)
        by_method = {}
        for method in ("LSODA", "DOP853", "Radau"):
            strict = run(
                plant,
                plant.nominal_state,
                segments,
                sample_period=0.01,
                method=method,
                rtol=1e-12,
                atol=1e-12,
            )
            by_method[method] = strict.peak(1)
        end_cold = trajectory.segments[0].states[-1]
        print(
            f"  {plant.name}: end of cold stage C_A = {end_cold[0]:.2f} mol/m^3, "
            f"T = {end_cold[1]:.2f} K; peak {check.peak_temperature:.4f} K at "
            f"{check.peak_time - CLOCK:.2f} s after the change; "
            f"{check.seconds_above_limit:.2f} s above {TEMPERATURE_ENVELOPE[1]} K; "
            f"accepted = {check.accepted}"
        )
        print(
            "    sampled / refined peak: "
            + "; ".join(f"{h} s: {a:.4f} / {b:.4f}" for h, (a, b) in by_sampling.items())
        )
        print(
            "    strict tolerances (1e-12, 0.01 s): "
            + "; ".join(f"{m}: {v:.4f} K at {t - CLOCK:.2f} s" for m, (v, t) in by_method.items())
        )
        print(
            f"    balances: mass {check.relative_mass_residual:.1e}, "
            f"energy {check.relative_energy_residual:.1e}"
        )
        results[plant.name] = {
            **summarise(check),
            "seconds_after_change": round(check.peak_time - CLOCK, 2),
            "end_of_cold_stage": [round(float(v), 3) for v in end_cold],
            "sampled_and_refined_peak_by_period": {str(h): v for h, v in by_sampling.items()},
            "strict_peak_by_method": {m: round(v[0], 5) for m, v in by_method.items()},
        }
    return results, trajectories


def part_b_steps_from_nominal(plants: dict[str, Plant]) -> dict:
    print("\n== B. Reference: steps from the nominal steady state, 24 cases, 40 min ==")
    results: dict[str, object] = {}
    for label, amplitudes in (("A10", A10), ("reduced thermal", REDUCED_THERMAL)):
        for plant in plants.values():
            cases = input_cases(plant.nominal_inputs, absolute_amplitudes(plant, amplitudes))
            checks = {
                name: check_trajectory(
                    run(plant, plant.nominal_state, [InputSegment(STEP_DURATION, u)]),
                    plant.parameters,
                )
                for name, u in cases
            }
            worst = max(checks, key=lambda name: checks[name].refined_peak_temperature)
            failed = [name for name, check in checks.items() if not check.accepted]
            print(
                f"  {label:16s} {plant.name}: T in "
                f"[{min(c.min_temperature for c in checks.values()):.2f}, "
                f"{checks[worst].refined_peak_temperature:.2f}] K, hottest {worst}, "
                f"rejected {len(failed)} of {len(checks)}"
            )
            results[f"{label}/{plant.name}"] = {
                "hottest_case": worst,
                "rejected": failed,
                **summarise(checks[worst]),
            }
    return results


def part_c_transitions(plants: dict[str, Plant]) -> tuple[dict, dict[str, np.ndarray]]:
    print("\n== C. Transitions between corners: corner i, then corner j held 600 s ==")
    corners = corner_levels()
    results: dict[str, object] = {}
    matrices: dict[str, np.ndarray] = {}
    for label, amplitudes in (("A10", A10), ("reduced thermal", REDUCED_THERMAL)):
        for plant in plants.values():
            u, amp = plant.nominal_inputs, absolute_amplitudes(plant, amplitudes)
            settled = {}
            for corner in corners:
                trajectory = run(
                    plant,
                    plant.nominal_state,
                    [InputSegment(SETTLING_TIME, u + corner * amp)],
                    sample_period=10.0,
                )
                settled[sign_string(corner)] = trajectory.states[-1]
            for variant in ("120 s dwell from nominal", "settled at corner i"):
                peaks = np.full((16, 16), np.nan)
                rejected = 0
                for (i, first), (j, second) in itertools.product(enumerate(corners), repeat=2):
                    if i == j:
                        continue
                    hold = InputSegment(SECOND_STAGE, u + second * amp)
                    if variant.startswith("120"):
                        trajectory = run(
                            plant, plant.nominal_state, [InputSegment(CLOCK, u + first * amp), hold]
                        )
                    else:
                        trajectory = run(plant, settled[sign_string(first)], [hold])
                    check = check_trajectory(trajectory, plant.parameters)
                    peaks[i, j] = check.refined_peak_temperature
                    rejected += not check.accepted
                i, j = np.unravel_index(np.nanargmax(peaks), peaks.shape)
                key = f"{label}/{plant.name}/{variant}"
                matrices[key] = peaks
                results[key] = {
                    "worst_peak_K": round(float(peaks[i, j]), 4),
                    "worst_transition": f"{sign_string(corners[i])} -> {sign_string(corners[j])}",
                    "rejected": int(rejected),
                    "of": 240,
                }
                print(
                    f"  {label:16s} {plant.name:6s} {variant:26s} worst {peaks[i, j]:.2f} K on "
                    f"{sign_string(corners[i])} -> {sign_string(corners[j])} "
                    f"(order q, C_Af, T_f, T_c); rejected {rejected} of 240"
                )
    return results, matrices


def part_d_seeded_sequences(plants: dict[str, Plant]) -> tuple[dict, dict[str, list[float]]]:
    print(f"\n== D. Seeded sequences: {len(SEEDS)} seeds, {SEQUENCE_DURATION / 3600:.0f} h each ==")
    results: dict[str, object] = {}
    peaks_by_key: dict[str, list[float]] = {}
    for protocol in PROTOCOLS:
        for plant in plants.values():
            checks = [
                check_trajectory(
                    run(plant, plant.nominal_state, segments_for(protocol, plant, seed)),
                    plant.parameters,
                )
                for seed in SEEDS
            ]
            peaks = [check.refined_peak_temperature for check in checks]
            failed = [seed for seed, check in zip(SEEDS, checks, strict=True) if not check.accepted]
            not_physical = [s for s, c in zip(SEEDS, checks, strict=True) if not c.states_physical]
            open_balances = [s for s, c in zip(SEEDS, checks, strict=True) if not c.balances_close]
            key = f"{protocol.key}/{plant.name}"
            peaks_by_key[key] = peaks
            results[key] = {
                "peak_K_by_seed": [round(value, 3) for value in peaks],
                "min_K": round(min(check.min_temperature for check in checks), 3),
                "rejected_seeds": failed,
                "unphysical_seeds": not_physical,
                "open_balance_seeds": open_balances,
                "worst_seed": int(SEEDS[int(np.argmax(peaks))]),
                "max_seconds_above_limit": round(max(c.seconds_above_limit for c in checks), 1),
            }
            print(
                f"  {protocol.key} {plant.name:6s} peak over seeds: min {min(peaks):.2f}, "
                f"median {np.median(peaks):.2f}, max {max(peaks):.2f} K; rejected "
                f"{len(failed)} of {len(SEEDS)} seeds {failed if failed else ''}"
            )
    return results, peaks_by_key


def part_e_adversarial_ramps(plants: dict[str, Plant]) -> tuple[dict, dict[str, Trajectory]]:
    print("\n== E. Adversarial ramps: the fastest cold-to-hot route each protocol allows ==")
    results: dict[str, object] = {}
    trajectories: dict[str, Trajectory] = {}
    for protocol in PROTOCOLS:
        dwells = (CLOCK,) if protocol.kind == "separated" else COLD_DWELLS
        for plant in plants.values():
            for dwell in dwells:
                trajectory = run(
                    plant, plant.nominal_state, adversarial_ramp(protocol, plant, dwell)
                )
                check = check_trajectory(trajectory, plant.parameters)
                key = f"{protocol.key}/{plant.name}/{dwell:.0f}"
                results[key] = summarise(check)
                if dwell == CLOCK:
                    trajectories[f"{protocol.key}/{plant.name}"] = trajectory
                print(
                    f"  {protocol.key} {plant.name:6s} cold dwell {dwell:4.0f} s: peak "
                    f"{check.refined_peak_temperature:.2f} K, accepted = {check.accepted}"
                )
    return results, trajectories


def part_f_cross_checks(plants: dict[str, Plant], seeded: dict) -> dict:
    print("\n== F. Cross-checks of the worst seeded sequence of every protocol and plant ==")
    results: dict[str, object] = {}
    for protocol in PROTOCOLS:
        for plant in plants.values():
            key = f"{protocol.key}/{plant.name}"
            seed = seeded[key]["worst_seed"]
            segments = segments_for(protocol, plant, seed)
            reference = run(plant, plant.nominal_state, segments).refined_peak(1)[0]
            strict = run(
                plant, plant.nominal_state, segments, method="DOP853", rtol=1e-12, atol=1e-12
            ).refined_peak(1)[0]
            fine = run(plant, plant.nominal_state, segments, sample_period=0.01).peak(1)[0]
            results[key] = {
                "seed": seed,
                "lsoda_1e-9_0.1s_refined": round(reference, 5),
                "dop853_1e-12_0.1s_refined": round(strict, 5),
                "lsoda_1e-9_0.01s_sampled": round(fine, 5),
            }
            spread = max(reference, strict, fine) - min(reference, strict, fine)
            print(
                f"  {key} seed {seed:2d}: LSODA {reference:.4f} K, DOP853 strict {strict:.4f} K, "
                f"0.01 s sampling {fine:.4f} K; spread {spread:.1e} K"
            )
    return results


def verdicts(transitions: dict, seeded: dict, ramps: dict) -> dict:
    print("\n== Verdict on the tested cases (not a guarantee over all sequences) ==")
    results: dict[str, object] = {}
    for protocol in PROTOCOLS:
        amplitude_label = "A10" if protocol.amplitudes == A10 else "reduced thermal"
        for plant in PLANTS:
            seeds_ok = not seeded[f"{protocol.key}/{plant}"]["rejected_seeds"]
            ramp_keys = [k for k in ramps if k.startswith(f"{protocol.key}/{plant}/")]
            ramps_ok = all(ramps[k]["accepted"] for k in ramp_keys)
            if protocol.kind == "binary":
                transition_keys = [
                    k for k in transitions if k.startswith(f"{amplitude_label}/{plant}/")
                ]
                transitions_ok: bool | None = all(
                    transitions[k]["rejected"] == 0 for k in transition_keys
                )
            else:
                transitions_ok = None  # corner-to-corner jumps are not part of this protocol
            passed = seeds_ok and ramps_ok and transitions_ok is not False
            results[f"{protocol.key}/{plant}"] = {
                "seeded_sequences": seeds_ok,
                "adversarial_ramps": ramps_ok,
                "corner_transitions": transitions_ok,
                "passes_on_tested_cases": passed,
            }
            shown = "n/a" if transitions_ok is None else str(transitions_ok)
            print(
                f"  {protocol.key} {plant:6s} seeds {seeds_ok!s:5s} ramps {ramps_ok!s:5s} "
                f"transitions {shown:5s} -> {'PASSES' if passed else 'FAILS'} on the tested cases"
            )
    return results


# =========================================================================== #
# Figures
# =========================================================================== #


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3, width=0.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_limit(ax: plt.Axes, x_text: float) -> None:
    limit = TEMPERATURE_ENVELOPE[1]
    ax.axhline(limit, color=LIMIT_COLOR, linewidth=1.0)
    ax.annotate(
        f"{limit:.0f} K limit",
        (x_text, limit),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8,
        color=INK,
        ha="left",
        va="bottom",
    )


def figure_counterexample(trajectories: dict[str, Trajectory], directory) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True, facecolor=SURFACE)
    target = trajectories["target"]
    for ax in axes:
        style_axes(ax)
        ax.axvline(CLOCK, color=AXIS, linewidth=0.8)

    for column, name in ((2, "feed temperature T_f"), (3, "coolant temperature T_c")):
        axes[0].step(
            target.times, target.inputs[:, column], where="post", color=INK_SECONDARY, linewidth=1.6
        )
        axes[0].annotate(
            name,
            (target.times[-1], target.inputs[-1, column]),
            xytext=(-4, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            ha="right",
        )
    axes[0].set_ylabel("input temperature, K", fontsize=9, color=INK_SECONDARY)
    axes[0].set_title(
        "A cold stage stores reactant; heating then releases its heat at once",
        fontsize=11,
        color=INK,
        loc="left",
        pad=14,
    )
    axes[0].text(
        0.0,
        1.03,
        "Both plants from their nominal steady state; "
        "q = 110 L/min and C_Af = 0.55 mol/L throughout (A10 levels)",
        transform=axes[0].transAxes,
        fontsize=8,
        color=INK_SECONDARY,
    )

    for plant_name, trajectory in trajectories.items():
        colour = PLANT_COLOR[plant_name]
        axes[1].plot(
            trajectory.times, trajectory.states[:, 0], color=colour, linewidth=1.8, label=plant_name
        )
        axes[2].plot(
            trajectory.times, trajectory.states[:, 1], color=colour, linewidth=1.8, label=plant_name
        )
        peak, when = trajectory.peak(1)
        axes[2].plot(
            [when],
            [peak],
            "o",
            color=colour,
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        axes[2].annotate(
            f"{plant_name}: {peak:.1f} K",
            (when, peak),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            va="center",
        )
    axes[1].set_ylabel("C_A, mol/m^3", fontsize=9, color=INK_SECONDARY)
    axes[2].set_ylabel("reactor temperature T, K", fontsize=9, color=INK_SECONDARY)
    axes[2].set_xlabel("time, s", fontsize=9, color=INK_SECONDARY)
    draw_limit(axes[2], 2.0)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    axes[1].annotate(
        "input change",
        (CLOCK, axes[1].get_ylim()[1]),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
    )
    fig.tight_layout()
    fig.savefig(directory / "fig1_counterexample.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_ramps(trajectories: dict[str, Trajectory], directory) -> None:
    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(13.5, 3.6), sharey=True, facecolor=SURFACE)
    for ax, protocol in zip(axes, PROTOCOLS, strict=True):
        style_axes(ax)
        for plant_name in PLANTS:
            trajectory = trajectories[f"{protocol.key}/{plant_name}"]
            ax.plot(
                trajectory.times,
                trajectory.states[:, 1],
                color=PLANT_COLOR[plant_name],
                linewidth=1.6,
                label=plant_name,
            )
        draw_limit(ax, 0.0)
        ax.set_title(
            f"{protocol.key}: {protocol.description}", fontsize=8, color=INK, loc="left", wrap=True
        )
        ax.set_xlabel("time, s", fontsize=8, color=INK_SECONDARY)
    axes[0].set_ylabel("reactor temperature T, K", fontsize=9, color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.suptitle(
        "Fastest cold-to-hot route each protocol allows (cold stage 120 s, q and C_Af high)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(directory / "fig2_adversarial_ramps.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_seed_peaks(peaks_by_key: dict[str, list[float]], directory) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2), facecolor=SURFACE)
    style_axes(ax)
    rng = np.random.default_rng(0)  # horizontal jitter only, so that equal peaks do not overlap
    for position, protocol in enumerate(PROTOCOLS):
        for offset, plant_name in ((-0.17, "source"), (0.17, "target")):
            peaks = peaks_by_key[f"{protocol.key}/{plant_name}"]
            x = position + offset + rng.uniform(-0.09, 0.09, len(peaks))
            ax.plot(
                x,
                peaks,
                "o",
                color=PLANT_COLOR[plant_name],
                markersize=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=plant_name if position == 0 else None,
            )
    draw_limit(ax, -0.45)
    ax.set_xticks(range(len(PROTOCOLS)), [protocol.key for protocol in PROTOCOLS])
    ax.set_xlim(-0.5, len(PROTOCOLS) - 0.5)
    ax.set_ylabel("peak temperature of the sequence, K", fontsize=9, color=INK_SECONDARY)
    ax.set_title(
        f"Peak temperature of every seeded sequence ({len(SEEDS)} seeds, 2 h each, none dropped)",
        fontsize=11,
        color=INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(directory / "fig3_seed_peaks.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_transitions(matrices: dict[str, np.ndarray], directory) -> None:
    diverging = LinearSegmentedColormap.from_list(
        "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ee8f8e", "#b3262a"]
    )
    limit = TEMPERATURE_ENVELOPE[1]
    keys = (
        "A10/target/120 s dwell from nominal",
        "reduced thermal/target/120 s dwell from nominal",
    )
    highest = max(float(np.nanmax(matrices[key])) for key in keys)
    lowest = min(float(np.nanmin(matrices[key])) for key in keys)
    norm = TwoSlopeNorm(
        vmin=min(lowest, limit - 1.0), vcenter=limit, vmax=max(highest, limit + 1.0)
    )
    labels = [sign_string(corner) for corner in corner_levels()]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0), facecolor=SURFACE)
    for ax, key, title in zip(
        axes, keys, ("A10 amplitudes", "thermal inputs +-2.5 K"), strict=True
    ):
        image = ax.imshow(matrices[key], cmap=diverging, norm=norm)
        ax.set_xticks(
            range(16), labels, rotation=90, fontsize=7, color=INK_SECONDARY, family="monospace"
        )
        ax.set_yticks(range(16), labels, fontsize=7, color=INK_SECONDARY, family="monospace")
        ax.set_xlabel("second corner, held 600 s", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("first corner, held 120 s", fontsize=9, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.5)  # surface gap between cells
        ax.tick_params(which="both", length=0)
    bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label(
        f"peak temperature, K (neutral at the {limit:.0f} K limit)", fontsize=9, color=INK_SECONDARY
    )
    bar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    bar.outline.set_visible(False)
    fig.suptitle(
        "Target plant: peak temperature after a change from one corner to another "
        "(signs of q, C_Af, T_f, T_c)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(
        directory / "fig4_transitions_target.png", dpi=200, facecolor=SURFACE, bbox_inches="tight"
    )
    plt.close(fig)


# =========================================================================== #


def main() -> None:
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    started = time.perf_counter()
    print(f"process_transfer {__version__}; M0-E03 sequential excitation")
    plants = {name: load_plant(name) for name in PLANTS}
    directory = output_dir("m0_e03")

    timings: dict[str, float] = {}
    summary: dict[str, object] = {"version": __version__, "seeds": list(SEEDS)}

    def timed(label: str, function, *args):  # noqa: ANN001, ANN202
        tick = time.perf_counter()
        result = function(*args)
        timings[label] = round(time.perf_counter() - tick, 1)
        print(f"  [{label}: {timings[label]:.1f} s]")
        return result

    summary["A_counterexample"], counterexample = timed("A", part_a_counterexample, plants)
    summary["B_steps_from_nominal"] = timed("B", part_b_steps_from_nominal, plants)
    summary["C_transitions"], matrices = timed("C", part_c_transitions, plants)
    summary["D_seeded_sequences"], peaks_by_key = timed("D", part_d_seeded_sequences, plants)
    summary["E_adversarial_ramps"], ramps = timed("E", part_e_adversarial_ramps, plants)
    summary["F_cross_checks"] = timed(
        "F", part_f_cross_checks, plants, summary["D_seeded_sequences"]
    )
    summary["verdicts"] = verdicts(
        summary["C_transitions"], summary["D_seeded_sequences"], summary["E_adversarial_ramps"]
    )

    tick = time.perf_counter()
    figure_counterexample(counterexample, directory)
    figure_ramps(ramps, directory)
    figure_seed_peaks(peaks_by_key, directory)
    figure_transitions(matrices, directory)
    timings["figures"] = round(time.perf_counter() - tick, 1)
    timings["total"] = round(time.perf_counter() - started, 1)
    summary["timings_s"] = timings

    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nrun times, s: {timings}")
    print(f"figures and summary.json written to {directory}")


if __name__ == "__main__":
    main()
