"""M1-E01, part 3: the oracle diagnostic, kept apart (``docs/m1_plan.md``, section 7.4; the
registration is in ``docs/experiment_log.md``).

This script reads the truth: the hidden physics of the target, its exact nominal steady
state and its noise-free trajectories. Its outputs go to PT_DATA_DIR/experiments/
m1_e01_oracle/<run id> only, marked as holding hidden parameters, and nothing in it is used to
choose or tune an ordinary model. The parts of M1-E01 on available information are another
script, ``experiments/m1_e01_identifiability.py``, which reads none of this.

Windows. The 16 corner windows of P3 on the target, each started from the exact nominal
steady state: 120 s at the corner and then the nominal inputs to 660 s, with the 110 scored
instants of section 4.3 of the plan, 6 s to 660 s after the onset. The true trajectories are
integrated as in M0 (LSODA, rtol = atol = 1e-9, restarted at every change of the inputs,
stored every 0.1 s), and again at 1e-11 to estimate the error of the first.

1. The pseudo-true fit. The first-order model of the modeller fitted to the noise-free true
   states at the scored instants of the 16 windows, every window started from the exact
   nominal steady state, with equal weight per corner (one window each, 110 readings each)
   and the objective of section 9.1, by the procedure of MR (``fit_mechanistic``, its five
   declared starts). Its minimiser, called pseudo-true, is defined by these windows and this
   objective. It is not the true value of any parameter. It would be the limit of MR as the
   budget grows if the initial states were exact, since uniform draws of the corners tend to
   equal weights; it differs from that limit through the error of the context means, the
   state an excursion carries into the next, and at finite budgets the bias of the
   estimator.
2. The kinetic mismatch alone, in the manner of M0-E08. Variant K: the true plant with its
   saturating rate replaced by a first-order rate with the same E/R, anchored to give the
   true rate at the nominal steady state (k0 / (1 + K_sat C_A,nominal), K_sat = 0), and the
   true conductance UA(T). Beside it, variant B of M0-E08 on the same windows: the true
   kinetics with the conductance constant at UA_true(T_nominal), the thermal mismatch alone.
   Both are anchored so that the nominal steady state is theirs too; each is compared with
   the true plant, A, on the 16 windows.

Run from the repository root, with PT_DATA_DIR set to a directory outside the repository:

    python experiments/m1_e01_oracle.py

The exit code is 0 only if every mandatory check of the registration holds: A accepted in
every window, K and B physically valid, both anchored at the nominal steady state, the
differences resolved by the integration, and a pseudo-true fit selected.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from process_transfer.config import load_sensors  # noqa: E402
from process_transfer.data.paths import data_dir, repository_root  # noqa: E402
from process_transfer.data.provenance import (  # noqa: E402
    copy_with_fingerprints,
    environment,
    git_state,
    new_run_directory,
)
from process_transfer.evaluation.metrics import Role, evaluate  # noqa: E402
from process_transfer.evaluation.plant import KnownPlant  # noqa: E402
from process_transfer.evaluation.windows import (  # noqa: E402
    CONTEXT_READINGS,
    P3_LAYOUT,
    Window,
    WindowData,
)
from process_transfer.generation.plants import VirtualPlant, load_virtual_plant  # noqa: E402
from process_transfer.models.fitting import (  # noqa: E402
    DEFAULT_FIT_SETTINGS,
    FitResult,
    FitSettings,
    fit_mechanistic,
)
from process_transfer.models.mechanistic import MechanisticModel, modeller_values  # noqa: E402
from process_transfer.models.rollout import predict_window  # noqa: E402
from process_transfer.simulation import cstr_true  # noqa: E402
from process_transfer.simulation.checks import check_trajectory, comparison_is_valid  # noqa: E402
from process_transfer.simulation.cstr_true import (  # noqa: E402
    TrueCSTRParameters,
    conductance,
    reaction_rate,
)
from process_transfer.simulation.integration import (  # noqa: E402
    InputSegment,
    RightHandSide,
    Trajectory,
    simulate_piecewise,
)
from process_transfer.simulation.operating_run import sensor_sample_indices  # noqa: E402
from process_transfer.simulation.protocols import (  # noqa: E402
    P3_HOLD,
    a10_amplitudes,
    corner_label,
    corner_levels,
)
from process_transfer.simulation.steady_state import find_steady_states  # noqa: E402

# =========================================================================== #
# Fixed before the first run
# =========================================================================== #

CONFIGS = repository_root() / "configs"
PLANT_FILE = CONFIGS / "target_cstr.yaml"
SENSORS_FILE = CONFIGS / "sensors_cstr.yaml"
SENSOR_PERIOD = 6.0  # s, D-020
WINDOW_DURATION = SENSOR_PERIOD * P3_LAYOUT.scored_readings  # 660 s
SIMULATION_PERIOD = 0.1  # s
TOLERANCE = 1.0e-9  # rtol = atol of the integration of M0
TIGHT_TOLERANCE = 1.0e-11
RESOLUTION_FRACTION = 0.01  # the integration error must be at most this fraction of an effect
# the right-hand side of a variant at the nominal steady state, against the feed terms, as
# generation.plants requires of the plant itself
ANCHOR_FRACTION = 1.0e-9
FIT_SETTINGS = DEFAULT_FIT_SETTINGS
STATE_NAMES = ("C_A", "T")

INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID_COLOR, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
EXCURSION_SHADE = "#e9e8e1"
VARIANT_COLOR = {"K": "#b8860b", "B": "#2a78d6", "pseudo-true": "#eb6834"}


def plain(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | range):
        return [plain(item) for item in value]
    return value


# =========================================================================== #
# The variants and their anchoring
# =========================================================================== #


def kinetic_variant(
    parameters: TrueCSTRParameters, nominal_state: np.ndarray
) -> TrueCSTRParameters:
    """First-order kinetics with the true E/R and the true rate at the nominal steady state;
    the true conductance UA(T)."""
    c_a = float(nominal_state[0])
    return dataclasses.replace(
        parameters,
        k0=parameters.k0 / (1.0 + parameters.saturation_constant * c_a),
        saturation_constant=0.0,
    )


def thermal_variant(
    parameters: TrueCSTRParameters, nominal_state: np.ndarray
) -> TrueCSTRParameters:
    """Variant B of M0-E08: the true kinetics, the conductance constant at UA_true(T_nominal)."""
    anchored = float(conductance(float(nominal_state[1]), parameters))
    return dataclasses.replace(parameters, ua_ref=anchored, alpha=0.0)


def anchoring(plant: VirtualPlant, variant: TrueCSTRParameters) -> dict[str, object]:
    x, u, p = plant.nominal_state, plant.nominal_inputs, plant.parameters
    dilution = u[0] / p.volume
    feed = np.array([dilution * u[1], dilution * u[2]])
    rhs = cstr_true.rhs(0.0, x, u, variant)
    rate_true = float(reaction_rate(x[0], x[1], p))
    rate_variant = float(reaction_rate(x[0], x[1], variant))
    ua_true = float(conductance(float(x[1]), p))
    ua_variant = float(conductance(float(x[1]), variant))
    return {
        "rate_true": rate_true,
        "rate_variant": rate_variant,
        "rate_relative_difference": abs(rate_variant - rate_true) / rate_true,
        "conductance_true_W_per_K": ua_true,
        "conductance_variant_W_per_K": ua_variant,
        "rhs_at_nominal_over_feed_terms": (rhs / feed).tolist(),
        "closed": bool(np.all(np.abs(rhs) <= ANCHOR_FRACTION * feed)),
    }


# =========================================================================== #
# The windows
# =========================================================================== #


def window_segments(nominal: np.ndarray, signs: np.ndarray) -> list[InputSegment]:
    corner = nominal + signs * a10_amplitudes(nominal)
    return [InputSegment(P3_HOLD, corner), InputSegment(WINDOW_DURATION - P3_HOLD, nominal)]


def simulate_windows(
    f: RightHandSide, x0: np.ndarray, nominal: np.ndarray, tolerance: float
) -> list[Trajectory]:
    return [
        simulate_piecewise(
            f,
            x0,
            window_segments(nominal, signs),
            SIMULATION_PERIOD,
            rtol=tolerance,
            atol=tolerance,
        )
        for signs in corner_levels()
    ]


def scored_states(trajectory: Trajectory) -> np.ndarray:
    """The states at the 110 scored instants, 6 s to 660 s after the onset."""
    indices = sensor_sample_indices(trajectory, SENSOR_PERIOD)
    states = trajectory.states[indices[1:]]
    if len(states) != P3_LAYOUT.scored_readings:
        raise ValueError(f"{len(states)} scored instants, not {P3_LAYOUT.scored_readings}")
    return np.array(states)


def window_inputs(nominal: np.ndarray, signs: np.ndarray) -> np.ndarray:
    inputs = np.tile(nominal, (P3_LAYOUT.scored_readings, 1))
    inputs[: P3_LAYOUT.hold] = nominal + signs * a10_amplitudes(nominal)
    return inputs


def truth_windows(
    states: list[np.ndarray], x0: np.ndarray, nominal: np.ndarray, sigma: np.ndarray
) -> tuple[WindowData, ...]:
    """The corner windows as a model is handed them: the context is ten copies of the exact
    starting state, so that its mean is that state, and the scored readings are exact."""
    windows = []
    for signs, scored in zip(corner_levels(), states, strict=True):
        window = Window(f"oracle.corner.{corner_label(signs)}", 1, CONTEXT_READINGS, P3_LAYOUT)
        windows.append(
            WindowData(
                window=window,
                sample_period=SENSOR_PERIOD,
                noise_std=sigma,
                onset_time=SENSOR_PERIOD * CONTEXT_READINGS,
                context=np.tile(x0, (CONTEXT_READINGS, 1)),
                scored=scored,
                inputs=window_inputs(nominal, signs),
            )
        )
    return tuple(windows)


# =========================================================================== #
# Comparisons
# =========================================================================== #


def differences(
    variant: list[np.ndarray], truth: list[np.ndarray], sigma: np.ndarray, labels: list[str]
) -> dict[str, object]:
    """Variant minus truth at the scored instants, in sigmas: largest and root mean square
    per channel, pooled over the windows, for each phase and for each window."""
    delta = np.stack([v - t for v, t in zip(variant, truth, strict=True)]) / sigma
    phases = {"whole window": slice(None), **dict(P3_LAYOUT.phase_slices())}
    found: dict[str, object] = {}
    for label, positions in phases.items():
        part = delta[:, positions]
        found[label] = {
            name: {
                "max_abs": float(np.max(np.abs(part[..., k]))),
                "rms": float(np.sqrt(np.mean(part[..., k] ** 2))),
            }
            for k, name in enumerate(STATE_NAMES)
        }
    found["by_window"] = {
        label: {
            name: {
                "max_abs": float(np.max(np.abs(delta[i, :, k]))),
                "rms": float(np.sqrt(np.mean(delta[i, :, k] ** 2))),
            }
            for k, name in enumerate(STATE_NAMES)
        }
        for i, label in enumerate(labels)
    }
    return found


def dense_largest(variant: list[Trajectory], truth: list[Trajectory]) -> list[float]:
    """The largest |variant - truth| of each state on the dense grid, in its unit."""
    return np.max(
        [np.max(np.abs(v.states - t.states), axis=0) for v, t in zip(variant, truth, strict=True)],
        axis=0,
    ).tolist()


def integration_error(project: list[Trajectory], tight: list[Trajectory]) -> list[float]:
    return dense_largest(project, tight)


def pseudo_true(
    windows: tuple[WindowData, ...], known: KnownPlant, settings: FitSettings | None = None
) -> tuple[FitResult, dict[str, object] | None]:
    """The fit of the first-order model to the noise-free windows, and its errors."""
    modeller = modeller_values()
    settings = FIT_SETTINGS if settings is None else settings
    fit = fit_mechanistic("pseudo-true", windows, known, modeller, settings)
    if fit.parameters is None:
        return fit, None
    model = fit.model(known)
    predictions = [predict_window(model, data) for data in windows]
    evaluation = evaluate(windows, predictions, Role.FITTING)
    rms_in_sigmas = {
        "pooled": dict(
            zip(STATE_NAMES, np.sqrt(evaluation.pooled.normalised_mse).tolist(), strict=True)
        ),
        **{
            name: dict(zip(STATE_NAMES, np.sqrt(score.normalised_mse).tolist(), strict=True))
            for name, score in evaluation.by_phase
        },
    }
    return fit, {
        "evaluation": evaluation,
        "rms_in_sigmas": rms_in_sigmas,
        "predictions": predictions,
    }


def model_steady_states(model: MechanisticModel, nominal: np.ndarray) -> list[list[float]]:
    found = find_steady_states(lambda x: model.rhs(x, nominal), c_a_upper=float(nominal[1]))
    return [steady.state.tolist() for steady in found]


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
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_traces(
    series: dict[str, list[np.ndarray]],
    truth: list[np.ndarray],
    sigma: np.ndarray,
    title: str,
    path: Path,
) -> Path:
    """Each series minus the truth at the scored instants, in sigmas, one line per window."""
    minutes = SENSOR_PERIOD * np.arange(1, P3_LAYOUT.scored_readings + 1) / 60.0
    fig, axes = plt.subplots(
        len(series),
        2,
        figsize=(12.0, 3.0 * len(series) + 0.8),
        sharex=True,
        squeeze=False,
        facecolor=SURFACE,
    )
    for row, (label, values) in enumerate(series.items()):
        for column, name in enumerate(STATE_NAMES):
            ax = axes[row, column]
            style(ax)
            ax.axvspan(0.0, P3_HOLD / 60.0, color=EXCURSION_SHADE, linewidth=0)
            for v, t in zip(values, truth, strict=True):
                ax.plot(
                    minutes,
                    (v[:, column] - t[:, column]) / sigma[column],
                    color=VARIANT_COLOR[label],
                    linewidth=0.7,
                    alpha=0.8,
                )
            for sign in (1.0, -1.0):
                ax.axhline(sign, color=INK_SECONDARY, linewidth=0.6, linestyle=":")
            ax.axhline(0.0, color=AXIS, linewidth=0.6)
            ax.set_ylabel(f"{label} - truth, {name} in sigmas", fontsize=9, color=INK_SECONDARY)
            if row == len(series) - 1:
                ax.set_xlabel("minutes after the onset", fontsize=9, color=INK_SECONDARY)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# =========================================================================== #
# Main
# =========================================================================== #


def main() -> int:
    started = time.perf_counter()
    data_directory, repository = data_dir().resolve(), repository_root().resolve()
    if data_directory == repository or repository in data_directory.parents:
        raise SystemExit(
            f"PT_DATA_DIR resolves to {data_directory}, inside the repository {repository}; the "
            "outputs "
            "of the oracle hold hidden parameters and must go to a new directory outside it"
        )
    state = git_state()
    directory = new_run_directory("m1_e01_oracle", state)
    if not state["code_identified"]:
        print(f"NOTE: the code of this run is not fully identified: {state['reason']}")
    plant = load_virtual_plant(PLANT_FILE)
    sensors = load_sensors(SENSORS_FILE)
    by_variable = {sensor.variable: sensor.noise_std.si for sensor in sensors.sensors}
    sigma = np.array([by_variable[name] for name in STATE_NAMES], dtype=np.float64)
    x0, nominal, p = plant.nominal_state, plant.nominal_inputs, plant.parameters
    known = KnownPlant(
        "target",
        p.volume,
        p.density,
        p.heat_capacity,
        p.reaction_enthalpy,
        tuple(float(v) for v in nominal),
    )

    variants = {"K": kinetic_variant(p, x0), "B": thermal_variant(p, x0)}
    rhs = {"A": plant.f}
    for label, parameters in variants.items():
        rhs[label] = lambda x, u, parameters=parameters: cstr_true.rhs(0.0, x, u, parameters)
    project = {label: simulate_windows(f, x0, nominal, TOLERANCE) for label, f in rhs.items()}
    tight = {label: simulate_windows(f, x0, nominal, TIGHT_TOLERANCE) for label, f in rhs.items()}
    scored = {label: [scored_states(t) for t in project[label]] for label in rhs}

    labels = [corner_label(signs) for signs in corner_levels()]
    check_a = [check_trajectory(t, p) for t in project["A"]]
    validity = {"A accepted in every window": all(c.accepted for c in check_a)}
    for label, parameters in variants.items():
        checks = [check_trajectory(t, parameters) for t in project[label]]
        validity[f"{label} physically valid in every window"] = all(
            comparison_is_valid(a, c) for a, c in zip(check_a, checks, strict=True)
        )
    anchors = {label: anchoring(plant, parameters) for label, parameters in variants.items()}
    errors = {label: integration_error(project[label], tight[label]) for label in rhs}
    effects = {}
    resolved = {}
    for label in variants:
        largest = dense_largest(project[label], project["A"])
        worst = [max(errors["A"][k], errors[label][k]) for k in range(2)]
        resolved[label] = all(
            w <= RESOLUTION_FRACTION * d for w, d in zip(worst, largest, strict=True)
        )
        effects[label] = {
            "sensor_grid_in_sigmas": differences(scored[label], scored["A"], sigma, labels),
            "dense_grid_largest_abs": dict(zip(("C_A_mol_per_m3", "T_K"), largest, strict=True)),
            "integration_error_largest_abs": worst,
            "resolved": resolved[label],
        }
    peaks = {label: float(max(np.max(t.states[:, 1]) for t in project[label])) for label in rhs}

    windows = truth_windows(scored["A"], x0, nominal, sigma)
    fit, found = pseudo_true(windows, known)
    pseudo = {"fit": fit}
    if found is not None:
        parameters = fit.parameters
        model = fit.model(known)
        pseudo.update(
            {
                "k_350_per_s": parameters.k_350,
                "activation_temperature_K": parameters.activation_temperature,
                "ua_W_per_K": parameters.ua,
                "k0_per_s": parameters.k0,
                "objective_J": fit.objective,
                "rms_in_sigmas": found["rms_in_sigmas"],
                "by_window_J": {
                    key[0].removeprefix("oracle.corner."): score.j
                    for key, score in found["evaluation"].by_window
                },
                "spread_over_starts": fit.spread(),
                "steady_states_of_the_fitted_model": model_steady_states(model, nominal),
            }
        )
    initial_exact = all(np.array_equal(w.initial_state, x0) for w in windows)
    initial_offset = max(float(np.max(np.abs(w.initial_state - x0))) for w in windows)
    checks = {
        **validity,
        "K and B anchored at the nominal steady state": all(a["closed"] for a in anchors.values()),
        "the differences are resolved by the integration": all(resolved.values()),
        "a pseudo-true fit was selected": fit.parameters is not None,
    }
    figures = [
        figure_traces(
            {label: scored[label] for label in variants},
            scored["A"],
            sigma,
            "M1-E01 part 3, oracle: the kinetic mismatch alone (K) and the thermal mismatch alone "
            "(B) against the true plant,\n16 corner windows from the exact nominal steady "
            "state; excursion shaded, dotted at one sigma",
            directory / "fig1_mismatch_alone.png",
        )
    ]
    if found is not None:
        figures.append(
            figure_traces(
                {"pseudo-true": found["predictions"]},
                scored["A"],
                sigma,
                "M1-E01 part 3, oracle: the pseudo-true first-order fit against the true plant,\n"
                "on its 16 corner windows, the windows it was fitted on",
                directory / "fig2_pseudo_true_residuals.png",
            )
        )
    summary = {
        "what": "M1-E01, part 3: an oracle diagnostic. It holds hidden parameters and exact "
        "states, and nothing in it may choose or tune an ordinary model.",
        "provenance": {
            "experiment": "M1-E01, part 3 (oracle)",
            "run_id": directory.name,
            "data_dir": str(data_directory),
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git": state,
            "environment": environment(),
            "configurations": copy_with_fingerprints(
                [PLANT_FILE, SENSORS_FILE, CONFIGS / "modeller_cstr.yaml"], directory / "configs"
            ),
            "oracle": True,
            "contains_hidden_parameters": True,
        },
        "fixed_before_the_run": {
            "simulation_period": SIMULATION_PERIOD,
            "tolerance": TOLERANCE,
            "tight_tolerance": TIGHT_TOLERANCE,
            "resolution_fraction": RESOLUTION_FRACTION,
            "anchor_fraction": ANCHOR_FRACTION,
            "fit_settings": FIT_SETTINGS,
            "window_duration_s": WINDOW_DURATION,
            "corners": [corner_label(c) for c in corner_levels()],
        },
        "truth": {
            "parameters": p,
            "nominal_state": x0,
            "nominal_inputs": nominal,
            "conductance_at_nominal_W_per_K": float(conductance(float(x0[1]), p)),
            "rate_at_nominal": float(reaction_rate(x0[0], x0[1], p)),
        },
        "noise_std": sigma,
        "initial_state_of_every_window_is_the_nominal_state_bit_for_bit": initial_exact,
        "largest_offset_of_an_initial_state": initial_offset,
        "integration_error_largest_abs": errors,
        "peak_temperatures_K": peaks,
        "variants": {
            label: {"parameters": v, "anchoring": anchors[label]} for label, v in variants.items()
        },
        "mismatch_alone": effects,
        "pseudo_true": pseudo,
        "checks": checks,
        "figures": [path.name for path in figures],
        "seconds": time.perf_counter() - started,
    }
    (directory / "summary.json").write_text(json.dumps(plain(summary), indent=2), encoding="utf-8")

    print(f"M1-E01 part 3, oracle; written to {directory}")
    for label in variants:
        pooled = effects[label]["sensor_grid_in_sigmas"]["whole window"]
        print(
            f"  {label} - truth: rms {pooled['C_A']['rms']:.3f} / {pooled['T']['rms']:.3f} sigma, "
            f"max {pooled['C_A']['max_abs']:.3f} / {pooled['T']['max_abs']:.3f} sigma (C_A / T)"
        )
    if found is not None:
        print(
            f"  pseudo-true: k_350 {fit.parameters.k_350:.6g} 1/s, E/R "
            f"{fit.parameters.activation_temperature:.1f} K, UA {fit.parameters.ua:.2f} W/K, "
            f"J {fit.objective:.4f}"
        )
    for label, holds in checks.items():
        print(f"  {label}: {'holds' if holds else 'FAILS'}")
    print(f"{summary['seconds']:.1f} s")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
