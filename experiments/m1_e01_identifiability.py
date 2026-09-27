"""M1-E01, parts 1 and 2: what the data can determine about the parameters of the modeller's
model, from available information only (``docs/m1_plan.md``, section 7.4; the registration
is in ``docs/experiment_log.md``).

Part 1, a priori. The information about theta = (ln k_350, E/R, ln UA) of the first-order
model under P3, with the noise of D-020, at the parameters that hold the nominal steady
state observed in the development data of M0: the mean of the readings of the steady run of
the target in `m0-e06`. The information of each of the 16 corner windows of P3 is computed
once, and that of a design is the sum over its windows (``models.identifiability``). Three
covariances for each design: with the initial state exact, the bound with the initial state
known through its ten context readings, and the sandwich of the estimator that M1 uses. The
designs are the numbers of windows that MR (b) and MR_F (b - n_V) receive at the budgets of
the plan: the expected design, n / 16 windows at each corner, and 20 000 draws of the
corners as P3 draws them.

Part 2, on the development data. The profile of the fitting objective of MR against E/R on
the target runs of `m0-e05`: each run as a replicate, its nine windows, and the 27 windows
pooled. E/R is held at each point of a declared grid and k_350 and UA are fitted by the
procedure of MR from its starts that do not move E/R. The fit of MR with E/R free locates
the minimum of each profile.

What it reads. The exports `m0-e05` and `m0-e06`, opened and verified by
``open_export_directory`` and read only; the known parameters of the target from them; the
modeller's values of ``configs/modeller_cstr.yaml``. It imports nothing of the simulation,
of the generator or of the private branch (``tests/test_boundaries.py``). The oracle part of
M1-E01 is another script, ``experiments/m1_e01_oracle.py``.

Usage, with PT_DATA_DIR set to a directory outside the repository and the exports of M0
given explicitly:

    python experiments/m1_e01_identifiability.py --exports <PT_DATA_DIR of M0>/available/exports

Outputs go to PT_DATA_DIR/experiments/m1_e01/<run id>: summary.json and the figures. The
exit code is 0 only if every mandatory check of the registration holds.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

from process_transfer.config import ModellerConfig  # noqa: E402
from process_transfer.data.export import Export, open_export_directory  # noqa: E402
from process_transfer.data.paths import repository_root  # noqa: E402
from process_transfer.data.provenance import (  # noqa: E402
    copy_with_fingerprints,
    environment,
    git_state,
    new_run_directory,
)
from process_transfer.evaluation.budgets import validation_windows  # noqa: E402
from process_transfer.evaluation.plant import KnownPlant, read_known_plant  # noqa: E402
from process_transfer.evaluation.windows import (  # noqa: E402
    CONTEXT_READINGS,
    P3_LAYOUT,
    WindowData,
    find_windows,
    layout_for,
    window_data,
)
from process_transfer.models.fitting import (  # noqa: E402
    DEFAULT_FIT_SETTINGS,
    FitResult,
    covariance,
    fit_mechanistic,
)
from process_transfer.models.identifiability import (  # noqa: E402
    CHI2_ONE_95,
    DesignCovariance,
    NoInformation,
    WindowInformation,
    design_covariance,
    profile_interval,
    profile_settings,
    steady_state_parameters,
    window_information,
)
from process_transfer.models.mechanistic import (  # noqa: E402
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
)
from process_transfer.models.rollout import EVALUATION_SETTINGS, RolloutSettings  # noqa: E402

# =========================================================================== #
# Fixed before the first run
# =========================================================================== #

PLANT = "target"
STEADY_EXPORT, STEADY_RUN = "m0-e06", "target.steady.d7200.n0"
P3_EXPORT = "m0-e05"
BUDGETS = (2, 5, 10, 20, 40)  # windows, section 5.4 of the plan
# the windows that MR (b) and MR_F (b - n_V) receive at those budgets
WINDOW_COUNTS = tuple(sorted({b for b in BUDGETS} | {b - validation_windows(b) for b in BUDGETS}))
# A10 (D-018): q and C_Af +-10 % of their nominal values, T_f and T_c +-5 K; the corners of
# P3 in the order of simulation.protocols.corner_levels, which a test checks
A10_RELATIVE = (0.10, 0.10)
A10_ABSOLUTE = (5.0, 5.0)  # K
CORNERS = tuple(itertools.product((1, -1), repeat=4))
DRAW_SEED = 20260927  # the Monte Carlo over the corners that P3 draws
DRAWS = 20_000
QUANTILES = (0.05, 0.5, 0.95)
# the resolution of part 1: its standard errors at the evaluation settings against a far
# tighter integration, largest relative difference allowed
TIGHT = RolloutSettings("LSODA", 1e-12, 1e-10, 10_000_000)
RESOLUTION = 1e-3
# the grid of part 2, K: a wide part for the shape, and a fine part for the minimum, placed
# where the smoke run of I1 found the estimates of MR on these data (disclosed in the log)
WIDE_GRID = tuple(float(v) for v in range(6000, 13001, 500))
FINE_GRID = tuple(float(v) for v in range(9000, 10001, 25))
GRID = tuple(sorted(set(WIDE_GRID) | set(FINE_GRID)))
# the fit held at the estimate of the free fit must reach its loss: the difference of the
# sums of squared normalised residuals, 2 N (J_held^2 - J_free^2), at most this in absolute
# value, a thousandth of what a unit of that sum is worth and far below 3.84
CHECK_POINT = 1e-3
POOLED = "pooled"
NAMES = ("ln k_350", "E/R", "ln UA")
KINDS = ("exact", "context", "sandwich")

INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID_COLOR, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
KIND_COLOR = {"exact": "#2a78d6", "context": "#7a5195", "sandwich": "#eb6834"}
KIND_LABEL = {
    "exact": "initial state exact",
    "context": "bound, initial state from its context",
    "sandwich": "sandwich of the estimator used",
}
PROFILE_COLOR = ["#2a78d6", "#1f9e89", "#b8860b", INK]


def plain(value: object) -> object:
    """What json can write: dataclasses as dictionaries, arrays as lists, enums as values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | range):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def fingerprints(directory: Path) -> dict[str, str]:
    """SHA-256 of every file under ``directory``, to show that nothing was written there."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


# =========================================================================== #
# The data
# =========================================================================== #


def a10_amplitudes(nominal: Sequence[float]) -> np.ndarray:
    """The A10 amplitudes of the inputs (q, C_Af, T_f, T_c), in SI."""
    return np.array(
        [A10_RELATIVE[0] * nominal[0], A10_RELATIVE[1] * nominal[1], *A10_ABSOLUTE],
        dtype=np.float64,
    )


def design_inputs(nominal: Sequence[float], signs: Sequence[int]) -> np.ndarray:
    """The inputs of the rows of a P3 window at a corner: held for the hold of the layout,
    then nominal to the last scored reading."""
    nominal = np.asarray(nominal, dtype=np.float64)
    inputs = np.tile(nominal, (P3_LAYOUT.scored_readings, 1))
    inputs[: P3_LAYOUT.hold] = nominal + np.asarray(signs, dtype=np.int64) * a10_amplitudes(nominal)
    return inputs


def target_windows(export: Export, known: KnownPlant) -> dict[str, tuple[WindowData, ...]]:
    """The window data of every target run of an export of P3, by run."""
    layout = layout_for("p3")
    found = {}
    for run_id in export.run_ids:
        if export.run(run_id)["plant_id"] != PLANT:
            continue
        observations = export.observations(run_id)
        found[run_id] = window_data(
            observations, find_windows(observations, known.nominal_inputs, layout).windows
        )
    return found


def profile_windows(
    runs: dict[str, tuple[WindowData, ...]],
) -> dict[str, tuple[WindowData, ...]]:
    """The windows of each profile: every run as a replicate, and all of them pooled."""
    return {**runs, POOLED: tuple(data for windows in runs.values() for data in windows)}


def load(exports: Path) -> tuple[KnownPlant, ModellerConfig, Export, Export]:
    p3 = open_export_directory(exports / P3_EXPORT)
    steady = open_export_directory(exports / STEADY_EXPORT)
    known = read_known_plant(p3, PLANT)
    if read_known_plant(steady, PLANT) != known:
        raise ValueError(f"{P3_EXPORT} and {STEADY_EXPORT} state different known parameters")
    return known, modeller_values(), p3, steady


def corners_are_the_design(runs: dict[str, tuple[WindowData, ...]], known: KnownPlant) -> bool:
    """Whether every excursion of the P3 runs moved the inputs to one of the 16 corners of
    the design, bit for bit: the design of part 1 is then P3 as it was run."""
    corners = {tuple(design_inputs(known.nominal_inputs, signs)[0]) for signs in CORNERS}
    return all(tuple(data.inputs[0]) in corners for windows in runs.values() for data in windows)


def corner_counts(windows: Sequence[WindowData], known: KnownPlant) -> np.ndarray:
    """How many of ``windows`` went to each corner of the design, in the order of CORNERS."""
    rows = [tuple(design_inputs(known.nominal_inputs, signs)[0]) for signs in CORNERS]
    counts = np.zeros(len(CORNERS))
    for data in windows:
        counts[rows.index(tuple(data.inputs[0]))] += 1
    return counts


# =========================================================================== #
# Part 1: information a priori
# =========================================================================== #


def point_of_evaluation(
    known: KnownPlant, state: np.ndarray, activation: float
) -> dict[str, object]:
    parameters = steady_state_parameters(known, state, activation)
    model = MechanisticModel("point", known, parameters)
    nominal = np.array(known.nominal_inputs)
    dilution = nominal[0] / known.volume
    feed = np.array([dilution * nominal[1], dilution * nominal[2]])
    eigenvalues = np.linalg.eigvals(model.jacobian_state(state, nominal))
    return {
        "parameters": parameters,
        "k_350": parameters.k_350,
        "residual_over_feed_terms": (model.rhs(state, nominal) / feed).tolist(),
        "eigenvalues_per_min": [[60.0 * v.real, 60.0 * v.imag] for v in eigenvalues],
        "stable": bool(np.all(eigenvalues.real < 0.0)),
        "model": model,
    }


def corner_blocks(
    model: MechanisticModel,
    known: KnownPlant,
    state: np.ndarray,
    sigma: np.ndarray,
    settings: RolloutSettings,
) -> list[WindowInformation | NoInformation]:
    return [
        window_information(
            model,
            state,
            design_inputs(known.nominal_inputs, signs),
            6.0,
            sigma,
            CONTEXT_READINGS,
            settings,
        )
        for signs in CORNERS
    ]


def describe(design: DesignCovariance) -> dict[str, object]:
    if design.reason is not None:
        return {"reason": design.reason, "singular_values": design.singular_values}
    return {
        "standard_errors": {kind: design.standard_errors(kind) for kind in KINDS},
        "correlations": {kind: design.correlations(kind) for kind in KINDS},
        "singular_values": design.singular_values,
        "condition_of_correlations": {
            kind: _condition(design.correlations(kind)) for kind in KINDS
        },
        "reason": None,
    }


def _condition(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(eigenvalues[-1] / eigenvalues[0])


def draws(n: int, rng: np.random.Generator) -> np.ndarray:
    """How many of n windows fall on each corner, for DRAWS sequences of P3."""
    return rng.multinomial(n, np.full(len(CORNERS), 1.0 / len(CORNERS)), size=DRAWS)


def monte_carlo(blocks: Sequence[WindowInformation]) -> dict[str, object]:
    """For each number of windows, the distribution over the corners P3 draws."""
    rng = np.random.default_rng(DRAW_SEED)
    found = {}
    for n in WINDOW_COUNTS:
        counts = draws(n, rng)
        errors = {kind: [] for kind in KINDS}
        correlations = {kind: [] for kind in ("exact", "sandwich")}
        conditions, deficient = [], 0
        for row in counts:
            design = design_covariance(blocks, row)
            if design.reason is not None:
                deficient += 1
                continue
            for kind in KINDS:
                errors[kind].append(list(design.standard_errors(kind).values()))
            for kind in correlations:
                c = design.correlations(kind)
                correlations[kind].append([c[0, 1], c[0, 2], c[1, 2]])
            conditions.append(_condition(design.correlations("sandwich")))
        summary = {"draws": DRAWS, "rank_deficient": deficient, "standard_errors": {}}
        for kind in KINDS:
            values = np.array(errors[kind])
            summary["standard_errors"][kind] = {
                name: {
                    "quantiles": dict(
                        zip(
                            map(str, QUANTILES),
                            np.quantile(values[:, i], QUANTILES).tolist(),
                            strict=True,
                        )
                    ),
                    "min": float(values[:, i].min()),
                    "max": float(values[:, i].max()),
                }
                for i, name in enumerate(NAMES)
            }
        summary["median_correlations"] = {
            kind: dict(
                zip(("k,E", "k,UA", "E,UA"), np.median(np.array(v), axis=0).tolist(), strict=True)
            )
            for kind, v in correlations.items()
        }
        summary["condition_of_sandwich_correlations"] = {
            "median": float(np.median(conditions)),
            "max": float(np.max(conditions)),
        }
        summary["distinct_corners"] = {
            "median": float(np.median((counts > 0).sum(axis=1))),
            "min": int((counts > 0).sum(axis=1).min()),
        }
        found[n] = summary
    return found


def part_one(
    known: KnownPlant,
    state: np.ndarray,
    sigma: np.ndarray,
    activation: float,
    label: str,
    with_monte_carlo: bool,
    observed_designs: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """The information at one point: every corner window, the expected design and the draws
    of P3 at each number of windows, and the designs of the corners that the runs of part 2
    visited, a priori, from the steady state."""
    point = point_of_evaluation(known, state, activation)
    model = point.pop("model")
    blocks = corner_blocks(model, known, state, sigma, EVALUATION_SETTINGS)
    failed = [
        {"corner": signs, "reason": block.reason}
        for signs, block in zip(CORNERS, blocks, strict=True)
        if isinstance(block, NoInformation)
    ]
    result: dict[str, object] = {"label": label, "activation": activation, "point": point}
    if failed:
        result["failed_windows"] = failed
        return result
    tight = corner_blocks(model, known, state, sigma, TIGHT)
    per_corner = {}
    worst = 0.0
    for signs, block, precise in zip(CORNERS, blocks, tight, strict=True):
        one = design_covariance([block])
        per_corner["".join("+" if s > 0 else "-" for s in signs)] = {
            **describe(one),
            "end_state_minus_steady_state": (block.end_state - state).tolist(),
        }
        if isinstance(precise, NoInformation):
            worst = math.inf
            continue
        reference = design_covariance([precise])
        for kind in KINDS:
            for name in NAMES:
                a, b = one.standard_errors(kind)[name], reference.standard_errors(kind)[name]
                worst = max(worst, abs(a - b) / b)
    expected = {
        n: describe(design_covariance(blocks, np.full(len(CORNERS), n / len(CORNERS))))
        for n in WINDOW_COUNTS
    }
    result.update(
        {
            "per_corner": per_corner,
            "expected_design": expected,
            "resolution": {
                "largest_relative_difference_of_standard_errors": worst,
                "allowed": RESOLUTION,
                "resolved": worst <= RESOLUTION,
            },
        }
    )
    if observed_designs:
        result["designs_of_the_runs_of_part_2"] = {
            name: {"corner_counts": counts, **describe(design_covariance(blocks, counts))}
            for name, counts in observed_designs.items()
        }
    if with_monte_carlo:
        result["p3_draws"] = monte_carlo(blocks)
    return result


# =========================================================================== #
# Part 2: the profile against E/R
# =========================================================================== #

_WINDOWS: dict[str, tuple[WindowData, ...]] = {}
_CONTEXT: dict[str, object] = {}


def _initialise(exports: str) -> None:
    """Load the windows of the profiles in a worker process, from the exports themselves."""
    known, modeller, p3, _ = load(Path(exports))
    _WINDOWS.update(profile_windows(target_windows(p3, known)))
    _CONTEXT.update(known=known, modeller=modeller)


def fit_task(task: tuple[str, float | None]) -> tuple[str, float | None, FitResult]:
    """One fit of a profile: E/R free when the activation is None, held otherwise."""
    name, activation = task
    settings = DEFAULT_FIT_SETTINGS if activation is None else profile_settings(activation)
    fit = fit_mechanistic(
        "MR" if activation is None else f"MR, E/R held at {activation} K",
        _WINDOWS[name],
        _CONTEXT["known"],
        _CONTEXT["modeller"],
        settings,
    )
    return name, activation, fit


def summarise_profile(
    free: FitResult, points: dict[float, FitResult], readings: int
) -> dict[str, object]:
    """The profile of one set of windows: its points, the increase of the loss above the
    minimum, where the increase crosses the declared thresholds, and its lowest point. The
    points are those of the grid and, when the free fit converged, its own E/R held."""
    grid = sorted(points)
    objectives = [points[a].objective for a in grid]
    known_values = [j for j in objectives if j is not None]
    if free.objective is None and not known_values:
        return {"readings": readings, "rows": [], "not computed": "no fit converged"}
    minimum = free.objective if free.objective is not None else min(known_values)
    increases = [None if j is None else 2.0 * readings * (j**2 - minimum**2) for j in objectives]
    estimate = None if free.parameters is None else free.parameters.activation_temperature
    rows = []
    for activation, j, increase in zip(grid, objectives, increases, strict=True):
        fit = points[activation]
        p = fit.parameters
        rows.append(
            {
                "activation": activation,
                "on_the_grid": activation in GRID,
                "at_the_free_estimate": activation == estimate,
                "objective": j,
                "increase": increase,
                "k_350": None if p is None else p.k_350,
                "ua": None if p is None else p.ua,
                "outcomes": [record.outcome for record in fit.starts],
                "spread": fit.spread(),
            }
        )
    on_grid = [
        (i, v)
        for i, (a, v) in enumerate(zip(grid, increases, strict=True))
        if a in GRID and v is not None
    ]
    lowest = min(on_grid, key=lambda item: (item[1], item[0]))[0] if on_grid else None
    intervals = {}
    for label, threshold in (
        ("nominal noise", CHI2_ONE_95),
        ("noise scaled by the lack of fit", CHI2_ONE_95 * minimum**2),
    ):
        try:
            intervals[label] = profile_interval(grid, increases, threshold)
        except ValueError as error:
            intervals[label] = {"not computed": str(error)}
    return {
        "readings": readings,
        "minimum_objective": minimum,
        "rows": rows,
        "failed_points": [row["activation"] for row in rows if row["objective"] is None],
        "lowest_grid_point": None if lowest is None else grid[lowest],
        "lowest_at_an_end_of_the_grid": (
            None if lowest is None else grid[lowest] in (GRID[0], GRID[-1])
        ),
        "estimate_inside_the_grid": None if estimate is None else GRID[0] < estimate < GRID[-1],
        "intervals": intervals,
    }


def part_two(
    known: KnownPlant, workers: int, exports: Path
) -> tuple[dict[str, object], dict[str, FitResult]]:
    names = list(_WINDOWS)
    mapper: Callable[..., Iterable] = map
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(workers, initializer=_initialise, initargs=(str(exports),))
        mapper = pool.map
    try:
        free = {name: fit for name, _, fit in mapper(fit_task, [(name, None) for name in names])}
        tasks = [(name, activation) for name in names for activation in GRID]
        tasks += [
            (name, free[name].parameters.activation_temperature)
            for name in names
            if free[name].parameters is not None
        ]
        results = list(mapper(fit_task, tasks))
    finally:
        if pool is not None:
            pool.shutdown()
    profiles = {}
    for name in names:
        fit = free[name]
        points = {a: f for n, a, f in results if n == name}
        check = [f for n, a, f in results[len(names) * len(GRID) :] if n == name]
        windows = _WINDOWS[name]
        summary = summarise_profile(fit, points, sum(len(d.scored) for d in windows))
        entry: dict[str, object] = {"free_fit": fit, "profile": summary}
        if fit.parameters is not None:
            spread = covariance(fit.parameters, windows, known)
            entry["covariance_at_the_estimate"] = spread
            entry["standard_errors_at_the_estimate"] = (
                {kind: spread.standard_errors(kind == "sandwich") for kind in ("exact", "sandwich")}
                if spread.reason is None
                else None
            )
            held = check[0]
            entry["check_point"] = {
                "activation": fit.parameters.activation_temperature,
                "objective_held": held.objective,
                "objective_free": fit.objective,
                "loss_difference": (
                    None
                    if held.objective is None
                    else 2.0 * summary["readings"] * (held.objective**2 - fit.objective**2)
                ),
            }
        profiles[name] = entry
    return profiles, free


def replicate_differences(profiles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """The estimates of E/R of every pair of replicates, their difference, and that
    difference in units of the combined sandwich standard errors, as a description."""
    estimates = {
        name: (
            entry["free_fit"].parameters.activation_temperature,
            entry["standard_errors_at_the_estimate"]["sandwich"]["E/R"],
        )
        for name, entry in profiles.items()
        if name != POOLED and entry.get("standard_errors_at_the_estimate")
    }
    pairs = []
    for (a, (ea, sa)), (b, (eb, sb)) in itertools.combinations(estimates.items(), 2):
        pairs.append(
            {
                "pair": [a, b],
                "difference_K": ea - eb,
                "in_combined_standard_errors": (ea - eb) / math.hypot(sa, sb),
            }
        )
    return pairs


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
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_standard_errors(part: dict[str, object], directory: Path) -> Path:
    """Standard errors against the number of windows, for the three covariances of the
    expected design, with the range of the sandwich over the draws of P3."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), facecolor=SURFACE)
    counts = np.array(WINDOW_COUNTS, dtype=float)
    scales = {"ln k_350": 100.0, "E/R": 1.0, "ln UA": 100.0}
    units = {"ln k_350": "k_350, %", "E/R": "E/R, K", "ln UA": "UA, %"}
    for ax, name in zip(axes, NAMES, strict=True):
        style(ax)
        for kind in KINDS:
            values = [
                part["expected_design"][n]["standard_errors"][kind][name] * scales[name]
                for n in WINDOW_COUNTS
            ]
            ax.plot(
                counts,
                values,
                color=KIND_COLOR[kind],
                linewidth=1.4,
                marker="o",
                markersize=3,
                label=KIND_LABEL[kind],
            )
        draws_ = part["p3_draws"]
        low = [
            draws_[n]["standard_errors"]["sandwich"][name]["quantiles"]["0.05"]
            for n in WINDOW_COUNTS
        ]
        high = [
            draws_[n]["standard_errors"]["sandwich"][name]["quantiles"]["0.95"]
            for n in WINDOW_COUNTS
        ]
        ax.fill_between(
            counts,
            np.array(low) * scales[name],
            np.array(high) * scales[name],
            color=KIND_COLOR["sandwich"],
            alpha=0.18,
            linewidth=0,
            label="sandwich, 5 % to 95 % over draws of P3",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_xticks(counts)
        ax.set_xticklabels([str(int(n)) for n in counts], fontsize=7)
        ax.set_xlabel("windows in the fit", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel(f"standard error of {units[name]}", fontsize=9, color=INK_SECONDARY)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(
        f"M1-E01 part 1: what n windows of P3 determine if the first-order model were right, at "
        f"E/R = {part['activation']:.0f} K and the observed steady state (local, noise of D-020)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = directory / "fig1_standard_errors_against_windows.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_corners(part: dict[str, object], directory: Path) -> Path:
    """The standard error of E/R from one window at each corner."""
    labels = list(part["per_corner"])
    fig, ax = plt.subplots(figsize=(11.0, 3.8), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(labels))
    width = 0.27
    for offset, kind in zip((-width, 0.0, width), KINDS, strict=True):
        values = [part["per_corner"][c]["standard_errors"][kind]["E/R"] for c in labels]
        ax.bar(x + offset, values, width, color=KIND_COLOR[kind], label=KIND_LABEL[kind])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, family="monospace")
    ax.set_xlabel("corner: signs of q, C_Af, T_f, T_c", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("standard error of E/R from one window, K", fontsize=9, color=INK_SECONDARY)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(
        "M1-E01 part 1: one window at each corner of P3", fontsize=10, color=INK, loc="left"
    )
    fig.tight_layout()
    path = directory / "fig2_one_window_per_corner.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_profiles(profiles: dict[str, dict[str, object]], directory: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), facecolor=SURFACE)
    for ax in axes:
        style(ax)
    for index, (name, entry) in enumerate(profiles.items()):
        color = PROFILE_COLOR[index % len(PROFILE_COLOR)]
        rows = entry["profile"]["rows"]
        a = np.array([r["activation"] for r in rows])
        j = np.array([np.nan if r["objective"] is None else r["objective"] for r in rows])
        inc = np.array([np.nan if r["increase"] is None else r["increase"] for r in rows])
        label = name if name == POOLED else name.replace("target.p3.", "")
        axes[0].plot(a, j, color=color, linewidth=1.2, marker="o", markersize=2, label=label)
        fine = (a >= FINE_GRID[0]) & (a <= FINE_GRID[-1])
        axes[1].plot(
            a[fine], inc[fine], color=color, linewidth=1.2, marker="o", markersize=2, label=label
        )
        free = entry["free_fit"].parameters
        if free is not None:
            for ax in axes:
                ax.axvline(free.activation_temperature, color=color, linewidth=0.7, linestyle=":")
            scaled = CHI2_ONE_95 * entry["profile"]["minimum_objective"] ** 2
            axes[1].axhline(scaled, color=color, linewidth=0.6, linestyle="--")
    axes[1].axhline(CHI2_ONE_95, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title(
        "dashed: 3.84 in grey, and 3.84 J_min^2 in the colour of each profile",
        fontsize=8,
        color=INK_SECONDARY,
        loc="left",
    )
    axes[0].set_xlabel("E/R held, K", fontsize=9, color=INK_SECONDARY)
    axes[1].set_xlabel("E/R held, K (fine part of the grid)", fontsize=9, color=INK_SECONDARY)
    axes[0].set_ylabel("J at the fitted k_350 and UA", fontsize=9, color=INK_SECONDARY)
    axes[1].set_ylabel("increase of the loss, 2N (J^2 - J_min^2)", fontsize=9, color=INK_SECONDARY)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "M1-E01 part 2: profile of the fitting objective of MR against E/R on the target runs "
        "of m0-e05 (dotted: E/R of the free fit). Development data, not the benchmark",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = directory / "fig3_profiles.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_compensation(
    profiles: dict[str, dict[str, object]], textbook: MechanisticParameters, directory: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2), facecolor=SURFACE)
    for ax in axes:
        style(ax)
    for index, (name, entry) in enumerate(profiles.items()):
        color = PROFILE_COLOR[index % len(PROFILE_COLOR)]
        rows = [r for r in entry["profile"]["rows"] if r["objective"] is not None]
        a = np.array([r["activation"] for r in rows])
        label = name if name == POOLED else name.replace("target.p3.", "")
        axes[0].plot(
            a,
            [r["k_350"] / textbook.k_350 for r in rows],
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=2,
            label=label,
        )
        axes[1].plot(
            a,
            [r["ua"] / textbook.ua for r in rows],
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=2,
            label=label,
        )
    axes[0].set_ylabel("fitted k_350 / textbook k_350", fontsize=9, color=INK_SECONDARY)
    axes[1].set_ylabel("fitted UA / textbook UA", fontsize=9, color=INK_SECONDARY)
    for ax in axes:
        ax.set_xlabel("E/R held, K", fontsize=9, color=INK_SECONDARY)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "M1-E01 part 2: how k_350 and UA move along the profile of E/R",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = directory / "fig4_compensation_along_the_profile.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# =========================================================================== #
# Main
# =========================================================================== #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exports", type=Path, required=True, help="the exports of M0")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    arguments = parser.parse_args()
    started = time.perf_counter()
    state = git_state()
    directory = new_run_directory("m1_e01", state)
    if not state["code_identified"]:
        print(f"NOTE: the code of this run is not fully identified: {state['reason']}")
    exports = arguments.exports.resolve()
    before = {name: fingerprints(exports / name) for name in (P3_EXPORT, STEADY_EXPORT)}

    known, modeller, p3, steady = load(exports)
    textbook = MechanisticParameters.textbook(modeller)
    runs = target_windows(p3, known)
    _WINDOWS.update(profile_windows(runs))
    _CONTEXT.update(known=known, modeller=modeller)
    observed = steady.observations(STEADY_RUN)
    sigma = np.array(observed.noise_std, dtype=np.float64)
    steady_state = np.mean(observed.measured, axis=0)
    context_means = np.array([d.initial_state for w in runs.values() for d in w])
    design_is_p3 = corners_are_the_design(runs, known)
    run_designs = (
        {name: corner_counts(w, known) for name, w in profile_windows(runs).items()}
        if design_is_p3
        else None
    )

    timings = {}
    tick = time.perf_counter()
    primary = part_one(
        known,
        steady_state,
        sigma,
        textbook.activation_temperature,
        "textbook E/R",
        with_monte_carlo=True,
        observed_designs=run_designs,
    )
    timings["part 1, textbook E/R"] = time.perf_counter() - tick

    tick = time.perf_counter()
    profiles, free = part_two(known, arguments.workers, exports)
    timings["part 2"] = time.perf_counter() - tick

    secondary = None
    pooled = free[POOLED].parameters
    if pooled is not None:
        tick = time.perf_counter()
        secondary = part_one(
            known,
            steady_state,
            sigma,
            pooled.activation_temperature,
            "E/R of the pooled free fit",
            with_monte_carlo=False,
            observed_designs=run_designs,
        )
        timings["part 1, pooled E/R"] = time.perf_counter() - tick

    after = {name: fingerprints(exports / name) for name in (P3_EXPORT, STEADY_EXPORT)}
    checks = {
        "every corner window of part 1 has information": "failed_windows" not in primary,
        "part 1 resolved against the tight integration": bool(
            primary.get("resolution", {}).get("resolved", False)
        ),
        "the point of evaluation is a stable steady state": primary["point"]["stable"],
        "the corners of m0-e05 are corners of the design": design_is_p3,
        "every profile with a free fit reproduces its loss at the estimate": all(
            entry["check_point"]["loss_difference"] is not None
            and abs(entry["check_point"]["loss_difference"]) <= CHECK_POINT
            for entry in profiles.values()
            if "check_point" in entry
        ),
        "the exports were not written to": before == after,
    }
    figures = [
        figure_standard_errors(primary, directory),
        figure_corners(primary, directory),
        figure_profiles(profiles, directory),
        figure_compensation(profiles, textbook, directory),
    ]
    summary = {
        "what": "M1-E01, parts 1 and 2: available information only. Development data of M0; "
        "not the benchmark of M1.",
        "provenance": {
            "experiment": "M1-E01, parts 1 and 2",
            "run_id": directory.name,
            "command": " ".join(sys.argv),
            "git": state,
            "environment": environment(),
            "workers": arguments.workers,
            "configurations": copy_with_fingerprints(
                [repository_root() / "configs" / "modeller_cstr.yaml"], directory / "configs"
            ),
            "exports": {
                export.dataset_id: {r: export.run(r)["content_sha256"] for r in export.run_ids}
                for export in (p3, steady)
            },
            "export_file_fingerprints": before,
            "oracle": False,
        },
        "fixed_before_the_run": {
            "budgets": BUDGETS,
            "window_counts": WINDOW_COUNTS,
            "corners": CORNERS,
            "a10": {"relative": A10_RELATIVE, "absolute_K": A10_ABSOLUTE},
            "draw_seed": DRAW_SEED,
            "draws": DRAWS,
            "quantiles": QUANTILES,
            "tight_rollout": TIGHT,
            "resolution": RESOLUTION,
            "grid": GRID,
            "check_point": CHECK_POINT,
            "fit_settings": DEFAULT_FIT_SETTINGS,
            "evaluation_rollout": EVALUATION_SETTINGS,
            "context_readings": CONTEXT_READINGS,
        },
        "known_plant": known,
        "noise_std": sigma,
        "steady_state": {
            "run": STEADY_RUN,
            "readings": observed.n_samples,
            "mean": steady_state,
            "standard_error_of_the_mean": (sigma / math.sqrt(observed.n_samples)).tolist(),
            "mean_of_the_contexts_of_m0_e05": context_means.mean(axis=0),
            "contexts": len(context_means),
        },
        "textbook": textbook,
        "part_1": primary,
        "part_1_at_the_pooled_estimate": secondary,
        "part_2": profiles,
        "replicate_differences": replicate_differences(profiles),
        "checks": checks,
        "figures": [path.name for path in figures],
        "seconds": {**timings, "total": time.perf_counter() - started},
    }
    (directory / "summary.json").write_text(json.dumps(plain(summary), indent=2), encoding="utf-8")

    print(f"M1-E01 parts 1 and 2; written to {directory}")
    print(f"steady state from {STEADY_RUN}: {steady_state.tolist()}")
    for n in WINDOW_COUNTS:
        e = primary["expected_design"][n]["standard_errors"]
        print(
            f"  n = {n:2d}: se(E/R) exact {e['exact']['E/R']:.1f} K, context "
            f"{e['context']['E/R']:.1f} K, sandwich {e['sandwich']['E/R']:.1f} K"
        )
    for name, entry in profiles.items():
        fit = entry["free_fit"]
        found = (
            "training failure"
            if fit.parameters is None
            else (f"E/R {fit.parameters.activation_temperature:.1f} K, J {fit.objective:.4f}")
        )
        profile = entry["profile"]
        print(
            f"  {name}: {found}; lowest grid point {profile.get('lowest_grid_point')} K, "
            f"at an end {profile.get('lowest_at_an_end_of_the_grid')}, failed points "
            f"{profile.get('failed_points')}"
        )
    for label, holds in checks.items():
        print(f"  {label}: {'holds' if holds else 'FAILS'}")
    print(f"{summary['seconds']['total']:.1f} s")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
