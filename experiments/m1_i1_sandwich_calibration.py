"""Development check of I1 of M1: are the sandwich standard errors of MR calibrated when the
model is right?

What this is. The recovery test of I1 (``tests/test_models_fitting.py``) asks each estimate
to lie within four sandwich standard errors of its true value, for five declared seeds.
That bounds gross errors; it says little about whether the standard errors are the right
size. This check repeats the same fit over many seeds of the noise and looks at the
distribution of z = (estimate - true value) / standard error, which is standard normal if
the standard errors are right. It is development, on synthetic data, not an experiment of
M1 and not a result.

Data. The modeller's own equations at the values of the recovery test (k_350 1.3 times the
textbook value, E/R = 9200 K, UA 0.85 times the textbook value), simulated on the truth
side from their steady state at the nominal inputs of the target: a lead of 60 s and eight
P3 excursions whose corners come from seed 2026, as in the test; readings with the noise
of D-020, one seed per replicate. MR is fitted on the eight windows from the textbook start,
and its covariance computed at the estimate, with the initial state taken as exact and with
the error of the context mean (the sandwich).

Usage:

    python experiments/m1_i1_sandwich_calibration.py --seeds 2000 2200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from process_transfer.cstr_variables import INPUT_NAMES, INPUT_UNITS, STATE_NAMES, STATE_UNITS
from process_transfer.data.paths import output_dir
from process_transfer.data.provenance import environment, git_state, new_directory
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.measurement.observations import Observations
from process_transfer.models.fitting import DEFAULT_STARTS, FitSettings, covariance, fit_mechanistic
from process_transfer.models.mechanistic import (
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
)
from process_transfer.simulation.integration import InputSegment, simulate_piecewise
from process_transfer.simulation.steady_state import find_steady_states

NOMINAL = np.array([0.1 / 60.0, 500.0, 350.0, 337.5])
AMPLITUDES = np.array([0.1 * (0.1 / 60.0), 50.0, 5.0, 5.0])
SIGMA = np.array([5.0, 0.5])
KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
CORNER_SEED = 2026


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--seeds", type=int, nargs=2, default=(2000, 2200), metavar=("FIRST", "STOP")
    )
    first, stop = parser.parse_args().seeds
    started = time.perf_counter()
    state = git_state()
    directory = new_directory(output_dir("development", "m1-i1-sandwich-calibration"), state)

    modeller = modeller_values()
    textbook = MechanisticParameters.textbook(modeller)
    true = MechanisticParameters.from_k_350(1.3 * textbook.k_350, 9200.0, 0.85 * textbook.ua)
    truth = MechanisticModel("truth", KNOWN, true)
    (steady,) = find_steady_states(lambda x: truth.rhs(x, NOMINAL), c_a_upper=NOMINAL[1])
    rng = np.random.default_rng(CORNER_SEED)
    corners = [rng.choice([-1, 1], size=4) for _ in range(8)]  # as random_corners of the tests
    segments = [InputSegment(60.0, NOMINAL)]
    for signs in corners:
        segments += [
            InputSegment(120.0, NOMINAL + signs * AMPLITUDES),
            InputSegment(600.0, NOMINAL),
        ]
    trajectory = simulate_piecewise(truth.rhs, steady.state, segments, sample_period=6.0)
    exact = np.array([math.log(true.k_350), true.activation_temperature, math.log(true.ua)])

    rows = []
    for seed in range(first, stop):
        noise = np.random.default_rng(seed).normal(0.0, 1.0, trajectory.states.shape) * SIGMA
        run = Observations(
            plant="target",
            run="target.p3.e2026.x8.n0",
            times=trajectory.times,
            measured=trajectory.states + noise,
            inputs=trajectory.inputs,
            measured_names=STATE_NAMES,
            measured_units=STATE_UNITS,
            input_names=INPUT_NAMES,
            input_units=INPUT_UNITS,
            sample_period=6.0,
            noise_std=tuple(SIGMA),
        )
        data = window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)
        fit = fit_mechanistic("MR", data, KNOWN, modeller, FitSettings(starts=DEFAULT_STARTS[:1]))
        if fit.parameters is None:
            rows.append({"seed": seed, "training_failure": fit.training_failure.reason})
            continue
        spread = covariance(fit.parameters, data, KNOWN)
        p = fit.parameters
        found = np.array([math.log(p.k_350), p.activation_temperature, math.log(p.ua)])
        sandwich = np.array(list(spread.standard_errors().values()))
        exact_initial = np.array(list(spread.standard_errors(sandwich=False).values()))
        rows.append(
            {
                "seed": seed,
                "estimate": found.tolist(),
                "z_sandwich": ((found - exact) / sandwich).tolist(),
                "z_exact_initial_state": ((found - exact) / exact_initial).tolist(),
                "se_sandwich": sandwich.tolist(),
            }
        )

    fitted = [row for row in rows if "z_sandwich" in row]
    z = np.array([row["z_sandwich"] for row in fitted])
    z_exact = np.array([row["z_exact_initial_state"] for row in fitted])
    boot = np.random.default_rng(0)
    sd = np.array([z[boot.integers(0, len(z), len(z))].std(axis=0, ddof=1) for _ in range(2000)])
    summary = {
        "what": "Development check of I1: calibration of the sandwich standard errors of MR "
        "on data of the modeller's own model. Not an experiment and not a result of M1.",
        "attempt": {"git": state, "environment": environment()},
        "parameters": ["ln k_350", "E/R", "ln UA"],
        "true": exact.tolist(),
        "seeds": [first, stop],
        "training_failures": len(rows) - len(fitted),
        "z_sandwich_mean": z.mean(axis=0).tolist(),
        "z_sandwich_sd": z.std(axis=0, ddof=1).tolist(),
        "z_sandwich_sd_bootstrap_95": np.percentile(sd, [2.5, 97.5], axis=0).tolist(),
        "z_sandwich_max_abs": np.abs(z).max(axis=0).tolist(),
        "fraction_abs_z_sandwich_above_2": (np.abs(z) > 2.0).mean(axis=0).tolist(),
        "z_exact_initial_state_sd": z_exact.std(axis=0, ddof=1).tolist(),
        "mean_se_sandwich": np.array([row["se_sandwich"] for row in fitted]).mean(axis=0).tolist(),
        "seconds": time.perf_counter() - started,
        "replicates": rows,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"development check, not a result of M1; written to {directory}")
    for key in (
        "training_failures",
        "z_sandwich_mean",
        "z_sandwich_sd",
        "z_sandwich_sd_bootstrap_95",
        "z_sandwich_max_abs",
        "fraction_abs_z_sandwich_above_2",
        "z_exact_initial_state_sd",
        "mean_se_sandwich",
        "seconds",
    ):
        print(f"{key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
