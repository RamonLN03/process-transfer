"""Synthetic runs for the tests of the evaluation contract and the models of M1.

Nothing here is read by the package. The runs simulated with the modeller's equations are
made on the truth side of the tests, with the integrator of the simulation, and reach the
code under test only as observations. The inputs follow the protocols of the plan (a lead
of 60 s, corners held 120 s, rests of 600 s; single-input steps of 600 s), with the
nominal inputs of both plants of M0 and the amplitudes A10 of D-019.
"""

from collections.abc import Callable, Sequence

import numpy as np

from process_transfer.cstr_variables import INPUT_NAMES, INPUT_UNITS, STATE_NAMES, STATE_UNITS
from process_transfer.measurement.observations import Observations
from process_transfer.simulation.integration import InputSegment, simulate_piecewise
from process_transfer.simulation.steady_state import find_steady_states

NOMINAL = np.array([0.1 / 60.0, 500.0, 350.0, 337.5])
AMPLITUDES = np.array([0.1 * (0.1 / 60.0), 50.0, 5.0, 5.0])
SIGMA = (5.0, 0.5)


def corner(signs: Sequence[int]) -> np.ndarray:
    return NOMINAL + np.asarray(signs, dtype=np.float64) * AMPLITUDES


def p3_inputs(corners: Sequence[Sequence[int]], lead: bool = True) -> np.ndarray:
    """Inputs of a P3 run: an optional lead of 10 rows, then for each corner 20 rows at the
    corner and 100 at the nominal inputs, and the final row."""
    first = 10 if lead else 0
    inputs = np.tile(NOMINAL, (first + 1 + 120 * len(corners), 1))
    for j, signs in enumerate(corners):
        onset = first + 120 * j
        inputs[onset : onset + 20] = corner(signs)
    return inputs


def step_inputs(input_index: int, direction: int) -> np.ndarray:
    """Inputs of a single-input step run: 600 s lead, 600 s hold, 600 s recovery."""
    inputs = np.tile(NOMINAL, (301, 1))
    inputs[100:200, input_index] += direction * AMPLITUDES[input_index]
    return inputs


def tick_coded(n: int) -> np.ndarray:
    """Readings that say which tick they were taken at: C_A = tick, T = 1000 + tick."""
    ticks = np.arange(n, dtype=np.float64)
    return np.column_stack([ticks, 1000.0 + ticks])


def observations(
    inputs: np.ndarray,
    measured: np.ndarray | None = None,
    run: str = "target.p3.e0.x2.n0",
    noise_std: Sequence[float] = SIGMA,
    sample_period: float = 6.0,
) -> Observations:
    n = len(inputs)
    return Observations(
        plant="target",
        run=run,
        times=sample_period * np.arange(n, dtype=np.float64),
        measured=tick_coded(n) if measured is None else measured,
        inputs=inputs,
        measured_names=STATE_NAMES,
        measured_units=STATE_UNITS,
        input_names=INPUT_NAMES,
        input_units=INPUT_UNITS,
        sample_period=sample_period,
        noise_std=tuple(noise_std),
    )


def random_corners(n: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [list(rng.choice([-1, 1], size=4)) for _ in range(n)]


def modeller_run(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    corners: Sequence[Sequence[int]],
    noise_seed: int | None,
    run: str = "target.p3.e0.x8.n0",
) -> Observations:
    """A P3 run with the lead of M1, simulated with the modeller's equations from their
    steady state at the nominal inputs, on the truth side of the tests. Readings with the
    noise of D-020 drawn from ``noise_seed``, or exact when it is None. ``f`` is the
    right-hand side of the modeller's model at the values chosen by the test."""
    (steady,) = find_steady_states(lambda x: f(x, NOMINAL), c_a_upper=NOMINAL[1])
    segments = [InputSegment(60.0, NOMINAL)]
    for signs in corners:
        segments += [InputSegment(120.0, corner(signs)), InputSegment(600.0, NOMINAL)]
    truth = simulate_piecewise(f, steady.state, segments, sample_period=6.0)
    readings = np.array(truth.states)
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        readings = readings + rng.normal(0.0, 1.0, readings.shape) * np.array(SIGMA)
    return observations(truth.inputs, measured=readings, run=run)
