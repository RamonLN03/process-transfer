"""Open-loop excitation sequences: piecewise-constant inputs that change on a clock.

An excitation is described by integer *levels*, one row per clock tick and one
column per input, and turned into inputs by ``u = nominal + level * amplitude``.
Every generator takes an explicit ``numpy.random.Generator``, so a sequence is a
function of its seed and nothing else.

This module describes experiments on the plant. It contains no hidden physics.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from process_transfer.cstr_variables import FloatArray
from process_transfer.simulation.integration import InputSegment

LevelArray = NDArray[np.int64]


def _check_count(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def binary_levels(n_ticks: int, n_inputs: int, rng: np.random.Generator) -> LevelArray:
    """Every input independently at -1 or +1, with equal probability, at every tick.

    An input may therefore jump across its whole range, from -1 to +1, in one tick.
    """
    _check_count("n_ticks", n_ticks)
    _check_count("n_inputs", n_inputs)
    return rng.choice(np.array([-1, 1], dtype=np.int64), size=(n_ticks, n_inputs))


def limited_move_levels(n_ticks: int, n_inputs: int, rng: np.random.Generator) -> LevelArray:
    """Three levels, -1, 0 and +1, each input moving by at most one level per tick.

    Every input starts at 0. At each tick it stays or moves to a neighbouring level,
    all allowed moves being equally likely, so a change from -1 to +1 takes at least
    two ticks and passes through the nominal value.
    """
    _check_count("n_ticks", n_ticks)
    _check_count("n_inputs", n_inputs)
    levels = np.zeros((n_ticks, n_inputs), dtype=np.int64)
    current = np.zeros(n_inputs, dtype=np.int64)
    for tick in range(n_ticks):
        for j in range(n_inputs):
            neighbours = (current[j] - 1, current[j], current[j] + 1)
            current[j] = rng.choice([level for level in neighbours if abs(level) <= 1])
        levels[tick] = current
    return levels


def levels_to_segments(
    nominal: FloatArray, amplitudes: FloatArray, levels: LevelArray, clock: float
) -> list[InputSegment]:
    """One segment of ``clock`` seconds per row of ``levels``."""
    if not math.isfinite(clock) or clock <= 0.0:
        raise ValueError(f"clock must be a finite positive number of seconds, got {clock!r}")
    nominal = np.asarray(nominal, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if not np.all(np.isfinite(nominal)):
        raise ValueError("nominal inputs must be finite")
    if np.any(amplitudes < 0.0) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must be finite and non-negative")
    levels = np.asarray(levels)
    if levels.ndim != 2 or levels.shape[1] != nominal.size:
        raise ValueError(f"levels must have shape (n_ticks, {nominal.size}), got {levels.shape}")
    if levels.size and np.max(np.abs(levels)) > 1:
        # a level of 2 would silently double the amplitude the protocol declares
        raise ValueError("levels must lie between -1 and +1")
    return [InputSegment(clock, nominal + row * amplitudes) for row in levels]


def excursions_with_rest(
    nominal: FloatArray, amplitudes: FloatArray, corners: LevelArray, hold: float, rest: float
) -> list[InputSegment]:
    """Each row of ``corners`` held for ``hold`` seconds, then the nominal inputs for
    ``rest`` seconds. The state is whatever the plant has reached: nothing resets it
    between excursions, so the residual left by one excursion is carried into the next.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    corners = np.asarray(corners)
    if corners.ndim != 2 or len(corners) == 0:
        raise ValueError("corners must be a non-empty array with one row per excursion")
    resting = np.zeros((1, nominal.size), dtype=np.int64)
    segments: list[InputSegment] = []
    for corner in corners:
        segments += levels_to_segments(nominal, amplitudes, corner[np.newaxis, :], hold)
        segments += levels_to_segments(nominal, amplitudes, resting, rest)
    return segments


def separated_excursions(
    nominal: FloatArray,
    amplitudes: FloatArray,
    n_excursions: int,
    hold: float,
    rest: float,
    rng: np.random.Generator,
) -> list[InputSegment]:
    """Excursions to random corners of the input box, each followed by a rest at the
    nominal inputs, so that every excursion starts close to the nominal steady state.

    Each excursion holds a corner, every input at -1 or +1, for ``hold`` seconds;
    the inputs then return to nominal for ``rest`` seconds.
    """
    _check_count("n_excursions", n_excursions)
    corners = binary_levels(n_excursions, np.asarray(nominal).size, rng)
    return excursions_with_rest(nominal, amplitudes, corners, hold, rest)
