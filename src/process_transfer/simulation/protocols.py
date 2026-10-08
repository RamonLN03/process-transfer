"""The operating runs of M0 as experiments on the plant (docs/decisions.md D-010, D-019).

P3, the excitation protocol accepted in D-019: excursions of 120 s to the corners of the
input box at the A10 amplitudes, each followed by 600 s at the nominal inputs. The state
of the plant is never reset: what an excursion leaves behind is carried into the next one.

M1 adds two things to P3 (``docs/m1_plan.md``, sections 4.2 and 6.1; D-039): a lead of
60 s at the nominal inputs before the first excursion, so that it has a context of ten
readings, and the amplitude A5, half of A10 on every input. Without them P3 is exactly
the protocol of M0. P3 at A5 may be used for data only once M1-E02 has verified it.

Scope. P3 was verified on the present source and target, with these amplitudes,
this hold and this rest (M0-E03 and M0-E03b). That is evidence about those plants and
conditions. A different plant, amplitude, hold or rest needs the same verification,
and the rest has been measured, not optimised.

The two other runs that D-010 promised, defined here so that every experiment on the
plant lives in one place. Neither draws anything at random, so neither has an excitation
seed:

* steady operation: the nominal inputs held for a whole run, from the nominal steady
  state, with nothing but the sensor noise on top;
* single-input step tests: 600 s at the nominal inputs, 600 s with one input moved by
  its A10 amplitude in one direction, 600 s at the nominal inputs to recover. Four inputs
  and two directions make eight tests per plant. Each is an independent run started at
  the nominal steady state; the state is carried across the three segments of a test and
  the eight tests are never chained into one trajectory, because the transitions between
  them have not been studied. What these holds do to the plant is verified in M0-E07,
  not inferred from P3.

This module describes experiments on the plant. It contains no hidden physics.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from process_transfer.cstr_variables import FloatArray
from process_transfer.simulation.excitation import (
    LevelArray,
    binary_levels,
    excursions_with_rest,
    levels_to_segments,
)
from process_transfer.simulation.integration import InputSegment

# A10 (D-018): q and C_Af +-10 % of their nominal values, T_f and T_c +-5 K.
A10_RELATIVE_FLOW = 0.10
A10_RELATIVE_FEED_CONCENTRATION = 0.10
A10_FEED_TEMPERATURE = 5.0  # K
A10_COOLANT_TEMPERATURE = 5.0  # K

P3_HOLD = 120.0  # s spent at a corner
P3_REST = 600.0  # s spent at the nominal inputs afterwards
# The lead of a P3 run of M1: the nominal inputs held from the verified steady state before
# the first excursion (plan of M1, section 4.2). A run of M0 has none.
P3_LEAD = 60.0  # s
# The amplitudes of P3, by the token that names them in the identity of a run (D-039).
P3_AMPLITUDES = ("a10", "a5")

STEADY_DURATION = 7200.0  # s of steady operation, the length of a P3 run

STEP_LEAD = 600.0  # s at the nominal inputs before the step
STEP_HOLD = 600.0  # s with one input moved, the "10 min hold" of D-010
STEP_RECOVERY = 600.0  # s at the nominal inputs after the step
# Short names of the inputs u = [q, C_Af, T_f, T_c], in that order, for run identities.
STEP_INPUTS = ("q", "caf", "tf", "tc")
STEP_DIRECTIONS = ("up", "down")

# What "back at the nominal steady state" means for P3. One hundredth of the sensor
# noise as it stood when the values were fixed: sigma_T = 0.5 K, and for C_A the stricter
# of the two readings then open in D-020, 3.8 mol/m^3 against 5 mol/m^3, so that the
# criterion did not depend on that decision. D-020 has since chosen 5 mol/m^3; the
# tolerance was verified at 0.038 and stays there, now 1/132 of sigma_CA. The decision is
# not a reason to relax it.
#
# This is a practical criterion, not a statistical guarantee. A residual far below the
# noise of one reading is not thereby invisible in every data set: averaging N
# independent readings resolves an offset of about 2 sigma / sqrt(N), so a constant
# offset of sigma / 100 would show after some 40 000 readings, 67 h at one reading every
# 6 s. What the criterion says is that the carried state is small against everything
# else in the data of this study. The residual is not constant either: it keeps
# decaying, and M0-E03b measured it at less than a third of the tolerance.
P3_RECOVERY_TOLERANCE_T = 0.005  # K
P3_RECOVERY_TOLERANCE_CA = 0.038  # mol/m^3
# How closely an excursion started from the carried state must reproduce the peak of the
# same excursion started from the exact steady state: a tenth of sigma_T, and about 1 %
# of the 3.8 K between the worst step from nominal (376.19 K) and the 380 K limit.
P3_PEAK_AGREEMENT = 0.05  # K


def a10_amplitudes(nominal_inputs: FloatArray) -> FloatArray:
    """Absolute A10 amplitudes, in SI, for inputs u = [q, C_Af, T_f, T_c]."""
    u = np.asarray(nominal_inputs, dtype=np.float64)
    return np.array(
        [
            A10_RELATIVE_FLOW * u[0],
            A10_RELATIVE_FEED_CONCENTRATION * u[1],
            A10_FEED_TEMPERATURE,
            A10_COOLANT_TEMPERATURE,
        ]
    )


def a5_amplitudes(nominal_inputs: FloatArray) -> FloatArray:
    """Absolute A5 amplitudes, in SI: half of A10 on every input, so q and C_Af +-5 % of
    their nominal values and T_f and T_c +-2.5 K (plan of M1, section 6.1). Halving is
    exact in binary arithmetic, so A5 is A10 / 2 to the last bit."""
    return 0.5 * a10_amplitudes(nominal_inputs)


def p3_amplitudes(nominal_inputs: FloatArray, amplitude: str) -> FloatArray:
    """The absolute amplitudes that ``amplitude``, ``a10`` or ``a5``, names."""
    if amplitude == "a10":
        return a10_amplitudes(nominal_inputs)
    if amplitude == "a5":
        return a5_amplitudes(nominal_inputs)
    raise ValueError(f"amplitude must be one of {P3_AMPLITUDES}, got {amplitude!r}")


def corner_levels() -> LevelArray:
    """The 16 corners of the input box as rows of -1 and +1, in a fixed order."""
    return np.array(list(itertools.product((1, -1), repeat=4)), dtype=np.int64)


def corner_label(levels: LevelArray) -> str:
    """Signs of q, C_Af, T_f and T_c, for example ``++--``."""
    return "".join("+" if level > 0 else "-" for level in levels)


def p3_corners(n_excursions: int, seed: int) -> LevelArray:
    """The corners of a seeded P3 sequence, each drawn with equal probability and
    independently, repeats allowed.

    A function of the excitation seed and of nothing else: the generator is built here
    and used for this only, so no other draw, of sensor noise for instance, can shift
    the sequence or be shifted by it. The same seed gives the sequence that
    ``separated_excursions`` draws from ``numpy.random.default_rng(seed)``, which is how
    M0-E03 built its P3 sequences.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be an explicit non-negative integer, got {seed!r}")
    return binary_levels(n_excursions, 4, np.random.default_rng(int(seed)))


def p3_segments(
    nominal_inputs: FloatArray,
    corners: LevelArray,
    amplitude: str = "a10",
    lead: float = 0.0,
) -> list[InputSegment]:
    """P3 for a given list of corners: each held ``P3_HOLD`` at ``amplitude``, then nominal
    for ``P3_REST``. A positive ``lead`` puts that many seconds at the nominal inputs before
    the first excursion; the state is carried through it as through a rest, never reset,
    and the corners are those given, whatever the lead. With the defaults this is P3 as
    in M0: A10 and no lead."""
    if (
        isinstance(lead, (bool, np.bool_))
        or not isinstance(lead, (int, float, np.integer, np.floating))
        or not math.isfinite(lead)
        or lead < 0.0
    ):
        raise ValueError(f"lead must be a finite number of seconds, zero or more, got {lead!r}")
    segments = excursions_with_rest(
        nominal_inputs, p3_amplitudes(nominal_inputs, amplitude), corners, P3_HOLD, P3_REST
    )
    if lead > 0.0:
        segments = steady_segments(nominal_inputs, float(lead)) + segments
    return segments


def steady_segments(
    nominal_inputs: FloatArray, duration: float = STEADY_DURATION
) -> list[InputSegment]:
    """Steady operation: one segment at the nominal inputs for ``duration`` seconds."""
    nominal = np.asarray(nominal_inputs, dtype=np.float64)
    still = np.zeros((1, nominal.size), dtype=np.int64)
    return levels_to_segments(nominal, np.zeros(nominal.size), still, duration)


def single_step_segments(
    nominal_inputs: FloatArray,
    input_name: str,
    direction: str,
    lead: float = STEP_LEAD,
    hold: float = STEP_HOLD,
    recovery: float = STEP_RECOVERY,
) -> list[InputSegment]:
    """One single-input step test: ``lead`` seconds at the nominal inputs, ``hold``
    seconds with ``input_name`` moved ``direction`` by its A10 amplitude while the other
    inputs stay nominal, then ``recovery`` seconds at the nominal inputs."""
    if input_name not in STEP_INPUTS:
        raise ValueError(f"input_name must be one of {STEP_INPUTS}, got {input_name!r}")
    if direction not in STEP_DIRECTIONS:
        raise ValueError(f"direction must be one of {STEP_DIRECTIONS}, got {direction!r}")
    nominal = np.asarray(nominal_inputs, dtype=np.float64)
    levels = np.zeros((3, nominal.size), dtype=np.int64)
    levels[1, STEP_INPUTS.index(input_name)] = 1 if direction == "up" else -1
    amplitudes = a10_amplitudes(nominal)
    segments: list[InputSegment] = []
    for row, seconds in zip(levels, (lead, hold, recovery), strict=True):
        segments += levels_to_segments(nominal, amplitudes, row[np.newaxis, :], seconds)
    return segments
