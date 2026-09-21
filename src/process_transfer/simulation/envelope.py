"""Operating envelope: how far the states travel under the planned input excursions.

Local eigenvalues say nothing about large steps, so stability verification
(docs/decisions.md D-009) also simulates the plant from its nominal steady state
under single-input excursions and under every corner of the input box, and
records the range the states visit.

This is a necessary check of an excitation, not a sufficient one. It covers
single steps, from the nominal steady state, at 24 input cases. It says nothing
about the interior of the input box or about chained input changes, where the
state at each change depends on the history: M0-E03 found sequences at amplitudes
that pass this check and still leave the envelope by more than 15 K. Sequences
are checked with ``simulation/checks.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from process_transfer.cstr_variables import INPUT_NAMES, FloatArray
from process_transfer.simulation.integration import sample_times

RightHandSide = Callable[[FloatArray, FloatArray], FloatArray]  # f(x, u) -> dx/dt


@dataclass(frozen=True)
class EnvelopeCase:
    """Response to one constant set of inputs applied as a step at t = 0."""

    label: str
    inputs: FloatArray
    final_state: FloatArray
    final_derivative: FloatArray
    duration: float  # s
    c_a_range: tuple[float, float]  # mol/m^3
    temperature_range: tuple[float, float]  # K

    @property
    def settled(self) -> bool:
        """True when running for another full duration at the final rate of change
        would move each state by less than one part in a million."""
        drift = np.abs(self.final_derivative) * self.duration
        return bool(np.all(drift < 1.0e-6 * np.abs(self.final_state)))


def input_cases(nominal: FloatArray, deviations: FloatArray) -> list[tuple[str, FloatArray]]:
    """Single-input excursions and all corners of the input box.

    ``deviations`` are absolute, non-negative SI deviations, one per input. For
    four inputs this gives 8 single-input cases and 16 corners.
    """
    cases: list[tuple[str, FloatArray]] = []
    for index, name in enumerate(INPUT_NAMES):
        for sign, symbol in ((+1.0, "+"), (-1.0, "-")):
            u = nominal.copy()
            u[index] += sign * deviations[index]
            cases.append((f"{name}{symbol}", u))
    for signs in itertools.product((+1.0, -1.0), repeat=len(INPUT_NAMES)):
        label = " ".join(
            f"{name}{'+' if sign > 0 else '-'}"
            for name, sign in zip(INPUT_NAMES, signs, strict=True)
        )
        cases.append((label, nominal + np.array(signs) * deviations))
    return cases


def simulate_envelope(
    f: RightHandSide,
    x0: FloatArray,
    cases: Sequence[tuple[str, FloatArray]],
    duration: float = 2400.0,
    sample_period: float = 1.0,
) -> list[EnvelopeCase]:
    """Apply each input case as a step from ``x0`` and record the ranges visited.

    The trajectory is sampled every ``sample_period`` seconds and at ``duration``
    itself, so ``final_state`` is always the state at the end of the simulation.
    """
    times = sample_times(duration, sample_period)
    results: list[EnvelopeCase] = []
    for label, u in cases:
        solution = solve_ivp(
            lambda t, x, u=u: f(x, u),
            (0.0, duration),
            x0,
            method="LSODA",
            t_eval=times,
            rtol=1.0e-9,
            atol=1.0e-9,
        )
        if not solution.success:
            raise RuntimeError(f"integration failed for case {label!r}: {solution.message}")
        c_a, temperature = solution.y
        final_state = solution.y[:, -1]
        results.append(
            EnvelopeCase(
                label=label,
                inputs=u,
                final_state=final_state,
                final_derivative=f(final_state, u),
                duration=duration,
                c_a_range=(float(c_a.min()), float(c_a.max())),
                temperature_range=(float(temperature.min()), float(temperature.max())),
            )
        )
    return results
