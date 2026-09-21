"""Steady states, numerical Jacobian and local stability of a CSTR model.

The functions take a right-hand side ``f(x)`` with inputs and parameters already
bound, so they work for the true plant and, in tests, for the modeller's model.
The state is x = [C_A, T] in SI.

Method (docs/decisions.md D-009): continuation in temperature. For each
temperature on a grid, the mass balance is solved for C_A; the energy-balance
residual along that curve is scanned for zeros at the grid points, both ends of
the range included, and for sign changes between them; each root is refined and
then polished on the full two-dimensional system.

Limitations of a sign-change scan. It is a verification aid for the operating
points of this study, not a bifurcation tool:

* a root of even multiplicity (a tangency, as at a fold) changes no sign and is
  missed unless it falls exactly on a grid point;
* two roots inside one grid cell cancel each other and are both missed;
* a root that lies within rounding error of an end of the range, without being
  exactly zero there, may be missed: choose a range that strictly contains the
  temperatures of interest;
* nothing outside ``temperature_range`` is seen;
* the mass balance is assumed to have exactly one root for C_A in
  ``[0, c_a_upper]`` at each temperature, which holds when the rate increases
  with C_A.

A count of steady states is therefore a statement about the scanned range at the
given grid, not a proof of uniqueness.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, root

from process_transfer.cstr_variables import FloatArray
from process_transfer.validation import require_finite, require_positive

BoundRightHandSide = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class SteadyState:
    """A steady state with its residual and the eigenvalues of the local Jacobian."""

    c_a: float  # mol/m^3
    temperature: float  # K
    residual: FloatArray  # f(x) at the steady state: [mol/(m^3 s), K/s]
    eigenvalues: NDArray[np.complex128]  # 1/s

    @property
    def state(self) -> FloatArray:
        return np.array([self.c_a, self.temperature], dtype=np.float64)

    @property
    def max_real_part(self) -> float:
        """Largest real part of the eigenvalues, in 1/s. Negative means locally stable."""
        return float(np.max(self.eigenvalues.real))

    def is_stable(self, margin: float = 0.0) -> bool:
        """True when every eigenvalue has real part below ``-margin`` (1/s)."""
        return self.max_real_part < -margin


def numerical_jacobian(
    f: BoundRightHandSide, x: FloatArray, rel_step: float = 1.0e-6
) -> FloatArray:
    """Jacobian df/dx by central differences, with a step scaled to each state.

    The step is ``rel_step * max(|x_i|, 1)`` and is therefore never zero. A
    right-hand side that is not finite around ``x`` is an error, not a Jacobian of
    NaN."""
    rel_step = require_positive("rel_step", rel_step)
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"the state must be finite, got {x!r}")
    n = x.size
    jacobian = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        step = rel_step * max(abs(x[j]), 1.0)
        forward, backward = x.copy(), x.copy()
        forward[j] += step
        backward[j] -= step
        ahead, behind = f(forward), f(backward)
        if not (np.all(np.isfinite(ahead)) and np.all(np.isfinite(behind))):
            # checked before subtracting, so that no arithmetic is done on infinities
            raise ValueError(f"the right-hand side is not finite around the state {x!r}")
        jacobian[:, j] = (ahead - behind) / (2.0 * step)
    return jacobian


def eigenvalues_at(f: BoundRightHandSide, x: FloatArray) -> NDArray[np.complex128]:
    """Eigenvalues (1/s) of the numerical Jacobian of ``f`` at ``x``."""
    return np.linalg.eigvals(numerical_jacobian(f, x)).astype(np.complex128)


def find_steady_states(
    f: BoundRightHandSide,
    c_a_upper: float,
    temperature_range: tuple[float, float] = (280.0, 480.0),
    n_grid: int = 801,
) -> list[SteadyState]:
    """All steady states of ``f`` with T inside ``temperature_range``, sorted by T.

    The default grid spacing is 0.25 K. Two steady states closer than that exist
    only in a very small neighbourhood of a fold bifurcation.

    ``c_a_upper`` is an upper bound for C_A at steady state; the feed
    concentration is the natural choice, since the reaction only consumes A.
    """

    c_a_upper = require_positive("c_a_upper", c_a_upper)
    low = require_positive("the lower end of temperature_range", temperature_range[0])
    high = require_finite("the upper end of temperature_range", temperature_range[1])
    if not low < high:
        raise ValueError(f"temperature_range must be increasing, got {temperature_range!r}")
    if n_grid != int(n_grid) or n_grid < 2:
        raise ValueError(f"n_grid must be an integer of at least 2, got {n_grid!r}")

    def c_a_on_mass_balance(temperature: float) -> float:
        """C_A in [0, c_a_upper] that closes the mass balance at this temperature."""

        def mass_residual(c_a: float) -> float:
            return float(f(np.array([c_a, temperature]))[0])

        at_zero, at_upper = mass_residual(0.0), mass_residual(c_a_upper)
        if not (math.isfinite(at_zero) and math.isfinite(at_upper)):
            raise ValueError(f"the right-hand side is not finite at T = {temperature} K")
        if at_zero * at_upper > 0.0:
            raise ValueError(
                f"at T = {temperature} K the mass balance does not change sign for C_A in "
                f"[0, {c_a_upper}]; c_a_upper must bound the steady-state concentration, "
                "the feed concentration being the natural choice"
            )
        return float(brentq(mass_residual, 0.0, c_a_upper, xtol=1e-12, rtol=1e-14))

    def energy_residual(temperature: float) -> float:
        c_a = c_a_on_mass_balance(temperature)
        residual = float(f(np.array([c_a, temperature]))[1])
        if not math.isfinite(residual):
            # A NaN never changes sign, so it would otherwise read as "no steady state".
            raise ValueError(f"the right-hand side is not finite at T = {temperature} K")
        return residual

    grid = np.linspace(low, high, int(n_grid))
    residuals = np.array([energy_residual(t) for t in grid])

    # Roots that fall exactly on a grid point, both ends of the range included, and
    # strict sign changes between neighbouring points that are both non-zero. A root
    # on a grid point is therefore found once, never again through its two intervals.
    candidates = [float(grid[i]) for i in range(n_grid) if residuals[i] == 0.0]
    for i in range(n_grid - 1):
        if residuals[i] * residuals[i + 1] < 0.0:
            candidates.append(float(brentq(energy_residual, grid[i], grid[i + 1], xtol=1e-11)))

    steady_states: list[SteadyState] = []
    for bracketed in sorted(candidates):
        guess = np.array([c_a_on_mass_balance(bracketed), bracketed])
        polished = root(f, guess, method="hybr", tol=1e-14)
        spacing = grid[1] - grid[0]
        x = polished.x if polished.success and abs(polished.x[1] - bracketed) < spacing else guess

        steady_states.append(
            SteadyState(
                c_a=float(x[0]),
                temperature=float(x[1]),
                residual=f(x),
                eigenvalues=eigenvalues_at(f, x),
            )
        )
    return steady_states
