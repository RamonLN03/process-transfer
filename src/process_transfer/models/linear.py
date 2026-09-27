"""BL, the linear black box (``docs/m1_plan.md``, sections 8.3 and 8.7): the simplest
data-driven reference.

    dx/dt = A (x - x_e) + B (u - u_n)

with u_n the known nominal inputs. A is 2 x 2 and x_e the equilibrium state under u_n. B is
estimated only along the directions of the inputs that the data of the fit excite, in level
units, and is zero on the complement: the minimum-norm convention, part of the definition
of BL (section 8.7, point 3).

Level units. The level of an input is its deviation from nominal divided by its amplitude
a_i, the largest deviation of that input in the rows of the fit: P3 moves each input by
+-a_i, so its levels are the signs of the corners, -1, 0 and +1. The excited directions are
the span of the level vectors of the rows of the fit. Their rank is computed exactly, on
the rational values of the levels (``evaluation.budgets.integer_rank``); an orthonormal
basis Q (4 x r) of their span is then computed in floating point. With v = (u - u_n) / a,

    B (u - u_n) = C Q^T v,     C of size 2 x r,

so B has 2 r free values and is zero on every direction of v orthogonal to the span. An
input that never moves has no amplitude; its level is zero in every row, so its column of
B is zero whatever amplitude is written for it, and 1 is written.

Parameters: A, x_e and C, 6 + 2 r values, at most 14. The optimiser works on them in
coordinates without units: with tau_0 = 60 s, the noise levels sigma of the sensors and x_0
the starting x_e,

    theta = (A_ij tau_0 sigma_j / sigma_i, (x_e - x_0) / sigma, C_ik tau_0 / sigma_i)

row by row, the states counted in noise levels and time in units of tau_0: a change of
units and an origin, nothing else. The trust region of the optimiser is taken in these
coordinates as they are (``x_scale=1``), not rescaled by the norms of the columns of the
Jacobian as MR's is. Those norms are large, and on noise-free data of a linear plant the
first steps of the rescaled region were long enough to leave the basin of the true values
for a stiff local minimum: 200 evaluations ended at J = 1.45, where the unscaled region
reached J = 4e-6 in 11 (``tests/test_models_linear.py``).

BL is fitted like MR, on all b windows with the loss of section 5.5, by
``scipy.optimize.least_squares`` with the Jacobian of the sensitivities integrated with the
states (``fitting.WindowLoss``), from declared starts; the fit keeps the converged endpoint
with the lowest objective, the first in the declared order on a tie. It selects nothing from
data.

Starts. x_e at the mean of the initial states of the windows of the fit, which are means of
contexts read at the nominal inputs after a rest; A = -I / tau for tau in STARTS_TAU, stable
matrices with the time scale of the plant's response, from a minute down and up; C = 0.

Derivatives, for the sensitivities: df/dx = A; df/dA_ij = (x - x_e)_j in row i;
df/dx_e = -A; df/dC_ik = w_k in row i, with w = Q^T v; each column then multiplied by the
derivative of its parameter with respect to its coordinate, sigma_i / (tau_0 sigma_j),
sigma_j or sigma_i / tau_0.

Domain: every finite state and every finite input. A matrix A with an eigenvalue of positive
real part makes a rollout grow; the rollout then reports its failure (``models.rollout``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.optimize import least_squares

from process_transfer.cstr_variables import FloatArray
from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.windows import WindowData
from process_transfer.models.fitting import (
    BUDGET_EXHAUSTED,
    CONVERGED,
    NUMERICAL_FAILURE,
    FitStop,
    WindowLoss,
    check_windows,
)
from process_transfer.models.rollout import EVALUATION_SETTINGS, RolloutSettings

STARTS_TAU = (60.0, 20.0, 180.0)  # s
COORDINATE_TIME = 60.0  # s, tau_0


@dataclass(frozen=True)
class InputDirections:
    """The directions of the inputs that the rows of a fit excite, in level units."""

    nominal: FloatArray  # u_n, (4,)
    amplitudes: FloatArray  # a, (4,): the largest deviation of each input, 1 if it never moves
    basis: FloatArray  # Q, (4, r), orthonormal
    rank: int
    levels: tuple[tuple[Fraction, ...], ...]  # the distinct level vectors, exactly

    def levels_of(self, u: FloatArray) -> FloatArray:
        return (np.asarray(u, dtype=np.float64) - self.nominal) / self.amplitudes


def _reduced_rows(rows: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    """The non-zero rows of the reduced row echelon form, in exact arithmetic: a basis of
    the span of ``rows``."""
    matrix = [list(row) for row in rows]
    if not matrix:
        return []
    rank = 0
    for column in range(len(matrix[0])):
        pivot = next((r for r in range(rank, len(matrix)) if matrix[r][column] != 0), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        lead = matrix[rank][column]
        matrix[rank] = [value / lead for value in matrix[rank]]
        for r in range(len(matrix)):
            if r != rank and matrix[r][column] != 0:
                factor = matrix[r][column]
                matrix[r] = [a - factor * b for a, b in zip(matrix[r], matrix[rank], strict=True)]
        rank += 1
    return matrix[:rank]


def excited_directions(
    windows: Sequence[WindowData], nominal_inputs: FloatArray
) -> InputDirections:
    """The level units and the excited directions of the rows of ``windows``."""
    nominal = np.asarray(nominal_inputs, dtype=np.float64)
    if nominal.shape != (4,) or not np.all(np.isfinite(nominal)):
        raise ValueError(f"the nominal inputs must be four finite values, got {nominal_inputs!r}")
    rows = np.concatenate([d.inputs for d in windows])
    deviations = rows - nominal
    largest = np.max(np.abs(deviations), axis=0)
    if not np.all(np.isfinite(largest)):
        raise ValueError("the deviations of the inputs from nominal are not representable")
    amplitudes = np.where(largest > 0.0, largest, 1.0)
    distinct = {
        tuple(Fraction(float(v)) / Fraction(float(a)) for v, a in zip(row, amplitudes, strict=True))
        for row in deviations
    }
    levels = tuple(sorted(distinct))
    reduced = _reduced_rows(levels)
    rank = len(reduced)
    if rank == 0:
        basis = np.zeros((4, 0))
    else:
        spanning = np.array([[float(v) for v in row] for row in reduced]).T  # (4, r)
        basis, _ = np.linalg.qr(spanning)
    return InputDirections(nominal, amplitudes, basis, rank, levels)


@dataclass(frozen=True)
class LinearParameters:
    a: FloatArray  # (2, 2), 1/s
    equilibrium: FloatArray  # x_e, (2,)
    c: FloatArray  # (2, r), state units per second per level


@dataclass(frozen=True)
class LinearCoordinates:
    """The coordinates without units in which BL is fitted."""

    rank: int
    sigma: FloatArray  # the noise levels of the sensors, (2,)
    origin: FloatArray  # x_0, the starting x_e, (2,)
    time: float = COORDINATE_TIME

    def theta(self, parameters: LinearParameters) -> FloatArray:
        return np.concatenate(
            [
                (parameters.a * self.time * self.sigma[None, :] / self.sigma[:, None]).ravel(),
                (parameters.equilibrium - self.origin) / self.sigma,
                (parameters.c * self.time / self.sigma[:, None]).ravel(),
            ]
        )

    def parameters(self, theta: FloatArray) -> LinearParameters:
        theta = np.asarray(theta, dtype=np.float64)
        r = self.rank
        if theta.shape != (6 + 2 * r,):
            raise ValueError(f"BL with rank {r} has {6 + 2 * r} parameters, got {theta.shape}")
        return LinearParameters(
            theta[:4].reshape(2, 2) * self.sigma[:, None] / (self.time * self.sigma[None, :]),
            self.origin + self.sigma * theta[4:6],
            theta[6:].reshape(2, r) * self.sigma[:, None] / self.time,
        )

    def derivatives(self) -> FloatArray:
        """d parameter / d coordinate, in the order of theta."""
        return np.concatenate(
            [
                (self.sigma[:, None] / (self.time * self.sigma[None, :])).ravel(),
                self.sigma,
                np.repeat(self.sigma / self.time, self.rank),
            ]
        )


class LinearModel:
    """BL with the given parameters, a continuous-time model with its derivatives with
    respect to the coordinates of its fit."""

    def __init__(
        self,
        name: str,
        directions: InputDirections,
        parameters: LinearParameters,
        coordinates: LinearCoordinates,
    ) -> None:
        if coordinates.rank != directions.rank:
            raise ValueError("the coordinates and the directions of BL differ in rank")
        self.name, self.directions, self.parameters = name, directions, parameters
        self.coordinates = coordinates
        rank = directions.rank
        self.parameter_names = (
            "A_11 tau_0",
            "A_12 tau_0 sigma_2 / sigma_1",
            "A_21 tau_0 sigma_1 / sigma_2",
            "A_22 tau_0",
            "(x_e_1 - x_0_1) / sigma_1",
            "(x_e_2 - x_0_2) / sigma_2",
            *(f"C_{i + 1}{k + 1} tau_0 / sigma_{i + 1}" for i in range(2) for k in range(rank)),
        )
        # B in physical units, C Q^T diag(1 / a): zero on the directions not excited
        self.input_matrix = parameters.c @ directions.basis.T / directions.amplitudes

    def rhs(self, x: FloatArray, u: FloatArray) -> FloatArray:
        p = self.parameters
        return p.a @ (x - p.equilibrium) + self.input_matrix @ (u - self.directions.nominal)

    def initial_state_problem(self, x0: FloatArray) -> str | None:
        return None

    def jacobian_state(self, x: FloatArray, u: FloatArray) -> FloatArray:
        return self.parameters.a.copy()

    def jacobian_parameters(self, x: FloatArray, u: FloatArray) -> FloatArray:
        p, rank = self.parameters, self.directions.rank
        offset = x - p.equilibrium
        w = self.directions.basis.T @ self.directions.levels_of(u)
        jac = np.zeros((2, 6 + 2 * rank))
        for i in range(2):
            jac[i, 2 * i : 2 * i + 2] = offset  # A_i1, A_i2
            jac[i, 6 + rank * i : 6 + rank * (i + 1)] = w
        jac[:, 4:6] = -p.a
        return jac * self.coordinates.derivatives()


@dataclass(frozen=True)
class LinearStart:
    tau: float  # s
    outcome: str
    message: str
    final: LinearParameters | None
    objective: float | None
    singular_values: tuple[float, ...] | None
    evaluations: int
    seconds: float = field(compare=False)


@dataclass(frozen=True)
class LinearFit:
    """A fit of BL: its directions, every start, the one selected, or the training failure."""

    name: str
    windows: tuple[tuple[str, int], ...]
    directions: InputDirections
    coordinates: LinearCoordinates
    starts: tuple[LinearStart, ...]
    selected: int | None
    training_failure: TrainingFailure | None

    @property
    def parameters(self) -> LinearParameters | None:
        return None if self.selected is None else self.starts[self.selected].final

    @property
    def objective(self) -> float | None:
        return None if self.selected is None else self.starts[self.selected].objective

    def model(self) -> LinearModel:
        if self.parameters is None:
            raise ValueError(f"{self.name} has no model: {self.training_failure}")
        return LinearModel(self.name, self.directions, self.parameters, self.coordinates)


def fit_linear(
    name: str,
    windows: Sequence[WindowData],
    nominal_inputs: FloatArray,
    rollout_settings: RolloutSettings = EVALUATION_SETTINGS,
    max_evaluations: int = 200,
    tolerance: float = 1e-10,
) -> LinearFit:
    """Estimate BL on ``windows``, from every declared start."""
    windows = tuple(windows)
    check_windows(windows)
    directions = excited_directions(windows, nominal_inputs)
    rank = directions.rank
    equilibrium = np.mean([d.initial_state for d in windows], axis=0)
    coordinates = LinearCoordinates(rank, np.array(windows[0].noise_std), equilibrium)
    records = []
    for tau in STARTS_TAU:
        initial = LinearParameters(-np.eye(2) / tau, equilibrium.copy(), np.zeros((2, rank)))

        def build(theta: FloatArray) -> LinearModel:
            return LinearModel(name, directions, coordinates.parameters(theta), coordinates)

        loss = WindowLoss(windows, build, rollout_settings)
        began = time.perf_counter()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                found = least_squares(
                    loss.residuals,
                    coordinates.theta(initial),
                    jac=loss.jacobian,
                    method="trf",
                    ftol=tolerance,
                    xtol=tolerance,
                    gtol=tolerance,
                    x_scale=1.0,
                    loss="linear",
                    max_nfev=max_evaluations,
                    tr_solver="exact",
                    verbose=0,
                )
        except (FitStop, FloatingPointError) as stop:
            records.append(
                LinearStart(
                    tau,
                    NUMERICAL_FAILURE,
                    str(stop),
                    None,
                    None,
                    None,
                    loss.evaluations,
                    time.perf_counter() - began,
                )
            )
            continue
        records.append(
            LinearStart(
                tau,
                CONVERGED if found.status > 0 else BUDGET_EXHAUSTED,
                f"status {found.status}: {found.message}",
                coordinates.parameters(found.x),
                math.sqrt(2.0 * found.cost),
                tuple(float(s) for s in np.linalg.svd(found.jac / loss.scale, compute_uv=False)),
                loss.evaluations,
                time.perf_counter() - began,
            )
        )
    converged = [i for i, r in enumerate(records) if r.outcome == CONVERGED]
    selected = min(converged, key=lambda i: (records[i].objective, i)) if converged else None
    failure = None
    if selected is None:
        failure = TrainingFailure(
            f"no start of {name} converged: "
            + "; ".join(f"tau {r.tau:g} s: {r.outcome} ({r.message})" for r in records)
        )
    return LinearFit(
        name,
        tuple(d.key for d in windows),
        directions,
        coordinates,
        tuple(records),
        selected,
        failure,
    )
