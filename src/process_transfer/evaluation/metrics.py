"""Metrics of predictions against readings (``docs/m1_plan.md``, section 9.1).

For each channel, the mean squared error and its root against the scored readings, in
physical units, mol/m^3 and K. To put the channels together, and to read an error against
the noise, each is divided by the noise level of its sensor from the data sheet:

    J = sqrt((MSE_CA / sigma_CA^2 + MSE_T / sigma_T^2) / 2)

pooled over the scored readings it is computed on. The sigmas are fixed and known; no
spread of the data enters. Every score is given pooled, for each phase of the layout, and
for each window. Only scored readings are ever compared with a prediction: a window hands
its context readings to the initial state and to nothing else.

What readings can say about the true state. A scored reading is y = x + e. When e has zero
mean and variance sigma^2 and is independent of everything the prediction p was computed
from, the fitted model, the data it was fitted and selected on, and the context of its
window, then E[(p - y)^2 | p] = (p - x)^2 + sigma^2, and MSE - sigma^2 pooled over scored
readings is an unbiased estimate of the mean squared error against the true states. It is
not one on the data a model was fitted on, which it was fitted to, nor on the validation
data a choice was made on, which selection makes optimistic. So every evaluation states the
role its readings played for the model, and the estimate is computed only for readings
held out from it; for the others it is absent, not zero. It is reported as it comes,
negative values included, and never clipped. Its square root is not an unbiased estimate
of a root mean square error and is not offered.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from process_transfer.cstr_variables import STATE_NAMES, FloatArray
from process_transfer.evaluation.windows import WindowData


class Role(Enum):
    """What the scored readings were to the model that is scored on them."""

    FITTING = "fitting"  # the model was fitted on them
    SELECTION = "selection"  # a checkpoint or a configuration was chosen on them
    HELD_OUT = "held out"  # neither the model nor any choice about it saw them


@dataclass(frozen=True)
class Score:
    """The errors of predictions against a set of scored readings."""

    readings: int
    mse: tuple[float, ...]  # per channel, in the square of its unit
    rmse: tuple[float, ...]  # per channel, in its unit
    normalised_mse: tuple[float, ...]  # MSE / sigma^2 per channel
    j: float
    # Only for held-out readings: MSE - sigma^2 per channel, and J^2 - 1. None otherwise.
    excess_mse: tuple[float, ...] | None
    excess_j_squared: float | None


def score(errors: FloatArray, sigma: FloatArray, role: Role) -> Score:
    """The score of prediction errors, one row per scored reading and one column per
    channel, against the noise levels ``sigma``."""
    errors = np.asarray(errors, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != len(STATE_NAMES) or errors.shape[0] == 0:
        raise ValueError(
            f"errors must have one row per scored reading and {len(STATE_NAMES)} columns, "
            f"and at least one row; got shape {errors.shape}"
        )
    if not np.all(np.isfinite(errors)):
        raise ValueError(
            "errors must be finite; a prediction that is not finite is an integration "
            "failure, recorded as such and never scored"
        )
    for name, value in zip(STATE_NAMES, sigma, strict=True):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(
                f"the score divides by the noise level of each channel, and {name} has "
                f"{value!r}; a normalised score does not exist for an exact sensor"
            )
    if not isinstance(role, Role):
        raise ValueError(f"role must be a Role, got {role!r}")
    # Errors are squared and averaged after dividing them by their largest magnitude, so that
    # no square overflows or underflows on the way to a result that is representable; a
    # result that is not representable is refused, never returned as inf or as zero.
    with np.errstate(over="ignore", under="ignore"):
        normalised_errors = errors / sigma
    if not np.all(np.isfinite(normalised_errors)):
        raise ValueError(
            "the errors divided by the noise levels are not representable in double "
            "precision; no score is computed for them"
        )
    channels = range(len(STATE_NAMES))
    mse = tuple(_mean_square(errors[:, c]) for c in channels)
    normalised = tuple(_mean_square(normalised_errors[:, c]) for c in channels)
    for label, values in (("MSE", mse), ("MSE / sigma^2", normalised)):
        for name, value in zip(STATE_NAMES, values, strict=True):
            if not math.isfinite(value):
                raise ValueError(
                    f"the {label} of {name} is not representable in double precision; no "
                    "score is computed"
                )
    excess = excess_j = None
    if role is Role.HELD_OUT:
        with np.errstate(over="ignore"):
            variance = sigma * sigma
        for name, value in zip(STATE_NAMES, variance, strict=True):
            if not math.isfinite(value):
                raise ValueError(
                    f"MSE - sigma^2 of {name} is not representable in double precision: the "
                    f"square of its noise level, {value!r}, overflows"
                )
        excess = tuple(m - v for m, v in zip(mse, variance.tolist(), strict=True))
        excess_j = sum(n / len(normalised) for n in normalised) - 1.0
    return Score(
        readings=int(errors.shape[0]),
        mse=mse,
        rmse=tuple(_root_mean_square(errors[:, c]) for c in channels),
        normalised_mse=normalised,
        j=_root_mean_square(normalised_errors.ravel()),
        excess_mse=excess,
        excess_j_squared=excess_j,
    )


def _mean_square(values: FloatArray) -> float:
    """mean(v^2), from the values divided by their largest magnitude m: m (m mean((v/m)^2)).
    Nothing overflows on the way, the scaled squares lose to underflow only what is below
    the resolution of the result, and the value is inf only when the result exceeds the
    largest double."""
    largest = float(np.max(np.abs(values)))
    if largest == 0.0:
        return 0.0
    with np.errstate(over="ignore", under="ignore"):
        return largest * (largest * float(np.mean((values / largest) ** 2)))


def _root_mean_square(values: FloatArray) -> float:
    """sqrt(mean(v^2)), scaled the same way. It never exceeds the largest magnitude of the
    values, so it is always representable."""
    largest = float(np.max(np.abs(values)))
    if largest == 0.0:
        return 0.0
    with np.errstate(under="ignore"):
        return largest * float(np.sqrt(np.mean((values / largest) ** 2)))


@dataclass(frozen=True)
class Evaluation:
    """The scores of one model on a set of windows of one layout."""

    role: Role
    pooled: Score
    by_phase: tuple[tuple[str, Score], ...]  # in the order of the layout
    by_window: tuple[tuple[tuple[str, int], Score], ...]  # (run, excursion) in the given order


def evaluate(
    windows: Sequence[WindowData], predictions: Sequence[FloatArray], role: Role
) -> Evaluation:
    """Score predictions, one array of states at the scored instants per window, against
    the scored readings of the windows."""
    if len(windows) == 0:
        raise ValueError("nothing to evaluate: no window")
    if len(predictions) != len(windows):
        raise ValueError(f"{len(predictions)} predictions for {len(windows)} windows")
    layout = windows[0].window.layout
    sigma = windows[0].noise_std
    for data in windows:
        if data.window.layout != layout:
            raise ValueError("an evaluation pools windows of one layout only")
        if not np.array_equal(data.noise_std, sigma):
            raise ValueError("an evaluation pools windows with the same noise levels only")
    keys = [data.key for data in windows]
    if len(set(keys)) != len(keys):
        raise ValueError("a window is given twice")
    errors = []
    for data, predicted in zip(windows, predictions, strict=True):
        predicted = np.asarray(predicted, dtype=np.float64)
        if predicted.shape != data.scored.shape:
            raise ValueError(
                f"the prediction of window {data.key} has shape {predicted.shape}; its scored "
                f"readings have shape {data.scored.shape}"
            )
        errors.append(predicted - data.scored)
    stacked = np.stack(errors)  # (windows, L, channels)
    by_phase = tuple(
        (name, score(stacked[:, positions].reshape(-1, stacked.shape[2]), sigma, role))
        for name, positions in layout.phase_slices()
    )
    return Evaluation(
        role=role,
        pooled=score(stacked.reshape(-1, stacked.shape[2]), sigma, role),
        by_phase=by_phase,
        by_window=tuple(
            (key, score(error, sigma, role)) for key, error in zip(keys, errors, strict=True)
        ),
    )
