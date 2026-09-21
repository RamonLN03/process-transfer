"""Does a series of measurement errors look like the noise that was specified?

A diagnostic of the simulated sensors. It works on measurement errors, reading minus
exact value, which exist only on the truth side of the project; no model is ever given
them. The functions take plain arrays and know nothing about the plant.

Every statistic is turned into a z-score: its distance from what independent zero-mean
Gaussian noise of standard deviation ``sigma`` would give, in units of the standard
deviation that such noise gives it over ``n`` samples. A realisation of random noise
never has a mean of exactly zero or a standard deviation of exactly ``sigma``; what can
be asked is that it lies where chance would put it. For correct noise each z-score is
close to standard normal, so |z| > 4 happens about once in 16 000 statistics.

    mean            z = mean sqrt(n) / sigma
    spread          Wilson-Hilferty: (S / n)^(1/3) is close to normal, with mean
                    1 - 2/(9n) and variance 2/(9n), where S = sum(e^2) / sigma^2 follows
                    a chi-square law with n degrees of freedom. The known zero mean is
                    used, not the sample mean, so a bias also shows up here
    lag-one         r1 = sum(e_k e_k+1) / sum(e_k^2), z = r1 sqrt(n)
    tails           the fraction of |e| beyond k sigma against 2 (1 - Phi(k)), with the
                    binomial standard deviation. This is what a clipped, truncated or
                    non-Gaussian noise of the right spread would fail
    correlation     r between two series, z = r sqrt(n). Valid when at least one of
                    the two is white, which the errors are under the hypothesis; the
                    other may be as autocorrelated as a state trajectory

The approximations are those of large n. They are meant for the hundreds or thousands
of samples of a run, not for a handful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from process_transfer.cstr_variables import FloatArray
from process_transfer.validation import require_positive

MINIMUM_SAMPLES = 100  # below this the large-sample z-scores are not trustworthy
_LARGEST_RATIO = 1.0e100  # |error| / sigma beyond which the squares would overflow
_LARGEST_VALUE = 1.0e100  # magnitude beyond which sums of squares would overflow


@dataclass(frozen=True)
class NoiseStatistics:
    """Empirical statistics of one error series, each with its z-score."""

    n: int
    sigma: float  # the specified standard deviation the series is compared with
    mean: float
    mean_z: float
    rms: float  # root mean square about zero, the estimate of sigma
    spread_z: float
    lag_one: float
    lag_one_z: float
    beyond_one_sigma: float  # fraction of samples with |error| > sigma
    beyond_one_sigma_z: float
    beyond_two_sigma: float
    beyond_two_sigma_z: float
    largest_in_sigmas: float  # max |error| / sigma: sigma is not a bound

    @property
    def z_scores(self) -> dict[str, float]:
        return {
            "mean": self.mean_z,
            "spread": self.spread_z,
            "lag_one": self.lag_one_z,
            "beyond_one_sigma": self.beyond_one_sigma_z,
            "beyond_two_sigma": self.beyond_two_sigma_z,
        }


def _series(name: str, values: FloatArray) -> FloatArray:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or len(series) < MINIMUM_SAMPLES:
        raise ValueError(
            f"{name} must be a vector of at least {MINIMUM_SAMPLES} samples, got shape "
            f"{series.shape}"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError(f"{name} must be finite")
    if float(np.max(np.abs(series))) > _LARGEST_VALUE:
        # sums of squares of larger numbers overflow; no error or state is of this size
        raise ValueError(f"{name} holds values beyond {_LARGEST_VALUE:g}, outside the domain")
    return series


def _tail_z(fraction: float, k: float, n: int) -> float:
    expected = 2.0 * float(norm.sf(k))
    return (fraction - expected) / math.sqrt(expected * (1.0 - expected) / n)


def noise_statistics(errors: FloatArray, sigma: float) -> NoiseStatistics:
    """Statistics of ``errors`` against zero-mean Gaussian noise of deviation ``sigma``.

    ``sigma`` must be positive. An exact sensor, ``sigma = 0``, is a valid sensor but has
    no noise to test: its errors are to be compared with zero exactly, not passed here.
    """
    e = _series("errors", errors)
    sigma = require_positive("sigma", sigma)
    n = len(e)
    largest = float(np.max(np.abs(e)))
    if not math.isfinite(largest / sigma) or largest / sigma > _LARGEST_RATIO:
        # finite errors and a finite sigma can still overflow in the ratio or its square
        raise ValueError(
            f"the largest error, {largest!r}, is more than {_LARGEST_RATIO:g} times sigma = "
            f"{sigma!r}; this is not noise of that specification, and its statistics would "
            "overflow"
        )
    scaled = e / sigma

    mean = float(np.mean(e))
    mean_square = float(np.mean(scaled**2))
    wilson_hilferty_mean = 1.0 - 2.0 / (9.0 * n)
    spread_z = (mean_square ** (1.0 / 3.0) - wilson_hilferty_mean) / math.sqrt(2.0 / (9.0 * n))

    sum_of_squares = float(np.sum(scaled**2))
    if sum_of_squares == 0.0:
        # every error is exactly zero although noise was specified: there is no
        # autocorrelation to speak of, and the spread and tail scores already say it
        lag_one = 0.0
    else:
        lag_one = float(np.sum(scaled[:-1] * scaled[1:])) / sum_of_squares

    beyond_one = float(np.mean(np.abs(scaled) > 1.0))
    beyond_two = float(np.mean(np.abs(scaled) > 2.0))
    return NoiseStatistics(
        n=n,
        sigma=sigma,
        mean=mean,
        mean_z=mean * math.sqrt(n) / sigma,
        rms=sigma * math.sqrt(mean_square),
        spread_z=spread_z,
        lag_one=lag_one,
        lag_one_z=lag_one * math.sqrt(n),
        beyond_one_sigma=beyond_one,
        beyond_one_sigma_z=_tail_z(beyond_one, 1.0, n),
        beyond_two_sigma=beyond_two,
        beyond_two_sigma_z=_tail_z(beyond_two, 2.0, n),
        largest_in_sigmas=float(np.max(np.abs(scaled))),
    )


def correlation(a: FloatArray, b: FloatArray) -> tuple[float, float]:
    """Pearson correlation of two series of equal length and its z-score, r sqrt(n).

    A series without any variation has no correlation with anything; that is an
    invalid input here, not a zero.
    """
    x, y = _series("a", a), _series("b", b)
    if len(x) != len(y):
        raise ValueError(f"the two series differ in length: {len(x)} and {len(y)}")
    dx, dy = x - np.mean(x), y - np.mean(y)
    norm_x, norm_y = float(np.sqrt(np.sum(dx**2))), float(np.sqrt(np.sum(dy**2)))
    if norm_x == 0.0 or norm_y == 0.0:
        raise ValueError("a constant series has no correlation to compute")
    r = float(np.sum((dx / norm_x) * (dy / norm_y)))
    return r, r * math.sqrt(len(x))
