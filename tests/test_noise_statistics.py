"""The z-scores of the noise diagnostics: calibrated on correct noise, and sensitive to
the mistakes they exist to catch.

All generators are seeded with fixed, arbitrary seeds. The limit |z| <= 4 used throughout
the project is not tuned to these seeds: the tests below either check the calibration
itself over thousands of replicates, or use defects whose z-scores are far beyond it.
"""

import numpy as np
import pytest

from process_transfer.measurement.noise_statistics import (
    MINIMUM_SAMPLES,
    correlation,
    noise_statistics,
)

N_RUN = 1201  # readings per variable in two hours at one reading every 6 s
SIGMA = 5.0


def test_the_z_scores_are_standard_normal_for_correct_noise() -> None:
    """Calibration: over many realisations of correct noise every z-score has mean 0 and
    standard deviation 1, so a limit of 4 means what it is said to mean."""
    rng = np.random.default_rng(101)
    replicates = 2000
    scores: dict[str, list[float]] = {}
    for _ in range(replicates):
        found = noise_statistics(SIGMA * rng.standard_normal(N_RUN), SIGMA)
        for name, z in found.z_scores.items():
            scores.setdefault(name, []).append(z)
    for name, values in scores.items():
        assert abs(np.mean(values)) < 0.10, name  # 4.5 standard errors of the mean
        assert 0.93 < np.std(values) < 1.07, name  # 4.4 standard errors of the deviation
        assert np.mean(np.abs(values) > 4.0) < 0.002, name


def test_the_correlation_score_is_calibrated_against_an_autocorrelated_series() -> None:
    """The errors are compared with the true states, which are anything but white. It is
    enough that one of the two series is white for r sqrt(n) to be standard normal."""
    rng = np.random.default_rng(102)
    scores = []
    for _ in range(2000):
        white = rng.standard_normal(N_RUN)
        wandering = np.cumsum(rng.standard_normal(N_RUN))  # a random walk
        scores.append(correlation(white, wandering)[1])
    assert abs(np.mean(scores)) < 0.10
    assert 0.93 < np.std(scores) < 1.07


def test_a_correct_series_is_described_correctly() -> None:
    errors = SIGMA * np.random.default_rng(103).standard_normal(100_000)
    found = noise_statistics(errors, SIGMA)
    assert found.n == 100_000 and found.sigma == SIGMA
    assert found.mean == pytest.approx(0.0, abs=4 * SIGMA / np.sqrt(found.n))
    assert found.rms == pytest.approx(SIGMA, rel=0.01)
    assert found.beyond_one_sigma == pytest.approx(0.3173, abs=0.006)
    assert found.beyond_two_sigma == pytest.approx(0.0455, abs=0.003)
    assert found.largest_in_sigmas > 3.5  # sigma is not a bound
    assert all(abs(z) <= 4.0 for z in found.z_scores.values())


def test_the_noise_level_that_was_not_chosen_would_be_seen_in_one_run() -> None:
    """D-020: 3.8 mol/m^3 on the target instead of 5 mol/m^3 is a spread 24 % too small,
    twelve standard deviations away over the 1201 readings of a two-hour run."""
    errors = 3.8 * np.random.default_rng(104).standard_normal(N_RUN)
    assert noise_statistics(errors, 5.0).spread_z < -8.0
    assert abs(noise_statistics(errors, 3.8).spread_z) <= 4.0


def test_a_bias_clipping_a_wrong_shape_and_memory_are_each_seen() -> None:
    rng = np.random.default_rng(105)
    noise = SIGMA * rng.standard_normal(N_RUN)

    biased = noise_statistics(noise + 0.5 * SIGMA, SIGMA)
    assert biased.mean_z > 10.0

    clipped = noise_statistics(np.clip(noise, -SIGMA, SIGMA), SIGMA)  # sigma used as a bound
    assert clipped.spread_z < -10.0 and clipped.beyond_one_sigma == 0.0
    assert clipped.largest_in_sigmas == 1.0

    half_width = SIGMA * np.sqrt(3.0)  # uniform noise with exactly the right deviation
    uniform = noise_statistics(rng.uniform(-half_width, half_width, N_RUN), SIGMA)
    assert abs(uniform.mean_z) <= 4.0 and uniform.beyond_one_sigma_z > 5.0

    remembered = np.empty(N_RUN)  # first-order autoregressive noise, correlation 0.3
    remembered[0] = noise[0]
    for k in range(1, N_RUN):
        remembered[k] = 0.3 * remembered[k - 1] + np.sqrt(1.0 - 0.3**2) * noise[k]
    assert noise_statistics(remembered, SIGMA).lag_one_z > 6.0


def test_errors_that_are_all_zero_are_reported_without_a_warning() -> None:
    """Noise was specified and none is there, as if a sensor had been left exact. The
    spread says so; nothing is divided by the zero sum of squares."""
    found = noise_statistics(np.zeros(N_RUN), SIGMA)
    assert found.rms == 0.0 and found.lag_one == 0.0
    # the most negative value the score can take at this n: (0 - (1 - 2/(9n))) / sqrt(2/(9n))
    variance = 2.0 / (9.0 * N_RUN)
    assert found.spread_z == pytest.approx(-(1.0 - variance) / np.sqrt(variance))
    assert found.spread_z < -70.0
    assert np.isfinite(list(found.z_scores.values())).all()


def test_correlation_of_related_and_unrelated_series() -> None:
    rng = np.random.default_rng(106)
    a, b = rng.standard_normal(N_RUN), rng.standard_normal(N_RUN)
    assert abs(correlation(a, b)[1]) <= 4.0
    r, z = correlation(a, a)
    assert r == pytest.approx(1.0) and z == pytest.approx(np.sqrt(N_RUN))
    assert correlation(a, -2.0 * a + 7.0)[0] == pytest.approx(-1.0)
    assert correlation(a, a + b)[1] > 15.0  # shared noise, as two runs on one stream would have


@pytest.mark.parametrize(
    ("errors", "sigma", "message"),
    [
        (np.zeros(N_RUN), 0.0, "sigma must be greater than zero"),
        (np.zeros(N_RUN), -1.0, "sigma must be greater than zero"),
        (np.zeros(N_RUN), np.nan, "sigma must be finite"),
        (np.zeros(MINIMUM_SAMPLES - 1), 1.0, "at least 100 samples"),
        (np.zeros((N_RUN, 2)), 1.0, "at least 100 samples"),
        (np.full(N_RUN, np.nan), 1.0, "must be finite"),
        (np.full(N_RUN, 1.0e200), 1.0, "outside the domain"),
        (np.full(N_RUN, 1.0), 1.0e-300, "times sigma"),
    ],
)
def test_invalid_input_is_rejected(errors: np.ndarray, sigma: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        noise_statistics(errors, sigma)


def test_correlation_rejects_what_it_cannot_compute() -> None:
    a = np.random.default_rng(107).standard_normal(N_RUN)
    with pytest.raises(ValueError, match="constant series"):
        correlation(a, np.full(N_RUN, 350.0))
    with pytest.raises(ValueError, match="differ in length"):
        correlation(a, a[:-1])
    with pytest.raises(ValueError, match="must be finite"):
        correlation(a, np.where(np.arange(N_RUN) == 5, np.inf, a))
