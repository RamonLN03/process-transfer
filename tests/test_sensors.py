"""The measurement component: additive Gaussian noise, seeded explicitly, never clipped.

The seeds used here are fixed and arbitrary. They are not those of any experiment, and
none was chosen for the result it gives. Statistical assertions use the z-scores of
``measurement/noise_statistics.py`` with a limit of 4, which correct noise exceeds about
once in 16 000 statistics; the sample sizes are large so that a wrong noise level is far
outside it.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from process_transfer.config import load_sensors
from process_transfer.measurement import sensors as sensors_module
from process_transfer.measurement.noise_statistics import correlation, noise_statistics
from process_transfer.measurement.sensors import (
    MeasurementSpec,
    SensorSpec,
    channel_generator,
    measure,
    noise_key,
    noise_stream,
)

Z_LIMIT = 4.0
SEED, STREAM = 11, (0, 0)
C_A = SensorSpec("C_A", "mol/m^3", 5.0)
T = SensorSpec("T", "K", 0.5)
N = 200_000


def clean(n: int = N) -> np.ndarray:
    return np.column_stack([np.full(n, 250.0), np.full(n, 350.0)])


# --------------------------------------------------------------------------- #
# Specification and units
# --------------------------------------------------------------------------- #


def test_the_specification_of_d020_in_si(configs_dir: Path) -> None:
    spec = MeasurementSpec.from_config(load_sensors(configs_dir / "sensors_cstr.yaml"))
    assert spec.sample_period == pytest.approx(6.0)  # 0.1 min
    assert spec.variables == ("C_A", "T")
    assert spec.noise_std == (pytest.approx(5.0), 0.5)  # 0.005 mol/L is 5 mol/m^3
    assert [sensor.unit for sensor in spec.sensors] == ["mol/m^3", "K"]


def test_the_empirical_deviation_is_the_specified_one_in_si_units() -> None:
    errors = measure(clean(), (C_A, T), SEED, STREAM) - clean()
    for column, sensor in enumerate((C_A, T)):
        found = noise_statistics(errors[:, column], sensor.noise_std)
        assert all(abs(z) <= Z_LIMIT for z in found.z_scores.values()), found
        assert found.rms == pytest.approx(sensor.noise_std, rel=0.01)
    # a mistake of units, mol/L read as mol/m^3, would be a factor of 1000
    assert abs(noise_statistics(errors[:, 0], 0.005).spread_z) > 1000.0


def test_sigma_is_a_standard_deviation_not_a_bound() -> None:
    errors = measure(clean(), (C_A, T), SEED, STREAM) - clean()
    beyond = np.mean(np.abs(errors[:, 1]) > T.noise_std)
    assert beyond == pytest.approx(0.3173, abs=0.005)  # about a third, as Gaussian noise gives
    assert np.max(np.abs(errors[:, 1])) > 3.5 * T.noise_std  # and nothing is truncated


def test_readings_are_not_clipped_to_a_physical_range() -> None:
    """A concentration of 1 mol/m^3 read with a noise of 5 mol/m^3 gives negative
    readings in about four cases out of ten. They are left as the sensor gave them."""
    low = np.column_stack([np.full(10_000, 1.0), np.full(10_000, 350.0)])
    readings = measure(low, (C_A, T), SEED, STREAM)
    assert 0.35 < np.mean(readings[:, 0] < 0.0) < 0.49
    assert readings[:, 0].min() < -10.0


# --------------------------------------------------------------------------- #
# Limits and invalid input
# --------------------------------------------------------------------------- #


def test_zero_noise_is_an_exact_sensor() -> None:
    values = np.column_stack([np.linspace(100.0, 300.0, 50), np.linspace(340.0, 370.0, 50)])
    exact_c_a = SensorSpec("C_A", "mol/m^3", 0.0)
    readings = measure(values, (exact_c_a, T), SEED, STREAM)
    np.testing.assert_array_equal(readings[:, 0], values[:, 0])  # bit for bit
    assert not np.array_equal(readings[:, 1], values[:, 1])
    # the other channel is what it would have been anyway: channels do not share a stream
    np.testing.assert_array_equal(readings[:, 1], measure(values, (C_A, T), SEED, STREAM)[:, 1])

    both_exact = measure(values, (exact_c_a, SensorSpec("T", "K", 0.0)), SEED, STREAM)
    np.testing.assert_array_equal(both_exact, values)
    assert both_exact is not values  # a copy: the caller's array is not handed back


@pytest.mark.parametrize("bad", [-0.5, np.nan, np.inf, -np.inf])
def test_an_invalid_noise_level_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="noise_std of T"):
        SensorSpec("T", "K", bad)


def test_an_invalid_specification_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_period"):
        MeasurementSpec(0.0, (T,))
    with pytest.raises(ValueError, match="sample_period"):
        MeasurementSpec(np.nan, (T,))
    with pytest.raises(ValueError, match="at least one sensor"):
        MeasurementSpec(6.0, ())
    with pytest.raises(ValueError, match="one sensor only"):
        MeasurementSpec(6.0, (T, T))
    with pytest.raises(ValueError, match="name the variable"):
        SensorSpec("", "K", 0.5)


def test_values_that_cannot_be_measured_are_rejected() -> None:
    with pytest.raises(ValueError, match="one column per sensor"):
        measure(np.zeros((10, 3)), (C_A, T), SEED, STREAM)
    with pytest.raises(ValueError, match="one column per sensor"):
        measure(np.zeros(10), (C_A, T), SEED, STREAM)
    bad = clean(10)
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite value"):
        measure(bad, (C_A, T), SEED, STREAM)
    huge = np.column_stack([np.full(10, 250.0), np.full(10, 1.7e308)])
    with pytest.raises(ValueError, match="overflows"):
        measure(huge, (C_A, SensorSpec("T", "K", 1.0e308)), SEED, STREAM)
    assert measure(np.zeros((0, 2)), (C_A, T), SEED, STREAM).shape == (0, 2)  # nothing to read


@pytest.mark.parametrize("bad", [None, -1, 1.5, "7", True])
def test_the_seed_must_be_named(bad: object) -> None:
    """No default and no ``None``: a seed taken from the clock cannot be reproduced."""
    with pytest.raises(ValueError, match="seed must be an explicit non-negative integer|negative"):
        measure(clean(10), (C_A, T), bad, STREAM)
    with pytest.raises(ValueError, match="stream"):
        measure(clean(10), (C_A, T), SEED, (0, bad))
    assert "seed" in inspect.signature(measure).parameters
    assert inspect.signature(measure).parameters["seed"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------- #
# Reproducibility and independence
# --------------------------------------------------------------------------- #


def test_the_same_seed_and_stream_reproduce_the_readings() -> None:
    first = measure(clean(1000), (C_A, T), SEED, STREAM)
    np.testing.assert_array_equal(first, measure(clean(1000), (C_A, T), SEED, STREAM))
    assert not np.array_equal(first, measure(clean(1000), (C_A, T), SEED + 1, STREAM))
    assert not np.array_equal(first, measure(clean(1000), (C_A, T), SEED, (1, 0)))
    assert not np.array_equal(first, measure(clean(1000), (C_A, T), SEED, (0, 1)))


def test_the_noise_does_not_depend_on_any_global_or_foreign_generator() -> None:
    """Draws made elsewhere, before or in between, change nothing, and measuring leaves
    the global generator of numpy where it was."""
    reference = measure(clean(1000), (C_A, T), SEED, STREAM)

    np.random.seed(12345)
    np.random.normal(size=777)
    np.random.default_rng(3).standard_normal(55)
    before = np.random.get_state()
    again = measure(clean(1000), (C_A, T), SEED, STREAM)
    after = np.random.get_state()

    np.testing.assert_array_equal(again, reference)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert "np.random.seed" not in inspect.getsource(sensors_module)
    assert "np.random.normal" not in inspect.getsource(sensors_module)


def test_the_noise_is_a_function_of_seed_stream_and_channel_only() -> None:
    """Not of the values measured, not of the noise level, not of the length of the run."""
    zeros = np.zeros((500, 2))
    unit = (SensorSpec("C_A", "mol/m^3", 1.0), SensorSpec("T", "K", 1.0))
    z = measure(zeros, unit, SEED, STREAM)  # the standard normal draws themselves

    np.testing.assert_array_equal(measure(zeros, (C_A, T), SEED, STREAM), z * [5.0, 0.5])
    np.testing.assert_array_equal(measure(np.zeros((200, 2)), unit, SEED, STREAM), z[:200])
    np.testing.assert_array_equal(
        z[:, 1], channel_generator(SEED, STREAM, channel=1).standard_normal(500)
    )
    shifted = measure(clean(500), (C_A, T), SEED, STREAM) - clean(500)
    np.testing.assert_allclose(shifted, z * [5.0, 0.5], rtol=0.0, atol=1e-12)  # rounding only


def test_channels_streams_and_seeds_are_uncorrelated() -> None:
    n = 50_000
    base = measure(np.zeros((n, 2)), (C_A, T), SEED, STREAM)
    other_stream = measure(np.zeros((n, 2)), (C_A, T), SEED, (1, 0))
    other_seed = measure(np.zeros((n, 2)), (C_A, T), SEED + 1, STREAM)
    pairs = {
        "C_A against T": (base[:, 0], base[:, 1]),
        "C_A of two plants": (base[:, 0], other_stream[:, 0]),
        "T of two plants": (base[:, 1], other_stream[:, 1]),
        "C_A of one plant against T of the other": (base[:, 0], other_stream[:, 1]),
        "C_A of two seeds": (base[:, 0], other_seed[:, 0]),
        "T of two seeds": (base[:, 1], other_seed[:, 1]),
    }
    for label, (a, b) in pairs.items():
        r, z = correlation(a, b)
        assert abs(z) <= Z_LIMIT, (label, r, z)


def test_the_diagnostics_are_calibrated_through_the_sensors_over_many_seeds() -> None:
    """No seed is chosen here: seeds 0 to 249, two streams and two channels, a thousand
    series of the 1201 readings of a two-hour run. Every family of z-scores has mean 0
    and deviation 1, so a run whose scores lean to one side, as one realisation in
    twenty does, is chance and not a bias of the sensors or of the statistics."""
    zeros = np.zeros((1201, 2))
    scores: dict[str, list[float]] = {}
    for seed in range(250):
        for plant in (0, 1):
            errors = measure(zeros, (C_A, T), seed, (plant, 0))
            for column, sensor in enumerate((C_A, T)):
                found = noise_statistics(errors[:, column], sensor.noise_std)
                for name, z in found.z_scores.items():
                    scores.setdefault(name, []).append(z)
    for name, values in scores.items():
        assert len(values) == 1000
        assert abs(np.mean(values)) < 0.15, name  # 4.7 standard errors of the mean
        assert 0.90 < np.std(values) < 1.10, name  # 4.5 standard errors of the deviation


def test_a_key_of_two_to_the_thirty_two_would_be_the_noise_of_another_stream() -> None:
    """Regression. numpy splits a key of 2**32 or more into several 32-bit words, so the
    stream (2**32,) became the words of the stream (0, 1) and replayed its noise exactly.
    Stream elements and channels must fit one word; the seed is not affected."""
    zeros = np.zeros((50, 2))
    with pytest.raises(ValueError, match="must be below 4294967296"):
        measure(zeros, (C_A, T), SEED, (2**32,))
    with pytest.raises(ValueError, match="must be below 4294967296"):
        channel_generator(SEED, (0,), channel=2**32)
    largest = measure(zeros, (C_A, T), SEED, (2**32 - 1,))
    assert not np.array_equal(largest, measure(zeros, (C_A, T), SEED, (2**32 - 1, 0)))
    assert not np.array_equal(
        measure(zeros, (C_A, T), 2**32, STREAM), measure(zeros, (C_A, T), 1, STREAM)
    )


def test_the_single_rule_for_noise_keys() -> None:
    assert noise_key("seed", np.int64(7)) == 7 and type(noise_key("seed", np.int64(7))) is int
    assert noise_stream((np.uint16(3), 0)) == (3, 0)
    assert noise_stream([]) == ()
    for bad in (0.9, 1.0, True, np.bool_(True), "0", None, -1, np.int64(-1), np.float64(2.0)):
        with pytest.raises(ValueError):
            noise_key("seed", bad)
    for bad in ("01", b"01", 5, None, (0, 0.5), (0, "1"), (True, 0)):
        with pytest.raises(ValueError, match="stream"):
            noise_stream(bad)


def test_the_channel_of_a_sensor_can_be_its_own_identity() -> None:
    """By default the channel is the position. With explicit channels a sensor keeps its
    noise when it is measured alone or in another order."""
    zeros = np.zeros((100, 2))
    both = measure(zeros, (C_A, T), SEED, STREAM)
    np.testing.assert_array_equal(measure(zeros, (C_A, T), SEED, STREAM, channels=(0, 1)), both)
    alone = measure(zeros[:, :1], (T,), SEED, STREAM, channels=(1,))
    np.testing.assert_array_equal(alone[:, 0], both[:, 1])
    swapped = measure(zeros, (T, C_A), SEED, STREAM, channels=(1, 0))
    np.testing.assert_array_equal(swapped[:, ::-1], both)
    for bad in ((0, 0), (0,), (0, 1, 2), (0, 0.5), (0, 2**32)):
        with pytest.raises(ValueError, match="channel"):
            measure(zeros, (C_A, T), SEED, STREAM, channels=bad)


def test_a_sensor_is_specified_in_si() -> None:
    assert SensorSpec("C_A", "mol/m^3", 5.0).unit == "mol/m^3"
    assert SensorSpec("pH", "-", 0.1).unit == "-"
    for unit in ("mol/L", "L", "1/min", "min"):
        with pytest.raises(ValueError, match="which is not SI"):
            SensorSpec("x", unit, 1.0)
    for unit in ("degC", "", None, 5):
        with pytest.raises(ValueError, match="is not known"):
            SensorSpec("x", unit, 1.0)
