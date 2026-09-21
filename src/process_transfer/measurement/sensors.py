"""Sensors: additive Gaussian noise on values handed over by the caller.

The measurement component does not know the plant. It receives the noise-free values
to be measured, an instrument specification and the identity of a noise stream, and
it returns readings. It needs no rate law, no parameter and no state of the simulator.

The noise model is that of ``docs/decisions.md`` D-020: a reading is the true value
plus Gaussian noise of zero mean, independent from sample to sample and from sensor to
sensor. ``noise_std`` is the standard deviation of that noise. It is not a bound on
the error: about a third of the readings lie further than one standard deviation from
the true value. Readings are never clipped, truncated or otherwise corrected, to the
noise level or to any physical range. A concentration reading may in principle be
negative; the sensor said so, and it is left as it is.

Randomness. The noise of channel ``c`` comes from a generator built from
``numpy.random.SeedSequence(entropy=seed, spawn_key=(*stream, c))``. It is a pure
function of an explicit seed, of the stream that identifies the run, and of the
channel. No global generator is involved and none is shared with the excitation, so
drawing an excitation cannot shift the noise, nor the reverse. Every channel has its
own stream, so adding a sensor or changing the noise level of one leaves the noise of
the others untouched, and a longer run starts with the noise of a shorter one.

Two runs that share seed and stream replay the same noise, sample for sample. That is
what makes a run reproducible, and it is a mistake when the runs are meant to be
different: every run must have its own stream. The convention is
``stream = (plant index, run index)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from process_transfer.config import SensorsConfig
from process_transfer.cstr_variables import FloatArray
from process_transfer.validation import require_non_negative, require_positive


@dataclass(frozen=True)
class SensorSpec:
    """One sensor, in SI: the variable it measures and the standard deviation of its
    noise. Zero is a valid limit, an exact sensor."""

    variable: str
    unit: str
    noise_std: float

    def __post_init__(self) -> None:
        if not self.variable:
            raise ValueError("a sensor must name the variable it measures")
        object.__setattr__(
            self, "noise_std", require_non_negative(f"noise_std of {self.variable}", self.noise_std)
        )


@dataclass(frozen=True)
class MeasurementSpec:
    """The instruments of a plant: sampling period in seconds and one sensor per
    measured variable, in the order of the columns they produce."""

    sample_period: float
    sensors: tuple[SensorSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sample_period", require_positive("sample_period", self.sample_period)
        )
        sensors = tuple(self.sensors)
        variables = [sensor.variable for sensor in sensors]
        if len(sensors) == 0:
            raise ValueError("at least one sensor is required")
        if len(set(variables)) != len(variables):
            raise ValueError(f"every variable may have one sensor only, got {variables}")
        object.__setattr__(self, "sensors", sensors)

    @classmethod
    def from_config(cls, cfg: SensorsConfig) -> MeasurementSpec:
        """The specification of a configuration file, converted to SI."""
        return cls(
            sample_period=cfg.sampling_period.si,
            sensors=tuple(
                SensorSpec(sensor.variable, sensor.noise_std.si_unit, sensor.noise_std.si)
                for sensor in cfg.sensors
            ),
        )

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sensor.variable for sensor in self.sensors)

    @property
    def noise_std(self) -> tuple[float, ...]:
        return tuple(sensor.noise_std for sensor in self.sensors)


def _require_key(name: str, value: object) -> int:
    """A seed or a stream index: a plain non-negative integer, given explicitly."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an explicit non-negative integer, got {value!r}; a run is "
            "reproducible only if its randomness is named"
        )
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return int(value)


def channel_generator(seed: int, stream: Sequence[int], channel: int) -> np.random.Generator:
    """The generator of one noise channel: a function of its three arguments only."""
    key = (
        *(_require_key("every element of stream", element) for element in stream),
        _require_key("channel", channel),
    )
    sequence = np.random.SeedSequence(entropy=_require_key("seed", seed), spawn_key=key)
    return np.random.default_rng(sequence)


def measure(
    clean: FloatArray, sensors: Sequence[SensorSpec], seed: int, stream: Sequence[int]
) -> FloatArray:
    """Readings of ``clean``, one column per sensor: ``clean + noise_std * z`` with ``z``
    standard normal, independent between rows and between columns.

    ``clean`` has shape (n_samples, n_sensors) and must be finite. A sensor with
    ``noise_std = 0`` returns its column exactly, without drawing anything. The result
    is checked for finiteness, since a finite value plus finite noise can overflow.
    Nothing is clipped.
    """
    values = np.asarray(clean, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(sensors):
        raise ValueError(
            f"clean must have shape (n_samples, {len(sensors)}), one column per sensor, "
            f"got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("a sensor cannot measure a non-finite value")
    # checked here and not only where a generator is built, so that a wrong seed or
    # stream is refused even when every sensor is exact and nothing is drawn
    seed = _require_key("seed", seed)
    stream = tuple(_require_key("every element of stream", element) for element in stream)

    readings = values.copy()
    for channel, sensor in enumerate(sensors):
        if sensor.noise_std == 0.0:
            continue  # an exact sensor: the reading is the value, bit for bit
        generator = channel_generator(seed, stream, channel)
        # an overflow is resolved just below, as an error; numpy need not warn about it
        with np.errstate(over="ignore"):
            readings[:, channel] += sensor.noise_std * generator.standard_normal(len(values))
    if not np.all(np.isfinite(readings)):
        raise ValueError("a reading is not finite: the value plus its noise overflows")
    return readings
