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

Keys. A seed, every element of a stream and a channel are plain non-negative integers,
Python or numpy, given explicitly. Fractions, booleans, strings and negative numbers
are refused and never rounded or converted: ``0.9`` silently read as ``0`` would hand
one run the noise of another. Stream elements and channels must also be below 2**32.
numpy splits a larger key into several 32-bit words, and ``(2**32,)`` then becomes the
same words as ``(0, 1)``: two different keys, one noise. There is one rule,
``noise_key`` and ``noise_stream``, and every entry point uses it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from process_transfer.config import SensorsConfig
from process_transfer.cstr_variables import FloatArray
from process_transfer.units import UnknownUnitError, si_unit_of
from process_transfer.validation import require_non_negative, require_positive


@dataclass(frozen=True)
class SensorSpec:
    """One sensor, in SI: the variable it measures, the SI unit of its readings and of
    its noise, and the standard deviation of that noise. Zero is a valid limit, an exact
    sensor.

    ``unit`` must be an SI unit known to ``process_transfer.units``. A noise level in
    engineering units is refused, not converted: the number and its unit would disagree
    by a factor that nothing downstream can see. ``MeasurementSpec.from_config`` is the
    place where a configuration in engineering units is converted.
    """

    variable: str
    unit: str
    noise_std: float

    def __post_init__(self) -> None:
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError("a sensor must name the variable it measures")
        try:
            si_unit = si_unit_of(self.unit)
        except (UnknownUnitError, TypeError) as error:
            raise ValueError(
                f"the unit of the {self.variable} sensor is not known: {error}"
            ) from None
        if si_unit != self.unit:
            raise ValueError(
                f"the {self.variable} sensor is given in {self.unit!r}, which is not SI. Inside "
                f"the simulation and the measurement everything is SI, here {si_unit!r}. Convert "
                "the noise level, or load the specification with MeasurementSpec.from_config; "
                "nothing is converted or relabelled here"
            )
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


STREAM_KEY_LIMIT = 2**32  # a stream element or a channel must fit one 32-bit word


def noise_key(name: str, value: object, limit: int | None = None) -> int:
    """``value`` as a plain ``int`` if it is a valid noise key, ``ValueError`` otherwise.

    Valid: a non-negative Python or numpy integer, below ``limit`` when one is given.
    Nothing is rounded or converted: ``0.9``, ``False``, ``"0"`` and ``-1`` are refused.
    This is the only place where that is decided.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an explicit non-negative integer, got {value!r}; a run is "
            "reproducible only if its randomness is named"
        )
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    if limit is not None and value >= limit:
        raise ValueError(
            f"{name} must be below {limit}, got {value!r}: a larger key is split into "
            "several words and would collide with a different, longer key"
        )
    return int(value)


def noise_stream(stream: object) -> tuple[int, ...]:
    """``stream`` as a tuple of plain ``int``, every element a valid key below 2**32."""
    if isinstance(stream, (str, bytes)) or not isinstance(stream, Iterable):
        raise ValueError(f"stream must be a sequence of non-negative integers, got {stream!r}")
    return tuple(
        noise_key("every element of stream", element, STREAM_KEY_LIMIT) for element in stream
    )


def channel_generator(seed: int, stream: Sequence[int], channel: int) -> np.random.Generator:
    """The generator of one noise channel: a function of its three arguments only."""
    key = (*noise_stream(stream), noise_key("channel", channel, STREAM_KEY_LIMIT))
    sequence = np.random.SeedSequence(entropy=noise_key("seed", seed), spawn_key=key)
    return np.random.default_rng(sequence)


def measure(
    clean: FloatArray,
    sensors: Sequence[SensorSpec],
    seed: int,
    stream: Sequence[int],
    channels: Sequence[int] | None = None,
) -> FloatArray:
    """Readings of ``clean``, one column per sensor: ``clean + noise_std * z`` with ``z``
    standard normal, independent between rows and between columns.

    ``clean`` has shape (n_samples, n_sensors) and must be finite. A sensor with
    ``noise_std = 0`` returns its column exactly, without drawing anything. The result
    is checked for finiteness, since a finite value plus finite noise can overflow.
    Nothing is clipped.

    ``channels`` gives the noise channel of each sensor, distinct keys below 2**32. By
    default a sensor's channel is its position. A caller for whom a sensor has an
    identity of its own passes that instead, so that the noise of a variable does not
    depend on which other variables are measured or in what order: measured alone, T
    must not receive the noise that C_A has when both are measured.
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
    seed = noise_key("seed", seed)
    stream = noise_stream(stream)
    if channels is None:
        channels = range(len(sensors))
    keys = [noise_key("every channel", channel, STREAM_KEY_LIMIT) for channel in channels]
    if len(keys) != len(sensors) or len(set(keys)) != len(keys):
        raise ValueError(
            f"channels must be {len(sensors)} distinct keys, one per sensor, got "
            f"{list(channels)!r}; two sensors on one channel would share their noise"
        )

    readings = values.copy()
    for column, (channel, sensor) in enumerate(zip(keys, sensors, strict=True)):
        if sensor.noise_std == 0.0:
            continue  # an exact sensor: the reading is the value, bit for bit
        generator = channel_generator(seed, stream, channel)
        # an overflow is resolved just below, as an error; numpy need not warn about it
        with np.errstate(over="ignore"):
            readings[:, column] += sensor.noise_std * generator.standard_normal(len(values))
    if not np.all(np.isfinite(readings)):
        raise ValueError("a reading is not finite: the value plus its noise overflows")
    return readings
