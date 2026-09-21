"""What is measured on a plant during one run, and nothing else.

``Observations`` is what modelling, transfer and evaluation code is given. Its fields
are a closed list, pinned by a test: instants, readings, known inputs and the metadata
an engineer would have, which is names, units, the sampling period and the instrument
specification. It holds no exact state, no measurement error, no parameter of the
plant and no seed. The seed of the sensor noise in particular stays out: whoever knows
it can regenerate the noise, subtract it and recover the exact states.

Convention of a row. Row k holds the instant t_k, the readings of the measured
variables at t_k, and the inputs applied from t_k on, until the next row at which they
differ (zero-order hold, right-continuous). At an instant where the inputs change, the
row therefore carries the new inputs, while the reading is of a state that has not
yet responded to them. The inputs are known without error (D-020).

The readings are what the sensors gave. They are not checked for positivity or for
closed balances, which are criteria for true states, and they are not corrected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from process_transfer.canonical import encode_list, encode_numbers, encode_string
from process_transfer.cstr_variables import FloatArray
from process_transfer.validation import require_non_negative, require_positive

CONTENT_ENCODING = "observations/v1"


def _read_only_copy(values: FloatArray) -> FloatArray:
    copy = np.array(values, dtype=np.float64)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class Observations:
    """Observations of one run of one plant, in SI."""

    plant: str
    run: str
    times: FloatArray  # s, strictly increasing
    measured: FloatArray  # shape (n_samples, n_measured): readings, noise included
    inputs: FloatArray  # shape (n_samples, n_inputs): known inputs, right-continuous
    measured_names: tuple[str, ...]
    measured_units: tuple[str, ...]
    input_names: tuple[str, ...]
    input_units: tuple[str, ...]
    sample_period: float  # s, nominal
    noise_std: tuple[float, ...]  # instrument specification, one per measured variable

    def __post_init__(self) -> None:
        times, measured, inputs = (
            _read_only_copy(values) for values in (self.times, self.measured, self.inputs)
        )
        if times.ndim != 1 or len(times) == 0:
            raise ValueError("times must be a non-empty vector")
        for name, values, labels, units in (
            ("measured", measured, self.measured_names, self.measured_units),
            ("inputs", inputs, self.input_names, self.input_units),
        ):
            if values.ndim != 2 or values.shape != (len(times), len(labels)):
                raise ValueError(
                    f"{name} must have shape ({len(times)}, {len(labels)}), one row per "
                    f"instant and one column per name, got {values.shape}"
                )
            if len(units) != len(labels):
                raise ValueError(f"{name}: {len(labels)} names against {len(units)} units")
        if len(self.noise_std) != len(self.measured_names):
            raise ValueError("noise_std must have one value per measured variable")
        for values in (times, measured, inputs):
            if not np.all(np.isfinite(values)):
                raise ValueError("observations must be finite; missing values are not part of M0")
        if np.any(times[1:] <= times[:-1]):
            raise ValueError("times must be strictly increasing, without duplicates")

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "measured", measured)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(
            self, "sample_period", require_positive("sample_period", self.sample_period)
        )
        object.__setattr__(
            self,
            "noise_std",
            tuple(require_non_negative("noise_std", value) for value in self.noise_std),
        )
        for name in ("measured_names", "measured_units", "input_names", "input_units"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        labels = (self.plant, self.run, *self.measured_names, *self.measured_units)
        labels += (*self.input_names, *self.input_units)
        if not all(isinstance(label, str) for label in labels):
            raise ValueError("plant, run, names and units must be strings")

    @property
    def n_samples(self) -> int:
        return len(self.times)

    def content_digest(self) -> str:
        """SHA-256 of the canonical encoding ``observations/v1`` of everything held here:
        labels, instrument specification and numbers, each with its name, its kind and
        its lengths (``docs/data_contract.md``). It identifies content independently of a
        file format. Equal digests mean equal content; across machines or library
        versions the last bits of a simulated state may differ, and the digest with them.
        """
        digest = hashlib.sha256()
        digest.update(f"process-transfer/{CONTENT_ENCODING}\n".encode("ascii"))
        fields: tuple[tuple[str, object], ...] = (
            ("plant", self.plant),
            ("run", self.run),
            ("measured_names", self.measured_names),
            ("measured_units", self.measured_units),
            ("input_names", self.input_names),
            ("input_units", self.input_units),
            ("sample_period", np.array([self.sample_period], dtype=np.float64)),
            ("noise_std", np.array(self.noise_std, dtype=np.float64)),
            ("times", self.times),
            ("measured", self.measured),
            ("inputs", self.inputs),
        )
        for name, value in fields:
            digest.update(encode_string(name))
            if isinstance(value, str):
                digest.update(encode_string(value))
            elif isinstance(value, tuple):
                digest.update(encode_list([encode_string(item) for item in value]))
            else:
                digest.update(encode_numbers(value))
        return digest.hexdigest()
