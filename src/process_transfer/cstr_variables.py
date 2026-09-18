"""State and input definitions of the CSTR, shared by the simulator and the modeller.

Only observable quantities live here: the ordering and SI units of the measured
states and of the inputs. Nothing in this module is hidden physics, so both
``process_transfer.simulation`` and ``process_transfer.modeller`` may import it
without breaking the ground-truth boundary of ``AGENTS.md``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from process_transfer.config import PlantSpec

FloatArray = NDArray[np.float64]

# State vector x = [C_A, T]
STATE_NAMES: tuple[str, ...] = ("C_A", "T")
STATE_UNITS: tuple[str, ...] = ("mol/m^3", "K")

# Input vector u = [q, C_Af, T_f, T_c]
INPUT_NAMES: tuple[str, ...] = ("q", "C_Af", "T_f", "T_c")
INPUT_UNITS: tuple[str, ...] = ("m^3/s", "mol/m^3", "K", "K")


def nominal_inputs(plant: PlantSpec) -> FloatArray:
    """Nominal input vector u = [q, C_Af, T_f, T_c] in SI."""
    nominal = plant.nominal_inputs
    return np.array(
        [
            nominal.feed_flow.si,
            nominal.feed_concentration.si,
            nominal.feed_temperature.si,
            nominal.coolant_temperature.si,
        ],
        dtype=np.float64,
    )
