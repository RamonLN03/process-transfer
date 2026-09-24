"""The known specification of a plant, as an export states it.

What a model and its evaluation may know about a plant besides its readings (D-024,
``docs/data_contract.md``): the reactor volume, the density and heat capacity of the
contents, the heat of reaction and the four nominal inputs, in SI. They are read from the
manifest of an export opened with ``data.export.open_export_directory``, which verified it.
They are never read from a plant configuration file: such a file also holds the hidden
physics, and an export was written without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from process_transfer.cstr_variables import INPUT_NAMES, FloatArray
from process_transfer.data.export import Export
from process_transfer.validation import require_finite, require_positive

# The known parameters of an export, with the SI unit each must be stated in. The nominal
# inputs are listed in the order of INPUT_NAMES: q, C_Af, T_f, T_c.
PROPERTY_UNITS = {
    "reactor_volume": "m^3",
    "density": "kg/m^3",
    "heat_capacity": "J/(kg*K)",
    "reaction_enthalpy": "J/mol",
}
NOMINAL_INPUT_UNITS = {
    "nominal_feed_flow": "m^3/s",
    "nominal_feed_concentration": "mol/m^3",
    "nominal_feed_temperature": "K",
    "nominal_coolant_temperature": "K",
}


@dataclass(frozen=True)
class KnownPlant:
    """The known parameters of one plant, in SI."""

    plant_id: str
    volume: float  # V, m^3
    density: float  # rho, kg/m^3
    heat_capacity: float  # cp, J/(kg K)
    reaction_enthalpy: float  # dH, J/mol; negative for an exothermic reaction
    nominal_inputs: FloatArray  # q, C_Af, T_f, T_c: m^3/s, mol/m^3, K, K

    def __post_init__(self) -> None:
        for name in ("volume", "density", "heat_capacity"):
            object.__setattr__(self, name, require_positive(name, getattr(self, name)))
        object.__setattr__(
            self, "reaction_enthalpy", require_finite("reaction_enthalpy", self.reaction_enthalpy)
        )
        # finite factors can still give a product that overflows or underflows to zero
        require_positive("volume * density * heat_capacity", self.thermal_mass)
        require_finite("reaction_enthalpy / (density * heat_capacity)", self.heat_release_per_mole)
        nominal = np.array(self.nominal_inputs, dtype=np.float64)
        if nominal.shape != (len(INPUT_NAMES),):
            raise ValueError(
                f"nominal_inputs must hold {len(INPUT_NAMES)} values, {INPUT_NAMES}, got "
                f"shape {nominal.shape}"
            )
        for name, value in zip(INPUT_NAMES, nominal, strict=True):
            require_positive(f"nominal {name}", value)
        nominal.setflags(write=False)
        object.__setattr__(self, "nominal_inputs", nominal)

    @property
    def thermal_mass(self) -> float:
        """V rho cp, J/K."""
        return self.volume * self.density * self.heat_capacity

    @property
    def heat_release_per_mole(self) -> float:
        """-dH / (rho cp), K m^3/mol: the rise in temperature per mole of A that reacts in
        a cubic metre. Positive for an exothermic reaction."""
        return -self.reaction_enthalpy / (self.density * self.heat_capacity)


def read_known_plant(export: Export, plant_id: str) -> KnownPlant:
    """The known parameters of ``plant_id`` as the manifest of ``export`` states them.

    Every expected parameter must be there once, in its SI unit, and nothing else: a
    parameter that is missing, repeated, unknown or in another unit is refused rather than
    guessed or converted."""
    plants = export.manifest["plants"]
    if plant_id not in plants:  # type: ignore[operator]
        raise KeyError(f"the export {export.dataset_id!r} has no plant {plant_id!r}")
    entries = plants[plant_id]["known_parameters"]  # type: ignore[index]
    expected = {**PROPERTY_UNITS, **NOMINAL_INPUT_UNITS}
    names = [entry["parameter"] for entry in entries]
    if sorted(names) != sorted(expected):
        raise ValueError(
            f"the known parameters of {plant_id!r} in {export.dataset_id!r} are {sorted(names)}; "
            f"expected exactly {sorted(expected)}"
        )
    values: dict[str, float] = {}
    for entry in entries:
        name, unit, value = entry["parameter"], entry["unit"], entry["value"]
        if unit != expected[name]:
            raise ValueError(
                f"the known parameter {name!r} of {plant_id!r} is stated in {unit!r}, not in "
                f"{expected[name]!r}; it is not converted here"
            )
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"the known parameter {name!r} must be a number, got {value!r}")
        values[name] = require_finite(name, value)
    return KnownPlant(
        plant_id=plant_id,
        volume=values["reactor_volume"],
        density=values["density"],
        heat_capacity=values["heat_capacity"],
        reaction_enthalpy=values["reaction_enthalpy"],
        nominal_inputs=np.array([values[name] for name in NOMINAL_INPUT_UNITS], dtype=np.float64),
    )
