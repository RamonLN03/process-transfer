"""Configuration models for plants, hidden physics, the modeller's knowledge and sensors.

Quantities are written in engineering units in YAML and exposed in SI through
``Quantity.si``. Every field states the physical dimension it expects, so a
known unit of the wrong kind (a volume in kelvin) is rejected, and every value
must be finite. The models are strict: unknown fields are errors, so a
true-plant file cannot be loaded as modeller knowledge by accident, and vice
versa. That strictness is the ground-truth boundary of ``AGENTS.md``
expressed in code: everything under ``TruePlantConfig.true_physics`` is
simulation truth and never reaches a model; ``TruePlantConfig.plant``,
``ModellerConfig`` and ``SensorsConfig`` are what an engineer would know.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from process_transfer.units import dimension_of, to_si


class StrictModel(BaseModel):
    """Base for all configuration models: no unknown fields, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Quantity(StrictModel):
    """A physical quantity as written in configuration: a finite value and a unit.

    Subclasses set ``DIMENSION`` to the physical dimension the field requires;
    the base class accepts any known unit.
    """

    DIMENSION: ClassVar[str | None] = None

    value: float = Field(allow_inf_nan=False)
    unit: str

    @model_validator(mode="after")
    def _unit_must_be_known_and_of_the_right_dimension(self) -> Quantity:
        dimension = dimension_of(self.unit)  # raises UnknownUnitError for anything unknown
        expected = type(self).DIMENSION
        if expected is not None and dimension != expected:
            raise ValueError(
                f"unit {self.unit!r} measures {dimension}, but this field requires {expected}"
            )
        return self

    @model_validator(mode="after")
    def _si_value_must_be_representable(self) -> Quantity:
        """A finite value does not guarantee a finite SI value: the conversion factor can
        overflow a very large number to infinity or underflow a very small one to zero.
        The equations use the SI value, so that is the one checked."""
        si_value, si_unit = to_si(self.value, self.unit)
        if not math.isfinite(si_value):
            raise ValueError(
                f"{self.value!r} {self.unit} is not representable in SI ({si_unit}): "
                "the conversion overflows"
            )
        if self.value != 0.0 and si_value == 0.0:
            raise ValueError(
                f"{self.value!r} {self.unit} underflows to zero in SI ({si_unit}); "
                "a non-zero quantity must stay non-zero"
            )
        return self

    @property
    def si(self) -> float:
        """The value converted to SI."""
        return to_si(self.value, self.unit)[0]

    @property
    def si_unit(self) -> str:
        """The SI unit string of this quantity."""
        return to_si(self.value, self.unit)[1]


class PositiveQuantity(Quantity):
    """A quantity that is physically required to be strictly positive."""

    value: float = Field(gt=0, allow_inf_nan=False)


class Volume(PositiveQuantity):
    DIMENSION = "volume"


class VolumetricFlow(PositiveQuantity):
    DIMENSION = "volumetric_flow"


class Concentration(PositiveQuantity):
    DIMENSION = "concentration"


class Temperature(PositiveQuantity):
    """An absolute temperature, or a quantity such as E/R expressed in kelvin."""

    DIMENSION = "temperature"


class Density(PositiveQuantity):
    DIMENSION = "density"


class SpecificHeatCapacity(PositiveQuantity):
    DIMENSION = "specific_heat_capacity"


class MolarEnergy(Quantity):
    """Signed on purpose: negative for an exothermic reaction (docs/decisions.md D-015)."""

    DIMENSION = "molar_energy"


class RateConstant(PositiveQuantity):
    DIMENSION = "inverse_time"


class ThermalConductance(PositiveQuantity):
    DIMENSION = "thermal_conductance"


class InverseConcentration(PositiveQuantity):
    DIMENSION = "inverse_concentration"


class InverseTemperature(Quantity):
    """Signed: a slope per kelvin may be positive or negative."""

    DIMENSION = "inverse_temperature"


class Duration(PositiveQuantity):
    DIMENSION = "time"


class NonNegativeQuantity(Quantity):
    """A quantity that may be zero but not negative, such as a noise level. Zero is a
    valid limit (an exact sensor), not an error."""

    value: float = Field(ge=0, allow_inf_nan=False)


# --------------------------------------------------------------------------- #
# Known plant information (available to the modeller)
# --------------------------------------------------------------------------- #


class ReactorDesign(StrictModel):
    volume: Volume


class PhysicalProperties(StrictModel):
    density: Density
    heat_capacity: SpecificHeatCapacity
    reaction_enthalpy: MolarEnergy


class NominalInputs(StrictModel):
    """Nominal values of the four inputs: feed flow and composition, feed and coolant
    temperature. Excitation moves the inputs around these values."""

    feed_flow: VolumetricFlow
    feed_concentration: Concentration
    feed_temperature: Temperature
    coolant_temperature: Temperature


class PlantSpec(StrictModel):
    """What is known about a plant: identity, design, properties and nominal inputs."""

    name: str
    process_type: Literal["cstr"]
    design: ReactorDesign
    properties: PhysicalProperties
    nominal_inputs: NominalInputs


# --------------------------------------------------------------------------- #
# Hidden physics (simulation truth only)
# --------------------------------------------------------------------------- #


class SaturatingKinetics(StrictModel):
    """Saturating, Langmuir-Hinshelwood-like rate law shared by source and target:

        r = k0 exp(-E/(R T)) C_A / (1 + K_sat C_A)

    No specific mechanism is claimed; the form is chosen so that no re-fit of a
    first-order law can reproduce it (docs/decisions.md D-004, D-006).
    """

    form: Literal["saturating"]
    k0: RateConstant
    activation_temperature: Temperature  # E/R
    saturation_constant: InverseConcentration  # K_sat


class TemperatureDependentConductance(StrictModel):
    """Plant-specific heat-transfer conductance UA(T) = UA_ref [1 + alpha (T - T_ref)]
    (docs/decisions.md D-007). The modeller sees a constant UA instead."""

    form: Literal["linear_in_temperature"]
    UA_ref: ThermalConductance
    alpha: InverseTemperature
    T_ref: Temperature


class TruePhysics(StrictModel):
    kinetics: SaturatingKinetics
    heat_transfer: TemperatureDependentConductance


class TruePlantConfig(StrictModel):
    """Simulation truth: the known plant specification plus the hidden physics."""

    plant: PlantSpec
    true_physics: TruePhysics


# --------------------------------------------------------------------------- #
# The modeller's simplified physics (available knowledge)
# --------------------------------------------------------------------------- #


class FirstOrderKinetics(StrictModel):
    """First-order Arrhenius kinetics, r = k0 exp(-E/(R T)) C_A, with nominal values
    a modeller would start from and later re-estimate."""

    form: Literal["first_order"]
    k0: RateConstant
    activation_temperature: Temperature


class ConstantConductance(StrictModel):
    form: Literal["constant"]
    UA: ThermalConductance


class ModellerConfig(StrictModel):
    """The modeller's simplified physics and nominal parameter values. Contains
    nothing hidden."""

    kinetics: FirstOrderKinetics
    heat_transfer: ConstantConductance


# --------------------------------------------------------------------------- #
# Sensors (an instrument specification, available knowledge)
# --------------------------------------------------------------------------- #

# The noise of a sensor is expressed in the dimension of the variable it measures.
_MEASURED_DIMENSIONS = {"C_A": "concentration", "T": "temperature"}


class SensorConfig(StrictModel):
    """One sensor: the variable it measures and its noise.

    ``noise_std`` is the standard deviation of additive, zero-mean Gaussian noise,
    independent from one sample to the next and from one sensor to another. It is not
    a bound on the error: about a third of the readings lie further than one standard
    deviation from the true value. Zero describes an exact sensor.
    """

    variable: Literal["C_A", "T"]
    form: Literal["additive_gaussian"]
    noise_std: NonNegativeQuantity

    @model_validator(mode="after")
    def _noise_has_the_dimension_of_the_variable(self) -> SensorConfig:
        expected = _MEASURED_DIMENSIONS[self.variable]
        found = dimension_of(self.noise_std.unit)
        if found != expected:
            raise ValueError(
                f"the noise of the {self.variable} sensor must be a {expected}, "
                f"but {self.noise_std.unit!r} measures {found}"
            )
        return self


class SensorsConfig(StrictModel):
    """The instrument specification of a process (docs/decisions.md D-020).

    There is one such file for the CSTR, not one per plant: source and target carry
    the same instruments, so their noise cannot differ by accident. Nothing here is
    hidden physics. It is what a data sheet would say.
    """

    process_type: Literal["cstr"]
    sampling_period: Duration
    sensors: tuple[SensorConfig, ...]

    @model_validator(mode="after")
    def _each_variable_is_measured_once(self) -> SensorsConfig:
        variables = [sensor.variable for sensor in self.sensors]
        if len(variables) == 0:
            raise ValueError("at least one sensor is required")
        if len(set(variables)) != len(variables):
            raise ValueError(f"every variable may have one sensor only, got {variables}")
        return self


# --------------------------------------------------------------------------- #
# Definition of a data set to generate (generation side; it holds a private seed)
# --------------------------------------------------------------------------- #


class DatasetDefinitionConfig(StrictModel):
    """What a data set is made of: plants, instruments, protocol, excitation seeds and
    the realisation of the noise. It is read by the generator only.

    ``sensor_master_seed`` is private. It is recorded in the private provenance of the
    data set and never in the available branch, because with it the noise could be
    regenerated and subtracted. Every run is one plant under one excitation seed; both
    plants receive the same excitation sequences, and their noise is independent,
    because the noise stream of a run follows from its identity.
    """

    dataset_id: str
    description: str
    plants: tuple[str, ...]  # configuration files of the plants, relative to this file
    sensors: str  # configuration file of the instruments, relative to this file
    protocol: Literal["p3"]
    n_excursions: int = Field(ge=1, le=1000)
    excitation_seeds: tuple[int, ...]
    noise_realisation: int = Field(ge=0)
    sensor_master_seed: int = Field(ge=0)
    simulation_period: Duration

    @model_validator(mode="after")
    def _runs_must_be_distinct(self) -> DatasetDefinitionConfig:
        if len(self.plants) == 0 or len(set(self.plants)) != len(self.plants):
            raise ValueError(f"plants must be one or more distinct files, got {self.plants}")
        seeds = self.excitation_seeds
        if len(seeds) == 0 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise ValueError(
                f"excitation_seeds must be one or more distinct non-negative integers, got {seeds}"
            )
        return self


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a plain dictionary."""
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_true_plant(path: str | Path) -> TruePlantConfig:
    """Load a true-plant configuration (known specification plus hidden physics)."""
    return TruePlantConfig.model_validate(load_yaml(path))


def load_modeller(path: str | Path) -> ModellerConfig:
    """Load the modeller's simplified physics and nominal parameter values."""
    return ModellerConfig.model_validate(load_yaml(path))


def load_sensors(path: str | Path) -> SensorsConfig:
    """Load the instrument specification shared by the plants of a process."""
    return SensorsConfig.model_validate(load_yaml(path))


def load_dataset_definition(path: str | Path) -> DatasetDefinitionConfig:
    """Load the definition of a data set to generate."""
    return DatasetDefinitionConfig.model_validate(load_yaml(path))
