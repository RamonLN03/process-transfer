"""Configuration models for plants, hidden physics and the modeller's knowledge.

Quantities are written in engineering units in YAML and exposed in SI through
``Quantity.si``. The models are strict: unknown fields are errors, so a
true-plant file cannot be loaded as modeller knowledge by accident, and vice
versa. That strictness is the ground-truth boundary of ``AGENTS.md``
expressed in code: everything under ``TruePlantConfig.true_physics`` is
simulation truth and never reaches a model; ``TruePlantConfig.plant`` and
``ModellerConfig`` are what an engineer would know.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from process_transfer.units import to_si


class StrictModel(BaseModel):
    """Base for all configuration models: no unknown fields, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Quantity(StrictModel):
    """A physical quantity as written in configuration: a value and a unit string."""

    value: float
    unit: str

    @model_validator(mode="after")
    def _unit_must_be_known(self) -> Quantity:
        to_si(self.value, self.unit)  # raises UnknownUnitError for anything unknown
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

    value: float = Field(gt=0)


# --------------------------------------------------------------------------- #
# Known plant information (available to the modeller)
# --------------------------------------------------------------------------- #


class ReactorDesign(StrictModel):
    volume: PositiveQuantity


class PhysicalProperties(StrictModel):
    density: PositiveQuantity
    heat_capacity: PositiveQuantity
    reaction_enthalpy: Quantity  # negative for an exothermic reaction


class NominalInputs(StrictModel):
    """Nominal values of the four inputs: feed flow and composition, feed and coolant
    temperature. Excitation moves the inputs around these values."""

    feed_flow: PositiveQuantity
    feed_concentration: PositiveQuantity
    feed_temperature: PositiveQuantity
    coolant_temperature: PositiveQuantity


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
    k0: PositiveQuantity
    activation_temperature: PositiveQuantity  # E/R
    saturation_constant: PositiveQuantity  # K_sat


class TemperatureDependentConductance(StrictModel):
    """Plant-specific heat-transfer conductance UA(T) = UA_ref [1 + alpha (T - T_ref)]
    (docs/decisions.md D-007). The modeller sees a constant UA instead."""

    form: Literal["linear_in_temperature"]
    UA_ref: PositiveQuantity
    alpha: Quantity
    T_ref: PositiveQuantity


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
    k0: PositiveQuantity
    activation_temperature: PositiveQuantity


class ConstantConductance(StrictModel):
    form: Literal["constant"]
    UA: PositiveQuantity


class ModellerConfig(StrictModel):
    """The modeller's simplified physics and nominal parameter values. Contains
    nothing hidden."""

    kinetics: FirstOrderKinetics
    heat_transfer: ConstantConductance


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
