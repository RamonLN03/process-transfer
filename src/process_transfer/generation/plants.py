"""A virtual plant ready to be run, with its starting point verified and not assumed.

Every run starts at the nominal steady state of its plant. That state is computed from
the true model, so it is part of the truth: it is used here and never exported. Before a
plant is used, the conditions of D-009 and D-017 are checked again, on the code and the
configuration of this run: one steady state in the scanned range for the nominal inputs,
the balances closed there, and every eigenvalue of the local Jacobian to the left of the
stability margin. A count of one is a statement about the scanned range, as
``simulation/steady_state.py`` explains, not a proof of uniqueness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from process_transfer.config import PlantSpec, load_true_plant
from process_transfer.cstr_variables import FloatArray, nominal_inputs
from process_transfer.simulation import cstr_true
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.steady_state import find_steady_states

# D-017: at the nominal inputs every eigenvalue has real part below -0.5 1/min.
NOMINAL_STABILITY_MARGIN = 0.5 / 60.0  # 1/s
# The steady state is polished on the full system; what is left of the right-hand side
# there is compared with the scale of each balance, the feed terms q/V * C_Af and
# q/V * T_f, and must be negligible against them.
RESIDUAL_FRACTION = 1.0e-9


class StartingPointError(ValueError):
    """The nominal operating point of a plant is not what the design requires."""


@dataclass(frozen=True)
class VirtualPlant:
    """A true plant with its nominal inputs and its verified nominal steady state, in SI.

    ``spec`` is the known part of the configuration, which is all that the available
    side is ever given. The rest of this object is truth.
    """

    plant_id: str
    configuration: Path
    spec: PlantSpec
    parameters: TrueCSTRParameters
    nominal_inputs: FloatArray
    nominal_state: FloatArray
    eigenvalues: tuple[complex, ...]  # 1/s, at the nominal steady state
    residual: FloatArray  # right-hand side at the nominal steady state

    def f(self, x: FloatArray, u: FloatArray) -> FloatArray:
        return cstr_true.rhs(0.0, x, u, self.parameters)

    @property
    def max_real_part(self) -> float:
        return float(max(value.real for value in self.eigenvalues))


def load_virtual_plant(configuration: Path) -> VirtualPlant:
    """Build a plant from its configuration file and verify its starting point.

    Raises :class:`StartingPointError` when the nominal inputs do not give exactly one
    steady state in the scanned range, when the balances are not closed there, or when
    it is not stable with the margin of D-017.
    """
    configuration = Path(configuration)
    cfg = load_true_plant(configuration)
    parameters = TrueCSTRParameters.from_config(cfg)
    u = nominal_inputs(cfg.plant)
    found = find_steady_states(lambda x: cstr_true.rhs(0.0, x, u, parameters), c_a_upper=u[1])
    if len(found) != 1:
        raise StartingPointError(
            f"{cfg.plant.name}: {len(found)} steady states were found for the nominal inputs, "
            "not one. The runs of this project start at a unique nominal steady state (D-009)"
        )
    (steady,) = found

    dilution = u[0] / parameters.volume
    scale = np.array([dilution * u[1], dilution * u[2]])  # mol/(m^3 s), K/s
    if not np.all(np.abs(steady.residual) <= RESIDUAL_FRACTION * scale):
        raise StartingPointError(
            f"{cfg.plant.name}: the balances are not closed at the nominal steady state: "
            f"residual {steady.residual.tolist()} against feed terms {scale.tolist()}"
        )
    if not steady.is_stable(NOMINAL_STABILITY_MARGIN):
        raise StartingPointError(
            f"{cfg.plant.name}: the nominal steady state is not stable with the margin of "
            f"D-017: largest real part {steady.max_real_part * 60.0:.4f} 1/min, required below "
            f"{-NOMINAL_STABILITY_MARGIN * 60.0:.2f} 1/min"
        )
    return VirtualPlant(
        plant_id=cfg.plant.name,
        configuration=configuration,
        spec=cfg.plant,
        parameters=parameters,
        nominal_inputs=u,
        nominal_state=steady.state,
        eigenvalues=tuple(complex(value) for value in steady.eigenvalues),
        residual=steady.residual,
    )
