"""Recovery of a planted correction by the hybrids (docs/m1_plan.md, section 13, the
criteria of I3; sections 7.3 and 7.4).

On data simulated by the modeller's model with a correction planted in it, and without the
truth, recovery is tested in two separate ways: the fitted hybrid reproduces the
trajectories, and the mechanism is recovered, either as the correction itself, with the
mechanistic parameters held at the values used to simulate, or, when parameters and
correction are fitted together, as the total rate (the total heat flow for HU). A joint fit
is not asked to recover the correction alone: its split with the parameters is not
identifiable.

The planted truth is written here, with the balances written out, and not with the code
under test. It is on the truth side: the training sees only the windows. The declared domain
of the states is that of the true states at the scored instants of the fitting windows,
with the inputs of those instants: where the data say something about the rate.

The planted factors are smooth and of the size of the target's own mismatch: the kinetic
factor runs over about 0.6 to 1.6 on the visited states, where the true rate against the
first-order rate of the target runs over 0.74 to 1.39 (``docs/m1_plan.md``, section 2).

The tolerances were set from a development run of these trainings on 2026-09-27: noise-free,
the rate was recovered to 0.4 % in root mean square and 5.7 % at worst for HK, the heat
flow to 0.04 % and 0.8 % for HU, and both to 0.35 % and 3 % for HKU. The tests ask 2 % and
10 %, and 0.2 sigma of the trajectories.
"""

import math

import numpy as np
import pytest

from m1_support import NOMINAL, modeller_run, random_corners
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.models import learned
from process_transfer.models.mechanistic import MechanisticParameters
from process_transfer.models.training import Configuration, TrainingSettings, train

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
TRUE = MechanisticParameters.from_k_350(0.0177, 9000.0, 1330.0)
CORNERS = random_corners(8, 31)
SETTINGS = TrainingSettings(learning_rate=3e-3, max_steps=2000, validation_every=200)


def kinetic_factor(c_a: float, temperature: float) -> float:
    return math.exp(-0.25 * (c_a - 190.0) / 100.0 + 0.01 * (temperature - 355.0))


def thermal_factor(temperature: float, coolant: float) -> float:
    return math.exp(0.02 * (temperature - 355.0) - 0.01 * (coolant - 337.5))


def true_rate(x: np.ndarray, kinetic: bool) -> float:
    c_a, temperature = x
    k = TRUE.k_350 * math.exp(-TRUE.activation_temperature * (1.0 / temperature - 1.0 / 350.0))
    return k * c_a * (kinetic_factor(c_a, temperature) if kinetic else 1.0)


def true_heat_flow(x: np.ndarray, u: np.ndarray, thermal: bool) -> float:
    temperature, coolant = x[1], u[3]
    factor = thermal_factor(temperature, coolant) if thermal else 1.0
    return TRUE.ua * factor * (temperature - coolant)


def planted(kinetic: bool, thermal: bool):  # noqa: ANN201
    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        d = u[0] / KNOWN.volume
        r = true_rate(x, kinetic)
        heat = true_heat_flow(x, u, thermal)
        return np.array(
            [
                d * (u[1] - x[0]) - r,
                d * (u[2] - x[1]) + KNOWN.heat_release_per_mole * r - heat / KNOWN.thermal_mass,
            ]
        )

    return f


def model_terms(model: learned.LearnedModel, x: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """The rate and the heat flow through the wall that the model's balances imply."""
    d = u[0] / KNOWN.volume
    f = model.rhs(x, u)
    rate = d * (u[1] - x[0]) - f[0]
    heat = KNOWN.thermal_mass * (d * (u[2] - x[1]) + KNOWN.heat_release_per_mole * rate - f[1])
    return rate, heat


def fitted(family: str, hold: bool, noise_seed: int | None = None):  # noqa: ANN201
    kinetic, thermal = family in ("HK", "HKU"), family in ("HU", "HKU")
    f = planted(kinetic, thermal)
    run = modeller_run(f, CORNERS, noise_seed=noise_seed)
    data = window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)
    fitting, validation = data[:6], data[6:]
    record = train(
        Configuration(family, (8,), 0.0),
        SETTINGS,
        fitting,
        validation,
        KNOWN,
        seed=3,
        start=TRUE,
        hold_mechanistic=hold,
    )
    assert record.failure is None, record.failure
    exact = modeller_run(f, CORNERS, noise_seed=None)
    truth = window_data(exact, find_windows(exact, NOMINAL, P3_LAYOUT).windows)[:6]
    states = np.concatenate([d.scored for d in truth])
    inputs = np.concatenate([d.inputs for d in truth])
    return record, record.model(KNOWN), states, inputs, kinetic, thermal


def relative_errors(found: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    rel = np.abs(found - true) / np.abs(true)
    return float(np.sqrt(np.mean(rel**2))), float(np.max(rel))


def test_hk_recovers_the_planted_kinetic_factor_with_the_parameters_held() -> None:
    record, model, states, inputs, _, _ = fitted("HK", hold=True)
    assert record.checkpoints[record.selected].fitting_loss < 0.2**2
    assert record.criterion < 0.2
    assert np.array_equal(model.parameters["theta"], TRUE.coordinates(None))
    scales = model.scales
    z = (states - scales.state_mean) / scales.state_std
    learned_log = learned.network(np, model.parameters["kinetic"], z)[:, 0]
    planted_log = np.log([kinetic_factor(c, t) for c, t in states])
    assert np.sqrt(np.mean((learned_log - planted_log) ** 2)) < 0.02
    assert np.max(np.abs(learned_log - planted_log)) < 0.1


def test_hk_fitted_jointly_recovers_the_total_rate() -> None:
    record, model, states, inputs, _, _ = fitted("HK", hold=False)
    assert record.checkpoints[record.selected].fitting_loss < 0.2**2
    assert record.criterion < 0.2
    rates = np.array([model_terms(model, x, u)[0] for x, u in zip(states, inputs, strict=True)])
    rms, largest = relative_errors(rates, np.array([true_rate(x, True) for x in states]))
    assert rms < 0.02 and largest < 0.10


@pytest.mark.parametrize("hold", [True, False])
def test_hu_recovers_the_total_heat_flow(hold) -> None:  # noqa: ANN001
    record, model, states, inputs, _, _ = fitted("HU", hold=hold)
    assert record.checkpoints[record.selected].fitting_loss < 0.2**2
    assert record.criterion < 0.2
    heat = np.array([model_terms(model, x, u)[1] for x, u in zip(states, inputs, strict=True)])
    true = np.array([true_heat_flow(x, u, True) for x, u in zip(states, inputs, strict=True)])
    rms, largest = relative_errors(heat, true)
    assert rms < 0.02 and largest < 0.10


def test_hku_fitted_jointly_recovers_the_total_rate_and_heat_flow() -> None:
    record, model, states, inputs, _, _ = fitted("HKU", hold=False)
    assert record.checkpoints[record.selected].fitting_loss < 0.2**2
    terms = np.array([model_terms(model, x, u) for x, u in zip(states, inputs, strict=True)])
    rate_rms, rate_largest = relative_errors(
        terms[:, 0], np.array([true_rate(x, True) for x in states])
    )
    heat_rms, heat_largest = relative_errors(
        terms[:, 1],
        np.array([true_heat_flow(x, u, True) for x, u in zip(states, inputs, strict=True)]),
    )
    assert rate_rms < 0.02 and rate_largest < 0.10
    assert heat_rms < 0.02 and heat_largest < 0.10


def test_hk_on_noisy_data_reaches_the_noise_and_recovers_the_total_rate() -> None:
    """With the noise of D-020 the trajectories can be reproduced only to the noise, J near
    one, and the rate is still recovered on the visited states."""
    record, model, states, inputs, _, _ = fitted("HK", hold=False, noise_seed=17)
    assert 0.8 < record.checkpoints[record.selected].fitting_loss < 1.2
    assert record.criterion < 1.2
    rates = np.array([model_terms(model, x, u)[0] for x, u in zip(states, inputs, strict=True)])
    rms, largest = relative_errors(rates, np.array([true_rate(x, True) for x in states]))
    assert rms < 0.02 and largest < 0.10
