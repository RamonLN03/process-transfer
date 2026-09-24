"""Validity bounds and implied terms (docs/m1_plan.md, sections 9.3 and 9.4): on the
modeller's own equations, which satisfy them by construction, on right-hand sides built to
break them, and at T = T_c, where nothing may be divided."""

from fractions import Fraction

import numpy as np
import pytest

from m1_support import NOMINAL, corner, observations, p3_inputs
from process_transfer.evaluation.physics import (
    check_implied_terms,
    implied_terms,
    implied_terms_along,
    rounding_factor,
    validity_violations,
)
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.modeller import cstr_first_order
from process_transfer.modeller.cstr_first_order import ModellerCSTRParameters

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, NOMINAL)
MODELLER = ModellerCSTRParameters(
    volume=0.1,
    density=1000.0,
    heat_capacity=239.0,
    reaction_enthalpy=-50000.0,
    k0=7.2e10 / 60.0,
    activation_temperature=8750.0,
    ua=1.0e5 / 60.0,
)


def modeller_f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return cstr_first_order.rhs(0.0, x, u, MODELLER)


def one_window(corners=([1, 1, -1, -1],)):
    run = observations(p3_inputs(list(corners)), measured=np.tile([200.0, 355.0], (131, 1)))
    (data,) = window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)
    return data


def test_a_prediction_inside_the_bounds_has_no_violation() -> None:
    data = one_window()
    assert validity_violations(data, np.tile([200.0, 355.0], (110, 1)), KNOWN) == ()


def test_each_bound_is_reported_at_the_tick_where_it_breaks() -> None:
    data = one_window([[1, 1, -1, -1]])  # C_Af = 550 and T_f, T_c 5 K low for 20 rows
    predicted = np.tile([200.0, 355.0], (110, 1))
    predicted[3, 0] = -0.5  # tick 14
    predicted[40, 0] = 551.0  # above max(200, 550): the corner fed 550 before tick 51
    predicted[60, 1] = 332.0  # below min(355, 345, 332.5)
    violations = validity_violations(data, predicted, KNOWN)
    assert [(v.tick, v.bound.split()[0], v.value, v.limit) for v in violations] == [
        (14, "C_A", -0.5, 0.0),
        (51, "C_A", 551.0, 550.0),
        (71, "T", 332.0, 332.5),
    ]


def test_the_bounds_use_only_the_inputs_applied_before_each_instant() -> None:
    data = one_window([[-1, -1, 1, 1]])  # C_Af = 450 during the corner, 500 after
    predicted = np.tile([200.0, 355.0], (110, 1))
    predicted[:, 0] = 520.0  # above every feed applied up to any instant
    violations = validity_violations(data, predicted, KNOWN)
    assert len(violations) == 110 and violations[0].limit == 450.0
    assert violations[-1].limit == 500.0  # nominal feed from row 30 on
    assert {v.limit for v in violations} == {450.0, 500.0}
    assert [v.tick for v in violations if v.limit == 450.0] == list(range(11, 31))


def test_the_temperature_bound_is_refused_for_an_endothermic_reaction() -> None:
    endothermic = KnownPlant("target", 0.1, 1000.0, 239.0, 50000.0, NOMINAL)
    with pytest.raises(ValueError, match="exothermic"):
        validity_violations(one_window(), np.tile([200.0, 355.0], (110, 1)), endothermic)


def random_points(seed: int, n: int, temperatures: tuple[float, float]):
    rng = np.random.default_rng(seed)
    inputs = np.tile(NOMINAL, (n, 1)) + rng.uniform(-1.0, 1.0, (n, 4)) * np.array(
        [1.6e-4, 50.0, 5.0, 5.0]
    )
    states = np.column_stack([rng.uniform(1.0, 500.0, n), rng.uniform(*temperatures, n)])
    return states, inputs


def test_the_implied_terms_of_the_modeller_are_its_rate_and_its_heat_flow() -> None:
    # T at least 2.5 K above every coolant temperature drawn, so that UA (T - T_c) is not
    # itself a cancellation and a relative comparison means something
    states, inputs = random_points(0, 200, (345.0, 390.0))
    derivatives = np.array([modeller_f(x, u) for x, u in zip(states, inputs, strict=True)])
    terms = implied_terms(derivatives, states, inputs, KNOWN)
    rate = np.array([cstr_first_order.reaction_rate(c, t, MODELLER) for c, t in states])
    np.testing.assert_allclose(terms.rate, rate, rtol=1e-10, atol=0.0)
    heat_flow = MODELLER.ua * (states[:, 1] - inputs[:, 3])
    np.testing.assert_allclose(terms.heat_flow, heat_flow, rtol=1e-10, atol=0.0)
    check = check_implied_terms(terms)
    assert (check.points, check.negative_rate, check.incompatible_heat_flow) == (200, 0, 0)


def test_the_rounding_bounds_hold_against_exact_rational_arithmetic() -> None:
    """The computed terms differ from their exact values, for the same floating-point
    derivatives, states and inputs, by no more than their bounds; the points include large
    derivatives that cancel and temperatures at and near the coolant's."""
    states, inputs = random_points(2, 300, (330.0, 345.0))
    rng = np.random.default_rng(3)
    derivatives = rng.normal(0.0, 1.0, (300, 2)) * rng.choice([1e-6, 1.0, 1e3], (300, 2))
    states[:100, 1] = inputs[:100, 3]
    terms = implied_terms(derivatives, states, inputs, KNOWN)
    V, rho, cp, dh = (Fraction(v) for v in (0.1, 1000.0, 239.0, -50000.0))
    for i in range(300):
        q, c_af, t_f, _ = (Fraction(float(v)) for v in inputs[i])
        c_a, t = (Fraction(float(v)) for v in states[i])
        f_ca, f_t = (Fraction(float(v)) for v in derivatives[i])
        rate = q / V * (c_af - c_a) - f_ca
        heat_flow = V * rho * cp * (q / V * (t_f - t) + (-dh / (rho * cp)) * rate - f_t)
        assert abs(Fraction(float(terms.rate[i])) - rate) <= Fraction(float(terms.rate_bound[i]))
        assert abs(Fraction(float(terms.heat_flow[i])) - heat_flow) <= Fraction(
            float(terms.heat_flow_bound[i])
        )


def test_at_the_coolant_temperature_the_modeller_implies_no_heat_flow() -> None:
    states, inputs = random_points(1, 500, (330.0, 345.0))
    states[:, 1] = inputs[:, 3]  # T = T_c exactly
    derivatives = np.array([modeller_f(x, u) for x, u in zip(states, inputs, strict=True)])
    terms = implied_terms(derivatives, states, inputs, KNOWN)
    check = check_implied_terms(terms)
    assert check.at_coolant_temperature == 500
    assert check.incompatible_heat_flow == 0
    # what is left of the heat flow is rounding, within its bound
    assert np.all(np.abs(terms.heat_flow) <= terms.heat_flow_bound)


def test_a_reversed_conductance_and_a_negative_rate_are_counted() -> None:
    def reversed_wall(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        f = modeller_f(x, u)
        exchange = MODELLER.ua / KNOWN.thermal_mass * (x[1] - u[3])
        return np.array([f[0], f[1] + 2.0 * exchange])  # heat flows in from the cold side

    def producing(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        f = modeller_f(x, u)
        rate = cstr_first_order.reaction_rate(x[0], x[1], MODELLER)
        return np.array([f[0] + 2.0 * rate, f[1]])  # A is produced

    states = np.array([[200.0, 360.0], [150.0, 340.0], [300.0, 337.5]])
    inputs = np.tile(NOMINAL, (3, 1))
    for f, rate_sign, incompatible in ((reversed_wall, 0, 2), (producing, 3, None)):
        derivatives = np.array([f(x, u) for x, u in zip(states, inputs, strict=True)])
        check = check_implied_terms(implied_terms(derivatives, states, inputs, KNOWN))
        assert check.negative_rate == rate_sign
        if incompatible is not None:
            # T = T_c at the third point: no heat flow there, so only two are incompatible
            assert check.incompatible_heat_flow == incompatible
            assert check.at_coolant_temperature == 1


def test_a_heat_flow_at_the_coolant_temperature_is_incompatible_without_a_division() -> None:
    states = np.array([[200.0, 337.5]])
    inputs = np.array([NOMINAL])
    f = modeller_f(states[0], inputs[0])
    extra = np.array([[f[0], f[1] - 1e-3]])  # 1e-3 K/s more cooling: 23.9 W through the wall
    terms = implied_terms(extra, states, inputs, KNOWN)
    assert terms.heat_flow[0] == pytest.approx(23.9, rel=1e-6)
    assert check_implied_terms(terms).incompatible_heat_flow == 1


def test_implied_terms_computed_by_hand() -> None:
    known = KnownPlant("hand", 2.0, 1.0, 1.0, -4.0, np.array([1.0, 1.0, 1.0, 1.0]))
    states = np.array([[1.0, 3.0]])
    inputs = np.array([[4.0, 5.0, 7.0, 2.0]])
    derivatives = np.array([[1.0, 2.0]])
    terms = implied_terms(derivatives, states, inputs, known)
    # q/V = 2; r = 2 (5 - 1) - 1 = 7; Q = 2 [2 (7 - 3) + 4 * 7 - 2] = 68
    assert terms.rate.tolist() == [7.0] and terms.heat_flow.tolist() == [68.0]
    assert terms.temperature_above_coolant.tolist() == [1.0]
    assert terms.rate_bound[0] == rounding_factor(4) * (8.0 + 1.0)
    assert terms.heat_flow_bound[0] == pytest.approx(rounding_factor(12) * 2.0 * 46.0, rel=1e-15)


def test_implied_terms_along_a_prediction_use_the_inputs_held_by_the_window() -> None:
    data = one_window()
    predicted = np.tile([200.0, 355.0], (110, 1))
    seen = []

    def recording(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        seen.append(u.copy())
        return modeller_f(x, u)

    terms = implied_terms_along(recording, data, predicted, KNOWN)
    np.testing.assert_array_equal(np.array(seen), data.inputs)
    assert check_implied_terms(terms).incompatible_heat_flow == 0
    np.testing.assert_array_equal(np.array(seen)[0], corner([1, 1, -1, -1]))


def test_non_finite_points_are_refused() -> None:
    with pytest.raises(ValueError, match="states must be finite"):
        implied_terms(np.zeros((1, 2)), np.array([[np.inf, 300.0]]), np.ones((1, 4)), KNOWN)
    with pytest.raises(ValueError, match="one row per point"):
        implied_terms(np.zeros((2, 2)), np.zeros((1, 2)), np.ones((1, 4)), KNOWN)
