"""Physical checks of predictions that use nothing hidden (``docs/m1_plan.md``, sections
9.3 and 9.4).

Validity bounds. For any rate that is not negative and vanishes without A, and any wall
conductance that is not negative, the balances of a CSTR keep, at every instant t,

    C_A(t) >= 0
    C_A(t) <= max(C_A(0), the richest feed applied up to t)
    T(t)   >= min(T(0), the coldest of feed and coolant applied up to t)   exothermic

The first two hold because the rate vanishes at C_A = 0 and never feeds A; the third
because, below both streams, flow, reaction heat and exchange all warm the reactor. "Up to
t" means the inputs of the rows before the instant: the row at a scored tick carries inputs
that only act after it. The temperature bound is derived for an exothermic reaction, dH <
0, and holds for dH = 0 too; for an endothermic reaction it does not hold and is refused,
and no plant of this project is endothermic. The comparisons are exact, with no tolerance:
a prediction is flagged only when its value lies beyond the bound. The initial state is the
context mean, a noisy estimate; a prediction that starts below zero is flagged as such.

Implied terms. For a model with right-hand side f = (f_CA, f_T), the balances imply

    r_imp = (q/V) (C_Af - C_A) - f_CA                                      mol/(m^3 s)
    Q_imp = V rho cp [ (q/V) (T_f - T) + (-dH / (rho cp)) r_imp - f_T ]    W

the rate of the reaction and the heat that leaves through the wall. A rate must not be
negative. A conductance that is not negative carries heat from the hotter side to the
colder, so Q_imp (T - T_c) >= 0, and Q_imp = 0 where T = T_c. The sign of T - T_c is exact
in floating point: the difference of two floating-point numbers is zero only when they are
equal, and otherwise has the right sign. So T = T_c is recognised exactly, and nothing is
divided by T - T_c.

Q_imp and r_imp are differences of terms that can be much larger than they are, and
forming them rounds. Their sign is therefore judged only where they lie beyond the rounding
bound of the expressions that form them. That bound is not a tolerance chosen here: in a
sum of products, each product carries a relative error of at most gamma_n = n u / (1 - n u),
where u = 2^-53 is the unit roundoff and n the number of roundings on its way to the
result (Higham, Accuracy and Stability of Numerical Algorithms, section 3.1). Q_imp as
computed below has at most 12 roundings on any path and r_imp at most 4, so their errors are
at most gamma_12 and gamma_4 times the sum of the magnitudes of their terms, the traffic
through each balance. A point is counted as incompatible with any non-negative conductance
when |Q_imp| exceeds its bound and its sign differs from that of T - T_c, which includes
T = T_c with a heat flow beyond the bound; a heat flow within its bound is zero to the
resolution of the arithmetic, which any conductance of zero allows. The quotient
Q_imp / (T - T_c) is not computed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from process_transfer.cstr_variables import FloatArray
from process_transfer.evaluation.outcomes import PhysicalViolation
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import WindowData

UNIT_ROUNDOFF = 2.0**-53


def rounding_factor(roundings: int) -> float:
    """gamma_n = n u / (1 - n u), the bound on the relative error of a product that went
    through n roundings."""
    return roundings * UNIT_ROUNDOFF / (1.0 - roundings * UNIT_ROUNDOFF)


def validity_violations(
    data: WindowData, predicted: FloatArray, known: KnownPlant
) -> tuple[PhysicalViolation, ...]:
    """The scored instants at which ``predicted`` breaks a validity bound."""
    if known.reaction_enthalpy > 0.0:
        raise ValueError(
            f"the temperature bound is derived for an exothermic reaction; plant "
            f"{known.plant_id!r} has dH = {known.reaction_enthalpy!r} J/mol"
        )
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != data.scored.shape or not np.all(np.isfinite(predicted)):
        raise ValueError(
            f"the prediction of window {data.key} must be finite, with shape {data.scored.shape}"
        )
    initial = data.initial_state
    # Row i of data.inputs is the row of tick onset + i; the scored instant i, tick
    # onset + 1 + i, follows the inputs of rows onset ... onset + i.
    richest_feed = np.maximum.accumulate(data.inputs[:, 1])
    coldest_stream = np.minimum.accumulate(np.minimum(data.inputs[:, 2], data.inputs[:, 3]))
    upper_concentration = np.maximum(initial[0], richest_feed)
    lower_temperature = np.minimum(initial[1], coldest_stream)
    checks = (
        ("C_A below zero", predicted[:, 0] < 0.0, predicted[:, 0], np.zeros(len(predicted))),
        (
            "C_A above its initial value and every feed applied",
            predicted[:, 0] > upper_concentration,
            predicted[:, 0],
            upper_concentration,
        ),
        (
            "T below its initial value and every feed and coolant applied",
            predicted[:, 1] < lower_temperature,
            predicted[:, 1],
            lower_temperature,
        ),
    )
    violations = []
    ticks = data.window.scored_ticks
    for bound, broken, values, limits in checks:
        for position in np.flatnonzero(broken):
            violations.append(
                PhysicalViolation(
                    window=data.key,
                    tick=ticks[position],
                    bound=bound,
                    value=float(values[position]),
                    limit=float(limits[position]),
                )
            )
    return tuple(sorted(violations, key=lambda violation: violation.tick))


@dataclass(frozen=True)
class ImpliedTerms:
    """The rate and the heat flow through the wall that a right-hand side implies, at a
    list of points, with the rounding bound of each."""

    rate: FloatArray  # mol/(m^3 s)
    rate_bound: FloatArray
    heat_flow: FloatArray  # W, leaving the reactor through the wall
    heat_flow_bound: FloatArray
    temperature_above_coolant: FloatArray  # sign of T - T_c: -1, 0 or +1, exact

    def __post_init__(self) -> None:
        """What ``check_implied_terms`` reads is checked here, where it enters: one finite
        value per point, bounds that are not negative, and signs of -1, 0 or +1. A term that
        is not finite has no sign to judge, and an infinite bound would call anything zero."""
        names = ("rate", "rate_bound", "heat_flow", "heat_flow_bound", "temperature_above_coolant")
        arrays = {name: np.array(getattr(self, name), dtype=np.float64) for name in names}
        points = len(arrays["rate"])
        for name, values in arrays.items():
            if values.ndim != 1 or len(values) != points:
                raise ValueError(
                    f"{name} has shape {values.shape}; the implied terms need one value per "
                    f"point, {points} points"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite; a term that is not has no sign to judge")
        for name in ("rate_bound", "heat_flow_bound"):
            if np.any(arrays[name] < 0.0):
                raise ValueError(f"{name} is a bound on a rounding error and must not be negative")
        if not np.all(np.isin(arrays["temperature_above_coolant"], (-1.0, 0.0, 1.0))):
            raise ValueError("temperature_above_coolant holds the sign of T - T_c: -1, 0 or +1")
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def implied_terms(
    derivatives: FloatArray, states: FloatArray, inputs: FloatArray, known: KnownPlant
) -> ImpliedTerms:
    """r_imp and Q_imp at points (x, u) with the derivatives f(x, u) there; one row each."""
    derivatives, states, inputs = (
        np.asarray(values, dtype=np.float64) for values in (derivatives, states, inputs)
    )
    n = len(states)
    if derivatives.shape != (n, 2) or states.shape != (n, 2) or inputs.shape != (n, 4):
        raise ValueError("derivatives, states and inputs need one row per point")
    for name, values in (("derivatives", derivatives), ("states", states), ("inputs", inputs)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
    # The order of the operations below is the one the rounding counts refer to. Finite
    # arguments can still overflow on the way, so the outcome is checked below instead of
    # trusted; the sign of T - T_c is exact even when its magnitude overflows.
    with np.errstate(over="ignore", invalid="ignore"):
        dilution = inputs[:, 0] / known.volume  # 1 rounding
        supply = dilution * (inputs[:, 1] - states[:, 0])  # 3 on this term
        rate = supply - derivatives[:, 0]  # 4 on the supply, 1 on f_CA
        heat_per_mole = -known.reaction_enthalpy / (known.density * known.heat_capacity)  # 2
        thermal_mass = known.volume * known.density * known.heat_capacity  # 2
        sensible = dilution * (inputs[:, 2] - states[:, 1])  # 3
        released = heat_per_mole * rate  # 7 on the supply, 4 on f_CA
        balance = sensible + released - derivatives[:, 1]  # 2 more on each term, 1 on f_T
        heat_flow = thermal_mass * balance  # 3 more on each term: 12, 9, 8 and 4 at most
        rate_bound = rounding_factor(4) * (np.abs(supply) + np.abs(derivatives[:, 0]))
        heat_flow_bound = (
            rounding_factor(12)
            * thermal_mass
            * (
                np.abs(sensible)
                + abs(heat_per_mole) * (np.abs(supply) + np.abs(derivatives[:, 0]))
                + np.abs(derivatives[:, 1])
            )
        )
        above_coolant = np.sign(states[:, 1] - inputs[:, 3])
    for name, values in (
        ("rate", rate),
        ("rounding bound of the rate", rate_bound),
        ("heat flow", heat_flow),
        ("rounding bound of the heat flow", heat_flow_bound),
    ):
        wrong = np.flatnonzero(~np.isfinite(values))
        if wrong.size:
            i = int(wrong[0])
            raise ValueError(
                f"the implied {name} is not representable in double precision at point {i} "
                f"(derivatives {derivatives[i].tolist()}, state {states[i].tolist()}, inputs "
                f"{inputs[i].tolist()}); its sign cannot be judged"
            )
    return ImpliedTerms(
        rate=rate,
        rate_bound=rate_bound,
        heat_flow=heat_flow,
        heat_flow_bound=heat_flow_bound,
        temperature_above_coolant=above_coolant,
    )


@dataclass(frozen=True)
class ImpliedTermCheck:
    points: int
    negative_rate: int  # r_imp below minus its rounding bound
    incompatible_heat_flow: int  # no non-negative conductance gives this heat flow
    at_coolant_temperature: int  # points with T = T_c exactly
    heat_flow_within_bound: int  # heat flows that are zero to the resolution of the arithmetic


def check_implied_terms(terms: ImpliedTerms) -> ImpliedTermCheck:
    beyond = np.abs(terms.heat_flow) > terms.heat_flow_bound
    incompatible = beyond & (np.sign(terms.heat_flow) != terms.temperature_above_coolant)
    return ImpliedTermCheck(
        points=len(terms.rate),
        negative_rate=int(np.sum(terms.rate < -terms.rate_bound)),
        incompatible_heat_flow=int(np.sum(incompatible)),
        at_coolant_temperature=int(np.sum(terms.temperature_above_coolant == 0)),
        heat_flow_within_bound=int(np.sum(~beyond)),
    )


def implied_terms_along(
    f: Callable[[FloatArray, FloatArray], FloatArray],
    data: WindowData,
    predicted: FloatArray,
    known: KnownPlant,
) -> ImpliedTerms:
    """The implied terms of a model along its own prediction of a window, at the scored
    instants. At each instant the right-hand side is evaluated with the inputs that led to
    it, those of the row before, so that every input used is one the window holds."""
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != data.scored.shape:
        raise ValueError(f"the prediction must have shape {data.scored.shape}")
    derivatives = np.array([f(x, u) for x, u in zip(predicted, data.inputs, strict=True)])
    return implied_terms(derivatives, predicted, data.inputs, known)
