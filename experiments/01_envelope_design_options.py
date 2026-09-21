"""M0-E02: design options after the target left the temperature envelope (D-018).

M0-E01 found that, under the D-010 excitation amplitudes, the target plant peaks
above the documented 380 K limit on the all-plus corner of the input box. This
script evaluates the candidate remedies with the same repository functions, so
that the decision rests on reproducible numbers. It changes no configuration
file; every variation is applied in memory.

Outcome and limit of this experiment. The project owner chose option A at +-10 %
(A10, D-018). Every option here is judged on single steps from the nominal steady
state at 24 input cases. M0-E03 later showed that this is not enough: with chained
input changes A10 takes the target well above the limit. The conditions below are
left as they were run.

Run from the repository root:

    python experiments/01_envelope_design_options.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from process_transfer import __version__
from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.simulation import cstr_true
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.envelope import input_cases, simulate_envelope
from process_transfer.simulation.steady_state import SteadyState, find_steady_states

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PER_MINUTE = 60.0
TEMPERATURE_ENVELOPE = (335.0, 380.0)  # K


def load(name: str) -> tuple[TrueCSTRParameters, np.ndarray]:
    cfg = load_true_plant(CONFIGS / f"{name}_cstr.yaml")
    return TrueCSTRParameters.from_config(cfg), nominal_inputs(cfg.plant)


def steady_states_at(u: np.ndarray, p: TrueCSTRParameters) -> list[SteadyState]:
    return find_steady_states(
        lambda x: cstr_true.rhs(0.0, x, u, p),
        c_a_upper=u[1],
        temperature_range=(300.0, 460.0),
        n_grid=321,
    )


def deviations(u: np.ndarray, relative: float, kelvin: float) -> np.ndarray:
    """q and C_Af +- ``relative`` of nominal; T_f and T_c +- ``kelvin``."""
    return np.array([relative * u[0], relative * u[1], kelvin, kelvin])


def evaluate(tag: str, p: TrueCSTRParameters, u: np.ndarray, dev: np.ndarray) -> None:
    (nominal,) = steady_states_at(u, p)
    eig = nominal.eigenvalues[np.argmax(nominal.eigenvalues.imag)] * PER_MINUTE
    cases = input_cases(u, dev)

    multiple = 0
    least_stable = (-np.inf, "")
    for label, inputs in cases:
        found = steady_states_at(inputs, p)
        multiple += len(found) != 1
        worst = max(steady.max_real_part for steady in found) * PER_MINUTE
        if worst > least_stable[0]:
            least_stable = (worst, label)

    envelope = simulate_envelope(
        lambda x, inputs: cstr_true.rhs(0.0, x, inputs, p), nominal.state, cases
    )
    t_low = min(case.temperature_range[0] for case in envelope)
    t_high = max(case.temperature_range[1] for case in envelope)
    c_low = min(case.c_a_range[0] for case in envelope)
    c_high = max(case.c_a_range[1] for case in envelope)
    inside = TEMPERATURE_ENVELOPE[0] <= t_low and t_high <= TEMPERATURE_ENVELOPE[1]

    # Visibility of the kinetic mismatch over the excited concentration range (D-006):
    # r_true / r_model = 2 / (1 + K_sat C_A), equal to one at the nominal concentration.
    ratio_dilute = 2.0 / (1.0 + p.saturation_constant * c_low)
    ratio_rich = 2.0 / (1.0 + p.saturation_constant * c_high)

    print(
        f"  {tag:8s} nominal ({nominal.c_a:6.1f} mol/m^3, {nominal.temperature:6.2f} K) "
        f"eig {eig.real:+.3f}{eig.imag:+.3f}j 1/min | multiple steady states: {multiple} cases | "
        f"least stable {least_stable[0]:+.3f} 1/min"
    )
    print(
        f"           T in [{t_low:.2f}, {t_high:.2f}] K, inside envelope: {inside} | "
        f"C_A in [{c_low:.0f}, {c_high:.0f}] mol/m^3, r_true/r_model from "
        f"{ratio_dilute:.2f} to {ratio_rich:.2f}"
    )


def main() -> None:
    print(f"process_transfer {__version__}")
    source, u_source = load("source")
    target, u_target = load("target")

    print("\nBaseline: current design, D-010 amplitudes (q, C_Af +-20 %; T_f, T_c +-5 K)")
    evaluate("source", source, u_source, deviations(u_source, 0.20, 5.0))
    evaluate("target", target, u_target, deviations(u_target, 0.20, 5.0))

    for relative in (0.15, 0.10):
        print(f"\nOption A: current design, q and C_Af +-{relative:.0%}; T_f, T_c +-5 K")
        evaluate("source", source, u_source, deviations(u_source, relative, 5.0))
        evaluate("target", target, u_target, deviations(u_target, relative, 5.0))

    print("\nOption B: UA_ref x1.5 on both plants, T_c = 341.667 K (keeps the source at")
    print("          250 mol/m^3 and 350 K); D-010 amplitudes")
    for tag, p, u in (("source", source, u_source), ("target", target, u_target)):
        stronger = dataclasses.replace(p, ua_ref=1.5 * p.ua_ref)
        inputs = u.copy()
        inputs[3] = 350.0 - (350.0 - 337.5) / 1.5
        evaluate(tag, stronger, inputs, deviations(inputs, 0.20, 5.0))

    print("\nOption C: milder domain shift, target UA_ref = 0.9 x source; D-010 amplitudes")
    milder = dataclasses.replace(target, ua_ref=0.9 * source.ua_ref)
    evaluate("target", milder, u_target, deviations(u_target, 0.20, 5.0))


if __name__ == "__main__":
    main()
