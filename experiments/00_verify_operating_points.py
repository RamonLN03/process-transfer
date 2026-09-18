"""M0-E01: steady states, local stability and operating envelope of source and target.

Reproduces, with repository code, the preliminary numbers first quoted in
docs/assumptions.md, and extends them to every corner of the input box.
Prints a plain-text report; writes nothing.

Run from the repository root:

    python experiments/00_verify_operating_points.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from process_transfer import __version__
from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import INPUT_NAMES, nominal_inputs
from process_transfer.simulation import cstr_true
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.envelope import input_cases, simulate_envelope
from process_transfer.simulation.steady_state import SteadyState, find_steady_states

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PER_MINUTE = 60.0

# Planned excitation amplitudes (docs/decisions.md D-010)
RELATIVE_DEVIATION = {"q": 0.20, "C_Af": 0.20}
ABSOLUTE_DEVIATION = {"T_f": 5.0, "T_c": 5.0}  # K

# Verification criteria (docs/decisions.md D-009, D-017; docs/assumptions.md)
STABILITY_MARGIN_PER_MIN = 0.5
TEMPERATURE_ENVELOPE = (335.0, 380.0)  # K


def deviations_for(nominal: np.ndarray) -> np.ndarray:
    values = []
    for name, value in zip(INPUT_NAMES, nominal, strict=True):
        if name in RELATIVE_DEVIATION:
            values.append(RELATIVE_DEVIATION[name] * value)
        else:
            values.append(ABSOLUTE_DEVIATION[name])
    return np.array(values)


def steady_states_at(u: np.ndarray, p: TrueCSTRParameters, fine: bool) -> list[SteadyState]:
    if fine:
        return find_steady_states(lambda x: cstr_true.rhs(0.0, x, u, p), c_a_upper=u[1])
    return find_steady_states(
        lambda x: cstr_true.rhs(0.0, x, u, p),
        c_a_upper=u[1],
        temperature_range=(300.0, 460.0),
        n_grid=321,
    )


def report(plant_name: str) -> None:
    cfg = load_true_plant(CONFIGS / f"{plant_name}_cstr.yaml")
    p = TrueCSTRParameters.from_config(cfg)
    u_nominal = nominal_inputs(cfg.plant)

    print(f"\n=== {plant_name} ===")
    steady_states = steady_states_at(u_nominal, p, fine=True)
    print(f"nominal inputs: steady states with T in [280, 480] K: {len(steady_states)}")
    for steady in steady_states:
        eig = steady.eigenvalues * PER_MINUTE
        conversion = 100.0 * (1.0 - steady.c_a / u_nominal[1])
        stable = steady.is_stable(STABILITY_MARGIN_PER_MIN / PER_MINUTE)
        print(
            f"  C_A = {steady.c_a:.6f} mol/m^3 ({steady.c_a / 1000.0:.6f} mol/L), "
            f"T = {steady.temperature:.6f} K, conversion = {conversion:.3f} %"
        )
        print(f"  eigenvalues = {eig[0]:.6f}, {eig[1]:.6f} 1/min")
        print(f"  stable with margin {STABILITY_MARGIN_PER_MIN} 1/min: {stable}")
        print(f"  residual = {steady.residual}")
    if len(steady_states) != 1:
        print("  envelope skipped: the operating point is not unique")
        return

    cases = input_cases(u_nominal, deviations_for(u_nominal))

    print("local stability on the boundary of the input box:")
    least_stable: tuple[float, str] | None = None
    for label, u in cases:
        found = steady_states_at(u, p, fine=False)
        worst = max(steady.max_real_part for steady in found) * PER_MINUTE
        if len(found) != 1 or worst >= 0.0:
            print(f"  CHECK {label}: {len(found)} steady state(s), max Re = {worst:+.3f} 1/min")
        if least_stable is None or worst > least_stable[0]:
            least_stable = (worst, label)
    assert least_stable is not None
    print(f"  least stable case: {least_stable[1]}, max Re = {least_stable[0]:+.4f} 1/min")

    envelope = simulate_envelope(
        lambda x, u: cstr_true.rhs(0.0, x, u, p), steady_states[0].state, cases
    )
    print(f"step responses from the nominal steady state, {len(envelope)} cases, 40 min each:")
    for case in envelope:
        t_min, t_max = case.temperature_range
        c_min, c_max = case.c_a_range
        print(
            f"  {case.label:20s} T in [{t_min:7.2f}, {t_max:7.2f}] K   "
            f"C_A in [{c_min:6.1f}, {c_max:6.1f}] mol/m^3   final T = {case.final_state[1]:7.2f} K"
        )
    t_low = min(case.temperature_range[0] for case in envelope)
    t_high = max(case.temperature_range[1] for case in envelope)
    c_low = min(case.c_a_range[0] for case in envelope)
    c_high = max(case.c_a_range[1] for case in envelope)
    hottest = max(envelope, key=lambda case: case.temperature_range[1])
    coldest = min(envelope, key=lambda case: case.temperature_range[0])
    slow = [case.label for case in envelope if not case.settled]
    inside = TEMPERATURE_ENVELOPE[0] <= t_low and t_high <= TEMPERATURE_ENVELOPE[1]
    print(f"  overall T in [{t_low:.2f}, {t_high:.2f}] K")
    print(f"  overall C_A in [{c_low:.1f}, {c_high:.1f}] mol/m^3")
    print(f"  hottest case: {hottest.label}; coldest case: {coldest.label}")
    print(f"  not settled to 1e-6 within 40 min (slow decay): {slow}")
    print(f"  inside documented envelope {TEMPERATURE_ENVELOPE} K: {inside}")


def main() -> None:
    print(f"process_transfer {__version__}")
    for plant_name in ("source", "target"):
        report(plant_name)


if __name__ == "__main__":
    main()
