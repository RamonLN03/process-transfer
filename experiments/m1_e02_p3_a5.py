"""M1-E02: the verification of P3 at A5 and of the lead, on the target.

Registered in ``docs/experiment_log.md`` (M1-E02) and amended on 2026-10-08 before this
script existed. It is the verification that section 6.2 of ``docs/m1_plan.md`` asks for
before any data at A5 exist, made as M0-E03b made it at A10. No reset anywhere: each
segment starts from the final state of the one before.

* Stability: the nominal steady state of the target as the generator verifies the starting
  point of every data set: exactly one, its balances closed, stable with the margin of
  D-017.
* Recovery: each of the 16 corners at A5 from the steady state, 120 s, then the rest of
  600 s; the distance to the steady state at the end.
* Carried state: all 256 ordered pairs of corners at A5, repeats included; the peak of the
  second excursion against the same excursion from the exact steady state.
* The lead: 60 s at the nominal inputs from the steady state, then one excursion and its
  rest, for each corner at A5 and at A10; 16 more at A10 without the lead give the peaks
  the lead is compared with.
* Every one of the 320 trajectories must be accepted by ``check_trajectory``: finite
  values, physical states, the envelope and closed balances.

The tolerances are those of P3 (``simulation/protocols.py``), fixed in M0-E03b. A
trajectory that cannot be simulated is a failed case with its cause; nothing is clipped,
replaced or retried. Reproducibility is checked by running the script twice from one clean
commit and comparing the two summaries with ``--compare``.

Run from the repository root, with PT_DATA_DIR outside the repository:

    python experiments/m1_e02_p3_a5.py
    python experiments/m1_e02_p3_a5.py --compare <run directory> <run directory>

Exit codes: 0 when every row passes (for ``--compare``, when the two runs agree and both
passed), 1 when a row fails, 2 when the run cannot complete. A summary saying which is
written in every case. Outputs go to PT_DATA_DIR/experiments/m1_e02/<run id>; they hold
the hidden parameters of the target and are diagnostics of the simulator. No data set of
observations is generated, and no noise is drawn.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import itertools
import json
import math
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from process_transfer import __version__
from process_transfer.config import load_true_plant
from process_transfer.cstr_variables import nominal_inputs
from process_transfer.data.paths import data_dir, output_dir, repository_root
from process_transfer.data.provenance import (
    copy_with_fingerprints,
    environment,
    git_state,
    new_directory,
    new_run_directory,
)
from process_transfer.generation.plants import NOMINAL_STABILITY_MARGIN, RESIDUAL_FRACTION
from process_transfer.simulation import cstr_true
from process_transfer.simulation.checks import (
    BALANCE_TOLERANCE,
    TEMPERATURE_ENVELOPE,
    check_trajectory,
)
from process_transfer.simulation.cstr_true import TrueCSTRParameters
from process_transfer.simulation.integration import (
    InputSegment,
    IntegrationError,
    Trajectory,
    simulate_piecewise,
)
from process_transfer.simulation.protocols import (
    A10_COOLANT_TEMPERATURE,
    A10_FEED_TEMPERATURE,
    A10_RELATIVE_FEED_CONCENTRATION,
    A10_RELATIVE_FLOW,
    P3_HOLD,
    P3_LEAD,
    P3_PEAK_AGREEMENT,
    P3_RECOVERY_TOLERANCE_CA,
    P3_RECOVERY_TOLERANCE_T,
    P3_REST,
    corner_label,
    corner_levels,
    p3_amplitudes,
    p3_segments,
)
from process_transfer.simulation.steady_state import find_steady_states

# =========================================================================== #
# Fixed by the registration and its amendment, before this script existed
# =========================================================================== #

PLANT = "target"
SAMPLE_PERIOD = 0.1  # s, the sampling of the truth, as in M0-E03b
LEAD = P3_LEAD  # 60 s
RECOVERY_TOLERANCE_T = P3_RECOVERY_TOLERANCE_T  # 0.005 K
RECOVERY_TOLERANCE_CA = P3_RECOVERY_TOLERANCE_CA  # 0.038 mol/m^3
PEAK_AGREEMENT = P3_PEAK_AGREEMENT  # 0.05 K
EXPECTED = {"recovery": 16, "pairs": 256, "lead": 32, "references_a10": 16}
# what differs between two runs by construction; everything else must be equal
VARIABLE_FIELDS = (("provenance", "run_id"), ("provenance", "started_utc"), ("timings_s",))


@dataclass(frozen=True)
class Plant:
    parameters: TrueCSTRParameters
    nominal_inputs: np.ndarray
    nominal_state: np.ndarray

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return cstr_true.rhs(0.0, x, u, self.parameters)


def plain(value: object) -> object:
    """JSON without NaN or infinity, which two runs could not compare as equal."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def refuse_a_data_directory_inside_the_repository() -> None:
    """The outputs hold the hidden parameters of the target and stay outside the repository
    (by default the data directory is <repository>/data)."""
    found, repository = data_dir().resolve(), repository_root().resolve()
    if found == repository or repository in found.parents:
        raise SystemExit(
            f"PT_DATA_DIR resolves to {found}, inside the repository {repository}; set it to a "
            "new directory outside the repository and outside OneDrive"
        )


# --------------------------------------------------------------------------- #
# The plant and its steady state
# --------------------------------------------------------------------------- #


def configuration_file() -> Path:
    return repository_root() / "configs" / f"{PLANT}_cstr.yaml"


def steady_state() -> tuple[dict[str, object], Plant | None]:
    """The nominal steady state, checked as ``generation.plants`` checks the starting point
    of a data set, with every quantity recorded rather than raised."""
    cfg = load_true_plant(configuration_file())
    p, u = TrueCSTRParameters.from_config(cfg), nominal_inputs(cfg.plant)
    found = find_steady_states(lambda x: cstr_true.rhs(0.0, x, u, p), c_a_upper=u[1])
    record: dict[str, object] = {"count": len(found), "nominal_inputs_SI": u.tolist()}
    if len(found) != 1:
        record.update(balances_closed=False, stable_with_margin=False)
        return record, None
    (steady,) = found
    dilution = u[0] / p.volume
    scale = np.array([dilution * u[1], dilution * u[2]])
    record.update(
        state_SI=steady.state.tolist(),
        residual=steady.residual.tolist(),
        residual_scale=scale.tolist(),
        balances_closed=bool(np.all(np.abs(steady.residual) <= RESIDUAL_FRACTION * scale)),
        eigenvalues_per_s=[[float(v.real), float(v.imag)] for v in steady.eigenvalues],
        max_real_part_per_s=steady.max_real_part,
        stable_with_margin=bool(steady.is_stable(NOMINAL_STABILITY_MARGIN)),
    )
    return record, Plant(p, u, steady.state)


# --------------------------------------------------------------------------- #
# One trajectory
# --------------------------------------------------------------------------- #


def continuous(trajectory: Trajectory, start: np.ndarray) -> bool:
    """The state starts at ``start`` and is carried from each segment into the next,
    exactly: what ``Trajectory`` enforces, checked again for the record."""
    segments = trajectory.segments
    if not np.array_equal(segments[0].states[0], start):
        return False
    return all(
        np.array_equal(after.states[0], before.states[-1]) and after.times[0] == before.times[-1]
        for before, after in itertools.pairwise(segments)
    )


def simulate(plant: Plant, segments: list[InputSegment]) -> tuple[Trajectory | None, dict]:
    """A trajectory and its record. A failure of the integration or of the structure of the
    trajectory is a failed case with its cause; any other error ends the run."""
    try:
        trajectory = simulate_piecewise(plant.f, plant.nominal_state, segments, SAMPLE_PERIOD)
    except (IntegrationError, ValueError) as error:
        return None, {
            "simulated": False,
            "failure": f"{type(error).__name__}: {error}",
            "accepted": False,
        }
    check = check_trajectory(trajectory, plant.parameters)
    return trajectory, {
        "simulated": True,
        "failure": None,
        "accepted": bool(check.accepted),
        "values_finite": bool(check.values_finite),
        "states_physical": bool(check.states_physical),
        "inside_envelope": bool(check.inside_envelope),
        "balances_close": bool(check.balances_close),
        "peak_K": check.peak_temperature,
        "refined_peak_K": check.refined_peak_temperature,
        "min_temperature_K": check.min_temperature,
        "c_a_range_mol_m3": list(check.c_a_range),
        "relative_mass_residual": check.relative_mass_residual,
        "relative_energy_residual": check.relative_energy_residual,
        "continuous": continuous(trajectory, plant.nominal_state),
    }


def peak_from(trajectory: Trajectory, first_segment: int) -> float:
    """The refined peak temperature of the trajectory from one segment on."""
    rest = dataclasses.replace(trajectory, segments=trajectory.segments[first_segment:])
    return float(rest.refined_peak(1)[0])


def gap(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else abs(a - b)


# --------------------------------------------------------------------------- #
# The cases
# --------------------------------------------------------------------------- #


def run_cases(plant: Plant) -> dict[str, list[dict]]:
    """The 320 trajectories of the registration, each with its record."""
    u, x_ss = plant.nominal_inputs, plant.nominal_state
    corners = corner_levels()

    recovery: list[dict] = []
    peak_a5: dict[str, float | None] = {}
    for corner in corners:
        label = corner_label(corner)
        trajectory, record = simulate(plant, p3_segments(u, corner[np.newaxis, :], "a5"))
        if trajectory is not None:
            residual = np.abs(trajectory.segments[-1].states[-1] - x_ss)
            record.update(residual_c_a_mol_m3=float(residual[0]), residual_t_K=float(residual[1]))
        peak_a5[label] = record.get("refined_peak_K")
        recovery.append({"corner": label, **record})

    references: list[dict] = []
    peak_a10: dict[str, float | None] = {}
    for corner in corners:
        label = corner_label(corner)
        _, record = simulate(plant, p3_segments(u, corner[np.newaxis, :], "a10"))
        peak_a10[label] = record.get("refined_peak_K")
        references.append({"corner": label, **record})

    pairs: list[dict] = []
    for first, second in itertools.product(corners, repeat=2):
        trajectory, record = simulate(plant, p3_segments(u, np.array([first, second]), "a5"))
        changed = None
        if trajectory is not None:
            changed = gap(peak_from(trajectory, 2), peak_a5[corner_label(second)])
        pairs.append(
            {
                "first": corner_label(first),
                "second": corner_label(second),
                "peak_change_K": changed,
                **record,
            }
        )

    lead: list[dict] = []
    for amplitude, without_lead in (("a5", peak_a5), ("a10", peak_a10)):
        for corner in corners:
            label = corner_label(corner)
            segments = p3_segments(u, corner[np.newaxis, :], amplitude, LEAD)
            trajectory, record = simulate(plant, segments)
            if trajectory is not None:
                end = np.abs(trajectory.segments[0].states[-1] - x_ss)
                record.update(
                    lead_at_nominal_inputs=bool(np.array_equal(trajectory.segments[0].inputs, u)),
                    end_of_lead_c_a_mol_m3=float(end[0]),
                    end_of_lead_t_K=float(end[1]),
                    peak_change_K=gap(peak_from(trajectory, 1), without_lead[label]),
                )
            lead.append({"amplitude": amplitude, "corner": label, **record})
    return {"recovery": recovery, "pairs": pairs, "lead": lead, "references_a10": references}


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


def at_most(value: object, limit: float) -> bool:
    """A number no larger than ``limit``. Not a number, a string or a missing value fails."""
    return isinstance(value, int | float) and not isinstance(value, bool) and value <= limit


def largest(cases: Sequence[dict], key: str) -> object:
    values = [c[key] for c in cases if isinstance(c.get(key), int | float)]
    return max(values) if values else None


def criteria(steady: dict, cases: dict[str, list[dict]]) -> dict[str, dict]:
    """Every row of the registration, from the records alone. A row passes only on its own
    evidence; none can make up for another."""
    counts = {name: len(cases.get(name, [])) for name in EXPECTED}
    complete = counts == EXPECTED
    everything = [case for name in EXPECTED for case in cases.get(name, [])]
    recovery, pairs, lead = cases.get("recovery", []), cases.get("pairs", []), cases.get("lead", [])

    def label(name: str, case: dict) -> str:
        if name == "pairs":
            return f"pair {case['first']} -> {case['second']}"
        if name == "lead":
            return f"lead {case['amplitude']} {case['corner']}"
        return f"{name} {case['corner']}"

    not_accepted = [
        label(name, case)
        for name in EXPECTED
        for case in cases.get(name, [])
        if case.get("accepted") is not True
    ]
    rows: dict[str, dict] = {
        "complete": {"passed": complete, "counts": counts, "expected": EXPECTED},
        "stability": {
            "passed": steady.get("count") == 1
            and steady.get("balances_closed") is True
            and steady.get("stable_with_margin") is True,
            "count": steady.get("count"),
            "max_real_part_per_s": steady.get("max_real_part_per_s"),
            "margin_per_s": -NOMINAL_STABILITY_MARGIN,
        },
        "physical_acceptance": {
            "passed": complete and not not_accepted,
            "trajectories": len(everything),
            "not_accepted": not_accepted,
        },
        "recovery": {
            "passed": counts["recovery"] == EXPECTED["recovery"]
            and all(
                at_most(c.get("residual_t_K"), RECOVERY_TOLERANCE_T)
                and at_most(c.get("residual_c_a_mol_m3"), RECOVERY_TOLERANCE_CA)
                for c in recovery
            ),
            "max_residual_t_K": largest(recovery, "residual_t_K"),
            "max_residual_c_a_mol_m3": largest(recovery, "residual_c_a_mol_m3"),
        },
        "carried_state": {
            "passed": counts["pairs"] == EXPECTED["pairs"]
            and all(c.get("accepted") is True for c in pairs)
            and all(at_most(c.get("peak_change_K"), PEAK_AGREEMENT) for c in pairs),
            "pairs_not_accepted": sum(c.get("accepted") is not True for c in pairs),
            "max_peak_change_K": largest(pairs, "peak_change_K"),
        },
        "envelope": {
            "passed": complete and all(c.get("inside_envelope") is True for c in everything),
            "max_refined_peak_K": largest(everything, "refined_peak_K"),
            "min_temperature_K": (
                min(
                    c["min_temperature_K"]
                    for c in everything
                    if isinstance(c.get("min_temperature_K"), int | float)
                )
                if any(isinstance(c.get("min_temperature_K"), int | float) for c in everything)
                else None
            ),
            "limits_K": list(TEMPERATURE_ENVELOPE),
        },
        "lead": {
            "passed": counts["lead"] == EXPECTED["lead"]
            and all(
                c.get("accepted") is True
                and c.get("continuous") is True
                and c.get("lead_at_nominal_inputs") is True
                and at_most(c.get("end_of_lead_t_K"), RECOVERY_TOLERANCE_T)
                and at_most(c.get("end_of_lead_c_a_mol_m3"), RECOVERY_TOLERANCE_CA)
                and at_most(c.get("peak_change_K"), PEAK_AGREEMENT)
                for c in lead
            ),
            "max_end_of_lead_t_K": largest(lead, "end_of_lead_t_K"),
            "max_end_of_lead_c_a_mol_m3": largest(lead, "end_of_lead_c_a_mol_m3"),
            "max_peak_change_K": largest(lead, "peak_change_K"),
            "discontinuous": [label("lead", c) for c in lead if c.get("continuous") is not True],
        },
    }
    # the state is carried, never reset, in every trajectory, not only those with the lead
    rows["continuity"] = {
        "passed": complete and all(c.get("continuous") is True for c in everything),
        "discontinuous": [
            label(name, case)
            for name in EXPECTED
            for case in cases.get(name, [])
            if case.get("continuous") is not True
        ],
    }
    return rows


# --------------------------------------------------------------------------- #
# Provenance, the run, and the comparison of two runs
# --------------------------------------------------------------------------- #


def provenance(directory: Path, state: dict[str, object], nominal: np.ndarray) -> dict:
    defaults = inspect.signature(simulate_piecewise).parameters
    a10 = [
        A10_RELATIVE_FLOW,
        A10_RELATIVE_FEED_CONCENTRATION,
        A10_FEED_TEMPERATURE,
        A10_COOLANT_TEMPERATURE,
    ]
    return {
        "experiment": "M1-E02",
        "run_id": directory.name,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(["python", *sys.argv]),
        "package_version": __version__,
        "git": state,
        "configurations": copy_with_fingerprints([configuration_file()], directory / "configs"),
        "environment": environment(),
        "plant": PLANT,
        "protocol": {
            "amplitudes_relative": {
                "a10": a10,
                "a5": [0.5 * value for value in a10],
                "meaning": "q and C_Af relative to their nominal values; T_f and T_c in K",
            },
            "amplitudes_SI": {
                name: p3_amplitudes(nominal, name).tolist() for name in ("a10", "a5")
            },
            "lead_s": LEAD,
            "hold_s": P3_HOLD,
            "rest_s": P3_REST,
            "corners": [corner_label(c) for c in corner_levels()],
            "expected_trajectories": EXPECTED,
        },
        "tolerances": {
            "recovery_t_K": RECOVERY_TOLERANCE_T,
            "recovery_c_a_mol_m3": RECOVERY_TOLERANCE_CA,
            "peak_agreement_K": PEAK_AGREEMENT,
        },
        "criteria": {
            "acceptance": "process_transfer.simulation.checks.check_trajectory(...).accepted",
            "temperature_envelope_K": list(TEMPERATURE_ENVELOPE),
            "balance_tolerance": BALANCE_TOLERANCE,
            "steady_state": "generation.plants: one steady state, residual within "
            "RESIDUAL_FRACTION of the feed terms, stable with NOMINAL_STABILITY_MARGIN",
            "residual_fraction": RESIDUAL_FRACTION,
            "stability_margin_per_s": NOMINAL_STABILITY_MARGIN,
        },
        "integration": {
            "method": defaults["method"].default,
            "rtol": defaults["rtol"].default,
            "atol": defaults["atol"].default,
            "sample_period_s": SAMPLE_PERIOD,
            "restart": "one solver call per input segment; the state is never reset",
        },
        "data_generated": "none: no observations, no noise, nothing for models",
        "contains_hidden_parameters": True,
    }


def report(summary: dict) -> None:
    for name, row in summary.get("criteria", {}).items():
        shown = {k: v for k, v in row.items() if k != "passed" and not isinstance(v, list | dict)}
        print(f"  {'pass' if row['passed'] else 'FAIL'}  {name:20s} {shown}")
    print(f"M1-E02 {'passed' if summary.get('passed') else 'did NOT pass'}")


def main(
    arguments: Sequence[str] | None = None,
    cases: Callable[[Plant], dict[str, list[dict]]] | None = None,
) -> int:
    options = parse(arguments)
    refuse_a_data_directory_inside_the_repository()
    if options.compare:
        return compare(*options.compare)
    started = time.perf_counter()
    print(f"process_transfer {__version__}; M1-E02, P3 at A5 and the lead, on the {PLANT}")
    state = git_state()
    directory = new_run_directory("m1_e02", state)
    if not state["code_identified"]:
        print(f"  NOTE: the code of this run is not fully identified: {state['reason']}")
    summary: dict[str, object] = {"passed": False, "exit_code": 2, "error": None}
    code = 2
    try:
        steady, plant = steady_state()
        summary["provenance"] = provenance(
            directory, state, np.asarray(steady["nominal_inputs_SI"], dtype=np.float64)
        )
        summary["steady_state"] = steady
        found = (cases or run_cases)(plant) if plant is not None else {}
        summary["cases"] = found
        rows = criteria(steady, found)
        summary["criteria"] = rows
        summary["passed"] = all(row["passed"] is True for row in rows.values())
        code = 0 if summary["passed"] else 1
    except Exception:  # the run could not complete: recorded, never passed
        summary["error"] = traceback.format_exc()
        code = 2
    finally:
        summary["exit_code"] = code
        summary["timings_s"] = {"total": round(time.perf_counter() - started, 1)}
        (directory / "summary.json").write_text(
            json.dumps(plain(summary), indent=2, allow_nan=False), encoding="utf-8"
        )
    if summary["error"]:
        print(summary["error"], file=sys.stderr)
    report(summary)
    print(f"summary.json written to {directory}")
    return code


def without_variable_fields(summary: dict) -> dict:
    kept = json.loads(json.dumps(summary))
    for path in VARIABLE_FIELDS:
        node = kept
        for key in path[:-1]:
            node = node.get(key, {})
        node.pop(path[-1], None)
    return kept


def differences(a: object, b: object, where: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        found = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                found.append(f"{where}/{key}: present in one run only")
            else:
                found += differences(a[key], b[key], f"{where}/{key}")
        return found
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} items against {len(b)}"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, f"{where}[{i}]")
        ]
    return [] if a == b and type(a) is type(b) else [f"{where}: {a!r} against {b!r}"]


def compare(first: str, second: str) -> int:
    """The reproducibility of M1-E02: two runs from one clean commit, equal in everything
    but the fields that differ by construction."""
    runs = [
        json.loads((Path(d) / "summary.json").read_text(encoding="utf-8")) for d in (first, second)
    ]
    found = differences(*(without_variable_fields(run) for run in runs))
    gits = [run.get("provenance", {}).get("git", {}) for run in runs]
    same_clean_commit = all(g.get("code_identified") is True for g in gits) and gits[0].get(
        "commit"
    ) == gits[1].get("commit")
    both_passed = all(run.get("passed") is True for run in runs)
    agree = not found and same_clean_commit and both_passed
    result = {
        "runs": [str(Path(d).resolve()) for d in (first, second)],
        "left_out": ["/".join(path) for path in VARIABLE_FIELDS],
        "differences": found,
        "same_clean_commit": same_clean_commit,
        "both_passed": both_passed,
        "reproduced": agree,
    }
    directory = new_directory(output_dir("experiments", "m1_e02_comparisons"))
    (directory / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{len(found)} differences outside {result['left_out']}; same clean commit: "
        f"{same_clean_commit}; both passed: {both_passed}"
    )
    for line in found[:20]:
        print(f"  {line}")
    print(f"reproduced: {agree}; written to {directory}")
    return 0 if agree else 1


def parse(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare", nargs=2, metavar="RUN_DIRECTORY")
    return parser.parse_args(arguments)


if __name__ == "__main__":
    sys.exit(main())
