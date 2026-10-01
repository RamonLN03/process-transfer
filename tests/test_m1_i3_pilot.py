"""The accounting of the pilot of I3 (experiments/m1_i3_pilot.py): every run, budget,
configuration and rate has one outcome, a training failure included (Codex's audit of
1a9fad4, F4). The script is imported and its functions run; nothing is trained here."""

import importlib.util
import sys
from pathlib import Path

from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.models.mechanistic import MechanisticParameters

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "m1_i3_pilot.py"


def load_pilot():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("m1_i3_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m1_i3_pilot"] = module
    spec.loader.exec_module(module)
    return module


PILOT = load_pilot()
RATES = {family: 1e-3 for family in PILOT.GRID}


def test_a_failed_start_leaves_its_hybrids_in_the_count_as_failures() -> None:
    """F4. Without MR_F for a run and budget the twelve hybrids of the grid were left out,
    and only the four trainings of BN remained. They are now outcomes, training failures
    whose reason names the failed start, and the outcomes cover what was expected."""
    failed = PILOT.starts_of(
        [
            {
                "kind": "MR_F",
                "run": "r",
                "budget": 2,
                "parameters": None,
                "failure": TrainingFailure("no start of MR_F converged: ..."),
            }
        ]
    )
    tasks, blocked = PILOT.trainings(PILOT.GRID, RATES, ["r"], [2], failed, 1, 1)
    assert [t["configuration"].family for t in tasks] == ["BN"] * 4
    assert len(blocked) == 12
    assert {b["family"] for b in blocked} == {"HK", "HU", "HKU"}
    for outcome in blocked:
        assert outcome["criterion"] is None and outcome["checkpoints"] == []
        assert "MR_F of r at 2 windows" in outcome["failure"].reason
        assert "no start of MR_F converged" in outcome["failure"].reason
    recorded = [PILOT.outcome_key(PILOT.blocked_outcome(t, "x")) for t in tasks] + [
        PILOT.outcome_key(b) for b in blocked
    ]
    expected = PILOT.expected_outcomes(PILOT.GRID, [RATES], ["r"], [2])
    assert len(recorded) == len(expected) == 16 and set(recorded) == expected


def test_a_start_that_was_never_fitted_is_a_failure_too() -> None:
    tasks, blocked = PILOT.trainings(PILOT.GRID, RATES, ["r"], [2], {}, 1, 1)
    assert len(tasks) == 4 and len(blocked) == 12
    assert all("not fitted" in b["failure"].reason for b in blocked)


def test_a_fitted_start_trains_every_configuration() -> None:
    start = MechanisticParameters.from_k_350(0.0177, 9000.0, 1330.0)
    starts = PILOT.starts_of(
        [{"kind": "MR_F", "run": "r", "budget": 2, "parameters": start, "failure": None}]
    )
    tasks, blocked = PILOT.trainings(PILOT.GRID, RATES, ["r"], [2], starts, 1, 1)
    assert len(tasks) == 16 and blocked == []
    assert all(t["start"] is start for t in tasks if t["configuration"].family != "BN")
