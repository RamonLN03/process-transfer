"""The verdict, the exit code and the comparison of M1-E02 (experiments/m1_e02_p3_a5.py).

The registered sweep is an experiment, not a test, and nothing here simulates P3 at A5:
the cases are synthetic records, built as the script builds them, with deliberate
failures. What is shown is that every row is judged on its own evidence, that a physical
failure cannot be hidden by other rows that pass, that a missing or unsimulated case
fails, and that only a run whose every row passes exits with 0.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "m1_e02_p3_a5.py"


def load_script():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("m1_e02_p3_a5", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m1_e02_p3_a5"] = module
    spec.loader.exec_module(module)
    return module


E02 = load_script()
CORNERS = [E02.corner_label(c) for c in E02.corner_levels()]
STEADY = {
    "count": 1,
    "balances_closed": True,
    "stable_with_margin": True,
    "max_real_part_per_s": -0.05,
}


def accepted(**extra: object) -> dict:
    return {
        "simulated": True,
        "failure": None,
        "accepted": True,
        "values_finite": True,
        "states_physical": True,
        "inside_envelope": True,
        "balances_close": True,
        "refined_peak_K": 370.0,
        "min_temperature_K": 340.0,
        "continuous": True,
        **extra,
    }


def passing() -> dict[str, list[dict]]:
    recovery = [
        {"corner": c, **accepted(residual_t_K=1e-3, residual_c_a_mol_m3=6e-3)} for c in CORNERS
    ]
    pairs = [
        {"first": a, "second": b, "peak_change_K": 1e-3, **accepted()}
        for a in CORNERS
        for b in CORNERS
    ]
    lead = [
        {
            "amplitude": amplitude,
            "corner": c,
            **accepted(
                lead_at_nominal_inputs=True,
                end_of_lead_t_K=1e-12,
                end_of_lead_c_a_mol_m3=1e-12,
                peak_change_K=1e-9,
            ),
        }
        for amplitude in ("a5", "a10")
        for c in CORNERS
    ]
    references = [{"corner": c, **accepted()} for c in CORNERS]
    return {"recovery": recovery, "pairs": pairs, "lead": lead, "references_a10": references}


def failed_rows(steady: dict, cases: dict) -> set[str]:
    return {name for name, row in E02.criteria(steady, cases).items() if row["passed"] is not True}


def physical_failure(case: dict) -> None:
    """A trajectory whose balances do not close, with every other number of it fine."""
    case.update(accepted=False, balances_close=False)


def test_every_row_passes_on_cases_that_satisfy_the_registration() -> None:
    rows = E02.criteria(STEADY, passing())
    assert set(rows) == {
        "complete",
        "stability",
        "physical_acceptance",
        "recovery",
        "carried_state",
        "envelope",
        "lead",
        "continuity",
    }
    assert all(row["passed"] is True for row in rows.values())
    assert rows["physical_acceptance"]["trajectories"] == 320


@pytest.mark.parametrize(
    ("group", "index", "also_failing"),
    [
        ("recovery", 3, set()),
        ("references_a10", 15, set()),
        ("pairs", 200, {"carried_state"}),
        ("lead", 20, {"lead"}),
    ],
)
def test_a_physical_failure_cannot_be_hidden_by_the_rows_that_pass(
    group: str, index: int, also_failing: set[str]
) -> None:
    cases = passing()
    physical_failure(cases[group][index])
    assert failed_rows(STEADY, cases) == {"physical_acceptance"} | also_failing
    assert E02.criteria(STEADY, cases)["physical_acceptance"]["not_accepted"]


def test_a_case_that_could_not_be_simulated_fails_its_rows() -> None:
    cases = passing()
    cases["recovery"][0] = {
        "corner": CORNERS[0],
        "simulated": False,
        "failure": "IntegrationError: the solver failed",
        "accepted": False,
    }
    assert {"physical_acceptance", "recovery", "envelope", "continuity"} <= failed_rows(
        STEADY, cases
    )


@pytest.mark.parametrize("group", ["recovery", "pairs", "lead", "references_a10"])
def test_a_missing_case_fails_the_run(group: str) -> None:
    cases = passing()
    cases[group].pop()
    assert {"complete", "physical_acceptance"} <= failed_rows(STEADY, cases)
    cases = passing()
    del cases[group]
    assert {"complete", "physical_acceptance"} <= failed_rows(STEADY, cases)


@pytest.mark.parametrize(
    ("group", "key", "value", "row"),
    [
        ("recovery", "residual_t_K", 0.0050001, "recovery"),
        ("recovery", "residual_c_a_mol_m3", 0.0381, "recovery"),
        ("recovery", "residual_t_K", "nan", "recovery"),
        ("recovery", "residual_t_K", None, "recovery"),
        ("pairs", "peak_change_K", 0.0500001, "carried_state"),
        ("pairs", "peak_change_K", "inf", "carried_state"),
        ("lead", "end_of_lead_t_K", 0.006, "lead"),
        ("lead", "end_of_lead_c_a_mol_m3", 0.04, "lead"),
        ("lead", "peak_change_K", 0.06, "lead"),
        ("lead", "lead_at_nominal_inputs", False, "lead"),
        ("pairs", "inside_envelope", False, "envelope"),
    ],
)
def test_each_tolerance_is_applied_and_not_a_number_fails(
    group: str, key: str, value: object, row: str
) -> None:
    cases = passing()
    cases[group][5][key] = value
    assert row in failed_rows(STEADY, cases)


def test_the_limits_themselves_pass() -> None:
    cases = passing()
    cases["recovery"][0].update(residual_t_K=0.005, residual_c_a_mol_m3=0.038)
    cases["pairs"][0]["peak_change_K"] = 0.05
    cases["lead"][0]["peak_change_K"] = 0.05
    assert failed_rows(STEADY, cases) == set()


def test_a_discontinuous_state_fails_the_lead_and_the_continuity() -> None:
    cases = passing()
    cases["lead"][7]["continuous"] = False
    assert failed_rows(STEADY, cases) == {"lead", "continuity"}
    cases = passing()
    cases["pairs"][7]["continuous"] = False
    assert failed_rows(STEADY, cases) == {"continuity"}


@pytest.mark.parametrize(
    "change",
    [{"count": 2}, {"count": 0}, {"balances_closed": False}, {"stable_with_margin": False}],
)
def test_a_steady_state_that_is_not_the_verified_one_fails_the_stability(change: dict) -> None:
    assert failed_rows({**STEADY, **change}, passing()) == {"stability"}


@pytest.fixture
def outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data" / "experiments" / "m1_e02"


def only_summary(directory: Path) -> dict:
    (run,) = list(directory.iterdir())
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def test_the_exit_code_is_zero_only_when_every_row_passes(outside: Path) -> None:
    assert E02.main([], cases=lambda plant: passing()) == 0
    summary = only_summary(outside)
    assert summary["passed"] is True and summary["exit_code"] == 0 and summary["error"] is None
    assert summary["steady_state"]["count"] == 1 and summary["steady_state"]["stable_with_margin"]
    assert summary["provenance"]["protocol"]["lead_s"] == 60.0
    assert summary["provenance"]["configurations"][0]["name"] == "target_cstr.yaml"


def test_a_failed_row_exits_with_one_and_says_so(outside: Path) -> None:
    cases = passing()
    physical_failure(cases["pairs"][9])
    assert E02.main([], cases=lambda plant: cases) == 1
    summary = only_summary(outside)
    assert summary["passed"] is False and summary["exit_code"] == 1
    assert summary["criteria"]["physical_acceptance"]["passed"] is False


def test_a_run_that_cannot_complete_exits_with_two_and_is_never_passed(outside: Path) -> None:
    def broken(plant: object) -> dict:
        raise RuntimeError("an error the script did not foresee")

    assert E02.main([], cases=broken) == 2
    summary = only_summary(outside)
    assert summary["passed"] is False and summary["exit_code"] == 2
    assert "an error the script did not foresee" in summary["error"]


def test_a_data_directory_inside_the_repository_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PT_DATA_DIR", "data")
    with pytest.raises(SystemExit, match="inside the repository"):
        E02.main([], cases=lambda plant: passing())


def summary_of_a_run(run_id: str, started: str, total: float) -> dict:
    return {
        "passed": True,
        "exit_code": 0,
        "error": None,
        "provenance": {
            "run_id": run_id,
            "started_utc": started,
            "git": {"commit": "abc", "code_identified": True},
        },
        "cases": {"recovery": [{"corner": "++++", "residual_t_K": 0.001}]},
        "timings_s": {"total": total},
    }


def write(directory: Path, summary: dict) -> str:
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return str(directory)


def test_two_runs_agree_when_only_the_variable_fields_differ(tmp_path: Path, outside: Path) -> None:
    a = write(tmp_path / "a", summary_of_a_run("run-a", "2026-10-08T10:00:00+00:00", 30.1))
    b = write(tmp_path / "b", summary_of_a_run("run-b", "2026-10-08T10:01:00+00:00", 29.7))
    assert E02.main(["--compare", a, b]) == 0
    (comparison,) = (outside.parent / "m1_e02_comparisons").iterdir()
    found = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
    assert found["reproduced"] is True and found["differences"] == []


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["cases"]["recovery"][0].update(residual_t_K=0.0010000000000000002),
        lambda s: s["provenance"]["git"].update(commit="abd"),
        lambda s: s["provenance"]["git"].update(code_identified=False),
        lambda s: s.update(passed=False),
        lambda s: s["provenance"].update(environment={"numpy": "other"}),
    ],
)
def test_any_other_difference_or_a_run_that_did_not_pass_is_not_a_reproduction(
    tmp_path: Path, outside: Path, change
) -> None:  # noqa: ANN001
    first = summary_of_a_run("run-a", "2026-10-08T10:00:00+00:00", 30.1)
    second = copy.deepcopy(first)
    change(second)
    assert E02.main(["--compare", write(tmp_path / "a", first), write(tmp_path / "b", second)]) == 1


def test_one_case_at_a10_is_simulated_checked_and_recorded() -> None:
    """At A10, the amplitude M0-E03b verified: a case with the lead, as the script runs it."""
    steady, plant = E02.steady_state()
    assert steady["count"] == 1 and steady["stable_with_margin"] and steady["balances_closed"]
    corner = E02.corner_levels()[:1]
    trajectory, record = E02.simulate(
        plant, E02.p3_segments(plant.nominal_inputs, corner, "a10", E02.LEAD)
    )
    assert record["simulated"] and record["accepted"] and record["continuous"]
    assert record["failure"] is None and trajectory is not None
    assert len(trajectory.segments) == 3


def test_a_case_that_cannot_be_simulated_is_recorded_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plant = E02.steady_state()

    def fails(*arguments: object, **options: object) -> None:
        raise E02.IntegrationError("the solver gave up")

    monkeypatch.setattr(E02, "simulate_piecewise", fails)
    trajectory, record = E02.simulate(
        plant, E02.p3_segments(plant.nominal_inputs, E02.corner_levels()[:1], "a10")
    )
    assert trajectory is None
    assert record == {
        "simulated": False,
        "failure": "IntegrationError: the solver gave up",
        "accepted": False,
    }
