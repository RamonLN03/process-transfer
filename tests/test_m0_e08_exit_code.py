"""The exit code of M0-E08, through the script's own ``main()``.

The audit of M0 left one limitation open (docs/m0_audit.md): the test of the exit code in
``tests/test_checks.py`` rebuilt the script's aggregation instead of running it, so a
disconnection between the criteria of ``experiments/08_oracle_conductance.py`` and its
exit code could pass unnoticed. Here the real ``main()`` runs, with its real anchoring,
simulation, acceptance checks and verdicts, on a short case: one P3 sequence of one
excursion per plant instead of three of ten. Faults are injected where they would arise:
the real ``check_trajectory`` is wrapped so that one verdict of A or of B is turned false,
the variant is mis-anchored, or the resolution asked of the integration is made
unreachable. Physics and tolerances are not changed, and outputs go to a temporary
PT_DATA_DIR.
"""

import dataclasses
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "08_oracle_conductance.py"


def load_script():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("m0_e08_oracle_conductance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m0_e08_oracle_conductance"] = module
    spec.loader.exec_module(module)
    return module


E08 = load_script()


@pytest.fixture
def short(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """One sequence of one excursion per plant, outputs under a temporary PT_DATA_DIR."""
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(E08, "SEEDS", (0,))
    monkeypatch.setattr(E08, "N_EXCURSIONS", 1)
    return tmp_path / "data" / "experiments" / "m0_e08"


def verdicts_of(directory: Path) -> dict[str, bool]:
    (run,) = list(directory.iterdir())
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))["verdicts"]


def faulty_checks(monkeypatch: pytest.MonkeyPatch, variant: str, **verdicts: bool) -> None:
    """The real ``check_trajectory``, with some of its verdicts turned false for A, the
    plant with UA(T), or for B, the variant with a constant UA (alpha = 0)."""
    real: Callable = E08.check_trajectory

    def check(trajectory, parameters):  # noqa: ANN001, ANN202
        found = real(trajectory, parameters)
        is_b = parameters.alpha == 0.0
        if (variant == "B") == is_b:
            return dataclasses.replace(found, **verdicts)
        return found

    monkeypatch.setattr(E08, "check_trajectory", check)


def test_a_valid_short_run_exits_with_zero(short: Path) -> None:
    assert E08.main() == 0
    assert verdicts_of(short) == {"H1_anchoring": True, "H2_resolved": True, "H3_valid": True}


@pytest.mark.parametrize(
    ("variant", "verdicts"),
    [
        ("A", {"balances_close": False}),
        ("B", {"balances_close": False}),
        ("A", {"inside_envelope": False}),
        ("A", {"states_physical": False}),
        ("B", {"states_physical": False}),
        ("B", {"values_finite": False}),
    ],
)
def test_a_pair_that_is_not_valid_fails_the_run(
    short: Path, monkeypatch: pytest.MonkeyPatch, variant: str, verdicts: dict[str, bool]
) -> None:
    faulty_checks(monkeypatch, variant, **verdicts)
    assert E08.main() == 1
    found = verdicts_of(short)
    assert found["H3_valid"] is False
    assert found["H1_anchoring"] is True and found["H2_resolved"] is True


def test_b_outside_the_envelope_alone_is_allowed_in_this_diagnostic(
    short: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Variant B probes how far the physics moves the plant; leaving the envelope is a
    result to report, not a reason to reject the comparison (D-027)."""
    faulty_checks(monkeypatch, "B", inside_envelope=False)
    assert E08.main() == 0
    assert verdicts_of(short)["H3_valid"] is True


def test_a_variant_that_is_not_anchored_fails_the_run(
    short: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = E08.constant_conductance_variant

    def mis_anchored(plant):  # noqa: ANN001, ANN202
        variant = real(plant)
        return dataclasses.replace(variant, ua_ref=variant.ua_ref * 1.001)

    monkeypatch.setattr(E08, "constant_conductance_variant", mis_anchored)
    assert E08.main() == 1
    assert verdicts_of(short)["H1_anchoring"] is False


def test_an_effect_the_integration_cannot_resolve_fails_the_run(
    short: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No integration error is zero, so asking it to be zero against the effect cannot
    be met: H2 fails, and with it the run."""
    monkeypatch.setattr(E08, "RESOLUTION_FRACTION", 0.0)
    assert E08.main() == 1
    assert verdicts_of(short)["H2_resolved"] is False
