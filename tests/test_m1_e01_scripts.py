"""The two scripts of M1-E01, run on synthetic data and on the configuration of the target,
never on the data the experiment reads: the design of part 1 is P3 as the protocol defines
it, the profile of part 2 finds the minimum of data simulated by the modeller's own model,
and the variants of the oracle are anchored at the nominal steady state. The scripts are
imported and their functions run; their logic is not copied here."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from m1_support import NOMINAL, SIGMA, modeller_run, observations, random_corners
from process_transfer.config import load_true_plant
from process_transfer.data import database
from process_transfer.data.export import export_dataset
from process_transfer.data.parquet_store import DatasetWriter, open_dataset
from process_transfer.data.records import RunRecord, known_plant
from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.models.fitting import (
    BUDGET_EXHAUSTED,
    CONVERGED,
    DEFAULT_FIT_SETTINGS,
    DEFAULT_STARTS,
    FitResult,
    FitSettings,
    StartRecord,
)
from process_transfer.models.identifiability import (
    CHI2_ONE_95,
    NoInformation,
    profile_interval,
    steady_state_parameters,
    window_information,
)
from process_transfer.models.mechanistic import (
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
)
from process_transfer.simulation.cstr_true import conductance, reaction_rate
from process_transfer.simulation.protocols import a10_amplitudes, corner_levels, p3_segments

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
MODELLER = modeller_values()
STATE = np.array([190.0, 355.0])


def load(name: str):  # noqa: ANN201
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def available():  # noqa: ANN201
    return load("m1_e01_identifiability")


@pytest.fixture(scope="module")
def oracle():  # noqa: ANN201
    return load("m1_e01_oracle")


# --------------------------------------------------------------------------- #
# Parts 1 and 2
# --------------------------------------------------------------------------- #


def test_the_design_of_part_one_is_p3_as_the_protocol_defines_it(available) -> None:  # noqa: ANN001
    assert np.array_equal(available.a10_amplitudes(NOMINAL), a10_amplitudes(NOMINAL))
    assert np.array_equal(np.array(available.CORNERS), corner_levels())
    for signs in corner_levels():
        inputs = available.design_inputs(NOMINAL, signs)
        hold, rest = p3_segments(NOMINAL, signs[np.newaxis, :])[:2]
        assert inputs.shape == (P3_LAYOUT.scored_readings, 4)
        assert np.array_equal(inputs[: P3_LAYOUT.hold], np.tile(hold.inputs, (20, 1)))
        assert np.array_equal(inputs[P3_LAYOUT.hold :], np.tile(rest.inputs, (90, 1)))


def test_the_numbers_of_windows_are_those_of_mr_and_mr_f_at_the_budgets(available) -> None:  # noqa: ANN001
    assert available.WINDOW_COUNTS == (1, 2, 4, 5, 8, 10, 16, 20, 32, 40)
    grid = available.GRID
    assert list(grid) == sorted(set(grid))
    assert grid[0] == 6000.0 and grid[-1] == 13000.0
    assert set(available.WIDE_GRID) <= set(grid) and set(available.FINE_GRID) <= set(grid)
    assert np.all(np.diff(available.FINE_GRID) == 25.0)


def test_the_draws_of_p3_are_reproducible_and_count_every_window(available) -> None:  # noqa: ANN001
    first = available.draws(5, np.random.default_rng(available.DRAW_SEED))
    again = available.draws(5, np.random.default_rng(available.DRAW_SEED))
    assert np.array_equal(first, again)
    assert first.shape == (available.DRAWS, 16) and np.all(first.sum(axis=1) == 5)


def p3_windows(true_activation: float, corners: int, seed: int, exact_context: bool = True):  # noqa: ANN201
    true = steady_state_parameters(KNOWN, STATE, true_activation)
    model = MechanisticModel("truth", KNOWN, true)
    run = modeller_run(model.rhs, random_corners(corners, seed), noise_seed=None)
    windows = find_windows(run, NOMINAL, P3_LAYOUT).windows
    measured = np.array(run.measured)
    if exact_context:
        for window in windows:
            measured[window.context_ticks.start : window.onset + 1] = run.measured[window.onset]
    return true, window_data(observations(run.inputs, measured=measured, run=run.run), windows)


def test_the_corners_of_a_run_are_recognised_as_corners_of_the_design(available) -> None:  # noqa: ANN001
    _, data = p3_windows(8750.0, 2, 3)
    assert available.corners_are_the_design({"r": data}, KNOWN)
    moved = dataclasses.replace(data[0], inputs=data[0].inputs + np.array([0.0, 1e-9, 0, 0]))
    assert not available.corners_are_the_design({"r": (moved,)}, KNOWN)
    counts = available.corner_counts(data, KNOWN)
    expected = np.zeros(16)
    for window in data:
        signs = tuple(int(v) for v in np.sign(window.inputs[0] - NOMINAL))
        expected[available.CORNERS.index(signs)] += 1
    assert np.array_equal(counts, expected) and counts.sum() == 2


def test_part_one_at_a_point_of_the_modellers_model(available, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(available, "WINDOW_COUNTS", (1, 4))
    monkeypatch.setattr(available, "DRAWS", 25)
    found = available.part_one(KNOWN, STATE, np.array(SIGMA), 8750.0, "test", True)
    assert found["point"]["stable"]
    assert np.all(np.abs(found["point"]["residual_over_feed_terms"]) <= 1e-12)
    assert found["resolution"]["resolved"]
    assert len(found["per_corner"]) == 16
    for n in (1, 4):
        errors = found["expected_design"][n]["standard_errors"]
        for name in available.NAMES:
            assert errors["exact"][name] < errors["context"][name] <= errors["sandwich"][name]
        draws = found["p3_draws"][n]
        assert draws["draws_without_covariance"] == 0 and draws["draws"] == 25
    one = found["expected_design"][1]["standard_errors"]["sandwich"]["E/R"]
    four = found["expected_design"][4]["standard_errors"]["sandwich"]["E/R"]
    assert four == pytest.approx(one / 2.0, rel=1e-12)


def test_the_profile_of_the_modellers_own_data_finds_its_activation(available, monkeypatch) -> None:  # noqa: ANN001
    """Noise-free data of the modeller's model at E/R = 9200 K: the free fit and the lowest
    point of the profile are there, the fit held at the estimate reproduces its objective,
    and the estimate is a point of the profile."""
    true, data = p3_windows(9200.0, 3, 5)
    monkeypatch.setattr(available, "GRID", (8700.0, 9200.0, 9700.0))
    monkeypatch.setattr(available, "_WINDOWS", {"r": data, available.POOLED: data})
    monkeypatch.setattr(available, "_CONTEXT", {"known": KNOWN, "modeller": MODELLER})
    profiles, free = available.part_two(KNOWN, 1, None, float(STATE[1]))
    for name in ("r", available.POOLED):
        entry = profiles[name]
        assert free[name].parameters.activation_temperature == pytest.approx(9200.0, abs=0.1)
        assert abs(entry["check_point"]["loss_difference"]) <= available.CHECK_POINT
        profile = entry["profile"]
        assert profile["lowest_grid_point"] == 9200.0
        assert not profile["lowest_at_an_end_of_the_grid"]
        assert profile["estimate_inside_the_grid"]
        assert sum(row["at_the_free_estimate"] for row in profile["rows"]) == 1
        assert len(profile["rows"]) == 4 and profile["failed_points"] == []
    differences = available.replicate_differences(profiles)
    assert differences == []  # one replicate and the pooled set: no pair of replicates


def test_the_profile_marks_a_minimum_at_the_end_of_the_grid(available, monkeypatch) -> None:  # noqa: ANN001
    _, data = p3_windows(9200.0, 2, 7)
    monkeypatch.setattr(available, "GRID", (7000.0, 7500.0, 8000.0))
    monkeypatch.setattr(available, "_WINDOWS", {"r": data})
    monkeypatch.setattr(available, "_CONTEXT", {"known": KNOWN, "modeller": MODELLER})
    profiles, _ = available.part_two(KNOWN, 1, None, float(STATE[1]))
    profile = profiles["r"]["profile"]
    assert profile["lowest_grid_point"] == 8000.0
    assert profile["lowest_at_an_end_of_the_grid"]
    assert not profile["estimate_inside_the_grid"]


# --------------------------------------------------------------------------- #
# Part 3, the oracle
# --------------------------------------------------------------------------- #


def test_the_variants_are_anchored_at_the_nominal_steady_state(oracle) -> None:  # noqa: ANN001
    plant = oracle.load_virtual_plant(oracle.PLANT_FILE)
    x0, p = plant.nominal_state, plant.parameters
    kinetic = oracle.kinetic_variant(p, x0)
    assert kinetic.saturation_constant == 0.0
    assert kinetic.activation_temperature == p.activation_temperature
    assert (kinetic.ua_ref, kinetic.alpha) == (p.ua_ref, p.alpha)
    assert reaction_rate(x0[0], x0[1], kinetic) == pytest.approx(
        reaction_rate(x0[0], x0[1], p), rel=1e-14
    )
    thermal = oracle.thermal_variant(p, x0)
    assert thermal.alpha == 0.0
    assert conductance(x0[1], thermal) == conductance(x0[1], p)
    for variant in (kinetic, thermal):
        assert oracle.anchoring(plant, variant)["closed"]
    # without saturation the kinetic variant is the plant itself
    plain = dataclasses.replace(p, saturation_constant=0.0)
    assert oracle.kinetic_variant(plain, x0) == plain


def test_the_scored_states_are_the_110_instants_after_the_onset(oracle) -> None:  # noqa: ANN001
    plant = oracle.load_virtual_plant(oracle.PLANT_FILE)
    (trajectory,) = oracle.simulate_windows(
        plant.f, plant.nominal_state, plant.nominal_inputs, oracle.TOLERANCE
    )[:1]
    scored = oracle.scored_states(trajectory)
    assert scored.shape == (110, 2)
    at_six = np.flatnonzero(np.isclose(trajectory.times, 6.0, rtol=0.0, atol=1e-9))
    assert np.array_equal(scored[0], trajectory.states[at_six[0]])
    assert trajectory.times[-1] == pytest.approx(660.0)
    same = oracle.differences([scored], [scored], np.array(SIGMA), ["w"])
    assert same["whole window"]["T"]["max_abs"] == 0.0


def test_the_pseudo_true_fit_of_the_modellers_own_model_is_its_parameters(oracle) -> None:  # noqa: ANN001
    """When the truth is the modeller's model itself, the oracle's fit on its noise-free
    corner windows recovers the parameters that simulated them."""
    true = steady_state_parameters(KNOWN, STATE, 9300.0)
    model = MechanisticModel("truth", KNOWN, true)
    trajectories = oracle.simulate_windows(model.rhs, STATE, NOMINAL, oracle.TIGHT_TOLERANCE)
    windows = oracle.truth_windows(
        [oracle.scored_states(t) for t in trajectories], STATE, NOMINAL, np.array(SIGMA)
    )
    assert all(np.allclose(w.initial_state, STATE, rtol=1e-15, atol=0.0) for w in windows)
    fit, found = oracle.pseudo_true(windows, KNOWN, FitSettings(starts=DEFAULT_STARTS[:1]))
    assert fit.objective < 1e-5
    assert fit.parameters.activation_temperature == pytest.approx(9300.0, rel=1e-5)
    assert fit.parameters.k_350 == pytest.approx(true.k_350, rel=1e-6)
    assert fit.parameters.ua == pytest.approx(true.ua, rel=1e-6)
    assert found["rms_in_sigmas"]["pooled"]["T"] < 1e-4
    steady = oracle.model_steady_states(fit.model(KNOWN), NOMINAL)
    assert len(steady) == 1 and np.allclose(steady[0], STATE, rtol=1e-6)


def test_the_oracle_differences_are_measured_in_sigmas_per_phase(oracle) -> None:  # noqa: ANN001
    base = [np.zeros((110, 2))]
    shifted = [np.zeros((110, 2))]
    shifted[0][:20, 1] = 1.0  # one kelvin during the excursion only
    found = oracle.differences(shifted, base, np.array(SIGMA), ["w"])
    assert found["excursion"]["T"]["max_abs"] == 2.0  # 1 K is two sigma_T
    assert found["return"]["T"]["max_abs"] == 0.0 and found["settled"]["T"]["rms"] == 0.0
    assert found["whole window"]["T"]["rms"] == pytest.approx(2.0 * np.sqrt(20 / 110))
    assert found["whole window"]["C_A"]["max_abs"] == 0.0


# --------------------------------------------------------------------------- #
# Both scripts from end to end, on data the experiment does not read
# --------------------------------------------------------------------------- #

ATTEMPT = {"generated_at_utc": "2026-09-27T00:00:00+00:00", "commit": "0" * 40}


def publish(dataset: str, mode: str, runs: list) -> None:  # noqa: ANN001
    plants = [known_plant(load_true_plant(CONFIGS / "target_cstr.yaml").plant)]
    with DatasetWriter(dataset, plants, ATTEMPT) as writer:
        for run in runs:
            writer.add_run(run, RunRecord(dataset, mode, "synthetic run for a test"))
        writer.publish()
    connection = database.connect()
    try:
        database.ingest_dataset(connection, open_dataset(dataset))
        assert export_dataset(connection, dataset, ATTEMPT).status == "created"
    finally:
        connection.close()


def test_parts_one_and_two_run_from_end_to_end_on_synthetic_exports(
    available,
    monkeypatch,
    tmp_path,  # noqa: ANN001
) -> None:
    """Exports named as the script expects, holding runs of the modeller's own model, not
    the data of M0: the script runs, checks what it must, and writes its summary and
    figures."""
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "pt-data"))
    true = steady_state_parameters(KNOWN, STATE, 9200.0)
    model = MechanisticModel("truth", KNOWN, true)
    p3_runs = [
        modeller_run(model.rhs, random_corners(3, seed), seed, run=f"target.p3.e{seed}.x3.n0")
        for seed in (0, 1)
    ]
    noise = np.random.default_rng(9).normal(0.0, 1.0, (121, 2)) * np.array(SIGMA)
    steady = observations(
        np.tile(NOMINAL, (121, 1)), measured=STATE + noise, run="target.steady.d720.n0"
    )
    publish("m0-e05", "p3", p3_runs)
    publish("m0-e06", "steady", [steady])
    monkeypatch.setattr(available, "STEADY_RUN", "target.steady.d720.n0")
    monkeypatch.setattr(available, "GRID", (8700.0, 9200.0, 9700.0))
    monkeypatch.setattr(available, "WINDOW_COUNTS", (1, 2))
    monkeypatch.setattr(available, "DRAWS", 20)
    monkeypatch.setattr(available, "_WINDOWS", {})
    monkeypatch.setattr(available, "_CONTEXT", {})
    exports = tmp_path / "pt-data" / "available" / "exports"
    monkeypatch.setattr(sys, "argv", ["m1_e01", "--exports", str(exports), "--workers", "1"])
    assert available.main() == 0
    (run,) = (tmp_path / "pt-data" / "experiments" / "m1_e01").iterdir()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert all(summary["checks"].values())
    assert set(summary["part_2"]) == {"target.p3.e0.x3.n0", "target.p3.e1.x3.n0", "pooled"}
    assert len(summary["replicate_differences"]) == 1
    rule = summary["noise_against_excitation"]["at the textbook point"]
    (pair,) = rule["pairs"]
    assert pair["in_combined_a_priori_standard_errors"] == pytest.approx(
        pair["difference_K"] / pair["combined_a_priori_standard_error_K"]
    )
    designs = summary["part_1"]["designs_of_the_runs_of_part_2"]
    assert sum(designs["pooled"]["corner_counts"]) == 6  # three windows in each of two runs
    assert summary["provenance"]["oracle"] is False
    assert summary["provenance"]["data_dir"] == str((tmp_path / "pt-data").resolve())
    assert sorted(path.name for path in run.glob("*.png")) == sorted(summary["figures"])


def test_the_oracle_runs_from_end_to_end_on_the_source_plant(oracle, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """The oracle's script on the source plant, which M1-E01 does not study, with one start:
    it checks what it must and writes its summary, marked as holding hidden parameters."""
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "pt-data"))
    monkeypatch.setattr(oracle, "PLANT_FILE", oracle.CONFIGS / "source_cstr.yaml")
    monkeypatch.setattr(oracle, "FIT_SETTINGS", FitSettings(starts=DEFAULT_STARTS[:1]))
    monkeypatch.setattr(sys, "argv", ["m1_e01_oracle"])
    assert oracle.main() == 0
    (run,) = (tmp_path / "pt-data" / "experiments" / "m1_e01_oracle").iterdir()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["contains_hidden_parameters"] is True
    assert summary["provenance"]["data_dir"] == str((tmp_path / "pt-data").resolve())
    assert all(summary["checks"].values())
    assert len(summary["mismatch_alone"]["K"]["sensor_grid_in_sigmas"]["by_window"]) == 16
    assert sorted(path.name for path in run.glob("*.png")) == sorted(summary["figures"])


# --------------------------------------------------------------------------- #
# Failures are records, as the registration says
# --------------------------------------------------------------------------- #


def with_corner_replaced(available, monkeypatch, position, replacement):  # noqa: ANN001, ANN201
    original = available.corner_blocks

    def replaced(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        blocks = original(*args, **kwargs)
        blocks[position] = replacement
        return blocks

    monkeypatch.setattr(available, "corner_blocks", replaced)


def test_a_corner_without_information_leaves_only_its_designs_without_covariance(
    available,
    monkeypatch,
    tmp_path,  # noqa: ANN001
) -> None:
    monkeypatch.setattr(available, "WINDOW_COUNTS", (1, 4))
    monkeypatch.setattr(available, "DRAWS", 10)
    with_corner_replaced(available, monkeypatch, 0, NoInformation("forced for a test"))
    avoids = np.zeros(16)
    avoids[1] = 2.0  # a run that never went to the first corner
    designs = {"avoids": avoids, "holds": np.ones(16)}
    found = available.part_one(KNOWN, STATE, np.array(SIGMA), 8750.0, "t", True, designs)
    assert found["failed_windows"] == [{"corner": "++++", "reason": "forced for a test"}]
    assert found["per_corner"]["++++"] == {"reason": "forced for a test"}
    assert "standard_errors" in found["per_corner"]["+++-"]
    assert found["resolution"]["corners_compared"] == 15
    assert not found["resolution"]["resolved"]
    assert "reason" in found["expected_design"][1] and "reason" in found["p3_draws"]
    assert "standard_errors" in found["designs_of_the_runs_of_part_2"]["avoids"]
    assert "reason" in found["designs_of_the_runs_of_part_2"]["holds"]
    assert available.figure_standard_errors(found, tmp_path) is None
    assert available.figure_corners(found, tmp_path).exists()


def test_a_rank_deficient_corner_is_reported_and_left_out_of_the_resolution(
    available,
    monkeypatch,  # noqa: ANN001
) -> None:
    monkeypatch.setattr(available, "WINDOW_COUNTS", (1,))
    point = steady_state_parameters(KNOWN, STATE, 8750.0)
    steady = window_information(
        MechanisticModel("m", KNOWN, point), STATE, np.tile(NOMINAL, (110, 1)), 6.0, SIGMA
    )
    with_corner_replaced(available, monkeypatch, 3, steady)
    found = available.part_one(KNOWN, STATE, np.array(SIGMA), 8750.0, "t", False)
    label = "".join("+" if s > 0 else "-" for s in available.CORNERS[3])
    assert found["per_corner"][label]["reason"] == "the design is rank deficient"
    assert found["resolution"]["corners_compared"] == 15 and not found["resolution"]["resolved"]
    assert "standard_errors" in found["expected_design"][1]


def fake_fit(objective: float | None, activation: float) -> FitResult:
    parameters = MechanisticParameters.from_k_350(0.0177, activation, 1330.0)
    outcome = CONVERGED if objective is not None else BUDGET_EXHAUSTED
    record = StartRecord(
        start=DEFAULT_STARTS[0],
        initial=parameters,
        outcome=outcome,
        message="",
        final=parameters,
        objective=2.0 if objective is None else objective,
        singular_values=(1.0, 1.0, 1.0),
        evaluations=1,
        jacobian_evaluations=1,
        seconds=0.0,
    )
    failure = None if objective is not None else TrainingFailure("no start converged")
    selected = 0 if objective is not None else None
    return FitResult("f", (), 990, DEFAULT_FIT_SETTINGS, parameters, (record,), selected, failure)


def test_a_profile_is_measured_from_its_free_fit_and_only_when_it_is_the_minimum(
    available,
    monkeypatch,  # noqa: ANN001
) -> None:
    monkeypatch.setattr(available, "GRID", (9000.0, 9500.0, 10000.0))
    points = {9000.0: fake_fit(2.4, 9000.0), 10000.0: fake_fit(2.45, 10000.0)}
    free = fake_fit(2.3, 9480.0)
    good = available.summarise_profile(
        free, {**points, 9500.0: fake_fit(2.301, 9500.0), 9480.0: fake_fit(2.3, 9480.0)}, 990, 355.0
    )
    assert good["free_fit_is_the_minimum"] and "nominal noise" in good["intervals"]
    below = available.summarise_profile(
        free, {**points, 9500.0: fake_fit(2.29, 9500.0), 9480.0: fake_fit(2.3, 9480.0)}, 990, 355.0
    )
    assert below["free_fit_is_the_minimum"] is False
    assert "not its minimum" in below["intervals"]["not computed"]
    failed = available.summarise_profile(
        fake_fit(None, 9480.0), {**points, 9500.0: fake_fit(2.3, 9500.0)}, 990, 355.0
    )
    assert failed["minimum_objective"] is None and failed["free_fit_is_the_minimum"] is None
    assert "no minimum" in failed["intervals"]["not computed"]
    assert all(row["increase"] is None for row in failed["rows"])
    assert failed["lowest_grid_point"] == 9500.0


def test_the_true_crossing_lies_between_the_interpolated_one_and_its_outer_limit(
    available,  # noqa: ANN001
) -> None:
    """On convex profiles, quadratic or steeper, sampled on the fine grid with the estimate
    among the points, the true half-width lies between the interpolated one and the outer
    limit, which is inside the bracketing point."""
    for centre, width, quartic in (
        (9530.0, 40.0, 0.0),
        (9512.3, 13.0, 0.0),
        (9488.0, 25.0, 0.0),
        (9530.0, 40.0, 3.0),
        (9471.7, 20.0, 10.0),
    ):
        points = sorted({*available.FINE_GRID, centre})

        def profile(a: float, centre: float = centre, width: float = width, q: float = quartic):
            z = (a - centre) / width
            return z**2 + q * z**4

        increases = [profile(a) for a in points]
        interval = profile_interval(points, increases, CHI2_ONE_95)
        # the true crossing, by bisection on the continuous profile
        lo, hi = 0.0, 500.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if profile(centre + mid) < CHI2_ONE_95 else (lo, mid)
        true = lo
        found = available.crossing_resolution(
            {"nominal noise": interval}, points, increases, centre
        )
        for side in ("low", "high"):
            entry = found["nominal noise"][side]
            assert entry["convex"]
            assert entry["half_width_K"] <= true + 1e-9
            assert true <= entry["outer_limit_K"] + 1e-9
            assert entry["outer_limit_K"] - entry["half_width_K"] <= entry["spacing_K"] + 1e-9
