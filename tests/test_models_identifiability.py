"""The information about the parameters of the modeller's model under a design of windows,
and the profile against E/R: the reusable parts of M1-E01 (docs/m1_plan.md, section 7.4).
The data are simulated with the modeller's own equations, on the truth side of the tests."""

import math

import numpy as np
import pytest

from m1_support import NOMINAL, SIGMA, corner, modeller_run, observations, random_corners
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.models.fitting import (
    CONVERGED,
    DEFAULT_FIT_SETTINGS,
    DEFAULT_STARTS,
    STARTS_WITH_FIXED_ACTIVATION,
    FitSettings,
    covariance,
    fit_mechanistic,
)
from process_transfer.models.identifiability import (
    CHI2_ONE_95,
    NoInformation,
    WindowInformation,
    design_covariance,
    profile_interval,
    profile_settings,
    steady_state_parameters,
    window_information,
)
from process_transfer.models.mechanistic import (
    MechanisticModel,
    MechanisticParameters,
    modeller_values,
)
from process_transfer.models.rollout import rollout

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
MODELLER = modeller_values()
STATE = np.array([190.0, 355.0])  # a steady state near the target's, mol/m^3 and K
PERIOD = 6.0


def corner_inputs(signs) -> np.ndarray:  # noqa: ANN001
    """The inputs of a P3 window: 20 rows at the corner, then 90 at the nominal inputs."""
    inputs = np.tile(NOMINAL, (110, 1))
    inputs[:20] = corner(signs)
    return inputs


def model_at(parameters: MechanisticParameters, fixed: float | None = None) -> MechanisticModel:
    return MechanisticModel("m", KNOWN, parameters, fixed)


def information(signs, parameters, fixed=None, **options) -> WindowInformation:  # noqa: ANN001
    found = window_information(
        model_at(parameters, fixed), STATE, corner_inputs(signs), PERIOD, SIGMA, **options
    )
    assert isinstance(found, WindowInformation), found
    return found


POINT = steady_state_parameters(KNOWN, STATE, 8750.0)
# The context information is S^T S minus a correction, so the context bound carries a
# relative error of about the machine epsilon times the square of the condition number of S,
# some 5e-12 for these windows (docs/numerical_robustness.md, fourteenth review). Comparisons
# that go through it are asked 1e-9, not 1e-12: a Linux runner gave 1.25e-12 on the same
# code that gave 5.7e-13 on Windows.
CONTEXT_ACCURACY = 1e-9
SIGNS = [(1, -1, 1, 1), (-1, 1, -1, -1), (1, 1, 1, -1)]


def test_the_point_of_evaluation_holds_the_observed_state_steady() -> None:
    for activation in (0.0, 8750.0, 12000.0):
        parameters = steady_state_parameters(KNOWN, STATE, activation)
        f = model_at(parameters).rhs(STATE, NOMINAL)
        dilution = NOMINAL[0] / KNOWN.volume
        feed = np.array([dilution * NOMINAL[1], dilution * NOMINAL[2]])
        assert np.all(np.abs(f) <= 1e-12 * feed), (activation, f)
        # the balances fix k(T) and UA whatever E/R is
        assert parameters.ua == pytest.approx(POINT.ua, rel=1e-14)
        k_at_t = parameters.k0 * math.exp(-parameters.activation_temperature / STATE[1])
        assert k_at_t == pytest.approx(POINT.k0 * math.exp(-8750.0 / STATE[1]), rel=1e-12)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ((0.0, 355.0), "between 0 and the nominal feed"),
        ((500.0, 355.0), "between 0 and the nominal feed"),
        ((190.0, 0.0), "divides by the temperature"),
        ((190.0, 337.5), "does not determine the conductance"),
        ((190.0, 330.0), "no positive conductance"),
        ((190.0, float("nan")), "two finite values"),
    ],
)
def test_a_state_the_model_cannot_hold_is_refused(state, message) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match=message):
        steady_state_parameters(KNOWN, np.array(state), 8750.0)


def test_an_activation_temperature_whose_rate_constant_overflows_is_refused() -> None:
    with pytest.raises(ValueError, match="overflows|must be finite"):
        steady_state_parameters(KNOWN, STATE, 1e6)


def normalised_sensitivities(signs, parameters) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    result = rollout(
        model_at(parameters),
        STATE,
        corner_inputs(signs),
        PERIOD,
        parameter_sensitivities=True,
        initial_state_sensitivities=True,
    )
    sigma = np.array(SIGMA)
    s = (result.parameter_sensitivities / sigma[None, :, None]).reshape(-1, 3)
    g = (result.initial_state_sensitivities / sigma[None, :, None]).reshape(-1, 2)
    return s, g


def test_the_context_information_is_the_generalised_least_squares_information() -> None:
    """Computed by Woodbury's identity, it equals S^T (I + G Sigma_0 G^T)^-1 S directly, and
    the Schur complement of theta in the joint information of theta and the initial state
    from the scored readings and the mean of ten context readings."""
    block = information(SIGNS[0], POINT)
    s, g = normalised_sensitivities(SIGNS[0], POINT)
    sigma_0 = np.diag(np.array(SIGMA) ** 2) / 10
    direct = s.T @ np.linalg.solve(np.eye(len(s)) + g @ sigma_0 @ g.T, s)
    joint = np.block([[s.T @ s, s.T @ g], [g.T @ s, g.T @ g + np.linalg.inv(sigma_0)]])
    schur = joint[:3, :3] - joint[:3, 3:] @ np.linalg.solve(joint[3:, 3:], joint[3:, :3])
    scale = np.max(np.abs(block.context))
    assert np.max(np.abs(block.context - direct)) <= 1e-10 * scale
    assert np.max(np.abs(block.context - schur)) <= 1e-10 * scale
    assert np.allclose(block.exact, s.T @ s, rtol=1e-12, atol=0.0)
    assert np.allclose(block.root.T @ block.root, block.exact, rtol=1e-10, atol=1e-10 * scale)


def loewner_gap(larger: np.ndarray, smaller: np.ndarray) -> float:
    """The smallest eigenvalue of larger - smaller, relative to the largest of larger."""
    return float(np.linalg.eigvalsh(larger - smaller)[0] / np.linalg.eigvalsh(larger)[-1])


def test_the_three_covariances_are_ordered() -> None:
    """exact^-1 <= context^-1 <= sandwich: the initial state known only through its context
    costs information, and the estimator that fixes it at the context mean costs more."""
    for n in (1, 3):
        design = design_covariance([information(signs, POINT) for signs in SIGNS[:n]])
        exact, bound, sandwich = design.exact_initial_state, design.context_bound, design.sandwich
        # the differences have rank two at most, so their smallest eigenvalue is zero up to
        # the accuracy of the context bound
        assert loewner_gap(bound, exact) >= -CONTEXT_ACCURACY
        assert loewner_gap(sandwich, bound) >= -CONTEXT_ACCURACY
        errors = {kind: design.standard_errors(kind) for kind in ("exact", "context", "sandwich")}
        for name in design.names:
            assert errors["exact"][name] < errors["context"][name] <= errors["sandwich"][name]


def test_with_many_context_readings_the_three_coincide() -> None:
    block = information(SIGNS[0], POINT, context_readings=10**12)
    design = design_covariance([block])
    for kind in ("context", "sandwich"):
        for name, value in design.standard_errors(kind).items():
            assert value == pytest.approx(design.standard_errors("exact")[name], rel=1e-6)


def test_the_design_agrees_with_the_covariance_of_a_fit_on_real_windows() -> None:
    """The a priori sandwich of the windows of a run, at the parameters that simulated it,
    is the covariance that fitting.covariance computes on those windows."""
    true = POINT
    run = modeller_run(model_at(true).rhs, random_corners(4, 11), noise_seed=None)
    data = window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)
    blocks = [
        window_information(model_at(true), d.initial_state, d.inputs, PERIOD, d.noise_std)
        for d in data
    ]
    design = design_covariance(blocks)
    reference = covariance(true, data, KNOWN)
    for kind, matrix in (
        ("exact", reference.exact_initial_state),
        ("sandwich", reference.sandwich),
    ):
        found = design._matrix(kind)
        assert np.allclose(found, matrix, rtol=1e-9, atol=0.0), kind
    assert np.allclose(design.singular_values, reference.singular_values, rtol=1e-10)


def test_blocks_add_and_a_weight_counts_windows() -> None:
    blocks = [information(signs, POINT) for signs in SIGNS[:2]]
    listed = design_covariance([blocks[0]] * 2 + [blocks[1]] * 3)
    weighted = design_covariance(blocks, weights=[2, 3])
    for kind in ("exact", "context", "sandwich"):
        assert np.allclose(
            listed._matrix(kind), weighted._matrix(kind), rtol=CONTEXT_ACCURACY, atol=0.0
        )
    one = design_covariance(blocks[:1])
    many = design_covariance(blocks[:1], weights=[40])
    for kind in ("exact", "context", "sandwich"):
        for name, value in many.standard_errors(kind).items():
            assert value == pytest.approx(
                one.standard_errors(kind)[name] / math.sqrt(40), rel=CONTEXT_ACCURACY
            )
    # a zero weight leaves a window out
    alone = design_covariance(blocks, weights=[0, 1])
    assert np.allclose(alone.sandwich, design_covariance(blocks[1:]).sandwich, rtol=1e-12, atol=0)


def test_a_window_at_steady_state_cannot_determine_three_parameters() -> None:
    """Section 7.3 of the plan: at the steady state the balances fix k(T) and UA, so a window
    that never leaves it cannot tell E/R from k_350. With E/R held fixed the two others are
    determined."""
    steady = np.tile(NOMINAL, (110, 1))
    free = window_information(model_at(POINT), STATE, steady, PERIOD, SIGMA)
    design = design_covariance([free])
    assert design.reason == "the design is rank deficient"
    assert design.sandwich is None and len(design.singular_values) == 3
    fixed = window_information(model_at(POINT, 8750.0), STATE, steady, PERIOD, SIGMA)
    held = design_covariance([fixed], fixed_activation=8750.0)
    assert held.reason is None and held.names == ("ln k_350", "ln UA")
    assert set(held.standard_errors("sandwich")) == {"ln k_350", "ln UA"}


def test_a_window_the_model_cannot_start_has_no_information() -> None:
    found = window_information(
        model_at(POINT), np.array([190.0, -1.0]), corner_inputs(SIGNS[0]), PERIOD, SIGMA
    )
    assert isinstance(found, NoInformation) and "initial state" in found.reason


@pytest.mark.parametrize(
    ("noise", "readings", "message"),
    [
        ((5.0, 0.0), 10, "noise levels"),
        ((5.0, float("inf")), 10, "noise levels"),
        ((5.0,), 10, "noise levels"),
        ((5.0, 0.5), 0, "context_readings"),
        ((5.0, 0.5), True, "context_readings"),
    ],
)
def test_invalid_arguments_of_a_window_are_refused(noise, readings, message) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match=message):
        window_information(model_at(POINT), STATE, corner_inputs(SIGNS[0]), PERIOD, noise, readings)


@pytest.mark.parametrize(
    ("weights", "message"),
    [([1.0], "weights for"), ([0.0, 0.0], "not all zero"), ([-1.0, 2.0], "not negative")],
)
def test_invalid_designs_are_refused(weights, message) -> None:  # noqa: ANN001
    blocks = [information(signs, POINT) for signs in SIGNS[:2]]
    with pytest.raises(ValueError, match=message):
        design_covariance(blocks, weights=weights)
    with pytest.raises(ValueError, match="at least one window"):
        design_covariance([])


def test_the_settings_of_a_profile_keep_the_starts_that_do_not_move_e_over_r() -> None:
    settings = profile_settings(9000.0)
    assert settings.fixed_activation_temperature == 9000.0
    assert settings.starts == STARTS_WITH_FIXED_ACTIVATION
    assert settings.max_evaluations == DEFAULT_FIT_SETTINGS.max_evaluations
    only_moving = FitSettings(starts=DEFAULT_STARTS[3:])
    with pytest.raises(ValueError, match="every declared start moves E/R"):
        profile_settings(9000.0, only_moving)


def test_the_profile_of_the_model_s_own_data_is_lowest_at_the_true_activation() -> None:
    true = steady_state_parameters(KNOWN, STATE, 9200.0)
    run = modeller_run(model_at(true).rhs, random_corners(3, 5), noise_seed=None)
    # every context reading replaced by the exact state at its onset, so that the context
    # mean is the exact initial state, as in the noise-free recovery test of I1
    measured = np.array(run.measured)
    windows = find_windows(run, NOMINAL, P3_LAYOUT).windows
    for window in windows:
        measured[window.context_ticks.start : window.onset + 1] = run.measured[window.onset]
    data = window_data(observations(run.inputs, measured=measured, run=run.run), windows)
    objectives = {}
    for activation in (8700.0, 9200.0, 9700.0):
        fit = fit_mechanistic("p", data, KNOWN, MODELLER, profile_settings(activation))
        assert all(record.outcome == CONVERGED for record in fit.starts)
        objectives[activation] = fit.objective
        if activation == 9200.0:
            assert fit.objective < 1e-5
            assert fit.parameters.k_350 == pytest.approx(true.k_350, rel=1e-6)
            assert fit.parameters.ua == pytest.approx(true.ua, rel=1e-6)
    assert objectives[9200.0] < min(objectives[8700.0], objectives[9700.0])


def test_the_interval_of_a_quadratic_profile_is_found_by_interpolation() -> None:
    grid = np.arange(9000.0, 10001.0, 100.0)
    increases = [((a - 9530.0) / 40.0) ** 2 for a in grid]
    found = profile_interval(grid, increases, CHI2_ONE_95)
    assert found.lowest == 9500.0
    half_width = 40.0 * math.sqrt(CHI2_ONE_95)  # 78.4 K
    # the chord of a convex profile lies above it, so the interpolated crossings fall inside
    # the true ones, by less than a quarter of the spacing here
    assert 9530.0 - half_width <= found.low <= 9530.0 - half_width + 25.0
    assert 9530.0 + half_width - 25.0 <= found.high <= 9530.0 + half_width
    exact = profile_interval([0.0, 1.0, 2.0], [4.0, 0.0, 4.0], 1.0)
    assert (exact.low, exact.high) == (0.75, 1.25)


def test_the_interval_says_why_a_side_has_no_crossing() -> None:
    grid = [1.0, 2.0, 3.0, 4.0]
    open_end = profile_interval(grid, [0.0, 0.5, 1.0, 2.0], 5.0)
    assert open_end.low is None and "end of the grid" in open_end.low_reason
    assert open_end.high is None and "end of the grid" in open_end.high_reason
    gap = profile_interval(grid, [9.0, None, 0.0, 9.0], 5.0)
    assert gap.low is None and "no start converged at E/R = 2.0" in gap.low_reason
    assert gap.high == pytest.approx(3.0 + 5.0 / 9.0)


@pytest.mark.parametrize(
    ("grid", "increases", "threshold", "message"),
    [
        ([1.0, 1.0], [0.0, 1.0], 1.0, "strictly increasing"),
        ([1.0, 2.0], [0.0], 1.0, "one increase"),
        ([1.0, 2.0], [None, None], 1.0, "no point"),
        ([1.0, 2.0], [0.0, 1.0], 0.0, "positive and finite"),
        ([1.0, 2.0], [2.0, 3.0], 1.0, "above the threshold"),
    ],
)
def test_invalid_profiles_are_refused(grid, increases, threshold, message) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match=message):
        profile_interval(grid, increases, threshold)


@pytest.mark.parametrize(
    ("noise", "readings", "message"),
    [
        ((1e200, 1e200), 10, r"sigma\^2 is"),  # the variance overflows
        ((1e-200, 0.5), 10, r"sigma\^2 is"),  # the variance underflows to zero
        ((5.0, 0.5), 10**400, "beyond the largest double"),
    ],
)
def test_noise_levels_whose_variance_is_not_a_double_are_refused(noise, readings, message) -> None:  # noqa: ANN001
    """Finite noise levels can still give a variance or its inverse that is not a positive
    double; the solve that forms the context information would then fail or divide by zero
    (the review of I2 before its registration)."""
    with pytest.raises(ValueError, match=message):
        window_information(model_at(POINT), STATE, corner_inputs(SIGNS[0]), PERIOD, noise, readings)


def test_a_design_with_fewer_rows_than_parameters_is_rank_deficient() -> None:
    """One scored reading gives two rows for three parameters: numpy returns two singular
    values, and the design is rank deficient, not judged on the smaller of two."""
    short = window_information(model_at(POINT), STATE, corner_inputs(SIGNS[0])[:1], PERIOD, SIGMA)
    assert isinstance(short, WindowInformation) and short.rows == 2
    design = design_covariance([short])
    assert design.reason == "the design is rank deficient" and len(design.singular_values) == 2


def test_weights_whose_rows_overflow_are_refused() -> None:
    block = information(SIGNS[0], POINT)
    with pytest.raises(ValueError, match="beyond the largest double"):
        design_covariance([block, block], weights=[1e306, 1e306])


def test_the_sandwich_is_formed_as_the_covariance_of_a_fit_forms_it() -> None:
    """exact^-1 + exact^-1 excess exact^-1, without forming the middle S^T Sigma S."""
    block = information(SIGNS[1], POINT)
    design = design_covariance([block])
    assert np.allclose(block.meat, block.exact + block.excess, rtol=0.0, atol=0.0)
    bread = np.linalg.inv(block.exact)
    units = np.array([1.0, 350.0, 1.0])
    expected = (bread @ block.meat @ bread) * np.outer(units, units)
    assert np.allclose(design.sandwich, expected, rtol=1e-8, atol=0.0)
