"""The equations of the learned models of M1 and the rollout that trains them
(docs/m1_plan.md, sections 8.2, 8.3 and 13, the criteria of I3): the hybrids at the identity,
the gradients, the changes of the inputs, the physical bounds and the domains. The data are
simulated on the truth side of the tests with the modeller's equations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from m1_support import NOMINAL, SIGMA, corner, modeller_run, random_corners
from process_transfer.evaluation.physics import validity_violations
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.models import learned
from process_transfer.models.mechanistic import MechanisticModel, MechanisticParameters
from process_transfer.models.rollout import EVALUATION_SETTINGS, predict_window, rollout
from process_transfer.models.training import rk4_rollout

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
TRUE = MechanisticParameters.from_k_350(0.0177, 9000.0, 1330.0)
SCALES = learned.Scales(
    state_mean=np.array([190.0, 355.0]),
    state_std=np.array([40.0, 6.0]),
    input_mean=np.array(NOMINAL),
    input_std=np.array([1e-4, 30.0, 3.0, 3.0]),
)
START = {"BN": None, "HK": TRUE, "HU": TRUE, "HKU": TRUE}


def windows_of(corners, noise_seed=None):  # noqa: ANN001, ANN201
    run = modeller_run(MechanisticModel("m", KNOWN, TRUE).rhs, corners, noise_seed=noise_seed)
    return window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)


def moved(parameters, rng, size=0.3):  # noqa: ANN001, ANN201
    """The same parameters with every weight moved, the last layers included."""
    found = {}
    for name, value in parameters.items():
        if name == "theta":
            found[name] = np.array(value)
        else:
            found[name] = [
                (w + size * rng.standard_normal(w.shape), b + size * rng.standard_normal(b.shape))
                for w, b in value
            ]
    return found


def corner_window(signs) -> np.ndarray:  # noqa: ANN001
    inputs = np.tile(NOMINAL, (110, 1))
    inputs[:20] = corner(signs)
    return inputs


@pytest.mark.parametrize("family", learned.HYBRIDS)
@pytest.mark.parametrize("fixed", [None, 9000.0])
def test_a_hybrid_at_the_identity_is_its_mechanistic_model(family, fixed) -> None:  # noqa: ANN001
    """The criterion of I3: HK with its correction at the identity equals MR_F to the
    precision of the integrator; so do HU and HKU, with E/R estimated or held."""
    rng = np.random.default_rng(3)
    parameters = learned.initial_parameters(family, (8,), rng, TRUE, fixed)
    model = learned.LearnedModel(family, family, parameters, KNOWN, SCALES, fixed)
    reference = MechanisticModel("MR_F", KNOWN, TRUE, fixed)
    for _ in range(20):
        x = np.array([rng.uniform(60.0, 350.0), rng.uniform(335.0, 380.0)])
        u = corner(rng.choice([-1, 1], size=4))
        # the two compute the rate by equivalent operations, k_350 exp(...) and the
        # modeller's k0 exp(-(E/R) / T), which agree to rounding; a derivative is a difference
        # of terms of the size of the feed terms, so it agrees to rounding of those
        feed = np.array([u[0] / KNOWN.volume * u[1], u[0] / KNOWN.volume * u[2]])
        assert np.all(np.abs(model.rhs(x, u) - reference.rhs(x, u)) <= 1e-13 * feed)
    state = np.array([190.0, 355.0])
    hybrid = rollout(model, state, corner_window((1, -1, 1, 1)), 6.0).states
    mechanistic = rollout(reference, state, corner_window((1, -1, 1, 1)), 6.0).states
    assert np.max(np.abs(hybrid - mechanistic) / np.array(SIGMA)) <= 1e-6
    found = model.mechanistic_parameters()
    for name in ("k0", "activation_temperature", "ua"):
        assert getattr(found, name) == pytest.approx(getattr(TRUE, name), rel=1e-14)


def test_the_networks_start_with_a_zero_last_layer_and_glorot_hidden_layers() -> None:
    rng = np.random.default_rng(0)
    layers = learned.initial_layers(rng, 6, (32, 32), 2)
    assert [w.shape for w, _ in layers] == [(6, 32), (32, 32), (32, 2)]
    assert np.all(layers[-1][0] == 0.0) and np.all(layers[-1][1] == 0.0)
    for (w, b), fan_in, fan_out in zip(layers[:-1], (6, 32), (32, 32), strict=True):
        assert np.all(np.abs(w) <= np.sqrt(6.0 / (fan_in + fan_out))) and np.any(w != 0.0)
        assert np.all(b == 0.0)
    bn = learned.initial_parameters("BN", (16, 16), rng, None)
    assert learned.network_size(bn) == 6 * 16 + 16 + 16 * 16 + 16 + 16 * 2 + 2
    hku = learned.initial_parameters("HKU", (8,), rng, TRUE)
    assert learned.network_size(hku) == 2 * (2 * 8 + 8 + 8 + 1)
    # BN starts as a model whose state does not move
    model = learned.LearnedModel("BN", "BN", bn, KNOWN, SCALES)
    assert np.all(model.rhs(np.array([150.0, 360.0]), corner((1, 1, 1, 1))) == 0.0)


@pytest.mark.parametrize(
    ("family", "hidden", "start", "message"),
    [
        ("HX", (8,), TRUE, "family must be one of"),
        ("HK", (), TRUE, "hidden"),
        ("HK", (0,), TRUE, "hidden"),
        ("HK", (8,), None, "starts from MR_F"),
        ("BN", (8,), TRUE, "no mechanistic parameters"),
    ],
)
def test_invalid_models_are_refused(family, hidden, start, message) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match=message):
        learned.initial_parameters(family, hidden, np.random.default_rng(0), start)


@pytest.mark.parametrize("family", learned.FAMILIES)
def test_gradients_agree_with_central_differences(family) -> None:  # noqa: ANN001
    """The criterion of I3: the gradient of the loss of the training, by automatic
    differentiation through the rollout, agrees with central differences at a point where
    every weight, the last layers included, and every mechanistic parameter is moved."""
    data = windows_of(random_corners(2, 8), noise_seed=4)
    rng = np.random.default_rng(12)
    hidden = (6, 6) if family == "BN" else (4,)
    point = moved(learned.initial_parameters(family, hidden, rng, START[family]), rng)
    flat, unravel = ravel_pytree(jax.tree_util.tree_map(jnp.asarray, point))
    x0 = jnp.asarray(np.array([d.initial_state for d in data]))
    inputs = jnp.asarray(np.array([d.inputs[:40] for d in data]))
    scored = jnp.asarray(np.array([d.scored[:40] for d in data]))
    sigma = jnp.asarray(SIGMA)

    def loss(vector):  # noqa: ANN001, ANN202
        predicted = rk4_rollout(family, unravel(vector), x0, inputs, 6.0, 2, KNOWN, SCALES, None)
        return jnp.mean(((predicted - scored) / sigma) ** 2)

    f = jax.jit(loss)
    gradient = np.asarray(jax.jit(jax.grad(loss))(flat))
    values = np.asarray(flat)
    largest = np.max(np.abs(gradient))
    assert largest > 0.0
    for i in np.unique(np.linspace(0, len(values) - 1, 15).astype(int)):
        step = 1e-6 * max(1.0, abs(values[i]))
        up, down = values.copy(), values.copy()
        up[i] += step
        down[i] -= step
        difference = (float(f(jnp.asarray(up))) - float(f(jnp.asarray(down)))) / (2.0 * step)
        assert abs(difference - gradient[i]) <= 1e-6 * largest, (i, difference, gradient[i])


def test_the_rollout_of_the_training_follows_the_inputs_row_by_row() -> None:
    """The criterion of I3 on changes of the inputs. Each row is integrated with its own
    inputs: a change of the inputs of row k moves the states from the end of row k on and
    none before. On a window of P3 the scheme agrees with the reference rollout, which
    restarts at every change, to 1e-3 sigma."""
    rng = np.random.default_rng(21)
    parameters = moved(learned.initial_parameters("HK", (8,), rng, TRUE), rng, 0.1)
    inputs = corner_window((-1, 1, 1, -1))
    x0 = np.array([[190.0, 355.0]])

    def scheme(rows: np.ndarray) -> np.ndarray:
        found = rk4_rollout("HK", parameters, x0, rows[None], 6.0, 2, KNOWN, SCALES, None)
        return np.asarray(found)[0]

    base = scheme(inputs)
    changed = inputs.copy()
    changed[50] = corner((1, 1, 1, 1))
    moved_states = scheme(changed)
    assert np.array_equal(moved_states[:50], base[:50])
    assert np.all(moved_states[50] != base[50])
    model = learned.LearnedModel("HK", "HK", parameters, KNOWN, SCALES)
    reference = rollout(model, x0[0], inputs, 6.0).states
    assert np.max(np.abs(base - reference) / np.array(SIGMA)) <= 1e-3


def test_the_scheme_integrates_a_linear_equation_to_its_order() -> None:
    """BN made linear by a network of one hidden unit with tiny weights, where tanh is its
    argument: against the exact solution of dx/dt = D (w2 w1 z), the error of the scheme
    falls sixteen times when its steps are halved, the order of the classical Runge-Kutta
    scheme, across a change of the inputs."""
    w1 = np.zeros((6, 1))
    w1[0, 0] = 1e-6
    parameters = {"dynamics": [(w1, np.zeros(1)), (np.array([[-2e5, 0.0]]), np.zeros(2))]}
    rate = SCALES.derivative[0] * -2e5 * 1e-6 / SCALES.state_std[0]  # dC/dt = rate (C - m_C)
    inputs = corner_window((1, 1, 1, 1))
    x0 = np.array([[250.0, 355.0]])
    errors = []
    for substeps in (1, 2):
        found = rk4_rollout("BN", parameters, x0, inputs[None], 6.0, substeps, KNOWN, SCALES, None)
        t = 6.0 * np.arange(1, 111)
        exact = SCALES.state_mean[0] + (250.0 - SCALES.state_mean[0]) * np.exp(rate * t)
        errors.append(np.max(np.abs(np.asarray(found)[0, :, 0] - exact)))
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.05)


VALIDITY_WINDOWS = windows_of([(1, 1, 1, 1), (-1, -1, -1, -1), (1, -1, -1, 1)])


@pytest.mark.parametrize("family", learned.HYBRIDS)
def test_the_hybrids_are_physically_valid_by_construction(family) -> None:  # noqa: ANN001
    """Whatever their weights, the factors are positive: the rate is not negative for
    C_A >= 0 and the conductance is positive, so a completed rollout never breaks the
    validity bounds of section 9.3."""
    rng = np.random.default_rng(31)
    for _ in range(5):
        parameters = moved(learned.initial_parameters(family, (8,), rng, TRUE), rng, 1.0)
        model = learned.LearnedModel(family, family, parameters, KNOWN, SCALES)
        for data in VALIDITY_WINDOWS:
            predicted = predict_window(model, data)
            if isinstance(predicted, np.ndarray):
                assert validity_violations(data, predicted, KNOWN) == ()
        for x in rng.uniform([0.0, 330.0], [400.0, 380.0], size=(20, 2)):
            u = corner(rng.choice([-1, 1], size=4))
            dilution = u[0] / KNOWN.volume
            rate = dilution * (u[1] - x[0]) - model.rhs(x, u)[0]
            assert rate >= 0.0


def test_an_exponential_that_overflows_is_a_failed_rollout_not_a_number() -> None:
    """A factor whose logarithm is 800 overflows: the rollout records the failure of the
    right-hand side, and nothing is clipped."""
    parameters = learned.initial_parameters("HK", (4,), np.random.default_rng(0), TRUE)
    parameters["kinetic"][-1] = (parameters["kinetic"][-1][0], np.array([800.0]))
    model = learned.LearnedModel("HK", "HK", parameters, KNOWN, SCALES)
    data = windows_of(random_corners(1, 3))[0]
    found = predict_window(model, data, EVALUATION_SETTINGS)
    assert not isinstance(found, np.ndarray) and found.cause == "right-hand side"


def test_a_hybrid_cannot_start_where_the_rate_law_divides_by_zero() -> None:
    parameters = learned.initial_parameters("HU", (4,), np.random.default_rng(0), TRUE)
    hybrid = learned.LearnedModel("HU", "HU", parameters, KNOWN, SCALES)
    assert "divides by the temperature" in hybrid.initial_state_problem(np.array([100.0, 0.0]))
    bn = learned.LearnedModel(
        "BN",
        "BN",
        learned.initial_parameters("BN", (4,), np.random.default_rng(0), None),
        KNOWN,
        SCALES,
    )
    assert bn.initial_state_problem(np.array([100.0, 0.0])) is None
    assert bn.mechanistic_parameters() is None


def test_scales_come_from_the_fitting_windows_and_a_zero_scale_is_refused() -> None:
    """Section 5.8: the mean and deviation of each variable over the rows of F; a variable
    that does not vary is refused with its name."""
    data = windows_of(random_corners(2, 5))
    scales = learned.fitting_scales(data)
    readings = np.concatenate([np.vstack([d.context, d.scored]) for d in data])
    assert np.array_equal(scales.state_mean, readings.mean(axis=0))
    assert np.array_equal(scales.input_std, np.concatenate([d.inputs for d in data]).std(axis=0))
    run = modeller_run(MechanisticModel("m", KNOWN, TRUE).rhs, [(1, 1, 1, 0)], noise_seed=None)
    steady_tc = window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)
    with pytest.raises(ValueError, match="standard deviation of T_c"):
        learned.fitting_scales(steady_tc)
    with pytest.raises(ValueError, match="at least one window"):
        learned.fitting_scales(())
