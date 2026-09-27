"""The training of the learned models and their selection on V (docs/m1_plan.md, sections
5.5, 5.8 and 8.7): where a training starts, what it may read, how it fails, how a checkpoint
and a configuration are chosen, and that it is reproducible. The data are simulated on the
truth side of the tests with the modeller's equations."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from m1_support import NOMINAL, SIGMA, modeller_run, random_corners
from process_transfer.evaluation.outcomes import TrainingFailure
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.models import learned
from process_transfer.models.mechanistic import MechanisticModel, MechanisticParameters
from process_transfer.models.rollout import EVALUATION_SETTINGS, predict_window
from process_transfer.models.training import (
    Checkpoint,
    Configuration,
    TrainingRecord,
    TrainingSettings,
    rk4_rollout,
    select_configuration,
    train,
    training_seed,
    validation_score,
)

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
TRUE = MechanisticParameters.from_k_350(0.0177, 9000.0, 1330.0)
SHORT = TrainingSettings(learning_rate=3e-3, max_steps=20, validation_every=10)


def windows_of(corners, noise_seed=None, rhs=None):  # noqa: ANN001, ANN201
    f = rhs or MechanisticModel("m", KNOWN, TRUE).rhs
    run = modeller_run(f, corners, noise_seed=noise_seed)
    return window_data(run, find_windows(run, NOMINAL, P3_LAYOUT).windows)


DATA = windows_of(random_corners(4, 5), noise_seed=11)
FITTING, VALIDATION = DATA[:3], DATA[3:]


def test_the_training_of_a_hybrid_starts_at_mr_f() -> None:
    """The first checkpoint of a hybrid is its start, MR_F with the factors at one: its
    criterion on V is MR_F's, so a hybrid that does not improve on V ends as MR_F."""
    for family in learned.HYBRIDS:
        record = train(
            Configuration(family, (8,), 1e-4), SHORT, FITTING, VALIDATION, KNOWN, 1, TRUE
        )
        start = validation_score(MechanisticModel("MR_F", KNOWN, TRUE), VALIDATION)[0]
        assert record.checkpoints[0].step == 0
        assert record.checkpoints[0].validation == pytest.approx(start, rel=1e-8)
        assert [c.step for c in record.checkpoints] == [0, 10, 20]


def test_the_identity_does_not_block_the_gradient() -> None:
    """At the start the last layer of each factor is zero and the hidden layers are not:
    the gradient reaches the last layer, and after one step it reaches the hidden layers."""
    x0 = jnp.asarray(np.array([d.initial_state for d in FITTING]))
    inputs = jnp.asarray(np.array([d.inputs for d in FITTING]))
    scored = jnp.asarray(np.array([d.scored for d in FITTING]))
    scales = learned.fitting_scales(FITTING)
    for family in learned.HYBRIDS:
        start = learned.initial_parameters(family, (8,), np.random.default_rng(5), TRUE)
        flat, unravel = ravel_pytree(jax.tree_util.tree_map(jnp.asarray, start))

        def loss(vector, family=family, unravel=unravel):  # noqa: ANN001, ANN202
            predicted = rk4_rollout(
                family, unravel(vector), x0, inputs, 6.0, 2, KNOWN, scales, None
            )
            return jnp.mean(((predicted - scored) / jnp.asarray(SIGMA)) ** 2)

        gradient = unravel(jax.grad(loss)(flat))
        for name in ("kinetic", "thermal"):
            if name in gradient:
                (hidden_w, _), (last_w, last_b) = gradient[name]
                assert float(jnp.max(jnp.abs(last_w))) > 0.0 and float(jnp.abs(last_b[0])) > 0.0
                assert float(jnp.max(jnp.abs(hidden_w))) == 0.0
        moved = flat - 1e-3 * jax.grad(loss)(flat)
        after = unravel(jax.grad(loss)(moved))
        for name in ("kinetic", "thermal"):
            if name in after:
                assert float(jnp.max(jnp.abs(after[name][0][0]))) > 0.0


def test_a_training_is_reproducible_bit_for_bit() -> None:
    first = train(Configuration("BN", (6, 6), 1e-3), SHORT, FITTING, VALIDATION, KNOWN, 42)
    second = train(Configuration("BN", (6, 6), 1e-3), SHORT, FITTING, VALIDATION, KNOWN, 42)
    assert first.checkpoints == second.checkpoints
    assert first.selected == second.selected
    for (w1, b1), (w2, b2) in zip(
        first.parameters["dynamics"], second.parameters["dynamics"], strict=True
    ):
        assert np.array_equal(w1, w2) and np.array_equal(b1, b2)
    other = train(Configuration("BN", (6, 6), 1e-3), SHORT, FITTING, VALIDATION, KNOWN, 43)
    assert other.checkpoints != first.checkpoints


def test_the_mechanistic_parameters_can_be_held() -> None:
    record = train(
        Configuration("HK", (8,), 0.0),
        SHORT,
        FITTING,
        VALIDATION,
        KNOWN,
        1,
        TRUE,
        hold_mechanistic=True,
    )
    held = TRUE.coordinates(None)
    selected = record.parameters["theta"]
    assert np.array_equal(selected, held) or record.selected == 0
    moved = train(Configuration("HK", (8,), 0.0), SHORT, FITTING, VALIDATION, KNOWN, 1, TRUE)
    if moved.selected:
        assert not np.array_equal(moved.parameters["theta"], held)
    with pytest.raises(ValueError, match="no mechanistic parameters to hold"):
        train(
            Configuration("BN", (4,), 0.0),
            SHORT,
            FITTING,
            VALIDATION,
            KNOWN,
            1,
            hold_mechanistic=True,
        )


def test_the_selected_model_is_evaluated_by_the_reference_rollout() -> None:
    record = train(Configuration("HK", (8,), 1e-4), SHORT, FITTING, VALIDATION, KNOWN, 3, TRUE)
    model = record.model(KNOWN)
    for data in VALIDATION:
        predicted = predict_window(model, data)
        assert isinstance(predicted, np.ndarray)
        reference = predict_window(model, data, EVALUATION_SETTINGS)
        assert np.array_equal(predicted, reference)
    score = validation_score(model, VALIDATION)[0]
    assert score == record.criterion


def test_a_scale_of_zero_refuses_the_training() -> None:
    """Section 5.8: a fitting part in which T_c never moves has no scale for it, and the
    training is a failure before any step."""
    flat = windows_of([(1, 1, 1, 0), (-1, 1, -1, 0), (1, -1, 1, 0)])
    record = train(Configuration("HU", (4,), 0.0), SHORT, flat[:2], flat[2:], KNOWN, 1, TRUE)
    assert record.failure is not None and "refused scale" in record.failure.reason
    assert "T_c" in record.failure.reason and record.checkpoints == () and record.steps_run == 0
    with pytest.raises(ValueError, match="has no model"):
        record.model(KNOWN)


def test_a_loss_that_is_not_finite_ends_the_training_as_a_failure() -> None:
    """A start whose rate constant, 1e300 1/s, overflows the rollout of the training: the
    loss is not finite at step 0, and the training is a failure whose checkpoints are not
    used (section 8.7)."""
    start = MechanisticParameters.from_k_350(1e300, 0.0, 1330.0)
    record = train(Configuration("HK", (4,), 0.0), SHORT, FITTING, VALIDATION, KNOWN, 1, start)
    assert record.failure is not None and "not finite at step 0" in record.failure.reason
    assert record.selected is None and record.parameters is None


def test_a_checkpoint_whose_rollout_of_v_fails_cannot_be_selected() -> None:
    """A validation window whose context puts T at zero: the hybrid cannot start there, so
    every checkpoint has a failed rollout of V and the training is a failure."""
    bad = dataclasses.replace(VALIDATION[0], context=np.tile([190.0, 0.0], (10, 1)))
    record = train(Configuration("HK", (4,), 0.0), SHORT, FITTING, (bad,), KNOWN, 1, TRUE)
    assert all(c.validation is None and c.failures for c in record.checkpoints)
    assert record.failure is not None and "every checkpoint" in record.failure.reason


def fake(criteria, failure=None):  # noqa: ANN001, ANN201
    checkpoints = tuple(
        Checkpoint(10 * i, 1.0, 0.0, value, () if value is not None else ("x",), 0.0, 0.0)
        for i, value in enumerate(criteria)
    )
    usable = [i for i, c in enumerate(checkpoints) if c.validation is not None]
    selected = None if failure or not usable else min(usable, key=lambda i: (criteria[i], i))
    return TrainingRecord(
        Configuration("BN", (4,), 0.0),
        SHORT,
        0,
        None,
        (),
        (),
        None,
        0,
        checkpoints,
        selected,
        None,
        None if failure is None else TrainingFailure(failure),
        0,
        {},
    )


def test_a_configuration_is_chosen_by_its_criterion_on_v_the_first_on_a_tie() -> None:
    records = [fake([2.0, 1.5]), fake([1.2, None]), fake([1.2, 1.3]), fake([0.1], "diverged")]
    assert select_configuration(records) == 1
    assert select_configuration([fake([3.0], "x"), fake([None])]) is None


def test_the_earliest_checkpoint_wins_a_tie() -> None:
    """Checkpoints of equal criterion: the training keeps the first."""
    record = train(
        Configuration("HU", (4,), 0.0),
        TrainingSettings(learning_rate=1e-12, max_steps=2, validation_every=1),
        FITTING,
        VALIDATION,
        KNOWN,
        1,
        TRUE,
    )
    values = [c.validation for c in record.checkpoints]
    assert record.selected == values.index(min(values))


def test_the_seed_of_a_configuration_is_declared_and_shared_by_budgets() -> None:
    assert training_seed(20260928, 3, 1) == training_seed(20260928, 3, 1)
    seeds = {training_seed(20260928, r, c) for r in range(10) for c in range(4)}
    assert len(seeds) == 40
    with pytest.raises(ValueError, match="replicate"):
        training_seed(20260928, -1, 0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": 15}, "multiple of validation_every"),
        ({"beta1": 1.0}, "beta1"),
        ({"substeps": True}, "substeps"),
    ],
)
def test_invalid_settings_are_refused(changes, message) -> None:  # noqa: ANN001
    arguments = {"learning_rate": 1e-3, "max_steps": 20, "validation_every": 10} | changes
    with pytest.raises(ValueError, match=message):
        TrainingSettings(**arguments)


def test_invalid_trainings_are_refused() -> None:
    config = Configuration("HK", (4,), 0.0)
    with pytest.raises(ValueError, match="penalty"):
        Configuration("HK", (4,), -1.0)
    with pytest.raises(ValueError, match="disjoint"):
        train(config, SHORT, FITTING, FITTING[:1], KNOWN, 1, TRUE)
    with pytest.raises(ValueError, match="windows of F and of V"):
        train(config, SHORT, FITTING, (), KNOWN, 1, TRUE)
    noisy = dataclasses.replace(VALIDATION[0], noise_std=np.array([4.0, 0.5]))
    with pytest.raises(ValueError, match="noise levels"):
        train(config, SHORT, FITTING, (noisy,), KNOWN, 1, TRUE)


def test_every_checkpoint_records_how_far_the_training_scheme_is_from_the_reference() -> None:
    record = train(Configuration("HK", (8,), 1e-4), SHORT, FITTING, VALIDATION, KNOWN, 3, TRUE)
    for checkpoint in record.checkpoints:
        assert 0.0 <= checkpoint.schemes_differ_in_sigmas < 1e-2
        assert math.isfinite(checkpoint.fitting_loss) and checkpoint.penalty >= 0.0
    assert set(record.seconds) == {"compile", "steps", "validation", "total"}
