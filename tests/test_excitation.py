"""Excitation sequences: reproducible from a seed, and obeying their stated rules."""

import numpy as np
import pytest

from process_transfer.simulation.excitation import (
    binary_levels,
    levels_to_segments,
    limited_move_levels,
    separated_excursions,
)

NOMINAL = np.array([1.0, 10.0, 100.0, 1000.0])
AMPLITUDES = np.array([0.1, 1.0, 5.0, 50.0])


@pytest.mark.parametrize("generator", [binary_levels, limited_move_levels])
def test_a_sequence_is_a_function_of_its_seed(generator) -> None:  # noqa: ANN001
    first = generator(200, 4, np.random.default_rng(7))
    again = generator(200, 4, np.random.default_rng(7))
    other = generator(200, 4, np.random.default_rng(8))
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.shape == (200, 4)


def test_binary_levels_use_both_extremes_of_every_input_and_nothing_else() -> None:
    levels = binary_levels(500, 4, np.random.default_rng(0))
    assert set(np.unique(levels)) == {-1, 1}
    for column in levels.T:
        assert 0.4 < np.mean(column == 1) < 0.6  # roughly balanced
    assert np.any(np.abs(np.diff(levels, axis=0)) == 2)  # full-range jumps do happen


def test_limited_moves_never_change_an_input_by_more_than_one_level() -> None:
    levels = limited_move_levels(2000, 4, np.random.default_rng(0))
    assert set(np.unique(levels)) == {-1, 0, 1}
    assert np.all(np.abs(levels[0]) <= 1)  # the first tick is one move away from zero
    assert np.max(np.abs(np.diff(levels, axis=0))) == 1


def test_levels_become_inputs_around_the_nominal_values() -> None:
    levels = np.array([[1, -1, 0, 1], [-1, -1, 1, 0]])
    segments = levels_to_segments(NOMINAL, AMPLITUDES, levels, clock=120.0)
    assert [segment.duration for segment in segments] == [120.0, 120.0]
    np.testing.assert_allclose(segments[0].inputs, [1.1, 9.0, 100.0, 1050.0])
    np.testing.assert_allclose(segments[1].inputs, [0.9, 9.0, 105.0, 1000.0])
    np.testing.assert_array_equal(NOMINAL, [1.0, 10.0, 100.0, 1000.0])  # not mutated


def test_separated_excursions_alternate_a_corner_with_a_rest_at_nominal() -> None:
    segments = separated_excursions(
        NOMINAL, AMPLITUDES, n_excursions=25, hold=120.0, rest=600.0, rng=np.random.default_rng(3)
    )
    assert len(segments) == 50
    for excursion, rest in zip(segments[0::2], segments[1::2], strict=True):
        assert excursion.duration == 120.0
        assert rest.duration == 600.0
        np.testing.assert_allclose(np.abs(excursion.inputs - NOMINAL), AMPLITUDES)  # a corner
        np.testing.assert_array_equal(rest.inputs, NOMINAL)
    assert sum(segment.duration for segment in segments) == 25 * 720.0


def test_invalid_arguments_are_rejected() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_ticks"):
        binary_levels(0, 4, rng)
    with pytest.raises(ValueError, match="n_inputs"):
        limited_move_levels(10, 0, rng)
    with pytest.raises(ValueError, match="clock"):
        levels_to_segments(NOMINAL, AMPLITUDES, np.zeros((2, 4), dtype=int), clock=0.0)
    with pytest.raises(ValueError, match="amplitudes"):
        levels_to_segments(NOMINAL, -AMPLITUDES, np.zeros((2, 4), dtype=int), clock=1.0)
    with pytest.raises(ValueError, match="shape"):
        levels_to_segments(NOMINAL, AMPLITUDES, np.zeros((2, 3), dtype=int), clock=1.0)
    with pytest.raises(ValueError, match="n_excursions"):
        separated_excursions(NOMINAL, AMPLITUDES, 0, 120.0, 600.0, rng)
