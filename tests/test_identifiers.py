"""Identifiers that are safe in a path on Windows and Linux, the identity of a run, and
the noise stream that follows from it."""

import os
import subprocess
import sys

import numpy as np
import pytest

from process_transfer.data.identifiers import (
    MAX_IDENTIFIER_LENGTH,
    noise_stream_words,
    path_identifier,
    require_distinct_runs,
    run_identifier,
)
from process_transfer.measurement.sensors import STREAM_KEY_LIMIT, SensorSpec, measure


@pytest.mark.parametrize("good", ["source", "m0-e05", "target.p3.e0.x10.n0", "a", "9lives", "a_b"])
def test_valid_identifiers(good: str) -> None:
    assert path_identifier("dataset_id", good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "Target",  # one file on Windows, two on Linux
        "a b",
        "a/b",
        "a\\b",
        "../escape",
        ".hidden",
        "-flag",
        "trailing.",
        "con",
        "nul.p3.e0",
        "com1",
        "lpt9.x",
        "caf\u00e9",
        "a" * (MAX_IDENTIFIER_LENGTH + 1),
        None,
        7,
        b"bytes",
    ],
)
def test_identifiers_that_are_not_safe_in_a_path_are_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        path_identifier("dataset_id", bad)
    assert path_identifier("x", "a" * MAX_IDENTIFIER_LENGTH)  # the limit itself is valid


def test_the_identity_of_a_run_is_built_from_its_definition() -> None:
    assert run_identifier("target", "p3", 0, 10, 0) == "target.p3.e0.x10.n0"
    assert (
        run_identifier("source", "p3", np.int64(12), np.int32(3), np.uint8(2))
        == "source.p3.e12.x3.n2"
    )
    # another realisation of the noise on the same excitation is another run
    assert run_identifier("target", "p3", 0, 10, 1) != run_identifier("target", "p3", 0, 10, 0)


@pytest.mark.parametrize(
    "arguments",
    [
        ("Target", "p3", 0, 10, 0),
        ("tar.get", "p3", 0, 10, 0),  # a dot separates the parts
        ("target", "p.3", 0, 10, 0),
        ("target", "p3", -1, 10, 0),
        ("target", "p3", 0.0, 10, 0),
        ("target", "p3", True, 10, 0),
        ("target", "p3", "0", 10, 0),
        ("target", "p3", 0, 0, 0),  # a run has at least one excursion
        ("target", "p3", 0, 10, -1),
        ("target", "p3", 0, 10, 1.5),
    ],
)
def test_an_invalid_definition_has_no_identity(arguments: tuple) -> None:
    with pytest.raises(ValueError):
        run_identifier(*arguments)


def test_the_noise_stream_follows_from_the_identity() -> None:
    words = noise_stream_words("target.p3.e0.x10.n0")
    assert words == noise_stream_words("target.p3.e0.x10.n0")
    assert len(words) == 4 and all(
        type(word) is int and 0 <= word < STREAM_KEY_LIMIT for word in words
    )
    others = {
        noise_stream_words(run_identifier(plant, "p3", seed, 10, realisation))
        for plant in ("source", "target")
        for seed in range(20)
        for realisation in range(5)
    }
    assert len(others) == 200  # every identity its own stream
    with pytest.raises(ValueError, match="run_id"):
        noise_stream_words("Not An Identifier")


def test_the_stream_is_the_same_in_another_process() -> None:
    """Python's ``hash()`` of a string changes from one process to the next; this must not."""
    code = (
        "from process_transfer.data.identifiers import noise_stream_words;"
        "print(noise_stream_words('target.p3.e0.x10.n0'))"
    )
    results = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert results == {str(noise_stream_words("target.p3.e0.x10.n0"))}


def test_a_repetition_replays_the_noise_and_a_new_realisation_does_not() -> None:
    sensors = (SensorSpec("C_A", "mol/m^3", 5.0), SensorSpec("T", "K", 0.5))
    zeros = np.zeros((200, 2))
    first = measure(zeros, sensors, 5, noise_stream_words("target.p3.e0.x10.n0"))
    again = measure(zeros, sensors, 5, noise_stream_words("target.p3.e0.x10.n0"))
    other = measure(zeros, sensors, 5, noise_stream_words("target.p3.e0.x10.n1"))
    np.testing.assert_array_equal(first, again)
    assert not np.any(first == other)


def test_two_runs_with_one_identity_are_refused_when_a_data_set_is_defined() -> None:
    streams = require_distinct_runs(["source.p3.e0.x10.n0", "target.p3.e0.x10.n0"])
    assert set(streams) == {"source.p3.e0.x10.n0", "target.p3.e0.x10.n0"}
    assert streams["source.p3.e0.x10.n0"] != streams["target.p3.e0.x10.n0"]
    with pytest.raises(ValueError, match="defined twice"):
        require_distinct_runs(["source.p3.e0.x10.n0", "target.p3.e0.x10.n0", "source.p3.e0.x10.n0"])
