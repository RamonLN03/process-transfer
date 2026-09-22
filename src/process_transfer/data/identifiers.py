"""Identifiers of data sets, plants and runs, and the noise stream that follows from a run.

An identifier that becomes part of a path must be the same file on Windows and on
Linux, so it is lower-case ASCII: a letter or digit first, then letters, digits, ``.``,
``_`` or ``-``. Two names that differ only in case are one file on Windows and two on
Linux; a name such as ``con`` or ``nul.p3`` is a device on Windows; a trailing dot is
dropped by Windows. All of these are refused rather than repaired.

The logical identity of a run says what the run is (``docs/data_contract.md``):

    <plant_id>.<protocol>.<definition>.n<noise realisation>

where the definition depends on the protocol and never on the clock or the content:

    p3       e<excitation seed>.x<number of excursions>       target.p3.e0.x10.n0
    steady   d<duration in seconds>                            target.steady.d7200.n0
    step     <input>-<direction>.l<lead>.h<hold>.r<recovery>   target.step.tc-up.l600.h600.r600.n0

The noise stream is derived from the identity, so that the same identity always means
the same noise and a new realisation always means a new one. The definition part is the
same on every plant that runs the same experiment, which is how the plants are paired.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np

MAX_IDENTIFIER_LENGTH = 100
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]*")
_WINDOWS_DEVICES = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{digit}" for digit in range(1, 10)]
    + [f"lpt{digit}" for digit in range(1, 10)]
)


def path_identifier(name: str, value: object) -> str:
    """``value`` if it is a valid identifier for use in a path, ``ValueError`` otherwise."""
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be lower-case ASCII, a letter or digit first and then letters, "
            f"digits, '.', '_' or '-', got {value!r}"
        )
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{name} is longer than {MAX_IDENTIFIER_LENGTH} characters: {value!r}")
    if value.endswith("."):
        raise ValueError(f"{name} must not end in a dot, which Windows drops: {value!r}")
    if value.split(".")[0] in _WINDOWS_DEVICES:
        raise ValueError(f"{name} starts with the name of a Windows device: {value!r}")
    return value


def _count(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return int(value)


_TOKEN = re.compile(r"[a-z0-9]+")


def _part(name: str, value: object) -> str:
    """A part of an identity: lower-case letters and digits, without the separators."""
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must be lower-case letters and digits only, got {value!r}")
    return value


def _identity(plant_id: str, protocol: str, definition: str, noise_realisation: int) -> str:
    path_identifier("plant_id", plant_id)
    if "." in plant_id:
        raise ValueError(
            f"plant_id must not contain a dot, which separates the parts: {plant_id!r}"
        )
    _part("protocol", protocol)
    realisation = _count("noise_realisation", noise_realisation, 0)
    return path_identifier("run_id", f"{plant_id}.{protocol}.{definition}.n{realisation}")


def run_identifier(
    plant_id: str, protocol: str, excitation_seed: int, n_excursions: int, noise_realisation: int
) -> str:
    """The logical identity of a seeded excursion run such as P3, built from its
    definition and from nothing else.

    The noise realisation is part of it, so two realisations of the noise on the same
    excitation are two runs. The seed of the noise is not part of it.
    """
    seed = _count("excitation_seed", excitation_seed, 0)
    excursions = _count("n_excursions", n_excursions, 1)
    return _identity(plant_id, protocol, f"e{seed}.x{excursions}", noise_realisation)


def steady_run_identifier(plant_id: str, duration_s: int, noise_realisation: int) -> str:
    """The identity of a run of steady operation: the nominal inputs held for a whole
    number of seconds. Nothing is drawn at random, so there is no seed to name."""
    seconds = _count("duration_s", duration_s, 1)
    return _identity(plant_id, "steady", f"d{seconds}", noise_realisation)


def step_run_identifier(
    plant_id: str,
    input_name: str,
    direction: str,
    lead_s: int,
    hold_s: int,
    recovery_s: int,
    noise_realisation: int,
) -> str:
    """The identity of a single-input step test: which input, which direction, and the
    three durations in whole seconds. Nothing is drawn at random."""
    _part("input_name", input_name)
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    lead = _count("lead_s", lead_s, 1)
    hold = _count("hold_s", hold_s, 1)
    recovery = _count("recovery_s", recovery_s, 1)
    definition = f"{input_name}-{direction}.l{lead}.h{hold}.r{recovery}"
    return _identity(plant_id, "step", definition, noise_realisation)


def run_definition(run_id: str) -> str:
    """The identity without its plant: what pairs the runs of two plants."""
    return path_identifier("run_id", run_id).split(".", 1)[1]


def noise_stream_words(run_id: str) -> tuple[int, int, int, int]:
    """The noise stream of a run: the first 16 bytes of the SHA-256 of its identity, as
    four big-endian 32-bit words. Deterministic across processes and machines, which
    Python's ``hash()`` is not, and every word fits the rule for noise keys."""
    digest = hashlib.sha256(path_identifier("run_id", run_id).encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[4 * i : 4 * i + 4], "big") for i in range(4))  # type: ignore[return-value]


def require_distinct_runs(run_ids: Iterable[str]) -> dict[str, tuple[int, int, int, int]]:
    """The noise stream of every run of a data set, or ``ValueError`` if two runs share
    an identity or a stream. The second is astronomically unlikely for different
    identities; it is checked, not assumed."""
    streams: dict[str, tuple[int, int, int, int]] = {}
    owners: dict[tuple[int, int, int, int], str] = {}
    for run_id in run_ids:
        if run_id in streams:
            raise ValueError(
                f"the run {run_id!r} is defined twice. A repetition is the same run; a new "
                "realisation of the noise needs a new n<k> in its identity"
            )
        words = noise_stream_words(run_id)
        if words in owners:
            raise ValueError(
                f"the runs {owners[words]!r} and {run_id!r} would share the noise stream {words}"
            )
        streams[run_id], owners[words] = words, run_id
    return streams
