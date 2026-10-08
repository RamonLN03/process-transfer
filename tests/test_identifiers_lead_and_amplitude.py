"""The lead and the amplitude in the identity of a P3 run (D-039).

An identity of M0 keeps its meaning and its noise stream; the new form writes both tokens
out and can never name a run of M0. The noise streams are SHA-256 of the identity, so the
values pinned here hold on every platform; they are those recorded in the private
provenance of the published data set ``m0-e05``.
"""

import numpy as np
import pytest

from process_transfer.data.identifiers import (
    MAX_IDENTIFIER_LENGTH,
    P3_AMPLITUDES,
    noise_stream_words,
    require_distinct_runs,
    run_definition,
    run_identifier,
)
from process_transfer.simulation import protocols


def test_the_identities_of_m0_and_their_noise_streams_are_unchanged() -> None:
    assert run_identifier("target", "p3", 0, 10, 0) == "target.p3.e0.x10.n0"
    assert run_identifier("source", "p3", 2, 10, 0) == "source.p3.e2.x10.n0"
    assert noise_stream_words("target.p3.e0.x10.n0") == (
        592392307,
        2625639068,
        1063869705,
        189434225,
    )
    assert noise_stream_words("source.p3.e2.x10.n0") == (
        325985538,
        3797910565,
        68039873,
        249781638,
    )


def test_a_run_of_m1_writes_its_lead_and_its_amplitude() -> None:
    assert (
        run_identifier("target", "p3", 7, 40, 0, lead_s=60, amplitude="a5")
        == "target.p3.e7.x40.l60.a5.n0"
    )
    assert (
        run_identifier("target", "p3", 7, 40, 0, lead_s=np.int64(60), amplitude="a10")
        == "target.p3.e7.x40.l60.a10.n0"
    )
    # the definition pairs the runs of two plants as before
    assert run_definition("target.p3.e7.x40.l60.a5.n0") == "p3.e7.x40.l60.a5.n0"


def test_the_two_forms_and_the_two_amplitudes_are_different_runs() -> None:
    ids = [
        run_identifier("target", "p3", 7, 40, 0),
        run_identifier("target", "p3", 7, 40, 0, lead_s=60, amplitude="a10"),
        run_identifier("target", "p3", 7, 40, 0, lead_s=60, amplitude="a5"),
        run_identifier("target", "p3", 7, 40, 0, lead_s=66, amplitude="a5"),
    ]
    assert len(set(ids)) == 4
    assert len(set(require_distinct_runs(ids).values())) == 4


@pytest.mark.parametrize(
    ("lead_s", "amplitude", "message"),
    [
        (60, None, "given together or not at all"),
        (None, "a5", "given together or not at all"),
        (0, "a10", "lead_s must be at least 1"),
        (0, "a5", "lead_s must be at least 1"),
        (-60, "a5", "lead_s must be at least 1"),
        (60.0, "a5", "lead_s must be an integer"),
        (True, "a5", "lead_s must be an integer"),
        ("60", "a5", "lead_s must be an integer"),
        (60, "A5", "amplitude must be one of"),
        (60, "a7", "amplitude must be one of"),
        (60, "5", "amplitude must be one of"),
        (60, "", "amplitude must be one of"),
    ],
)
def test_a_lead_or_an_amplitude_outside_its_domain_has_no_identity(
    lead_s: object, amplitude: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_identifier("target", "p3", 7, 40, 0, lead_s=lead_s, amplitude=amplitude)  # type: ignore[arg-type]


def test_an_identity_too_long_for_a_path_is_refused() -> None:
    plant = "p" * (MAX_IDENTIFIER_LENGTH - len(".p3.e7.x40.n0"))
    assert len(run_identifier(plant, "p3", 7, 40, 0)) == MAX_IDENTIFIER_LENGTH
    with pytest.raises(ValueError, match="longer than"):
        run_identifier(plant, "p3", 7, 40, 0, lead_s=60, amplitude="a5")


def test_the_amplitudes_of_the_identities_are_those_of_the_protocol() -> None:
    assert P3_AMPLITUDES == protocols.P3_AMPLITUDES
