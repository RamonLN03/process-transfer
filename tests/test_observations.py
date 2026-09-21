"""``Observations``: what modelling is given, and nothing else."""

import dataclasses
import hashlib
import struct

import numpy as np
import pytest

from process_transfer.measurement.observations import CONTENT_ENCODING, Observations

# The closed list of what an observation set may hold. Adding a field means editing
# this test, which is the point: nothing of the truth gets in unnoticed.
ALLOWED_FIELDS = {
    "plant",
    "run",
    "times",
    "measured",
    "inputs",
    "measured_names",
    "measured_units",
    "input_names",
    "input_units",
    "sample_period",
    "noise_std",
}
FORBIDDEN_WORDS = ("seed", "stream", "exact", "true", "truth", "error", "parameter", "hidden")


def build(**changes: object) -> Observations:
    n = 5
    arguments = dict(
        plant="target",
        run="p3-0",
        times=6.0 * np.arange(n),
        measured=np.column_stack([np.linspace(180.0, 200.0, n), np.linspace(350.0, 356.0, n)]),
        inputs=np.tile([1.7e-3, 500.0, 350.0, 337.5], (n, 1)),
        measured_names=("C_A", "T"),
        measured_units=("mol/m^3", "K"),
        input_names=("q", "C_Af", "T_f", "T_c"),
        input_units=("m^3/s", "mol/m^3", "K", "K"),
        sample_period=6.0,
        noise_std=(5.0, 0.5),
    )
    return Observations(**{**arguments, **changes})


def test_the_fields_are_a_closed_list_without_any_truth() -> None:
    names = {field.name for field in dataclasses.fields(Observations)}
    assert names == ALLOWED_FIELDS
    for name in names:
        assert not any(word in name.lower() for word in FORBIDDEN_WORDS), name
    public = {name for name in vars(Observations) if not name.startswith("_")}
    for name in public:
        assert not any(word in name.lower() for word in FORBIDDEN_WORDS), name


def test_a_valid_set_is_kept_as_read_only_copies() -> None:
    measured = np.column_stack([np.linspace(180.0, 200.0, 5), np.linspace(350.0, 356.0, 5)])
    found = build(measured=measured)
    assert found.n_samples == 5
    measured[0, 0] = -1.0  # the caller's array is theirs
    assert found.measured[0, 0] == 180.0
    with pytest.raises(ValueError, match="read-only"):
        found.measured[0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        found.times[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.plant = "source"


def test_readings_are_not_judged_as_true_states() -> None:
    """A negative concentration reading and a reading above the feed are what a noisy
    sensor can give. Neither is rejected nor altered."""
    measured = np.column_stack([[-3.2, 610.0, 190.0, 191.0, 189.0], np.full(5, 355.0)])
    found = build(measured=measured)
    np.testing.assert_array_equal(found.measured, measured)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"times": np.array([0.0, 6.0, 6.0, 18.0, 24.0])}, "strictly increasing"),
        ({"times": np.array([0.0, 12.0, 6.0, 18.0, 24.0])}, "strictly increasing"),
        ({"times": np.array([0.0, 6.0, np.nan, 18.0, 24.0])}, "must be finite"),
        ({"times": np.zeros((5, 1))}, "non-empty vector"),
        ({"times": np.array([])}, "non-empty vector"),
        ({"measured": np.zeros((4, 2))}, "measured must have shape"),
        ({"measured": np.zeros((5, 3))}, "measured must have shape"),
        ({"inputs": np.zeros((5, 3))}, "inputs must have shape"),
        ({"measured": np.full((5, 2), np.inf)}, "must be finite"),
        ({"measured_units": ("K",)}, "2 names against 1 units"),
        ({"noise_std": (5.0,)}, "one value per measured variable"),
        ({"noise_std": (5.0, -0.5)}, "noise_std"),
        ({"sample_period": 0.0}, "sample_period"),
    ],
)
def test_malformed_sets_are_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build(**changes)


def test_the_digest_follows_everything_the_set_holds() -> None:
    reference = build().content_digest()
    assert reference == build().content_digest()
    assert len(reference) == 64

    measured = build().measured.copy()
    measured[2, 1] = np.nextafter(measured[2, 1], np.inf)  # one unit in the last place
    assert build(measured=measured).content_digest() != reference
    assert build(plant="source").content_digest() != reference
    assert build(measured_units=("mol/L", "K")).content_digest() != reference
    assert build(noise_std=(5.0, 0.0)).content_digest() != reference
    assert build(sample_period=60.0).content_digest() != reference


def test_two_sets_of_different_shapes_never_share_a_digest() -> None:
    """Regression. The first digest joined the labels with a separator and appended the
    arrays without their shapes. One row with four inputs and two rows with one input
    had the same digest, their numbers lining up and one label holding the separator.
    Both digests were ee94d10a19179cb9... before the change."""
    one_row = Observations(
        plant="p",
        run="r",
        times=np.array([0.0]),
        measured=np.array([[1.0]]),
        inputs=np.array([[2.0, 3.0, 5.0, 6.0]]),
        measured_names=("m1",),
        measured_units=("K",),
        input_names=("i1", "i2", "i3", "i4"),
        input_units=("K", "K", "K", "K"),
        sample_period=6.0,
        noise_std=(5.0,),
    )
    two_rows = Observations(
        plant="p",
        run="r",
        times=np.array([0.0, 1.0]),
        measured=np.array([[2.0], [3.0]]),
        inputs=np.array([[5.0], [6.0]]),
        measured_names=("m1",),
        measured_units=("K",),
        input_names=("\x1f".join(["i1", "i2", "i3", "i4", "K", "K", "K"]),),
        input_units=("K",),
        sample_period=6.0,
        noise_std=(5.0,),
    )
    assert one_row.content_digest() != two_rows.content_digest()


def test_the_digest_is_the_documented_encoding_byte_for_byte() -> None:
    """``observations/v1`` as written in docs/data_contract.md, spelled out here by hand
    for the smallest set, so that the encoding cannot change without this test saying so."""

    def length(n: int) -> bytes:
        return n.to_bytes(8, "big")

    def string(s: str) -> bytes:
        return b"S" + length(len(s.encode())) + s.encode()

    def strings(*items: str) -> bytes:
        return b"L" + length(len(items)) + b"".join(string(item) for item in items)

    def numbers(shape: tuple[int, ...], *values: float) -> bytes:
        dims = b"".join(length(size) for size in shape)
        return b"A" + length(len(shape)) + dims + struct.pack(f"<{len(values)}d", *values)

    expected = b"".join(
        [
            b"process-transfer/observations/v1\n",
            string("plant") + string("target"),
            string("run") + string("r0"),
            string("measured_names") + strings("T"),
            string("measured_units") + strings("K"),
            string("input_names") + strings("q", "T_c"),
            string("input_units") + strings("m^3/s", "K"),
            string("sample_period") + numbers((1,), 6.0),
            string("noise_std") + numbers((1,), 0.5),
            string("times") + numbers((2,), 0.0, 6.0),
            string("measured") + numbers((2, 1), 355.25, 354.5),
            string("inputs") + numbers((2, 2), 0.0016, 337.5, 0.0017, 342.5),
        ]
    )
    found = Observations(
        plant="target",
        run="r0",
        times=np.array([0.0, 6.0]),
        measured=np.array([[355.25], [354.5]]),
        inputs=np.array([[0.0016, 337.5], [0.0017, 342.5]]),
        measured_names=("T",),
        measured_units=("K",),
        input_names=("q", "T_c"),
        input_units=("m^3/s", "K"),
        sample_period=6.0,
        noise_std=(0.5,),
    )
    assert CONTENT_ENCODING == "observations/v1"
    assert found.content_digest() == hashlib.sha256(expected).hexdigest()


def test_labels_must_be_strings() -> None:
    with pytest.raises(ValueError, match="must be strings"):
        build(measured_names=("C_A", 7))
    with pytest.raises(ValueError, match="must be strings"):
        build(plant=None)
