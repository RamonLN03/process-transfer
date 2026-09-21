"""``Observations``: what modelling is given, and nothing else."""

import dataclasses

import numpy as np
import pytest

from process_transfer.measurement.observations import Observations

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
