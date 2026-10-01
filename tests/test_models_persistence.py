"""The selected model of a training written to a file and read back: the same predictions,
the record of how it was obtained, and a refusal of a file that was changed."""

import json

import numpy as np
import pytest

from m1_support import NOMINAL, modeller_run, random_corners
from process_transfer.evaluation.plant import KnownPlant
from process_transfer.evaluation.windows import P3_LAYOUT, find_windows, window_data
from process_transfer.models.mechanistic import MechanisticModel, MechanisticParameters
from process_transfer.models.persistence import load_selected, save_selected
from process_transfer.models.rollout import predict_window
from process_transfer.models.training import Configuration, TrainingSettings, train

KNOWN = KnownPlant("target", 0.1, 1000.0, 239.0, -50000.0, tuple(NOMINAL))
TRUE = MechanisticParameters.from_k_350(0.0177, 9000.0, 1330.0)
SETTINGS = TrainingSettings(learning_rate=3e-3, max_steps=20, validation_every=10)
RUN = modeller_run(MechanisticModel("m", KNOWN, TRUE).rhs, random_corners(4, 5), noise_seed=11)
DATA = window_data(RUN, find_windows(RUN, NOMINAL, P3_LAYOUT).windows)


@pytest.mark.parametrize(
    ("configuration", "start", "fixed"),
    [
        (Configuration("HKU", (4,), 1e-4), TRUE, None),
        (
            Configuration("HK", (4,), 1e-4),
            MechanisticParameters.from_k_350(0.0177, 9000.0, 1330.0),
            9000.0,
        ),
        (Configuration("BN", (4, 4), 1e-3), None, None),
    ],
)
def test_a_model_read_back_predicts_what_it_predicted(
    configuration, start, fixed, tmp_path
) -> None:  # noqa: ANN001
    record = train(configuration, SETTINGS, DATA[:3], DATA[3:], KNOWN, 5, start, fixed)
    path = tmp_path / "model.json"
    digest = save_selected(record, KNOWN, path)
    saved = load_selected(path)
    assert saved.sha256 == digest
    assert saved.configuration == record.configuration and saved.settings == record.settings
    assert saved.seed == 5 and saved.criterion == record.criterion
    assert saved.selected_step == record.checkpoints[record.selected].step
    assert saved.fitting_windows == record.fitting_windows
    original = record.model(KNOWN)
    for data in DATA:
        assert np.array_equal(predict_window(saved.model, data), predict_window(original, data))
    if configuration.family != "BN":
        assert saved.model.mechanistic_parameters() == original.mechanistic_parameters()
    with pytest.raises(ValueError, match="never overwritten"):
        save_selected(record, KNOWN, path)


def test_a_changed_file_is_refused(tmp_path) -> None:  # noqa: ANN001
    record = train(Configuration("HK", (4,), 1e-4), SETTINGS, DATA[:3], DATA[3:], KNOWN, 5, TRUE)
    path = tmp_path / "model.json"
    save_selected(record, KNOWN, path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["content"]["parameters"]["theta"][1] += 1e-12
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its hash"):
        load_selected(path)


def test_a_failed_training_has_no_model_to_write(tmp_path) -> None:  # noqa: ANN001
    start = MechanisticParameters.from_k_350(1e300, 0.0, 1330.0)
    record = train(Configuration("HK", (4,), 0.0), SETTINGS, DATA[:3], DATA[3:], KNOWN, 1, start)
    with pytest.raises(ValueError, match="no selected model"):
        save_selected(record, KNOWN, tmp_path / "model.json")
