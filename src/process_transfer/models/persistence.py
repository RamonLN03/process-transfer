"""The selected model of a training, written to a file and read back (``docs/m1_plan.md``,
sections 8.5 and 12: the identities of the fitted models are fixed at the technical freeze).

One JSON file holds everything needed to rebuild the model and to say how it was obtained:

* the family, the hidden layers and the penalty of its configuration;
* every weight and bias of its networks, and for a hybrid its mechanistic coordinates;
* the scales computed from F;
* E/R when it was held;
* the known parameters of the plant the model was built for;
* the seed, the settings of the training, the step and the criterion of the selected
  checkpoint, and the keys of the windows of F and V.

Numbers are written as JSON numbers, which Python writes with the shortest representation
that reads back to the same double, so nothing is rounded. The file carries the SHA-256 of
its own content, the canonical JSON of everything but the hash, sorted keys and no spaces.
Reading it checks the hash and the format, and rebuilds the model through
``learned.LearnedModel``, which refuses parameters outside the domain of the family. A
model read back predicts what the model written predicted, bit for bit.

This is the record of one fitted model, not a registry: where the files go and how they
are named is the business of the script that writes them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from process_transfer.evaluation.plant import KnownPlant
from process_transfer.models import learned
from process_transfer.models.rollout import RolloutSettings
from process_transfer.models.training import Configuration, TrainingRecord, TrainingSettings

FORMAT = "process-transfer learned model 1"


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _arrays(parameters: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, value in parameters.items():
        if name == "theta":
            found[name] = np.asarray(value, dtype=np.float64).tolist()
        else:
            found[name] = [
                [np.asarray(w, dtype=np.float64).tolist(), np.asarray(b, dtype=np.float64).tolist()]
                for w, b in value
            ]
    return found


def _parameters(stored: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, value in stored.items():
        if name == "theta":
            found[name] = np.array(value, dtype=np.float64)
        else:
            found[name] = [
                (np.array(w, dtype=np.float64), np.array(b, dtype=np.float64)) for w, b in value
            ]
    return found


def save_selected(record: TrainingRecord, known: KnownPlant, path: Path) -> str:
    """Write the selected model of ``record`` to ``path``, which must not exist, and return
    the SHA-256 of its content."""
    if record.parameters is None or record.scales is None or record.selected is None:
        raise ValueError(f"{record.configuration.label} has no selected model: {record.failure}")
    path = Path(path)
    if path.exists():
        raise ValueError(f"{path} exists; a written model is never overwritten")
    checkpoint = record.checkpoints[record.selected]
    payload = {
        "format": FORMAT,
        "configuration": dataclasses.asdict(record.configuration),
        "parameters": _arrays(record.parameters),
        "scales": {
            f.name: getattr(record.scales, f.name).tolist()
            for f in dataclasses.fields(record.scales)
        },
        "fixed_activation": record.fixed_activation,
        "known_plant": dataclasses.asdict(known),
        "seed": record.seed,
        "settings": dataclasses.asdict(record.settings),
        "selected_step": checkpoint.step,
        "criterion": checkpoint.validation,
        "fitting_windows": [list(k) for k in record.fitting_windows],
        "validation_windows": [list(k) for k in record.validation_windows],
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    path.write_text(
        json.dumps({"sha256": digest, "content": payload}, indent=1, allow_nan=False),
        encoding="utf-8",
    )
    return digest


@dataclasses.dataclass(frozen=True)
class SavedModel:
    """A model read back, and how it was obtained."""

    model: learned.LearnedModel
    configuration: Configuration
    settings: TrainingSettings
    seed: int
    selected_step: int
    criterion: float
    fitting_windows: tuple[tuple[str, int], ...]
    validation_windows: tuple[tuple[str, int], ...]
    sha256: str


def load_selected(path: Path) -> SavedModel:
    """Read a model written by ``save_selected``, after checking its hash and format."""
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(stored) != {"sha256", "content"}:
        raise ValueError(f"{path} is not a written model: its keys are {sorted(stored)}")
    payload = stored["content"]
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != stored["sha256"]:
        raise ValueError(
            f"{path} does not match its hash: {digest} against the {stored['sha256']} written"
        )
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} has format {payload.get('format')!r}, not {FORMAT!r}")
    configuration = payload["configuration"]
    configuration = Configuration(
        configuration["family"], tuple(configuration["hidden"]), configuration["penalty"]
    )
    settings = dict(payload["settings"])
    settings["reference"] = RolloutSettings(**settings["reference"])
    known = dict(payload["known_plant"])
    known["nominal_inputs"] = tuple(known["nominal_inputs"])
    scales = learned.Scales(
        **{k: np.array(v, dtype=np.float64) for k, v in payload["scales"].items()}
    )
    model = learned.LearnedModel(
        configuration.family,
        configuration.family,
        _parameters(payload["parameters"]),
        KnownPlant(**known),
        scales,
        payload["fixed_activation"],
    )
    return SavedModel(
        model=model,
        configuration=configuration,
        settings=TrainingSettings(**settings),
        seed=payload["seed"],
        selected_step=payload["selected_step"],
        criterion=payload["criterion"],
        fitting_windows=tuple(tuple(k) for k in payload["fitting_windows"]),
        validation_windows=tuple(tuple(k) for k in payload["validation_windows"]),
        sha256=digest,
    )
