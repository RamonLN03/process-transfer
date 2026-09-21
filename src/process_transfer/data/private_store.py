"""What is needed to regenerate and to diagnose a data set, kept apart from it.

    PT_DATA_DIR/private/datasets/<dataset_id>/<attempt>/
        provenance.json     code, configurations, seeds, noise streams, numerical
                            settings, library versions, schema version, checks of the truth
        configs/            the configuration files used, in full, hidden physics included

One directory per generation attempt, never reused, as for the runs of the experiments.
A second attempt at the same data set adds a second record next to the first; that the
two report the same content hashes is the evidence that the data set is reproducible.

Nothing here is read by ``parquet_store``, by the database or by the export, which work
when this branch does not exist. It is a sibling of ``available/``, not a directory
inside a data set, so that no listing of a data set can reach it. It is not protected by
the operating system: whoever can read the disk can read it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from process_transfer.data.identifiers import path_identifier
from process_transfer.data.paths import data_dir
from process_transfer.data.provenance import copy_with_fingerprints, new_directory


def private_root() -> Path:
    """``PT_DATA_DIR/private/datasets``. It is not created here."""
    return data_dir() / "private" / "datasets"


def write_private_attempt(
    dataset_id: str,
    provenance: Mapping[str, object],
    configuration_files: Sequence[Path],
    state: dict[str, object] | None = None,
) -> Path:
    """Record one generation attempt. ``provenance`` must be serialisable as JSON; the
    configuration files are copied in full and fingerprinted. Returns the new directory."""
    parent = private_root() / path_identifier("dataset_id", dataset_id)
    try:
        text = json.dumps(dict(provenance), indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the provenance is not serialisable as strict JSON: {error}") from None
    directory = new_directory(parent, state)
    fingerprints = copy_with_fingerprints(list(configuration_files), directory / "configs")
    record = json.loads(text)
    record["configurations"] = fingerprints
    record["attempt_directory"] = directory.name
    (directory / "provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return directory
