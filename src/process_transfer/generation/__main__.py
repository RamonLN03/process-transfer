"""Generate a data set from its definition, all the way to a database and an export.

    python -m process_transfer.generation configs/datasets/m0_e05.yaml

From an IDE, run the module ``process_transfer.generation`` with the path of a
definition as its parameter. The working directory does not matter: a relative path is
looked for under the repository root, and generated files go under ``PT_DATA_DIR``.

The exit code is 0 only if every mandatory check passed. It is 1 when a check failed, and
the report, written to the private record of the attempt whatever happens, says which.
Any other failure ends the program with its error and a non-zero code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from process_transfer.data.paths import repository_root
from process_transfer.generation.pipeline import run_pipeline


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m process_transfer.generation",
        description="Generate, store, ingest, check and export one data set of the virtual plants.",
    )
    parser.add_argument("definition", type=Path, help="YAML definition of the data set")
    parser.add_argument("--no-figures", action="store_true", help="skip the basic figures")
    options = parser.parse_args(arguments)

    definition = options.definition
    if not definition.is_absolute() and not definition.is_file():
        definition = repository_root() / definition
    if not definition.is_file():
        print(f"no definition file at {options.definition}", file=sys.stderr)
        return 2

    report = run_pipeline(definition, figures=not options.no_figures)
    print(f"data set {report['dataset_id']}: {report['dataset']['status']}")  # type: ignore[index]
    for name, seconds in report["stages_s"].items():  # type: ignore[union-attr]
        print(f"  {name:52s} {seconds:8.3f} s")
    for name, passed in report["checks"].items():  # type: ignore[union-attr]
        print(f"  {'pass' if passed else 'FAIL'}  {name}")
    print(f"report and private record: {report['private_record']}")
    if not report["ok"]:
        print("at least one mandatory check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
