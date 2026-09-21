"""Look for hidden information in what was written to the available branch.

The separation of truth and observation is built into the interfaces: the writers are
never handed the truth. This scan is the check from the other end. It reads the files a
model would read, data sets, database and exports, the way a model would, and looks for
what must not be there:

* names: a column, a key of a JSON document or a metadata entry that speaks of a seed, a
  noise stream, an exact or true value, an error, the hidden physics or a diagnostic of
  the truth;
* numbers: the value of a hidden parameter, the nominal steady state, any exact state at
  a sensor instant, the master seed of the noise or a word of a noise stream;
* files: anything that is not Parquet, JSON or the database.

It knows the hidden values, so it lives on the truth side. It is a check against
accidents, not a proof: a hidden value written in a transformed way, scaled or rounded,
would not be found by comparing numbers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FORBIDDEN_IN_NAMES = (
    "seed",
    "stream",
    "true",
    "truth",
    "exact",
    "error",
    "hidden",
    "k0",
    "saturation",
    "activation",
    "ua_ref",
    "alpha",
    "t_ref",
    "steady",
    "initial_state",
    "peak",
    "residual",
    "derivative",
)
FORBIDDEN_IN_TEXT = ("true_physics", "sensor_master_seed", "saturation_constant", "UA_ref")
RELATIVE_TOLERANCE = 1.0e-12  # a parameter may have gone through a unit conversion and back


@dataclass(frozen=True)
class HiddenValues:
    """What must not be found. ``parameters`` are compared to a relative 1e-12; ``states``
    are compared bit for bit, there being thousands of them; ``integers`` exactly."""

    parameters: dict[str, float]
    states: np.ndarray
    integers: dict[str, int] = field(default_factory=dict)


def _name_findings(where: str, names: Iterable[str]) -> list[str]:
    findings = []
    for name in names:
        lowered = str(name).lower()
        for word in FORBIDDEN_IN_NAMES:
            if word in lowered:
                findings.append(f"{where}: the name {name!r} contains {word!r}")
    return findings


def _float_findings(where: str, values: np.ndarray, hidden: HiddenValues) -> list[str]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    findings = []
    if values.size == 0:
        return findings
    for label, secret in hidden.parameters.items():
        if np.any(np.isclose(values, secret, rtol=RELATIVE_TOLERANCE, atol=0.0)):
            findings.append(f"{where}: holds the hidden value of {label}, {secret!r}")
    leaked = int(np.count_nonzero(np.isin(values, hidden.states)))
    if leaked:
        findings.append(f"{where}: {leaked} value(s) equal an exact state bit for bit")
    return findings


def _integer_findings(where: str, values: Iterable[int], hidden: HiddenValues) -> list[str]:
    present = set(int(value) for value in values)
    return [
        f"{where}: holds {label}, {secret}"
        for label, secret in hidden.integers.items()
        if secret in present
    ]


def _scan_json(where: str, document: object, hidden: HiddenValues) -> list[str]:
    findings: list[str] = []
    floats: list[float] = []
    integers: list[int] = []

    def walk(node: object, trail: str) -> None:
        if isinstance(node, dict):
            findings.extend(_name_findings(f"{where} {trail}", node.keys()))
            for key, value in node.items():
                walk(value, f"{trail}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, int):
            integers.append(node)
        elif isinstance(node, float):
            floats.append(node)
        elif isinstance(node, str):
            for word in FORBIDDEN_IN_TEXT:
                if word.lower() in node.lower():
                    findings.append(f"{where} {trail}: the text mentions {word!r}")

    walk(document, "")
    findings += _float_findings(where, np.array(floats, dtype=np.float64), hidden)
    findings += _integer_findings(where, integers, hidden)
    return findings


def _scan_columns(where: str, columns: dict[str, np.ndarray], hidden: HiddenValues) -> list[str]:
    """Names and values of the columns of one relation, whatever it was read from."""
    findings = _name_findings(where, columns)
    for name, values in columns.items():
        values = np.ma.getdata(values)[~np.ma.getmaskarray(values)]  # nulls hold no value
        if values.dtype.kind == "f":
            findings += _float_findings(f"{where}.{name}", values, hidden)
        elif values.dtype.kind in "iu":
            findings += _integer_findings(f"{where}.{name}", values.tolist(), hidden)
        elif values.dtype.kind in "OUS":
            for text in {str(value) for value in values.tolist() if value is not None}:
                findings += [
                    f"{where}.{name}: the text mentions {word!r}"
                    for word in FORBIDDEN_IN_TEXT
                    if word.lower() in text.lower()
                ]
    return findings


def _scan_parquet(path: Path, hidden: HiddenValues) -> list[str]:
    table = pq.read_table(path)
    columns = {
        name: np.ma.masked_invalid(table[name].to_numpy(zero_copy_only=False))
        if pa.types.is_floating(table[name].type)
        else np.asarray(
            table[name].to_pylist(), dtype=object if pa.types.is_string(table[name].type) else None
        )
        for name in table.schema.names
    }
    findings = _scan_columns(path.name, columns, hidden)
    entries: dict[str, str] = {}
    for key, value in (pq.read_metadata(path).metadata or {}).items():
        entries[key.decode()] = value.decode("utf-8", "replace")
    for column in table.schema:
        for key, value in (column.metadata or {}).items():
            entries[f"{column.name}:{key.decode()}"] = value.decode("utf-8", "replace")
    findings += _name_findings(f"{path.name} metadata", (k for k in entries if k != "ARROW:schema"))
    for key, value in entries.items():
        if key == "ARROW:schema":
            continue  # the serialised schema, already examined field by field above
        findings += [
            f"{path.name} metadata {key}: the text mentions {word!r}"
            for word in FORBIDDEN_IN_TEXT
            if word.lower() in value.lower()
        ]
    return findings


def _scan_database(path: Path, hidden: HiddenValues, allowed: Sequence[str]) -> list[str]:
    findings: list[str] = []
    connection = duckdb.connect(str(path), read_only=True)
    try:
        relations = connection.execute(
            "SELECT table_schema, table_name FROM information_schema.tables ORDER BY ALL"
        ).fetchall()
        for schema_name, table_name in relations:
            where = f"{path.name}:{schema_name}.{table_name}"
            if table_name not in allowed:
                findings.append(f"{where}: a relation that the contract does not have")
                continue
            columns = connection.execute(
                f'SELECT * FROM "{schema_name}"."{table_name}"'
            ).fetchnumpy()
            findings += _scan_columns(where, columns, hidden)
    finally:
        connection.close()
    return findings


def scan_available(
    roots: Sequence[Path],
    hidden: HiddenValues,
    allowed_relations: Sequence[str],
) -> dict[str, object]:
    """Scan every file under ``roots``. Returns what was read and what was found; an
    empty list of findings is a pass. Files are listed and opened one by one."""
    for label, value in hidden.parameters.items():
        if not math.isfinite(value) or value == 0.0:
            raise ValueError(f"the hidden value of {label} is {value!r}; it cannot be searched for")
    findings: list[str] = []
    files: list[str] = []
    for root in roots:
        root = Path(root)
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        if not paths:
            raise ValueError(f"there is nothing to scan under {root}")
        for path in paths:
            files.append(str(path))
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                findings += _scan_parquet(path, hidden)
            elif suffix == ".json":
                document = json.loads(path.read_text(encoding="utf-8"))
                findings += _scan_json(path.name, document, hidden)
            elif suffix == ".duckdb":
                findings += _scan_database(path, hidden, allowed_relations)
            else:
                findings.append(f"{path}: a file of a kind that the available branch does not hold")
    return {"files_scanned": files, "findings": findings}
