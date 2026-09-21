"""The ground-truth boundary of AGENTS.md, enforced on the import graph.

Simulation truth and the modeller's physics must stay distinct: neither package
may import the other, and the shared variable definitions import no physics.
The check parses the source, so an import hidden inside a function is caught too.
"""

import ast
from pathlib import Path

import process_transfer

PACKAGE_ROOT = Path(process_transfer.__file__).parent


def _imports(py_file: Path) -> tuple[set[str], bool]:
    """Absolute module names imported by a file, and whether it uses relative imports."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    modules: set[str] = set()
    has_relative = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                has_relative = True
            elif node.module:
                modules.add(node.module)
    return modules, has_relative


def _assert_never_imports(package: str, forbidden_prefix: str) -> None:
    files = sorted((PACKAGE_ROOT / package).rglob("*.py"))
    assert files, f"no source files found under {package}"
    for py_file in files:
        modules, has_relative = _imports(py_file)
        assert not has_relative, f"{py_file} uses relative imports, which this check cannot follow"
        offending = {m for m in modules if m.startswith(forbidden_prefix)}
        assert not offending, f"{py_file} imports {sorted(offending)}"


def test_modeller_never_imports_simulation_truth() -> None:
    _assert_never_imports("modeller", "process_transfer.simulation")


def test_simulation_never_imports_the_modeller() -> None:
    _assert_never_imports("simulation", "process_transfer.modeller")


def test_measurement_never_imports_simulation_truth_or_the_modeller() -> None:
    """A sensor is handed the values it measures. It has no need of the rate law, of a
    true parameter or of the simulator, and models will read ``Observations`` from this
    package, so an import of the truth here would carry it to them."""
    _assert_never_imports("measurement", "process_transfer.simulation")
    _assert_never_imports("measurement", "process_transfer.modeller")


def test_shared_variable_definitions_import_no_physics() -> None:
    modules, has_relative = _imports(PACKAGE_ROOT / "cstr_variables.py")
    assert not has_relative
    assert not {m for m in modules if m.startswith("process_transfer.simulation")}
    assert not {m for m in modules if m.startswith("process_transfer.modeller")}
