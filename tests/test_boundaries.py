"""The ground-truth boundary of AGENTS.md, enforced on the import graph.

Simulation truth and the modeller's physics must stay distinct: neither package
may import the other, and the shared variable definitions import no physics.
The check parses the source, so an import hidden inside a function is caught too.
"""

import ast
from pathlib import Path

import pytest

import process_transfer

PACKAGE_ROOT = Path(process_transfer.__file__).parent


def _imports(py_file: Path) -> tuple[set[str], bool]:
    """Absolute module names imported by a file, and whether it uses relative imports.

    ``from package import name`` counts as an import of ``package.name`` too, since the name
    may be a module (``from process_transfer import simulation``), and a string given to
    ``importlib.import_module`` or ``__import__`` counts as the module it names."""
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
                modules.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            called = node.func
            name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
            if name in ("import_module", "__import__") and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    modules.add(argument.value)
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


def test_the_data_layer_never_imports_simulation_truth_or_the_modeller() -> None:
    """Storage, database and export are on the available side. They are handed
    ``Observations`` and the known specification of a plant; an import of the simulation
    would put the hidden physics within reach of what writes the files a model reads."""
    _assert_never_imports("data", "process_transfer.simulation")
    _assert_never_imports("data", "process_transfer.modeller")


# What reads a plant configuration file, which holds the hidden physics, or the private
# branch of the data, which holds the seeds of the noise.
_TRUTH_READERS = {
    "load_true_plant",
    "TruePlantConfig",
    "TruePhysics",
    "PlantSpec",
    "from_config",
    "private_store",
    "load_dataset_definition",
}


def _names_used(py_file: Path) -> set[str]:
    """Every name, attribute and imported name that a file mentions."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom | ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
    return names


def _assert_reads_only_available_information(package: str) -> None:
    for prefix in (
        "process_transfer.simulation",
        "process_transfer.generation",
        "process_transfer.data.private_store",
    ):
        _assert_never_imports(package, prefix)
    for py_file in sorted((PACKAGE_ROOT / package).rglob("*.py")):
        found = _names_used(py_file) & _TRUTH_READERS
        assert not found, f"{py_file} mentions {sorted(found)}"


def test_the_evaluation_reads_only_available_information() -> None:
    """The evaluation contract of M1 reads exports, their known parameters and readings. It
    never imports the simulation, the generator or the private branch, and never mentions
    what reads a plant configuration file (docs/m1_plan.md, sections 5.6 and 12)."""
    _assert_reads_only_available_information("evaluation")


def test_the_models_read_only_available_information() -> None:
    """The models of M1 are built from the known parameters of an export and the modeller's
    values, and fitted on the windows they are handed. They never import the simulation,
    the generator or the private branch, and never mention what reads a plant configuration
    file (docs/m1_plan.md, sections 5.6 and 12)."""
    _assert_reads_only_available_information("models")


def test_the_check_of_names_sees_what_it_is_meant_to_see(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from process_transfer.config import load_true_plant\n"
        "import process_transfer.data.private_store as store\n"
        "x = config.PlantSpec\n",
        encoding="utf-8",
    )
    assert {"load_true_plant", "private_store", "PlantSpec"} <= _names_used(probe)


def test_the_neutral_modules_import_no_physics() -> None:
    for module in ("canonical.py", "sampling_clock.py", "validation.py", "units.py"):
        modules, has_relative = _imports(PACKAGE_ROOT / module)
        assert not has_relative
        assert not {m for m in modules if m.startswith("process_transfer.simulation")}, module
        assert not {m for m in modules if m.startswith("process_transfer.modeller")}, module


def test_shared_variable_definitions_import_no_physics() -> None:
    modules, has_relative = _imports(PACKAGE_ROOT / "cstr_variables.py")
    assert not has_relative
    assert not {m for m in modules if m.startswith("process_transfer.simulation")}
    assert not {m for m in modules if m.startswith("process_transfer.modeller")}


def test_the_import_check_sees_a_module_imported_by_name(tmp_path: Path) -> None:
    """The check would miss nothing that names a module of the truth: an import of the
    package's module by name, or a string given to importlib."""
    probe = tmp_path / "probe.py"
    lines = [
        "import importlib",
        "from process_transfer import simulation",
        "from process_transfer import generation as g",
        "importlib.import_module('process_transfer.data.private_store')",
    ]
    probe.write_text("\n".join(lines) + "\n", encoding="utf-8")
    modules, _ = _imports(probe)
    assert {
        "process_transfer.simulation",
        "process_transfer.generation",
        "process_transfer.data.private_store",
    } <= modules


EXPERIMENTS = PACKAGE_ROOT.parents[1] / "experiments"


def test_the_available_parts_of_m1_e01_read_only_available_information() -> None:
    """Parts 1 and 2 of M1-E01 use available information only (docs/m1_plan.md, section
    7.4): the script never imports the simulation, the generator or the private branch, and
    never mentions what reads a plant configuration file. Part 3, the oracle, is another
    script."""
    script = EXPERIMENTS / "m1_e01_identifiability.py"
    modules, has_relative = _imports(script)
    assert not has_relative
    for prefix in (
        "process_transfer.simulation",
        "process_transfer.generation",
        "process_transfer.data.private_store",
    ):
        assert not {m for m in modules if m.startswith(prefix)}, prefix
    found = _names_used(script) & _TRUTH_READERS
    assert not found, f"{script} mentions {sorted(found)}"
    assert "m1_e01_oracle" not in script.read_text(encoding="utf-8").replace(
        "experiments/m1_e01_oracle.py", ""
    )


@pytest.mark.parametrize("name", ["m1_i3_framework_comparison.py", "m1_i3_pilot.py"])
def test_the_development_scripts_of_i3_read_only_available_information(name: str) -> None:
    """The comparison of frameworks and the pilot of I3 read the exports of M0 and simulate
    their synthetic runs with the model side's rollout: they never import the simulation,
    the generator or the private branch, and never mention what reads a plant
    configuration file."""
    script = EXPERIMENTS / name
    modules, has_relative = _imports(script)
    assert not has_relative
    for prefix in (
        "process_transfer.simulation",
        "process_transfer.generation",
        "process_transfer.data.private_store",
    ):
        assert not {m for m in modules if m.startswith(prefix)}, prefix
    found = _names_used(script) & _TRUTH_READERS
    assert not found, f"{script} mentions {sorted(found)}"
