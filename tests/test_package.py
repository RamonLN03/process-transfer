"""Smoke test: the package imports and reports a version string."""

import process_transfer


def test_package_imports_and_has_version() -> None:
    assert isinstance(process_transfer.__version__, str)
    assert process_transfer.__version__ != ""
