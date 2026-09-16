"""ProcessTransfer: physics-aware transfer of hybrid process models between plants.

Subpackages are added milestone by milestone and kept as distinct concerns
(simulation truth, modeller physics, data infrastructure, and later models,
transfer and evaluation). See ``docs/architecture.md`` and ``AGENTS.md``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("process-transfer")
except PackageNotFoundError:  # running from a checkout without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
