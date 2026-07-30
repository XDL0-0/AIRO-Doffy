"""AIRO-Doffy public package.

Importing :mod:`airo_doffy` is intentionally side-effect free. Hardware and
optional transports are loaded only by their factories or application entry
points.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("airo-doffy")
except PackageNotFoundError:
    __version__ = "2.0.0.dev0"

__all__ = ["__version__"]

