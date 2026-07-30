"""Compatibility entry point for the retired 41-taxel serial reader.

The supported runtime uses the 4-taxel BLE backend. This wrapper keeps the old
``python tactile.py`` diagnostic command and legacy imports available during the
v2 migration without making deprecated code part of the installed package.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
import warnings
from pathlib import Path

_TARGET = (
    Path(__file__).resolve().parent
    / "deprecated"
    / "tactile"
    / "magtouch_ilias_41taxel.py"
)
_WARNING = (
    "The 41-taxel serial tactile reader is deprecated and unsupported; "
    "select the 4-taxel BLE backend for supported runtime use."
)


def _load_legacy_module():
    warnings.warn(_WARNING, DeprecationWarning, stacklevel=2)
    module_name = "_airo_doffy_deprecated_magtouch_ilias_41taxel"
    spec = importlib.util.spec_from_file_location(module_name, _TARGET)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy tactile reader from {_TARGET}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    warnings.warn(_WARNING, DeprecationWarning, stacklevel=1)
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    _legacy = _load_legacy_module()
    MagtouchIliasSerialReader = _legacy.MagtouchIliasSerialReader
    MagtouchIliasSerialReaderConfig = _legacy.MagtouchIliasSerialReaderConfig
    __all__ = [
        "MagtouchIliasSerialReader",
        "MagtouchIliasSerialReaderConfig",
    ]
