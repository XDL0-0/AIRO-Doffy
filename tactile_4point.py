"""Compatibility facade for the deprecated BLE4 reader."""

from __future__ import annotations

import importlib
import runpy
import warnings

_LEGACY_MODULE = "airo_doffy.devices.tactile.legacy"
_LEGACY_EXPORTS = {
    "FourPointTactileBleReader",
    "_format_panel_data",
}

__all__ = sorted(_LEGACY_EXPORTS)


def __getattr__(name: str):
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        "tactile_4point is deprecated; use airo_doffy.devices.tactile instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(_LEGACY_MODULE), name)


def __dir__() -> list[str]:
    return sorted((*globals(), *_LEGACY_EXPORTS))


if __name__ == "__main__":
    runpy.run_module(_LEGACY_MODULE, run_name="__main__")
