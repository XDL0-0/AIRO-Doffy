"""Helpers for temporary legacy script-path wrappers."""

from __future__ import annotations

import runpy
import warnings


def run_moved_module(target: str, legacy_path: str) -> None:
    """Run a relocated module as ``__main__`` and report its new location."""
    warnings.warn(
        f"{legacy_path} moved to `python -m {target}`; update scripts and documentation.",
        DeprecationWarning,
        stacklevel=2,
    )
    runpy.run_module(target, run_name="__main__", alter_sys=True)


def block_unsafe_legacy_command(target: str, legacy_path: str) -> None:
    """Refuse a legacy command whose hard-coded defaults mutate external state."""
    raise SystemExit(
        f"{legacy_path} was retired because it performed an external-state "
        f"operation with hard-coded values. Use `python -m {target} --help` and explicitly "
        "confirm the operation."
    )
