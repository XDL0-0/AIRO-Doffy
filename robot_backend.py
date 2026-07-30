"""Compatibility facade for robot backends during the v2 migration.

New code should use ``airo_doffy.robots``. The broad legacy backend API remains
available here until the root teleoperation and inference entry points migrate.
"""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "robot_backend is a compatibility module; use airo_doffy.robots for new code",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CommandResult",
    "FastRobotiq2F85",
    "NullGripper",
    "PositionManipulatorBackend",
    "RealManBackend",
    "RobotBackend",
    "URPositionBackend",
    "URTorqueBackend",
    "make_robot",
    "make_robot_backend",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    legacy = importlib.import_module("airo_doffy.robots.legacy")
    value = getattr(legacy, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
