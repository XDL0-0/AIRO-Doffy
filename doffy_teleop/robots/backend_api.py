"""Public imports for the modular robot backend package.

The concrete implementations live in :mod:`doffy_teleop.robots`; this module
provides the backend API used by teleop, inference, evaluation and replay tools.
"""

from doffy_teleop.robots.contracts import CommandResult, RobotBackend
from doffy_teleop.robots.gripper import FastRobotiq2F85, NullGripper
from doffy_teleop.robots.backends import (
    PositionManipulatorBackend,
    RealManBackend,
    URPositionBackend,
    URTorqueBackend,
)
from doffy_teleop.robots.factory import make_robot, make_robot_backend

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
