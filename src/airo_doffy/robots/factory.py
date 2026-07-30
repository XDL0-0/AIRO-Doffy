"""Composition helpers selecting unstarted robot and gripper adapters."""

from __future__ import annotations

from ..config.models import RobotConfig
from ..core.errors import ModelValidationError
from .base import RobotBackend
from .grippers.base import Gripper
from .grippers.mock import NullGripper
from .grippers.robotiq_2f85 import create_robotiq_2f85
from .realman import create_realman_backend
from .ur import create_ur_backend


def create_robot_backend(config: RobotConfig) -> RobotBackend:
    """Select an unstarted vendor adapter from typed robot values."""

    if config.robot_type in {"ur3e", "ur5e"}:
        return create_ur_backend(config)
    if config.robot_type == "realman":
        return create_realman_backend(config)
    raise ModelValidationError(f"unsupported robot_type: {config.robot_type!r}")


def create_gripper(config: RobotConfig) -> Gripper:
    """Select a disabled in-memory gripper or an unstarted Robotiq adapter."""

    if not config.gripper_enabled:
        return NullGripper(max_width_m=config.gripper_max_width_m)
    if config.robot_type not in {"ur3e", "ur5e"}:
        raise ModelValidationError("Robotiq gripper is supported only with UR robots")
    return create_robotiq_2f85(config)
