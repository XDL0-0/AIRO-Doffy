"""Public API for the classic robot teleoperation runtime.

The implementation is split into ``doffy_teleop.robots.legacy`` mixins so
callers share a stable class and patchable backend factory.
"""

from __future__ import annotations

import gc
import threading
import time

import cv2
import numpy as np
import doffy_teleop.utils as utils

from doffy_teleop.robots.legacy.runtime import RobotTeleop as _RobotTeleop
from doffy_teleop.config import Config
from doffy_teleop.sensors.force_filter import WrenchFilter
from doffy_teleop.robots.backend_api import FastRobotiq2F85 as _FastRobotiq2F85
from doffy_teleop.robots.backend_api import make_robot as _make_robot
from doffy_teleop.robots.backend_api import make_robot_backend
from airo_spatial_algebra.se3 import SE3Container
from doffy_teleop.visualization.config import VisualizerConfig


class FastRobotiq2F85(_FastRobotiq2F85):
    """Backward-compatible export for replay scripts."""


def make_robot(ur_ip: str, robot_type: str, torque_mode: bool, initial_joint=None, ruckig_params=None):
    """Backward-compatible export for replay scripts."""
    return _make_robot(ur_ip, robot_type, torque_mode, initial_joint, ruckig_params)


class RobotTeleop(_RobotTeleop):
    """Historical class with a patchable backend factory."""

    def __init__(
        self,
        initial_data: list[dict],
        *,
        cfg: Config | None = None,
        visualizer_config: VisualizerConfig | None = None,
    ):
        super().__init__(
            initial_data,
            cfg=cfg,
            visualizer_config=visualizer_config,
            backend_factory=make_robot_backend,
        )


URTeleop = RobotTeleop

__all__ = [
    "Config",
    "FastRobotiq2F85",
    "RobotTeleop",
    "SE3Container",
    "URTeleop",
    "VisualizerConfig",
    "WrenchFilter",
    "make_robot",
    "make_robot_backend",
]
