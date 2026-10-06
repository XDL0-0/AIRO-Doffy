"""Robot backend contracts and shared target semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import doffy_teleop.utils as utils
from .gripper import NullGripper

@dataclass
class CommandResult:
    accepted: bool
    tcp_pose: np.ndarray
    joint_configuration: np.ndarray | None


class RobotBackend:
    name = "robot"
    supports_freedrive = False
    supports_torque_mode = False
    supports_force = False
    is_ur = False

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.tcp_tool = cfg.TCP_TOOL
        self.tcp_transform = np.asarray(cfg.TCP_TRANSFORM, dtype=float)
        self.inv_tcp_transform = np.linalg.inv(self.tcp_transform)
        self.dof = 0
        self.robot = None
        self.ik_solver = None
        self.gripper = NullGripper(cfg.GRIPPER_MAX)
        self.hand = None

    @property
    def dataset_robot_type(self) -> str:
        return self.name

    def to_robot_tcp_pose(self, tool_tcp_pose: np.ndarray) -> np.ndarray:
        return np.asarray(tool_tcp_pose, dtype=float) @ self.inv_tcp_transform

    def to_tool_tcp_pose(self, robot_tcp_pose: np.ndarray) -> np.ndarray:
        return np.asarray(robot_tcp_pose, dtype=float) @ self.tcp_transform

    def initial_joint_configuration(self, configured: np.ndarray) -> np.ndarray:
        configured = np.asarray(configured, dtype=float).reshape(-1)
        if configured.shape == (self.dof,):
            return configured.copy()
        current = self.get_joint_configuration()
        utils.logger.warning(
            f"Configured INITIAL_JOINT has shape {configured.shape}, but {self.name} has "
            f"{self.dof} DoF. Using current joints as initial pose."
        )
        return current

    def get_joint_configuration(self) -> np.ndarray:
        raise NotImplementedError

    def get_tcp_pose(self) -> np.ndarray:
        raise NotImplementedError

    def get_tcp_force(self) -> np.ndarray | None:
        return None

    def solve_tcp_ik(
        self,
        tcp_pose: np.ndarray,
        seed: np.ndarray | None = None,
    ) -> np.ndarray | None:
        raise NotImplementedError

    def is_joint_target_safe(
        self,
        joints: np.ndarray,
        previous_joints: np.ndarray | None,
        tcp_position: np.ndarray | None,
        joint_threshold: np.ndarray,
    ) -> bool:
        if previous_joints is not None and not utils.is_joint_change_safe(previous_joints, joints, joint_threshold):
            return False
        return True

    def clip_joint_configuration(self, joints: np.ndarray) -> np.ndarray:
        return np.asarray(joints, dtype=float)

    def command_joint_configuration(self, joints: np.ndarray, dt: float) -> CommandResult:
        raise NotImplementedError

    def command_tcp_pose(self, tcp_pose: np.ndarray, dt: float) -> CommandResult:
        raise NotImplementedError

    def move_to_joint_configuration(self, joints: np.ndarray, speed: float | None = None):
        raise NotImplementedError

    def reset(self, joints: np.ndarray) -> None:
        action = self.move_to_joint_configuration(joints, self.cfg.RESET_JOINT_SPEED)
        if action is not None and hasattr(action, "wait"):
            action.wait()

    def start_freedrive(self) -> None:
        raise NotImplementedError(f"{self.name} does not support freedrive through this backend.")

    def stop_freedrive(self) -> None:
        raise NotImplementedError(f"{self.name} does not support freedrive through this backend.")

    def set_freedrive_sensitivity(self, grade: int) -> None:
        raise NotImplementedError(
            f"{self.name} does not support freedrive sensitivity through this backend."
        )

    def cleanup(self) -> None:
        close_hand = getattr(self.hand, "close", None)
        if callable(close_hand):
            close_hand()
        close = getattr(self.gripper, "close", None)
        if callable(close):
            close()
