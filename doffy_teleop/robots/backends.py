"""UR, RealMan and generic position backend implementations."""

from __future__ import annotations

import time

import numpy as np

import doffy_teleop.utils as utils
from .contracts import CommandResult, RobotBackend
from .gripper import NullGripper

class PositionManipulatorBackend(RobotBackend):
    """Backend for any airo_robots PositionManipulator implementation."""

    def __init__(self, cfg, manipulator, name: str, gripper=None) -> None:
        super().__init__(cfg)
        self.robot = manipulator
        self.name = name
        self.gripper = gripper if gripper is not None else NullGripper(cfg.GRIPPER_MAX)
        self.dof = int(self.robot.manipulator_specs.dof)
        self.supports_force = (
            hasattr(self.robot, "get_tcp_force")
            or hasattr(getattr(self.robot, "rtde_receive", None), "getActualTCPForce")
            or hasattr(getattr(self.robot, "robot", None), "rm_get_force_data")
        )

    def get_joint_configuration(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_configuration(), dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        return self.to_tool_tcp_pose(np.asarray(self.robot.get_tcp_pose(), dtype=float))

    def get_tcp_force(self) -> np.ndarray | None:
        if hasattr(self.robot, "get_tcp_force"):
            return np.asarray(self.robot.get_tcp_force(), dtype=float)
        rtde_receive = getattr(self.robot, "rtde_receive", None)
        if rtde_receive is not None and hasattr(rtde_receive, "getActualTCPForce"):
            return np.asarray(rtde_receive.getActualTCPForce(), dtype=float)
        realman_robot = getattr(self.robot, "robot", None)
        if realman_robot is not None and hasattr(realman_robot, "rm_get_force_data"):
            error_code, data = realman_robot.rm_get_force_data()
            if error_code != 0:
                utils.logger.warning(f"RealMan rm_get_force_data failed with error code {error_code}.")
                return None
            # RealMan exposes several 6D vectors. Prefer zeroed external wrench
            # when available, then fall back to raw sensor data.
            for key in ("zero_force_data", "work_zero_force_data", "tool_zero_force_data", "force_data"):
                values = data.get(key) if isinstance(data, dict) else None
                if values is not None:
                    wrench = np.asarray(values, dtype=float).reshape(-1)
                    if wrench.size >= 6:
                        return wrench[:6]
            utils.logger.warning(f"RealMan force data did not contain a 6D wrench: {data}")
        return None

    def solve_tcp_ik(
        self,
        tcp_pose: np.ndarray,
        seed: np.ndarray | None = None,
    ) -> np.ndarray | None:
        seed = self.get_joint_configuration() if seed is None else np.asarray(seed, dtype=float)
        solution = self.robot.inverse_kinematics(self.to_robot_tcp_pose(tcp_pose), seed)
        if solution is None:
            return None
        solution = np.asarray(solution, dtype=float)
        if solution.shape != (self.dof,):
            return None
        return solution

    def command_joint_configuration(self, joints: np.ndarray, dt: float) -> CommandResult:
        joints = np.asarray(joints, dtype=float)
        self.robot.servo_to_joint_configuration(joints, dt)
        return CommandResult(True, self.get_tcp_pose(), joints)

    def command_tcp_pose(self, tcp_pose: np.ndarray, dt: float) -> CommandResult:
        joint_target = self.solve_tcp_ik(tcp_pose)
        self.robot.servo_to_tcp_pose(self.to_robot_tcp_pose(tcp_pose), dt)
        return CommandResult(True, np.asarray(tcp_pose, dtype=float), joint_target)

    def move_to_joint_configuration(self, joints: np.ndarray, speed: float | None = None):
        return self.robot.move_to_joint_configuration(np.asarray(joints, dtype=float), speed)


class URPositionBackend(PositionManipulatorBackend):
    is_ur = True
    supports_freedrive = True

    def __init__(self, cfg, robot, ik_solver, gripper) -> None:
        super().__init__(cfg, robot, cfg.ROBOT_TYPE, gripper)
        self.ik_solver = ik_solver

    def solve_tcp_ik(
        self,
        tcp_pose: np.ndarray,
        seed: np.ndarray | None = None,
    ) -> np.ndarray | None:
        seed = self.get_joint_configuration() if seed is None else np.asarray(seed, dtype=float)
        solutions = self.ik_solver.inverse_kinematics_closest_with_tcp(
            np.asarray(tcp_pose, dtype=float), self.tcp_transform, *seed
        )
        if not solutions:
            return None
        return np.asarray(solutions[0], dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        joints = self.get_joint_configuration()
        return self.ik_solver.forward_kinematics(*joints) @ self.tcp_transform

    def is_joint_target_safe(
        self,
        joints: np.ndarray,
        previous_joints: np.ndarray | None,
        tcp_position: np.ndarray | None,
        joint_threshold: np.ndarray,
    ) -> bool:
        joints = np.asarray(joints, dtype=float)
        if not utils.is_joint_within_limits(joints):
            utils.logger.warning("No valid IK solution within UR joint limits, keeping previous pose!")
            return False
        if not utils.is_pose_safe(joints, tcp_position, robot_type=self.cfg.ROBOT_TYPE):
            return False
        return super().is_joint_target_safe(joints, previous_joints, tcp_position, joint_threshold)

    def clip_joint_configuration(self, joints: np.ndarray) -> np.ndarray:
        low, high = utils.UR3E_JOINT_LIMITS
        return np.clip(np.asarray(joints, dtype=float), low, high)

    def start_freedrive(self) -> None:
        self.robot.rtde_control.servoStop()
        time.sleep(0.1)
        self.robot.rtde_control.teachMode()

    def stop_freedrive(self) -> None:
        self.robot.rtde_control.endTeachMode()


class RealManBackend(PositionManipulatorBackend):
    """RealMan backend with bounded retries for transient SDK read timeouts."""

    supports_freedrive = True

    def __init__(self, cfg, robot) -> None:
        super().__init__(cfg, robot, "realman", NullGripper(cfg.GRIPPER_MAX))
        self._read_retries = int(getattr(cfg, "REALMAN_READ_RETRIES", 3))
        self._retry_delay = float(getattr(cfg, "REALMAN_RETRY_DELAY", 0.05))
        if self.tcp_tool == "Hand" and getattr(cfg, "BRAINCO_HAND_ENABLE", True):
            self._connect_brainco_hand()

    def _connect_brainco_hand(self) -> None:
        """Recognize and connect a six-DoF BrainCo hand through RM_ARM+."""
        from doffy_teleop.robots.brainco_hand import BrainCoHandDriver

        arm = getattr(self.robot, "robot", None)
        try:
            if arm is None:
                raise RuntimeError("RealMan wrapper does not expose its SDK arm object.")
            self.hand = BrainCoHandDriver(
                arm,
                baudrate=self.cfg.BRAINCO_HAND_BAUDRATE,
                read_retries=self.cfg.BRAINCO_HAND_READ_RETRIES,
                retry_delay=self.cfg.BRAINCO_HAND_RETRY_DELAY,
                mode_settle_delay=self.cfg.BRAINCO_HAND_MODE_SETTLE_DELAY,
                max_send_hz=self.cfg.BRAINCO_HAND_MAX_SEND_HZ,
                filter_cutoff_hz=self.cfg.BRAINCO_HAND_FILTER_CUTOFF_HZ,
                dead_zone=self.cfg.BRAINCO_HAND_DEAD_ZONE,
                thumb_rotate_progress_range=(
                    self.cfg.BRAINCO_THUMB_ROTATE_PROGRESS_RANGE
                ),
            )
        except Exception as exc:
            self.hand = None
            self.tcp_tool = "None"
            self.cfg.TCP_TOOL = "None"
            self.cfg.GRIPPER = False
            utils.logger.warning(
                "TCP_TOOL='Hand' is unavailable (%s); falling back to "
                "TCP_TOOL='None'.",
                exc,
            )
            return

        utils.logger.info(
            "BrainCo six-DoF hand recognized through RealMan RM_ARM+ "
            "(limits=%s..%s, dead-zone=%.3f, filter=%.1f Hz).",
            self.hand.lower.tolist(),
            self.hand.upper.tolist(),
            self.cfg.BRAINCO_HAND_DEAD_ZONE,
            self.cfg.BRAINCO_HAND_FILTER_CUTOFF_HZ,
        )

    def _read_with_retry(self, method_name: str, read):
        for attempt in range(1, self._read_retries + 1):
            try:
                return read()
            except RuntimeError as exc:
                transient_timeout = "error code -2" in str(exc)
                if not transient_timeout or attempt == self._read_retries:
                    raise
                utils.logger.warning(
                    f"RealMan {method_name} timed out "
                    f"(attempt {attempt}/{self._read_retries}); retrying..."
                )
                time.sleep(self._retry_delay)
        raise RuntimeError(f"RealMan {method_name} retry loop ended unexpectedly.")

    def get_joint_configuration(self) -> np.ndarray:
        joints = self._read_with_retry(
            "rm_get_joint_degree",
            self.robot.get_joint_configuration,
        )
        return np.asarray(joints, dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        tcp_pose = self._read_with_retry(
            "rm_get_current_arm_state",
            self.robot.get_tcp_pose,
        )
        return self.to_tool_tcp_pose(np.asarray(tcp_pose, dtype=float))

    def get_tcp_force(self) -> np.ndarray | None:
        return self._read_with_retry("rm_get_force_data", super().get_tcp_force)

    def start_freedrive(self) -> None:
        self.robot.start_freedrive()

    def stop_freedrive(self) -> None:
        hold_joints = None
        try:
            hold_joints = self.get_joint_configuration()
        except Exception:
            utils.logger.exception(
                "Could not read RealMan joints before leaving drag-teach"
            )
        self.robot.stop_freedrive()
        if hold_joints is None or not np.all(np.isfinite(hold_joints)):
            return
        # Drag-teach interrupts an earlier MoveJ/CAN-FD setpoint. Without a new
        # hold command the arm snaps back to that old target (usually the
        # episode start / initial pose) as soon as teach mode ends.
        dt = 1.0 / max(float(getattr(self.cfg, "COLLECT_RATE", 24.0)), 1.0)
        self.command_joint_configuration(hold_joints, dt)

    def set_freedrive_sensitivity(self, grade: int) -> None:
        grade = int(grade)
        if not 0 <= grade <= 100:
            raise ValueError("RealMan freedrive sensitivity must be between 0 and 100.")
        arm = getattr(self.robot, "robot", None)
        setter = getattr(arm, "rm_set_drag_teach_sensitivity", None)
        if not callable(setter):
            raise RuntimeError(
                "The installed RealMan SDK does not expose "
                "rm_set_drag_teach_sensitivity()."
            )
        result = int(setter(grade))
        if result != 0:
            raise RuntimeError(
                "rm_set_drag_teach_sensitivity failed with RealMan error code "
                f"{result}."
            )

    def cleanup(self) -> None:
        try:
            close = getattr(self.robot, "close", None)
            if callable(close):
                close()
        finally:
            super().cleanup()


class URTorqueBackend(RobotBackend):
    name = "ur_torque"
    supports_torque_mode = True
    supports_freedrive = True
    supports_force = True
    is_ur = True

    def __init__(self, cfg, robot, ik_solver, gripper) -> None:
        super().__init__(cfg)
        self.robot = robot
        self.ik_solver = ik_solver
        self.gripper = gripper
        self.dof = 6

    @property
    def dataset_robot_type(self) -> str:
        return f"{self.cfg.ROBOT_TYPE}_torque"

    def get_joint_configuration(self) -> np.ndarray:
        return np.asarray(self.robot.get_cached_joint_configuration(), dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        return self.to_tool_tcp_pose(np.asarray(self.robot.get_cached_tcp_pose(), dtype=float))

    def get_tcp_force(self) -> np.ndarray | None:
        return np.asarray(self.robot.get_cached_tcp_force(), dtype=float)

    def solve_tcp_ik(
        self,
        tcp_pose: np.ndarray,
        seed: np.ndarray | None = None,
    ) -> np.ndarray | None:
        seed = self.get_joint_configuration() if seed is None else np.asarray(seed, dtype=float)
        solutions = self.ik_solver.inverse_kinematics_closest_with_tcp(
            np.asarray(tcp_pose, dtype=float), self.tcp_transform, *seed
        )
        if not solutions:
            return None
        return np.asarray(solutions[0], dtype=float)

    def is_joint_target_safe(
        self,
        joints: np.ndarray,
        previous_joints: np.ndarray | None,
        tcp_position: np.ndarray | None,
        joint_threshold: np.ndarray,
    ) -> bool:
        joints = np.asarray(joints, dtype=float)
        if not utils.is_joint_within_limits(joints):
            return False
        if not utils.is_pose_safe(joints, tcp_position, robot_type=self.cfg.ROBOT_TYPE):
            return False
        return super().is_joint_target_safe(joints, previous_joints, tcp_position, joint_threshold)

    def clip_joint_configuration(self, joints: np.ndarray) -> np.ndarray:
        low, high = utils.UR3E_JOINT_LIMITS
        return np.clip(np.asarray(joints, dtype=float), low, high)

    def command_joint_configuration(self, joints: np.ndarray, dt: float) -> CommandResult:
        del dt
        joints = np.asarray(joints, dtype=float)
        self.robot.target_pos = joints
        return CommandResult(True, self.get_tcp_pose(), joints)

    def command_tcp_pose(self, tcp_pose: np.ndarray, dt: float) -> CommandResult:
        del dt
        joint_target = self.solve_tcp_ik(tcp_pose)
        if joint_target is None:
            utils.logger.warning("Torque TCP IK failed, skipping action.")
            return CommandResult(False, np.asarray(tcp_pose, dtype=float), None)
        self.robot.target_pos = joint_target
        return CommandResult(True, np.asarray(tcp_pose, dtype=float), joint_target)

    def move_to_joint_configuration(self, joints: np.ndarray, speed: float | None = None):
        del speed
        self.robot.tmp_move(np.asarray(joints, dtype=float))
        return None

    def reset(self, joints: np.ndarray) -> None:
        self.robot.tmp_move(np.asarray(joints, dtype=float))

    def start_freedrive(self) -> None:
        utils.logger.warning(
            "Freedrive requested in torque mode. Trying teachMode without disabling torque control; "
            "switch TORQUE_MODE=False if this fails."
        )
        self.robot.rtde_control.teachMode()

    def stop_freedrive(self) -> None:
        self.robot.rtde_control.endTeachMode()
        self.robot.target_pos = self.get_joint_configuration()

    def cleanup(self) -> None:
        try:
            self.robot.disable_torque_control()
        finally:
            super().cleanup()
