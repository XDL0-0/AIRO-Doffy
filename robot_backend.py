"""Robot backend adapters used by teleoperation and inference.

The upper layers produce TCP or joint targets. Backends translate those targets
to the concrete robot implementation: a generic PositionManipulator, UR RTDE, or
the UR torque worker's shared joint target.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import time

import numpy as np

import utils
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85, rescale_range


class NullGripper:
    """Small gripper stand-in for manipulators without a configured gripper."""

    def __init__(self, max_width: float) -> None:
        self._width = float(max_width)
        self._max_width = float(max_width)

    def open(self) -> None:
        self._width = self._max_width

    def move(self, target_width_in_meters: float, *_, **__):
        self._set_target_width(target_width_in_meters)
        return None

    def _set_target_width(self, target_width_in_meters: float) -> None:
        self._width = float(np.clip(target_width_in_meters, 0.0, self._max_width))

    def get_current_width(self) -> float:
        return self._width

    def close(self) -> None:
        pass


class FastRobotiq2F85(Robotiq2F85):
    """Robotiq 2F-85 with a persistent TCP socket and non-blocking setpoint writes."""

    def __init__(self, host_ip: str, port: int = 63352, fingers_max_stroke=None):
        super().__init__(host_ip, port, fingers_max_stroke)
        self._persistent_sock = self._make_sock(host_ip, port)
        self._fast_communicate("SET SPE 255")
        verify = self._fast_communicate("GET SPE")
        utils.logger.info(f"FastRobotiq2F85 ready (SPE={verify})")

    def _make_sock(self, host: str, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(0.02)
        sock.connect((host, port))
        return sock

    def _fast_communicate(self, command: str) -> str:
        try:
            self._persistent_sock.sendall((command.strip() + "\n").encode())
            data = self._persistent_sock.recv(2**10)
            return data.decode()[:-1]
        except (socket.timeout, BrokenPipeError, ConnectionResetError, OSError) as exc:
            utils.logger.warning(f"Gripper socket error: {exc}, reconnecting...")
            try:
                self._persistent_sock.close()
            except Exception:
                pass
            self._persistent_sock = self._make_sock(self.host_ip, self.port)
            self._persistent_sock.sendall((command.strip() + "\n").encode())
            data = self._persistent_sock.recv(2**10)
            return data.decode()[:-1]

    def _set_target_width(self, target_width_in_meters: float) -> None:
        target_width_in_meters = np.clip(
            target_width_in_meters,
            self._gripper_specs.min_width,
            self._gripper_specs.max_width,
        )
        register_val = round(
            rescale_range(
                target_width_in_meters,
                self._gripper_specs.min_width,
                self._gripper_specs.max_width,
                230,
                0,
            )
        )
        self._fast_communicate(f"SET  POS {register_val}")

    def get_current_width(self) -> float:
        register_value = int(self._fast_communicate("GET POS").split(" ")[1])
        return rescale_range(
            register_value,
            0,
            230,
            self._gripper_specs.max_width,
            self._gripper_specs.min_width,
        )

    def close(self) -> None:
        try:
            self._persistent_sock.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


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
        self.tcp_transform = np.asarray(cfg.TCP_TRANSFORM, dtype=float)
        self.inv_tcp_transform = np.linalg.inv(self.tcp_transform)
        self.dof = 0
        self.robot = None
        self.ik_solver = None
        self.gripper = NullGripper(cfg.GRIPPER_MAX)

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
        action = self.move_to_joint_configuration(joints, 1.0)
        if action is not None and hasattr(action, "wait"):
            action.wait()

    def start_freedrive(self) -> None:
        raise NotImplementedError(f"{self.name} does not support freedrive through this backend.")

    def stop_freedrive(self) -> None:
        raise NotImplementedError(f"{self.name} does not support freedrive through this backend.")

    def cleanup(self) -> None:
        close = getattr(self.gripper, "close", None)
        if callable(close):
            close()


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

    def __init__(self, cfg, robot) -> None:
        super().__init__(cfg, robot, "realman", NullGripper(cfg.GRIPPER_MAX))
        self._read_retries = int(getattr(cfg, "REALMAN_READ_RETRIES", 3))
        self._retry_delay = float(getattr(cfg, "REALMAN_RETRY_DELAY", 0.05))

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


def _ur_config_and_ik(robot_type: str, robot_cls):
    if robot_type == "ur3e":
        from ur_analytic_ik import ur3e as ik

        return robot_cls.UR3E_CONFIG, ik
    if robot_type == "ur5e":
        from ur_analytic_ik import ur5e as ik

        return robot_cls.UR5E_CONFIG, ik
    raise ValueError(f"Unsupported UR robot type: {robot_type}")


def make_robot_backend(cfg) -> RobotBackend:
    robot_type = cfg.ROBOT_TYPE.lower()
    robot_ip = cfg.ROBOT_IP or cfg.UR_IP

    if robot_type in {"ur3e", "ur5e"}:
        gripper = FastRobotiq2F85(cfg.UR_IP)
        gripper.open()
        if cfg.TORQUE_MODE:
            from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque

            robot_config, ik = _ur_config_and_ik(robot_type, URrtdeTorque)
            kwargs = {"initial_joint_configuration": np.asarray(cfg.INITIAL_JOINT, dtype=float)}
            if getattr(cfg, "RUCKIG_ENABLE", False):
                ruckig_params = {
                    "max_vel": cfg.RUCKIG_MAX_VEL,
                    "max_acc": cfg.RUCKIG_MAX_ACC,
                    "max_jerk": cfg.RUCKIG_MAX_JERK,
                }
                kwargs["ruckig_params"] = ruckig_params
            try:
                robot = URrtdeTorque(robot_ip, robot_config, **kwargs)
            except TypeError:
                kwargs.pop("ruckig_params", None)
                robot = URrtdeTorque(robot_ip, robot_config, **kwargs)
            return URTorqueBackend(cfg, robot, ik, gripper)

        from airo_robots.manipulators.hardware.ur_rtde import URrtde

        robot_config, ik = _ur_config_and_ik(robot_type, URrtde)
        robot = URrtde(robot_ip, robot_config)
        return URPositionBackend(cfg, robot, ik, gripper)

    if robot_type == "realman":
        from airo_robots.manipulators.hardware.realman import RealmanControl

        robot = RealmanControl(ip_address=robot_ip, port=cfg.REALMAN_PORT)
        return RealManBackend(cfg, robot)

    raise ValueError(
        f"Unsupported ROBOT_TYPE '{cfg.ROBOT_TYPE}'. Use 'ur3e', 'ur5e', or 'realman'."
    )


def make_robot(ur_ip: str, robot_type: str, torque_mode: bool, initial_joint=None, ruckig_params=None):
    """Backward-compatible factory returning the raw UR object and analytic IK module."""
    del ruckig_params
    if not torque_mode:
        from airo_robots.manipulators.hardware.ur_rtde import URrtde

        robot_cls = URrtde
    else:
        from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque

        robot_cls = URrtdeTorque

    robot_config, ik = _ur_config_and_ik(robot_type, robot_cls)
    kwargs = {}
    if torque_mode:
        kwargs["initial_joint_configuration"] = initial_joint
    return robot_cls(ur_ip, robot_config, **kwargs), ik
