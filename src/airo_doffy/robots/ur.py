"""UR RTDE adapter with explicit lifecycle and lazy vendor imports."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from ..config.models import RobotConfig
from ..core.clocks import Clock, MonotonicClock
from ..core.errors import LifecycleError, ModelValidationError, OptionalDependencyError
from ..core.types import ClockDomain, RobotAction, RobotCommandType, RobotState

ManipulatorFactory = Callable[[RobotConfig], object]
InverseKinematics = Callable[[object, tuple[float, ...]], Sequence[float] | None]


def _numpy_array(values: object):
    try:
        import numpy as np
    except ImportError as exc:
        raise OptionalDependencyError(
            "UR adapters require the 'robot-ur' optional dependency"
        ) from exc
    return np.asarray(values, dtype=float)


def _vector(values: object, size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must contain {size} numbers") from exc
    if len(result) != size:
        raise ModelValidationError(f"{name} must contain {size} numbers")
    return result


def _pose(values: object) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(float(value) for value in row) for row in values)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("UR TCP pose must have shape (4, 4)") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ModelValidationError("UR TCP pose must have shape (4, 4)")
    return rows


def _default_manipulator_factory(config: RobotConfig) -> object:
    if config.ip is None:
        raise LifecycleError("robot.ip must be configured before starting a UR backend")
    try:
        if config.torque_mode:
            from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque

            robot_class = URrtdeTorque
        else:
            from airo_robots.manipulators.hardware.ur_rtde import URrtde

            robot_class = URrtde
    except ImportError as exc:
        raise OptionalDependencyError(
            "UR adapters require the 'robot-ur' optional dependency"
        ) from exc

    robot_config = (
        robot_class.UR3E_CONFIG if config.robot_type == "ur3e" else robot_class.UR5E_CONFIG
    )
    kwargs: dict[str, Any] = {}
    if config.torque_mode:
        kwargs["initial_joint_configuration"] = _numpy_array(config.initial_joints_rad)
    return robot_class(config.ip, robot_config, **kwargs)


def _default_torque_ik(config: RobotConfig) -> InverseKinematics:
    try:
        if config.robot_type == "ur3e":
            from ur_analytic_ik import ur3e as analytic_ik
        else:
            from ur_analytic_ik import ur5e as analytic_ik
    except ImportError as exc:
        raise OptionalDependencyError(
            "UR torque TCP commands require the 'robot-ur' optional dependency"
        ) from exc

    def solve(pose: object, seed: tuple[float, ...]) -> Sequence[float] | None:
        solutions = analytic_ik.inverse_kinematics_closest_with_tcp(
            _numpy_array(pose),
            _numpy_array(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
            *seed,
        )
        return None if not solutions else solutions[0]

    return solve


class URRobotBackend:
    """Backend-neutral facade over position or torque-mode UR RTDE objects."""

    def __init__(
        self,
        config: RobotConfig,
        *,
        manipulator: object | None = None,
        manipulator_factory: ManipulatorFactory | None = None,
        inverse_kinematics: InverseKinematics | None = None,
        clock: Clock | None = None,
    ) -> None:
        if config.robot_type not in {"ur3e", "ur5e"}:
            raise ModelValidationError("URRobotBackend requires robot_type 'ur3e' or 'ur5e'")
        if manipulator is not None and manipulator_factory is not None:
            raise ModelValidationError(
                "provide either an injected manipulator or manipulator_factory, not both"
            )
        self._config = config
        self._manipulator = manipulator
        self._factory = manipulator_factory or _default_manipulator_factory
        self._inverse_kinematics = inverse_kinematics
        self._clock = clock or MonotonicClock()
        self._lock = threading.RLock()
        self._sequence = 0
        self._started = False
        self._closed = False
        self._terminal_stop = False

    @property
    def name(self) -> str:
        suffix = "_torque" if self._config.torque_mode else ""
        return f"{self._config.robot_type}{suffix}"

    @property
    def dof(self) -> int:
        return 6

    @property
    def manipulator(self) -> object | None:
        return self._manipulator

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed UR backend")
            if self._started:
                raise LifecycleError("UR backend is already started")
            if self._manipulator is None:
                self._manipulator = self._factory(self._config)
            self._started = True

    def _require_started(self) -> object:
        if self._closed:
            raise LifecycleError("UR backend is closed")
        if not self._started or self._manipulator is None:
            raise LifecycleError("UR backend has not been started")
        return self._manipulator

    @staticmethod
    def _call(robot: object, method_name: str, *args: object) -> object:
        method = getattr(robot, method_name, None)
        if not callable(method):
            raise LifecycleError(f"UR adapter does not expose {method_name}()")
        return method(*args)

    def _read_joints(self, robot: object) -> tuple[float, ...]:
        method = (
            "get_cached_joint_configuration"
            if self._config.torque_mode
            else "get_joint_configuration"
        )
        return _vector(self._call(robot, method), self.dof, "UR joints")

    def _read_tcp_pose(self, robot: object) -> tuple[tuple[float, ...], ...]:
        method = "get_cached_tcp_pose" if self._config.torque_mode else "get_tcp_pose"
        return _pose(self._call(robot, method))

    def _read_wrench(self, robot: object) -> tuple[float, ...] | None:
        direct_names = (
            ("get_cached_tcp_force", "get_tcp_force")
            if self._config.torque_mode
            else ("get_tcp_force",)
        )
        for method_name in direct_names:
            method = getattr(robot, method_name, None)
            if callable(method):
                return _vector(method(), 6, "UR wrench")
        receiver = getattr(robot, "rtde_receive", None)
        method = getattr(receiver, "getActualTCPForce", None)
        if callable(method):
            return _vector(method(), 6, "UR wrench")
        return None

    def read_state(self) -> RobotState:
        with self._lock:
            robot = self._require_started()
            state = RobotState(
                sequence=self._sequence,
                source_timestamp_ns=self._clock.now_ns(),
                clock_domain=ClockDomain.MONOTONIC,
                joints_rad=self._read_joints(robot),
                tcp_pose=self._read_tcp_pose(robot),
                wrench=self._read_wrench(robot),
            )
            self._sequence += 1
            return state

    @staticmethod
    def _flattened_pose(action: RobotAction) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(action.values[row * 4 + column] for column in range(4))
            for row in range(4)
        )

    def _apply_joint_position(self, robot: object, action: RobotAction) -> None:
        if len(action.values) != self.dof:
            raise ModelValidationError("UR joint command must contain 6 values")
        target = _numpy_array(action.values)
        if self._config.torque_mode:
            if not hasattr(robot, "target_pos"):
                raise LifecycleError("UR torque adapter does not expose target_pos")
            robot.target_pos = target
            return
        duration = action.duration_s or (1.0 / 60.0)
        self._call(robot, "servo_to_joint_configuration", target, duration)

    def _apply_tcp_pose(self, robot: object, action: RobotAction) -> None:
        if self._config.torque_mode:
            solver = self._inverse_kinematics
            if solver is None:
                solver = _default_torque_ik(self._config)
                self._inverse_kinematics = solver
            pose = self._flattened_pose(action)
            solution = solver(pose, self._read_joints(robot))
            if solution is None:
                raise LifecycleError("UR torque TCP inverse kinematics failed")
            target = _vector(solution, self.dof, "UR torque IK solution")
            robot.target_pos = _numpy_array(target)
            return
        duration = action.duration_s or (1.0 / 60.0)
        self._call(
            robot,
            "servo_to_tcp_pose",
            _numpy_array(self._flattened_pose(action)),
            duration,
        )

    def _apply_velocity(
        self,
        robot: object,
        action: RobotAction,
        *,
        method_name: str,
    ) -> None:
        if self._config.torque_mode:
            raise LifecycleError("velocity commands are unavailable in UR torque mode")
        duration = action.duration_s
        if duration is None:
            raise ModelValidationError("UR velocity commands require duration_s")
        self._call(robot, method_name, _numpy_array(action.values), duration)

    @staticmethod
    def _control_object(robot: object) -> object:
        return getattr(robot, "rtde_control", robot)

    def _hold(self, robot: object) -> None:
        control = self._control_object(robot)
        for method_name in ("servoStop", "speedStop"):
            method = getattr(control, method_name, None)
            if callable(method):
                method()
                return
        raise LifecycleError("UR adapter does not expose a hold operation")

    def _stop(self, robot: object) -> None:
        control = self._control_object(robot)
        for method_name in ("stopJ", "servoStop", "speedStop"):
            method = getattr(control, method_name, None)
            if callable(method):
                method()
                return
        raise LifecycleError("UR adapter does not expose a controlled stop operation")

    def apply_action(self, action: RobotAction) -> None:
        if not isinstance(action, RobotAction):
            raise ModelValidationError("UR action must be a RobotAction")
        with self._lock:
            robot = self._require_started()
            if self._terminal_stop and action.command_type is not RobotCommandType.STOP:
                raise LifecycleError("UR backend received a terminal stop")
            if action.command_type is RobotCommandType.JOINT_POSITION:
                self._apply_joint_position(robot, action)
            elif action.command_type is RobotCommandType.TCP_POSE:
                self._apply_tcp_pose(robot, action)
            elif action.command_type is RobotCommandType.JOINT_VELOCITY:
                self._apply_velocity(
                    robot,
                    action,
                    method_name="servo_to_joint_velocity",
                )
            elif action.command_type is RobotCommandType.TCP_TWIST:
                self._apply_velocity(robot, action, method_name="servo_to_tcp_velocity")
            elif action.command_type is RobotCommandType.HOLD:
                self._hold(robot)
            elif action.command_type is RobotCommandType.STOP:
                self._stop(robot)
                self._terminal_stop = True
            else:
                raise ModelValidationError(
                    f"unsupported UR command type: {action.command_type.value}"
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            robot = self._manipulator
            self._started = False
            self._closed = True
        if robot is None:
            return
        try:
            if self._config.torque_mode:
                disable = getattr(robot, "disable_torque_control", None)
                if callable(disable):
                    disable()
            else:
                try:
                    self._stop(robot)
                except LifecycleError:
                    pass
        finally:
            for method_name in ("close", "disconnect"):
                method = getattr(robot, method_name, None)
                if callable(method):
                    method()
                    break


def create_ur_backend(config: RobotConfig) -> URRobotBackend:
    """Factory target that does not connect until the backend is started."""

    return URRobotBackend(config)
