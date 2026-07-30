"""RealMan RM75 adapter with isolated SDK types and high-follow commands."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

from ..config.models import RobotConfig
from ..core.clocks import Clock, MonotonicClock
from ..core.errors import LifecycleError, ModelValidationError, OptionalDependencyError
from ..core.types import ClockDomain, RobotAction, RobotCommandType, RobotState

ControllerFactory = Callable[[RobotConfig], object]


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
        result = tuple(tuple(float(value) for value in row) for row in values)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("RealMan TCP pose must have shape (4, 4)") from exc
    if len(result) != 4 or any(len(row) != 4 for row in result):
        raise ModelValidationError("RealMan TCP pose must have shape (4, 4)")
    return result


def _rotation_vector(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, float, float]:
    cosine = max(-1.0, min(1.0, (matrix[0][0] + matrix[1][1] + matrix[2][2] - 1) / 2))
    angle = math.acos(cosine)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0)
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        axes = (
            math.sqrt(max(0.0, (matrix[0][0] + 1) / 2)),
            math.sqrt(max(0.0, (matrix[1][1] + 1) / 2)),
            math.sqrt(max(0.0, (matrix[2][2] + 1) / 2)),
        )
        return tuple(axis * angle for axis in axes)
    scale = angle / (2 * sine)
    return (
        (matrix[2][1] - matrix[1][2]) * scale,
        (matrix[0][2] - matrix[2][0]) * scale,
        (matrix[1][0] - matrix[0][1]) * scale,
    )


def _realman_pose(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    rotation = _rotation_vector(matrix)
    return (
        matrix[0][3],
        matrix[1][3],
        matrix[2][3],
        rotation[0],
        rotation[1],
        rotation[2],
    )


def _default_controller_factory(config: RobotConfig) -> object:
    if config.ip is None:
        raise LifecycleError("robot.ip must be configured before starting a RealMan backend")
    try:
        from airo_robots.manipulators.hardware.realman import RealmanControl
    except ImportError as exc:
        raise OptionalDependencyError(
            "RealMan adapters require the 'robot-realman' optional dependency"
        ) from exc
    return RealmanControl(ip_address=config.ip, port=config.realman_port)


class RealManRobotBackend:
    """Atomic RM75 state and CAN-FD command adapter.

    Cadence, slew limiting, watchdogs, and state-push ownership belong to the
    RealMan executor. This class translates exactly one validated action.
    """

    def __init__(
        self,
        config: RobotConfig,
        *,
        controller: object | None = None,
        controller_factory: ControllerFactory | None = None,
        clock: Clock | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.robot_type != "realman":
            raise ModelValidationError("RealManRobotBackend requires robot_type 'realman'")
        if controller is not None and controller_factory is not None:
            raise ModelValidationError(
                "provide either an injected controller or controller_factory, not both"
            )
        self._config = config
        self._controller = controller
        self._factory = controller_factory or _default_controller_factory
        self._clock = clock or MonotonicClock()
        self._sleep = sleep
        self._lock = threading.RLock()
        self._sequence = 0
        self._started = False
        self._closed = False
        self._terminal_stop = False
        self._last_motion: RobotAction | None = None

    @property
    def name(self) -> str:
        return "realman"

    @property
    def dof(self) -> int:
        return 7

    @property
    def controller(self) -> object | None:
        return self._controller

    @property
    def raw_arm(self) -> object | None:
        controller = self._controller
        return None if controller is None else getattr(controller, "robot", controller)

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed RealMan backend")
            if self._started:
                raise LifecycleError("RealMan backend is already started")
            if self._controller is None:
                self._controller = self._factory(self._config)
            self._started = True

    def _require_started(self) -> tuple[object, object]:
        if self._closed:
            raise LifecycleError("RealMan backend is closed")
        if not self._started or self._controller is None:
            raise LifecycleError("RealMan backend has not been started")
        return self._controller, getattr(self._controller, "robot", self._controller)

    def _read_with_retry(self, method_name: str, read: Callable[[], object]) -> object:
        for attempt in range(1, self._config.realman_read_retries + 1):
            try:
                return read()
            except RuntimeError as exc:
                is_timeout = "error code -2" in str(exc)
                if not is_timeout or attempt == self._config.realman_read_retries:
                    raise
                self._sleep(self._config.realman_retry_delay_s)
        raise RuntimeError(f"RealMan {method_name} retry loop ended unexpectedly")

    def _read_wrench(self, controller: object, raw_arm: object) -> tuple[float, ...] | None:
        method = getattr(controller, "get_tcp_force", None)
        if callable(method):
            return _vector(method(), 6, "RealMan wrench")
        raw_read = getattr(raw_arm, "rm_get_force_data", None)
        if not callable(raw_read):
            return None
        error_code, data = raw_read()
        if error_code != 0 or not isinstance(data, dict):
            return None
        for key in (
            "zero_force_data",
            "work_zero_force_data",
            "tool_zero_force_data",
            "force_data",
        ):
            values = data.get(key)
            if values is not None:
                try:
                    return _vector(values, 6, "RealMan wrench")
                except ModelValidationError:
                    continue
        return None

    def read_state(self) -> RobotState:
        with self._lock:
            controller, raw_arm = self._require_started()
            joint_read = getattr(controller, "get_joint_configuration", None)
            tcp_read = getattr(controller, "get_tcp_pose", None)
            if not callable(joint_read) or not callable(tcp_read):
                raise LifecycleError("RealMan controller does not expose state reads")
            joints = self._read_with_retry("get_joint_configuration", joint_read)
            tcp_pose = self._read_with_retry("get_tcp_pose", tcp_read)
            state = RobotState(
                sequence=self._sequence,
                source_timestamp_ns=self._clock.now_ns(),
                clock_domain=ClockDomain.MONOTONIC,
                joints_rad=_vector(joints, 7, "RealMan joints"),
                tcp_pose=_pose(tcp_pose),
                wrench=self._read_wrench(controller, raw_arm),
            )
            self._sequence += 1
            return state

    def _send_joint(self, raw_arm: object, values: object) -> None:
        joints = _vector(values, 7, "RealMan joint command")
        method = getattr(raw_arm, "rm_movej_canfd", None)
        if not callable(method):
            raise LifecycleError("RealMan arm does not expose rm_movej_canfd()")
        result = method(
            [math.degrees(value) for value in joints],
            True,
            0,
            self._config.realman_canfd_trajectory_mode,
            self._config.realman_canfd_radio,
        )
        if result != 0:
            raise RuntimeError(f"rm_movej_canfd failed with RealMan error code {result}")

    def _send_tcp(self, raw_arm: object, values: object) -> None:
        flat = _vector(values, 16, "RealMan TCP command")
        matrix = tuple(
            tuple(flat[row * 4 + column] for column in range(4)) for row in range(4)
        )
        method = getattr(raw_arm, "rm_movep_canfd", None)
        if not callable(method):
            raise LifecycleError("RealMan arm does not expose rm_movep_canfd()")
        result = method(
            list(_realman_pose(matrix)),
            True,
            self._config.realman_canfd_trajectory_mode,
            self._config.realman_canfd_radio,
        )
        if result != 0:
            raise RuntimeError(f"rm_movep_canfd failed with RealMan error code {result}")

    @staticmethod
    def _controlled_stop(raw_arm: object) -> None:
        for method_name in ("rm_set_arm_stop", "rm_stop_arm", "stop"):
            method = getattr(raw_arm, method_name, None)
            if callable(method):
                result = method()
                if result not in {None, 0}:
                    raise RuntimeError(
                        f"{method_name} failed with RealMan error code {result}"
                    )
                return
        raise LifecycleError("RealMan arm does not expose a controlled stop operation")

    def apply_action(self, action: RobotAction) -> None:
        if not isinstance(action, RobotAction):
            raise ModelValidationError("RealMan action must be a RobotAction")
        with self._lock:
            _controller, raw_arm = self._require_started()
            if self._terminal_stop and action.command_type is not RobotCommandType.STOP:
                raise LifecycleError("RealMan backend received a terminal stop")
            if action.command_type is RobotCommandType.JOINT_POSITION:
                self._send_joint(raw_arm, action.values)
                self._last_motion = action
            elif action.command_type is RobotCommandType.TCP_POSE:
                self._send_tcp(raw_arm, action.values)
                self._last_motion = action
            elif action.command_type is RobotCommandType.HOLD:
                if self._last_motion is None:
                    raise LifecycleError("RealMan hold requires an established setpoint")
                if self._last_motion.command_type is RobotCommandType.JOINT_POSITION:
                    self._send_joint(raw_arm, self._last_motion.values)
                else:
                    self._send_tcp(raw_arm, self._last_motion.values)
            elif action.command_type is RobotCommandType.STOP:
                self._controlled_stop(raw_arm)
                self._terminal_stop = True
            elif action.command_type in {
                RobotCommandType.JOINT_VELOCITY,
                RobotCommandType.TCP_TWIST,
            }:
                raise LifecycleError("RealMan velocity modes are not configured")
            else:
                raise ModelValidationError(
                    f"unsupported RealMan command type: {action.command_type.value}"
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            controller = self._controller
            self._started = False
            self._closed = True
        if controller is None:
            return
        for method_name in ("close", "disconnect"):
            method = getattr(controller, method_name, None)
            if callable(method):
                method()
                break


def create_realman_backend(config: RobotConfig) -> RealManRobotBackend:
    """Factory target that delays RealMan SDK connection until start."""

    return RealManRobotBackend(config)
