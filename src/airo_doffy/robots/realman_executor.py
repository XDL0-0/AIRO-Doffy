"""RealMan high-follow scheduling, slew limiting, and timing supervision."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from ..config.models import RobotConfig
from ..core.buffers import LatestValueBuffer
from ..core.clocks import Clock, MonotonicClock
from ..core.errors import LifecycleError, ModelValidationError
from ..core.types import RobotAction, RobotCommandType
from .base import RobotBackend

_HIGH_FOLLOW_LIMIT_NS = 10_000_000


def _matrix(values: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(values[row * 4 + column] for column in range(4)) for row in range(4)
    )


def _flatten(values: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(value for row in values for value in row)


def _matmul3(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def _transpose3(
    matrix: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def _axis_angle(
    rotation: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, float, float], float]:
    cosine = max(-1.0, min(1.0, (sum(rotation[i][i] for i in range(3)) - 1) / 2))
    angle = math.acos(cosine)
    if angle < 1e-12:
        return (1.0, 0.0, 0.0), 0.0
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        axis = [
            math.sqrt(max(0.0, (rotation[i][i] + 1) / 2)) for i in range(3)
        ]
        norm = math.sqrt(sum(value * value for value in axis))
        return tuple(value / norm for value in axis), angle
    scale = 1 / (2 * sine)
    return (
        (
            (rotation[2][1] - rotation[1][2]) * scale,
            (rotation[0][2] - rotation[2][0]) * scale,
            (rotation[1][0] - rotation[0][1]) * scale,
        ),
        angle,
    )


def _rodrigues(
    axis: tuple[float, float, float],
    angle: float,
) -> tuple[tuple[float, ...], ...]:
    x, y, z = axis
    cosine, sine, one_minus = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return (
        (
            cosine + x * x * one_minus,
            x * y * one_minus - z * sine,
            x * z * one_minus + y * sine,
        ),
        (
            y * x * one_minus + z * sine,
            cosine + y * y * one_minus,
            y * z * one_minus - x * sine,
        ),
        (
            z * x * one_minus - y * sine,
            z * y * one_minus + x * sine,
            cosine + z * z * one_minus,
        ),
    )


@dataclass(frozen=True, slots=True)
class RealManExecutorSnapshot:
    """Immutable high-follow timing and command snapshot."""

    running: bool
    ready: bool
    terminal_stop: bool
    packets_sent: int
    achieved_hz: float | None
    command_gap_violations: int
    sdk_call_overruns: int
    last_command_start_ns: int | None
    last_success_ns: int | None
    latest_sequence: int | None
    rejected_count: int
    error: str | None


class RealManCanfdExecutor:
    """Continuously resend a slew-limited latest target on one owner thread."""

    def __init__(
        self,
        backend: RobotBackend,
        config: RobotConfig,
        *,
        control_mode: str = "joint",
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(backend, RobotBackend) or backend.dof != 7:
            raise ModelValidationError("RealMan executor requires a 7-DoF RobotBackend")
        if config.robot_type != "realman":
            raise ModelValidationError("RealMan executor requires realman RobotConfig")
        mode = control_mode.lower()
        if mode not in {"joint", "tcp"}:
            raise ModelValidationError("RealMan control_mode must be 'joint' or 'tcp'")
        self._backend = backend
        self._config = config
        self._mode = mode
        self._clock = clock or MonotonicClock()
        self._actions = LatestValueBuffer[RobotAction]()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._lock = threading.RLock()
        self._started = False
        self._running = False
        self._inflight = False
        self._inflight_started_ns: int | None = None
        self._closed = False
        self._terminal_stop = False
        self._target: tuple[float, ...] | None = None
        self._setpoint: tuple[float, ...] | None = None
        self._observed_sequence: int | None = None
        self._packets_sent = 0
        self._achieved_hz: float | None = None
        self._gap_violations = 0
        self._sdk_overruns = 0
        self._last_command_start_ns: int | None = None
        self._last_success_ns: int | None = None
        self._error: str | None = None

    @property
    def target_hz(self) -> float:
        return float(self._config.realman_control_rate_hz)

    def submit(self, action: RobotAction) -> bool:
        if not isinstance(action, RobotAction):
            raise ModelValidationError("RealMan executor action must be a RobotAction")
        expected = (
            RobotCommandType.JOINT_POSITION
            if self._mode == "joint"
            else RobotCommandType.TCP_POSE
        )
        if action.command_type not in {expected, RobotCommandType.HOLD, RobotCommandType.STOP}:
            raise ModelValidationError(
                f"{self._mode} RealMan executor does not accept {action.command_type.value}"
            )
        with self._lock:
            if self._closed:
                raise LifecycleError("RealMan executor is closed")
            if self._terminal_stop:
                raise LifecycleError("RealMan executor received a terminal stop")
        return self._actions.publish(action)

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed RealMan executor")
            if self._started:
                raise LifecycleError("RealMan executor is already started")
        self._backend.start()
        try:
            state = self._backend.read_state()
        except Exception:
            self._backend.close()
            raise
        initial = (
            state.joints_rad
            if self._mode == "joint"
            else _flatten(state.tcp_pose)
        )
        with self._lock:
            self._target = initial
            self._setpoint = initial
            self._last_success_ns = self._clock.now_ns()
            self._started = True

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("RealMan executor is closed")
        if not self._started:
            raise LifecycleError("RealMan executor has not been started")

    def _refresh_target(self) -> RobotAction | None:
        action = self._actions.read()
        if action is None or action.sequence == self._observed_sequence:
            return None
        self._observed_sequence = action.sequence
        if action.command_type is RobotCommandType.STOP:
            self._terminal_stop = True
            return action
        if action.command_type is RobotCommandType.HOLD:
            self._target = self._setpoint
        else:
            self._target = action.values
        return action

    def _next_joint_setpoint(self) -> tuple[float, ...]:
        assert self._target is not None and self._setpoint is not None
        maximum_step = (
            self._config.realman_max_joint_speed_rad_s
            / self._config.realman_control_rate_hz
        )
        return tuple(
            current + min(max(target - current, -maximum_step), maximum_step)
            for current, target in zip(self._setpoint, self._target, strict=True)
        )

    def _next_tcp_setpoint(self) -> tuple[float, ...]:
        assert self._target is not None and self._setpoint is not None
        current, target = _matrix(self._setpoint), _matrix(self._target)
        translation_delta = tuple(target[i][3] - current[i][3] for i in range(3))
        distance = math.sqrt(sum(value * value for value in translation_delta))
        linear_step = (
            self._config.realman_max_linear_speed_m_s
            / self._config.realman_control_rate_hz
        )
        translation_scale = 1.0 if distance <= linear_step else linear_step / distance
        translation = tuple(
            current[i][3] + translation_delta[i] * translation_scale for i in range(3)
        )

        current_rotation = tuple(tuple(current[i][j] for j in range(3)) for i in range(3))
        target_rotation = tuple(tuple(target[i][j] for j in range(3)) for i in range(3))
        delta_rotation = _matmul3(target_rotation, _transpose3(current_rotation))
        axis, angle = _axis_angle(delta_rotation)
        angular_step = (
            self._config.realman_max_angular_speed_rad_s
            / self._config.realman_control_rate_hz
        )
        step_rotation = _rodrigues(axis, min(angle, angular_step))
        rotation = _matmul3(step_rotation, current_rotation)
        matrix = (
            (*rotation[0], translation[0]),
            (*rotation[1], translation[1]),
            (*rotation[2], translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
        return _flatten(matrix)

    def execute_once(self) -> bool:
        """Send one high-follow packet from the current latest target."""

        with self._lock:
            self._require_started()
            if self._terminal_stop:
                return False
            new_action = self._refresh_target()
            if new_action is not None and new_action.command_type is RobotCommandType.STOP:
                self._inflight = True
                self._inflight_started_ns = self._clock.now_ns()
                outgoing = new_action
            else:
                next_values = (
                    self._next_joint_setpoint()
                    if self._mode == "joint"
                    else self._next_tcp_setpoint()
                )
                self._setpoint = next_values
                sequence = self._observed_sequence or 0
                outgoing = RobotAction(
                    sequence=sequence,
                    source_timestamp_ns=self._clock.now_ns(),
                    command_type=(
                        RobotCommandType.JOINT_POSITION
                        if self._mode == "joint"
                        else RobotCommandType.TCP_POSE
                    ),
                    values=next_values,
                    duration_s=1.0 / self._config.realman_control_rate_hz,
                )
                self._inflight = True
                self._inflight_started_ns = self._clock.now_ns()
        try:
            self._backend.apply_action(outgoing)
        except Exception as exc:
            with self._lock:
                self._inflight = False
                self._inflight_started_ns = None
                self._error = f"{type(exc).__name__}: {exc}"
                self._stop_event.set()
            raise
        with self._lock:
            self._inflight = False
            self._inflight_started_ns = None
            self._packets_sent += 1
            self._last_success_ns = self._clock.now_ns()
            if outgoing.command_type is RobotCommandType.STOP:
                self._stop_event.set()
        return True

    def _record_timing(self, start_ns: int, end_ns: int, *, ready: bool) -> bool:
        violation = False
        with self._lock:
            if (
                self._last_command_start_ns is not None
                and start_ns - self._last_command_start_ns > _HIGH_FOLLOW_LIMIT_NS
            ):
                self._gap_violations += 1
                violation = True
            if end_ns - start_ns > _HIGH_FOLLOW_LIMIT_NS:
                self._sdk_overruns += 1
                violation = True
            self._last_command_start_ns = start_ns
        if ready and violation:
            raise RuntimeError("RealMan high-follow timing exceeded 10 ms after readiness")
        return violation

    def run(self, external_stop: threading.Event | None = None) -> None:
        """Run high-follow scheduling and enforce startup/runtime timing gates."""

        with self._lock:
            self._require_started()
            if self._running:
                raise LifecycleError("RealMan executor run loop is already active")
            self._running = True
        period_s = 1.0 / self._config.realman_control_rate_hz
        window_s = self._config.realman_rate_check_window_s
        deadline = time.perf_counter()
        window_start = deadline
        window_packets = 0
        window_violation = False
        failed_windows = 0
        try:
            while not self._stop_event.is_set() and not (
                external_stop is not None and external_stop.is_set()
            ):
                call_start_ns = time.perf_counter_ns()
                self.execute_once()
                call_end_ns = time.perf_counter_ns()
                window_packets += 1
                window_violation |= self._record_timing(
                    call_start_ns,
                    call_end_ns,
                    ready=self._ready_event.is_set(),
                )
                now = time.perf_counter()
                elapsed = now - window_start
                if elapsed >= window_s:
                    achieved = window_packets / elapsed
                    with self._lock:
                        self._achieved_hz = achieved
                    healthy = (
                        achieved > self._config.realman_min_canfd_rate_hz
                        and not window_violation
                    )
                    if healthy:
                        failed_windows = 0
                        self._ready_event.set()
                    else:
                        failed_windows += 1
                        if self._ready_event.is_set() or (
                            failed_windows >= self._config.realman_rate_failure_windows
                        ):
                            raise RuntimeError(
                                "RealMan CAN-FD failed its >"
                                f"{self._config.realman_min_canfd_rate_hz:g} Hz timing gate"
                            )
                    window_start = now
                    window_packets = 0
                    window_violation = False

                deadline += period_s
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    deadline = time.perf_counter()
                    continue
                # Event.wait() has coarse timer behavior on some Windows builds
                # and can create >10 ms gaps at a 5 ms target period.
                time.sleep(remaining)
        except Exception as exc:
            with self._lock:
                if self._error is None:
                    self._error = f"{type(exc).__name__}: {exc}"
                self._stop_event.set()
            if external_stop is not None:
                external_stop.set()
            raise
        finally:
            with self._lock:
                self._running = False

    def wait_until_healthy(self, timeout: float | None = None) -> float:
        """Wait for a clean timing window or raise the recorded failure."""

        timeout_s = (
            self._config.realman_rate_check_window_s
            * (self._config.realman_rate_failure_windows + 1)
            if timeout is None
            else float(timeout)
        )
        deadline = time.monotonic() + timeout_s
        while not self._ready_event.is_set():
            with self._lock:
                if self._error is not None:
                    raise LifecycleError(f"RealMan executor failed: {self._error}")
                if not self._running and self._stop_event.is_set():
                    raise LifecycleError("RealMan executor stopped before becoming healthy")
            if time.monotonic() >= deadline:
                raise LifecycleError("RealMan executor timing gate timed out")
            self._ready_event.wait(min(0.01, max(0.0, deadline - time.monotonic())))
        with self._lock:
            assert self._achieved_hz is not None
            return self._achieved_hz

    def heartbeat_error(self) -> str | None:
        now_ns = self._clock.now_ns()
        timeout_ns = int(self._config.realman_heartbeat_timeout_s * 1e9)
        with self._lock:
            if self._inflight and self._inflight_started_ns is not None:
                if now_ns - self._inflight_started_ns > timeout_ns:
                    return "RealMan CAN-FD SDK call exceeded heartbeat timeout"
            if self._last_success_ns is not None and now_ns - self._last_success_ns > timeout_ns:
                return "RealMan CAN-FD has no recent successful packet"
            return self._error

    def request_stop(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> RealManExecutorSnapshot:
        with self._lock:
            return RealManExecutorSnapshot(
                running=self._running,
                ready=self._ready_event.is_set(),
                terminal_stop=self._terminal_stop,
                packets_sent=self._packets_sent,
                achieved_hz=self._achieved_hz,
                command_gap_violations=self._gap_violations,
                sdk_call_overruns=self._sdk_overruns,
                last_command_start_ns=self._last_command_start_ns,
                last_success_ns=self._last_success_ns,
                latest_sequence=self._actions.latest_sequence,
                rejected_count=self._actions.rejected_count,
                error=self._error,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._running or self._inflight:
                raise LifecycleError(
                    "stop and join the RealMan SDK-owner thread before close"
                )
            self._closed = True
            self._started = False
            self._stop_event.set()
            self._actions.close()
        self._backend.close()
