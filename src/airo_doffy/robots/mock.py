"""Deterministic robot backend for unit and integration tests."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from ..core.clocks import Clock, MonotonicClock
from ..core.errors import LifecycleError, ModelValidationError
from ..core.types import ClockDomain, RobotAction, RobotCommandType, RobotState

_IDENTITY_POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_OPERATIONS = frozenset({"start", "read_state", "apply_action", "close"})


class InjectedRobotError(RuntimeError):
    """Failure raised by a mock backend when a test requests fault injection."""


class MockRobotBackend:
    """Thread-safe backend with captured commands, latency, and queued failures."""

    def __init__(
        self,
        *,
        name: str = "mock",
        dof: int = 6,
        initial_state: RobotState | None = None,
        latency_s: float = 0.0,
        clock: Clock | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ModelValidationError("mock robot name must be a non-empty string")
        if dof not in {6, 7}:
            raise ModelValidationError("mock robot dof must be 6 or 7")
        latency = float(latency_s)
        if not math.isfinite(latency) or latency < 0:
            raise ModelValidationError("mock robot latency_s must be finite and non-negative")
        self._name = name
        self._dof = dof
        self._clock = clock or MonotonicClock()
        self._sleep = sleep
        self._latency_s = latency
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._holding = False
        self._stopped = False
        self._actions: list[RobotAction] = []
        self._failures: dict[str, deque[Exception]] = {
            operation: deque() for operation in _OPERATIONS
        }
        state = initial_state or RobotState(
            sequence=0,
            source_timestamp_ns=self._clock.now_ns(),
            clock_domain=ClockDomain.MONOTONIC,
            joints_rad=(0.0,) * dof,
            tcp_pose=_IDENTITY_POSE,
        )
        self._validate_state(state)
        self._state = state

    @property
    def name(self) -> str:
        return self._name

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def holding(self) -> bool:
        with self._lock:
            return self._holding

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    @property
    def captured_actions(self) -> tuple[RobotAction, ...]:
        with self._lock:
            return tuple(self._actions)

    def _delay(self) -> None:
        if self._latency_s:
            self._sleep(self._latency_s)

    def _raise_injected(self, operation: str) -> None:
        failure = self._failures[operation].popleft() if self._failures[operation] else None
        if failure is not None:
            raise failure

    def _validate_state(self, state: RobotState) -> None:
        if not isinstance(state, RobotState):
            raise ModelValidationError("mock state must be a RobotState")
        if len(state.joints_rad) != self._dof:
            raise ModelValidationError(
                f"mock state has {len(state.joints_rad)} joints, expected {self._dof}"
            )

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("mock robot is closed")
        if not self._started:
            raise LifecycleError("mock robot has not been started")

    def inject_failure(
        self,
        operation: str,
        error: Exception | None = None,
        *,
        count: int = 1,
    ) -> None:
        """Queue failures for the next matching operations."""

        if operation not in _OPERATIONS:
            supported = ", ".join(sorted(_OPERATIONS))
            raise ModelValidationError(f"operation must be one of: {supported}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ModelValidationError("failure count must be a positive integer")
        failure = error or InjectedRobotError(f"injected mock robot {operation} failure")
        if not isinstance(failure, Exception):
            raise ModelValidationError("injected failure must be an Exception")
        with self._lock:
            self._failures[operation].extend(failure for _ in range(count))

    def set_state(self, state: RobotState) -> None:
        """Replace the synthetic sensor state without recording a command."""

        self._validate_state(state)
        with self._lock:
            if self._closed:
                raise LifecycleError("mock robot is closed")
            self._state = state

    def clear_captured_actions(self) -> None:
        with self._lock:
            self._actions.clear()

    def start(self) -> None:
        self._delay()
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed mock robot")
            if self._started:
                raise LifecycleError("mock robot is already started")
            self._raise_injected("start")
            self._started = True

    def read_state(self) -> RobotState:
        self._delay()
        with self._lock:
            self._require_started()
            self._raise_injected("read_state")
            return self._state

    def _updated_state(self, action: RobotAction) -> RobotState:
        joints = self._state.joints_rad
        tcp_pose = self._state.tcp_pose
        if action.command_type is RobotCommandType.JOINT_POSITION:
            if len(action.values) != self._dof:
                raise ModelValidationError(
                    f"joint action has {len(action.values)} values, expected {self._dof}"
                )
            joints = action.values
        elif action.command_type is RobotCommandType.TCP_POSE:
            tcp_pose = tuple(
                tuple(action.values[row * 4 + column] for column in range(4))
                for row in range(4)
            )
        elif (
            action.command_type is RobotCommandType.JOINT_VELOCITY
            and action.duration_s is not None
        ):
            if len(action.values) != self._dof:
                raise ModelValidationError(
                    f"joint action has {len(action.values)} values, expected {self._dof}"
                )
            joints = tuple(
                position + velocity * action.duration_s
                for position, velocity in zip(joints, action.values, strict=True)
            )
        return RobotState(
            sequence=self._state.sequence + 1,
            source_timestamp_ns=self._clock.now_ns(),
            clock_domain=ClockDomain.MONOTONIC,
            joints_rad=joints,
            tcp_pose=tcp_pose,
            gripper_width_m=(
                self._state.gripper_width_m
                if action.gripper_width_m is None
                else action.gripper_width_m
            ),
            wrench=self._state.wrench,
        )

    def apply_action(self, action: RobotAction) -> None:
        if not isinstance(action, RobotAction):
            raise ModelValidationError("mock action must be a RobotAction")
        self._delay()
        with self._lock:
            self._require_started()
            self._raise_injected("apply_action")
            if self._stopped and action.command_type not in {
                RobotCommandType.HOLD,
                RobotCommandType.STOP,
            }:
                raise LifecycleError("mock robot is stopped")
            self._state = self._updated_state(action)
            self._actions.append(action)
            if action.command_type is RobotCommandType.STOP:
                self._stopped = True
                self._holding = True
            elif action.command_type is RobotCommandType.HOLD:
                self._holding = True
            else:
                self._holding = False

    def close(self) -> None:
        self._delay()
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True
            self._holding = True
            self._raise_injected("close")
