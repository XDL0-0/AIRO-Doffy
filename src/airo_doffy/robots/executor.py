"""Latest-command scheduling separated from robot mappings and adapters."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from ..core.buffers import LatestValueBuffer
from ..core.clocks import Clock, MonotonicClock
from ..core.errors import LifecycleError, ModelValidationError
from ..core.types import RobotAction, RobotCommandType
from .base import RobotBackend

_REPEATED_COMMANDS = frozenset(
    {
        RobotCommandType.JOINT_POSITION,
        RobotCommandType.TCP_POSE,
        RobotCommandType.JOINT_VELOCITY,
        RobotCommandType.TCP_TWIST,
    }
)


@dataclass(frozen=True, slots=True)
class ExecutorSnapshot:
    """Small immutable health snapshot safe to publish to monitoring code."""

    running: bool
    terminal_stop: bool
    latest_sequence: int | None
    last_applied_sequence: int | None
    applied_count: int
    rejected_count: int
    last_applied_ns: int | None
    error: str | None


class LatestActionExecutor:
    """Apply one latest action at a configured cadence on the caller-owned thread.

    ``start()`` owns backend startup. ``run()`` is intentionally blocking so a
    runtime can choose and track the SDK-owner thread. ``execute_once()`` offers
    the same behavior for deterministic tests and externally scheduled loops.
    """

    def __init__(
        self,
        backend: RobotBackend,
        *,
        target_hz: float,
        repeat_active_commands: bool = True,
        clock: Clock | None = None,
    ) -> None:
        rate = float(target_hz)
        if not math.isfinite(rate) or rate <= 0:
            raise ModelValidationError("executor target_hz must be positive and finite")
        if not isinstance(backend, RobotBackend):
            raise ModelValidationError("backend must satisfy RobotBackend")
        self._backend = backend
        self._target_hz = rate
        self._repeat_active_commands = bool(repeat_active_commands)
        self._clock = clock or MonotonicClock()
        self._actions = LatestValueBuffer[RobotAction]()
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._running = False
        self._inflight = False
        self._terminal_stop = False
        self._last_applied_sequence: int | None = None
        self._applied_count = 0
        self._last_applied_ns: int | None = None
        self._error: str | None = None

    @property
    def backend(self) -> RobotBackend:
        return self._backend

    @property
    def target_hz(self) -> float:
        return self._target_hz

    def submit(self, action: RobotAction) -> bool:
        """Publish a new action, rejecting stale sequence numbers."""

        if not isinstance(action, RobotAction):
            raise ModelValidationError("executor action must be a RobotAction")
        with self._lock:
            if self._closed:
                raise LifecycleError("executor is closed")
            if self._terminal_stop:
                raise LifecycleError("executor received a terminal stop")
        return self._actions.publish(action)

    def start(self) -> None:
        """Start the owned backend without creating an untracked worker."""

        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed executor")
            if self._started:
                raise LifecycleError("executor is already started")
        self._backend.start()
        with self._lock:
            self._started = True

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("executor is closed")
        if not self._started:
            raise LifecycleError("executor has not been started")

    def execute_once(self) -> bool:
        """Apply the latest command once, returning whether an SDK call occurred."""

        with self._lock:
            self._require_started()
            if self._terminal_stop:
                return False
            action = self._actions.read()
            if action is None:
                return False
            is_new = action.sequence != self._last_applied_sequence
            should_repeat = (
                self._repeat_active_commands and action.command_type in _REPEATED_COMMANDS
            )
            if not is_new and not should_repeat:
                return False
            self._inflight = True
            if action.command_type is RobotCommandType.STOP:
                self._terminal_stop = True
        try:
            self._backend.apply_action(action)
        except Exception as exc:
            with self._lock:
                self._inflight = False
                self._error = f"{type(exc).__name__}: {exc}"
                self._stop_event.set()
            raise
        with self._lock:
            self._inflight = False
            self._last_applied_sequence = action.sequence
            self._last_applied_ns = self._clock.now_ns()
            self._applied_count += 1
            if action.command_type is RobotCommandType.STOP:
                self._terminal_stop = True
                self._stop_event.set()
        return True

    def run(self, external_stop: threading.Event | None = None) -> None:
        """Run at the target rate without burst catch-up after an overrun."""

        with self._lock:
            self._require_started()
            if self._running:
                raise LifecycleError("executor run loop is already active")
            self._running = True
        period_s = 1.0 / self._target_hz
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set() and not (
                external_stop is not None and external_stop.is_set()
            ):
                self.execute_once()
                deadline += period_s
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    deadline = time.monotonic()
                    continue
                self._stop_event.wait(remaining)
        finally:
            with self._lock:
                self._running = False

    def request_stop(self) -> None:
        """Stop the scheduler loop without closing the backend."""

        self._stop_event.set()

    def check_health(self) -> None:
        with self._lock:
            if self._error is not None:
                raise LifecycleError(f"executor failed: {self._error}")

    def snapshot(self) -> ExecutorSnapshot:
        with self._lock:
            return ExecutorSnapshot(
                running=self._running,
                terminal_stop=self._terminal_stop,
                latest_sequence=self._actions.latest_sequence,
                last_applied_sequence=self._last_applied_sequence,
                applied_count=self._applied_count,
                rejected_count=self._actions.rejected_count,
                last_applied_ns=self._last_applied_ns,
                error=self._error,
            )

    def close(self) -> None:
        """Stop scheduling and close the owned backend idempotently."""

        with self._lock:
            if self._closed:
                return
            if self._running or self._inflight:
                raise LifecycleError(
                    "stop and join the caller-owned executor thread before close"
                )
            self._closed = True
            self._started = False
            self._stop_event.set()
            self._actions.close()
        self._backend.close()
