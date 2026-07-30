"""Composable teleoperation session orchestration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from threading import Event, RLock

from ..core.clocks import Clock, MonotonicClock
from ..core.errors import LifecycleError, ModelValidationError
from ..core.events import RuntimeEvent
from ..core.types import (
    ClockDomain,
    RobotAction,
    RobotCommandType,
    RobotState,
    VRInputState,
)
from ..devices.vr.base import VRInputSource
from ..teleop.mappings.base import TeleopMapping
from ..teleop.safety.base import ActionFilter
from ..teleop.safety.watchdog import (
    TeleopWatchdog,
    WatchdogDecision,
    WatchdogState,
)
from .lifecycle import LifecycleManager, ManagedWorker
from .ports import (
    ActionExecutor,
    CommandDispatcher,
    CommandSource,
    RobotStateSource,
    SessionExtension,
)


@dataclass(frozen=True, slots=True)
class TeleopCycle:
    """One immutable result of runtime coordination."""

    sequence: int
    timestamp_ns: int
    dt_s: float
    vr_input: VRInputState | None
    robot_state: RobotState
    mapped_action: RobotAction | None
    safe_action: RobotAction | None
    submitted: bool
    watchdog: WatchdogDecision | None = None
    events: tuple[RuntimeEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class TeleopSessionMetrics:
    """Small immutable control-loop status."""

    running: bool
    cycles: int
    mapped_actions: int
    submitted_actions: int
    rejected_actions: int
    command_events: int
    last_cycle_ns: int | None
    last_error: str | None


class TeleopSession:
    """Coordinate typed components without implementing their internals."""

    def __init__(
        self,
        *,
        vr_source: VRInputSource,
        state_source: RobotStateSource,
        mapping: TeleopMapping,
        action_filter: ActionFilter,
        executor: ActionExecutor,
        target_hz: float,
        watchdog: TeleopWatchdog | None = None,
        extensions: tuple[SessionExtension, ...] = (),
        command_source: CommandSource | None = None,
        command_dispatcher: CommandDispatcher | None = None,
        clock: Clock | None = None,
        max_dt_s: float = 0.05,
    ) -> None:
        rate = float(target_hz)
        if not math.isfinite(rate) or rate <= 0:
            raise ModelValidationError("target_hz must be positive and finite")
        if not math.isfinite(max_dt_s) or max_dt_s <= 0:
            raise ModelValidationError("max_dt_s must be positive and finite")
        if not isinstance(vr_source, VRInputSource):
            raise ModelValidationError("vr_source must satisfy VRInputSource")
        if not isinstance(state_source, RobotStateSource):
            raise ModelValidationError("state_source must satisfy RobotStateSource")
        if not isinstance(mapping, TeleopMapping):
            raise ModelValidationError("mapping must satisfy TeleopMapping")
        if not isinstance(action_filter, ActionFilter):
            raise ModelValidationError("action_filter must satisfy ActionFilter")
        if not isinstance(executor, ActionExecutor):
            raise ModelValidationError("executor must satisfy ActionExecutor")
        checked_extensions = tuple(extensions)
        if any(
            not isinstance(extension, SessionExtension)
            for extension in checked_extensions
        ):
            raise ModelValidationError(
                "extensions must satisfy SessionExtension"
            )
        if (command_source is None) != (command_dispatcher is None):
            raise ModelValidationError(
                "command_source and command_dispatcher must be configured together"
            )
        if command_source is not None and not isinstance(
            command_source,
            CommandSource,
        ):
            raise ModelValidationError("command_source must satisfy CommandSource")
        if command_dispatcher is not None and not isinstance(
            command_dispatcher,
            CommandDispatcher,
        ):
            raise ModelValidationError(
                "command_dispatcher must satisfy CommandDispatcher"
            )

        self._vr_source = vr_source
        self._state_source = state_source
        self._mapping = mapping
        self._action_filter = action_filter
        self._executor = executor
        self._target_hz = rate
        self._watchdog = watchdog
        self._extensions = checked_extensions
        self._command_source = command_source
        self._command_dispatcher = command_dispatcher
        self._clock = clock or MonotonicClock()
        self._max_dt_s = float(max_dt_s)
        self._stop = Event()
        self._run_finished = Event()
        self._run_finished.set()
        self._lock = RLock()
        self._started = False
        self._closed = False
        self._running = False
        self._cycle_sequence = 0
        self._last_step_ns: int | None = None
        self._last_candidate: RobotAction | None = None
        self._last_output_sequence: int | None = None
        self._cycles = 0
        self._mapped_actions = 0
        self._submitted_actions = 0
        self._rejected_actions = 0
        self._command_events = 0
        self._last_cycle_ns: int | None = None
        self._last_error: str | None = None

        self._executor_worker = ManagedWorker(
            lambda stop: self._executor.run(stop),
            name="robot-action-executor",
        )
        resources = [
            ("action_executor", executor),
            ("action_executor_worker", self._executor_worker),
            ("vr_source", vr_source),
        ]
        resources.extend(
            (f"extension_{index}", extension)
            for index, extension in enumerate(checked_extensions)
        )
        self._lifecycle = LifecycleManager(resources)

    @property
    def extensions(self) -> tuple[SessionExtension, ...]:
        return self._extensions

    def metrics(self) -> TeleopSessionMetrics:
        with self._lock:
            return TeleopSessionMetrics(
                running=self._running,
                cycles=self._cycles,
                mapped_actions=self._mapped_actions,
                submitted_actions=self._submitted_actions,
                rejected_actions=self._rejected_actions,
                command_events=self._command_events,
                last_cycle_ns=self._last_cycle_ns,
                last_error=self._last_error,
            )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed teleop session")
            if self._started:
                raise LifecycleError("teleop session is already started")
        self._lifecycle.start()
        with self._lock:
            self._started = True

    def step_once(self) -> TeleopCycle:
        with self._lock:
            self._require_started()
            if self._running and self._stop.is_set():
                raise LifecycleError("teleop session is stopping")
            now_ns = self._clock.now_ns()
            dt_s = self._step_dt(now_ns)
            sequence = self._cycle_sequence
            self._cycle_sequence += 1

        events = self._dispatch_commands()
        robot_state = self._state_source.read_state()
        vr_input = self._vr_source.read_latest()
        decision = self._evaluate_watchdog(vr_input, robot_state, now_ns)
        mapped = None
        safe = None

        if vr_input is not None and (
            decision is None or decision.state is WatchdogState.ACTIVE
        ):
            mapped = self._mapping.map_input(vr_input, robot_state, dt_s)
            self._last_candidate = mapped
            filtered = self._action_filter.apply(mapped, robot_state, now_ns)
            safe = filtered if filtered is not None else self._hold_action(now_ns)
            if filtered is None:
                with self._lock:
                    self._rejected_actions += 1
        elif (
            vr_input is not None
            and decision is not None
            and decision.reference_ready
        ):
            mapped = self._mapping.map_input(vr_input, robot_state, dt_s)
            self._last_candidate = mapped
            self._watchdog.acknowledge_reference(
                vr_sequence=vr_input.sequence,
                robot_sequence=robot_state.sequence,
            )
            decision = self._watchdog.evaluate(
                vr_input,
                robot_state,
                now_ns=now_ns,
            )
            filtered = self._action_filter.apply(mapped, robot_state, now_ns)
            safe = filtered if filtered is not None else self._hold_action(now_ns)
            if filtered is None:
                with self._lock:
                    self._rejected_actions += 1
        elif decision is not None:
            safe = self._last_candidate or self._hold_action(now_ns)
        elif vr_input is None:
            safe = self._hold_action(now_ns)

        if mapped is not None:
            with self._lock:
                self._mapped_actions += 1
        if decision is not None and safe is not None:
            safe = self._watchdog.guard(safe, decision)
        elif safe is not None:
            safe = self._monotonic_action(safe)
        submitted = False
        if safe is not None:
            submitted = self._executor.submit(safe)
            self._remember_output_sequence(safe.sequence)
            with self._lock:
                self._submitted_actions += int(submitted)
                self._rejected_actions += int(not submitted)

        cycle = TeleopCycle(
            sequence=sequence,
            timestamp_ns=now_ns,
            dt_s=dt_s,
            vr_input=vr_input,
            robot_state=robot_state,
            mapped_action=mapped,
            safe_action=safe,
            submitted=submitted,
            watchdog=decision,
            events=events,
        )
        for extension in self._extensions:
            extension.on_cycle(cycle)
        with self._lock:
            self._cycles += 1
            self._last_cycle_ns = now_ns
        return cycle

    def run(self, *, max_cycles: int | None = None) -> None:
        if max_cycles is not None and (
            isinstance(max_cycles, bool)
            or not isinstance(max_cycles, int)
            or max_cycles < 0
        ):
            raise ModelValidationError("max_cycles must be a non-negative integer or None")
        with self._lock:
            self._require_started()
            if self._running:
                raise LifecycleError("teleop session loop is already running")
            self._running = True
            self._run_finished.clear()
        period_s = 1.0 / self._target_hz
        deadline = time.monotonic()
        completed = 0
        try:
            while not self._stop.is_set() and (
                max_cycles is None or completed < max_cycles
            ):
                self.step_once()
                completed += 1
                self.check_health()
                deadline += period_s
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    deadline = time.monotonic()
                else:
                    self._stop.wait(remaining)
        except BaseException as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._stop.set()
            raise
        finally:
            with self._lock:
                self._running = False
            self._run_finished.set()

    def check_health(self) -> None:
        self._executor_worker.check_health()
        check_executor = getattr(self._executor, "check_health", None)
        if callable(check_executor):
            check_executor()
        for extension in self._extensions:
            check_extension = getattr(extension, "check_health", None)
            if callable(check_extension):
                check_extension()

    def request_stop(self) -> None:
        self._stop.set()

    def close(self, *, timeout_s: float = 5.0) -> None:
        if timeout_s < 0:
            raise ModelValidationError("timeout_s must be non-negative")
        with self._lock:
            if self._closed:
                return
            self._stop.set()
            running = self._running
        if running and not self._run_finished.wait(timeout_s):
            raise LifecycleError("teleop session loop did not stop before timeout")
        try:
            self._lifecycle.close()
        finally:
            with self._lock:
                self._closed = True
                self._started = False

    def _dispatch_commands(self) -> tuple[RuntimeEvent, ...]:
        if self._command_source is None or self._command_dispatcher is None:
            return ()
        events = tuple(
            self._command_dispatcher.dispatch(command)
            for command in self._command_source.drain()
        )
        with self._lock:
            self._command_events += len(events)
        return events

    def _evaluate_watchdog(
        self,
        vr_input: VRInputState | None,
        robot_state: RobotState,
        now_ns: int,
    ) -> WatchdogDecision | None:
        if self._watchdog is None:
            return None
        return self._watchdog.evaluate(
            vr_input,
            robot_state,
            now_ns=now_ns,
        )

    def _hold_action(self, now_ns: int) -> RobotAction:
        with self._lock:
            sequence = (
                0
                if self._last_output_sequence is None
                else self._last_output_sequence + 1
            )
        return RobotAction(
            sequence=sequence,
            source_timestamp_ns=now_ns,
            clock_domain=ClockDomain.MONOTONIC,
            command_type=RobotCommandType.HOLD,
            values=(),
        )

    def _remember_output_sequence(self, sequence: int) -> None:
        with self._lock:
            if (
                self._last_output_sequence is None
                or sequence > self._last_output_sequence
            ):
                self._last_output_sequence = sequence

    def _monotonic_action(self, action: RobotAction) -> RobotAction:
        with self._lock:
            if (
                self._last_output_sequence is None
                or action.sequence > self._last_output_sequence
            ):
                return action
            return replace(
                action,
                sequence=self._last_output_sequence + 1,
            )

    def _step_dt(self, now_ns: int) -> float:
        if self._last_step_ns is None:
            result = min(1.0 / self._target_hz, self._max_dt_s)
        else:
            result = min(
                max((now_ns - self._last_step_ns) / 1_000_000_000, 0.0),
                self._max_dt_s,
            )
        self._last_step_ns = now_ns
        return result

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("teleop session is closed")
        if not self._started:
            raise LifecycleError("teleop session has not been started")
