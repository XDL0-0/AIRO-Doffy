"""Dependency-injected routing for typed runtime commands."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...core.clocks import Clock, MonotonicClock
from ...core.errors import CommandRejectedError, ModelValidationError
from ...core.events import (
    RuntimeCommand,
    RuntimeCommandType,
    RuntimeEvent,
    RuntimeEventSeverity,
    RuntimeEventType,
)
from ...core.types import ClockDomain

CommandHandler = Callable[[RuntimeCommand], str | None]


@dataclass(frozen=True, slots=True)
class CommandRouterMetrics:
    """Snapshot of accepted, rejected, failed, and unhandled dispatches."""

    dispatched: int
    accepted: int
    rejected: int
    errors: int
    unhandled: int


class CommandRouter:
    """Route command enums to injected handlers without component knowledge."""

    def __init__(
        self,
        handlers: Mapping[RuntimeCommandType, CommandHandler],
        *,
        clock: Clock | None = None,
        component: str = "command_router",
    ) -> None:
        if not isinstance(component, str) or not component.strip():
            raise ModelValidationError("router component must be a non-empty string")
        checked: dict[RuntimeCommandType, CommandHandler] = {}
        try:
            items = handlers.items()
        except AttributeError as exc:
            raise ModelValidationError("handlers must be a command mapping") from exc
        for raw_kind, handler in items:
            try:
                kind = RuntimeCommandType(raw_kind)
            except (TypeError, ValueError) as exc:
                raise ModelValidationError(
                    f"unsupported command handler key: {raw_kind!r}"
                ) from exc
            if not callable(handler):
                raise ModelValidationError(f"handler for {kind.value} must be callable")
            checked[kind] = handler
        self._handlers = checked
        self._clock = clock or MonotonicClock()
        self._component = component
        self._lock = threading.Lock()
        self._event_sequence = 0
        self._dispatched = 0
        self._accepted = 0
        self._rejected = 0
        self._errors = 0
        self._unhandled = 0

    @property
    def metrics(self) -> CommandRouterMetrics:
        with self._lock:
            return CommandRouterMetrics(
                dispatched=self._dispatched,
                accepted=self._accepted,
                rejected=self._rejected,
                errors=self._errors,
                unhandled=self._unhandled,
            )

    def _next_sequence(self) -> int:
        with self._lock:
            sequence = self._event_sequence
            self._event_sequence += 1
            self._dispatched += 1
            return sequence

    def _record_outcome(
        self,
        *,
        accepted: bool = False,
        rejected: bool = False,
        error: bool = False,
        unhandled: bool = False,
    ) -> None:
        with self._lock:
            self._accepted += int(accepted)
            self._rejected += int(rejected)
            self._errors += int(error)
            self._unhandled += int(unhandled)

    def _event(
        self,
        command: RuntimeCommand,
        *,
        sequence: int,
        kind: RuntimeEventType,
        severity: RuntimeEventSeverity,
        message: str,
        extra_details: tuple[tuple[str, str], ...] = (),
    ) -> RuntimeEvent:
        return RuntimeEvent(
            kind=kind,
            sequence=sequence,
            timestamp_ns=self._clock.now_ns(),
            severity=severity,
            component=self._component,
            message=message,
            command_id=command.command_id,
            details=(
                ("command_kind", command.kind.value),
                ("origin", command.origin),
                *extra_details,
            ),
            clock_domain=ClockDomain.MONOTONIC,
        )

    def dispatch(self, command: RuntimeCommand) -> RuntimeEvent:
        """Invoke one selected handler and return an accepted/rejected event."""

        if not isinstance(command, RuntimeCommand):
            raise ModelValidationError("command must be a RuntimeCommand")
        sequence = self._next_sequence()
        handler = self._handlers.get(command.kind)
        if handler is None:
            self._record_outcome(rejected=True, unhandled=True)
            return self._event(
                command,
                sequence=sequence,
                kind=RuntimeEventType.COMMAND_REJECTED,
                severity=RuntimeEventSeverity.WARNING,
                message=f"no handler registered for {command.kind.value}",
            )
        try:
            message = handler(command)
            if message is not None and not isinstance(message, str):
                raise TypeError("command handler must return str or None")
        except CommandRejectedError as exc:
            self._record_outcome(rejected=True)
            return self._event(
                command,
                sequence=sequence,
                kind=RuntimeEventType.COMMAND_REJECTED,
                severity=RuntimeEventSeverity.WARNING,
                message=str(exc) or "command rejected",
            )
        except Exception as exc:
            self._record_outcome(rejected=True, error=True)
            return self._event(
                command,
                sequence=sequence,
                kind=RuntimeEventType.COMMAND_REJECTED,
                severity=RuntimeEventSeverity.ERROR,
                message=f"command handler failed: {type(exc).__name__}: {exc}",
                extra_details=(("error_type", type(exc).__name__),),
            )
        self._record_outcome(accepted=True)
        return self._event(
            command,
            sequence=sequence,
            kind=RuntimeEventType.COMMAND_ACCEPTED,
            severity=RuntimeEventSeverity.INFO,
            message=message or f"routed {command.kind.value}",
        )
