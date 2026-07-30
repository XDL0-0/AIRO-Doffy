"""Typed runtime commands and observable events for reliable control paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from .errors import ModelValidationError
from .types import ClockDomain


class RuntimeCommandType(str, Enum):
    """Reliable commands currently produced by VR, UI, and runtime clients."""

    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    ROLLBACK_LAST_EPISODE = "rollback_last_episode"
    RECALIBRATE_TACTILE = "recalibrate_tactile"
    RESET_TELEOP_REFERENCE = "reset_teleop_reference"
    PAUSE = "pause"
    RESUME = "resume"
    SET_VIDEO_PROFILE = "set_video_profile"
    SHUTDOWN = "shutdown"


class RuntimeEventType(str, Enum):
    """Lifecycle, command, recording, and safety events."""

    COMPONENT_STARTED = "component_started"
    COMPONENT_STOPPED = "component_stopped"
    COMMAND_ACCEPTED = "command_accepted"
    COMMAND_REJECTED = "command_rejected"
    RECORDING_STARTED = "recording_started"
    RECORDING_STOPPED = "recording_stopped"
    ROLLBACK_COMPLETED = "rollback_completed"
    WATCHDOG_TRIPPED = "watchdog_tripped"
    COMPONENT_ERROR = "component_error"
    SHUTDOWN_COMPLETE = "shutdown_complete"


class RuntimeEventSeverity(str, Enum):
    """Portable severity independent of a logging framework."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelValidationError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeCommand:
    """One idempotency-addressable command on the reliable command path."""

    kind: RuntimeCommandType
    sequence: int
    source_timestamp_ns: int
    value: str | None = None
    origin: str = "unknown"
    command_id: str = field(default_factory=lambda: uuid4().hex)
    clock_domain: ClockDomain = ClockDomain.UNSPECIFIED

    def __post_init__(self) -> None:
        try:
            kind = RuntimeCommandType(self.kind)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError(f"unsupported runtime command: {self.kind!r}") from exc
        if kind is RuntimeCommandType.SET_VIDEO_PROFILE:
            if not isinstance(self.value, str) or not self.value.strip():
                raise ModelValidationError("set_video_profile requires a non-empty string value")
        elif self.value is not None:
            raise ModelValidationError(f"{kind.value} does not accept a value")
        if not isinstance(self.origin, str) or not self.origin.strip():
            raise ModelValidationError("origin must be a non-empty string")
        if not isinstance(self.command_id, str) or not self.command_id.strip():
            raise ModelValidationError("command_id must be a non-empty string")
        try:
            domain = ClockDomain(self.clock_domain)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError(f"unsupported clock domain: {self.clock_domain!r}") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "sequence", _non_negative_int(self.sequence, "sequence"))
        object.__setattr__(
            self,
            "source_timestamp_ns",
            _non_negative_int(self.source_timestamp_ns, "source_timestamp_ns"),
        )
        object.__setattr__(self, "clock_domain", domain)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeEvent:
    """Immutable event emitted by runtime components and command routing."""

    kind: RuntimeEventType
    sequence: int
    timestamp_ns: int
    severity: RuntimeEventSeverity = RuntimeEventSeverity.INFO
    component: str = "runtime"
    message: str = ""
    command_id: str | None = None
    details: tuple[tuple[str, str], ...] = ()
    clock_domain: ClockDomain = ClockDomain.MONOTONIC

    def __post_init__(self) -> None:
        try:
            kind = RuntimeEventType(self.kind)
            severity = RuntimeEventSeverity(self.severity)
            domain = ClockDomain(self.clock_domain)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("invalid runtime event enum value") from exc
        if not isinstance(self.component, str) or not self.component.strip():
            raise ModelValidationError("component must be a non-empty string")
        if not isinstance(self.message, str):
            raise ModelValidationError("message must be a string")
        if self.command_id is not None and (
            not isinstance(self.command_id, str) or not self.command_id.strip()
        ):
            raise ModelValidationError("command_id must be None or a non-empty string")
        try:
            details = tuple((str(key), str(value)) for key, value in self.details)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("details must contain key/value pairs") from exc
        keys = [key for key, _ in details]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ModelValidationError("event detail keys must be non-empty and unique")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "sequence", _non_negative_int(self.sequence, "sequence"))
        object.__setattr__(
            self,
            "timestamp_ns",
            _non_negative_int(self.timestamp_ns, "timestamp_ns"),
        )
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "clock_domain", domain)
