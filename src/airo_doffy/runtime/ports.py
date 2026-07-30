"""Runtime-only orchestration ports."""

from __future__ import annotations

from threading import Event
from typing import Protocol, runtime_checkable

from ..core.events import RuntimeCommand, RuntimeEvent
from ..core.interfaces import Lifecycle
from ..core.types import RobotAction, RobotState


@runtime_checkable
class RobotStateSource(Protocol):
    """Provide immutable robot state without transferring lifecycle ownership."""

    def read_state(self) -> RobotState:
        """Return the newest robot state."""


@runtime_checkable
class ActionExecutor(Lifecycle, Protocol):
    """Caller-owned action execution loop."""

    def submit(self, action: RobotAction) -> bool:
        """Submit a safe latest action."""

    def run(self, external_stop: Event | None = None) -> None:
        """Run until the executor or external stop is set."""


@runtime_checkable
class CommandSource(Protocol):
    """Drain typed commands from one input path."""

    def drain(self) -> tuple[RuntimeCommand, ...]:
        """Return currently queued commands without waiting."""


@runtime_checkable
class CommandDispatcher(Protocol):
    """Route one typed command to injected handlers."""

    def dispatch(self, command: RuntimeCommand) -> RuntimeEvent:
        """Dispatch a command and return an observable event."""


@runtime_checkable
class SessionExtension(Lifecycle, Protocol):
    """Optional session behavior that observes each completed teleop cycle."""

    def on_cycle(self, cycle: object) -> None:
        """Observe one immutable cycle without owning the teleop loop."""
