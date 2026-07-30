"""Cross-domain lifecycle protocols shared by domain-owned ports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Startable(Protocol):
    """Component that can acquire its resources explicitly."""

    def start(self) -> None:
        """Acquire resources and become ready."""


@runtime_checkable
class Closable(Protocol):
    """Component with an idempotent explicit shutdown."""

    def close(self) -> None:
        """Release resources; repeated calls must be safe."""


@runtime_checkable
class Lifecycle(Startable, Closable, Protocol):
    """Component supporting explicit start and idempotent close."""
