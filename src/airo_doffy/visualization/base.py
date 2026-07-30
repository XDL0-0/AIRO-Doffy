"""Narrow visualization ports with no GUI dependency."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.events import RuntimeCommand
from .models import VisualizationSnapshot


@runtime_checkable
class SnapshotRenderer(Protocol):
    """Render typed snapshots without reading application components."""

    def start(self) -> None:
        """Acquire renderer resources."""

    def render(self, snapshot: VisualizationSnapshot) -> bool:
        """Render one snapshot and return false when the view has closed."""

    def close(self) -> None:
        """Release renderer resources; repeated calls must be safe."""


@runtime_checkable
class SnapshotConsumer(Protocol):
    """Latest-only visualization input owned by the runtime."""

    def start(self) -> None:
        """Start consuming snapshots."""

    def publish(self, snapshot: VisualizationSnapshot) -> bool:
        """Publish a snapshot without blocking the producer."""

    def close(self) -> None:
        """Stop consuming and release resources."""


@runtime_checkable
class VisualizationCommandSink(Protocol):
    """Typed command boundary injected into interactive renderers."""

    def submit(self, command: RuntimeCommand) -> bool:
        """Submit one UI command without mutating runtime state directly."""
