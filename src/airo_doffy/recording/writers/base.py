"""Narrow serializer and rollback ports."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..samples import Episode


@runtime_checkable
class EpisodeWriter(Protocol):
    """Serialize complete immutable episodes without reading hardware."""

    def write(self, episode: Episode) -> Path | None:
        """Persist one episode and return its primary path when applicable."""

    def close(self) -> None:
        """Release serializer resources; repeated calls must be safe."""


@runtime_checkable
class EpisodeRollback(Protocol):
    """Remove one explicitly indexed persisted episode."""

    def rollback(self, episode_index: int) -> bool:
        """Remove an episode and return whether storage changed."""
