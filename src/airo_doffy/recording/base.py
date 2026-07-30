"""Episode recording port independent of HDF5 and LeRobot serializers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.interfaces import Closable
from ..core.types import Observation


@runtime_checkable
class EpisodeRecorder(Closable, Protocol):
    """Append immutable observations to an explicitly bounded episode."""

    def start_episode(self) -> None:
        """Begin a new episode."""

    def append(self, observation: Observation) -> None:
        """Append one immutable observation."""

    def finish_episode(self) -> int:
        """Finalize the episode and return its stable index."""

    def rollback_last_episode(self) -> bool:
        """Remove or mark the latest completed episode for rollback."""
