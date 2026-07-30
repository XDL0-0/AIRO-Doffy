"""VR input source port independent of sockets and protocol serialization."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.interfaces import Lifecycle
from ...core.types import VRInputState


@runtime_checkable
class VRInputSource(Lifecycle, Protocol):
    """Explicitly started latest-only controller or hand input source."""

    def read_latest(self) -> VRInputState | None:
        """Return the newest validated VR input state without waiting."""
