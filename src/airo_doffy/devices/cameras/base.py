"""Camera acquisition port independent of camera vendor SDKs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.interfaces import Lifecycle
from ...core.types import CameraFrame


@runtime_checkable
class CameraSource(Lifecycle, Protocol):
    """Explicitly started source providing latest-only camera frames."""

    def read_latest(self) -> CameraFrame | None:
        """Return the newest captured frame without waiting."""
