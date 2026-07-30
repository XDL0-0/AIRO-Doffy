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


@runtime_checkable
class DepthCameraSource(CameraSource, Protocol):
    """Color source that can also expose an aligned or paired depth frame."""

    def read_latest_depth(self) -> CameraFrame | None:
        """Return the newest packed depth frame without waiting."""
