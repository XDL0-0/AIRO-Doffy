"""Supported tactile sensor port independent of BLE and DDS libraries."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.interfaces import Lifecycle
from ...core.types import TactileSample


@runtime_checkable
class TactileSensor(Lifecycle, Protocol):
    """Latest-only tactile source with an explicit recalibration request."""

    def read_latest(self) -> TactileSample | None:
        """Return the newest 4-taxel sample without waiting."""

    def recalibrate(self) -> None:
        """Request or perform baseline recalibration."""
