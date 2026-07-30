"""Video processing, encoding, and transport ports without codec SDK imports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.interfaces import Lifecycle
from ...core.types import CameraFrame, EncodedFrame, ProcessedFrame


@runtime_checkable
class FrameProcessor(Protocol):
    """Pure camera-frame transformation."""

    def process(self, frame: CameraFrame) -> ProcessedFrame:
        """Crop, resize, or convert one frame."""


@runtime_checkable
class VideoEncoder(Protocol):
    """Encoder that converts a processed frame to one access unit."""

    def encode(self, frame: ProcessedFrame) -> EncodedFrame:
        """Encode one frame without owning capture or transport."""


@runtime_checkable
class VideoEncodingPipeline(Lifecycle, Protocol):
    """Bounded asynchronous bridge from processed to encoded latest frames."""

    def submit(self, frame: ProcessedFrame) -> bool:
        """Submit a frame without blocking; return whether it was accepted."""

    def read_latest(self) -> EncodedFrame | None:
        """Return the newest encoded frame without waiting."""


@runtime_checkable
class VideoTransport(Lifecycle, Protocol):
    """Transport that sends encoded frames without interpreting pixels."""

    def send(self, frame: EncodedFrame) -> None:
        """Send one encoded access unit."""
