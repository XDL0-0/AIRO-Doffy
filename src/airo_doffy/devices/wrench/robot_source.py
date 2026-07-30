"""Expose a robot backend's raw wrench through the typed wrench port."""

from __future__ import annotations

from ...core.errors import ModelValidationError
from ...core.types import WrenchSample
from ...robots.base import RobotBackend


class RobotStateWrenchSource:
    """Read raw wrench snapshots without owning the robot backend lifecycle."""

    def __init__(self, backend: RobotBackend, *, frame_id: str) -> None:
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ModelValidationError("frame_id must be a non-empty string")
        self._backend = backend
        self._frame_id = frame_id

    def read_latest(self) -> WrenchSample | None:
        state = self._backend.read_state()
        if state.wrench is None:
            return None
        return WrenchSample(
            sequence=state.sequence,
            source_timestamp_ns=state.source_timestamp_ns,
            receive_timestamp_ns=state.receive_timestamp_ns,
            clock_domain=state.clock_domain,
            values=state.wrench,
            frame_id=self._frame_id,
        )
