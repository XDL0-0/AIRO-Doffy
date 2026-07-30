"""Hardware-free visualization renderer for tests and mock sessions."""

from __future__ import annotations

from threading import Lock

from ..core.errors import LifecycleError, ModelValidationError
from .models import VisualizationSnapshot


class MemorySnapshotRenderer:
    """Collect snapshots in memory and optionally emulate a closed window."""

    def __init__(self, *, close_after: int | None = None) -> None:
        if close_after is not None and (
            isinstance(close_after, bool)
            or not isinstance(close_after, int)
            or close_after <= 0
        ):
            raise ModelValidationError("close_after must be a positive integer or None")
        self._close_after = close_after
        self._snapshots: list[VisualizationSnapshot] = []
        self._started = False
        self._closed = False
        self._lock = Lock()

    @property
    def snapshots(self) -> tuple[VisualizationSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshots)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("closed memory renderer cannot start")
            if self._started:
                raise LifecycleError("memory renderer is already started")
            self._started = True

    def render(self, snapshot: VisualizationSnapshot) -> bool:
        with self._lock:
            if not self._started or self._closed:
                return False
            self._snapshots.append(snapshot)
            return (
                self._close_after is None
                or len(self._snapshots) < self._close_after
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._started = False
