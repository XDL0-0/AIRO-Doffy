"""Bounded typed command outbox for interactive visualizers."""

from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Lock

from ..core.errors import ModelValidationError
from ..core.events import RuntimeCommand


class VisualizationCommandOutbox:
    """Buffer UI commands without owning their runtime handlers."""

    def __init__(self, *, capacity: int = 8) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ModelValidationError("capacity must be a positive integer")
        self._queue: Queue[RuntimeCommand] = Queue(maxsize=capacity)
        self._lock = Lock()
        self._closed = False
        self._accepted = 0
        self._rejected = 0

    @property
    def accepted_count(self) -> int:
        with self._lock:
            return self._accepted

    @property
    def rejected_count(self) -> int:
        with self._lock:
            return self._rejected

    def submit(self, command: RuntimeCommand) -> bool:
        if not isinstance(command, RuntimeCommand):
            raise ModelValidationError(
                "visualization command outbox accepts only RuntimeCommand"
            )
        with self._lock:
            if self._closed:
                self._rejected += 1
                return False
            try:
                self._queue.put_nowait(command)
            except Full:
                self._rejected += 1
                return False
            self._accepted += 1
            return True

    def drain(self, *, limit: int | None = None) -> tuple[RuntimeCommand, ...]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ModelValidationError("limit must be a non-negative integer or None")
        commands = []
        while limit is None or len(commands) < limit:
            try:
                commands.append(self._queue.get_nowait())
            except Empty:
                break
        return tuple(commands)

    def close(self) -> None:
        with self._lock:
            self._closed = True
