"""Injectable clock contracts with explicit monotonic and wall-clock sources."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Nanosecond clock used by components that need deterministic testing."""

    def now_ns(self) -> int:
        """Return the current time in this clock's documented domain."""


class MonotonicClock:
    """Steady process-local clock suitable for durations and freshness."""

    __slots__ = ()

    def now_ns(self) -> int:
        return time.monotonic_ns()


class WallClock:
    """Unix wall clock for externally meaningful timestamps."""

    __slots__ = ()

    def now_ns(self) -> int:
        return time.time_ns()
