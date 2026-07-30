"""Typed wrench-source boundary independent of robot and sensor SDKs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.types import WrenchSample


@runtime_checkable
class WrenchSource(Protocol):
    """Non-owning latest-sample view over one raw six-axis source."""

    def read_latest(self) -> WrenchSample | None:
        """Return the latest raw wrench snapshot or ``None`` when unavailable."""
