"""Backend-neutral gripper contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.interfaces import Lifecycle


@runtime_checkable
class Gripper(Lifecycle, Protocol):
    """Explicit-lifecycle width-controlled gripper."""

    @property
    def name(self) -> str:
        """Stable gripper name."""

    @property
    def max_width_m(self) -> float:
        """Maximum opening width in meters."""

    def read_width(self) -> float:
        """Read the current opening width in meters."""

    def move(self, width_m: float) -> None:
        """Command an opening width in meters."""
