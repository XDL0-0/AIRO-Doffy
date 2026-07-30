"""Backend-neutral atomic robot capabilities without vendor SDK imports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.interfaces import Lifecycle
from ..core.types import RobotAction, RobotState


@runtime_checkable
class RobotBackend(Lifecycle, Protocol):
    """Atomic state and command port; control cadence belongs to an executor."""

    @property
    def name(self) -> str:
        """Stable backend name."""

    @property
    def dof(self) -> int:
        """Robot arm degrees of freedom."""

    def read_state(self) -> RobotState:
        """Read one immutable state snapshot."""

    def apply_action(self, action: RobotAction) -> None:
        """Apply one already validated atomic action."""
