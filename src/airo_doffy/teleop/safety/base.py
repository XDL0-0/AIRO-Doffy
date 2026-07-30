"""Composable robot-action safety filter port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.types import RobotAction, RobotState


@runtime_checkable
class ActionFilter(Protocol):
    """Validate, clip, transform, or reject an action deterministically."""

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        """Return a safe action, or ``None`` when the input is rejected."""
