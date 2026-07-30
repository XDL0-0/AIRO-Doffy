"""Deterministic VR-to-robot mapping port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.types import RobotAction, RobotState, VRInputState


@runtime_checkable
class TeleopMapping(Protocol):
    """Map validated VR and robot states without hardware access."""

    def map_input(
        self,
        vr_input: VRInputState,
        robot_state: RobotState,
        dt_s: float,
    ) -> RobotAction:
        """Produce one unfiltered robot action."""
