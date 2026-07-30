"""IK conversion filter using the same dependency-injected solver boundary."""

from __future__ import annotations

from ...core.errors import ModelValidationError
from ...core.types import RobotAction, RobotCommandType, RobotState
from ..mappings import CommandModeSelector, InverseKinematicsSolver


class InverseKinematicsFilter:
    """Convert TCP poses to joint positions or reject invalid IK results."""

    def __init__(self, solver: InverseKinematicsSolver) -> None:
        self._selector = CommandModeSelector("joint", ik_solver=solver)

    @property
    def rejection_count(self) -> int:
        return self._selector.metrics.ik_rejected

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        del now_ns
        if not isinstance(action, RobotAction) or not isinstance(robot_state, RobotState):
            raise ModelValidationError("IK filter requires RobotAction and RobotState")
        if action.command_type is not RobotCommandType.TCP_POSE:
            return action
        return self._selector.select(action, robot_state)
