"""TCP versus joint command selection through an injected IK boundary."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...core.errors import ModelValidationError
from ...core.types import RobotAction, RobotCommandType, RobotState
from ..transforms import validate_transform


@runtime_checkable
class InverseKinematicsSolver(Protocol):
    """Pure/injected IK boundary; implementations must not command hardware."""

    def solve(
        self,
        tcp_pose: tuple[tuple[float, ...], ...],
        seed_joints_rad: tuple[float, ...],
    ) -> tuple[float, ...] | None:
        """Return one joint solution or ``None`` when no solution is valid."""


@dataclass(frozen=True, slots=True)
class CommandModeMetrics:
    """Snapshot of TCP selections, joint selections, and IK rejections."""

    tcp_selected: int
    joint_selected: int
    ik_rejected: int


class CommandModeSelector:
    """Convert mapped TCP actions to the configured backend-neutral command type."""

    def __init__(
        self,
        mode: str,
        *,
        ik_solver: InverseKinematicsSolver | None = None,
    ) -> None:
        checked_mode = mode.lower()
        if checked_mode not in {"joint", "tcp"}:
            raise ModelValidationError("command mode must be 'joint' or 'tcp'")
        if checked_mode == "joint" and (
            ik_solver is None or not isinstance(ik_solver, InverseKinematicsSolver)
        ):
            raise ModelValidationError("joint command mode requires an IK solver")
        self._mode = checked_mode
        self._ik_solver = ik_solver
        self._lock = threading.Lock()
        self._tcp_selected = 0
        self._joint_selected = 0
        self._ik_rejected = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def metrics(self) -> CommandModeMetrics:
        with self._lock:
            return CommandModeMetrics(
                tcp_selected=self._tcp_selected,
                joint_selected=self._joint_selected,
                ik_rejected=self._ik_rejected,
            )

    def select(
        self,
        tcp_action: RobotAction,
        robot_state: RobotState,
    ) -> RobotAction | None:
        """Return TCP directly or a validated joint action from injected IK."""

        if (
            not isinstance(tcp_action, RobotAction)
            or tcp_action.command_type is not RobotCommandType.TCP_POSE
        ):
            raise ModelValidationError("command selector requires a TCP_POSE action")
        if not isinstance(robot_state, RobotState):
            raise ModelValidationError("command selector requires RobotState")
        if self._mode == "tcp":
            with self._lock:
                self._tcp_selected += 1
            return tcp_action
        assert self._ik_solver is not None
        tcp_pose = validate_transform(
            tuple(
                tuple(tcp_action.values[row * 4 + column] for column in range(4))
                for row in range(4)
            )
        )
        try:
            solution = self._ik_solver.solve(tcp_pose, robot_state.joints_rad)
            if solution is None:
                raise ModelValidationError("IK returned no solution")
            selected = RobotAction(
                sequence=tcp_action.sequence,
                source_timestamp_ns=tcp_action.source_timestamp_ns,
                receive_timestamp_ns=tcp_action.receive_timestamp_ns,
                clock_domain=tcp_action.clock_domain,
                command_type=RobotCommandType.JOINT_POSITION,
                values=solution,
                duration_s=tcp_action.duration_s,
                gripper_width_m=tcp_action.gripper_width_m,
            )
            if len(selected.values) != len(robot_state.joints_rad):
                raise ModelValidationError("IK solution DOF does not match robot state")
        except (TypeError, ValueError, ModelValidationError):
            with self._lock:
                self._ik_rejected += 1
            return None
        with self._lock:
            self._joint_selected += 1
        return selected
