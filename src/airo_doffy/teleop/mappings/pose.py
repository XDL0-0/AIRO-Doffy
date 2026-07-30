"""Stateful controller and hand reference mapping built from pure pose operations."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from ...config.models import TeleopConfig
from ...core.errors import ModelValidationError
from ...core.types import (
    ClockDomain,
    ControllerState,
    HandSide,
    HandState,
    RobotAction,
    RobotCommandType,
    RobotState,
    VRInputMode,
    VRInputState,
)
from ..transforms import (
    RotationComposition,
    Transform4,
    flatten_transform,
    map_relative_pose,
    validate_transform,
    vr_pose_to_transform,
)
from .command_mode import CommandModeSelector
from .gripper import (
    ControllerGripperMapping,
    GripperDirection,
    HandGripperMapping,
    IncrementalGripperMapper,
)

_THUMB_TIP_INDEX = 5
_INDEX_TIP_INDEX = 10


@dataclass(frozen=True, slots=True)
class TeleopReference:
    """One source pose paired with the robot pose measured at engagement."""

    source_pose: Transform4
    robot_pose: Transform4


@dataclass(frozen=True, slots=True)
class PoseMappingMetrics:
    """Snapshot of mapped, hold, rebase, missing, jump, and IK outcomes."""

    mapped: int
    holds: int
    rebases: int
    missing_input: int
    jump_rejections: int
    ik_rejections: int


def _positive_dt(dt_s: float) -> float:
    value = float(dt_s)
    if not math.isfinite(value) or value <= 0:
        raise ModelValidationError("dt_s must be positive and finite")
    return value


def _metadata(vr_input: VRInputState) -> dict[str, object]:
    return {
        "sequence": vr_input.sequence,
        "source_timestamp_ns": vr_input.source_timestamp_ns,
        "receive_timestamp_ns": vr_input.receive_timestamp_ns,
        "clock_domain": vr_input.clock_domain,
    }


def _hold(vr_input: VRInputState, gripper_width_m: float | None) -> RobotAction:
    return RobotAction(
        **_metadata(vr_input),
        command_type=RobotCommandType.HOLD,
        values=(),
        gripper_width_m=gripper_width_m,
    )


def _finger_distance(hand: HandState) -> float:
    thumb = hand.joints_m[_THUMB_TIP_INDEX]
    index = hand.joints_m[_INDEX_TIP_INDEX]
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(thumb, index)))


class _RelativePoseMapping:
    def __init__(
        self,
        config: TeleopConfig,
        command_selector: CommandModeSelector,
        *,
        gripper_speed_m_s: float,
        gripper_max_width_m: float,
    ) -> None:
        if not isinstance(config, TeleopConfig):
            raise ModelValidationError("mapping config must be TeleopConfig")
        if not isinstance(command_selector, CommandModeSelector):
            raise ModelValidationError("mapping requires CommandModeSelector")
        if command_selector.mode != config.command_mode:
            raise ModelValidationError("mapping and command selector modes must match")
        self._config = config
        self._selector = command_selector
        self._integrator = IncrementalGripperMapper(
            speed_m_s=gripper_speed_m_s,
            max_width_m=gripper_max_width_m,
        )
        self._reference: TeleopReference | None = None
        self._gripper_target_m: float | None = None
        self._fine_mode = False
        self._rebase_requested = False
        self._lock = threading.RLock()
        self._mapped = 0
        self._holds = 0
        self._rebases = 0
        self._missing_input = 0
        self._jump_rejections = 0
        self._ik_rejections = 0

    @property
    def reference(self) -> TeleopReference | None:
        with self._lock:
            return self._reference

    @property
    def fine_mode(self) -> bool:
        with self._lock:
            return self._fine_mode

    @property
    def metrics(self) -> PoseMappingMetrics:
        with self._lock:
            return PoseMappingMetrics(
                mapped=self._mapped,
                holds=self._holds,
                rebases=self._rebases,
                missing_input=self._missing_input,
                jump_rejections=self._jump_rejections,
                ik_rejections=self._ik_rejections,
            )

    def set_fine_mode(self, enabled: bool) -> None:
        """Request a zero-motion rebase when the fine-mode edge is consumed."""

        if not isinstance(enabled, bool):
            raise ModelValidationError("fine mode must be a boolean")
        with self._lock:
            if enabled != self._fine_mode:
                self._fine_mode = enabled
                self._rebase_requested = True

    def reset_reference(self) -> None:
        with self._lock:
            self._reference = None
            self._rebase_requested = False

    def _set_reference(self, source_pose: Transform4, robot_state: RobotState) -> None:
        self._reference = TeleopReference(
            source_pose=source_pose,
            robot_pose=validate_transform(robot_state.tcp_pose),
        )
        self._rebase_requested = False
        self._rebases += 1

    def _target(
        self,
        source_pose: Transform4,
        robot_state: RobotState,
        vr_input: VRInputState,
        *,
        dt_s: float,
        gripper_width_m: float | None,
    ) -> RobotAction:
        assert self._reference is not None
        translation_scale = (
            self._config.fine_translation_scale
            if self._fine_mode
            else self._config.translation_scale
        )
        rotation_scale = (
            self._config.fine_rotation_scale
            if self._fine_mode
            else self._config.rotation_scale
        )
        tcp_target = map_relative_pose(
            self._reference.source_pose,
            source_pose,
            self._reference.robot_pose,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            freeze_rotation=self._config.freeze_rotation,
            rotation_composition=RotationComposition(
                self._config.rotation_composition
            ),
        )
        action = RobotAction(
            **_metadata(vr_input),
            command_type=RobotCommandType.TCP_POSE,
            values=flatten_transform(tcp_target),
            duration_s=dt_s,
            gripper_width_m=gripper_width_m,
        )
        selected = self._selector.select(action, robot_state)
        if selected is None:
            self._ik_rejections += 1
            self._holds += 1
            return _hold(vr_input, gripper_width_m)
        self._mapped += 1
        return selected

    def _gripper_target(
        self,
        direction: GripperDirection,
        *,
        measured_width_m: float | None,
        dt_s: float,
    ) -> float | None:
        if measured_width_m is None:
            self._gripper_target_m = None
            return None
        if self._gripper_target_m is None or direction is GripperDirection.HOLD:
            self._gripper_target_m = measured_width_m
        self._gripper_target_m = self._integrator.target(
            direction,
            current_width_m=self._gripper_target_m,
            dt_s=dt_s,
        )
        return self._gripper_target_m


class ControllerPoseMapping(_RelativePoseMapping):
    """Map right-controller relative motion and joystick gripper input."""

    def __init__(
        self,
        config: TeleopConfig,
        command_selector: CommandModeSelector,
        *,
        gripper_speed_m_s: float = 0.1,
        gripper_max_width_m: float = 0.085,
        side: HandSide = HandSide.RIGHT,
    ) -> None:
        super().__init__(
            config,
            command_selector,
            gripper_speed_m_s=gripper_speed_m_s,
            gripper_max_width_m=gripper_max_width_m,
        )
        self._side = HandSide(side)
        self._gripper = ControllerGripperMapping(
            integrator=self._integrator,
            deadzone=config.controller_gripper_deadzone,
        )

    def _controller(self, vr_input: VRInputState) -> ControllerState | None:
        if vr_input.mode is not VRInputMode.CONTROLLERS:
            return None
        return next(
            (
                controller
                for controller in vr_input.controllers
                if controller.side is self._side
            ),
            None,
        )

    def map_input(
        self,
        vr_input: VRInputState,
        robot_state: RobotState,
        dt_s: float,
    ) -> RobotAction:
        """Map one typed controller update without hardware or socket access."""

        if not isinstance(vr_input, VRInputState) or not isinstance(robot_state, RobotState):
            raise ModelValidationError("controller mapping requires VRInputState and RobotState")
        dt = _positive_dt(dt_s)
        with self._lock:
            controller = self._controller(vr_input)
            if controller is None:
                self._missing_input += 1
                self._holds += 1
                return _hold(vr_input, robot_state.gripper_width_m)
            source_pose = vr_pose_to_transform(
                controller.position_m,
                controller.orientation_xyzw,
                self._config.vr_to_robot_axes,
            )
            gripper = self._gripper_target(
                self._gripper.direction(controller.joystick_xy[1]),
                measured_width_m=robot_state.gripper_width_m,
                dt_s=dt,
            )
            engaged = controller.grip_trigger > self._config.controller_grip_threshold
            if not engaged:
                self._set_reference(source_pose, robot_state)
                self._holds += 1
                return _hold(vr_input, gripper)
            if self._reference is None or self._rebase_requested:
                self._set_reference(source_pose, robot_state)
                self._holds += 1
                return _hold(vr_input, gripper)
            return self._target(
                source_pose,
                robot_state,
                vr_input,
                dt_s=dt,
                gripper_width_m=gripper,
            )


class HandPoseMapping(_RelativePoseMapping):
    """Map right-hand wrist/palm motion and thumb-index gripper distance."""

    def __init__(
        self,
        config: TeleopConfig,
        command_selector: CommandModeSelector,
        *,
        gripper_speed_m_s: float = 0.1,
        gripper_max_width_m: float = 0.085,
        side: HandSide = HandSide.RIGHT,
    ) -> None:
        super().__init__(
            config,
            command_selector,
            gripper_speed_m_s=gripper_speed_m_s,
            gripper_max_width_m=gripper_max_width_m,
        )
        self._side = HandSide(side)
        self._last_palm: tuple[float, float, float] | None = None
        self._gripper = HandGripperMapping(
            integrator=self._integrator,
            open_distance_m=config.hand_gripper_open_distance_m,
            close_distance_m=config.hand_gripper_close_distance_m,
        )

    def reset_reference(self) -> None:
        with self._lock:
            super().reset_reference()
            self._last_palm = None

    def _hand(self, vr_input: VRInputState) -> HandState | None:
        if vr_input.mode is not VRInputMode.HANDS:
            return None
        return next(
            (hand for hand in vr_input.hands if hand.side is self._side),
            None,
        )

    def _source_pose(self, hand: HandState) -> Transform4:
        position = hand.wrist_position_m or hand.joints_m[0]
        orientation = hand.wrist_orientation_xyzw or (0.0, 0.0, 0.0, 1.0)
        return vr_pose_to_transform(
            position,
            orientation,
            self._config.vr_to_robot_axes,
        )

    def map_input(
        self,
        vr_input: VRInputState,
        robot_state: RobotState,
        dt_s: float,
    ) -> RobotAction:
        """Map one typed OpenXR hand update without gesture side effects."""

        if not isinstance(vr_input, VRInputState) or not isinstance(robot_state, RobotState):
            raise ModelValidationError("hand mapping requires VRInputState and RobotState")
        dt = _positive_dt(dt_s)
        with self._lock:
            hand = self._hand(vr_input)
            if hand is None:
                self._missing_input += 1
                self._holds += 1
                return _hold(vr_input, robot_state.gripper_width_m)
            source_pose = self._source_pose(hand)
            palm = tuple(row[3] for row in source_pose[:3])
            if self._last_palm is not None:
                jump = math.sqrt(
                    sum(
                        (current - previous) ** 2
                        for current, previous in zip(palm, self._last_palm)
                    )
                )
                if jump > self._config.hand_palm_jump_threshold_m:
                    self._reference = None
                    self._last_palm = None
                    self._jump_rejections += 1
                    self._holds += 1
                    return _hold(vr_input, robot_state.gripper_width_m)
            self._last_palm = palm
            gripper = self._gripper_target(
                self._gripper.direction(_finger_distance(hand)),
                measured_width_m=robot_state.gripper_width_m,
                dt_s=dt,
            )
            if self._reference is None or self._rebase_requested:
                self._set_reference(source_pose, robot_state)
                self._holds += 1
                return _hold(vr_input, gripper)
            return self._target(
                source_pose,
                robot_state,
                vr_input,
                dt_s=dt,
                gripper_width_m=gripper,
            )


class ModeAwareTeleopMapping:
    """Select controller or hand pose mapping from the typed VR input mode."""

    def __init__(
        self,
        controller: ControllerPoseMapping,
        hand: HandPoseMapping,
    ) -> None:
        self._controller = controller
        self._hand = hand

    def map_input(
        self,
        vr_input: VRInputState,
        robot_state: RobotState,
        dt_s: float,
    ) -> RobotAction:
        if vr_input.mode is VRInputMode.CONTROLLERS:
            return self._controller.map_input(vr_input, robot_state, dt_s)
        return self._hand.map_input(vr_input, robot_state, dt_s)

    def set_fine_mode(self, enabled: bool) -> None:
        self._controller.set_fine_mode(enabled)
        self._hand.set_fine_mode(enabled)

    def reset_references(self) -> None:
        self._controller.reset_reference()
        self._hand.reset_reference()
