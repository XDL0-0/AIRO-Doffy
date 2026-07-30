"""Immutable cross-domain samples with boundary shape validation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from .errors import ModelValidationError

_EnumT = TypeVar("_EnumT", bound=Enum)
_MAX_SEQUENCE = (1 << 64) - 1


class ClockDomain(str, Enum):
    """Clock domain associated with a source timestamp."""

    MONOTONIC = "monotonic"
    UNIX = "unix"
    DEVICE = "device"
    UNSPECIFIED = "unspecified"


class PixelFormat(str, Enum):
    """Packed pixel layouts accepted by camera frame models."""

    RGB8 = "rgb8"
    BGR8 = "bgr8"
    GRAY8 = "gray8"
    DEPTH_U16 = "depth_u16"


class VideoCodec(str, Enum):
    """Encoded video payload formats."""

    H264 = "h264"
    JPEG = "jpeg"
    VP8 = "vp8"


class HandSide(str, Enum):
    """Left or right controller/hand identity."""

    LEFT = "left"
    RIGHT = "right"


class ControllerButton(str, Enum):
    """Buttons represented by the current Quest controller protocol."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    JOYSTICK = "joystick"


class VRInputMode(str, Enum):
    """Mutually exclusive VR input representations."""

    CONTROLLERS = "controllers"
    HANDS = "hands"


class RobotCommandType(str, Enum):
    """Robot actions shared by mappings, filters, executors, and backends."""

    JOINT_POSITION = "joint_position"
    TCP_POSE = "tcp_pose"
    JOINT_VELOCITY = "joint_velocity"
    TCP_TWIST = "tcp_twist"
    HOLD = "hold"
    STOP = "stop"


def _enum(value: object, enum_type: type[_EnumT], name: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(str(item.value) for item in enum_type)
        raise ModelValidationError(f"{name} must be one of: {supported}") from exc


def _non_negative_int(value: object, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelValidationError(f"{name} must be an integer")
    if value < 0:
        raise ModelValidationError(f"{name} must be non-negative")
    if maximum is not None and value > maximum:
        raise ModelValidationError(f"{name} must be <= {maximum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ModelValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ModelValidationError(f"{name} must be finite")
    return result


def _vector(values: Iterable[object], size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(
            _finite_float(value, f"{name}[{index}]") for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise ModelValidationError(f"{name} must be an iterable of {size} numbers") from exc
    if len(result) != size:
        raise ModelValidationError(f"{name} must have shape ({size},), got ({len(result)},)")
    return result


def _variable_vector(
    values: Iterable[object],
    allowed_sizes: set[int],
    name: str,
) -> tuple[float, ...]:
    try:
        result = tuple(
            _finite_float(value, f"{name}[{index}]") for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise ModelValidationError(f"{name} must be an iterable") from exc
    if len(result) not in allowed_sizes:
        expected = ", ".join(str(size) for size in sorted(allowed_sizes))
        raise ModelValidationError(
            f"{name} length must be one of {{{expected}}}, got {len(result)}"
        )
    return result


def _matrix(
    values: Iterable[Iterable[object]],
    rows: int,
    columns: int,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    try:
        result = tuple(
            _vector(row, columns, f"{name}[{index}]") for index, row in enumerate(values)
        )
    except TypeError as exc:
        raise ModelValidationError(
            f"{name} must be an iterable with shape ({rows}, {columns})"
        ) from exc
    if len(result) != rows:
        raise ModelValidationError(
            f"{name} must have shape ({rows}, {columns}), got ({len(result)}, {columns})"
        )
    return result


def _shape(values: Sequence[object], name: str) -> tuple[int, ...]:
    result = tuple(
        _non_negative_int(value, f"{name}[{index}]") for index, value in enumerate(values)
    )
    if not result or any(dimension == 0 for dimension in result):
        raise ModelValidationError(f"{name} dimensions must be positive")
    return result


def _image_payload(
    data: bytes | bytearray | memoryview,
    shape: Sequence[object],
    pixel_format: object,
) -> tuple[bytes, tuple[int, ...], PixelFormat]:
    try:
        payload = bytes(data)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("data must support the bytes protocol") from exc
    checked_shape = _shape(shape, "shape")
    checked_format = _enum(pixel_format, PixelFormat, "pixel_format")

    if checked_format in {PixelFormat.RGB8, PixelFormat.BGR8}:
        if len(checked_shape) != 3 or checked_shape[2] != 3:
            raise ModelValidationError(
                f"{checked_format.value} frames require shape (height, width, 3)"
            )
        expected_bytes = checked_shape[0] * checked_shape[1] * 3
    elif checked_format is PixelFormat.GRAY8:
        if len(checked_shape) == 3 and checked_shape[2] == 1:
            expected_bytes = checked_shape[0] * checked_shape[1]
        elif len(checked_shape) == 2:
            expected_bytes = checked_shape[0] * checked_shape[1]
        else:
            raise ModelValidationError(
                "gray8 frames require shape (height, width) or (height, width, 1)"
            )
    else:
        if len(checked_shape) != 2:
            raise ModelValidationError("depth_u16 frames require shape (height, width)")
        expected_bytes = checked_shape[0] * checked_shape[1] * 2

    if len(payload) != expected_bytes:
        raise ModelValidationError(
            f"data has {len(payload)} bytes; {checked_format.value} {checked_shape} "
            f"requires {expected_bytes}"
        )
    return payload, checked_shape, checked_format


def _stream_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError("stream_id must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class SequencedSample:
    """Common metadata present on every high-frequency sample."""

    sequence: int
    source_timestamp_ns: int
    receive_timestamp_ns: int | None = None
    clock_domain: ClockDomain = ClockDomain.MONOTONIC

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _non_negative_int(self.sequence, "sequence", _MAX_SEQUENCE),
        )
        object.__setattr__(
            self,
            "source_timestamp_ns",
            _non_negative_int(self.source_timestamp_ns, "source_timestamp_ns"),
        )
        object.__setattr__(
            self,
            "receive_timestamp_ns",
            None
            if self.receive_timestamp_ns is None
            else _non_negative_int(self.receive_timestamp_ns, "receive_timestamp_ns"),
        )
        object.__setattr__(
            self,
            "clock_domain",
            _enum(self.clock_domain, ClockDomain, "clock_domain"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ImageFrame(SequencedSample):
    stream_id: str
    data: bytes
    shape: tuple[int, ...]
    pixel_format: PixelFormat

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        payload, shape, pixel_format = _image_payload(self.data, self.shape, self.pixel_format)
        object.__setattr__(self, "stream_id", _stream_id(self.stream_id))
        object.__setattr__(self, "data", payload)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "pixel_format", pixel_format)


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraFrame(_ImageFrame):
    """Immutable packed frame produced directly by a camera source."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessedFrame(_ImageFrame):
    """Immutable packed frame after crop, resize, color, or depth processing."""

    processing_timestamp_ns: int

    def __post_init__(self) -> None:
        _ImageFrame.__post_init__(self)
        object.__setattr__(
            self,
            "processing_timestamp_ns",
            _non_negative_int(self.processing_timestamp_ns, "processing_timestamp_ns"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EncodedFrame(SequencedSample):
    """Immutable compressed video access unit."""

    stream_id: str
    data: bytes
    codec: VideoCodec
    width: int
    height: int
    encoded_timestamp_ns: int
    keyframe: bool = False

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        try:
            payload = bytes(self.data)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("data must support the bytes protocol") from exc
        if not payload:
            raise ModelValidationError("encoded frame data must not be empty")
        width = _non_negative_int(self.width, "width")
        height = _non_negative_int(self.height, "height")
        if width == 0 or height == 0:
            raise ModelValidationError("encoded frame width and height must be positive")
        object.__setattr__(self, "stream_id", _stream_id(self.stream_id))
        object.__setattr__(self, "data", payload)
        object.__setattr__(self, "codec", _enum(self.codec, VideoCodec, "codec"))
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(
            self,
            "encoded_timestamp_ns",
            _non_negative_int(self.encoded_timestamp_ns, "encoded_timestamp_ns"),
        )
        object.__setattr__(self, "keyframe", bool(self.keyframe))


@dataclass(frozen=True, slots=True, kw_only=True)
class TactileSample(SequencedSample):
    """Supported 4-taxel sample in fixed ``(4, 3)`` xyz order."""

    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        object.__setattr__(self, "values", _matrix(self.values, 4, 3, "values"))


@dataclass(frozen=True, slots=True, kw_only=True)
class WrenchSample(SequencedSample):
    """Force/torque vector ordered as ``Fx,Fy,Fz,Tx,Ty,Tz``."""

    values: tuple[float, ...]
    frame_id: str = "tool"

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ModelValidationError("frame_id must be a non-empty string")
        object.__setattr__(self, "values", _vector(self.values, 6, "values"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ControllerState(SequencedSample):
    """One Quest controller state in the current 14-field per-hand format."""

    side: HandSide
    position_m: tuple[float, ...]
    orientation_xyzw: tuple[float, ...]
    joystick_xy: tuple[float, ...]
    index_trigger: float
    grip_trigger: float
    buttons: frozenset[ControllerButton]
    clock_domain: ClockDomain = ClockDomain.DEVICE

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        try:
            buttons = frozenset(
                _enum(button, ControllerButton, "buttons") for button in self.buttons
            )
        except TypeError as exc:
            raise ModelValidationError("buttons must be iterable") from exc
        index_trigger = _finite_float(self.index_trigger, "index_trigger")
        grip_trigger = _finite_float(self.grip_trigger, "grip_trigger")
        if not 0.0 <= index_trigger <= 1.0 or not 0.0 <= grip_trigger <= 1.0:
            raise ModelValidationError("controller triggers must be within [0, 1]")
        orientation = _vector(self.orientation_xyzw, 4, "orientation_xyzw")
        if math.sqrt(sum(value * value for value in orientation)) == 0.0:
            raise ModelValidationError("orientation_xyzw must not be the zero quaternion")
        object.__setattr__(self, "side", _enum(self.side, HandSide, "side"))
        object.__setattr__(self, "position_m", _vector(self.position_m, 3, "position_m"))
        object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(self, "joystick_xy", _vector(self.joystick_xy, 2, "joystick_xy"))
        object.__setattr__(self, "index_trigger", index_trigger)
        object.__setattr__(self, "grip_trigger", grip_trigger)
        object.__setattr__(self, "buttons", buttons)


@dataclass(frozen=True, slots=True, kw_only=True)
class HandState(SequencedSample):
    """One OpenXR hand containing exactly 26 xyz joints."""

    side: HandSide
    joints_m: tuple[tuple[float, ...], ...]
    wrist_position_m: tuple[float, ...] | None = None
    wrist_orientation_xyzw: tuple[float, ...] | None = None
    clock_domain: ClockDomain = ClockDomain.DEVICE

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        if (self.wrist_position_m is None) != (self.wrist_orientation_xyzw is None):
            raise ModelValidationError(
                "wrist position and orientation must either both be set or both be omitted"
            )
        wrist_position = (
            None
            if self.wrist_position_m is None
            else _vector(self.wrist_position_m, 3, "wrist_position_m")
        )
        wrist_orientation = (
            None
            if self.wrist_orientation_xyzw is None
            else _vector(self.wrist_orientation_xyzw, 4, "wrist_orientation_xyzw")
        )
        if wrist_orientation is not None and sum(value * value for value in wrist_orientation) == 0:
            raise ModelValidationError("wrist_orientation_xyzw must not be the zero quaternion")
        object.__setattr__(self, "side", _enum(self.side, HandSide, "side"))
        object.__setattr__(self, "joints_m", _matrix(self.joints_m, 26, 3, "joints_m"))
        object.__setattr__(self, "wrist_position_m", wrist_position)
        object.__setattr__(self, "wrist_orientation_xyzw", wrist_orientation)


@dataclass(frozen=True, slots=True, kw_only=True)
class VRInputState(SequencedSample):
    """Complete controller or hand-tracking sample for one VR update."""

    mode: VRInputMode
    controllers: tuple[ControllerState, ...] = ()
    hands: tuple[HandState, ...] = ()
    clock_domain: ClockDomain = ClockDomain.DEVICE

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        mode = _enum(self.mode, VRInputMode, "mode")
        controllers = tuple(self.controllers)
        hands = tuple(self.hands)
        if any(not isinstance(item, ControllerState) for item in controllers):
            raise ModelValidationError("controllers must contain ControllerState values")
        if any(not isinstance(item, HandState) for item in hands):
            raise ModelValidationError("hands must contain HandState values")
        if mode is VRInputMode.CONTROLLERS:
            if len(controllers) != 2 or hands:
                raise ModelValidationError(
                    "controller mode requires exactly two controllers and no hands"
                )
        elif not 1 <= len(hands) <= 2 or controllers:
            raise ModelValidationError("hand mode requires one or two hands and no controllers")
        sides = [item.side for item in controllers or hands]
        if len(set(sides)) != len(sides):
            raise ModelValidationError("VR input cannot contain duplicate hand sides")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "controllers", controllers)
        object.__setattr__(self, "hands", hands)


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotState(SequencedSample):
    """Backend-neutral robot state for UR (6 DOF) or RealMan RM75 (7 DOF)."""

    joints_rad: tuple[float, ...]
    tcp_pose: tuple[tuple[float, ...], ...]
    gripper_width_m: float | None = None
    wrench: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        width = (
            None
            if self.gripper_width_m is None
            else _finite_float(self.gripper_width_m, "gripper_width_m")
        )
        if width is not None and width < 0:
            raise ModelValidationError("gripper_width_m must be non-negative")
        object.__setattr__(
            self,
            "joints_rad",
            _variable_vector(self.joints_rad, {6, 7}, "joints_rad"),
        )
        object.__setattr__(self, "tcp_pose", _matrix(self.tcp_pose, 4, 4, "tcp_pose"))
        object.__setattr__(self, "gripper_width_m", width)
        object.__setattr__(
            self,
            "wrench",
            None if self.wrench is None else _vector(self.wrench, 6, "wrench"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotAction(SequencedSample):
    """Validated backend-neutral robot command plus optional gripper target."""

    command_type: RobotCommandType
    values: tuple[float, ...]
    duration_s: float | None = None
    gripper_width_m: float | None = None

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        command_type = _enum(self.command_type, RobotCommandType, "command_type")
        if command_type in {
            RobotCommandType.JOINT_POSITION,
            RobotCommandType.JOINT_VELOCITY,
        }:
            values = _variable_vector(self.values, {6, 7}, "values")
        elif command_type is RobotCommandType.TCP_POSE:
            values = _vector(self.values, 16, "values")
        elif command_type is RobotCommandType.TCP_TWIST:
            values = _vector(self.values, 6, "values")
        else:
            values = tuple(self.values)
            if values:
                raise ModelValidationError(f"{command_type.value} commands cannot contain values")

        duration = (
            None if self.duration_s is None else _finite_float(self.duration_s, "duration_s")
        )
        if duration is not None and duration <= 0:
            raise ModelValidationError("duration_s must be positive")
        width = (
            None
            if self.gripper_width_m is None
            else _finite_float(self.gripper_width_m, "gripper_width_m")
        )
        if width is not None and width < 0:
            raise ModelValidationError("gripper_width_m must be non-negative")
        object.__setattr__(self, "command_type", command_type)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "gripper_width_m", width)


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation(SequencedSample):
    """Immutable runtime snapshot assembled from independently optional sources."""

    robot: RobotState | None = None
    frames: tuple[ProcessedFrame, ...] = ()
    vr_input: VRInputState | None = None
    tactile: TactileSample | None = None
    wrench: WrenchSample | None = None

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        frames = tuple(self.frames)
        if self.robot is not None and not isinstance(self.robot, RobotState):
            raise ModelValidationError("robot must be a RobotState or None")
        if any(not isinstance(frame, ProcessedFrame) for frame in frames):
            raise ModelValidationError("frames must contain ProcessedFrame values")
        if self.vr_input is not None and not isinstance(self.vr_input, VRInputState):
            raise ModelValidationError("vr_input must be a VRInputState or None")
        if self.tactile is not None and not isinstance(self.tactile, TactileSample):
            raise ModelValidationError("tactile must be a TactileSample or None")
        if self.wrench is not None and not isinstance(self.wrench, WrenchSample):
            raise ModelValidationError("wrench must be a WrenchSample or None")
        stream_ids = [frame.stream_id for frame in frames]
        if len(set(stream_ids)) != len(stream_ids):
            raise ModelValidationError("observation frames must have unique stream_id values")
        object.__setattr__(self, "frames", frames)
