"""Versioned binary envelopes for high-frequency VR and robot state."""

from __future__ import annotations

import struct
from dataclasses import replace
from enum import IntEnum

from ...core.errors import ModelValidationError
from ...core.types import ClockDomain, RobotState, VRInputState
from ...devices.vr.binary_v2 import (
    HEADER as VR_V2_HEADER,
    decode_vr_binary_v2,
    encode_vr_binary_v2,
)

STATE_MAGIC = 0xAD20
STATE_VERSION = 1
STATE_HEADER = struct.Struct("<HBBIQHH")
STATE_HEADER_SIZE = STATE_HEADER.size
STATE_SEQUENCE_MODULUS = 1 << 32

_CLOCK_MASK = 0x0003
_KNOWN_FLAGS = _CLOCK_MASK
_ROBOT_PREFIX = struct.Struct("<BBBB")
_FLOAT32 = struct.Struct("<f")
_ROBOT_GRIPPER = 1 << 0
_ROBOT_WRENCH = 1 << 1
_ROBOT_KNOWN_OPTIONS = _ROBOT_GRIPPER | _ROBOT_WRENCH

StateSample = VRInputState | RobotState


class StateMessageType(IntEnum):
    """State payload kinds carried by the common realtime envelope."""

    VR_INPUT = 1
    ROBOT_STATE = 2


_CLOCK_TO_FLAG = {
    ClockDomain.MONOTONIC: 0,
    ClockDomain.UNIX: 1,
    ClockDomain.DEVICE: 2,
    ClockDomain.UNSPECIFIED: 3,
}
_FLAG_TO_CLOCK = {value: key for key, value in _CLOCK_TO_FLAG.items()}


def state_message_type(sample: StateSample) -> StateMessageType:
    """Return the wire message type for a supported state sample."""

    if isinstance(sample, VRInputState):
        return StateMessageType.VR_INPUT
    if isinstance(sample, RobotState):
        return StateMessageType.ROBOT_STATE
    raise ModelValidationError("state sample must be VRInputState or RobotState")


def _encode_robot(state: RobotState) -> bytes:
    options = 0
    values = [*state.joints_rad, *(value for row in state.tcp_pose for value in row)]
    if state.gripper_width_m is not None:
        options |= _ROBOT_GRIPPER
        values.append(state.gripper_width_m)
    if state.wrench is not None:
        options |= _ROBOT_WRENCH
        values.extend(state.wrench)
    floats = struct.pack(f"<{len(values)}f", *values)
    return _ROBOT_PREFIX.pack(len(state.joints_rad), options, 0, 0) + floats


def _decode_robot(
    payload: bytes,
    *,
    sequence: int,
    source_timestamp_ns: int,
    receive_timestamp_ns: int,
    clock_domain: ClockDomain,
) -> RobotState:
    if len(payload) < _ROBOT_PREFIX.size:
        raise ModelValidationError("robot state payload is truncated")
    dof, options, reserved_0, reserved_1 = _ROBOT_PREFIX.unpack_from(payload)
    if dof not in {6, 7}:
        raise ModelValidationError("robot state DOF must be 6 or 7")
    if options & ~_ROBOT_KNOWN_OPTIONS or reserved_0 or reserved_1:
        raise ModelValidationError("robot state payload contains unsupported flags")
    float_count = dof + 16
    if options & _ROBOT_GRIPPER:
        float_count += 1
    if options & _ROBOT_WRENCH:
        float_count += 6
    expected_size = _ROBOT_PREFIX.size + float_count * _FLOAT32.size
    if len(payload) != expected_size:
        raise ModelValidationError("robot state payload length does not match its flags")
    values = struct.unpack_from(f"<{float_count}f", payload, _ROBOT_PREFIX.size)
    offset = dof
    tcp_values = values[offset : offset + 16]
    offset += 16
    gripper = None
    if options & _ROBOT_GRIPPER:
        gripper = values[offset]
        offset += 1
    wrench = None
    if options & _ROBOT_WRENCH:
        wrench = tuple(values[offset : offset + 6])
    return RobotState(
        sequence=sequence,
        source_timestamp_ns=source_timestamp_ns,
        receive_timestamp_ns=receive_timestamp_ns,
        clock_domain=clock_domain,
        joints_rad=tuple(values[:dof]),
        tcp_pose=tuple(
            tuple(tcp_values[index : index + 4]) for index in range(0, 16, 4)
        ),
        gripper_width_m=gripper,
        wrench=wrench,
    )


def encode_state_packet(sample: StateSample) -> bytes:
    """Serialize one typed sample into the transport-independent state envelope."""

    message_type = state_message_type(sample)
    if sample.sequence >= STATE_SEQUENCE_MODULUS:
        raise ModelValidationError("state packet sequence must fit uint32")
    if sample.source_timestamp_ns >= 1 << 64:
        raise ModelValidationError("state packet timestamp must fit uint64")
    try:
        flags = _CLOCK_TO_FLAG[sample.clock_domain]
    except KeyError as exc:
        raise ModelValidationError("unsupported state clock domain") from exc
    if message_type is StateMessageType.VR_INPUT:
        payload = encode_vr_binary_v2(sample)
    else:
        payload = _encode_robot(sample)
    if len(payload) > 0xFFFF:
        raise ModelValidationError("state payload must fit uint16 length")
    return STATE_HEADER.pack(
        STATE_MAGIC,
        STATE_VERSION,
        message_type,
        sample.sequence,
        sample.source_timestamp_ns,
        len(payload),
        flags,
    ) + payload


def _decode_vr(
    payload: bytes,
    *,
    sequence: int,
    source_timestamp_ns: int,
    receive_timestamp_ns: int,
    clock_domain: ClockDomain,
) -> VRInputState:
    if len(payload) < VR_V2_HEADER.size:
        raise ModelValidationError("VR state payload is truncated")
    _magic, _version, _mode, _count, _flags, inner_sequence, inner_timestamp, _length = (
        VR_V2_HEADER.unpack_from(payload)
    )
    if inner_sequence != sequence or inner_timestamp != source_timestamp_ns:
        raise ModelValidationError("VR payload metadata does not match state envelope")
    state = decode_vr_binary_v2(
        payload,
        receive_timestamp_ns=receive_timestamp_ns,
    )
    if state is None:
        raise ModelValidationError("VR state payload is malformed")
    controllers = tuple(
        replace(controller, clock_domain=clock_domain) for controller in state.controllers
    )
    hands = tuple(replace(hand, clock_domain=clock_domain) for hand in state.hands)
    return replace(
        state,
        clock_domain=clock_domain,
        controllers=controllers,
        hands=hands,
    )


def decode_state_packet(
    data: bytes | bytearray | memoryview,
    *,
    receive_timestamp_ns: int,
) -> StateSample:
    """Strictly decode one state packet or raise ``ModelValidationError``."""

    if (
        isinstance(receive_timestamp_ns, bool)
        or not isinstance(receive_timestamp_ns, int)
        or receive_timestamp_ns < 0
    ):
        raise ModelValidationError("receive_timestamp_ns must be a non-negative integer")
    try:
        raw = bytes(data)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("state packet must support the bytes protocol") from exc
    if len(raw) < STATE_HEADER.size:
        raise ModelValidationError("state packet is shorter than its header")
    try:
        magic, version, type_code, sequence, timestamp_ns, payload_length, flags = (
            STATE_HEADER.unpack_from(raw)
        )
    except struct.error as exc:
        raise ModelValidationError("state packet header is malformed") from exc
    if magic != STATE_MAGIC:
        raise ModelValidationError("state packet magic does not match")
    if version != STATE_VERSION:
        raise ModelValidationError(f"unsupported state packet version: {version}")
    if flags & ~_KNOWN_FLAGS:
        raise ModelValidationError("state packet contains unsupported flags")
    payload = raw[STATE_HEADER.size :]
    if len(payload) != payload_length:
        raise ModelValidationError("state packet payload length does not match header")
    try:
        message_type = StateMessageType(type_code)
        clock_domain = _FLAG_TO_CLOCK[flags & _CLOCK_MASK]
    except (ValueError, KeyError) as exc:
        raise ModelValidationError("state packet type or clock domain is unsupported") from exc
    if message_type is StateMessageType.VR_INPUT:
        return _decode_vr(
            payload,
            sequence=sequence,
            source_timestamp_ns=timestamp_ns,
            receive_timestamp_ns=receive_timestamp_ns,
            clock_domain=clock_domain,
        )
    return _decode_robot(
        payload,
        sequence=sequence,
        source_timestamp_ns=timestamp_ns,
        receive_timestamp_ns=receive_timestamp_ns,
        clock_domain=clock_domain,
    )
