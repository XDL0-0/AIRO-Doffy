"""Versioned little-endian binary codec for typed VR input state."""

from __future__ import annotations

import struct

from ...core.errors import ModelValidationError
from ...core.types import (
    ClockDomain,
    ControllerButton,
    ControllerState,
    HandSide,
    HandState,
    VRInputMode,
    VRInputState,
)

MAGIC = b"AVR2"
VERSION = 2
MODE_CONTROLLERS = 1
MODE_HANDS = 2
HEADER = struct.Struct("<4sBBBBIQI")
CONTROLLER = struct.Struct("<BBH11f")
HAND_PREFIX = struct.Struct("<BBH")
WRIST = struct.Struct("<7f")
JOINT = struct.Struct("<3f")
JOINT_COUNT = 26

_PRIMARY = 1 << 0
_SECONDARY = 1 << 1
_JOYSTICK = 1 << 2


def _side_code(side: HandSide) -> int:
    return 0 if side is HandSide.LEFT else 1


def _side(code: int) -> HandSide:
    if code == 0:
        return HandSide.LEFT
    if code == 1:
        return HandSide.RIGHT
    raise ModelValidationError(f"unknown VR side code: {code}")


def _button_bits(buttons: frozenset[ControllerButton]) -> int:
    value = 0
    if ControllerButton.PRIMARY in buttons:
        value |= _PRIMARY
    if ControllerButton.SECONDARY in buttons:
        value |= _SECONDARY
    if ControllerButton.JOYSTICK in buttons:
        value |= _JOYSTICK
    return value


def _buttons(bits: int) -> frozenset[ControllerButton]:
    if bits & ~(_PRIMARY | _SECONDARY | _JOYSTICK):
        raise ModelValidationError("controller button bits contain unknown flags")
    values = set()
    if bits & _PRIMARY:
        values.add(ControllerButton.PRIMARY)
    if bits & _SECONDARY:
        values.add(ControllerButton.SECONDARY)
    if bits & _JOYSTICK:
        values.add(ControllerButton.JOYSTICK)
    return frozenset(values)


def encode_vr_binary_v2(state: VRInputState) -> bytes:
    """Encode one typed state using protocol version 2."""

    if not isinstance(state, VRInputState):
        raise ModelValidationError("state must be a VRInputState")
    if state.sequence >= 2**32:
        raise ModelValidationError("binary v2 sequence must fit uint32")
    if state.source_timestamp_ns >= 2**64:
        raise ModelValidationError("binary v2 timestamp must fit uint64")
    payload = bytearray()
    if state.mode is VRInputMode.CONTROLLERS:
        mode = MODE_CONTROLLERS
        entities = state.controllers
        for controller in entities:
            payload.extend(
                CONTROLLER.pack(
                    _side_code(controller.side),
                    _button_bits(controller.buttons),
                    0,
                    *controller.position_m,
                    *controller.orientation_xyzw,
                    *controller.joystick_xy,
                    controller.index_trigger,
                    controller.grip_trigger,
                )
            )
    else:
        mode = MODE_HANDS
        entities = state.hands
        for hand in entities:
            wrist_present = int(hand.wrist_position_m is not None)
            payload.extend(
                HAND_PREFIX.pack(
                    _side_code(hand.side),
                    wrist_present,
                    JOINT_COUNT,
                )
            )
            if wrist_present:
                assert hand.wrist_position_m is not None
                assert hand.wrist_orientation_xyzw is not None
                payload.extend(
                    WRIST.pack(
                        *hand.wrist_position_m,
                        *hand.wrist_orientation_xyzw,
                    )
                )
            for joint in hand.joints_m:
                payload.extend(JOINT.pack(*joint))
    return HEADER.pack(
        MAGIC,
        VERSION,
        mode,
        len(entities),
        0,
        state.sequence,
        state.source_timestamp_ns,
        len(payload),
    ) + bytes(payload)


def _decode_controllers(
    payload: bytes,
    count: int,
    sequence: int,
    source_timestamp_ns: int,
    receive_timestamp_ns: int,
) -> tuple[ControllerState, ...]:
    if count != 2 or len(payload) != count * CONTROLLER.size:
        raise ModelValidationError("controller payload must contain exactly two entities")
    controllers = []
    for index in range(count):
        values = CONTROLLER.unpack_from(payload, index * CONTROLLER.size)
        side_code, button_bits, reserved, *numbers = values
        if reserved != 0:
            raise ModelValidationError("controller reserved field must be zero")
        controllers.append(
            ControllerState(
                sequence=sequence,
                source_timestamp_ns=source_timestamp_ns,
                receive_timestamp_ns=receive_timestamp_ns,
                clock_domain=ClockDomain.DEVICE,
                side=_side(side_code),
                position_m=tuple(numbers[0:3]),
                orientation_xyzw=tuple(numbers[3:7]),
                joystick_xy=tuple(numbers[7:9]),
                index_trigger=numbers[9],
                grip_trigger=numbers[10],
                buttons=_buttons(button_bits),
            )
        )
    return tuple(controllers)


def _decode_hands(
    payload: bytes,
    count: int,
    sequence: int,
    source_timestamp_ns: int,
    receive_timestamp_ns: int,
) -> tuple[HandState, ...]:
    if count not in {1, 2}:
        raise ModelValidationError("hand payload must contain one or two entities")
    hands = []
    offset = 0
    for _index in range(count):
        if offset + HAND_PREFIX.size > len(payload):
            raise ModelValidationError("hand payload is truncated")
        side_code, wrist_present, joint_count = HAND_PREFIX.unpack_from(payload, offset)
        offset += HAND_PREFIX.size
        if wrist_present not in {0, 1} or joint_count != JOINT_COUNT:
            raise ModelValidationError("invalid hand wrist flag or joint count")
        wrist_position = None
        wrist_orientation = None
        if wrist_present:
            if offset + WRIST.size > len(payload):
                raise ModelValidationError("hand wrist payload is truncated")
            wrist = WRIST.unpack_from(payload, offset)
            offset += WRIST.size
            wrist_position = tuple(wrist[0:3])
            wrist_orientation = tuple(wrist[3:7])
        joints = []
        for _joint_index in range(JOINT_COUNT):
            if offset + JOINT.size > len(payload):
                raise ModelValidationError("hand joint payload is truncated")
            joints.append(JOINT.unpack_from(payload, offset))
            offset += JOINT.size
        hands.append(
            HandState(
                sequence=sequence,
                source_timestamp_ns=source_timestamp_ns,
                receive_timestamp_ns=receive_timestamp_ns,
                clock_domain=ClockDomain.DEVICE,
                side=_side(side_code),
                joints_m=tuple(joints),
                wrist_position_m=wrist_position,
                wrist_orientation_xyzw=wrist_orientation,
            )
        )
    if offset != len(payload):
        raise ModelValidationError("hand payload contains trailing bytes")
    return tuple(hands)


def decode_vr_binary_v2(
    data: bytes | bytearray | memoryview,
    *,
    receive_timestamp_ns: int,
) -> VRInputState | None:
    """Decode binary v2, returning ``None`` for malformed or unknown packets."""

    try:
        raw = bytes(data)
        if len(raw) < HEADER.size:
            return None
        magic, version, mode, count, flags, sequence, timestamp_ns, length = (
            HEADER.unpack_from(raw)
        )
        if magic != MAGIC or version != VERSION or flags != 0:
            return None
        payload = raw[HEADER.size:]
        if length != len(payload):
            return None
        if mode == MODE_CONTROLLERS:
            controllers = _decode_controllers(
                payload,
                count,
                sequence,
                timestamp_ns,
                receive_timestamp_ns,
            )
            return VRInputState(
                sequence=sequence,
                source_timestamp_ns=timestamp_ns,
                receive_timestamp_ns=receive_timestamp_ns,
                clock_domain=ClockDomain.DEVICE,
                mode=VRInputMode.CONTROLLERS,
                controllers=controllers,
            )
        if mode == MODE_HANDS:
            hands = _decode_hands(
                payload,
                count,
                sequence,
                timestamp_ns,
                receive_timestamp_ns,
            )
            return VRInputState(
                sequence=sequence,
                source_timestamp_ns=timestamp_ns,
                receive_timestamp_ns=receive_timestamp_ns,
                clock_domain=ClockDomain.DEVICE,
                mode=VRInputMode.HANDS,
                hands=hands,
            )
        return None
    except (ModelValidationError, TypeError, ValueError, struct.error):
        return None
