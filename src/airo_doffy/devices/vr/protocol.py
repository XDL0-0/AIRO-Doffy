"""Pure decoders for current Unity controller and OpenXR hand packets."""

from __future__ import annotations

import base64
import struct
import time

from ...core.types import (
    ClockDomain,
    ControllerButton,
    ControllerState,
    HandSide,
    HandState,
    VRInputMode,
    VRInputState,
)

CONTROLLER_FIELDS_PER_HAND = 14
HAND_JOINT_COUNT = 26
HAND_BYTES_PER_BONE = 12
HAND_BINARY_HEADER = 8
HAND_TEXT_FIELDS = 2 + 2 + 7 + HAND_JOINT_COUNT * 3


def detect_packet_type(input_data: str) -> str:
    """Return the legacy packet family without parsing its payload."""

    if not isinstance(input_data, str):
        return "unknown"
    value = input_data.strip()
    if value.startswith("HB,"):
        return "hand_binary"
    if value.startswith("H,"):
        return "hand_text"
    if value.startswith("C,") or (value and value[0].isdigit()):
        return "controller"
    return "unknown"


def parse_data(input_data: str | None) -> list[dict] | None:
    """Decode legacy/new controller CSV into the historical dictionary shape."""

    if not input_data:
        return None
    parts = input_data.strip().split(",")
    frame_id = 0
    if parts[0] == "C":
        if len(parts) != 3 + 2 * CONTROLLER_FIELDS_PER_HAND:
            return None
        try:
            frame_id = int(parts[1])
            timestamp = int(parts[2])
        except ValueError:
            return None
        data_start = 3
    else:
        if len(parts) != 1 + 2 * CONTROLLER_FIELDS_PER_HAND:
            return None
        try:
            timestamp = int(parts[0])
        except ValueError:
            return None
        data_start = 1
    try:
        numbers = [float(value) for value in parts[data_start:]]
    except ValueError:
        return None
    controllers = []
    for index, name in enumerate(("LTouch", "RTouch")):
        offset = index * CONTROLLER_FIELDS_PER_HAND
        values = numbers[offset : offset + CONTROLLER_FIELDS_PER_HAND]
        controllers.append(
            {
                "ControllerType": name,
                "FrameId": frame_id,
                "Timestamp": timestamp,
                "Position": tuple(values[0:3]),
                "Rotation": tuple(values[3:7]),
                "Joystick": tuple(values[7:9]),
                "IndexTrigger": values[9],
                "GripTrigger": values[10],
                "Button_AX": int(values[11]),
                "Button_BY": int(values[12]),
                "Joystick_Press": int(values[13]),
            }
        )
    return controllers


def _parse_hand_text(input_data: str) -> dict | None:
    try:
        parts = input_data.strip().rstrip("\n").split(",")
        if len(parts) != HAND_TEXT_FIELDS or parts[0] != "H":
            return None
        wrist = [float(parts[index]) for index in range(4, 11)]
        bones = []
        offset = 11
        for _index in range(HAND_JOINT_COUNT):
            bones.append(
                (
                    float(parts[offset]),
                    float(parts[offset + 1]),
                    float(parts[offset + 2]),
                )
            )
            offset += 3
        return {
            "side": parts[1],
            "frame_id": int(parts[2]),
            "timestamp": int(parts[3]),
            "wrist_pose": {
                "position": tuple(wrist[0:3]),
                "rotation": tuple(wrist[3:7]),
            },
            "bones": bones,
        }
    except (IndexError, TypeError, ValueError):
        return None


def _parse_hand_binary(
    input_data: str,
    *,
    wall_time_ms: int | None = None,
) -> dict | None:
    try:
        raw = base64.b64decode(input_data.strip()[3:])
        if len(raw) < HAND_BINARY_HEADER or raw[0] != 0x48:
            return None
        side = chr(raw[1])
        bone_count = raw[2] | raw[3] << 8
        if bone_count != HAND_JOINT_COUNT:
            return None
        expected = HAND_BINARY_HEADER + HAND_JOINT_COUNT * HAND_BYTES_PER_BONE
        if len(raw) < expected:
            return None
        frame_id = struct.unpack_from("<I", raw, 4)[0]
        bones = tuple(
            struct.unpack_from("<fff", raw, HAND_BINARY_HEADER + index * 12)
            for index in range(HAND_JOINT_COUNT)
        )
    except (TypeError, ValueError, struct.error):
        return None
    return {
        "side": side,
        "frame_id": frame_id,
        "timestamp": (
            int(time.time() * 1000)
            if wall_time_ms is None
            else int(wall_time_ms)
        ),
        "wrist_pose": None,
        "bones": list(bones),
    }


def parse_hand_data(
    input_data: str | None,
    *,
    wall_time_ms: int | None = None,
) -> dict | None:
    """Decode current hand text or HB packet into the historical dictionary."""

    if not input_data:
        return None
    packet_type = detect_packet_type(input_data)
    if packet_type == "hand_text":
        return _parse_hand_text(input_data)
    if packet_type == "hand_binary":
        return _parse_hand_binary(input_data, wall_time_ms=wall_time_ms)
    return None


def _side(value: str) -> HandSide:
    normalized = value.strip().lower()
    if normalized in {"l", "left", "ltouch"}:
        return HandSide.LEFT
    if normalized in {"r", "right", "rtouch"}:
        return HandSide.RIGHT
    raise ValueError(f"unsupported hand side: {value!r}")


def _buttons(values: dict) -> frozenset[ControllerButton]:
    buttons = set()
    if values["Button_AX"]:
        buttons.add(ControllerButton.PRIMARY)
    if values["Button_BY"]:
        buttons.add(ControllerButton.SECONDARY)
    if values["Joystick_Press"]:
        buttons.add(ControllerButton.JOYSTICK)
    return frozenset(buttons)


def decode_vr_input(
    input_data: str,
    *,
    receive_timestamp_ns: int,
) -> VRInputState | None:
    """Decode a current legacy packet directly into the typed v2 input model."""

    packet_type = detect_packet_type(input_data)
    if packet_type == "controller":
        values = parse_data(input_data)
        if values is None:
            return None
        is_new = input_data.strip().startswith("C,")
        sequence = (
            values[0]["FrameId"]
            if is_new
            else receive_timestamp_ns
        )
        source_timestamp_ns = (
            values[0]["Timestamp"]
            if is_new
            else values[0]["Timestamp"] * 1_000_000
        )
        try:
            controllers = tuple(
                ControllerState(
                    sequence=sequence,
                    source_timestamp_ns=source_timestamp_ns,
                    receive_timestamp_ns=receive_timestamp_ns,
                    clock_domain=ClockDomain.DEVICE,
                    side=_side(item["ControllerType"]),
                    position_m=item["Position"],
                    orientation_xyzw=item["Rotation"],
                    joystick_xy=item["Joystick"],
                    index_trigger=item["IndexTrigger"],
                    grip_trigger=item["GripTrigger"],
                    buttons=_buttons(item),
                )
                for item in values
            )
            return VRInputState(
                sequence=sequence,
                source_timestamp_ns=source_timestamp_ns,
                receive_timestamp_ns=receive_timestamp_ns,
                clock_domain=ClockDomain.DEVICE,
                mode=VRInputMode.CONTROLLERS,
                controllers=controllers,
            )
        except (TypeError, ValueError):
            return None

    hand = parse_hand_data(
        input_data,
        wall_time_ms=receive_timestamp_ns // 1_000_000,
    )
    if hand is None:
        return None
    wrist = hand["wrist_pose"]
    source_timestamp_ns = (
        hand["timestamp"]
        if packet_type == "hand_text"
        else receive_timestamp_ns
    )
    try:
        typed_hand = HandState(
            sequence=hand["frame_id"],
            source_timestamp_ns=source_timestamp_ns,
            receive_timestamp_ns=receive_timestamp_ns,
            clock_domain=ClockDomain.DEVICE,
            side=_side(hand["side"]),
            joints_m=tuple(hand["bones"]),
            wrist_position_m=None if wrist is None else wrist["position"],
            wrist_orientation_xyzw=None if wrist is None else wrist["rotation"],
        )
        return VRInputState(
            sequence=hand["frame_id"],
            source_timestamp_ns=source_timestamp_ns,
            receive_timestamp_ns=receive_timestamp_ns,
            clock_domain=ClockDomain.DEVICE,
            mode=VRInputMode.HANDS,
            hands=(typed_hand,),
        )
    except (TypeError, ValueError):
        return None
