"""Parse VR data from Unity — controller old/new formats; hand = OpenXR 26 joints only.

Controller formats:
  Old (DualControllerSender, Time.time):
    <timestamp_ms>,<left_14>,<right_14>                      (29 fields)
  New (DualControllerSender, Stopwatch):
    C,<frameId>,<timestamp_ns>,<left_14>,<right_14>          (prefix "C,")

Hand (OpenXR XR_EXT_hand_tracking, 26 joints):
  Text: H,<side>,<frameId>,<timestamp_ns>,<wrist_pos3>,<wrist_quat4>,<joint_xyz * 26>
  Binary: HB,<base64>  header 8B [0x48, side, count_lo, count_hi, frameId_u32_LE] + 26×12B joints
"""

from __future__ import annotations

import base64
import struct
from typing import Tuple

import utils

CONTROLLER_FIELDS_PER_HAND = 14
HAND_JOINT_COUNT = 26
HAND_BYTES_PER_BONE = 12  # 3 × float32
HAND_BINARY_HEADER = 8
HAND_TEXT_FIELDS = 2 + 2 + 7 + HAND_JOINT_COUNT * 3  # H, side, frameId, ts, wrist, bones


# ── Packet type detection ────────────────────────────────────────────────

def detect_packet_type(input_data: str) -> str:
    """Return 'controller', 'hand_text', 'hand_binary', or 'unknown'."""
    s = input_data.strip()
    if s.startswith("HB,"):
        return "hand_binary"
    if s.startswith("H,"):
        return "hand_text"
    if s.startswith("C,"):
        return "controller"
    if s and s[0].isdigit():
        return "controller"
    return "unknown"


# ── Controller parsing ───────────────────────────────────────────────────

def parse_data(input_data: str | None) -> list[dict] | None:
    """Parse controller CSV — auto-detects old and new format.

    Old: <timestamp_ms>,<left_14>,<right_14>              (29 fields)
    New: C,<frameId>,<timestamp_ns>,<left_14>,<right_14>  (31 fields)

    Returns list of 2 dicts [left, right] on success, None on failure.
    Each dict has keys: ControllerType, Timestamp, FrameId, Position,
    Rotation, Joystick, IndexTrigger, GripTrigger, Button_AX, Button_BY,
    Joystick_Press.
    """
    if not input_data:
        return None

    raw = input_data.strip()
    parts = raw.split(",")

    frame_id = 0
    timestamp = 0
    data_start = 0

    if parts[0] == "C":
        # New format: C,<frameId>,<timestamp_ns>,<left_14>,<right_14>
        expected = 3 + 2 * CONTROLLER_FIELDS_PER_HAND  # 31
        if len(parts) != expected:
            utils.logger.warning(
                f"Controller packet (new): expected {expected} fields, got {len(parts)}"
            )
            return None
        try:
            frame_id = int(parts[1])
            timestamp = int(parts[2])
        except ValueError:
            utils.logger.warning(f"Non-numeric header in new controller data: {raw[:80]}")
            return None
        data_start = 3
    else:
        # Old format: <timestamp_ms>,<left_14>,<right_14>
        expected = 1 + 2 * CONTROLLER_FIELDS_PER_HAND  # 29
        if len(parts) != expected:
            utils.logger.warning(
                f"Controller packet (old): expected {expected} fields, got {len(parts)}"
            )
            return None
        try:
            timestamp = int(parts[0])
        except ValueError:
            utils.logger.warning(f"Non-numeric timestamp in controller data: {raw[:80]}")
            return None
        data_start = 1

    try:
        nums = [float(p) for p in parts[data_start:]]
    except ValueError:
        utils.logger.warning(f"Non-numeric value in controller data: {raw[:80]}")
        return None

    controllers: list[dict] = []
    for i, name in enumerate(("LTouch", "RTouch")):
        off = i * CONTROLLER_FIELDS_PER_HAND
        v = nums[off : off + CONTROLLER_FIELDS_PER_HAND]
        controllers.append(
            {
                "ControllerType": name,
                "FrameId": frame_id,
                "Timestamp": timestamp,
                "Position": (v[0], v[1], v[2]),
                "Rotation": (v[3], v[4], v[5], v[6]),
                "Joystick": (v[7], v[8]),
                "IndexTrigger": v[9],
                "GripTrigger": v[10],
                "Button_AX": int(v[11]),
                "Button_BY": int(v[12]),
                "Joystick_Press": int(v[13]),
            }
        )
    return controllers


# ── Hand-tracking parsing ────────────────────────────────────────────────

def parse_hand_data(input_data: str | None) -> dict | None:
    """Parse hand-tracking packet (text or binary). OpenXR 26-joint format only.

    Returns dict with keys: side, frame_id, timestamp, wrist_pose, bones (26×(x,y,z)).
    Returns None on failure.
    """
    if not input_data:
        return None

    ptype = detect_packet_type(input_data)
    if ptype == "hand_text":
        return _parse_hand_text(input_data)
    elif ptype == "hand_binary":
        return _parse_hand_binary(input_data)
    return None


def _parse_hand_text(input_data: str) -> dict | None:
    """Parse text hand data: H,<side>,<frameId>,<timestamp_ns>,<wrist_7>,<26 joint xyz>."""
    try:
        parts = input_data.strip().rstrip("\n").split(",")
        if len(parts) != HAND_TEXT_FIELDS or parts[0] != "H":
            if parts[0] == "H" and len(parts) != HAND_TEXT_FIELDS:
                utils.logger.warning(
                    f"Hand text: expected {HAND_TEXT_FIELDS} fields, got {len(parts)}"
                )
            return None

        side = parts[1]
        frame_id = int(parts[2])
        timestamp = int(parts[3])

        wrist_vals = [float(parts[i]) for i in range(4, 11)]
        wrist_pose = {
            "position": (wrist_vals[0], wrist_vals[1], wrist_vals[2]),
            "rotation": (wrist_vals[3], wrist_vals[4], wrist_vals[5], wrist_vals[6]),
        }

        bones: list[Tuple[float, float, float]] = []
        idx = 11
        for _ in range(HAND_JOINT_COUNT):
            bones.append((float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])))
            idx += 3

        return {
            "side": side,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "wrist_pose": wrist_pose,
            "bones": bones,
        }

    except Exception as e:
        utils.logger.warning(f"Hand text parse error: {e}")
        return None


def _parse_hand_binary(input_data: str) -> dict | None:
    """Parse binary hand data: 8B header + 26 joints × 12B (little-endian float xyz)."""
    try:
        payload_b64 = input_data.strip()[3:]  # skip 'HB,'
        raw = base64.b64decode(payload_b64)

        if len(raw) < HAND_BINARY_HEADER or raw[0] != 0x48:
            return None

        side = chr(raw[1])
        bone_count = raw[2] | (raw[3] << 8)
        if bone_count != HAND_JOINT_COUNT:
            utils.logger.warning(
                f"Hand binary: expected {HAND_JOINT_COUNT} joints, got {bone_count}"
            )
            return None

        expected_len = HAND_BINARY_HEADER + HAND_JOINT_COUNT * HAND_BYTES_PER_BONE
        if len(raw) < expected_len:
            utils.logger.warning(
                f"Hand binary too short: {len(raw)} bytes, need {expected_len}"
            )
            return None

        frame_id = struct.unpack_from("<I", raw, 4)[0]
        offset = HAND_BINARY_HEADER
        bones: list[Tuple[float, float, float]] = []
        for _ in range(HAND_JOINT_COUNT):
            x = struct.unpack_from("<f", raw, offset)[0]
            y = struct.unpack_from("<f", raw, offset + 4)[0]
            z = struct.unpack_from("<f", raw, offset + 8)[0]
            bones.append((x, y, z))
            offset += HAND_BYTES_PER_BONE

        import time as _time
        timestamp = int(_time.time() * 1000)

        return {
            "side": side,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "wrist_pose": None,
            "bones": bones,
        }

    except Exception as e:
        utils.logger.warning(f"Hand binary parse error: {e}")
        return None
