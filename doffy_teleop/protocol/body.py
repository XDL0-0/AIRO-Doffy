"""Strict BODY v1 telemetry decoding without viewer or robot dependencies.

BODY positions use Unity world metres (Y up) and XYZW rotations. NumPy is
needed only when callers request the array-based ``BodyFrame.positions`` view.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _joint_names() -> tuple[str, ...]:
    names = ["Root", "Hips", "SpineLower", "SpineMiddle", "SpineUpper", "Chest",
             "Neck", "Head", "LeftShoulder", "LeftScapula", "LeftArmUpper",
             "LeftArmLower", "LeftHandWristTwist", "RightShoulder", "RightScapula",
             "RightArmUpper", "RightArmLower", "RightHandWristTwist"]
    for side in ("Left", "Right"):
        names.extend((side + "HandPalm", side + "HandWrist"))
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            parts = ("Metacarpal", "Proximal", "Distal", "Tip") if finger == "Thumb" else (
                "Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")
            names.extend(side + "Hand" + finger + part for part in parts)
    for side in ("Left", "Right"):
        names.extend(side + part for part in ("UpperLeg", "LowerLeg", "FootAnkleTwist",
                                             "FootAnkle", "FootSubtalar", "FootTransverse", "FootBall"))
    return tuple(names)


JOINT_NAMES = _joint_names()
JOINT_IDS = {name: index for index, name in enumerate(JOINT_NAMES)}


@dataclass(frozen=True)
class BodyJoint:
    id: int
    name: str
    position: tuple[float, float, float] | None
    rotation: tuple[float, float, float, float] | None
    position_tracked: bool | None = None
    orientation_tracked: bool | None = None


@dataclass(frozen=True)
class BodyFrame:
    frame_id: int
    timestamp_ns: int
    received_ns: int
    joint_set: str
    confidence: float
    tracking_valid: bool
    joints: tuple[BodyJoint, ...]
    wrm: dict

    def positions(self, *, show_hands: bool = True) -> dict[str, np.ndarray]:
        if not self.tracking_valid:
            return {}
        import numpy as np

        return {joint.name: np.asarray(joint.position) for joint in self.joints
                if joint.position is not None and (show_hands or not is_finger(joint.name))}


def is_finger(name: str) -> bool:
    return "Hand" in name and not name.endswith(("Wrist", "WristTwist", "Palm"))


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected finite number")
    return float(value)


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("expected boolean")
    return value


def _vector(value: object, count: int) -> tuple | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("invalid vector length")
    return tuple(_number(element) for element in value)


def parse_body_packet(raw: bytes | str, *, received_ns: int | None = None) -> BodyFrame | None:
    """Strictly decode BODY v1. Bad, incompatible, and nonfinite packets are ignored."""
    try:
        if len(raw) > 65507:
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("type") != "BODY":
            return None
        if type(payload.get("version")) is not int or payload["version"] != 1:
            return None
        if payload.get("coordinate_space") != "unity_world":
            return None
        joint_set = payload["joint_set"]
        if joint_set not in ("upper_body", "full_body"):
            return None
        count = 70 if joint_set == "upper_body" else 84
        records = payload["joints"]
        if not isinstance(records, list) or len(records) > count:
            return None
        confidence = _number(payload["confidence"])
        if not 0 <= confidence <= 1:
            return None
        joints = []
        seen = set()
        for record in records:
            joint_id = _integer(record["id"])
            if joint_id >= count or joint_id in seen or record["name"] != JOINT_NAMES[joint_id]:
                return None
            seen.add(joint_id)
            position = _vector(record["position"], 3)
            rotation = _vector(record["rotation"], 4)
            position_valid = _boolean(record["position_valid"])
            orientation_valid = _boolean(record["orientation_valid"])
            position_tracked = (
                _boolean(record["position_tracked"]) if "position_tracked" in record else None)
            orientation_tracked = (
                _boolean(record["orientation_tracked"]) if "orientation_tracked" in record else None)
            if position_valid and position is None:
                return None
            if orientation_valid and (rotation is None or math.hypot(*rotation) < 1e-8):
                return None
            joints.append(BodyJoint(joint_id, record["name"],
                                    position if position_valid else None,
                                    rotation if orientation_valid else None,
                                    position_tracked, orientation_tracked))
        wrm = payload.get("wrm", {})
        if not isinstance(wrm, dict):
            return None
        for key in ("elbow_alpha", "confidence"):
            if key in wrm and not 0 <= _number(wrm[key]) <= 1:
                return None
        for key in ("enabled", "calibrated"):
            if key in wrm:
                _boolean(wrm[key])
        return BodyFrame(_integer(payload["frame_id"]), _integer(payload["timestamp_ns"]),
                         time.monotonic_ns() if received_ns is None else received_ns,
                         joint_set, confidence, _boolean(payload["tracking_valid"]),
                         tuple(joints), wrm)
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        return None


__all__ = ["JOINT_NAMES", "JOINT_IDS", "BodyJoint", "BodyFrame", "is_finger", "parse_body_packet"]
