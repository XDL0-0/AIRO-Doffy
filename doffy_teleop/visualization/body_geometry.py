"""BODY skeleton topology and coordinate/angle helpers for display."""

from __future__ import annotations

import numpy as np

from doffy_teleop.protocol.body import BodyFrame, is_finger


def _edges() -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []

    def chain(*names: str) -> None:
        result.extend(zip(names, names[1:]))

    chain("Hips", "SpineLower", "SpineMiddle", "SpineUpper", "Chest", "Neck", "Head")
    for side in ("Left", "Right"):
        chain("Chest", side + "Shoulder", side + "ArmUpper", side + "ArmLower",
              side + "HandWristTwist", side + "HandWrist", side + "HandPalm")
        chain(side + "Shoulder", side + "Scapula", side + "ArmUpper")
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            parts = ("Metacarpal", "Proximal", "Distal", "Tip") if finger == "Thumb" else (
                "Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")
            chain(side + "HandWrist", *(side + "Hand" + finger + part for part in parts))
        chain("Hips", side + "UpperLeg", side + "LowerLeg", side + "FootAnkleTwist",
              side + "FootAnkle", side + "FootSubtalar", side + "FootTransverse", side + "FootBall")
    return tuple(result)

BODY_EDGES = _edges()
KEY_JOINTS = ("Head", "Hips", "LeftShoulder", "RightShoulder", "LeftArmLower",
              "RightArmLower", "LeftHandWrist", "RightHandWrist", "LeftLowerLeg",
              "RightLowerLeg", "LeftFootAnkle", "RightFootAnkle")
COLORS = {"Left": "#38bdf8", "Right": "#fb923c", "Center": "#dbeafe"}


def joint_angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float | None:
    """Inner angle in degrees: a straight elbow or knee is 180 degrees."""
    first, second = a - vertex, b - vertex
    divisor = np.linalg.norm(first) * np.linalg.norm(second)
    if divisor < 1e-10:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / divisor, -1, 1))))


def pose_angles(frame: BodyFrame) -> dict[str, float | None]:
    points = frame.positions()
    angles = {}
    for side in ("Left", "Right"):
        for label, names in (("elbow", (side + "Shoulder", side + "ArmLower", side + "HandWrist")),
                             ("knee", (side + "UpperLeg", side + "LowerLeg", side + "FootAnkle"))):
            angles[side + " " + label] = joint_angle(*(points[name] for name in names)) if all(
                name in points for name in names) else None
        shoulder, elbow = points.get(side + "Shoulder"), points.get(side + "ArmLower")
        angles[side + " arm raise"] = joint_angle(shoulder + (0, -1, 0), shoulder, elbow) if (
            shoulder is not None and elbow is not None) else None
    return angles


def display_coordinates(points: np.ndarray) -> np.ndarray:
    """Reorder Unity XYZ to X/Z/Y for a Y-up Matplotlib scene, without sign flips."""
    return np.asarray(points)[..., [0, 2, 1]]


def _color(name: str) -> str:
    return COLORS["Left"] if name.startswith("Left") else COLORS["Right"] if name.startswith("Right") else COLORS["Center"]
