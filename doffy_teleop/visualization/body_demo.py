"""Synthetic BODY poses for explicitly labelled dashboard demos."""

from __future__ import annotations

import math
import time

import numpy as np

from doffy_teleop.protocol.body import BodyFrame, BodyJoint, JOINT_NAMES


def demo_frame(t: float = 0.0) -> BodyFrame:
    """Clearly marked synthetic standing pose for checking UI without hardware."""
    positions = {"Root": (0, 0, 0), "Hips": (0, 0.96, 0), "SpineLower": (0, 1.1, 0),
                 "SpineMiddle": (0, 1.23, 0), "SpineUpper": (0, 1.36, 0), "Chest": (0, 1.47, 0),
                 "Neck": (0, 1.62, 0), "Head": (0, 1.75, 0)}
    for side, sign, phase in (("Left", -1, 0), ("Right", 1, 1.8)):
        shoulder = np.array([sign * 0.18, 1.48, 0])
        elevation = 0.8 + 0.55 * math.sin(t + phase)
        elbow = shoulder + np.array([sign * 0.30 * math.sin(elevation), -0.30 * math.cos(elevation), 0.02])
        wrist = elbow + np.array([sign * 0.07, -0.18, 0.20 + 0.04 * math.sin(t + phase)])
        positions.update({side + "Shoulder": shoulder, side + "Scapula": shoulder + [0, -0.015, -0.04],
                          side + "ArmUpper": shoulder + [sign * 0.025, -0.02, 0], side + "ArmLower": elbow,
                          side + "HandWristTwist": wrist + [0, 0.02, -0.025], side + "HandWrist": wrist,
                          side + "HandPalm": wrist + [0, 0, 0.07]})
        for finger_index, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Little")):
            parts = ("Metacarpal", "Proximal", "Distal", "Tip") if finger == "Thumb" else (
                "Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")
            for index, part in enumerate(parts):
                positions[side + "Hand" + finger + part] = wrist + [sign * (finger_index - 2) * 0.02, -0.008 * index, 0.045 + index * 0.024]
        for name, point in (("UpperLeg", (sign * 0.09, 0.94, 0)), ("LowerLeg", (sign * 0.1, 0.52, 0.04)),
                            ("FootAnkleTwist", (sign * 0.1, 0.13, 0)), ("FootAnkle", (sign * 0.1, 0.10, 0)),
                            ("FootSubtalar", (sign * 0.1, 0.055, 0)), ("FootTransverse", (sign * 0.1, 0.055, 0.08)),
                            ("FootBall", (sign * 0.1, 0.055, 0.16))):
            positions[side + name] = point
    return BodyFrame(int(t * 30), time.time_ns(), time.monotonic_ns(), "full_body", 1.0, True,
                     tuple(BodyJoint(index, name, tuple(positions[name]), (0, 0, 0, 1)) for index, name in enumerate(JOINT_NAMES)),
                     {"elbow_alpha": 0.5 + 0.5 * math.sin(t), "confidence": 1.0, "enabled": True, "calibrated": True})
