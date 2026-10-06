"""Compatibility exports for the modular BODY telemetry viewer.

Protocol, reception, rendering, and lifecycle implementations live in their
respective doffy_teleop subpackages. Existing imports remain available here.
"""

from doffy_teleop.protocol.body import (
    JOINT_IDS, JOINT_NAMES, BodyFrame, BodyJoint, is_finger, parse_body_packet,
)
from doffy_teleop.media.body import BodyReceiver
from doffy_teleop.visualization.body import BodyDashboard
from doffy_teleop.visualization.body_demo import demo_frame
from doffy_teleop.visualization.body_geometry import (
    BODY_EDGES, COLORS, KEY_JOINTS, display_coordinates, joint_angle, pose_angles,
)
from doffy_teleop.runtime.body_viewer import run_visualizer

__all__ = [
    "BODY_EDGES", "COLORS", "JOINT_IDS", "JOINT_NAMES", "KEY_JOINTS",
    "BodyDashboard", "BodyFrame", "BodyJoint", "BodyReceiver", "demo_frame",
    "display_coordinates", "is_finger", "joint_angle", "parse_body_packet",
    "pose_angles", "run_visualizer",
]
