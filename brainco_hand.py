"""BrainCo Revo2 hand driver for a hand connected through RealMan RM_ARM+.

The driver owns hardware discovery, RM_ARM+ setup, controller-triggered hand
presets, and conversion from the OpenXR 26-joint hand skeleton to the Revo2
six-motor position command.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np


# RealMan RM_ARM+ / BrainCo Revo2 motor order.
THUMB_FLEX = 0
INDEX = 1
MIDDLE = 2
RING = 3
PINKY = 4
THUMB_ROTATE = 5
HAND_DOF = 6

# Preset fractions and timing mirror
# tests/revo2_keyboard_teleop_single_send.py exactly.
GRAB_FOUR_FINGER_CLOSE = 0.90
GRAB_THUMB_ROTATE = 0.80
GRAB_THUMB_CLOSE = 0.88
GRAB_THUMB_PRE_CLOSE = GRAB_THUMB_CLOSE - 0.60

# OpenXR XR_EXT_hand_tracking joint indices.
THUMB_JOINTS = (2, 3, 4, 5)
INDEX_JOINTS = (6, 7, 8, 9, 10)
MIDDLE_JOINTS = (11, 12, 13, 14, 15)
RING_JOINTS = (16, 17, 18, 19, 20)
PINKY_JOINTS = (21, 22, 23, 24, 25)
OPENXR_JOINT_COUNT = 26


class BrainCoHandUnavailableError(RuntimeError):
    """Raised when RM_ARM+ does not expose a usable BrainCo six-DoF hand."""


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _result_pair(result: Any, operation: str) -> tuple[int, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise BrainCoHandUnavailableError(
            f"{operation} returned an unexpected value: {result!r}"
        )
    code, value = result
    if int(code) != 0:
        raise BrainCoHandUnavailableError(
            f"{operation} failed with RealMan error code {code}."
        )
    return int(code), value


def _joint_bend(points: np.ndarray, indices: Sequence[int], full_bend: float) -> float:
    """Return normalized accumulated bend for one OpenXR digit chain."""
    chain = points[np.asarray(indices, dtype=int)]
    segment = np.diff(chain, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError("OpenXR hand contains a zero-length finger segment.")

    unit = segment / lengths[:, None]
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    bend = float(np.sum(np.arccos(np.clip(dots, -1.0, 1.0))))
    return float(np.clip(bend / full_bend, 0.0, 1.0))


def _validated_openxr_points(
    bones: Sequence[Sequence[float]],
) -> np.ndarray:
    points = np.asarray(bones, dtype=float)
    if points.shape != (OPENXR_JOINT_COUNT, 3):
        raise ValueError(
            f"Expected an OpenXR ({OPENXR_JOINT_COUNT}, 3) skeleton, "
            f"received {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("OpenXR hand contains non-finite joint positions.")
    return points


def _thumb_flex_from_points(points: np.ndarray) -> float:
    """Estimate thumb curvature using all thumb joints, including its tip."""
    chain = points[np.asarray(THUMB_JOINTS, dtype=int)]
    segments = np.diff(chain, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError("OpenXR hand contains a zero-length thumb segment.")
    unit = segments / lengths[:, None]

    # Local curvature at the proximal and distal joints. Distal bending has a
    # stronger effect on the Revo2's first motor, so weight it more heavily.
    proximal_bend = float(
        np.arccos(np.clip(np.dot(unit[0], unit[1]), -1.0, 1.0))
    )
    distal_bend = float(
        np.arccos(np.clip(np.dot(unit[1], unit[2]), -1.0, 1.0))
    )
    weighted_joint_bend = 0.35 * proximal_bend + 0.65 * distal_bend
    joint_curve = weighted_joint_bend / np.deg2rad(55.0)

    # The base-to-tip chord captures accumulated curvature that can be small
    # or noisy at either individual OpenXR joint. Comparing it with the first
    # thumb segment uses the tip together with every intermediate joint.
    base_to_tip = chain[-1] - chain[0]
    chord_length = float(np.linalg.norm(base_to_tip))
    if chord_length < 1e-6:
        raise ValueError("OpenXR thumb base and tip positions overlap.")
    tip_sweep = float(
        np.arccos(
            np.clip(
                np.dot(unit[0], base_to_tip / chord_length),
                -1.0,
                1.0,
            )
        )
    )
    tip_curve = tip_sweep / np.deg2rad(35.0)

    # Square-root response expands modest human thumb bends into useful Revo2
    # motion without changing the zero point of a straight thumb.
    return float(np.sqrt(np.clip(max(joint_curve, tip_curve), 0.0, 1.0)))


def openxr_thumb_flexion(bones: Sequence[Sequence[float]]) -> float:
    """Return normalized first-motor flexion from the four OpenXR thumb joints."""
    return _thumb_flex_from_points(_validated_openxr_points(bones))


def openxr_thumb_opposition_progress(
    bones: Sequence[Sequence[float]],
) -> float:
    """Return scale-invariant thumb progress from the index to pinky side."""
    points = _validated_openxr_points(bones)
    index_mcp = points[INDEX_JOINTS[0]]
    across_palm = points[PINKY_JOINTS[0]] - index_mcp
    palm_width_sq = float(np.dot(across_palm, across_palm))
    if palm_width_sq < 1e-8:
        raise ValueError("OpenXR hand has an invalid palm width.")
    return float(
        np.dot(points[THUMB_JOINTS[-1]] - index_mcp, across_palm)
        / palm_width_sq
    )


def openxr_to_brainco_joints(
    bones: Sequence[Sequence[float]],
    *,
    thumb_rotate_open_progress: float = -0.65,
    thumb_rotate_progress_range: float = 1.5,
) -> np.ndarray:
    """Map an OpenXR right-hand skeleton to normalized Revo2 motor positions.

    The returned order is ``[thumb flex, index, middle, ring, pinky,
    thumb rotation]`` and every value is in ``[0, 1]``. Finger values use
    accumulated anatomical bend. The first thumb-flex value also recognizes
    fingertip contact, while thumb opposition uses thumb-tip progress from the
    index side toward the pinky side of the palm. All measures are invariant to
    wrist pose and hand size.
    """
    points = _validated_openxr_points(bones)
    if thumb_rotate_progress_range <= 0.0:
        raise ValueError("Thumb-rotation progress range must be positive.")

    # Unlike the four fingers, thumb flexion combines local joint bend with the
    # accumulated base-to-tip curve so modest OpenXR motion closes motor 1.
    thumb_bend = _thumb_flex_from_points(points)
    index = _joint_bend(points, INDEX_JOINTS, 1.5 * np.pi)
    middle = _joint_bend(points, MIDDLE_JOINTS, 1.5 * np.pi)
    ring = _joint_bend(points, RING_JOINTS, 1.5 * np.pi)
    pinky = _joint_bend(points, PINKY_JOINTS, 1.5 * np.pi)

    index_mcp = points[INDEX_JOINTS[0]]
    pinky_mcp = points[PINKY_JOINTS[0]]
    across_palm = pinky_mcp - index_mcp
    palm_width_sq = float(np.dot(across_palm, across_palm))
    if palm_width_sq < 1e-8:
        raise ValueError("OpenXR hand has an invalid palm width.")
    palm_width = float(np.sqrt(palm_width_sq))

    # Closing/pinching may be dominated by thumb opposition rather than local
    # thumb bend. Use proximity to the three grasping fingertips to make the
    # first Revo2 joint close decisively in those poses. An open hand normally
    # has a nearest-tip distance above 1.25 palm widths; contact is normally
    # below roughly 0.35 palm widths.
    grasping_tips = points[[INDEX_JOINTS[-1], MIDDLE_JOINTS[-1], RING_JOINTS[-1]]]
    nearest_tip_distance = float(
        np.min(np.linalg.norm(grasping_tips - points[THUMB_JOINTS[-1]], axis=1))
    )
    thumb_contact_close = float(
        np.clip(
            (1.25 - nearest_tip_distance / palm_width) / 0.9,
            0.0,
            1.0,
        )
    )
    thumb_flex = max(thumb_bend, thumb_contact_close)

    thumb_progress = openxr_thumb_opposition_progress(points)
    # The driver calibrates thumb_rotate_open_progress from the initial hand
    # frame. Therefore motion begins immediately instead of traversing a fixed
    # clipped region near the open pose.
    thumb_rotate = float(
        np.clip(
            (thumb_progress - thumb_rotate_open_progress)
            / thumb_rotate_progress_range,
            0.0,
            1.0,
        )
    )

    return np.asarray(
        [thumb_flex, index, middle, ring, pinky, thumb_rotate],
        dtype=float,
    )


class BrainCoHandMotionFilter:
    """Per-joint dead zone followed by a time-aware low-pass filter."""

    def __init__(
        self,
        *,
        cutoff_hz: float,
        dead_zone: float,
        initial: Sequence[float],
    ) -> None:
        self.cutoff_hz = float(cutoff_hz)
        self.dead_zone = float(dead_zone)
        if self.cutoff_hz < 0.0:
            raise ValueError("BrainCo hand filter cutoff cannot be negative.")
        if not 0.0 <= self.dead_zone < 1.0:
            raise ValueError("BrainCo hand dead zone must be in [0, 1).")

        initial_values = self._validated(initial)
        self._anchor = initial_values.copy()
        self.value = initial_values.copy()

    @staticmethod
    def _validated(joints: Sequence[float]) -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.shape != (HAND_DOF,) or not np.all(np.isfinite(values)):
            raise ValueError("BrainCo hand target must contain six finite values.")
        return np.clip(values, 0.0, 1.0)

    def update(self, joints: Sequence[float], dt: float) -> np.ndarray:
        values = self._validated(joints)
        changed = np.abs(values - self._anchor) > self.dead_zone
        self._anchor[changed] = values[changed]

        if self.cutoff_hz == 0.0:
            self.value = self._anchor.copy()
        else:
            dt = max(float(dt), 0.0)
            alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)
            self.value += alpha * (self._anchor - self.value)
        return self.value.copy()

    def reset(self, joints: Sequence[float]) -> None:
        """Seed both filter state and dead-zone anchor from a preset pose."""
        values = self._validated(joints)
        self._anchor = values.copy()
        self.value = values.copy()


class BrainCoHandDriver:
    """Drive a BrainCo Revo2 through an existing RealMan SDK arm object."""

    def __init__(
        self,
        arm: Any,
        *,
        baudrate: int = 460800,
        read_retries: int = 10,
        retry_delay: float = 0.25,
        mode_settle_delay: float = 2.0,
        max_send_hz: float = 50.0,
        filter_cutoff_hz: float = 8.0,
        dead_zone: float = 0.015,
        thumb_rotate_progress_range: float = 1.2,
    ) -> None:
        self.arm = arm
        self.baudrate = int(baudrate)
        self.read_retries = int(read_retries)
        self.retry_delay = float(retry_delay)
        self.mode_settle_delay = float(mode_settle_delay)
        self.min_send_interval = 1.0 / float(max_send_hz)
        self.thumb_rotate_progress_range = float(thumb_rotate_progress_range)
        self._thumb_rotate_open_progress: float | None = None
        self.last_send_time = 0.0

        if self.read_retries < 1:
            raise ValueError("read_retries must be at least 1.")
        if self.retry_delay < 0.0 or self.mode_settle_delay < 0.0:
            raise ValueError("BrainCo hand delays cannot be negative.")
        if max_send_hz <= 0.0:
            raise ValueError("max_send_hz must be positive.")
        if self.thumb_rotate_progress_range <= 0.0:
            raise ValueError("thumb_rotate_progress_range must be positive.")

        self._configure_rm_plus_mode()
        _, base_info = _result_pair(
            self.arm.rm_get_rm_plus_base_info(),
            "rm_get_rm_plus_base_info",
        )
        if not isinstance(base_info, dict):
            raise BrainCoHandUnavailableError(
                f"RM_ARM+ base information is invalid: {base_info!r}"
            )
        if int(base_info.get("type", -1)) != 2:
            raise BrainCoHandUnavailableError(
                "RM_ARM+ end-effector is not a dexterous hand "
                f"(type={base_info.get('type')!r})."
            )
        if int(base_info.get("dof", -1)) != HAND_DOF:
            raise BrainCoHandUnavailableError(
                "BrainCo mapping requires a six-DoF hand "
                f"(dof={base_info.get('dof')!r})."
            )

        self.lower, self.upper = self._extract_limits(base_info)
        measured = self._measured_position()
        if measured is None:
            raise BrainCoHandUnavailableError(
                "Could not read the initial BrainCo hand position."
            )
        self.target = measured
        span = self.upper - self.lower
        measured_normalized = np.divide(
            measured - self.lower,
            span,
            out=np.zeros(HAND_DOF, dtype=float),
            where=span != 0,
        )
        self.motion_filter = BrainCoHandMotionFilter(
            cutoff_hz=filter_cutoff_hz,
            dead_zone=dead_zone,
            initial=measured_normalized,
        )
        self.filtered_joints = measured_normalized
        self._last_filter_time = time.monotonic()
        self._motion_stages: list[tuple[np.ndarray, float]] = []
        self._next_motion_stage_time = 0.0
        self.active_motion: str | None = None

    def _configure_rm_plus_mode(self) -> None:
        _, current_mode = _result_pair(
            self.arm.rm_get_rm_plus_mode(),
            "rm_get_rm_plus_mode",
        )
        if int(current_mode) == self.baudrate:
            return
        result = int(self.arm.rm_set_rm_plus_mode(self.baudrate))
        if result != 0:
            raise BrainCoHandUnavailableError(
                "rm_set_rm_plus_mode failed with RealMan error code "
                f"{result}."
            )
        if self.mode_settle_delay:
            time.sleep(self.mode_settle_delay)

    @staticmethod
    def _extract_limits(base_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        low = base_info.get("pos_low")
        high = base_info.get("pos_up")
        if not isinstance(low, list) or not isinstance(high, list):
            low, high = [0] * HAND_DOF, [1000] * HAND_DOF
        if len(low) < HAND_DOF or len(high) < HAND_DOF:
            low, high = [0] * HAND_DOF, [1000] * HAND_DOF

        lower = np.asarray(low[:HAND_DOF], dtype=int)
        upper = np.asarray(high[:HAND_DOF], dtype=int)
        return np.minimum(lower, upper), np.maximum(lower, upper)

    def _measured_position(self) -> np.ndarray | None:
        for attempt in range(self.read_retries):
            result, state = self.arm.rm_get_rm_plus_state_info()
            if int(result) == 0 and isinstance(state, dict):
                position = state.get("pos")
                if isinstance(position, list) and len(position) >= HAND_DOF:
                    return np.asarray(
                        [
                            _clamp(position[i], self.lower[i], self.upper[i])
                            for i in range(HAND_DOF)
                        ],
                        dtype=int,
                    )
            if attempt + 1 < self.read_retries and self.retry_delay:
                time.sleep(self.retry_delay)
        return None

    def normalized_to_position(self, joints: Sequence[float]) -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.shape != (HAND_DOF,) or not np.all(np.isfinite(values)):
            raise ValueError("BrainCo hand target must contain six finite values.")
        values = np.clip(values, 0.0, 1.0)
        return np.rint(self.lower + values * (self.upper - self.lower)).astype(int)

    def send_normalized(self, joints: Sequence[float]) -> bool:
        """Filter and send one target, returning False when no send is needed."""
        now = time.monotonic()
        dt = now - self._last_filter_time
        self._last_filter_time = now
        self.filtered_joints = self.motion_filter.update(joints, dt)
        if now - self.last_send_time < self.min_send_interval:
            return False

        target = self.normalized_to_position(self.filtered_joints)
        if np.array_equal(target, self.target):
            return False
        result = int(self.arm.rm_set_hand_follow_pos(target.tolist(), True))
        self.last_send_time = time.monotonic()
        if result != 0:
            raise RuntimeError(
                "rm_set_hand_follow_pos failed with RealMan error code "
                f"{result}."
            )
        self.target = target
        return True

    def request_motion(self, motion: str, *, now: float | None = None) -> bool:
        """Start a non-blocking ``grab`` or ``release`` preset.

        Grab uses the same three targets and inter-stage delays as the keyboard
        teleop script. Calling :meth:`advance_motion` from the VR packet loop
        progresses it without pausing arm teleoperation.
        """
        motion = str(motion).strip().lower()
        if motion == "grab":
            thumb_rotate = np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, GRAB_THUMB_ROTATE],
                dtype=float,
            )
            four_fingers = np.asarray(
                [
                    GRAB_THUMB_PRE_CLOSE,
                    GRAB_FOUR_FINGER_CLOSE,
                    GRAB_FOUR_FINGER_CLOSE,
                    GRAB_FOUR_FINGER_CLOSE,
                    GRAB_FOUR_FINGER_CLOSE,
                    GRAB_THUMB_ROTATE,
                ],
                dtype=float,
            )
            thumb_close = four_fingers.copy()
            thumb_close[THUMB_FLEX] = GRAB_THUMB_CLOSE
            self._motion_stages = [
                (thumb_rotate, 0.80),
                (four_fingers, 0.80),
                (thumb_close, 0.90),
            ]
        elif motion == "release":
            self._motion_stages = [(np.zeros(HAND_DOF, dtype=float), 0.0)]
        else:
            raise ValueError(
                f"Unsupported BrainCo hand motion {motion!r}; "
                "expected 'grab' or 'release'."
            )

        self.active_motion = motion
        self._next_motion_stage_time = time.monotonic() if now is None else float(now)
        return self.advance_motion(now=now)

    def advance_motion(self, *, now: float | None = None) -> bool:
        """Send the next due preset stage, if one is ready."""
        if not self._motion_stages:
            return False

        current_time = time.monotonic() if now is None else float(now)
        earliest_send = max(
            self._next_motion_stage_time,
            self.last_send_time + self.min_send_interval,
        )
        if current_time < earliest_send:
            return False

        normalized, wait_after_s = self._motion_stages.pop(0)
        target = self.normalized_to_position(normalized)
        result = int(self.arm.rm_set_hand_follow_pos(target.tolist(), True))
        if result != 0:
            self._motion_stages.clear()
            self.active_motion = None
            raise RuntimeError(
                "rm_set_hand_follow_pos failed with RealMan error code "
                f"{result}."
            )

        self.last_send_time = current_time
        self.target = target
        self.filtered_joints = normalized.copy()
        self.motion_filter.reset(normalized)
        self._last_filter_time = current_time
        if self._motion_stages:
            self._next_motion_stage_time = current_time + wait_after_s
        else:
            self.active_motion = None
        return True

    def map_openxr_hand(self, bones: Sequence[Sequence[float]]) -> np.ndarray:
        """Map a frame using its initial thumb pose as the rotation endpoint."""
        thumb_progress = openxr_thumb_opposition_progress(bones)
        if self._thumb_rotate_open_progress is None:
            self._thumb_rotate_open_progress = thumb_progress
        return openxr_to_brainco_joints(
            bones,
            thumb_rotate_open_progress=self._thumb_rotate_open_progress,
            thumb_rotate_progress_range=self.thumb_rotate_progress_range,
        )

    def follow_openxr_hand(self, bones: Sequence[Sequence[float]]) -> np.ndarray:
        """Recognize, filter, send, and return the filtered six-DoF target."""
        # Live hand tracking supersedes any controller preset that had not yet
        # completed when the tracking mode changed.
        self._motion_stages.clear()
        self.active_motion = None
        joints = self.map_openxr_hand(bones)
        self.send_normalized(joints)
        return self.filtered_joints.copy()

    def close(self) -> None:
        """The RealMan arm owns the shared RM_ARM+ connection."""
