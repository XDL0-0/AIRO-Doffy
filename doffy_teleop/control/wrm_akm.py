"""Abstract kinematic mapping for a vertically mounted RealMan RM75.

This module is deliberately independent from the legacy VR/camera scripts.  It
owns the optional Unity UDP channel on port 8005 and the RM75 arm-angle IK used
by :mod:`realman_teleop` when ``Config.WRM_enable`` is true.

Unity packet formats
--------------------
The recommended format is JSON::

    {"type":"WRM","frame_id":42,"timestamp_ns":123,
     "position":[x,y,z],"rotation":[qx,qy,qz,qw],
     "elbow_alpha":0.35,"confidence":0.92,
     "grip_trigger":1.0,"index_trigger":0.0,
     "joystick":[0,0],"joystick_press":false}

An elbow-only JSON packet is also valid.  This lets an existing controller
pose stream remain on port 8001.  For simple Unity senders, these CSV forms are
accepted as well::

    WRM,frame_id,timestamp_ns,elbow_alpha,confidence
    WRM,frame_id,timestamp_ns,px,py,pz,qx,qy,qz,qw,elbow_alpha,confidence

Human elbow data never becomes a robot joint command.  ``elbow_alpha`` selects
an abstract RM75 arm-angle target between a high-elbow configuration and a
shoulder-height configuration; the vendor IK then solves the TCP and redundant
configuration together from the previous accepted solution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import threading
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WrmTrackingSample:
    """One parsed Unity upper-body tracking sample."""

    frame_id: int
    timestamp_ns: int
    received_ns: int
    elbow_alpha: float
    confidence: float
    position: tuple[float, float, float] | None = None
    rotation_xyzw: tuple[float, float, float, float] | None = None
    joystick: tuple[float, float] = (0.0, 0.0)
    index_trigger: float = 0.0
    grip_trigger: float = 0.0
    button_ax: int = 0
    button_by: int = 0
    joystick_press: int = 0

    @property
    def has_controller_pose(self) -> bool:
        return self.position is not None and self.rotation_xyzw is not None

    def as_controller_data(self, fallback: list[dict] | None = None) -> list[dict] | None:
        """Return the legacy two-controller shape consumed by RealManTeleop."""

        if not self.has_controller_pose:
            return fallback
        if fallback is not None and len(fallback) == 2:
            left = dict(fallback[0])
        else:
            left = {
                "ControllerType": "LTouch",
                "FrameId": self.frame_id,
                "Timestamp": self.timestamp_ns,
                "Position": (0.0, 0.0, 0.0),
                "Rotation": (0.0, 0.0, 0.0, 1.0),
                "Joystick": (0.0, 0.0),
                "IndexTrigger": 0.0,
                "GripTrigger": 0.0,
                "Button_AX": 0,
                "Button_BY": 0,
                "Joystick_Press": 0,
            }
        right = {
            "ControllerType": "RTouch",
            "FrameId": self.frame_id,
            "Timestamp": self.timestamp_ns,
            "Position": self.position,
            "Rotation": self.rotation_xyzw,
            "Joystick": self.joystick,
            "IndexTrigger": self.index_trigger,
            "GripTrigger": self.grip_trigger,
            "Button_AX": self.button_ax,
            "Button_BY": self.button_by,
            "Joystick_Press": self.joystick_press,
        }
        return [left, right]


def _first(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, dict):
        keys = ("x", "y", "z", "w")[:length]
        value = [value[key] for key in keys]
    result = tuple(float(item) for item in value)
    if len(result) != length or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def parse_wrm_unity_packet(data: bytes | str, *, received_ns: int | None = None) -> WrmTrackingSample:
    """Parse one WRM JSON/CSV datagram and validate all safety-relevant fields."""

    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = str(data)
    text = text.strip()
    if not text:
        raise ValueError("empty WRM packet")
    received = time.monotonic_ns() if received_ns is None else int(received_ns)

    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("WRM JSON root must be an object")
        nested = _first(payload, "right_arm", "rightArm", "wrm", default={})
        if isinstance(nested, dict):
            payload = {**payload, **nested}
        alpha = float(_first(payload, "elbow_alpha", "elbowAlpha"))
        confidence = float(
            _first(payload, "confidence", "tracking_confidence", "trackingConfidence", default=1.0)
        )
        position_value = _first(payload, "position", "controller_position", "controllerPosition")
        rotation_value = _first(
            payload,
            "rotation",
            "rotation_xyzw",
            "controller_rotation",
            "controllerRotation",
        )
        position = None if position_value is None else _vector(position_value, 3, "position")
        rotation = None if rotation_value is None else _vector(rotation_value, 4, "rotation")
        if (position is None) != (rotation is None):
            raise ValueError("controller position and rotation must be supplied together")
        joystick = _vector(_first(payload, "joystick", default=(0.0, 0.0)), 2, "joystick")
        sample = WrmTrackingSample(
            frame_id=int(_first(payload, "frame_id", "frameId", default=0)),
            timestamp_ns=int(_first(payload, "timestamp_ns", "timestampNs", "timestamp", default=0)),
            received_ns=received,
            elbow_alpha=alpha,
            confidence=confidence,
            position=position,
            rotation_xyzw=rotation,
            joystick=joystick,
            index_trigger=float(_first(payload, "index_trigger", "indexTrigger", default=0.0)),
            grip_trigger=float(_first(payload, "grip_trigger", "gripTrigger", default=0.0)),
            button_ax=int(bool(_first(payload, "button_ax", "buttonAX", default=0))),
            button_by=int(bool(_first(payload, "button_by", "buttonBY", default=0))),
            joystick_press=int(bool(_first(payload, "joystick_press", "joystickPress", default=0))),
        )
    else:
        fields = [field.strip() for field in text.split(",")]
        if fields[0].upper() not in {"WRM", "AKM"}:
            raise ValueError("WRM CSV packet must begin with WRM or AKM")
        if len(fields) == 5:
            frame_id, timestamp_ns = int(fields[1]), int(fields[2])
            sample = WrmTrackingSample(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                received_ns=received,
                elbow_alpha=float(fields[3]),
                confidence=float(fields[4]),
            )
        elif len(fields) >= 12:
            values = [float(value) for value in fields[3:12]]
            extras = [float(value) for value in fields[12:]]
            extras += [0.0] * (7 - len(extras))
            sample = WrmTrackingSample(
                frame_id=int(fields[1]),
                timestamp_ns=int(fields[2]),
                received_ns=received,
                position=tuple(values[0:3]),
                rotation_xyzw=tuple(values[3:7]),
                elbow_alpha=values[7],
                confidence=values[8],
                joystick=(extras[0], extras[1]),
                index_trigger=extras[2],
                grip_trigger=extras[3],
                button_ax=int(bool(extras[4])),
                button_by=int(bool(extras[5])),
                joystick_press=int(bool(extras[6])),
            )
        else:
            raise ValueError("WRM CSV packet has an unsupported field count")

    if not np.isfinite(sample.elbow_alpha) or not 0.0 <= sample.elbow_alpha <= 1.0:
        raise ValueError("elbow_alpha must be finite and in [0, 1]")
    if not np.isfinite(sample.confidence) or not 0.0 <= sample.confidence <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    if sample.rotation_xyzw is not None:
        quaternion = np.asarray(sample.rotation_xyzw, dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-6:
            raise ValueError("controller quaternion has zero norm")
        object.__setattr__(sample, "rotation_xyzw", tuple(quaternion / norm))
    return sample


class WrmUdpReceiver:
    """Latest-value UDP receiver for Unity WRM tracking packets."""

    def __init__(self, bind_ip: str, port: int, *, socket_factory=socket.socket) -> None:
        if not 1 <= int(port) <= 65535:
            raise ValueError("WRM UDP port must be between 1 and 65535")
        self._lock = threading.Lock()
        self._sample: WrmTrackingSample | None = None
        self._sequence = 0
        self._invalid_packets = 0
        self._last_error = ""
        self._socket = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((str(bind_ip), int(port)))
        self._socket.settimeout(0.1)
        self.address = (str(bind_ip), int(port))

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                packet, _ = self._socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                if stop_event.is_set():
                    break
                raise
            try:
                sample = parse_wrm_unity_packet(packet)
            except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
                with self._lock:
                    self._invalid_packets += 1
                    self._last_error = str(exc)
                continue
            with self._lock:
                self._sample = sample
                self._sequence += 1
                self._last_error = ""

    def snapshot(self) -> tuple[WrmTrackingSample | None, int]:
        with self._lock:
            return self._sample, self._sequence

    @property
    def diagnostics(self) -> tuple[int, str]:
        with self._lock:
            return self._invalid_packets, self._last_error

    def close(self) -> None:
        self._socket.close()


@dataclass(frozen=True)
class Rm75AkmSettings:
    confidence_threshold: float = 0.55
    tracking_timeout_s: float = 0.25
    minimum_elbow_height_m: float = 0.20
    horizontal_elbow_height_m: float = 0.2405
    calibration_span_deg: float = 120.0
    calibration_step_deg: float = 5.0
    maximum_arm_angle_rate_deg_s: float = 90.0
    elbow_singularity_margin_deg: float = 3.0


class Rm75ArmAngleIk:
    """RM75 TCP + abstract arm-angle IK with a continuous previous-frame seed."""

    # Official RM75 standard-DH geometry.  Joint-4 origin is the physical elbow.
    _D = np.array([0.2405, 0.0, 0.256, 0.0, 0.210, 0.0, 0.0])
    _ALPHA = np.radians(np.array([-90.0, 90.0, -90.0, 90.0, -90.0, 90.0, 0.0]))

    def __init__(
        self,
        arm: Any,
        api: Any,
        initial_joints_radians: np.ndarray,
        initial_robot_tcp: np.ndarray,
        *,
        lower_limits_radians: np.ndarray | None = None,
        upper_limits_radians: np.ndarray | None = None,
        settings: Rm75AkmSettings | None = None,
    ) -> None:
        self.arm = arm
        self.api = api
        self.settings = Rm75AkmSettings() if settings is None else settings
        self.dof = 7
        self.last_status = 0
        self.last_rejection = ""
        self._state_lock = threading.Lock()
        self._last_solution = self._validate_joints(initial_joints_radians).copy()
        self._initial_elbow_sign = np.sign(self._last_solution[3]) or 1.0
        self._lower = self._optional_limits(lower_limits_radians)
        self._upper = self._optional_limits(upper_limits_radians)

        required = (
            "rm_algo_calculate_arm_angle_from_config_rm75",
            "rm_algo_inverse_kinematics_rm75_for_arm_angle",
            "rm_set_self_collision_enable",
        )
        missing = [name for name in required if not callable(getattr(arm, name, None))]
        params_type = getattr(api, "rm_inverse_kinematics_params_t", None)
        if missing or params_type is None:
            details = missing + ([] if params_type is not None else ["rm_inverse_kinematics_params_t"])
            raise RuntimeError("WRM AKM requires RM75 SDK feature(s): " + ", ".join(details))
        self._params_type = params_type

        collision_result = int(self.arm.rm_set_self_collision_enable(True))
        if collision_result != 0:
            raise RuntimeError(
                "Cannot enable RM75 controller self-collision protection "
                f"(error {collision_result})"
            )

        status, high_angle = self.arm.rm_algo_calculate_arm_angle_from_config_rm75(
            np.degrees(self._last_solution).tolist()
        )
        if int(status) != 0 or not np.isfinite(high_angle):
            raise RuntimeError(f"Cannot calculate the initial RM75 arm angle (error {status})")
        self.robot_elbow_high = float(high_angle)
        self.robot_elbow_horizontal = self._calibrate_horizontal_angle(initial_robot_tcp)
        self._raw_target = self.robot_elbow_high
        self._filtered_target = self.robot_elbow_high
        self._last_target_update_ns = time.monotonic_ns()
        self._latest_elbow_alpha: float | None = None
        self._latest_confidence: float | None = None
        # The TCP coupling is reference-relative. Re-referencing while the
        # clutch is open therefore cannot apply the same drop a second time.
        self._tcp_z_reference_progress = 0.0
        self.tracking_frozen = True

    @staticmethod
    def _validate_joints(joints: np.ndarray) -> np.ndarray:
        result = np.asarray(joints, dtype=float)
        if result.shape != (7,) or not np.all(np.isfinite(result)):
            raise ValueError("RM75 AKM requires seven finite joint angles")
        return result

    def _optional_limits(self, limits: np.ndarray | None) -> np.ndarray | None:
        if limits is None:
            return None
        return self._validate_joints(limits)

    @classmethod
    def elbow_position(cls, joints_radians: np.ndarray) -> np.ndarray:
        """Return the joint-4/elbow origin in the vertically mounted base frame."""

        joints = cls._validate_joints(joints_radians)
        transform = np.eye(4)
        for index in range(3):
            theta = joints[index]
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(cls._ALPHA[index]), np.sin(cls._ALPHA[index])
            transform = transform @ np.array(
                [
                    [ct, -st * ca, st * sa, 0.0],
                    [st, ct * ca, -ct * sa, 0.0],
                    [0.0, sa, ca, cls._D[index]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        return transform[:3, 3].copy()

    @staticmethod
    def _pose_vector(pose: np.ndarray) -> list[float]:
        matrix = np.asarray(pose, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("RM75 AKM TCP target must be a finite 4x4 pose")
        # Proper rotation only: no mirror/reflection matrix is ever applied.
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("RM75 AKM TCP rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.0:
            raise ValueError("RM75 AKM refuses reflected/mirrored rotations")
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            index = int(np.argmax(np.diag(rotation)))
            if index == 0:
                scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
                qw = (rotation[2, 1] - rotation[1, 2]) / scale
                qx = 0.25 * scale
                qy = (rotation[0, 1] + rotation[1, 0]) / scale
                qz = (rotation[0, 2] + rotation[2, 0]) / scale
            elif index == 1:
                scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
                qw = (rotation[0, 2] - rotation[2, 0]) / scale
                qx = (rotation[0, 1] + rotation[1, 0]) / scale
                qy = 0.25 * scale
                qz = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
                qw = (rotation[1, 0] - rotation[0, 1]) / scale
                qx = (rotation[0, 2] + rotation[2, 0]) / scale
                qy = (rotation[1, 2] + rotation[2, 1]) / scale
                qz = 0.25 * scale
        quaternion_wxyz = np.asarray([qw, qx, qy, qz], dtype=float)
        quaternion_wxyz /= np.linalg.norm(quaternion_wxyz)
        return [
            *matrix[:3, 3].tolist(),
            *quaternion_wxyz.tolist(),
        ]

    def _call_solver(
        self,
        tcp_target: np.ndarray,
        arm_angle_deg: float,
        seed: np.ndarray,
    ) -> tuple[int, np.ndarray | None]:
        params = self._params_type(
            np.degrees(seed).tolist(),
            self._pose_vector(tcp_target),
            0,
        )
        status, output = self.arm.rm_algo_inverse_kinematics_rm75_for_arm_angle(
            params,
            float(arm_angle_deg),
        )
        status = int(status)
        if status != 0:
            return status, None
        candidate = np.radians(np.asarray(output, dtype=float))
        if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
            return -10, None
        return 0, candidate

    def _candidate_is_safe(self, candidate: np.ndarray) -> tuple[bool, str]:
        if self._lower is not None and np.any(candidate < self._lower):
            return False, "joint lower limit"
        if self._upper is not None and np.any(candidate > self._upper):
            return False, "joint upper limit"
        margin = np.radians(self.settings.elbow_singularity_margin_deg)
        if np.sign(candidate[3]) != self._initial_elbow_sign or abs(candidate[3]) < margin:
            return False, "J4 elbow flip/singularity boundary"
        elbow_z = float(self.elbow_position(candidate)[2])
        if elbow_z < self.settings.minimum_elbow_height_m:
            return False, f"elbow below minimum height ({elbow_z:.3f} m)"
        return True, ""

    def _calibrate_horizontal_angle(self, tcp_pose: np.ndarray) -> float:
        best: tuple[float, float] | None = None
        span = self.settings.calibration_span_deg
        step = self.settings.calibration_step_deg
        offsets = np.arange(-span, span + 0.5 * step, step)
        for offset in offsets:
            angle = self.robot_elbow_high + float(offset)
            status, candidate = self._call_solver(tcp_pose, angle, self._last_solution)
            if status != 0 or candidate is None:
                continue
            safe, _ = self._candidate_is_safe(candidate)
            if not safe:
                continue
            elbow_z = float(self.elbow_position(candidate)[2])
            score = abs(elbow_z - self.settings.horizontal_elbow_height_m)
            # Prefer the smaller redundancy excursion when heights tie.
            score += 1e-5 * abs(offset)
            if best is None or score < best[0]:
                best = (score, angle)
        if best is None:
            raise RuntimeError("No safe shoulder-height RM75 elbow configuration was found")
        return float(best[1])

    def update_tracking(self, sample: WrmTrackingSample | None, *, now_ns: int | None = None) -> bool:
        """Update the arm-angle objective, freezing it on low/stale confidence."""

        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        valid = (
            sample is not None
            and sample.confidence >= self.settings.confidence_threshold
            and (now - sample.received_ns) / 1e9 <= self.settings.tracking_timeout_s
        )
        with self._state_lock:
            if sample is not None:
                self._latest_elbow_alpha = float(sample.elbow_alpha)
                self._latest_confidence = float(sample.confidence)
            if not valid:
                self.tracking_frozen = True
                self._last_target_update_ns = now
                return False

            # Required inverse abstract mapping. Alpha=0 is high; alpha=1 is
            # shoulder-horizontal. These values are arm angles, never joints.
            alpha = float(sample.elbow_alpha)
            self._raw_target = (
                (1.0 - alpha) * self.robot_elbow_high
                + alpha * self.robot_elbow_horizontal
            )
            dt = max(0.0, (now - self._last_target_update_ns) / 1e9)
            maximum_step = (
                self.settings.maximum_arm_angle_rate_deg_s * min(dt, 0.05)
            )
            delta = np.clip(
                self._raw_target - self._filtered_target,
                -maximum_step,
                maximum_step,
            )
            self._filtered_target += float(delta)
            self._last_target_update_ns = now
            self.tracking_frozen = False
            return True

    @property
    def elbow_target_deg(self) -> float:
        with self._state_lock:
            return float(self._filtered_target)

    def _arm_angle_progress_unlocked(self) -> float:
        span = self.robot_elbow_horizontal - self.robot_elbow_high
        if abs(span) < 1e-9:
            return 0.0
        progress = (self._filtered_target - self.robot_elbow_high) / span
        return float(np.clip(progress, 0.0, 1.0))

    @property
    def arm_angle_progress(self) -> float:
        """Return the rate-limited high-to-horizontal arm-angle progress."""

        with self._state_lock:
            return self._arm_angle_progress_unlocked()

    def set_tcp_z_reference(self) -> None:
        """Make the current arm-angle progress the zero TCP-Z offset.

        This is called whenever controller/robot references are refreshed.
        It prevents an already-applied drop from accumulating after the user
        releases and re-engages the clutch.
        """

        with self._state_lock:
            self._tcp_z_reference_progress = self._arm_angle_progress_unlocked()

    def tcp_z_offset(self, maximum_drop_m: float) -> float:
        """Return the base-frame Z offset for the current WRM progress."""

        maximum_drop = float(maximum_drop_m)
        if not np.isfinite(maximum_drop) or maximum_drop < 0.0:
            raise ValueError("maximum_drop_m must be finite and non-negative")
        with self._state_lock:
            progress_delta = (
                self._arm_angle_progress_unlocked()
                - self._tcp_z_reference_progress
            )
        # Increasing alpha/progress must lower TCP Z.
        return -maximum_drop * progress_delta

    def couple_tcp_target_z(
        self,
        tcp_target: np.ndarray,
        maximum_drop_m: float,
    ) -> np.ndarray:
        """Apply smooth, reference-relative WRM progress to TCP base-frame Z."""

        target = np.asarray(tcp_target, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("WRM TCP coupling requires a finite 4x4 target pose")
        coupled = target.copy()
        coupled[2, 3] += self.tcp_z_offset(maximum_drop_m)
        return coupled

    def visualizer_state(self) -> dict[str, float | bool | None]:
        """Return a coherent, read-only snapshot for the teleop dashboard."""

        with self._state_lock:
            return {
                "elbow_alpha": self._latest_elbow_alpha,
                "confidence": self._latest_confidence,
                "arm_angle_target_deg": float(self._filtered_target),
                "arm_angle_high_deg": float(self.robot_elbow_high),
                "arm_angle_horizontal_deg": float(
                    self.robot_elbow_horizontal
                ),
                "arm_angle_progress": self._arm_angle_progress_unlocked(),
                "tcp_z_reference_progress": float(
                    self._tcp_z_reference_progress
                ),
                "tracking_frozen": bool(self.tracking_frozen),
            }

    def reset_seed(self, joints_radians: np.ndarray) -> None:
        with self._state_lock:
            self._last_solution = self._validate_joints(joints_radians).copy()

    def accept_solution(self, joints_radians: np.ndarray) -> None:
        """Commit an externally safety-checked/filter-applied solution as seed."""

        with self._state_lock:
            self._last_solution = self._validate_joints(joints_radians).copy()

    def solve(self, tcp_target: np.ndarray) -> np.ndarray | None:
        with self._state_lock:
            arm_angle = float(self._filtered_target)
            seed = self._last_solution.copy()
        status, candidate = self._call_solver(
            tcp_target,
            arm_angle,
            seed,
        )
        self.last_status = status
        if candidate is None:
            self.last_rejection = f"IK status {status}"
            return None
        safe, reason = self._candidate_is_safe(candidate)
        if not safe:
            self.last_status = -20
            self.last_rejection = reason
            return None
        self.last_rejection = ""
        return candidate
