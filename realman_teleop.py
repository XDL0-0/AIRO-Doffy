"""Lean RealMan VR teleoperation with a dedicated high-rate CAN-FD stream.

This entry point intentionally contains only:

* VR controller -> joint or TCP control;
* RealSense capture/streaming through the existing camera managers;
* the RealMan robot's integrated six-axis force sensor; and
* the shared teleoperation visualizer.

With the default realtime UDP state push, camera work, robot state/force
updates, and visualization stay off the CAN-FD deadline thread. The deadline
thread continuously resends the latest safe target at 200 Hz by default and
verifies that its measured rate remains strictly above 100 Hz. The explicit
legacy polling fallback serializes state reads on that same owner thread so no
SDK handle is used concurrently.
"""

from __future__ import annotations

from ctypes import c_float
from dataclasses import dataclass
import json
import socket
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
from airo_spatial_algebra.se3 import SE3Container

import utils
from config import Config
from dataset import DatasetRecorder
from force_filter import WrenchFilter
from robot_backend import RealManBackend, make_robot_backend
from visualizer_config import VisualizerConfig


_UNCLOSED_REALMAN_TELEOPS: list["RealManTeleop"] = []


def pack_quest_tcp_state_packet(tcp_pose: np.ndarray, wrench: np.ndarray, tcp_display_axes: np.ndarray) -> bytes:
    """Encode one RealMan TCP state for Quest ``TCPPoseReceiver``.

    Position and rotation are converted from RealMan base frame (X forward,
    Y left, Z up) into the Unity world frame (X right, Y up, Z forward)
    so that they line up correctly after the user's calibration.

    Force ``[Fx, Fy, Fz]`` is in newtons; torque is not currently sent.
    """

    pose = np.asarray(tcp_pose, dtype=float)
    values = np.asarray(wrench, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("Quest TCP state packets require a finite 4x4 TCP pose.")
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Quest TCP state packets require six finite wrench values.")

    to_unity = np.asarray(tcp_display_axes, dtype=float)

    position_unity = to_unity @ pose[:3, 3]
    rotation_unity = to_unity @ pose[:3, :3] @ to_unity.T
    force_unity = to_unity @ values[:3]

    quaternion_xyzw = utils.quat_cal(rotation_unity)
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    message = {
        "rightTCP": {
            "position": position_unity.tolist(),
            "rotation": quaternion_wxyz.tolist(),
            "force": force_unity.tolist(),
        }
    }
    return json.dumps(message).encode("utf-8")


@dataclass(frozen=True)
class CanfdLoopSnapshot:
    """Thread-safe diagnostic snapshot for the high-rate command loop."""

    target_hz: float
    achieved_hz: float | None
    total_commands: int
    deadline_misses: int
    high_follow_gap_violations: int
    sdk_call_overruns: int
    max_gap_ms: float
    max_sdk_call_ms: float
    last_command_start_ns: int
    last_command_success_ns: int
    completed_timing_windows: int
    consecutive_timing_failure_windows: int
    timing_verified: bool
    running: bool
    error: str


@dataclass(frozen=True)
class RealManStateSnapshot:
    """Latest measured robot state, copied out of the sensor cache."""

    joints: np.ndarray
    tcp_pose: np.ndarray
    wrench: np.ndarray
    state_timestamp_ns: int
    force_timestamp_ns: int
    state_error: str
    force_error: str
    input_stale: bool


class QuestTcpStateSender:
    """Publish cached RealMan TCP pose and force to Quest at a fixed rate."""

    def __init__(
        self,
        teleop: "RealManTeleop",
        *,
        quest_ip: str,
        port: int,
        send_rate_hz: float,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ) -> None:
        if not str(quest_ip).strip():
            raise ValueError("quest_ip cannot be empty.")
        if not 1 <= int(port) <= 65535:
            raise ValueError("Quest TCP state port must be between 1 and 65535.")
        if float(send_rate_hz) <= 0.0:
            raise ValueError("Quest TCP state send rate must be positive.")

        self.teleop = teleop
        self.destination = (str(quest_ip), int(port))
        self.period_s = 1.0 / float(send_rate_hz)
        self._socket_factory = socket_factory

    def run(
        self,
        stop_event: threading.Event,
        background_errors: list[str] | None = None,
    ) -> None:
        """Send the latest fresh TCP state until the stop event is set."""

        udp_socket = None
        next_send = time.monotonic()
        skipped_reason = ""
        try:
            udp_socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            utils.logger.info(
                "Quest TCP state transfer enabled: %s:%d at %.1f Hz.",
                self.destination[0],
                self.destination[1],
                1.0 / self.period_s,
            )
            while not stop_event.is_set():
                snapshot = self.teleop.state_snapshot()
                force_age_s = max(
                    0.0,
                    (time.monotonic_ns() - snapshot.force_timestamp_ns) / 1e9,
                )
                state_age_s = max(
                    0.0,
                    (time.monotonic_ns() - snapshot.state_timestamp_ns) / 1e9,
                )
                reason = snapshot.state_error or snapshot.force_error
                if not reason and state_age_s > self.teleop.sensor_stale_after_s:
                    reason = f"TCP pose is stale ({state_age_s * 1000.0:.0f} ms)"
                if not reason and force_age_s > self.teleop.sensor_stale_after_s:
                    reason = f"force is stale ({force_age_s * 1000.0:.0f} ms)"

                if reason:
                    if reason != skipped_reason:
                        utils.logger.warning(
                            "Quest TCP state transfer paused: %s.",
                            reason,
                        )
                    skipped_reason = reason
                else:
                    if skipped_reason:
                        utils.logger.info("Quest TCP state transfer resumed.")
                    skipped_reason = ""
                    packet = pack_quest_tcp_state_packet(
                        snapshot.tcp_pose,
                        snapshot.wrench,
                        self.teleop.cfg.TCP_DISPLAY_AXES,
                    )
                    udp_socket.sendto(packet, self.destination)

                next_send += self.period_s
                now = time.monotonic()
                if next_send < now:
                    next_send = now
                stop_event.wait(next_send - now)
        except Exception as exc:
            message = f"Quest TCP state sender failed: {exc}"
            utils.logger.exception(message)
            if background_errors is not None:
                background_errors.append(message)
            stop_event.set()
        finally:
            if udp_socket is not None:
                udp_socket.close()


class RealManRemoteIkSolver:
    """Thin radians-facing adapter for RealMan's teleoperation QP solver."""

    def __init__(
        self,
        arm: Any,
        api: Any,
        *,
        dof: int,
        period_s: float,
        initial_joints_radians: np.ndarray,
        joint_acceleration_limits_radians: np.ndarray,
        dq_weight: float,
        limit_holdon: bool,
        elbow_margin_degrees: float,
    ) -> None:
        required = (
            "rm_algo_ik_remote_init",
            "rm_algo_set_error_weight",
            "rm_algo_set_dq_weight",
            "rm_algo_ik_remote",
        )
        missing = [name for name in required if not callable(getattr(arm, name, None))]
        if missing:
            raise TypeError(
                "RealMan SDK is missing teleoperation solver method(s): "
                + ", ".join(missing)
            )
        matrix_type = getattr(api, "rm_Mat_t", None)
        if matrix_type is None:
            raise TypeError("RealMan SDK does not expose rm_Mat_t.")

        self.arm = arm
        self.api = api
        self.dof = int(dof)
        self._matrix_type = matrix_type
        self._output_type = c_float * self.dof
        self.last_status = 0

        # Targets in this application are absolute poses in the robot work/base
        # frame, hence tool_or_work=1. dT must match the fixed solver cadence.
        self.arm.rm_algo_ik_remote_init(float(period_s), 1)
        self.arm.rm_algo_set_error_weight([1.0] * 6)
        self.arm.rm_algo_set_dq_weight([float(dq_weight)] * self.dof)
        set_joint_max_acc = getattr(self.arm, "rm_algo_set_joint_max_acc", None)
        if callable(set_joint_max_acc):
            acceleration_limits = np.asarray(
                joint_acceleration_limits_radians,
                dtype=float,
            )
            if (
                acceleration_limits.shape != (self.dof,)
                or np.any(acceleration_limits <= 0.0)
            ):
                raise ValueError(
                    "joint_acceleration_limits_radians must be positive with "
                    f"shape ({self.dof},)."
                )
            # RealMan's algorithm API uses RPM/s; the rest of this application
            # uses SI radians/s^2.
            set_joint_max_acc(
                (acceleration_limits * 60.0 / (2.0 * np.pi)).tolist()
            )
        set_limit_holdon = getattr(self.arm, "rm_algo_set_enable_limit_holdon", None)
        if callable(set_limit_holdon):
            set_limit_holdon(int(limit_holdon))

        self._configure_nonzero_elbow_limit(
            np.asarray(initial_joints_radians, dtype=float),
            float(elbow_margin_degrees),
        )

    @staticmethod
    def is_available(arm: Any, api: Any) -> bool:
        return (
            api is not None
            and getattr(api, "rm_Mat_t", None) is not None
            and callable(getattr(arm, "rm_algo_ik_remote_init", None))
            and callable(getattr(arm, "rm_algo_ik_remote", None))
        )

    def _configure_nonzero_elbow_limit(
        self,
        initial_joints_radians: np.ndarray,
        margin_degrees: float,
    ) -> None:
        set_limit = getattr(self.arm, "rm_algo_set_joint_limit_angle", None)
        if not callable(set_limit):
            utils.logger.warning(
                "RealMan SDK cannot configure the QP elbow limit; configure a "
                "nonzero elbow soft limit on the teach pendant."
            )
            return

        elbow_index = 3 if self.dof == 7 else 2
        if initial_joints_radians.shape != (self.dof,):
            raise ValueError(f"Expected {self.dof} initial RealMan joint angles.")
        initial_elbow_degrees = float(np.degrees(initial_joints_radians[elbow_index]))
        if abs(initial_elbow_degrees) <= margin_degrees:
            raise ValueError(
                "Initial RealMan elbow is too close to the QP singular boundary: "
                f"{initial_elbow_degrees:.2f} degrees. Move it beyond "
                f"+/-{margin_degrees:g} degrees before teleoperation."
            )

        dof_type = (
            self.api.rm_dofType_e.DOF_TYPE_7
            if self.dof == 7
            else self.api.rm_dofType_e.DOF_TYPE_6
        )
        joint = (
            self.api.rm_jointType_e.JOINT_Q4
            if self.dof == 7
            else self.api.rm_jointType_e.JOINT_Q3
        )
        if initial_elbow_degrees < 0.0:
            limit_type = self.api.rm_limitType_e.LIMIT_MAX
            limit_angle = -margin_degrees
        else:
            limit_type = self.api.rm_limitType_e.LIMIT_MIN
            limit_angle = margin_degrees
        result = set_limit(dof_type, joint, limit_type, limit_angle)
        if result != 0:
            raise RuntimeError(
                "rm_algo_set_joint_limit_angle failed with RealMan error code "
                f"{result}."
            )

    def solve(
        self,
        tcp_target: np.ndarray,
        current_joints_radians: np.ndarray,
    ) -> np.ndarray | None:
        tcp_target = np.asarray(tcp_target, dtype=float)
        current = np.asarray(current_joints_radians, dtype=float)
        if tcp_target.shape != (4, 4) or not np.all(np.isfinite(tcp_target)):
            raise ValueError("Expected a finite 4x4 QP TCP target.")
        if current.shape != (self.dof,) or not np.all(np.isfinite(current)):
            raise ValueError(f"Expected {self.dof} finite current joint angles.")

        matrix = self._matrix_type(row=4, col=4, data=tcp_target.tolist())
        q_out = self._output_type()
        self.last_status = int(
            self.arm.rm_algo_ik_remote(
                matrix,
                np.degrees(current).tolist(),
                q_out,
            )
        )
        if self.last_status != 0:
            return None
        result = np.radians(np.asarray(list(q_out), dtype=float))
        if result.shape != (self.dof,) or not np.all(np.isfinite(result)):
            return None
        return result


class CanfdCommandLoop:
    """Continuously send the latest joint or TCP target through RealMan CAN-FD."""

    def __init__(
        self,
        arm: Any,
        *,
        control_mode: str,
        dof: int,
        target_hz: float,
        minimum_hz: float,
        rate_check_window: float,
        maximum_failure_windows: int,
        trajectory_mode: int = 1,
        radio: int = 60,
        joint_speed_limits: float | np.ndarray | None = None,
        joint_acceleration_limits: float | np.ndarray | None = None,
        linear_speed_limit: float | None = None,
        linear_acceleration_limit: float | None = None,
        angular_speed_limit: float | None = None,
        angular_acceleration_limit: float | None = None,
        heartbeat_timeout: float = 0.05,
    ) -> None:
        if control_mode not in {"joint", "tcp"}:
            raise ValueError(f"Unsupported CAN-FD control mode: {control_mode}")
        if target_hz <= minimum_hz:
            raise ValueError(
                f"CAN-FD target rate ({target_hz:g} Hz) must be strictly above "
                f"the minimum ({minimum_hz:g} Hz)."
            )
        if minimum_hz < 100.0:
            raise ValueError("CAN-FD minimum rate must be at least 100 Hz.")
        if rate_check_window <= 0.0:
            raise ValueError("CAN-FD rate check window must be positive.")
        if maximum_failure_windows < 1:
            raise ValueError("maximum_failure_windows must be at least 1.")
        if trajectory_mode not in {0, 1, 2}:
            raise ValueError("trajectory_mode must be 0, 1, or 2.")
        if radio < 0:
            raise ValueError("radio cannot be negative.")
        if trajectory_mode == 1 and radio > 100:
            raise ValueError("curve-fit CAN-FD radio must be between 0 and 100.")
        if trajectory_mode == 2 and radio > 999:
            raise ValueError("filter CAN-FD radio must be between 0 and 999.")

        required_method = "rm_movej_canfd" if control_mode == "joint" else "rm_movep_canfd"
        if not callable(getattr(arm, required_method, None)):
            raise TypeError(f"RealMan SDK object does not expose {required_method}().")

        self.arm = arm
        self.control_mode = control_mode
        self.dof = int(dof)
        self.target_hz = float(target_hz)
        self.minimum_hz = float(minimum_hz)
        self.period_s = 1.0 / self.target_hz
        self.rate_check_window = float(rate_check_window)
        self.maximum_failure_windows = int(maximum_failure_windows)
        self.trajectory_mode = int(trajectory_mode)
        self.radio = int(radio)
        if joint_speed_limits is None:
            self.joint_speed_limits = np.full(self.dof, np.inf)
        else:
            limits = np.asarray(joint_speed_limits, dtype=float)
            if limits.ndim == 0:
                limits = np.full(self.dof, float(limits))
            if limits.shape != (self.dof,) or np.any(limits <= 0.0):
                raise ValueError(
                    f"joint_speed_limits must be positive with shape ({self.dof},)."
                )
            self.joint_speed_limits = limits
        if joint_acceleration_limits is None:
            self.joint_acceleration_limits = np.full(self.dof, np.inf)
        else:
            acceleration_limits = np.asarray(
                joint_acceleration_limits,
                dtype=float,
            )
            if acceleration_limits.ndim == 0:
                acceleration_limits = np.full(
                    self.dof,
                    float(acceleration_limits),
                )
            if (
                acceleration_limits.shape != (self.dof,)
                or np.any(acceleration_limits <= 0.0)
            ):
                raise ValueError(
                    "joint_acceleration_limits must be positive with shape "
                    f"({self.dof},)."
                )
            self.joint_acceleration_limits = acceleration_limits
        self.linear_speed_limit = (
            np.inf if linear_speed_limit is None else float(linear_speed_limit)
        )
        self.linear_acceleration_limit = (
            np.inf
            if linear_acceleration_limit is None
            else float(linear_acceleration_limit)
        )
        self.angular_speed_limit = (
            np.inf if angular_speed_limit is None else float(angular_speed_limit)
        )
        self.angular_acceleration_limit = (
            np.inf
            if angular_acceleration_limit is None
            else float(angular_acceleration_limit)
        )
        if self.linear_speed_limit <= 0.0:
            raise ValueError("linear_speed_limit must be positive.")
        if self.linear_acceleration_limit <= 0.0:
            raise ValueError("linear_acceleration_limit must be positive.")
        if self.angular_speed_limit <= 0.0:
            raise ValueError("angular_speed_limit must be positive.")
        if self.angular_acceleration_limit <= 0.0:
            raise ValueError("angular_acceleration_limit must be positive.")
        self.heartbeat_timeout = float(heartbeat_timeout)
        if self.heartbeat_timeout <= 0.01:
            raise ValueError("heartbeat_timeout must be greater than 10 ms.")

        self._target_lock = threading.Lock()
        self._target: np.ndarray | None = None
        self._setpoint: np.ndarray | None = None
        self._joint_velocity = np.zeros(self.dof)
        self._linear_velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._planned_joint_velocity = np.zeros(self.dof)
        self._planned_linear_velocity = np.zeros(3)
        self._planned_angular_velocity = np.zeros(3)
        self._hold_requested = False
        self._pending_lock = threading.Lock()
        self._pending_joint_target: (
            tuple[np.ndarray, float, int] | None
        ) = None
        self._joint_request_generation = 0
        self._joint_target_resolver: (
            Callable[[np.ndarray, float], np.ndarray | None] | None
        ) = None
        self._continuous_joint_resolution = False
        self._maintenance_callback: Callable[[], None] | None = None
        self._maintenance_period_ns = 0

        self._stats_lock = threading.Lock()
        self._achieved_hz: float | None = None
        self._total_commands = 0
        self._deadline_misses = 0
        self._high_follow_gap_violations = 0
        self._sdk_call_overruns = 0
        self._max_gap_ms = 0.0
        self._max_sdk_call_ms = 0.0
        self._last_command_start_ns = 0
        self._last_command_success_ns = 0
        self._completed_timing_windows = 0
        self._consecutive_timing_failure_windows = 0
        self._timing_verified = False
        self._running = False
        self._error = ""

    def set_joint_target(self, joints_radians: np.ndarray) -> None:
        if self.control_mode != "joint":
            raise RuntimeError("Cannot set a joint target while CAN-FD is in TCP mode.")
        target = np.asarray(joints_radians, dtype=float)
        if target.shape != (self.dof,) or not np.all(np.isfinite(target)):
            raise ValueError(f"Expected a finite joint target with shape ({self.dof},).")
        with self._target_lock:
            self._target = target.copy()
            self._hold_requested = False
            if self._setpoint is None:
                self._setpoint = target.copy()
                self._reset_velocity_state()

    def set_tcp_target(self, realman_pose: np.ndarray) -> None:
        if self.control_mode != "tcp":
            raise RuntimeError("Cannot set a TCP target while CAN-FD is in joint mode.")
        target = np.asarray(realman_pose, dtype=float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("Expected a finite RealMan TCP target [x,y,z,rx,ry,rz].")
        with self._target_lock:
            self._target = target.copy()
            self._hold_requested = False
            if self._setpoint is None:
                self._setpoint = target.copy()
                self._reset_velocity_state()

    def set_joint_target_resolver(
        self,
        resolver: Callable[[np.ndarray, float], np.ndarray | None],
        *,
        continuous: bool = False,
    ) -> None:
        if self.control_mode != "joint":
            raise RuntimeError("A joint target resolver is only valid in joint mode.")
        self._joint_target_resolver = resolver
        self._continuous_joint_resolution = bool(continuous)

    def request_joint_target(self, tcp_pose: np.ndarray, dt: float) -> None:
        """Publish the newest TCP target for IK on the CAN-FD owner thread."""

        if self.control_mode != "joint":
            raise RuntimeError("Joint target requests are only valid in joint mode.")
        pose = np.asarray(tcp_pose, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("Expected a finite 4x4 TCP target.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Joint target request dt must be positive and finite.")
        with self._pending_lock:
            self._joint_request_generation += 1
            self._pending_joint_target = (
                pose.copy(),
                float(dt),
                self._joint_request_generation,
            )

    def set_maintenance_callback(
        self,
        callback: Callable[[], None],
        rate_hz: float,
    ) -> None:
        """Schedule low-rate SDK maintenance on the CAN-FD owner thread."""

        if rate_hz <= 0.0:
            raise ValueError("Maintenance rate must be positive.")
        self._maintenance_callback = callback
        self._maintenance_period_ns = max(1, round(1e9 / rate_hz))

    def resolve_pending_target(self) -> bool:
        if self.control_mode != "joint":
            return False
        with self._pending_lock:
            pending = self._pending_joint_target
            if not self._continuous_joint_resolution:
                self._pending_joint_target = None
        if pending is None:
            return False
        if self._joint_target_resolver is None:
            raise RuntimeError("No joint target resolver was configured.")
        tcp_pose, request_dt, generation = pending
        dt = self.period_s if self._continuous_joint_resolution else request_dt
        resolved = self._joint_target_resolver(tcp_pose, dt)
        if resolved is None:
            return False
        resolved = np.asarray(resolved, dtype=float)
        if resolved.shape != (self.dof,) or not np.all(np.isfinite(resolved)):
            raise ValueError(
                f"Expected a finite joint target with shape ({self.dof},)."
            )
        # Keep the generation check and target commit atomic with respect to
        # hold_current_setpoint(). A release/timeout that happens while IK is
        # in flight invalidates this result before it can reach CAN-FD.
        with self._pending_lock:
            if generation != self._joint_request_generation:
                return False
            with self._target_lock:
                self._target = resolved.copy()
                self._hold_requested = False
        return True

    @staticmethod
    def _limit_vector_step(delta: np.ndarray, maximum_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(delta))
        if norm <= maximum_norm or norm == 0.0:
            return delta
        return delta * (maximum_norm / norm)

    def _reset_velocity_state(self) -> None:
        self._joint_velocity.fill(0.0)
        self._linear_velocity.fill(0.0)
        self._angular_velocity.fill(0.0)
        self._planned_joint_velocity.fill(0.0)
        self._planned_linear_velocity.fill(0.0)
        self._planned_angular_velocity.fill(0.0)

    def _limited_joint_step(
        self,
        error: np.ndarray,
        velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        braking_speed = np.full(self.dof, np.inf)
        finite_acceleration = np.isfinite(self.joint_acceleration_limits)
        braking_speed[finite_acceleration] = np.sqrt(
            2.0
            * self.joint_acceleration_limits[finite_acceleration]
            * np.abs(error[finite_acceleration])
        )
        desired_velocity = np.zeros(self.dof)
        moving = error != 0.0
        desired_velocity[moving] = np.sign(error[moving]) * np.minimum(
            self.joint_speed_limits[moving],
            braking_speed[moving],
        )
        maximum_velocity_change = (
            self.joint_acceleration_limits * self.period_s
        )
        next_velocity = velocity + np.clip(
            desired_velocity - velocity,
            -maximum_velocity_change,
            maximum_velocity_change,
        )
        step = next_velocity * self.period_s
        reaches_target = (
            (step * error >= 0.0)
            & (np.abs(step) >= np.abs(error))
            & (error != 0.0)
        )
        step[reaches_target] = error[reaches_target]
        next_velocity[reaches_target] = 0.0
        return step, next_velocity

    def _limited_vector_velocity_step(
        self,
        error: np.ndarray,
        velocity: np.ndarray,
        speed_limit: float,
        acceleration_limit: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        distance = float(np.linalg.norm(error))
        if distance == 0.0:
            desired_velocity = np.zeros(3)
        else:
            desired_speed = min(
                speed_limit,
                float(np.sqrt(2.0 * acceleration_limit * distance)),
            )
            desired_velocity = error * (desired_speed / distance)
        next_velocity = velocity + self._limit_vector_step(
            desired_velocity - velocity,
            acceleration_limit * self.period_s,
        )
        step = next_velocity * self.period_s
        if (
            distance > 0.0
            and float(np.dot(step, error)) >= 0.0
            and float(np.linalg.norm(step)) >= distance
        ):
            return error, np.zeros(3)
        return step, next_velocity

    def _next_setpoint(self) -> np.ndarray:
        with self._target_lock:
            if self._target is None or self._setpoint is None:
                raise RuntimeError("CAN-FD target was not initialized before the loop started.")
            if self._hold_requested:
                self._target = self._setpoint.copy()
                self._hold_requested = False
                self._reset_velocity_state()
            target = self._target.copy()
            current = self._setpoint.copy()
            joint_velocity = self._joint_velocity.copy()
            linear_velocity = self._linear_velocity.copy()
            angular_velocity = self._angular_velocity.copy()

        if self.control_mode == "joint":
            # RealMan joints are bounded, not continuous modulo 2π. Direct
            # interpolation stays inside the interval between two validated
            # joint configurations and cannot wrap across a physical limit.
            step, next_velocity = self._limited_joint_step(
                target - current,
                joint_velocity,
            )
            with self._target_lock:
                self._planned_joint_velocity = next_velocity
            return current + step

        translation_delta, next_linear_velocity = (
            self._limited_vector_velocity_step(
                target[:3] - current[:3],
                linear_velocity,
                self.linear_speed_limit,
                self.linear_acceleration_limit,
            )
        )
        current_se3 = SE3Container.from_euler_angles_and_translation(
            current[3:],
            current[:3],
        )
        target_se3 = SE3Container.from_euler_angles_and_translation(
            target[3:],
            target[:3],
        )
        rotation_delta = current_se3.rotation_matrix.T @ target_se3.rotation_matrix
        rotation_vector, _ = cv2.Rodrigues(rotation_delta)
        rotation_step, next_angular_velocity = (
            self._limited_vector_velocity_step(
                rotation_vector.reshape(3),
                angular_velocity,
                self.angular_speed_limit,
                self.angular_acceleration_limit,
            )
        )
        rotation_increment, _ = cv2.Rodrigues(rotation_step)
        next_se3 = SE3Container.from_rotation_matrix_and_translation(
            current_se3.rotation_matrix @ rotation_increment,
            current[:3] + translation_delta,
        )
        next_setpoint = np.concatenate(
            [next_se3.translation, next_se3.orientation_as_euler_angles]
        )
        with self._target_lock:
            self._planned_linear_velocity = next_linear_velocity
            self._planned_angular_velocity = next_angular_velocity
        return next_setpoint

    def _commit_setpoint(self, setpoint: np.ndarray) -> None:
        with self._target_lock:
            self._setpoint = np.asarray(setpoint, dtype=float).copy()
            if self._hold_requested:
                self._target = self._setpoint.copy()
                self._hold_requested = False
                self._reset_velocity_state()
            elif self.control_mode == "joint":
                self._joint_velocity = self._planned_joint_velocity.copy()
            else:
                self._linear_velocity = self._planned_linear_velocity.copy()
                self._angular_velocity = self._planned_angular_velocity.copy()

    def hold_current_setpoint(self) -> None:
        """Stop progressing toward a pending target at the next CAN-FD packet."""

        with self._pending_lock:
            self._joint_request_generation += 1
            self._pending_joint_target = None
        with self._target_lock:
            if self._setpoint is not None:
                self._target = self._setpoint.copy()
                self._hold_requested = True

    def target_reached(self) -> bool:
        """Return whether the rate-limited setpoint has reached its target."""

        with self._target_lock:
            if self._target is None or self._setpoint is None:
                return False
            target = self._target.copy()
            setpoint = self._setpoint.copy()

        if self.control_mode == "joint":
            return bool(np.allclose(setpoint, target, rtol=0.0, atol=1e-6))

        if not np.allclose(setpoint[:3], target[:3], rtol=0.0, atol=1e-5):
            return False
        setpoint_rotation = SE3Container.from_euler_angles_and_translation(
            setpoint[3:],
            setpoint[:3],
        ).rotation_matrix
        target_rotation = SE3Container.from_euler_angles_and_translation(
            target[3:],
            target[:3],
        ).rotation_matrix
        rotation_delta = setpoint_rotation.T @ target_rotation
        rotation_vector, _ = cv2.Rodrigues(rotation_delta)
        return float(np.linalg.norm(rotation_vector)) <= 1e-4

    def send_once(self) -> None:
        """Send one high-follow packet. Exposed for hardware-adapter tests."""

        target = self._next_setpoint()
        if self.control_mode == "joint":
            result = self.arm.rm_movej_canfd(
                np.degrees(target).tolist(),
                True,
                0,
                self.trajectory_mode,
                self.radio,
            )
            method_name = "rm_movej_canfd"
        else:
            result = self.arm.rm_movep_canfd(
                target.tolist(),
                True,
                self.trajectory_mode,
                self.radio,
            )
            method_name = "rm_movep_canfd"

        if result != 0:
            raise RuntimeError(f"{method_name} failed with RealMan error code {result}.")
        self._commit_setpoint(target)

    def _record_command_gap(
        self,
        previous_start_ns: int | None,
        start_ns: int,
    ) -> tuple[bool, int]:
        if previous_start_ns is None:
            return False, 0
        gap_ns = start_ns - previous_start_ns
        gap_ms = gap_ns / 1e6
        violated = gap_ms > 10.0
        with self._stats_lock:
            self._max_gap_ms = max(self._max_gap_ms, gap_ms)
            # High-follow requires no more than 10 ms between packets.
            if violated:
                self._high_follow_gap_violations += 1
        return violated, gap_ns

    def _record_sdk_call_duration(self, duration_ns: int) -> bool:
        duration_ms = duration_ns / 1e6
        violated = duration_ms > 10.0
        with self._stats_lock:
            self._max_sdk_call_ms = max(self._max_sdk_call_ms, duration_ms)
            if violated:
                self._sdk_call_overruns += 1
        return violated

    def _finish_rate_window(
        self,
        interval_count: int,
        elapsed_s: float,
        gap_violations: int,
    ) -> tuple[bool, bool]:
        achieved_hz = interval_count / elapsed_s
        with self._stats_lock:
            self._completed_timing_windows += 1
            first_measurement = self._achieved_hz is None
            self._achieved_hz = achieved_hz
            rate_too_slow = achieved_hz <= self.minimum_hz
            if rate_too_slow or gap_violations:
                self._consecutive_timing_failure_windows += 1
            else:
                self._consecutive_timing_failure_windows = 0
            timing_failed = (
                self._consecutive_timing_failure_windows
                >= self.maximum_failure_windows
            )

        if first_measurement:
            utils.logger.info(
                f"RealMan CAN-FD measured rate: {achieved_hz:.1f} Hz "
                f"(target {self.target_hz:g} Hz)"
            )
        if rate_too_slow:
            utils.logger.warning(
                f"RealMan CAN-FD rate is too slow: {achieved_hz:.1f} Hz "
                f"(minimum must be > {self.minimum_hz:g} Hz)."
            )
        if gap_violations:
            utils.logger.warning(
                f"RealMan CAN-FD had {gap_violations} packet gap(s) above 10 ms "
                "in the latest timing window."
            )
        return timing_failed, rate_too_slow

    def run(self, stop_event: threading.Event) -> None:
        """Run until stopped, or stop the application after sustained low rate."""

        period_ns = max(1, round(self.period_s * 1e9))
        window_ns = max(1, round(self.rate_check_window * 1e9))
        previous_start_ns: int | None = None
        window_start_ns = time.perf_counter_ns()
        window_intervals = 0
        window_interval_ns = 0
        window_gap_violations = 0
        next_tick_ns = window_start_ns
        next_maintenance_ns = window_start_ns

        with self._stats_lock:
            self._running = True
            self._error = ""
            self._timing_verified = False

        try:
            while not stop_event.is_set():
                command_start_ns = time.perf_counter_ns()
                gap_violated, gap_ns = self._record_command_gap(
                    previous_start_ns,
                    command_start_ns,
                )
                with self._stats_lock:
                    timing_verified = self._timing_verified
                if timing_verified and gap_violated:
                    raise RuntimeError(
                        "CAN-FD packet gap exceeded the 10 ms high-follow "
                        f"limit ({gap_ns / 1e6:.2f} ms)."
                    )
                with self._stats_lock:
                    self._last_command_start_ns = command_start_ns
                # The teleoperation QP solver is a continuous controller: use
                # the latest Cartesian target at this loop's fixed dT before
                # emitting each joint packet. Legacy one-shot IK remains after
                # the packet so existing adapters do not change behavior.
                if self._continuous_joint_resolution:
                    self.resolve_pending_target()
                self.send_once()
                command_complete_ns = time.perf_counter_ns()
                with self._stats_lock:
                    self._last_command_success_ns = command_complete_ns
                call_violated = self._record_sdk_call_duration(
                    command_complete_ns - command_start_ns
                )
                window_gap_violations += int(gap_violated or call_violated)
                if timing_verified and call_violated:
                    raise RuntimeError(
                        "CAN-FD SDK call exceeded the 10 ms high-follow limit "
                        f"({(command_complete_ns - command_start_ns) / 1e6:.2f} ms)."
                    )
                if gap_ns:
                    window_intervals += 1
                    window_interval_ns += gap_ns
                previous_start_ns = command_start_ns

                with self._stats_lock:
                    self._total_commands += 1

                now_ns = time.perf_counter_ns()
                window_elapsed_ns = now_ns - window_start_ns
                if window_elapsed_ns >= window_ns and window_intervals:
                    interval_elapsed_s = window_interval_ns / 1e9
                    timing_failed, rate_too_slow = self._finish_rate_window(
                        window_intervals,
                        interval_elapsed_s,
                        window_gap_violations,
                    )
                    if timing_failed:
                        snapshot = self.snapshot()
                        if rate_too_slow:
                            reason = (
                                f"send rate remained at or below {self.minimum_hz:g} Hz "
                                f"(latest: {snapshot.achieved_hz:.1f} Hz)"
                            )
                        else:
                            reason = "packet timing repeatedly exceeded the 10 ms high-follow limit"
                        raise RuntimeError(
                            f"CAN-FD {reason} for "
                            f"{snapshot.consecutive_timing_failure_windows} "
                            "consecutive timing windows."
                        )
                    window_start_ns = now_ns
                    window_intervals = 0
                    window_interval_ns = 0
                    window_gap_violations = 0

                if not self._continuous_joint_resolution:
                    self.resolve_pending_target()
                if (
                    self._maintenance_callback is not None
                    and time.perf_counter_ns() >= next_maintenance_ns
                ):
                    self._maintenance_callback()
                    next_maintenance_ns += self._maintenance_period_ns
                    now_ns = time.perf_counter_ns()
                    if next_maintenance_ns <= now_ns:
                        next_maintenance_ns = (
                            now_ns + self._maintenance_period_ns
                        )

                next_tick_ns += period_ns
                remaining_s = (next_tick_ns - time.perf_counter_ns()) / 1e9
                if remaining_s > 0.0:
                    stop_event.wait(remaining_s)
                else:
                    with self._stats_lock:
                        self._deadline_misses += 1
                    # Do not issue a burst of stale packets after an overrun.
                    next_tick_ns = time.perf_counter_ns()

        except Exception as exc:
            with self._stats_lock:
                self._error = str(exc)
            utils.logger.exception("RealMan CAN-FD loop stopped")
            stop_event.set()
        finally:
            with self._stats_lock:
                self._running = False

    def snapshot(self) -> CanfdLoopSnapshot:
        with self._stats_lock:
            return CanfdLoopSnapshot(
                target_hz=self.target_hz,
                achieved_hz=self._achieved_hz,
                total_commands=self._total_commands,
                deadline_misses=self._deadline_misses,
                high_follow_gap_violations=self._high_follow_gap_violations,
                sdk_call_overruns=self._sdk_call_overruns,
                max_gap_ms=self._max_gap_ms,
                max_sdk_call_ms=self._max_sdk_call_ms,
                last_command_start_ns=self._last_command_start_ns,
                last_command_success_ns=self._last_command_success_ns,
                completed_timing_windows=self._completed_timing_windows,
                consecutive_timing_failure_windows=(
                    self._consecutive_timing_failure_windows
                ),
                timing_verified=self._timing_verified,
                running=self._running,
                error=self._error,
            )

    def heartbeat_error(self) -> str:
        snapshot = self.snapshot()
        now_ns = time.perf_counter_ns()
        timeout_ns = self.heartbeat_timeout * 1e9
        if (
            snapshot.last_command_start_ns > snapshot.last_command_success_ns
            and now_ns - snapshot.last_command_start_ns > timeout_ns
        ):
            return (
                "CAN-FD SDK call has not completed for "
                f"{(now_ns - snapshot.last_command_start_ns) / 1e6:.1f} ms."
            )
        if (
            snapshot.running
            and snapshot.last_command_success_ns
            and now_ns - snapshot.last_command_success_ns > timeout_ns
        ):
            return (
                "CAN-FD has not completed a successful packet for "
                f"{(now_ns - snapshot.last_command_success_ns) / 1e6:.1f} ms."
            )
        return ""

    def report_external_failure(
        self,
        message: str,
        stop_event: threading.Event,
    ) -> None:
        with self._stats_lock:
            if not self._error:
                self._error = message
        stop_event.set()

    def wait_until_healthy(
        self,
        stop_event: threading.Event,
        timeout: float | None = None,
    ) -> CanfdLoopSnapshot:
        """Wait for a new clean timing window before enabling motion input."""

        if timeout is None:
            timeout = self.rate_check_window * (self.maximum_failure_windows + 2)
        deadline = time.monotonic() + timeout
        initial_window_count = self.snapshot().completed_timing_windows

        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if snapshot.error:
                raise RuntimeError(snapshot.error)
            heartbeat_error = self.heartbeat_error()
            if heartbeat_error:
                self.report_external_failure(heartbeat_error, stop_event)
                raise RuntimeError(heartbeat_error)
            if (
                snapshot.completed_timing_windows > initial_window_count
                and snapshot.achieved_hz is not None
                and snapshot.achieved_hz > self.minimum_hz
                and snapshot.consecutive_timing_failure_windows == 0
            ):
                with self._stats_lock:
                    self._timing_verified = True
                return self.snapshot()
            if stop_event.wait(0.01):
                snapshot = self.snapshot()
                if snapshot.error:
                    raise RuntimeError(snapshot.error)
                raise RuntimeError("CAN-FD timing check stopped before it completed.")

        raise TimeoutError(
            f"CAN-FD did not produce a clean >{self.minimum_hz:g} Hz timing "
            f"window within {timeout:.1f} seconds."
        )


class RealManTeleop:
    """RealMan controller/hand mapping and cached robot sensor state."""

    def __init__(
        self,
        initial_controller_data: list[dict],
        *,
        cfg: Config | None = None,
        backend: RealManBackend | None = None,
    ) -> None:
        self.cfg = Config() if cfg is None else cfg
        if self.cfg.ROBOT_TYPE != "realman":
            raise ValueError("realman_teleop.py requires Config.ROBOT_TYPE='realman'.")
        if self.cfg.GRIPPER:
            raise ValueError("Disable Config.GRIPPER for the RealMan-only teleoperation script.")
        if self.cfg.TACTILE_TRANSFER:
            raise ValueError("Disable Config.TACTILE_TRANSFER; this script uses robot force only.")

        created_backend = backend is None
        self.backend = make_robot_backend(self.cfg) if backend is None else backend
        try:
            if self.backend.name != "realman":
                raise TypeError("The selected backend is not a RealMan backend.")
            if not self.backend.supports_force:
                raise RuntimeError("The connected RealMan backend does not expose a force sensor.")

            self.dof = int(self.backend.dof)
            self.control_mode = self.cfg.TELEOP_COMMAND_MODE
            self.tracking_mode = self.cfg.TRACKING_MODE
            self.tcp_tool = getattr(self.backend, "tcp_tool", self.cfg.TCP_TOOL)
            self.hand = getattr(self.backend, "hand", None)
            self.joint_threshold = self._joint_threshold_for_dof(self.cfg.MOVE_THRESHOLD)
            self.vr_to_robot_axes = np.asarray(self.cfg.VR_TO_ROBOT_AXES, dtype=float)
            self.vr_to_robot_handedness = float(np.linalg.det(self.vr_to_robot_axes))
            self.tcp_transform = np.asarray(self.cfg.TCP_TRANSFORM, dtype=float)
            self.inv_tcp_transform = np.linalg.inv(self.tcp_transform)
            controller_reference = self._extract_controller_se3(
                initial_controller_data
            )

            initial_joint = self.backend.initial_joint_configuration(self.cfg.INITIAL_JOINT)
            self.initial_joint = np.asarray(initial_joint, dtype=float).copy()
            utils.logger.info(f"Moving RealMan to initial joint configuration: {initial_joint}")
            self.backend.reset(initial_joint)

            joints = np.asarray(
                self.backend.get_joint_configuration(),
                dtype=float,
            )
            tcp_pose = np.asarray(self.backend.get_tcp_pose(), dtype=float)
            raw_wrench = self.backend.get_tcp_force()
            if raw_wrench is None:
                raise RuntimeError(
                    "RealMan force sensor did not return a six-axis wrench during startup."
                )
            raw_wrench = np.asarray(raw_wrench, dtype=float)
            self._validate_sensor_state(joints, tcp_pose)
            self._validate_wrench(raw_wrench)
            self._finish_initialization(
                controller_reference,
                joints,
                tcp_pose,
                raw_wrench,
            )

        except Exception:
            if created_backend:
                self.backend.cleanup()
            raise

    def _finish_initialization(
        self,
        controller_reference: SE3Container,
        joints: np.ndarray,
        tcp_pose: np.ndarray,
        raw_wrench: np.ndarray,
    ) -> None:
        """Initialize local state after the robot connection has been verified."""

        self.wrench_filter = WrenchFilter(
            moving_average_window=self.cfg.FORCE_MOVING_AVERAGE_WINDOW,
            low_pass_alpha=self.cfg.FORCE_LOW_PASS_ALPHA,
        )
        wrench = self.wrench_filter.process(raw_wrench)

        self._sensor_lock = threading.Lock()
        now_ns = time.monotonic_ns()
        self._joints = joints.copy()
        self._tcp_pose = tcp_pose.copy()
        self._wrench = np.asarray(wrench, dtype=float)
        self._state_timestamp_ns = now_ns
        self._force_timestamp_ns = now_ns
        self._action_timestamp_ns = now_ns
        self._state_error = ""
        self._force_error = ""

        self._control_lock = threading.Lock()
        self._input_stale = False
        self._requires_reference = True
        self._grip_active = False
        self._controller_reset_held = False
        self._hand_joystick_motion: str | None = None
        self._hand_joystick_threshold = (
            self.cfg.BRAINCO_HAND_JOYSTICK_THRESHOLD
        )
        self._wrist_joint_bias = 0.0
        self._reset_in_progress = False
        self._reset_requires_grip_release = False
        self._last_joint_target = self._joints.copy()
        self._last_tcp_target = self._tcp_pose.copy()
        self._initial_tcp_pose = self._tcp_pose.copy()
        self._controller_reference = controller_reference
        self._robot_reference = SE3Container.from_homogeneous_matrix(self._tcp_pose)
        self._hand_initialized = False
        self._hand_last_palm: np.ndarray | None = None

        self._position_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=self.cfg.CARTESIAN_POS_FILTER_CUTOFF_HZ,
            dim=3,
        )
        self._rotation_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=self.cfg.CARTESIAN_ROT_FILTER_CUTOFF_HZ,
            dim=3,
        )
        self._joint_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=self.cfg.HAND_JOINT_FILTER_CUTOFF_HZ,
            dim=self.dof,
        )
        self._seed_joint_filter(self._joints)

        raw_arm = getattr(self.backend.robot, "robot", None)
        if raw_arm is None:
            raise TypeError("RealMan backend does not expose the vendor SDK arm object.")
        self._raw_arm = raw_arm
        self._realman_api = getattr(self.backend.robot, "_api", None)
        self._remote_ik_solver: RealManRemoteIkSolver | None = None
        self._last_remote_ik_status = 0
        self._state_push_callback = None
        self._state_push_active = False
        self._state_push_received = threading.Event()
        self._state_callback_condition = threading.Condition()
        self._accept_state_callbacks = False
        self._state_callbacks_in_flight = 0
        self._lifecycle_lock = threading.Lock()
        self._sdk_worker_threads: list[threading.Thread] = []
        self._started = False
        self._closed = False
        self._quarantined = False

        joint_speed_limits = np.full(
            self.dof,
            self.cfg.REALMAN_MAX_JOINT_SPEED,
            dtype=float,
        )
        joint_acceleration_limits = np.full(
            self.dof,
            self.cfg.REALMAN_MAX_JOINT_ACCELERATION,
            dtype=float,
        )
        linear_speed_limit = float(self.cfg.REALMAN_MAX_LINEAR_SPEED)
        specs = getattr(self.backend.robot, "manipulator_specs", None)
        if specs is not None:
            controller_joint_speeds = np.asarray(
                specs.max_joint_speeds,
                dtype=float,
            )
            if controller_joint_speeds.shape == (self.dof,):
                joint_speed_limits = np.minimum(
                    joint_speed_limits,
                    controller_joint_speeds,
                )
            controller_linear_speed = float(specs.max_linear_speed)
            if controller_linear_speed > 0.0:
                linear_speed_limit = min(
                    linear_speed_limit,
                    controller_linear_speed,
                )
        self._joint_speed_limits = joint_speed_limits

        if self.control_mode == "joint" and self.cfg.REALMAN_QP_IK_ENABLE:
            if RealManRemoteIkSolver.is_available(raw_arm, self._realman_api):
                self._remote_ik_solver = RealManRemoteIkSolver(
                    raw_arm,
                    self._realman_api,
                    dof=self.dof,
                    period_s=1.0 / self.cfg.REALMAN_CTRL_RATE,
                    initial_joints_radians=self._joints,
                    joint_acceleration_limits_radians=(
                        joint_acceleration_limits
                    ),
                    dq_weight=self.cfg.REALMAN_QP_DQ_WEIGHT,
                    limit_holdon=self.cfg.REALMAN_QP_LIMIT_HOLDON,
                    elbow_margin_degrees=self.cfg.REALMAN_QP_ELBOW_MARGIN_DEG,
                )
            else:
                utils.logger.warning(
                    "RealMan teleoperation QP solver is unavailable in this SDK; "
                    "falling back to one-shot inverse kinematics."
                )

        self.canfd = CanfdCommandLoop(
            raw_arm,
            control_mode=self.control_mode,
            dof=self.dof,
            target_hz=self.cfg.REALMAN_CTRL_RATE,
            minimum_hz=self.cfg.REALMAN_MIN_CANFD_RATE,
            rate_check_window=self.cfg.REALMAN_RATE_CHECK_WINDOW,
            maximum_failure_windows=self.cfg.REALMAN_RATE_FAILURE_WINDOWS,
            trajectory_mode=self.cfg.REALMAN_CANFD_TRAJECTORY_MODE,
            radio=self.cfg.REALMAN_CANFD_RADIO,
            joint_speed_limits=joint_speed_limits,
            joint_acceleration_limits=joint_acceleration_limits,
            linear_speed_limit=linear_speed_limit,
            linear_acceleration_limit=(
                self.cfg.REALMAN_MAX_LINEAR_ACCELERATION
            ),
            angular_speed_limit=self.cfg.REALMAN_MAX_ANGULAR_SPEED,
            angular_acceleration_limit=(
                self.cfg.REALMAN_MAX_ANGULAR_ACCELERATION
            ),
            heartbeat_timeout=self.cfg.REALMAN_CANFD_HEARTBEAT_TIMEOUT,
        )
        if self.control_mode == "joint":
            self.canfd.set_joint_target(self._last_joint_target)
            self.canfd.set_joint_target_resolver(
                self._resolve_joint_target,
                continuous=self._remote_ik_solver is not None,
            )
        else:
            self.canfd.set_tcp_target(self._tool_tcp_to_realman_pose(self._last_tcp_target))
        if not self.cfg.REALMAN_REALTIME_STATE_PUSH:
            self.canfd.set_maintenance_callback(
                self._refresh_sensors,
                self.cfg.REALMAN_SENSOR_RATE,
            )

        if self.cfg.GRAVITY_COMP:
            utils.logger.warning(
                "GRAVITY_COMP is ignored by realman_teleop.py because RealMan "
                "zero_force_data is already controller-compensated."
            )
        if self.cfg.REALMAN_REALTIME_STATE_PUSH:
            sensor_description = (
                f"UDP push {1000.0 / self.cfg.REALMAN_STATE_PUSH_CYCLE_MS:g} Hz"
            )
        else:
            sensor_description = (
                f"polling {self.cfg.REALMAN_SENSOR_RATE:g} Hz"
            )
        utils.logger.info(
            f"RealMan teleop ready - DoF:{self.dof}, mode:{self.control_mode}, "
            f"tracking:{self.tracking_mode}, tool:{self.tcp_tool}, "
            f"CAN-FD:{self.cfg.REALMAN_CTRL_RATE} Hz, "
            f"IK:{'QP continuous' if self._remote_ik_solver else 'legacy'}, "
            f"sensors:{sensor_description}"
        )

    def _joint_threshold_for_dof(self, threshold: np.ndarray | float) -> np.ndarray:
        values = np.asarray(threshold, dtype=float)
        if values.ndim == 0:
            return np.full(self.dof, float(values))
        if values.shape == (self.dof,):
            return values
        if values.size == 1:
            return np.full(self.dof, float(values.item()))
        return np.resize(values, self.dof)

    def _vr_pose_to_robot_se3(
        self,
        position: tuple[float, float, float] | np.ndarray,
        quaternion: tuple[float, float, float, float] | np.ndarray,
    ) -> SE3Container:
        position_vr = np.asarray(position, dtype=float)
        quaternion_vr = np.asarray(quaternion, dtype=float)
        position_robot = self.vr_to_robot_axes @ position_vr
        quaternion_robot = np.concatenate(
            [
                self.vr_to_robot_handedness
                * (self.vr_to_robot_axes @ quaternion_vr[:3]),
                quaternion_vr[3:],
            ]
        )
        return SE3Container.from_quaternion_and_translation(
            quaternion_robot,
            position_robot,
        )

    def _extract_controller_se3(self, controller_data: list[dict]) -> SE3Container:
        right = controller_data[1]
        return self._vr_pose_to_robot_se3(right["Position"], right["Rotation"])

    def _extract_hand_se3(self, right_hand: dict) -> SE3Container:
        wrist_pose = right_hand.get("wrist_pose")
        if wrist_pose is not None:
            return self._vr_pose_to_robot_se3(
                wrist_pose["position"],
                wrist_pose["rotation"],
            )
        return self._vr_pose_to_robot_se3(
            right_hand["bones"][0],
            np.array([0.0, 0.0, 0.0, 1.0]),
        )

    def _tool_tcp_to_realman_pose(self, tool_tcp_pose: np.ndarray) -> np.ndarray:
        robot_tcp_pose = np.asarray(tool_tcp_pose, dtype=float) @ self.inv_tcp_transform
        pose = SE3Container.from_homogeneous_matrix(robot_tcp_pose)
        return np.concatenate([pose.translation, pose.orientation_as_euler_angles])

    def _seed_joint_filter(self, joints: np.ndarray) -> None:
        self._joint_filter.value = np.asarray(joints, dtype=float).copy()
        self._joint_filter.initialized = True

    def _set_reference(
        self,
        controller_data: list[dict],
        measured_tcp: np.ndarray,
        measured_joints: np.ndarray,
    ) -> None:
        self._wrist_joint_bias = 0.0
        self._controller_reference = self._extract_controller_se3(controller_data)
        self._robot_reference = SE3Container.from_homogeneous_matrix(measured_tcp)
        self._position_filter.reset()
        self._rotation_filter.reset()
        self._seed_joint_filter(measured_joints)
        self._last_joint_target = np.asarray(
            measured_joints,
            dtype=float,
        ).copy()
        self._last_tcp_target = np.asarray(
            measured_tcp,
            dtype=float,
        ).copy()

    def _set_hand_reference(
        self,
        hand_pose: SE3Container,
        measured_tcp: np.ndarray,
        measured_joints: np.ndarray,
    ) -> None:
        self._controller_reference = hand_pose
        self._robot_reference = SE3Container.from_homogeneous_matrix(measured_tcp)
        self._position_filter.reset()
        self._rotation_filter.reset()
        self._seed_joint_filter(measured_joints)
        self._last_joint_target = np.asarray(measured_joints, dtype=float).copy()
        self._last_tcp_target = np.asarray(measured_tcp, dtype=float).copy()
        self._hand_initialized = True

    def _request_reset(self) -> None:
        """Return to the startup pose through the active CAN-FD stream."""

        # Cancel any in-flight IK result before publishing the reset target.
        # Keeping reset inside this command stream avoids concurrent access to
        # the vendor SDK from the VR thread.
        self.canfd.hold_current_setpoint()
        if self.control_mode == "joint":
            self.canfd.set_joint_target(self.initial_joint)
            self._last_joint_target = self.initial_joint.copy()
        else:
            self.canfd.set_tcp_target(
                self._tool_tcp_to_realman_pose(self._initial_tcp_pose)
            )
        self._last_tcp_target = self._initial_tcp_pose.copy()
        self._position_filter.reset()
        self._rotation_filter.reset()
        self._seed_joint_filter(self.initial_joint)
        self._requires_reference = True
        self._grip_active = False
        self._reset_in_progress = True
        self._reset_requires_grip_release = True
        self._action_timestamp_ns = time.monotonic_ns()
        utils.logger.info("RealMan reset requested; returning to the startup pose.")

    def _disable_hand_tool(self, reason: str) -> None:
        """Permanently downgrade the active BrainCo tool for this run."""
        close = getattr(self.hand, "close", None)
        if callable(close):
            close()
        self.hand = None
        self.backend.hand = None
        self.tcp_tool = "None"
        self.backend.tcp_tool = "None"
        self.cfg.TCP_TOOL = "None"
        utils.logger.warning("%s; falling back to TCP_TOOL='None'.", reason)

    def _process_brainco_joystick(self, right: dict) -> None:
        """Edge-trigger BrainCo grab/release and advance staged motion."""
        if self.hand is None or self.tcp_tool != "Hand":
            self._hand_joystick_motion = None
            return

        joystick = right.get("Joystick", (0.0, 0.0))
        joystick_y = float(joystick[1])
        if joystick_y > self._hand_joystick_threshold:
            motion = "grab"
        elif joystick_y < -self._hand_joystick_threshold:
            motion = "release"
        else:
            motion = None

        try:
            if motion is not None and motion != self._hand_joystick_motion:
                self.hand.request_motion(motion)
                utils.logger.info(
                    "BrainCo hand motion: %s (right joystick %s).",
                    motion,
                    "forward" if motion == "grab" else "backward",
                )
            else:
                advance = getattr(self.hand, "advance_motion", None)
                if callable(advance):
                    advance()
        except Exception as exc:
            self._disable_hand_tool(f"BrainCo hand command failed ({exc})")
        self._hand_joystick_motion = motion

    def _update_wrist_joint_bias(self, right: dict) -> None:
        """Accumulate right-stick X motion for the robot's final wrist joint."""
        joystick_x = float(right.get("Joystick", (0.0, 0.0))[0])
        direction = (joystick_x > 0.8) - (joystick_x < -0.8)
        self._wrist_joint_bias += float(direction) * 0.01

    def _apply_wrist_roll_to_tcp(self, tcp_target: np.ndarray) -> np.ndarray:
        """Represent final-joint bias as local tool-axis roll in TCP mode."""
        target = np.asarray(tcp_target, dtype=float).copy()
        rotation_increment, _ = cv2.Rodrigues(
            np.asarray([0.0, 0.0, self._wrist_joint_bias], dtype=float)
        )
        target[:3, :3] = target[:3, :3] @ rotation_increment
        return target

    def _target_from_controller(self, controller: SE3Container, dt: float) -> np.ndarray:
        controller_matrix = controller.homogeneous_matrix
        translation_delta = controller_matrix[:3, 3] - self._controller_reference.translation
        # Both orientations have already been converted from Unity into
        # RealMan coordinates. Their spatial difference is therefore a
        # rotation expressed directly in the RealMan base frame.
        rotation_delta = (
            controller_matrix[:3, :3]
            @ self._controller_reference.rotation_matrix.T
        )

        translation_delta = self._position_filter.update(translation_delta, dt)
        rotation_vector, _ = cv2.Rodrigues(rotation_delta)
        rotation_vector = self._rotation_filter.update(
            rotation_vector.reshape(3),
            dt,
        )

        rotation_delta, _ = cv2.Rodrigues(rotation_vector)

        target_translation = self._robot_reference.translation + translation_delta
        if self.cfg.FREEZE_ROTATION:
            target_rotation = self._robot_reference.rotation_matrix
        else:
            # Apply the controller's base-frame rotation directly. No Euler
            # axis permutation or RealMan-specific sign correction is needed.
            target_rotation = rotation_delta @ self._robot_reference.rotation_matrix
        return SE3Container.from_rotation_matrix_and_translation(
            target_rotation,
            target_translation,
        ).homogeneous_matrix

    def _joint_target_is_safe(
        self,
        target: np.ndarray,
        tcp_target: np.ndarray,
    ) -> bool:
        if target.shape != (self.dof,) or not np.all(np.isfinite(target)):
            return False
        if not self.backend.is_joint_target_safe(
            target,
            self._last_joint_target,
            tcp_target[:3, 3],
            self.joint_threshold,
        ):
            return False

        lower = getattr(self.backend.robot, "_joint_lower_limits", None)
        upper = getattr(self.backend.robot, "_joint_upper_limits", None)
        if lower is not None and upper is not None:
            if np.any(target < np.asarray(lower)) or np.any(target > np.asarray(upper)):
                utils.logger.warning("RealMan IK target exceeds a controller joint limit.")
                return False
        return True

    def _filter_joint_target(self, target: np.ndarray, dt: float) -> np.ndarray:
        return self._joint_filter.update(target, dt)

    def _resolve_joint_target(
        self,
        tcp_target: np.ndarray,
        dt: float,
    ) -> np.ndarray | None:
        """Run IK on the same fixed-rate thread that owns CAN-FD SDK calls."""

        snapshot = self.state_snapshot()
        if self._remote_ik_solver is not None:
            # VR targets describe the configured tool TCP. RealMan's QP matrix
            # describes the controller/flange TCP, matching the legacy backend
            # IK boundary and preserving a non-identity TCP_TRANSFORM.
            robot_tcp_target = self.backend.to_robot_tcp_pose(tcp_target)
            joint_target = self._remote_ik_solver.solve(
                robot_tcp_target,
                snapshot.joints,
            )
            solver_status = self._remote_ik_solver.last_status
        else:
            joint_target = self.backend.solve_tcp_ik(tcp_target, snapshot.joints)
            solver_status = 0 if joint_target is not None else -1
        if joint_target is None:
            if solver_status != self._last_remote_ik_status:
                utils.logger.warning(
                    "RealMan IK failed with status "
                    f"{solver_status}; keeping the previous CAN-FD target."
                )
            self._last_remote_ik_status = solver_status
            return None
        self._last_remote_ik_status = 0

        with self._control_lock:
            if self._input_stale or not self._grip_active:
                return None
            joint_target = np.asarray(joint_target, dtype=float)
            unadjusted_last_joint = float(joint_target[-1])
            joint_target[-1] += self._wrist_joint_bias
            lower = getattr(self.backend.robot, "_joint_lower_limits", None)
            upper = getattr(self.backend.robot, "_joint_upper_limits", None)
            if lower is not None and upper is not None:
                joint_target[-1] = np.clip(
                    joint_target[-1],
                    np.asarray(lower, dtype=float)[-1],
                    np.asarray(upper, dtype=float)[-1],
                )
                self._wrist_joint_bias = (
                    float(joint_target[-1]) - unadjusted_last_joint
                )
            if not self._joint_target_is_safe(joint_target, tcp_target):
                utils.logger.warning(
                    "Unsafe RealMan target rejected; holding position."
                )
                return None
            filtered = self._filter_joint_target(joint_target, dt)
            if not self._joint_target_is_safe(filtered, tcp_target):
                return None
            self._last_joint_target = filtered
            self._last_tcp_target = tcp_target
            return filtered

    def process_controller(
        self,
        controller_data: list[dict],
        dt: float,
    ) -> bool:
        """Consume one new VR packet and publish a new safe CAN-FD target."""

        snapshot = self.state_snapshot()
        maximum_state_age = self.sensor_stale_after_s
        state_age = (time.monotonic_ns() - snapshot.state_timestamp_ns) / 1e9
        if state_age > maximum_state_age:
            self.mark_input_stale(
                f"robot state is stale ({state_age * 1000.0:.0f} ms)"
            )
            return False

        with self._control_lock:
            right = controller_data[1]
            controller = self._extract_controller_se3(controller_data)

            reset_pressed = (
                bool(right.get("Joystick_Press", False))
                and float(right.get("IndexTrigger", 0.0))
                >= self.cfg.CONTROLLER_RESET_TRIGGER_THRESHOLD
            )
            if reset_pressed:
                if not self._controller_reset_held:
                    self._request_reset()
                self._controller_reset_held = True
                return False
            self._controller_reset_held = False

            self._process_brainco_joystick(right)

            if self._reset_in_progress:
                self._set_reference(
                    controller_data,
                    snapshot.tcp_pose,
                    snapshot.joints,
                )
                if self.canfd.target_reached():
                    self._reset_in_progress = False
                    utils.logger.info("RealMan reset motion complete.")
                return False

            # Do not let a trigger still held after reset immediately replace
            # the reset target. A fresh grip starts from a fresh reference.
            if self._reset_requires_grip_release:
                self._set_reference(
                    controller_data,
                    snapshot.tcp_pose,
                    snapshot.joints,
                )
                self._requires_reference = False
                self._grip_active = False
                if not right["GripTrigger"]:
                    self._reset_requires_grip_release = False
                return False

            if self._requires_reference or self._input_stale:
                self._set_reference(controller_data, snapshot.tcp_pose, snapshot.joints)
                self._requires_reference = False
                self._input_stale = False
                self._grip_active = bool(right["GripTrigger"])
                return False

            if not right["GripTrigger"]:
                if self._grip_active:
                    self.canfd.hold_current_setpoint()
                self._set_reference(controller_data, snapshot.tcp_pose, snapshot.joints)
                self._grip_active = False
                return False

            self._grip_active = True
            self._update_wrist_joint_bias(right)
            tcp_target = self._target_from_controller(controller, dt)
            if self.control_mode == "joint":
                self.canfd.request_joint_target(tcp_target, dt)
            else:
                tcp_target = self._apply_wrist_roll_to_tcp(tcp_target)
                self.canfd.set_tcp_target(self._tool_tcp_to_realman_pose(tcp_target))
                self._last_tcp_target = tcp_target
            self._action_timestamp_ns = time.monotonic_ns()
            return True

    def process_hand(self, hand_data: dict, dt: float) -> bool:
        """Consume one OpenXR right-hand packet for wrist and Revo2 control."""
        right_hand = hand_data.get("R")
        if right_hand is None:
            return False
        bones = right_hand.get("bones")
        if bones is None or len(bones) != 26:
            return False
        if np.linalg.norm(np.asarray(bones[10], dtype=float)) < 1e-4:
            return False

        snapshot = self.state_snapshot()
        state_age = (time.monotonic_ns() - snapshot.state_timestamp_ns) / 1e9
        if state_age > self.sensor_stale_after_s:
            self.mark_input_stale(
                f"robot state is stale ({state_age * 1000.0:.0f} ms)"
            )
            return False

        hand_pose = self._extract_hand_se3(right_hand)
        if self._hand_last_palm is not None:
            jump = float(
                np.linalg.norm(hand_pose.translation - self._hand_last_palm)
            )
            if jump > self.cfg.HAND_PALM_JUMP_THRESHOLD:
                utils.logger.warning(
                    "Hand wrist jump %.3fm > %.3fm; holding RealMan target.",
                    jump,
                    self.cfg.HAND_PALM_JUMP_THRESHOLD,
                )
                self._hand_initialized = False
                self._hand_last_palm = None
                self.canfd.hold_current_setpoint()
                return False
        self._hand_last_palm = hand_pose.translation.copy()

        if self.hand is not None:
            try:
                self.hand.follow_openxr_hand(bones)
            except ValueError as exc:
                utils.logger.debug("Ignoring invalid OpenXR hand frame: %s", exc)
            except Exception as exc:
                self._disable_hand_tool(f"BrainCo hand command failed ({exc})")

        with self._control_lock:
            if (
                not self._hand_initialized
                or self._requires_reference
                or self._input_stale
            ):
                self._set_hand_reference(
                    hand_pose,
                    snapshot.tcp_pose,
                    snapshot.joints,
                )
                self._requires_reference = False
                self._input_stale = False
                self._grip_active = True
                return False

            self._grip_active = True
            tcp_target = self._target_from_controller(hand_pose, dt)
            if self.control_mode == "joint":
                self.canfd.request_joint_target(tcp_target, dt)
            else:
                self.canfd.set_tcp_target(
                    self._tool_tcp_to_realman_pose(tcp_target)
                )
                self._last_tcp_target = tcp_target
            self._action_timestamp_ns = time.monotonic_ns()
            return True

    def mark_input_stale(self, reason: str = "VR input timed out") -> None:
        with self._control_lock:
            if self._input_stale:
                return
            utils.logger.warning(f"{reason}; holding the last RealMan target.")
            self.canfd.hold_current_setpoint()
            self._input_stale = True
            self._requires_reference = True
            self._grip_active = False
            self._hand_initialized = False
            self._hand_last_palm = None

    def _refresh_sensors(self) -> None:
        state_error = ""
        force_error = ""
        joints: np.ndarray | None = None
        tcp_pose: np.ndarray | None = None
        wrench: np.ndarray | None = None

        try:
            joints = np.asarray(self.backend.get_joint_configuration(), dtype=float)
            tcp_pose = np.asarray(self.backend.get_tcp_pose(), dtype=float)
            self._validate_sensor_state(joints, tcp_pose)
        except Exception as exc:
            state_error = str(exc)

        try:
            raw_wrench = self.backend.get_tcp_force()
            if raw_wrench is None:
                raise RuntimeError("RealMan force sensor returned no wrench.")
            raw_wrench = np.asarray(raw_wrench, dtype=float)
            self._validate_wrench(raw_wrench)
            wrench = self.wrench_filter.process(raw_wrench)
        except Exception as exc:
            force_error = str(exc)

        now_ns = time.monotonic_ns()
        with self._sensor_lock:
            previous_state_error = self._state_error
            previous_force_error = self._force_error
            if joints is not None and tcp_pose is not None:
                self._joints = joints
                self._tcp_pose = tcp_pose
                self._state_timestamp_ns = now_ns
            if wrench is not None:
                self._wrench = wrench
                self._force_timestamp_ns = now_ns
            self._state_error = state_error
            self._force_error = force_error

        if state_error and state_error != previous_state_error:
            utils.logger.warning(f"RealMan state read failed: {state_error}")
        if force_error and force_error != previous_force_error:
            utils.logger.warning(f"RealMan force read failed: {force_error}")

    def _validate_sensor_state(
        self,
        joints: np.ndarray,
        tcp_pose: np.ndarray,
    ) -> None:
        if joints.shape != (self.dof,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(
                f"RealMan returned invalid joints with shape {joints.shape}."
            )
        if tcp_pose.shape != (4, 4) or not np.all(np.isfinite(tcp_pose)):
            raise RuntimeError(
                f"RealMan returned an invalid TCP pose with shape {tcp_pose.shape}."
            )

    @staticmethod
    def _validate_wrench(wrench: np.ndarray) -> None:
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise RuntimeError(
                f"RealMan returned an invalid wrench with shape {wrench.shape}."
            )

    def _handle_realtime_state(self, state) -> None:
        """Copy one vendor UDP state-push callback into the local cache."""

        try:
            if int(state.errCode) != 0:
                raise RuntimeError(
                    f"RealMan realtime state parse error {int(state.errCode)}."
                )

            joints = np.radians(
                np.asarray(
                    list(state.joint_status.joint_position)[: self.dof],
                    dtype=float,
                )
            )
            waypoint = state.waypoint
            robot_tcp_pose = (
                SE3Container.from_euler_angles_and_translation(
                    np.array(
                        [
                            waypoint.euler.rx,
                            waypoint.euler.ry,
                            waypoint.euler.rz,
                        ],
                        dtype=float,
                    ),
                    np.array(
                        [
                            waypoint.position.x,
                            waypoint.position.y,
                            waypoint.position.z,
                        ],
                        dtype=float,
                    ),
                ).homogeneous_matrix
            )
            tcp_pose = robot_tcp_pose @ self.tcp_transform
            raw_wrench = np.asarray(
                list(state.force_sensor.zero_force),
                dtype=float,
            )
            force_coordinate = int(
                getattr(
                    state.force_sensor,
                    "coordinate",
                    self.cfg.REALMAN_FORCE_COORDINATE,
                )
            )
            if force_coordinate != self.cfg.REALMAN_FORCE_COORDINATE:
                raise RuntimeError(
                    "RealMan force frame mismatch: expected "
                    f"{self.cfg.REALMAN_FORCE_COORDINATE}, got "
                    f"{force_coordinate}."
                )
            self._validate_sensor_state(joints, tcp_pose)
            self._validate_wrench(raw_wrench)
            if not self._state_push_received.is_set():
                # The synchronous startup read is in the sensor frame. Reset
                # before the first pushed sample so work/tool-frame filtering
                # never mixes wrenches expressed in different coordinates.
                self.wrench_filter.reset()
            wrench = self.wrench_filter.process(raw_wrench)

            now_ns = time.monotonic_ns()
            with self._sensor_lock:
                self._joints = joints
                self._tcp_pose = tcp_pose
                self._wrench = wrench
                self._state_timestamp_ns = now_ns
                self._force_timestamp_ns = now_ns
                self._state_error = ""
                self._force_error = ""
            self._state_push_received.set()

        except Exception as exc:
            message = str(exc)
            with self._sensor_lock:
                self._state_error = message
                self._force_error = message

    def _state_push_callback_entry(self, state) -> None:
        """Reject late packets during shutdown and track callbacks in flight."""

        with self._state_callback_condition:
            if not self._accept_state_callbacks:
                return
            self._state_callbacks_in_flight += 1
        try:
            self._handle_realtime_state(state)
        finally:
            with self._state_callback_condition:
                self._state_callbacks_in_flight -= 1
                if self._state_callbacks_in_flight == 0:
                    self._state_callback_condition.notify_all()

    def _stop_accepting_state_callbacks(self) -> None:
        deadline = time.monotonic() + self.cfg.REALMAN_STATE_PUSH_TIMEOUT
        with self._state_callback_condition:
            self._accept_state_callbacks = False
            while self._state_callbacks_in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Timed out waiting for a RealMan realtime state "
                        "callback to finish."
                    )
                self._state_callback_condition.wait(remaining)

    def _start_realtime_state_push(self) -> None:
        api = self._realman_api
        if api is None:
            raise RuntimeError(
                "Installed airo-robots RealMan wrapper does not expose its SDK module."
            )
        callback_type = getattr(api, "rm_realtime_arm_state_callback_ptr", None)
        config_type = getattr(api, "rm_realtime_push_config_t", None)
        if callback_type is None or config_type is None:
            raise RuntimeError(
                "Installed RealMan SDK does not support realtime UDP state push."
            )
        if not callable(
            getattr(self._raw_arm, "rm_realtime_arm_state_call_back", None)
        ) or not callable(getattr(self._raw_arm, "rm_set_realtime_push", None)):
            raise RuntimeError(
                "RealMan SDK arm does not expose realtime state-push methods."
            )

        self._state_push_received.clear()
        with self._state_callback_condition:
            self._accept_state_callbacks = True
        self._state_push_callback = callback_type(
            self._state_push_callback_entry
        )
        self._raw_arm.rm_realtime_arm_state_call_back(
            self._state_push_callback
        )
        config = config_type(
            cycle=self.cfg.REALMAN_STATE_PUSH_CYCLE_MS,
            enable=True,
            port=self.cfg.REALMAN_STATE_PUSH_PORT,
            force_coordinate=self.cfg.REALMAN_FORCE_COORDINATE,
            ip=self.cfg.PC_IP,
        )
        result = self._raw_arm.rm_set_realtime_push(config)
        if result != 0:
            self._stop_accepting_state_callbacks()
            raise RuntimeError(
                "rm_set_realtime_push failed with RealMan error code "
                f"{result}."
            )
        self._state_push_active = True

        if not self._state_push_received.wait(self.cfg.REALMAN_STATE_PUSH_TIMEOUT):
            self._stop_realtime_state_push()
            raise TimeoutError(
                "No RealMan realtime state packet arrived. Check Config.PC_IP, "
                f"UDP port {self.cfg.REALMAN_STATE_PUSH_PORT}, and the firewall."
            )
        utils.logger.info(
            "RealMan realtime joint/TCP/force push enabled at "
            f"{self.cfg.REALMAN_STATE_PUSH_CYCLE_MS} ms."
        )

    def _stop_realtime_state_push(self) -> None:
        disable_error: Exception | None = None
        if self._state_push_active:
            try:
                config_type = getattr(
                    self._realman_api,
                    "rm_realtime_push_config_t",
                )
                config = config_type(
                    cycle=self.cfg.REALMAN_STATE_PUSH_CYCLE_MS,
                    enable=False,
                    port=self.cfg.REALMAN_STATE_PUSH_PORT,
                    force_coordinate=self.cfg.REALMAN_FORCE_COORDINATE,
                    ip=self.cfg.PC_IP,
                )
                result = self._raw_arm.rm_set_realtime_push(config)
                if result != 0:
                    disable_error = RuntimeError(
                        "Disabling RealMan realtime state push failed with "
                        f"error code {result}."
                    )
            except Exception as exc:
                disable_error = exc
            finally:
                self._state_push_active = False

        self._stop_accepting_state_callbacks()
        if disable_error is not None:
            utils.logger.warning(
                f"Could not disable RealMan realtime state push: "
                f"{disable_error}"
            )

    def start(self, stop_event: threading.Event) -> list[threading.Thread]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Cannot start a closed RealMan teleoperation.")
            if self._quarantined:
                raise RuntimeError(
                    "Cannot restart a quarantined RealMan teleoperation."
                )
            if self._started:
                raise RuntimeError(
                    "RealMan teleoperation has already been started."
                )
            self._started = True

            if self.cfg.REALMAN_REALTIME_STATE_PUSH:
                self._start_realtime_state_push()
            else:
                utils.logger.warning(
                    "REALMAN_REALTIME_STATE_PUSH is disabled; synchronous "
                    "state reads are serialized on the CAN-FD owner thread "
                    "and may consume the timing budget."
                )

            command_thread = threading.Thread(
                target=self.canfd.run,
                args=(stop_event,),
                name="realman-canfd",
                daemon=True,
            )
            started: list[threading.Thread] = []
            self._sdk_worker_threads = started
            try:
                command_thread.start()
                started.append(command_thread)
            except Exception:
                stop_event.set()
                for thread in started:
                    thread.join(timeout=1.0)
                if not self.live_sdk_workers():
                    self._stop_realtime_state_push()
                raise
            return list(started)

    @property
    def sensor_stale_after_s(self) -> float:
        if self.cfg.REALMAN_REALTIME_STATE_PUSH:
            push_period_s = self.cfg.REALMAN_STATE_PUSH_CYCLE_MS / 1000.0
            return max(0.1, 4.0 * push_period_s)
        return max(0.2, 3.0 / self.cfg.REALMAN_SENSOR_RATE)

    def live_sdk_workers(self) -> list[str]:
        return [
            thread.name
            for thread in self._sdk_worker_threads
            if thread.is_alive()
        ]

    def sdk_worker_threads(self) -> tuple[threading.Thread, ...]:
        return tuple(self._sdk_worker_threads)

    def state_snapshot(self) -> RealManStateSnapshot:
        with self._sensor_lock:
            joints = self._joints.copy()
            tcp_pose = self._tcp_pose.copy()
            wrench = self._wrench.copy()
            state_timestamp_ns = self._state_timestamp_ns
            force_timestamp_ns = self._force_timestamp_ns
            state_error = self._state_error
            force_error = self._force_error
        with self._control_lock:
            input_stale = self._input_stale
        return RealManStateSnapshot(
            joints=joints,
            tcp_pose=tcp_pose,
            wrench=wrench,
            state_timestamp_ns=state_timestamp_ns,
            force_timestamp_ns=force_timestamp_ns,
            state_error=state_error,
            force_error=force_error,
            input_stale=input_stale,
        )

    def recording_snapshot(
        self,
    ) -> tuple[RealManStateSnapshot, np.ndarray, np.ndarray, int]:
        """Copy cached measured state and the latest teleoperation targets."""

        state = self.state_snapshot()
        with self._control_lock:
            if self.control_mode == "joint":
                action_joints = self._last_joint_target.copy()
            else:
                # Direct Cartesian CAN-FD has no host-side joint target. The
                # measured joints provide a replayable joint demonstration
                # when DATA_TYPE is qpos/both; DATA_TYPE=tcp records the actual
                # Cartesian command below.
                action_joints = state.joints.copy()
            action_tcp = self._last_tcp_target.copy()
            action_timestamp_ns = self._action_timestamp_ns
        return state, action_joints, action_tcp, action_timestamp_ns

    def reset_active(self) -> bool:
        with self._control_lock:
            return self._reset_in_progress

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            live_workers = self.live_sdk_workers()
            if live_workers:
                raise RuntimeError(
                    "Refusing to close the RealMan SDK handle while worker(s) "
                    f"are still alive: {', '.join(live_workers)}."
                )
            self._stop_realtime_state_push()
            self.backend.cleanup()
            self._closed = True
            _UNCLOSED_REALMAN_TELEOPS[:] = [
                teleop
                for teleop in _UNCLOSED_REALMAN_TELEOPS
                if teleop is not self
            ]

    def quarantine_without_sdk_cleanup(self) -> None:
        """Keep callback memory alive when an in-flight SDK call prevents close."""

        callback_error: Exception | None = None
        with self._lifecycle_lock:
            if self._closed:
                return
            self._quarantined = True
            try:
                # This only changes Python-side callback state. It deliberately
                # makes no SDK call while another thread may be stuck inside one.
                self._stop_accepting_state_callbacks()
            except Exception as exc:
                callback_error = exc
            finally:
                if not any(
                    retained is self
                    for retained in _UNCLOSED_REALMAN_TELEOPS
                ):
                    _UNCLOSED_REALMAN_TELEOPS.append(self)
        if callback_error is not None:
            raise callback_error


def _create_camera_manager(cfg: Config):
    if cfg.VIDEO_TRANSPORT.lower() == "webrtc":
        from WebRTC_udp import WebRTCUDPManager

        return WebRTCUDPManager()

    if cfg.VIDEO_TRANSPORT.lower() == "udp":
        from camera_udp import CameraUDPManager

        return CameraUDPManager()

    raise ValueError(f"Unsupported VIDEO_TRANSPORT: {cfg.VIDEO_TRANSPORT}")


def _visualizer_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    step_y = max(1, image.shape[0] // 240)
    step_x = max(1, image.shape[1] // 320)
    return image[::step_y, ::step_x, :3].copy()


def _visualizer_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    previews: dict[str, np.ndarray] = {}
    for name, image in sorted(images.items()):
        preview = _visualizer_image(image)
        if preview is not None:
            previews[name] = preview
    return previews


class RealManEpisodeRecorder:
    """Record cached RealMan state and camera frames without touching CAN-FD."""

    def __init__(
        self,
        cfg: Config,
        teleop: RealManTeleop,
        camera_manager,
        *,
        dataset: DatasetRecorder | None = None,
        background_errors: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.teleop = teleop
        self.camera_manager = camera_manager
        self.background_errors = background_errors
        self.dataset = (
            dataset
            if dataset is not None
            else DatasetRecorder(
                camera_manager.camera_num,
                robot_dof=teleop.dof,
                robot_type=getattr(
                    teleop.backend,
                    "dataset_robot_type",
                    teleop.backend.name,
                ),
                force_collect=cfg.FORCE_COLLECT,
                torque_collect=cfg.TORQUE_COLLECT,
                gripper=False,
                config=cfg,
            )
        )
        self.pause_event = threading.Event()
        self._dataset_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._last_state_quaternion: np.ndarray | None = None
        self._last_action_quaternion: np.ndarray | None = None

    @staticmethod
    def _pose_vector(
        tcp_pose: np.ndarray,
        previous_quaternion: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(tcp_pose, dtype=float)
        quaternion = utils.quat_cal(
            pose[:3, :3],
            previous_quaternion,
        )
        vector = np.concatenate(
            [quaternion, pose[:3, 3]]
        ).astype(np.float32)
        return vector, quaternion

    @staticmethod
    def _delta_tcp_action(
        measured_tcp: np.ndarray,
        target_tcp: np.ndarray,
    ) -> np.ndarray:
        delta_translation = target_tcp[:3, 3] - measured_tcp[:3, 3]
        delta_rotation = target_tcp[:3, :3] @ measured_tcp[:3, :3].T
        delta_rotvec, _ = cv2.Rodrigues(delta_rotation)
        return np.concatenate(
            [delta_translation, delta_rotvec.reshape(3)]
        ).astype(np.float32)

    def _state_and_action(
        self,
        robot: RealManStateSnapshot,
        action_joints: np.ndarray,
        action_tcp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_tcp, self._last_state_quaternion = self._pose_vector(
            robot.tcp_pose,
            self._last_state_quaternion,
        )
        if self.cfg.DATA_TYPE == "tcp":
            action, self._last_action_quaternion = self._pose_vector(
                action_tcp,
                self._last_action_quaternion,
            )
            return state_tcp, action, state_tcp
        if self.cfg.DATA_TYPE == "delta_tcp":
            action = self._delta_tcp_action(robot.tcp_pose, action_tcp)
            return robot.joints.astype(np.float32), action, state_tcp
        return (
            robot.joints.astype(np.float32),
            np.asarray(action_joints, dtype=np.float32),
            state_tcp,
        )

    def collect_once(self) -> bool:
        """Collect one configured-rate frame when an episode is active."""

        if self.pause_event.is_set():
            return False
        with self.camera_manager._lock:
            if not self.camera_manager.data_collecting_state:
                return False
            images = {
                name: np.asarray(image).copy()
                for name, image in self.camera_manager.camera_images.items()
            }
            image_timestamps = dict(
                getattr(
                    self.camera_manager,
                    "camera_image_timestamps_ns",
                    {},
                )
            )
            depth_images = (
                {
                    name: np.asarray(depth).copy()
                    for name, depth in self.camera_manager.depth_images.items()
                }
                if self.cfg.DEPTH_INFO_ENABLE
                else None
            )
            vr_input_timestamp_ns = int(
                getattr(self.camera_manager, "vr_input_timestamp_ns", 0)
            )

        if not (
            self.camera_manager.is_movement_exist()
            or self.teleop.reset_active()
        ):
            return False

        collect_timestamp_ns = time.monotonic_ns()
        robot, action_joints, action_tcp, action_timestamp_ns = (
            self.teleop.recording_snapshot()
        )
        state, action, tcp_vector = self._state_and_action(
            robot,
            action_joints,
            action_tcp,
        )
        extra = {
            "collect_timestamp_ns": np.array(
                collect_timestamp_ns,
                dtype=np.int64,
            ),
            "robot_state_timestamp_ns": np.array(
                robot.state_timestamp_ns,
                dtype=np.int64,
            ),
            "robot_action_timestamp_ns": np.array(
                action_timestamp_ns,
                dtype=np.int64,
            ),
            "vr_input_timestamp_ns": np.array(
                vr_input_timestamp_ns,
                dtype=np.int64,
            ),
            "tactile_timestamp_ns": np.array(0, dtype=np.int64),
            "camera_timestamps_ns": image_timestamps,
            "tcp_pose": tcp_vector,
        }
        wrench = (
            robot.wrench
            if self.cfg.FORCE_COLLECT or self.cfg.TORQUE_COLLECT
            else None
        )
        with self._dataset_lock:
            if self.pause_event.is_set():
                return False
            self.dataset.data_collection(
                state,
                action,
                images,
                None,
                wrench,
                depth_images,
                extra,
            )
        return True

    def process_pending_once(self) -> bool:
        """Handle one export or rollback request from VR/the visualizer."""

        with self.camera_manager._lock:
            rollback = bool(self.camera_manager.data_rollback_state)
            export = bool(self.camera_manager.data_export_state)
        if not rollback and not export:
            return False

        self.pause_event.set()
        try:
            with self._dataset_lock:
                if rollback:
                    removed = self.dataset.rollback_last_episode()
                    if removed:
                        utils.logger.info(
                            "Rollback complete. Next episode: "
                            f"{self.dataset.recorded_episodes}"
                        )
                    else:
                        utils.logger.warning(
                            "Rollback requested, but no episode was removed."
                        )
                elif self.dataset.collect_step:
                    self.dataset.data_export(self.camera_manager)
                    self.dataset._reset_data_dict()
                    utils.logger.info(
                        "Episode "
                        f"{self.dataset.recorded_episodes - 1} exported successfully"
                    )
                else:
                    utils.logger.warning("No data to export.")
            return True
        finally:
            with self.camera_manager._lock:
                if rollback:
                    self.camera_manager.data_rollback_state = False
                if export:
                    self.camera_manager.data_export_state = False
            self.pause_event.clear()

    def handle_visualizer_commands(self, visualizer_handle) -> None:
        if visualizer_handle is None:
            return
        for command in visualizer_handle.drain_commands():
            name = command.get("command")
            if name == "rollback_last_episode":
                with self.camera_manager._lock:
                    self.camera_manager.data_collecting_state = False
                    self.camera_manager.data_export_state = False
                    self.camera_manager.data_rollback_state = True
                utils.logger.info("Rollback requested from visualizer.")

    def recording_status(self) -> dict[str, object]:
        with self.camera_manager._lock:
            collecting = bool(self.camera_manager.data_collecting_state)
        with self._dataset_lock:
            return self.dataset.recording_status(collecting=collecting)

    def _report_background_failure(
        self,
        label: str,
        exc: Exception,
        stop_event: threading.Event,
    ) -> None:
        message = f"RealMan {label} failed: {exc}"
        utils.logger.exception(message)
        if self.background_errors is not None:
            self.background_errors.append(message)
        stop_event.set()

    def _collect_loop(self, stop_event: threading.Event) -> None:
        period_s = 1.0 / self.cfg.COLLECT_RATE
        next_tick = time.monotonic()
        try:
            while not stop_event.is_set():
                self.collect_once()
                next_tick += period_s
                remaining = next_tick - time.monotonic()
                if remaining > 0.0:
                    stop_event.wait(remaining)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:
            self._report_background_failure("dataset collection", exc, stop_event)

    def _export_loop(self, stop_event: threading.Event) -> None:
        try:
            while not stop_event.is_set():
                self.process_pending_once()
                stop_event.wait(0.05)
        except Exception as exc:
            self._report_background_failure("dataset export", exc, stop_event)
        finally:
            self.close()

    def start(self, stop_event: threading.Event) -> list[threading.Thread]:
        threads = [
            threading.Thread(
                target=self._collect_loop,
                args=(stop_event,),
                name="realman-dataset-collector",
                daemon=True,
            ),
            threading.Thread(
                target=self._export_loop,
                args=(stop_event,),
                name="realman-dataset-exporter",
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        return threads

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self.pause_event.set()
            try:
                with self.camera_manager._lock:
                    rollback = bool(
                        self.camera_manager.data_rollback_state
                    )
                    self.camera_manager.data_collecting_state = False
                    self.camera_manager.data_export_state = False
                    self.camera_manager.data_rollback_state = False
                with self._dataset_lock:
                    if rollback:
                        self.dataset.rollback_last_episode()
                    elif self.dataset.collect_step:
                        utils.logger.info(
                            "Saving the active RealMan episode before exit."
                        )
                        self.dataset.data_export(self.camera_manager)
                        self.dataset._reset_data_dict()
                    self.dataset.close()
                self._closed = True
            finally:
                self.pause_event.clear()


def visualizer_publish_loop(
    visualizer_handle,
    teleop: RealManTeleop,
    camera_manager,
    viz_cfg: VisualizerConfig,
    stop_event: threading.Event,
    background_errors: list[str] | None = None,
    recorder: RealManEpisodeRecorder | None = None,
) -> None:
    period_s = 1.0 / viz_cfg.HZ
    force_frame = ("sensor", "work", "tool")[
        teleop.cfg.REALMAN_FORCE_COORDINATE
    ]
    try:
        while not stop_event.is_set():
            now_ns = time.monotonic_ns()
            robot = teleop.state_snapshot()
            canfd = teleop.canfd.snapshot()
            with camera_manager._lock:
                images = dict(camera_manager.camera_data)
                images.update(camera_manager.camera_images)
                capture_timestamps = dict(
                    getattr(
                        camera_manager,
                        "camera_data_timestamps_ns",
                        {},
                    )
                )
                display_timestamps = dict(
                    getattr(
                        camera_manager,
                        "camera_image_timestamps_ns",
                        {},
                    )
                )

            image_timestamps = capture_timestamps
            image_timestamps.update(display_timestamps)
            camera_timestamp_support = hasattr(
                camera_manager,
                "camera_data_timestamps_ns",
            )
            camera_stale_after_s = max(
                0.25,
                5.0
                / max(
                    1.0,
                    float(getattr(camera_manager, "realsense_fps", 30.0)),
                ),
            )
            camera_ages_s: list[float] = []
            camera_errors: list[str] = []
            if camera_manager.camera_num <= 0:
                camera_errors.append("no camera connected")
            for camera_index in range(camera_manager.camera_num):
                key = f"camera_{camera_index}"
                if key not in images:
                    camera_errors.append(f"{key} frame is unavailable")
                    continue
                timestamp_ns = int(image_timestamps.get(key, 0))
                if timestamp_ns:
                    age_s = max(0.0, (now_ns - timestamp_ns) / 1e9)
                    camera_ages_s.append(age_s)
                    if age_s > camera_stale_after_s:
                        camera_errors.append(
                            f"{key} frame is stale ({age_s * 1000.0:.0f} ms)"
                        )
                elif camera_timestamp_support:
                    camera_errors.append(f"{key} timestamp is unavailable")

            state_age_s = max(
                0.0,
                (now_ns - robot.state_timestamp_ns) / 1e9,
            )
            force_age_s = max(
                0.0,
                (now_ns - robot.force_timestamp_ns) / 1e9,
            )
            achieved = (
                "measuring"
                if canfd.achieved_hz is None
                else f"{canfd.achieved_hz:.1f} Hz"
            )
            status = (
                f"{teleop.control_mode} CAN-FD {achieved} "
                f"(target {canfd.target_hz:g} Hz)\n"
                f"max gap {canfd.max_gap_ms:.2f} ms, "
                f"control step {canfd.max_sdk_call_ms:.2f} ms, "
                f">10 ms violations "
                f"{canfd.high_follow_gap_violations + canfd.sdk_call_overruns}\n"
                f"state age {state_age_s * 1000.0:.0f} ms, "
                f"force age {force_age_s * 1000.0:.0f} ms"
            )
            if camera_ages_s:
                status += (
                    f", camera age "
                    f"{max(camera_ages_s) * 1000.0:.0f} ms"
                )
            errors = [
                message
                for message in (
                    canfd.error,
                    robot.state_error,
                    robot.force_error,
                    teleop.canfd.heartbeat_error(),
                )
                if message
            ]
            errors.extend(camera_errors)
            if not canfd.running:
                errors.append("CAN-FD sender is not running")
            if robot.input_stale:
                errors.append("VR controller input is stale")
            if state_age_s > teleop.sensor_stale_after_s:
                errors.append(
                    f"robot state is stale ({state_age_s * 1000.0:.0f} ms)"
                )
            if force_age_s > teleop.sensor_stale_after_s:
                errors.append(
                    f"robot force is stale ({force_age_s * 1000.0:.0f} ms)"
                )
            visualizer_handle.publish(
                {
                    "timestamp": min(
                        robot.state_timestamp_ns,
                        robot.force_timestamp_ns,
                    )
                    / 1e9,
                    "wrench": robot.wrench,
                    "joints": robot.joints,
                    "tcp_translation": robot.tcp_pose[:3, 3],
                    "images": _visualizer_images(images),
                    "camera_count": camera_manager.camera_num,
                    "source_label": (
                        f"RealMan robot force ({force_frame} frame)"
                    ),
                    "status_extra": status,
                    **(
                        {"dataset": recorder.recording_status()}
                        if recorder is not None
                        else {}
                    ),
                    "connected": not errors,
                    "error": " | ".join(errors),
                }
            )
            stop_event.wait(period_s)
    except Exception as exc:
        message = f"RealMan visualizer publisher failed: {exc}"
        utils.logger.exception(message)
        if background_errors is not None:
            background_errors.append(message)
        stop_event.set()


def main() -> None:
    cfg = Config()
    viz_cfg = VisualizerConfig()
    stop_event = threading.Event()
    camera_manager = None
    teleop: RealManTeleop | None = None
    recorder: RealManEpisodeRecorder | None = None
    tcp_state_sender: QuestTcpStateSender | None = None
    visualizer_handle = None
    worker_threads: list[threading.Thread] = []
    background_errors: list[str] = []
    interrupted = False
    fatal_error = ""
    shutdown_failed = False

    utils.logger.info(
        f"Starting RealMan-only teleoperation: {cfg.TELEOP_COMMAND_MODE} control, "
        f"{cfg.REALMAN_CTRL_RATE} Hz CAN-FD, {cfg.VIDEO_TRANSPORT} video"
    )

    try:
        camera_manager = _create_camera_manager(cfg)
        initial_data = camera_manager.test_connection()
        teleop = RealManTeleop(initial_data, cfg=cfg)
        # Dataset discovery/import can be expensive. Finish it before the
        # high-follow sender starts so it cannot steal time from CAN-FD's
        # strict 10 ms packet deadline.
        recorder = RealManEpisodeRecorder(
            cfg,
            teleop,
            camera_manager,
            background_errors=background_errors,
        )
        camera_manager.start_comms_threads()
        worker_threads.extend(teleop.start(stop_event))
        if cfg.FORCE_ENABLE:
            tcp_state_sender = QuestTcpStateSender(
                teleop,
                quest_ip=cfg.VR_IP,
                port=cfg.FORCE_PORT,
                send_rate_hz=cfg.FORCE_SEND_RATE,
            )
            tcp_state_thread = threading.Thread(
                target=tcp_state_sender.run,
                args=(stop_event, background_errors),
                name="quest-tcp-state-sender",
                daemon=True,
            )
            tcp_state_thread.start()
            worker_threads.append(tcp_state_thread)
        worker_threads.extend(recorder.start(stop_event))

        if viz_cfg.ENABLED:
            from visualizer import start_visualizer

            visualizer_handle = start_visualizer(
                hz=viz_cfg.HZ,
                window_s=viz_cfg.WINDOW_S,
                title="RealMan Teleop Visualizer",
                force_panel_range=viz_cfg.FORCE_PANEL_RANGE,
                camera_num=camera_manager.camera_num,
                show_rollback_button=True,
                show_record_button=False,
            )
            visualizer_thread = threading.Thread(
                target=visualizer_publish_loop,
                args=(
                    visualizer_handle,
                    teleop,
                    camera_manager,
                    viz_cfg,
                    stop_event,
                    background_errors,
                    recorder,
                ),
                name="realman-visualizer-publisher",
                daemon=True,
            )
            visualizer_thread.start()
            worker_threads.append(visualizer_thread)

        # Starting worker threads and, especially, spawning the visualizer can
        # briefly preempt the CAN-FD sender. Keep the fail-fast watchdog in its
        # startup probation state until every background service is running,
        # then require a fresh clean window under the final system load.
        timing = teleop.canfd.wait_until_healthy(stop_event)
        verified_hz = timing.achieved_hz
        if verified_hz is None:
            raise RuntimeError(
                "CAN-FD timing verification returned no measured rate."
            )
        utils.logger.info(
            f"RealMan CAN-FD timing verified at {verified_hz:.1f} Hz; "
            "VR motion input enabled."
        )

        previous_input_timestamp_ns = 0
        previous_control_time = time.monotonic()
        while not stop_event.is_set():
            recorder.handle_visualizer_commands(visualizer_handle)

            heartbeat_error = teleop.canfd.heartbeat_error()
            if heartbeat_error:
                fatal_error = heartbeat_error
                teleop.canfd.report_external_failure(
                    heartbeat_error,
                    stop_event,
                )
                break

            robot_snapshot = teleop.state_snapshot()
            state_age_s = (
                time.monotonic_ns() - robot_snapshot.state_timestamp_ns
            ) / 1e9
            if state_age_s > teleop.sensor_stale_after_s:
                teleop.mark_input_stale(
                    f"robot state is stale ({state_age_s * 1000.0:.0f} ms)"
                )

            with camera_manager._lock:
                controller_data = camera_manager.data
                hand_data = (
                    dict(camera_manager.hand_data)
                    if camera_manager.hand_data
                    else None
                )
                input_timestamp_ns = camera_manager.vr_input_timestamp_ns

            new_input = input_timestamp_ns > previous_input_timestamp_ns
            if cfg.TRACKING_MODE == "hand" and hand_data is not None and new_input:
                now = time.monotonic()
                dt = min(max(now - previous_control_time, 1e-4), 0.05)
                previous_control_time = now
                teleop.process_hand(hand_data, dt)
                previous_input_timestamp_ns = input_timestamp_ns
            elif cfg.TRACKING_MODE == "controller" and controller_data is not None and new_input:
                now = time.monotonic()
                dt = min(max(now - previous_control_time, 1e-4), 0.05)
                previous_control_time = now
                teleop.process_controller(controller_data, dt)
                previous_input_timestamp_ns = input_timestamp_ns
            elif new_input:
                expected = "hand" if cfg.TRACKING_MODE == "hand" else "controller"
                teleop.mark_input_stale(
                    f"Received non-{expected} VR input in {expected} tracking mode"
                )
                previous_control_time = time.monotonic()
                previous_input_timestamp_ns = input_timestamp_ns

            if input_timestamp_ns:
                input_age_s = (time.monotonic_ns() - input_timestamp_ns) / 1e9
                if input_age_s > cfg.REALMAN_VR_TIMEOUT:
                    teleop.mark_input_stale(
                        f"VR input timed out ({input_age_s * 1000.0:.0f} ms)"
                    )

            canfd_error = teleop.canfd.snapshot().error
            if canfd_error:
                fatal_error = canfd_error
                stop_event.set()
                break

            stop_event.wait(0.005)

        fatal_error = (
            teleop.canfd.snapshot().error
            or (background_errors[0] if background_errors else "")
            or fatal_error
        )

    except KeyboardInterrupt:
        interrupted = True
        utils.logger.info("Stopping RealMan teleoperation...")
    finally:
        stop_event.set()

        if camera_manager is not None:
            try:
                camera_manager.close()
            except Exception as exc:
                utils.logger.warning(f"Error closing camera manager: {exc}")

        all_threads = list(worker_threads)
        if teleop is not None:
            for thread in teleop.sdk_worker_threads():
                if thread not in all_threads:
                    all_threads.append(thread)
        for thread in all_threads:
            thread.join(timeout=2.0)

        if visualizer_handle is not None:
            try:
                visualizer_handle.close()
            except Exception as exc:
                utils.logger.warning(
                    f"Error closing RealMan visualizer: {exc}"
                )

        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                fatal_error = fatal_error or f"Error closing dataset: {exc}"
                utils.logger.error(f"Error closing RealMan dataset: {exc}")

        if teleop is not None:
            live_workers = teleop.live_sdk_workers()
            if live_workers:
                shutdown_error = (
                    "RealMan worker(s) did not stop; the SDK handle was left "
                    f"open to avoid invalidating an in-flight call: "
                    f"{', '.join(live_workers)}."
                )
                fatal_error = fatal_error or shutdown_error
                shutdown_failed = True
                utils.logger.critical(shutdown_error)
                try:
                    teleop.quarantine_without_sdk_cleanup()
                except Exception as exc:
                    utils.logger.critical(
                        "Could not drain RealMan state callbacks while "
                        f"quarantining the live SDK handle: {exc}"
                    )
            else:
                try:
                    teleop.close()
                except Exception as exc:
                    shutdown_error = (
                        f"Error closing RealMan connection: {exc}"
                    )
                    fatal_error = fatal_error or shutdown_error
                    shutdown_failed = True
                    utils.logger.error(shutdown_error)
                    try:
                        teleop.quarantine_without_sdk_cleanup()
                    except Exception as callback_exc:
                        utils.logger.critical(
                            "Could not drain RealMan state callbacks after "
                            f"SDK cleanup failed: {callback_exc}"
                        )

        utils.logger.info("RealMan teleoperation shutdown complete.")

    if fatal_error and (not interrupted or shutdown_failed):
        raise RuntimeError(f"RealMan teleoperation stopped: {fatal_error}")


if __name__ == "__main__":
    main()
