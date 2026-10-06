"""High-follow RealMan CAN-FD command clock and setpoint policy."""

from __future__ import annotations

from typing import Any, Callable
import threading
import time

import cv2
import numpy as np
from airo_spatial_algebra.se3 import SE3Container

import doffy_teleop.utils as utils


class CanfdSetpointMixin:
    """Thread-safe target, interpolation and pending-IK policy."""

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
