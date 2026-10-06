"""Safety-gated target resolution and VR/hand input handlers."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from airo_spatial_algebra.se3 import SE3Container

import doffy_teleop.utils as utils


class TargetPolicyMixin:
    """Resolve IK targets and update control state from parsed input."""

    def _resolve_joint_target(
        self,
        tcp_target: np.ndarray,
        dt: float,
    ) -> np.ndarray | None:
        """Run IK on the same fixed-rate thread that owns CAN-FD SDK calls."""

        snapshot = self.state_snapshot()
        if self._wrm_akm is not None:
            tcp_target = self._wrm_akm.couple_tcp_target_z(
                tcp_target,
                self.cfg.WRM_TCP_Z_DROP_M,
            )
            robot_tcp_target = self.backend.to_robot_tcp_pose(tcp_target)
            joint_target = self._wrm_akm.solve(robot_tcp_target)
            solver_status = self._wrm_akm.last_status
        elif self._remote_ik_solver is not None:
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
            if self._wrm_akm is not None:
                self._wrm_akm.accept_solution(filtered)
            return filtered

    def update_wrm_tracking(self, sample) -> bool:
        """Update only the abstract elbow objective from one Unity sample.

        Invalid, stale, or low-confidence samples freeze the last elbow target;
        controller/TCP safety remains governed by the existing input path.
        """

        if self._wrm_akm is None:
            return False
        with self._control_lock:
            return bool(self._wrm_akm.update_tracking(sample))

    def wrm_visualizer_state(self) -> dict | None:
        """Return WRM alpha/configuration state when AKM is enabled."""

        if self._wrm_akm is None:
            return None
        state = self._wrm_akm.visualizer_state()
        state["tcp_z_offset_m"] = self._wrm_akm.tcp_z_offset(
            self.cfg.WRM_TCP_Z_DROP_M
        )
        state["tcp_z_drop_limit_m"] = float(self.cfg.WRM_TCP_Z_DROP_M)
        return state

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
