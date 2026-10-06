"""Pure VR/hand reference mapping and target preparation."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from airo_spatial_algebra.se3 import SE3Container

import doffy_teleop.utils as utils


class InputMappingMixin:
    """Reference frames, filtering and low-level controller transforms."""

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
        if self._wrm_akm is not None:
            self._wrm_akm.reset_seed(measured_joints)
            self._wrm_akm.set_tcp_z_reference()

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
        if self._wrm_akm is not None:
            self._wrm_akm.reset_seed(measured_joints)
            self._wrm_akm.set_tcp_z_reference()
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
        if self._wrm_akm is not None:
            self._wrm_akm.reset_seed(self.initial_joint)
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
