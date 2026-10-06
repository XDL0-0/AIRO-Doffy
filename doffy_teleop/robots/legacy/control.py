"""Robot command, reset, and controller-mode helpers for classic teleop."""

from __future__ import annotations

import time

import cv2
import numpy as np

import doffy_teleop.utils as utils
from airo_spatial_algebra.se3 import SE3Container

class LegacyControlMixin:
    def _update_gripper(self, gripper_state: int, dt: float, gripper_width: float) -> None:
        if not self.gripper_enabled:
            return
        if gripper_state:
            self.gripper_solution_width += self.gripper_speed * dt * gripper_state
            self.gripper_solution_width = np.clip(self.gripper_solution_width, 0.0, self.gripper_max)
            if gripper_state < 0:
                self.gripper_solution_width = max(self.gripper_solution_width, gripper_width)
            else:
                self.gripper_solution_width = min(self.gripper_solution_width, gripper_width)

            if not self.gripper_stop_control_sign or gripper_state != self._gripper_direction:
                self._gripper_direction = gripper_state
                destination = 0.0 if gripper_state < 0 else self.gripper_max
                self.gripper._set_target_width(destination)
                self.gripper_stop_control_sign = True
        elif self.gripper_stop_control_sign:
            self.gripper_solution_width = np.clip(gripper_width, 0.0, self.gripper_max)
            self.gripper_stop_control_sign = False
            self._gripper_direction = 0
            self.gripper._set_target_width(self.gripper_solution_width)

    def reset_robot_and_gripper(self) -> None:
        if self.gripper_enabled:
            self.gripper.move(self.gripper_max)
        self.backend.reset(self.initial_joint)
        if self.gripper_enabled:
            self.gripper_solution_width = self.gripper.get_current_width()
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(self._read_tcp_pose())
        self.filtered_joint_target = np.array(self.initial_joint)
        self.last_sent_target = np.array(self.initial_joint)
        self.previous_joint_action = np.array(self.initial_joint)
        self.previous_tcp_action = self._read_tcp_pose()
        if self.ruckig_enable:
            self._reset_ruckig_state(self.initial_joint)
        gripper_norm = self._normalize_gripper_width(self.gripper_solution_width)
        self._publish_snapshot(
            self._read_joints(),
            self._read_tcp_pose(),
            gripper_norm,
            self.initial_joint,
            self.previous_tcp_action,
            gripper_norm,
        )
        self._zero_force_baseline_after_reset()
        utils.logger.info("---- Robot reset complete ----")

    # ── Command helpers ───────────────────────────────────────────────────

    def _safe_joint_target(
        self,
        tcp_target: np.ndarray,
        current_joints: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        joint_target = self.backend.solve_tcp_ik(tcp_target, current_joints)
        if joint_target is None:
            utils.logger.warning("No valid IK solution, keeping previous pose!")
            return None, False
        safe = self.backend.is_joint_target_safe(
            joint_target,
            self.previous_joint_action,
            tcp_target[:3, 3],
            self.joint_threshold,
        )
        return joint_target, safe

    def _send_target(self, tcp_target: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        current_joints = self._read_joints()
        joint_target, safe = self._safe_joint_target(tcp_target, current_joints)
        if not safe or joint_target is None:
            joint_target = self.previous_joint_action.copy()
            tcp_target = self.previous_tcp_action.copy()

        final_target = np.asarray(joint_target, dtype=float)
        if self.control_mode == "joint":
            if self.torque_mode:
                pass
            elif self.ruckig_enable:
                final_target = self._ruckig_step(final_target)
            else:
                beta = 0.7
                self.filtered_joint_target = (
                    beta * final_target + (1 - beta) * self.filtered_joint_target
                )
                raw_delta = self.filtered_joint_target - self.last_sent_target
                max_step = 0.02
                max_change = np.max(np.abs(raw_delta))
                if max_change > max_step:
                    raw_delta *= max_step / max_change
                final_target = self.last_sent_target + raw_delta
            self.backend.command_joint_configuration(final_target, dt)
        else:
            result = self.backend.command_tcp_pose(tcp_target, dt)
            if result.joint_configuration is not None:
                final_target = result.joint_configuration

        final_target = self.backend.clip_joint_configuration(final_target)
        self.filtered_joint_target = final_target.copy()
        self.last_sent_target = final_target.copy()
        return final_target, tcp_target

    def _target_from_delta(self, reference: SE3Container, current: SE3Container, dt: float) -> np.ndarray:
        se3_mat = current.homogeneous_matrix
        translation_diff = se3_mat[:3, 3] - reference.translation
        rotation_diff = reference.rotation_matrix.T @ se3_mat[:3, :3]
        translation_diff = self.pos_filter.update(translation_diff, dt)

        rvec, _ = cv2.Rodrigues(rotation_diff)
        rvec = self._remap_controller_rotation_vector(rvec.flatten())
        rvec = self.rot_filter.update(rvec, dt)
        rotation_diff, _ = cv2.Rodrigues(rvec)

        target_translation = self.SE3_tcp_pose_in_base_frame_std.translation + translation_diff
        if self.freeze_rotation:
            target_rotation = self.SE3_tcp_pose_in_base_frame_std.rotation_matrix
        else:
            target_rotation = rotation_diff @ self.SE3_tcp_pose_in_base_frame_std.rotation_matrix
        return SE3Container.from_rotation_matrix_and_translation(
            target_rotation, target_translation
        ).homogeneous_matrix

    # ── Standby / controller teleop ───────────────────────────────────────

    def _standby_mode(self, controller_data: list[dict]) -> bool:
        if not controller_data[1]["GripTrigger"]:
            self._set_reference(controller_data)
            utils.logger.debug("Standby mode active")
            return True
        return False

    def _teleop_mode(
        self,
        controller_data: list[dict],
        dt: float,
        state_joints: np.ndarray,
        state_tcp: np.ndarray,
        state_gripper_norm: float,
    ) -> None:
        if self.reset_sign:
            self.reset_sign = False
            self._set_reference(controller_data)

        tcp_target = self._target_from_delta(
            self.SE3_controller_std,
            self._extract_se3(controller_data),
            dt,
        )

        if self.backend.is_ur and self.dof == 6 and not self.freeze_rotation:
            joint_target, safe = self._safe_joint_target(tcp_target, state_joints)
            if safe and joint_target is not None:
                joystick_x = controller_data[1]["Joystick"][0]
                proposed_bias = self.last_joint_bias + ((joystick_x > 0.8) - (joystick_x < -0.8)) * 0.01
                target_j5 = joint_target[5] + proposed_bias
                target_j5_clipped = np.clip(target_j5, utils.UR3E_JOINT_LIMITS[0], utils.UR3E_JOINT_LIMITS[1])
                self.last_joint_bias = target_j5_clipped - joint_target[5]
                joint_target[5] = target_j5_clipped
                tcp_target = self.backend.ik_solver.forward_kinematics(*joint_target) @ self.tcp_transform
        elif self.backend.name == "realman":
            joystick_x = float(controller_data[1]["Joystick"][0])
            self.last_joint_bias += (
                (joystick_x > 0.8) - (joystick_x < -0.8)
            ) * 0.01
            rotation_increment, _ = cv2.Rodrigues(
                np.asarray([0.0, 0.0, self.last_joint_bias], dtype=float)
            )
            tcp_target = np.asarray(tcp_target, dtype=float).copy()
            tcp_target[:3, :3] = tcp_target[:3, :3] @ rotation_increment

        final_joints, final_tcp = self._send_target(tcp_target, dt)
        action_gripper_norm = self._normalize_gripper_width(self.gripper_solution_width)
        self._publish_snapshot(
            state_joints,
            state_tcp,
            state_gripper_norm,
            final_joints,
            final_tcp,
            action_gripper_norm,
        )
        utils.logger.debug("Teleop step executed successfully.")
