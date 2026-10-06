"""OpenXR hand tracking helpers for classic robot teleop."""

from __future__ import annotations

import numpy as np

import doffy_teleop.utils as utils
from airo_spatial_algebra.se3 import SE3Container

class LegacyHandMixin:
    # ── Hand-tracking helpers ─────────────────────────────────────────────

    _THUMB_TIP_IDX = 5
    _INDEX_TIP_IDX = 10
    _RING_TIP_IDX = 20
    _PINKY_TIP_IDX = 25

    def _extract_hand_se3(self, rh: dict) -> SE3Container:
        wrist_pose = rh.get("wrist_pose")
        if wrist_pose is not None:
            p = wrist_pose["position"]
            r = wrist_pose["rotation"]
            return self._vr_pose_to_robot_se3(p, r)

        p = rh["bones"][0]
        return self._vr_pose_to_robot_se3(
            p,
            np.array([0.0, 0.0, 0.0, 1.0]),
        )

    def _hand_set_reference(self, hand_se3: SE3Container) -> None:
        self._hand_ref_se3 = hand_se3
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(self._read_tcp_pose())
        self.pos_filter.reset()
        self.rot_filter.reset()
        self._seed_hand_joint_filter(self._read_joints())
        self._hand_initialized = True
        utils.logger.info("Hand reference set")

    def _reset_hand_reference_state(self) -> None:
        self._hand_initialized = False
        self._hand_last_palm = None
        self._hand_ref_se3 = None
        self.hand_joint_filter.reset()

    def _filter_hand_joint_target(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if self.hand_joint_filter.initialized:
            base = self.hand_joint_filter.value
            target = base + np.arctan2(np.sin(target - base), np.cos(target - base))
        filtered = self.hand_joint_filter.update(target, dt)
        return self.backend.clip_joint_configuration(filtered)

    def _seed_hand_joint_filter(self, joints: np.ndarray) -> None:
        self.hand_joint_filter.value = np.asarray(joints, dtype=float).copy()
        self.hand_joint_filter.initialized = True

    def _hand_teleop_step(
        self,
        hand_data: dict | None,
        dt: float,
        state_joints: np.ndarray,
        state_tcp: np.ndarray,
        state_gripper_norm: float,
    ) -> None:
        if hand_data is None or "R" not in hand_data:
            return

        rh = hand_data["R"]
        bones = rh.get("bones")
        if bones is None or len(bones) != 26:
            return
        if np.linalg.norm(bones[self._INDEX_TIP_IDX]) < 1e-4:
            return

        hand_se3 = self._extract_hand_se3(rh)
        if self._hand_last_palm is not None:
            jump = np.linalg.norm(hand_se3.translation - self._hand_last_palm)
            if jump > self._hand_palm_jump:
                utils.logger.warning(
                    f"Hand wrist jump {jump:.3f}m > {self._hand_palm_jump}m, ignoring frame"
                )
                self._hand_initialized = False
                self._hand_last_palm = None
                return
        self._hand_last_palm = hand_se3.translation.copy()

        if self.hand is not None:
            try:
                hand_joints = self.hand.follow_openxr_hand(bones)
                utils.logger.debug(
                    "BrainCo hand target: %s",
                    np.round(hand_joints, 3).tolist(),
                )
            except ValueError as exc:
                # A partially tracked OpenXR frame must not disable otherwise
                # healthy hand hardware.
                utils.logger.debug("Ignoring invalid OpenXR hand frame: %s", exc)
            except Exception as exc:
                self._disable_hand_tool(f"BrainCo hand command failed ({exc})")

        if not self._hand_initialized:
            self._hand_set_reference(hand_se3)
            self.reset_sign = False
            return

        tcp_target = self._target_from_delta(self._hand_ref_se3, hand_se3, dt)
        final_joints, final_tcp = self._send_target(tcp_target, dt)

        if self.gripper_enabled:
            thumb_tip = np.array(bones[self._THUMB_TIP_IDX])
            index_tip = np.array(bones[self._INDEX_TIP_IDX])
            finger_dist = np.linalg.norm(thumb_tip - index_tip)
            if finger_dist > self._hand_gripper_open:
                gripper_state = 1
            elif finger_dist < self._hand_gripper_close:
                gripper_state = -1
            else:
                gripper_state = 0

            gripper_width_m = self.capture_gripper_width()
            self._update_gripper(gripper_state, dt, gripper_width_m)
            utils.logger.debug(
                f"Hand teleop: finger_dist={finger_dist:.3f}m  "
                f"gripper_state={gripper_state}"
            )
        action_gripper_norm = self._normalize_gripper_width(self.gripper_solution_width)
        self._publish_snapshot(
            state_joints,
            state_tcp,
            state_gripper_norm,
            final_joints,
            final_tcp,
            action_gripper_norm,
        )
