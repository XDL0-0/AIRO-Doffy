"""Top-level classic teleop step orchestration."""

from __future__ import annotations

import time

import numpy as np

import doffy_teleop.utils as utils

class LegacyStepMixin:
    # ── Main step ─────────────────────────────────────────────────────────

    def step(
        self,
        controller_data: list[dict] | None,
        dt: float = 0.01,
        hand_data: dict | None = None,
    ) -> None:
        ctrl_active = False
        if controller_data is not None:
            rd = controller_data[1]
            ctrl_active = (
                rd["GripTrigger"]
                or rd["IndexTrigger"] > 0.5
                or rd["Button_AX"]
                or rd["Button_BY"]
                or abs(rd["Joystick"][0]) > 0.3
                or abs(rd["Joystick"][1]) > 0.3
            )

        if not ctrl_active and hand_data is not None and "R" in hand_data:
            bones = hand_data["R"].get("bones")
            if bones is not None and len(bones) == 26 and np.linalg.norm(bones[self._INDEX_TIP_IDX]) >= 1e-4:
                thumb_tip = np.array(bones[self._THUMB_TIP_IDX])
                pinky_tip = np.array(bones[self._PINKY_TIP_IDX])
                ring_tip = np.array(bones[self._RING_TIP_IDX])
                current_time = time.time()
                if np.linalg.norm(thumb_tip - pinky_tip) < self._hand_mode_toggle_dist:
                    if (current_time - self._last_toggle_time) > 1.0:
                        if self.tracking_mode == "hand":
                            utils.logger.warning("Gesture: Switched to CONTROLLER mode")
                            self.tracking_mode = "controller"
                        else:
                            utils.logger.warning("Gesture: Switched to HAND mode")
                            self.tracking_mode = "hand"
                        self._reset_hand_reference_state()
                        self._last_toggle_time = current_time
                elif np.linalg.norm(thumb_tip - ring_tip) < self._hand_reset_dist:
                    if (current_time - self._last_reset_time) > 2.0:
                        utils.logger.warning("Gesture: Resetting robot to initial position")
                        self.reset_sign = True
                        self.reset_robot_and_gripper()
                        self._reset_hand_reference_state()
                        self._last_reset_time = current_time
                        return

        state_joints = self.capture_joint_pose()
        state_tcp = self._read_tcp_pose()
        gripper_width_m = self.capture_gripper_width()
        state_gripper_norm = self._normalize_gripper_width(gripper_width_m)

        if self.wrench_mode:
            with self._state_lock:
                self.tcp_wrench = self.capture_tcp_wrench()

        if self.tracking_mode == "hand":
            if ctrl_active:
                utils.logger.debug("Hand mode: controller active, ignoring hand data")
                return
            self._hand_teleop_step(hand_data, dt, state_joints, state_tcp, state_gripper_norm)
            return

        if controller_data is None:
            return

        if self.gripper_enabled:
            x = -controller_data[1]["Joystick"][1]
            gripper_state = (x > 0.7) - (x < -0.7)
            self._update_gripper(gripper_state, dt, gripper_width_m)

        reset_pressed = (
            bool(controller_data[1]["Joystick_Press"])
            and controller_data[1]["IndexTrigger"] >= self._controller_reset_trigger_threshold
        )
        if reset_pressed:
            if not self._controller_reset_held:
                self.reset_sign = True
                self.reset_robot_and_gripper()
            self._controller_reset_held = True
            return
        self._controller_reset_held = False

        self._process_brainco_joystick(controller_data[1])

        if self._standby_mode(controller_data):
            action_gripper_norm = self._normalize_gripper_width(self.gripper_solution_width)
            self._publish_snapshot(
                state_joints,
                state_tcp,
                state_gripper_norm,
                self.previous_joint_action,
                self.previous_tcp_action,
                action_gripper_norm,
            )
            return

        self._teleop_mode(
            controller_data,
            dt,
            state_joints,
            state_tcp,
            state_gripper_norm,
        )
