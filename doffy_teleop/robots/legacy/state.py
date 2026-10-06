"""State, sensor, dataset, and reference helpers for classic robot teleop."""

from __future__ import annotations

import gc
import time

import cv2
import numpy as np

import doffy_teleop.utils as utils
from airo_spatial_algebra.se3 import SE3Container

class LegacyStateMixin:
    def _joint_threshold_for_dof(self, threshold: np.ndarray | float) -> np.ndarray:
        dof = self.dof
        values = np.asarray(threshold, dtype=float)
        if values.ndim == 0:
            return np.full(dof, float(values))
        if values.shape == (dof,):
            return values
        if values.size == 1:
            return np.full(dof, float(values.item()))
        return np.resize(values, dof)

    def _read_joints(self) -> np.ndarray:
        return np.asarray(self.backend.get_joint_configuration(), dtype=float)

    def _read_tcp_pose(self) -> np.ndarray:
        return np.asarray(self.backend.get_tcp_pose(), dtype=float)

    def _read_raw_force(self) -> np.ndarray:
        force = self.backend.get_tcp_force()
        return np.zeros(6) if force is None else np.asarray(force, dtype=float)

    def close(self) -> None:
        self.backend.cleanup()
        for attr in ("_otg_out", "_otg_inp", "_otg"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()

    def _disable_hand_tool(self, reason: str) -> None:
        """Permanently downgrade the active hand tool to None for this run."""
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

        joystick_y = float(right.get("Joystick", (0.0, 0.0))[1])
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

    def _get_tool_rotation(self) -> np.ndarray:
        return self._read_tcp_pose()[:3, :3]

    @staticmethod
    def _normalize_gripper_width(gripper_width_m: float) -> float:
        return float(np.clip(gripper_width_m / 0.085, 0.0, 1.0))

    def _gripper_array_from_width(self, gripper_width_m: float) -> np.ndarray:
        return np.array([self._normalize_gripper_width(gripper_width_m)])

    # ── Dataset vector helpers ────────────────────────────────────────────

    def _tcp_vector(self, tcp_pose: np.ndarray, gripper_norm: float, action: bool = False) -> np.ndarray:
        se3 = SE3Container.from_homogeneous_matrix(tcp_pose)
        if action:
            self.last_action_quat = utils.quat_cal(se3.rotation_matrix, self.last_action_quat)
            quat = self.last_action_quat
        else:
            self.last_quat = utils.quat_cal(se3.rotation_matrix, self.last_quat)
            quat = self.last_quat
        values = [quat, se3.translation]
        if self.gripper_enabled:
            values.append(np.array([gripper_norm]))
        return np.concatenate(values).astype(np.float32)

    def _delta_tcp_vector(
        self,
        reference_tcp: np.ndarray,
        target_tcp: np.ndarray,
        gripper_norm: float,
    ) -> np.ndarray:
        reference = SE3Container.from_homogeneous_matrix(reference_tcp)
        target = SE3Container.from_homogeneous_matrix(target_tcp)
        delta_translation = target.translation - reference.translation
        delta_rotation = target.rotation_matrix @ reference.rotation_matrix.T
        delta_rotvec, _ = cv2.Rodrigues(delta_rotation)
        values = [delta_translation, delta_rotvec.reshape(3)]
        if self.gripper_enabled:
            values.append(np.array([gripper_norm]))
        return np.concatenate(values).astype(np.float32)

    def _build_state_vector(self, joints: np.ndarray, tcp_pose: np.ndarray, gripper_norm: float) -> np.ndarray:
        if self.state_representation == "tcp":
            return self._tcp_vector(tcp_pose, gripper_norm, action=False)
        state = np.asarray(joints, dtype=np.float32)
        if self.gripper_enabled:
            state = np.concatenate([state, [gripper_norm]]).astype(np.float32)
        return state

    def _build_action_vector(
        self,
        joint_target: np.ndarray | None,
        tcp_target: np.ndarray,
        reference_tcp: np.ndarray,
        gripper_norm: float,
    ) -> np.ndarray:
        if self.action_representation == "tcp":
            return self._tcp_vector(tcp_target, gripper_norm, action=True)
        if self.action_representation == "delta_tcp":
            return self._delta_tcp_vector(reference_tcp, tcp_target, gripper_norm)
        if joint_target is None:
            joint_target = self.previous_joint_action
        action = np.asarray(joint_target, dtype=np.float32)
        if self.gripper_enabled:
            action = np.concatenate([action, [gripper_norm]]).astype(np.float32)
        return action

    def _publish_snapshot(
        self,
        state_joints: np.ndarray,
        state_tcp: np.ndarray,
        state_gripper_norm: float,
        action_joints: np.ndarray | None,
        action_tcp: np.ndarray,
        action_gripper_norm: float,
    ) -> None:
        with self._state_lock:
            self.state_timestamp_ns = time.monotonic_ns()
            self.action_timestamp_ns = self.state_timestamp_ns
            self.state = self._build_state_vector(state_joints, state_tcp, state_gripper_norm)
            self.previous_solution = self._build_action_vector(
                action_joints,
                action_tcp,
                state_tcp,
                action_gripper_norm,
            )
            if action_joints is not None:
                self.previous_joint_action = np.asarray(action_joints, dtype=float)
            self.previous_tcp_action = np.asarray(action_tcp, dtype=float)

    def get_state_snapshot(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        with self._state_lock:
            state = self.state.copy()
            action = self.previous_solution.copy()
            wrench = self.tcp_wrench.copy()
            state_timestamp_ns = self.state_timestamp_ns
            action_timestamp_ns = self.action_timestamp_ns

        extra = {
            "robot_state_timestamp_ns": np.array(state_timestamp_ns, dtype=np.int64),
            "robot_action_timestamp_ns": np.array(action_timestamp_ns, dtype=np.int64),
        }
        if self.collect_tcp_extra:
            extra["tcp_pose"] = self.capture_tcp_pose()
        return state, action, wrench, extra

    # ── Ruckig OTG ────────────────────────────────────────────────────────

    def _init_ruckig(self, cfg) -> None:
        from ruckig import Ruckig, InputParameter, OutputParameter

        self._otg = Ruckig(self.dof, 1.0 / self.control_rate)
        self._otg_inp = InputParameter(self.dof)
        self._otg_out = OutputParameter(self.dof)
        self._otg_inp.max_velocity = self._ruckig_limits(cfg.RUCKIG_MAX_VEL, "velocity")
        self._otg_inp.max_acceleration = self._ruckig_limits(cfg.RUCKIG_MAX_ACC, "acceleration")
        self._otg_inp.max_jerk = self._ruckig_limits(cfg.RUCKIG_MAX_JERK, "jerk")
        self._reset_ruckig_state(self.initial_joint)
        utils.logger.info("Ruckig OTG enabled for joint servo mode")

    def _ruckig_limits(self, value: float | np.ndarray, name: str) -> list[float]:
        limits = np.asarray(value, dtype=float)
        if limits.ndim == 0:
            return [float(limits)] * self.dof
        if limits.shape != (self.dof,):
            if limits.shape == (6,) and self.dof != 6:
                utils.logger.warning(
                    f"Ruckig {name} limits are shape (6,) but robot has {self.dof} DoF; resizing."
                )
            limits = np.resize(limits, self.dof)
        return limits.tolist()

    def _reset_ruckig_state(self, position: np.ndarray) -> None:
        self._otg_inp.current_position = list(position)
        self._otg_inp.current_velocity = [0.0] * self.dof
        self._otg_inp.current_acceleration = [0.0] * self.dof

    def _set_ruckig_target(self, target: np.ndarray) -> None:
        self._otg_inp.target_position = target.tolist()
        self._otg_inp.target_velocity = [0.0] * self.dof
        self._otg_inp.target_acceleration = [0.0] * self.dof

    def _apply_ruckig_result(self) -> np.ndarray:
        self._otg_out.pass_to_input(self._otg_inp)
        return np.array(self._otg_out.new_position)

    def _ruckig_step(self, target: np.ndarray) -> np.ndarray:
        from ruckig import Result

        target = np.asarray(target, dtype=float)
        self._set_ruckig_target(target)
        try:
            result = self._otg.update(self._otg_inp, self._otg_out)
        except Exception as exc:
            utils.logger.warning(
                f"Ruckig OTG exception ({type(exc).__name__}), resetting state and retrying once"
            )
            measured_joints = self._read_joints()
            self._reset_ruckig_state(measured_joints)
            self._set_ruckig_target(target)
            try:
                result = self._otg.update(self._otg_inp, self._otg_out)
            except Exception as retry_exc:
                utils.logger.warning(
                    f"Ruckig OTG retry failed ({type(retry_exc).__name__}), holding measured pose"
                )
                return measured_joints

        if result == Result.Working or result == Result.Finished:
            return self._apply_ruckig_result()
        utils.logger.warning(f"Ruckig OTG error ({result}), resetting state")
        self._reset_ruckig_state(self._read_joints())
        return np.array(self._otg_inp.current_position)

    # ── Reference frame ───────────────────────────────────────────────────

    def _set_reference(self, data: list[dict]) -> None:
        self.last_joint_bias = 0.0
        if hasattr(self, "pos_filter"):
            self.pos_filter.reset()
            self.rot_filter.reset()
        self.SE3_controller_std = self._extract_se3(data)
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(self._read_tcp_pose())

    def _vr_pose_to_robot_se3(
        self,
        position: tuple[float, float, float] | np.ndarray,
        quaternion: tuple[float, float, float, float] | np.ndarray,
    ) -> SE3Container:
        """Convert a Unity/Quest pose into the configured robot base axes."""
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

    def _extract_se3(self, controller_data: list[dict]) -> SE3Container:
        r = controller_data[1]["Rotation"]
        p = controller_data[1]["Position"]
        return self._vr_pose_to_robot_se3(p, r)

    def _remap_controller_rotation_vector(
        self,
        rotation_vector_robot: np.ndarray,
    ) -> np.ndarray:
        """Map Unity-local pitch/yaw/roll onto EEF pitch/yaw/roll."""

        rotation_vector_vr = (
            self.vr_angular_axes.T
            @ np.asarray(rotation_vector_robot, dtype=float)
        )
        rotation_vector_vr *= self.vr_rotation_axis_signs
        pitch, yaw, roll = rotation_vector_vr
        return np.array([roll, pitch, yaw])

    # ── Sensor capture ────────────────────────────────────────────────────

    def capture_joint_pose(self) -> np.ndarray:
        return self._read_joints()

    def capture_eef_pose(self, last_quat: np.ndarray | None) -> np.ndarray:
        tcp = self._read_tcp_pose()
        se3 = SE3Container.from_homogeneous_matrix(tcp)
        self.last_quat = utils.quat_cal(se3.rotation_matrix, last_quat)
        return np.concatenate([self.last_quat, se3.translation])

    def capture_tcp_pose(self) -> np.ndarray:
        tcp = self._read_tcp_pose()
        se3 = SE3Container.from_homogeneous_matrix(tcp)
        self.last_tcp_quat = utils.quat_cal(se3.rotation_matrix, self.last_tcp_quat)
        return np.concatenate([self.last_tcp_quat, se3.translation]).astype(np.float32)

    def capture_tcp_force(self) -> np.ndarray:
        raw = self._read_raw_force()
        if self.gravity_comp:
            return self.gravity_compensator.compensate(raw, self._get_tool_rotation())
        return raw

    def capture_tcp_wrench(self) -> np.ndarray:
        return self.wrench_filter.process(self.capture_tcp_force())

    def refresh_wrench_snapshot(self) -> np.ndarray:
        if not self.wrench_mode:
            return self.tcp_wrench.copy()
        wrench = self.capture_tcp_wrench()
        with self._state_lock:
            self.tcp_wrench = wrench.copy()
        return wrench

    def capture_gripper_width(self) -> float:
        if not self.gripper_enabled:
            return float(self.gripper_solution_width)
        return float(self.gripper.get_current_width())

    def capture_gripper(self) -> np.ndarray:
        if not self.gripper_enabled:
            return np.empty((0,), dtype=float)
        return self._gripper_array_from_width(self.capture_gripper_width())

    def calibrate_force_sensor(self) -> None:
        if not self.gravity_comp:
            return
        n = self._calib_samples_needed
        utils.logger.info(f"Calibrating force sensor ({n} samples)...")
        for _ in range(n):
            self.gravity_compensator.add_calibration_sample(
                self._read_raw_force(), self._get_tool_rotation()
            )
            time.sleep(0.005)
        self.gravity_compensator.finish_calibration()
        self.wrench_filter.reset()

    def _zero_force_baseline_after_reset(self) -> None:
        if not self.gravity_comp:
            return
        try:
            self.calibrate_force_sensor()
        except Exception as exc:
            utils.logger.warning(f"Force sensor zero calibration failed after reset: {exc}")
