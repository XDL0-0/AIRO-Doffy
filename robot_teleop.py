"""Robot teleoperation via VR controllers or hand tracking.

The teleop layer now speaks to a robot backend instead of directly depending on
UR RTDE or UR analytic IK. UR position control, UR torque control, and RealMan
position control can share the same VR mapping and dataset publication path.
"""

from __future__ import annotations

import gc
import threading
import time

import cv2
import numpy as np

import utils
from config import Config
from force_filter import WrenchFilter
from visualizer_config import VisualizerConfig
from data_schema import (
    action_representation,
    build_data_schema,
    should_store_extra_tcp_pose,
    state_representation,
)
from robot_backend import FastRobotiq2F85 as _FastRobotiq2F85
from robot_backend import make_robot as _make_robot
from robot_backend import make_robot_backend
from airo_spatial_algebra.se3 import SE3Container


class FastRobotiq2F85(_FastRobotiq2F85):
    """Backward-compatible export for replay scripts."""


def make_robot(ur_ip: str, robot_type: str, torque_mode: bool, initial_joint=None, ruckig_params=None):
    """Backward-compatible export for replay scripts."""
    return _make_robot(ur_ip, robot_type, torque_mode, initial_joint, ruckig_params)


class RobotTeleop:
    """Main teleop class. The historical name is kept for import compatibility."""

    def __init__(self, initial_data: list[dict]):
        cfg = Config()
        viz_cfg = VisualizerConfig()
        self.cfg = cfg
        self.backend = make_robot_backend(cfg)
        self.ur = self.backend.robot
        self.ik = self.backend.ik_solver
        self.gripper = self.backend.gripper
        self.hand = self.backend.hand

        self.dof = self.backend.dof
        self.initial_joint = self.backend.initial_joint_configuration(cfg.INITIAL_JOINT)
        self.robot_type = cfg.ROBOT_TYPE
        self.control_rate = cfg.UR_CTRL_RATE
        self.control_mode = cfg.TELEOP_COMMAND_MODE
        self.tcp_tool = self.backend.tcp_tool
        self.gripper_enabled = self.tcp_tool == "Gripper"
        self.gripper_speed = cfg.GRIPPER_SPEED
        self.gripper_max = cfg.GRIPPER_MAX
        self.data_type = cfg.DATA_TYPE
        self.freeze_rotation = cfg.FREEZE_ROTATION
        self.schema = build_data_schema(
            self.data_type,
            self.dof,
            gripper=self.gripper_enabled,
        )
        self.state_representation = state_representation(self.data_type)
        self.action_representation = action_representation(self.data_type)
        self.collect_tcp_extra = should_store_extra_tcp_pose(self.data_type)
        self.tcp_transform = cfg.TCP_TRANSFORM
        self.vr_to_robot_axes = np.asarray(cfg.VR_TO_ROBOT_AXES, dtype=float)
        self.vr_to_robot_handedness = float(np.linalg.det(self.vr_to_robot_axes))
        self.vr_angular_axes = self.vr_to_robot_handedness * self.vr_to_robot_axes
        self.vr_rotation_axis_signs = np.asarray(
            cfg.VR_ROTATION_AXIS_SIGNS,
            dtype=float,
        )
        self.joint_threshold = self._joint_threshold_for_dof(cfg.MOVE_THRESHOLD)
        self.last_quat: np.ndarray | None = None
        self.last_action_quat: np.ndarray | None = None
        self.last_tcp_quat: np.ndarray | None = None
        self.reset_sign = False
        self.torque_mode = cfg.TORQUE_MODE
        self.gripper_stop_control_sign = False
        self._gripper_direction = 0

        self.tracking_mode = cfg.TRACKING_MODE
        self._controller_reset_trigger_threshold = cfg.CONTROLLER_RESET_TRIGGER_THRESHOLD
        self._controller_reset_held = False
        self._hand_joystick_motion: str | None = None
        self._hand_joystick_threshold = cfg.BRAINCO_HAND_JOYSTICK_THRESHOLD
        self._hand_ref_se3: SE3Container | None = None
        self._hand_last_palm: np.ndarray | None = None
        self._hand_initialized = False
        self._hand_palm_jump = cfg.HAND_PALM_JUMP_THRESHOLD
        self._hand_gripper_open = cfg.HAND_GRIPPER_OPEN_DIST
        self._hand_gripper_close = cfg.HAND_GRIPPER_CLOSE_DIST
        self._hand_mode_toggle_dist = cfg.HAND_MODE_TOGGLE_DIST
        self._hand_reset_dist = cfg.HAND_RESET_DIST
        self._last_toggle_time = 0.0
        self._last_reset_time = 0.0

        self._state_lock = threading.Lock()

        utils.logger.info(
            f"Teleop initialized - robot:{self.backend.dataset_robot_type}, "
            f"DoF:{self.dof}, mode:{self.control_mode}, data:{self.data_type}"
        )
        utils.logger.info(f"Freeze rotation: {self.freeze_rotation}")
        utils.logger.info(f"Tracking mode: {self.tracking_mode}")
        utils.logger.info(f"TCP tool: {self.tcp_tool}")
        utils.logger.info(f"Gripper: {self.gripper_enabled}")
        utils.logger.info(f"Moving to initial joint: {self.initial_joint}")

        self.backend.reset(self.initial_joint)

        if self.gripper_enabled:
            self.gripper.open()
            self.gripper_solution_width = self.gripper.get_current_width()
        else:
            self.gripper_solution_width = self.gripper_max
        gripper_init = self._normalize_gripper_width(self.gripper_solution_width)
        self.last_joint_bias = 0.0
        self.filtered_joint_target = np.array(self.initial_joint, dtype=float)
        self.last_sent_target = np.array(self.initial_joint, dtype=float)
        self.previous_joint_action = np.array(self.initial_joint, dtype=float)
        self.previous_tcp_action = self._read_tcp_pose()

        self.pos_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=cfg.CARTESIAN_POS_FILTER_CUTOFF_HZ, dim=3
        )
        self.rot_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=cfg.CARTESIAN_ROT_FILTER_CUTOFF_HZ, dim=3
        )
        self.hand_joint_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=cfg.HAND_JOINT_FILTER_CUTOFF_HZ, dim=self.dof
        )

        self.ruckig_enable = (
            cfg.RUCKIG_ENABLE
            and not self.torque_mode
            and self.control_mode == "joint"
        )
        if self.ruckig_enable:
            self._init_ruckig(cfg)

        self.wrench_mode = (
            cfg.FORCE_COLLECT or cfg.TORQUE_COLLECT or viz_cfg.ENABLED
        ) and self.backend.supports_force
        self.force_mode = cfg.FORCE_COLLECT and self.backend.supports_force
        self.torque_collect = cfg.TORQUE_COLLECT and self.backend.supports_force
        self.tcp_wrench = np.zeros(6)
        self.wrench_filter = WrenchFilter(
            moving_average_window=cfg.FORCE_MOVING_AVERAGE_WINDOW,
            low_pass_alpha=cfg.FORCE_LOW_PASS_ALPHA,
        )
        self.gravity_comp = cfg.GRAVITY_COMP and self.wrench_mode
        if self.gravity_comp:
            self.gravity_compensator = utils.GravityCompensator(
                mass=cfg.TOOL_MASS,
                com=cfg.TOOL_COM,
                filter_alpha=cfg.GRAVITY_COMP_FILTER_ALPHA,
            )
            self._calib_samples_needed = cfg.GRAVITY_CALIB_SAMPLES

        self._set_reference(initial_data)
        now_ns = time.monotonic_ns()
        self.state_timestamp_ns = now_ns
        self.action_timestamp_ns = now_ns
        self.state = self._build_state_vector(self._read_joints(), self._read_tcp_pose(), gripper_init)
        self.previous_solution = self._build_action_vector(
            self.previous_joint_action,
            self.previous_tcp_action,
            self.previous_tcp_action,
            gripper_init,
        )

        if self.gravity_comp:
            time.sleep(1.0)
            self.calibrate_force_sensor()

    # ── Basic robot reads ─────────────────────────────────────────────────

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

    # ── Gripper and reset ─────────────────────────────────────────────────

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


URTeleop = RobotTeleop
