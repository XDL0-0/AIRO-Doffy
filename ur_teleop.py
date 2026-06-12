"""UR robot teleoperation via VR controllers.

Provides:
    - make_robot()         — factory for UR arm + IK solver
    - FastRobotiq2F85      — low-latency gripper controller
    - URTeleop             — main teleop class (VR → IK → joint servo)
"""

from __future__ import annotations

import socket
import time
import threading
import gc
import cv2
import numpy as np

import utils
from config import Config
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85, rescale_range
from airo_spatial_algebra.se3 import SE3Container


# ── Robot factory ─────────────────────────────────────────────────────────

def make_robot(ur_ip: str, robot_type: str, torque_mode: bool, initial_joint=None, ruckig_params=None):
    kwargs = {}
    if not torque_mode:
        from airo_robots.manipulators.hardware.ur_rtde import URrtde
    else:
        from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque as URrtde
        kwargs = dict(initial_joint_configuration=initial_joint)
    if torque_mode and ruckig_params is not None:
        kwargs["ruckig_params"] = ruckig_params

    if robot_type == "ur3e":
        from ur_analytic_ik import ur3e as ik
        return URrtde(ur_ip, URrtde.UR3E_CONFIG, **kwargs), ik
    elif robot_type == "ur5e":
        from ur_analytic_ik import ur5e as ik
        return URrtde(ur_ip, URrtde.UR5E_CONFIG, **kwargs), ik
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")


# ── Fast Gripper ──────────────────────────────────────────────────────────

class FastRobotiq2F85(Robotiq2F85):
    """Robotiq 2F-85 with persistent TCP connection and non-blocking control.

    The stock Robotiq2F85 opens a new TCP socket for every command and blocks
    after each SET POS. This subclass keeps one connection open and skips the
    blocking wait — suitable for the teleop hot-loop.
    """

    def __init__(self, host_ip: str, port: int = 63352, fingers_max_stroke=None):
        super().__init__(host_ip, port, fingers_max_stroke)
        self._persistent_sock = self._make_sock(host_ip, port)
        self._fast_communicate("SET SPE 255")
        verify = self._fast_communicate("GET SPE")
        utils.logger.info(f"FastRobotiq2F85 ready (SPE={verify})")

    def _make_sock(self, host: str, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(0.02)
        s.connect((host, port))
        return s

    def _fast_communicate(self, command: str) -> str:
        try:
            self._persistent_sock.sendall((command.strip() + "\n").encode())
            data = self._persistent_sock.recv(2 ** 10)
            return data.decode()[:-1]
        except (socket.timeout, BrokenPipeError, ConnectionResetError, OSError) as e:
            utils.logger.warning(f"Gripper socket error: {e}, reconnecting...")
            try:
                self._persistent_sock.close()
            except Exception:
                pass
            self._persistent_sock = self._make_sock(self.host_ip, self.port)
            self._persistent_sock.sendall((command.strip() + "\n").encode())
            data = self._persistent_sock.recv(2 ** 10)
            return data.decode()[:-1]

    def _set_target_width(self, target_width_in_meters: float) -> None:
        target_width_in_meters = np.clip(
            target_width_in_meters,
            self._gripper_specs.min_width,
            self._gripper_specs.max_width,
        )
        register_val = round(
            rescale_range(
                target_width_in_meters,
                self._gripper_specs.min_width,
                self._gripper_specs.max_width,
                230, 0,
            )
        )
        self._fast_communicate(f"SET  POS {register_val}")

    def get_current_width(self) -> float:
        register_value = int(self._fast_communicate("GET POS").split(" ")[1])
        return rescale_range(
            register_value, 0, 230,
            self._gripper_specs.max_width, self._gripper_specs.min_width,
        )

    def __del__(self):
        try:
            self._persistent_sock.close()
        except Exception:
            pass


# ── Teleop ────────────────────────────────────────────────────────────────

class URTeleop:
    def __init__(self, initial_data: list[dict]):
        cfg = Config()
        self.initial_joint = cfg.INITIAL_JOINT
        self.robot_type = cfg.ROBOT_TYPE
        ruckig_params = None
        # Torque mode uses the URrtdeTorque servo-compatible target path; its
        # worker smooths the 60 Hz teleop stream into a 500 Hz torque reference.
        self.ur, self.ik = make_robot(
            cfg.UR_IP, cfg.ROBOT_TYPE, cfg.TORQUE_MODE, self.initial_joint, ruckig_params
        )
        self.gripper = FastRobotiq2F85(cfg.UR_IP)
        self.gripper.open()
        self.gripper_solution_width = self.gripper.get_current_width()

        self.last_joint_bias = 0.0
        self.control_rate = cfg.UR_CTRL_RATE
        self.gripper_speed = cfg.GRIPPER_SPEED
        self.gripper_max = cfg.GRIPPER_MAX
        self.data_type = cfg.DATA_TYPE
        self.collect_tcp_extra = self.data_type == "both"
        self.save_eef = cfg.SAVE_EEF and not self.collect_tcp_extra
        self.tcp_transform = cfg.TCP_TRANSFORM
        self.joint_threshold = cfg.MOVE_THRESHOLD
        self.fine_mode = False
        self.last_quat: np.ndarray | None = None
        self.last_tcp_quat: np.ndarray | None = None
        self.reset_sign = False
        self.torque_mode = cfg.TORQUE_MODE
        self.gripper_stop_control_sign = False
        self._gripper_direction = 0

        # ── Hand-tracking mode state ──────────────────────────────────────
        self.tracking_mode = cfg.TRACKING_MODE          # "controller" or "hand"
        self._controller_reset_trigger_threshold = cfg.CONTROLLER_RESET_TRIGGER_THRESHOLD
        self._hand_ref_se3: SE3Container | None = None  # reference wrist SE3
        self._hand_last_palm: np.ndarray | None = None  # previous frame wrist pos (jump det)
        self._hand_initialized = False
        self._hand_palm_jump = cfg.HAND_PALM_JUMP_THRESHOLD
        self._hand_gripper_open = cfg.HAND_GRIPPER_OPEN_DIST
        self._hand_gripper_close = cfg.HAND_GRIPPER_CLOSE_DIST
        self._hand_mode_toggle_dist = cfg.HAND_MODE_TOGGLE_DIST
        self._hand_reset_dist = cfg.HAND_RESET_DIST
        self._last_toggle_time = 0.0
        self._last_reset_time = 0.0

        self._state_lock = threading.Lock()

        utils.logger.info(f"Teleop initialized — UR:{cfg.UR_IP}, VR:{cfg.VR_IP}")
        utils.logger.info(f"Tracking mode: {self.tracking_mode}")
        utils.logger.info(f"Moving to initial joint: {self.initial_joint}")

        if self.torque_mode:
            self.ur.target_pos = self.initial_joint
        else:
            self.ur.move_to_joint_configuration(self.initial_joint, 0.5).wait()

        self.previous_solution = np.append(self.initial_joint, 0)
        self.filtered_joint_target = np.array(self.initial_joint)
        self.last_sent_target = np.array(self.initial_joint)
        self.pos_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=cfg.CARTESIAN_POS_FILTER_CUTOFF_HZ, dim=3
        )
        self.rot_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=cfg.CARTESIAN_ROT_FILTER_CUTOFF_HZ, dim=3
        )
        self.hand_joint_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=cfg.HAND_JOINT_FILTER_CUTOFF_HZ, dim=6
        )
        now_ns = time.monotonic_ns()
        self.state_timestamp_ns = now_ns
        self.action_timestamp_ns = now_ns
        self.ruckig_enable = cfg.RUCKIG_ENABLE and not self.torque_mode
        if self.ruckig_enable:
            self._init_ruckig(cfg)
        self.force_mode = cfg.FORCE_COLLECT and cfg.TORQUE_MODE
        self.tcp_force = np.zeros(6)
        self.gravity_comp = cfg.GRAVITY_COMP and self.force_mode

        if self.gravity_comp:
            self.gravity_compensator = utils.GravityCompensator(
                mass=cfg.TOOL_MASS,
                com=cfg.TOOL_COM,
                filter_alpha=cfg.FORCE_FILTER_ALPHA,
            )
            self._calib_samples_needed = cfg.GRAVITY_CALIB_SAMPLES

        self._set_reference(initial_data)
        joints = self._read_joints()
        gripper_init = self._normalize_gripper_width(self.gripper_solution_width)
        self.previous_solution = np.append(joints, gripper_init)

        if self.save_eef:
            self.last_quat = utils.quat_cal(
                self.SE3_tcp_pose_in_base_frame_std.rotation_matrix, self.last_quat
            )
            self.ur_eef_capture = np.concatenate(
                [self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation]
            )
            self.previous_solution_eef = np.append(
                self.ur_eef_capture, gripper_init,
            )
            self.state_eef = np.concatenate(
                [self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation,
                 [gripper_init]]
            )
        else:
            self.state = np.concatenate(
                [joints, [gripper_init]]
            )

        if self.gravity_comp:
            time.sleep(1.0)
            self.calibrate_force_sensor()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _read_joints(self) -> np.ndarray:
        if self.torque_mode:
            return np.array(self.ur.get_cached_joint_configuration())
        return np.array(self.ur.get_joint_configuration())

    def _read_tcp_pose(self) -> np.ndarray:
        joints = self._read_joints()
        flange_pose = self.ik.forward_kinematics(*joints)
        return flange_pose @ self.tcp_transform

    def _read_raw_force(self) -> np.ndarray:
        if self.torque_mode:
            return np.array(self.ur.get_cached_tcp_force())
        return np.array(self.ur.get_tcp_force())

    def close(self) -> None:
        """Release resources owned by the teleop wrapper before interpreter exit."""
        if hasattr(self, "gripper"):
            try:
                self.gripper._persistent_sock.close()
            except Exception as e:
                utils.logger.warning(f"Error closing gripper socket: {e}")

        for attr in ("_otg_out", "_otg_inp", "_otg"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()

    def _get_tool_rotation(self) -> np.ndarray:
        return self._read_tcp_pose()[:3, :3]

    @staticmethod
    def _normalize_gripper_width(gripper_width_m: float) -> float:
        return float(np.clip(gripper_width_m / 0.085, 0.0, 1.0))

    def _gripper_array_from_width(self, gripper_width_m: float) -> np.ndarray:
        return np.array([self._normalize_gripper_width(gripper_width_m)])

    def get_state_snapshot(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        """Thread-safe snapshot of (state, action, force, optional extras)."""
        with self._state_lock:
            if self.save_eef:
                state = (
                    self.state_eef.copy(),
                    self.previous_solution_eef.copy(),
                    self.tcp_force.copy(),
                )
            else:
                state = (
                    self.state.copy(),
                    self.previous_solution.copy(),
                    self.tcp_force.copy(),
                )
            state_timestamp_ns = self.state_timestamp_ns
            action_timestamp_ns = self.action_timestamp_ns
        extra = {
            "robot_state_timestamp_ns": np.array(state_timestamp_ns, dtype=np.int64),
            "robot_action_timestamp_ns": np.array(action_timestamp_ns, dtype=np.int64),
        }
        if self.collect_tcp_extra:
            extra["tcp_pose"] = self.capture_tcp_pose()
        return (*state, extra)

    # ── Ruckig OTG ────────────────────────────────────────────────────────

    def _init_ruckig(self, cfg) -> None:
        from ruckig import Ruckig, InputParameter, OutputParameter
        self._otg = Ruckig(6, 1.0 / self.control_rate)
        self._otg_inp = InputParameter(6)
        self._otg_out = OutputParameter(6)
        self._otg_inp.max_velocity = self._ruckig_limits(
            cfg.RUCKIG_MAX_VEL, "velocity"
        )
        self._otg_inp.max_acceleration = self._ruckig_limits(
            cfg.RUCKIG_MAX_ACC, "acceleration"
        )
        self._otg_inp.max_jerk = self._ruckig_limits(cfg.RUCKIG_MAX_JERK, "jerk")
        self._reset_ruckig_state(self.initial_joint)
        utils.logger.info("Ruckig OTG enabled for servo mode")

    @staticmethod
    def _ruckig_limits(value: float | np.ndarray, name: str) -> list[float]:
        limits = np.asarray(value, dtype=float)
        if limits.ndim == 0:
            return [float(limits)] * 6
        if limits.shape != (6,):
            raise ValueError(
                f"Ruckig {name} limits must be scalar or shape (6,), got {limits.shape}"
            )
        return limits.tolist()

    def _reset_ruckig_state(self, position: np.ndarray) -> None:
        self._otg_inp.current_position = list(position)
        self._otg_inp.current_velocity = [0.0] * 6
        self._otg_inp.current_acceleration = [0.0] * 6

    def _set_ruckig_target(self, target: np.ndarray) -> None:
        self._otg_inp.target_position = target.tolist()
        self._otg_inp.target_velocity = [0.0] * 6
        self._otg_inp.target_acceleration = [0.0] * 6

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
        tcp = self._read_tcp_pose()
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(tcp)

    @staticmethod
    def _extract_se3(controller_data: list[dict]) -> SE3Container:
        r = controller_data[1]["Rotation"]
        p = controller_data[1]["Position"]
        rotation_rh = np.array([r[0], r[2], -r[1], r[3]])
        position_rh = np.array([-p[0], -p[2], p[1]])
        return SE3Container.from_quaternion_and_translation(rotation_rh, position_rh)

    # ── Fine-mode toggle ──────────────────────────────────────────────────

    def _update_fine_mode(self, controller_data: list[dict], mode_status: str | None) -> None:
        if mode_status == "ON" and not self.fine_mode:
            utils.logger.info("Fine Control Mode: ON")
            self.fine_mode = True
            self._set_reference(controller_data)
        elif mode_status == "OFF" and self.fine_mode:
            utils.logger.info("Fine Control Mode: OFF")
            self.fine_mode = False
            self._set_reference(controller_data)

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
        return np.concatenate([self.last_tcp_quat, se3.translation])

    def capture_tcp_force(self) -> np.ndarray:
        raw = self._read_raw_force()
        if self.gravity_comp:
            R = self._get_tool_rotation()
            return self.gravity_compensator.compensate(raw, R)
        return raw

    def capture_gripper_width(self) -> float:
        return float(self.gripper.get_current_width())

    def capture_gripper(self) -> np.ndarray:
        return self._gripper_array_from_width(self.capture_gripper_width())

    # ── Force calibration ─────────────────────────────────────────────────

    def calibrate_force_sensor(self) -> None:
        if not self.gravity_comp:
            return
        n = self._calib_samples_needed
        utils.logger.info(f"Calibrating force sensor ({n} samples)...")
        for _ in range(n):
            raw = self._read_raw_force()
            R = self._get_tool_rotation()
            self.gravity_compensator.add_calibration_sample(raw, R)
            time.sleep(0.005)
        self.gravity_compensator.finish_calibration()

    # ── Gripper update ────────────────────────────────────────────────────

    def _update_gripper(self, gripper_state: int, dt: float, gripper_width: float, ur_pose: np.ndarray | None = None) -> None:
        joints_for_state = ur_pose if ur_pose is not None else self.previous_solution[:6]
        gripper_width_norm = self._normalize_gripper_width(gripper_width)

        if gripper_state:
            self.gripper_solution_width += self.gripper_speed * dt * gripper_state
            self.gripper_solution_width = np.clip(
                self.gripper_solution_width, 0.0, self.gripper_max
            )
            if gripper_state < 0:
                self.gripper_solution_width = max(self.gripper_solution_width, gripper_width)
            else:
                self.gripper_solution_width = min(self.gripper_solution_width, gripper_width)

            if not self.gripper_stop_control_sign or gripper_state != self._gripper_direction:
                self._gripper_direction = gripper_state
                destination = 0.0 if gripper_state < 0 else self.gripper_max
                self.gripper._set_target_width(destination)
                self.gripper_stop_control_sign = True

            with self._state_lock:
                self.state_timestamp_ns = time.monotonic_ns()
                gripper_target_norm = self._normalize_gripper_width(
                    self.gripper_solution_width
                )
                self.previous_solution = np.concatenate(
                    [self.previous_solution[:6], [gripper_target_norm]]
                )
                self.state = np.concatenate(
                    [joints_for_state, [gripper_width_norm]]
                )

            if self.save_eef:
                gripper_target_norm = self._normalize_gripper_width(
                    self.gripper_solution_width
                )
                self.previous_solution_eef = np.concatenate(
                    [self.previous_solution_eef[:7], [gripper_target_norm]]
                )

        elif not gripper_state and self.gripper_stop_control_sign:
            self.gripper_solution_width = gripper_width
            self.gripper_stop_control_sign = False
            self._gripper_direction = 0
            self.gripper_solution_width = np.clip(
                self.gripper_solution_width, 0.0, self.gripper_max
            )
            self.gripper._set_target_width(self.gripper_solution_width)

            with self._state_lock:
                self.state_timestamp_ns = time.monotonic_ns()
                gripper_target_norm = self._normalize_gripper_width(
                    self.gripper_solution_width
                )
                self.previous_solution = np.concatenate(
                    [self.previous_solution[:6], [gripper_target_norm]]
                )
                self.state = np.concatenate(
                    [joints_for_state, [gripper_width_norm]]
                )

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset_robot_and_gripper(self) -> None:
        gripper_to_default = self.gripper.move(self.gripper_max)

        if self.torque_mode:
            self.ur.tmp_move(self.initial_joint)
            with self._state_lock:
                now_ns = time.monotonic_ns()
                self.state_timestamp_ns = now_ns
                self.action_timestamp_ns = now_ns
                self.previous_solution = np.concatenate([self.initial_joint, [1.0]])
            self.gripper_solution_width = self.gripper.get_current_width()
            self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(
                self._read_tcp_pose()
            )
            self.filtered_joint_target = np.array(self.initial_joint)
            self.last_sent_target = np.array(self.initial_joint)
            return

        robot_to_default = self.ur.move_to_joint_configuration(self.initial_joint, 1)

        while not robot_to_default.is_action_done():
            gripper_capture = self.capture_gripper()
            gripper_target_norm = self._normalize_gripper_width(self.gripper_max)

            if self.save_eef:
                eef_pose = self.capture_eef_pose(self.last_quat)
                self.previous_solution_eef = np.concatenate([eef_pose, [gripper_target_norm]])
                self.state_eef = np.concatenate(
                    [self.previous_solution_eef[:7], gripper_capture]
                )
            else:
                with self._state_lock:
                    self.state_timestamp_ns = time.monotonic_ns()
                    self.previous_solution = np.concatenate(
                        [self.capture_joint_pose(), [gripper_target_norm]]
                    )
                    self.state = np.concatenate(
                        [self.previous_solution[:6], gripper_capture]
                    )
            time.sleep(1 / self.control_rate)

        utils.logger.info("Gripper and robot in default pose!")

        with self._state_lock:
            now_ns = time.monotonic_ns()
            self.state_timestamp_ns = now_ns
            self.action_timestamp_ns = now_ns
            self.previous_solution = np.concatenate([self.initial_joint, [1.0]])
        self.gripper_solution_width = self.gripper.get_current_width()
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(
            self.ur.get_tcp_pose()
        )
        self.filtered_joint_target = np.array(self.initial_joint)
        self.last_sent_target = np.array(self.initial_joint)
        if self.ruckig_enable:
            self._reset_ruckig_state(self.initial_joint)

        if self.save_eef:
            self.last_quat = utils.quat_cal(
                self.SE3_tcp_pose_in_base_frame_std.rotation_matrix
            )
            self.ur_eef_capture = np.concatenate(
                [self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation]
            )
            self.previous_solution_eef = np.concatenate([self.ur_eef_capture, [1.0]])

        utils.logger.info("---- Reset complete ----")

    # ── Standby / Teleop modes ────────────────────────────────────────────

    def _standby_mode(self, controller_data: list[dict]) -> bool:
        if not controller_data[1]["GripTrigger"]:
            self._set_reference(controller_data)
            utils.logger.debug("Standby mode active")
            return True
        return False

    def _teleop_mode(
        self, controller_data: list[dict], gripper_state: int, dt: float
    ) -> None:
        if self.reset_sign:
            self.reset_sign = False
            self._set_reference(controller_data)

        SE3_controller = self._extract_se3(controller_data)
        se3_mat = SE3_controller.homogeneous_matrix

        translation_diff = se3_mat[:3, 3] - self.SE3_controller_std.translation
        rotation_diff = self.SE3_controller_std.rotation_matrix.T @ se3_mat[:3, :3]

        translation_diff = self.pos_filter.update(translation_diff, dt)

        if self.fine_mode:
            alpha_t, alpha_r, beta = 0.3, 0.4, 0.3
        else:
            alpha_t, alpha_r, beta = 1.0, 1.0, 0.7

        rvec, _ = cv2.Rodrigues(rotation_diff)
        rvec = self.rot_filter.update(rvec.flatten(), dt) * alpha_r
        rotation_diff, _ = cv2.Rodrigues(rvec)
        translation_diff *= alpha_t

        target_translation = (
            self.SE3_tcp_pose_in_base_frame_std.translation + translation_diff
        )
        target_rotation = rotation_diff @ self.SE3_tcp_pose_in_base_frame_std.rotation_matrix
        tcp_target = SE3Container.from_rotation_matrix_and_translation(
            target_rotation, target_translation
        )

        current_joints = self._read_joints()
        # Remove the external joystick bias from the IK seed to prevent the 
        # IK solver from abruptly jumping to a +/- 2pi solution when crossing pi.
        ik_seed_joints = current_joints.copy()
        ik_seed_joints[5] -= self.last_joint_bias
        
        joint_solution = self.ik.inverse_kinematics_closest_with_tcp(
            tcp_target.homogeneous_matrix, self.tcp_transform, *ik_seed_joints
        )

        with self._state_lock:
            prev = self.previous_solution[:6]

        unsafe = False
        if not joint_solution or not utils.is_joint_within_limits(joint_solution[0]):
            utils.logger.warning("No valid IK solution, keeping previous pose!")
            unsafe = True
        else:
            joystick_x = controller_data[1]["Joystick"][0]
            proposed_bias = self.last_joint_bias + ((joystick_x > 0.8) - (joystick_x < -0.8)) * 0.01
            
            # Clamping the final joint angle to UR limits to avoid wind-up
            target_j5 = joint_solution[0][5] + proposed_bias
            target_j5_clipped = np.clip(target_j5, utils.UR3E_JOINT_LIMITS[0], utils.UR3E_JOINT_LIMITS[1])
            proposed_bias = target_j5_clipped - joint_solution[0][5]
            
            joint_solution[0][5] = target_j5_clipped

            if not utils.is_pose_safe(
                joint_solution[0], tcp_target.translation, robot_type=self.robot_type
            ):
                unsafe = True
            elif not utils.is_joint_change_safe(prev, joint_solution[0], self.joint_threshold):
                utils.logger.warning("Joint change unsafe, keeping previous pose!")
                unsafe = True

        if unsafe:
            joint_solution = [prev]
        else:
            self.last_joint_bias = proposed_bias

        final_target = np.array(joint_solution[0])

        if self.torque_mode:
            self.ur.servo_to_joint_configuration(final_target, 1 / self.control_rate)
        elif self.ruckig_enable:
            final_target = self._ruckig_step(final_target)
            self.ur.servo_to_joint_configuration(final_target, 1 / self.control_rate)
        else:
            self.filtered_joint_target = (
                beta * final_target + (1 - beta) * self.filtered_joint_target
            )
            raw_delta = self.filtered_joint_target - self.last_sent_target
            max_step = 0.02
            max_change = np.max(np.abs(raw_delta))
            if max_change > max_step:
                raw_delta *= max_step / max_change
            final_target = self.last_sent_target + raw_delta
            self.last_sent_target = final_target
            self.ur.servo_to_joint_configuration(final_target, 1 / self.control_rate)

        self.filtered_joint_target = final_target.copy()
        self.last_sent_target = final_target.copy()

        with self._state_lock:
            self.action_timestamp_ns = time.monotonic_ns()
            gripper_target_norm = self._normalize_gripper_width(
                self.gripper_solution_width
            )
            self.previous_solution = np.concatenate(
                [final_target, [gripper_target_norm]]
            )

        if self.save_eef:
            tcp_fk = self.ik.forward_kinematics(*joint_solution[0])
            self.last_quat = utils.quat_cal(tcp_fk[:3, :3], self.last_quat)
            solution_eef = np.concatenate([self.last_quat, tcp_fk[:3, 3]])
            self.previous_solution_eef = np.concatenate(
                [solution_eef, [gripper_target_norm]]
            )

        utils.logger.debug("Teleop step executed successfully.")

    # ── Hand-tracking helpers ──────────────────────────────────────────────

    # OpenXR 26-joint indices
    _THUMB_TIP_IDX = 5
    _INDEX_TIP_IDX = 10
    _MIDDLE_TIP_IDX = 15
    _RING_TIP_IDX = 20
    _PINKY_TIP_IDX = 25

    @staticmethod
    def _extract_hand_se3(rh: dict) -> SE3Container:
        """Convert hand data to SE3. Uses wrist_pose if available, else bones[0] (palm)."""
        wrist_pose = rh.get("wrist_pose")
        if wrist_pose is not None:
            p = wrist_pose["position"]
            r = wrist_pose["rotation"]
            rotation_rh = np.array([r[0], r[2], -r[1], r[3]])
            position_rh = np.array([-p[0], -p[2], p[1]])
            return SE3Container.from_quaternion_and_translation(rotation_rh, position_rh)
        else:
            # Binary format (HB) only has positions, no wrist rotation
            p = rh["bones"][0]  # Palm
            position_rh = np.array([-p[0], -p[2], p[1]])
            return SE3Container.from_quaternion_and_translation(
                np.array([0.0, 0.0, 0.0, 1.0]), position_rh
            )

    def _hand_set_reference(self, hand_se3: SE3Container) -> None:
        """Record current hand SE3 and robot TCP as reference for delta control."""
        self._hand_ref_se3 = hand_se3
        tcp = self._read_tcp_pose()
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(tcp)
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
        low, high = utils.UR3E_JOINT_LIMITS
        return np.clip(filtered, low, high)

    def _seed_hand_joint_filter(self, joints: np.ndarray) -> None:
        self.hand_joint_filter.value = np.asarray(joints, dtype=float).copy()
        self.hand_joint_filter.initialized = True

    def _hand_teleop_step(self, hand_data: dict | None, dt: float) -> None:
        """One teleop step using right-hand wrist_pose for TCP control.

        - wrist_pose position + rotation → TCP (SE3 delta, same as controller)
        - Thumb tip / index tip distance → gripper 3-zone control
        """
        if hand_data is None or "R" not in hand_data:
            return

        rh = hand_data["R"]
        bones = rh.get("bones")
        if bones is None or len(bones) != 26:
            return  # Only accept OpenXR 26-joint hand data
        utils.logger.debug(f"Hand bones sample: {bones}")

        # Check for fake packet (Unity sometimes sends hand packets with 0,0,0 fingers when holding controller)
        if np.linalg.norm(bones[self._INDEX_TIP_IDX]) < 1e-4:
            return

        hand_se3 = self._extract_hand_se3(rh)

        # ── Jump detection ────────────────────────────────────────────────
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

        # ── Initialise reference on first valid frame ─────────────────────
        if not self._hand_initialized:
            self._hand_set_reference(hand_se3)
            self.reset_sign = False
            return

        # ── SE3 delta (same approach as _teleop_mode) ─────────────────────
        se3_mat = hand_se3.homogeneous_matrix
        translation_diff = se3_mat[:3, 3] - self._hand_ref_se3.translation
        rotation_diff = self._hand_ref_se3.rotation_matrix.T @ se3_mat[:3, :3]

        translation_diff = self.pos_filter.update(translation_diff, dt)

        rvec, _ = cv2.Rodrigues(rotation_diff)
        rvec = self.rot_filter.update(rvec.flatten(), dt)
        rotation_diff, _ = cv2.Rodrigues(rvec)

        target_translation = (
            self.SE3_tcp_pose_in_base_frame_std.translation + translation_diff
        )
        target_rotation = rotation_diff @ self.SE3_tcp_pose_in_base_frame_std.rotation_matrix
        tcp_target = SE3Container.from_rotation_matrix_and_translation(
            target_rotation, target_translation
        )

        current_joints = self._read_joints()
        joint_solution = self.ik.inverse_kinematics_closest_with_tcp(
            tcp_target.homogeneous_matrix, self.tcp_transform, *current_joints
        )

        with self._state_lock:
            prev = self.previous_solution[:6]

        unsafe = False
        final_target = None
        if not joint_solution or not utils.is_joint_within_limits(joint_solution[0]):
            utils.logger.warning("Hand IK: no valid solution, keeping pose")
            unsafe = True
        else:
            final_target = np.array(joint_solution[0])
            if self.ruckig_enable:
                final_target = self._filter_hand_joint_target(final_target, dt)

            if not utils.is_pose_safe(
                final_target, tcp_target.translation, robot_type=self.robot_type
            ):
                unsafe = True
            elif not utils.is_joint_change_safe(prev, final_target, self.joint_threshold):
                utils.logger.warning("Hand IK: joint change unsafe, keeping pose")
                unsafe = True

        if unsafe:
            final_target = prev
            if self.ruckig_enable:
                self._seed_hand_joint_filter(prev)

        if self.torque_mode:
            self.ur.servo_to_joint_configuration(final_target, 1 / self.control_rate)
        elif self.ruckig_enable:
            final_target = self._ruckig_step(final_target)
            self.ur.servo_to_joint_configuration(final_target, 1 / self.control_rate)
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
            self.last_sent_target = final_target
            self.ur.servo_to_joint_configuration(final_target, 1 / self.control_rate)

        self.filtered_joint_target = final_target.copy()
        self.last_sent_target = final_target.copy()

        with self._state_lock:
            self.action_timestamp_ns = time.monotonic_ns()
            gripper_target_norm = self._normalize_gripper_width(
                self.gripper_solution_width
            )
            self.previous_solution = np.concatenate(
                [final_target, [gripper_target_norm]]
            )

        # ── Gripper from thumb–index distance ─────────────────────────────
        thumb_tip = np.array(bones[self._THUMB_TIP_IDX])
        index_tip = np.array(bones[self._INDEX_TIP_IDX])
        finger_dist = np.linalg.norm(thumb_tip - index_tip)

        if finger_dist > self._hand_gripper_open:
            gripper_state = 1    # open
        elif finger_dist < self._hand_gripper_close:
            gripper_state = -1   # close
        else:
            gripper_state = 0    # dead-zone, hold

        ur_pose = self.capture_joint_pose()
        gripper_width_m = self.capture_gripper_width()
        gripper_capture = self._gripper_array_from_width(gripper_width_m)
        self._update_gripper(gripper_state, dt, gripper_width_m, ur_pose)

        with self._state_lock:
            self.state_timestamp_ns = time.monotonic_ns()
            self.state = np.concatenate([ur_pose, gripper_capture])

        utils.logger.debug(
            f"Hand teleop: finger_dist={finger_dist:.3f}m  gripper_state={gripper_state}"
        )

    # ── Main step ─────────────────────────────────────────────────────────

    def step(self, controller_data: list[dict], fine_mode_status: str | None,
             dt: float = 0.01, hand_data: dict | None = None) -> None:
        
        # Determine if controller is actively being operated
        ctrl_active = False
        if controller_data is not None:
            rd = controller_data[1]  # Right hand
            ctrl_active = (
                rd["GripTrigger"]
                or rd["IndexTrigger"] > 0.5
                or rd["Button_AX"]
                or rd["Button_BY"]
                or abs(rd["Joystick"][0]) > 0.3
                or abs(rd["Joystick"][1]) > 0.3
            )

        # ── Mode Toggle Gesture ───────────────────────────────────────────
        # Hand toggle gesture is only valid if we aren't actively using the controller
        if not ctrl_active and hand_data is not None and "R" in hand_data:
            bones = hand_data["R"].get("bones")
            if bones is not None and len(bones) == 26:
                if np.linalg.norm(bones[self._INDEX_TIP_IDX]) >= 1e-4:
                    thumb_tip = np.array(bones[self._THUMB_TIP_IDX])
                    pinky_tip = np.array(bones[self._PINKY_TIP_IDX])
                    ring_tip = np.array(bones[self._RING_TIP_IDX])
                    
                    if np.linalg.norm(thumb_tip - pinky_tip) < self._hand_mode_toggle_dist:
                        current_time = time.time()
                        if (current_time - self._last_toggle_time) > 1.0:
                            if self.tracking_mode == "hand":
                                utils.logger.warning("Gesture: Switched to CONTROLLER mode")
                                self.tracking_mode = "controller"
                                self._reset_hand_reference_state()
                            else:
                                utils.logger.warning("Gesture: Switched to HAND mode")
                                self.tracking_mode = "hand"
                                self._reset_hand_reference_state()
                            self._last_toggle_time = current_time
                    elif np.linalg.norm(thumb_tip - ring_tip) < self._hand_reset_dist:
                        current_time = time.time()
                        if (current_time - self._last_reset_time) > 2.0:
                            utils.logger.warning("Gesture: Resetting Robot and Gripper to Initial Position")
                            self.reset_sign = True
                            self.reset_robot_and_gripper()
                            self._reset_hand_reference_state()
                            self._last_reset_time = current_time
                            return

        # ── Hand tracking mode ────────────────────────────────────────────
        if self.tracking_mode == "hand":
            # Reject hand data when controllers are actively being used.
            # Quest sends hand bones even while holding controllers.
            if ctrl_active:
                utils.logger.debug("Hand mode: controller active, ignoring hand data")
                return

            ur_pose = self.capture_joint_pose()
            if self.save_eef:
                self.ur_eef_capture = self.capture_eef_pose(self.last_quat)

            if self.force_mode:
                force = self.capture_tcp_force()
                with self._state_lock:
                    self.tcp_force = force

            self._hand_teleop_step(hand_data, dt)
            return

        # ── Controller mode (original logic) ──────────────────────────────
        if controller_data is None:
            return

        self._update_fine_mode(controller_data, fine_mode_status)

        ur_pose = self.capture_joint_pose()
        if self.save_eef:
            self.ur_eef_capture = self.capture_eef_pose(self.last_quat)
        gripper_width_m = self.capture_gripper_width()
        gripper_capture = self._gripper_array_from_width(gripper_width_m)

        if self.force_mode:
            force = self.capture_tcp_force()
            with self._state_lock:
                self.tcp_force = force

        x = -controller_data[1]["Joystick"][1]
        gripper_state = (x > 0.7) - (x < -0.7)

        self._update_gripper(gripper_state, dt, gripper_width_m, ur_pose)

        reset_requested = (
            bool(controller_data[1]["Joystick_Press"])
            and controller_data[1]["IndexTrigger"] >= self._controller_reset_trigger_threshold
        )
        if reset_requested:
            self.reset_sign = True
            self.reset_robot_and_gripper()
            return

        if not self._standby_mode(controller_data):
            with self._state_lock:
                self.state_timestamp_ns = time.monotonic_ns()
                self.state = np.concatenate([ur_pose, gripper_capture])
            self._teleop_mode(controller_data, gripper_state, dt)
