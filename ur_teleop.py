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
import cv2
import numpy as np

import utils
from config import Config
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85, rescale_range
from airo_spatial_algebra.se3 import SE3Container


# ── Robot factory ─────────────────────────────────────────────────────────

def make_robot(ur_ip: str, robot_type: str, torque_mode: bool, initial_joint=None):
    if not torque_mode:
        from airo_robots.manipulators.hardware.ur_rtde import URrtde
    else:
        from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque as URrtde

    if robot_type == "ur3e":
        from ur_analytic_ik import ur3e as ik
        return URrtde(ur_ip, URrtde.UR3E_CONFIG, initial_joint_configuration=initial_joint), ik
    elif robot_type == "ur5e":
        from ur_analytic_ik import ur5e as ik
        return URrtde(ur_ip, URrtde.UR5E_CONFIG, initial_joint_configuration=initial_joint), ik
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
        self.ur, self.ik = make_robot(
            cfg.UR_IP, cfg.ROBOT_TYPE, cfg.TORQUE_MODE, self.initial_joint
        )
        self.gripper = FastRobotiq2F85(cfg.UR_IP)
        self.gripper.open()
        self.gripper_solution_width = self.gripper.get_current_width()

        self.last_joint_bias = 0.0
        self.control_rate = cfg.UR_CTRL_RATE
        self.gripper_speed = cfg.GRIPPER_SPEED
        self.gripper_max = cfg.GRIPPER_MAX
        self.save_eef = cfg.SAVE_EEF
        self.tcp_transform = cfg.TCP_TRANSFORM
        self.joint_threshold = cfg.MOVE_THRESHOLD
        self.fine_mode = False
        self.last_quat: np.ndarray | None = None
        self.reset_sign = False
        self.torque_mode = cfg.TORQUE_MODE
        self.gripper_stop_control_sign = False
        self._gripper_direction = 0

        self._state_lock = threading.Lock()

        utils.logger.info(f"Teleop initialized — UR:{cfg.UR_IP}, VR:{cfg.VR_IP}")
        utils.logger.info(f"Moving to initial joint: {self.initial_joint}")

        if self.torque_mode:
            self.ur.target_pos = self.initial_joint
        else:
            self.ur.move_to_joint_configuration(self.initial_joint, 0.5).wait()

        self.previous_solution = np.append(self.initial_joint, 0)
        self.filtered_joint_target = np.array(self.initial_joint)
        self.last_sent_target = np.array(self.initial_joint)
        self.pos_filter = utils.ExponentialFilter(alpha=0.6, dim=3)
        self.rot_filter = utils.ExponentialFilter(alpha=0.6, dim=3)
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
        self.previous_solution = np.append(
            joints, initial_data[1]["Button_BY"] - initial_data[1]["Button_AX"]
        )

        if self.save_eef:
            self.last_quat = utils.quat_cal(
                self.SE3_tcp_pose_in_base_frame_std.rotation_matrix, self.last_quat
            )
            self.ur_eef_capture = np.concatenate(
                [self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation]
            )
            self.previous_solution_eef = np.append(
                self.ur_eef_capture,
                initial_data[1]["Button_BY"] - initial_data[1]["Button_AX"],
            )
            self.state_eef = np.concatenate(
                [self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation,
                 [self.gripper_solution_width]]
            )
        else:
            self.state = np.concatenate(
                [self.previous_solution[:6],
                 self.SE3_tcp_pose_in_base_frame_std.translation,
                 [self.gripper_solution_width]]
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
        if self.torque_mode:
            return self.ur.get_cached_tcp_pose()
        return self.ur.get_tcp_pose()

    def _read_raw_force(self) -> np.ndarray:
        if self.torque_mode:
            return np.array(self.ur.get_cached_tcp_force())
        return np.array(self.ur.get_tcp_force())

    def _get_tool_rotation(self) -> np.ndarray:
        return self._read_tcp_pose()[:3, :3]

    def get_state_snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Thread-safe snapshot of (state, previous_solution, tcp_force)."""
        with self._state_lock:
            return (
                self.state.copy(),
                self.previous_solution.copy(),
                self.tcp_force.copy(),
            )

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
        if self.torque_mode:
            se3 = SE3Container.from_rotation_vector_and_translation(tcp[3:6], tcp[0:3])
        else:
            se3 = SE3Container.from_homogeneous_matrix(tcp)
        self.last_quat = utils.quat_cal(se3.rotation_matrix, last_quat)
        return np.concatenate([self.last_quat, se3.translation])

    def capture_tcp_force(self) -> np.ndarray:
        raw = self._read_raw_force()
        if self.gravity_comp:
            R = self._get_tool_rotation()
            return self.gravity_compensator.compensate(raw, R)
        return raw

    def capture_gripper(self) -> np.ndarray:
        return np.array([self.gripper.get_current_width()])

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

    def _update_gripper(self, gripper_state: int, dt: float, gripper_width: float) -> None:
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
                self.previous_solution = np.concatenate(
                    [self.previous_solution[:6], [self.gripper_solution_width]]
                )
                self.state = np.concatenate(
                    [self.previous_solution[:6], [gripper_width]]
                )

            if self.save_eef:
                self.previous_solution_eef = np.concatenate(
                    [self.previous_solution_eef[:7], [gripper_state]]
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
                self.previous_solution = np.concatenate(
                    [self.previous_solution[:6], [self.gripper_solution_width]]
                )
                self.state = np.concatenate(
                    [self.previous_solution[:6], [gripper_width]]
                )

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset_robot_and_gripper(self) -> None:
        gripper_to_default = self.gripper.move(self.gripper_max)

        if self.torque_mode:
            self.ur.tmp_move(self.initial_joint)
            with self._state_lock:
                self.previous_solution = np.concatenate([self.initial_joint, [0]])
            self.gripper_solution_width = self.gripper.get_current_width()
            self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(
                self.ur.get_cached_tcp_pose()
            )
            self.filtered_joint_target = np.array(self.initial_joint)
            self.last_sent_target = np.array(self.initial_joint)
            return

        robot_to_default = self.ur.move_to_joint_configuration(self.initial_joint, 1)

        while not robot_to_default.is_action_done():
            gripper_condition = int(not gripper_to_default.is_action_done())
            gripper_capture = self.capture_gripper()

            if self.save_eef:
                eef_pose = self.capture_eef_pose(self.last_quat)
                self.previous_solution_eef = np.concatenate([eef_pose, [gripper_condition]])
                self.state_eef = np.concatenate(
                    [self.previous_solution_eef[:7], gripper_capture]
                )
            else:
                with self._state_lock:
                    self.previous_solution = np.concatenate(
                        [self.capture_joint_pose(), [gripper_condition]]
                    )
                    self.state = np.concatenate(
                        [self.previous_solution[:6], gripper_capture]
                    )
            time.sleep(1 / self.control_rate)

        utils.logger.info("Gripper and robot in default pose!")

        with self._state_lock:
            self.previous_solution = np.concatenate([self.initial_joint, [0]])
        self.gripper_solution_width = self.gripper.get_current_width()
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(
            self.ur.get_tcp_pose()
        )
        self.filtered_joint_target = np.array(self.initial_joint)
        self.last_sent_target = np.array(self.initial_joint)

        if self.save_eef:
            self.last_quat = utils.quat_cal(
                self.SE3_tcp_pose_in_base_frame_std.rotation_matrix
            )
            self.ur_eef_capture = np.concatenate(
                [self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation]
            )
            self.previous_solution_eef = np.concatenate([self.ur_eef_capture, [0]])

        utils.logger.info("---- Reset complete ----")

    # ── Standby / Teleop modes ────────────────────────────────────────────

    def _standby_mode(self, controller_data: list[dict]) -> bool:
        if not controller_data[1]["GripTrigger"]:
            self._set_reference(controller_data)
            utils.logger.debug("Standby mode active")
            return True
        return False

    def _teleop_mode(self, controller_data: list[dict], gripper_state: int) -> None:
        if self.reset_sign:
            self.reset_sign = False
            self._set_reference(controller_data)

        SE3_controller = self._extract_se3(controller_data)
        se3_mat = SE3_controller.homogeneous_matrix

        translation_diff = se3_mat[:3, 3] - self.SE3_controller_std.translation
        rotation_diff = self.SE3_controller_std.rotation_matrix.T @ se3_mat[:3, :3]

        translation_diff = self.pos_filter.update(translation_diff)

        if self.fine_mode:
            alpha_t, alpha_r, beta = 0.3, 0.4, 0.3
        else:
            alpha_t, alpha_r, beta = 1.0, 1.0, 0.7

        rvec, _ = cv2.Rodrigues(rotation_diff)
        rvec = self.rot_filter.update(rvec.flatten()) * alpha_r
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
        joint_solution = self.ik.inverse_kinematics_closest_with_tcp(
            tcp_target.homogeneous_matrix, self.tcp_transform, *current_joints
        )

        if not joint_solution or not utils.is_joint_within_limits(joint_solution[0]):
            utils.logger.warning("No valid IK solution, keeping previous pose!")
            return

        joystick_x = controller_data[1]["Joystick"][0]
        self.last_joint_bias += ((joystick_x > 0.8) - (joystick_x < -0.8)) * 0.01
        self.last_joint_bias = np.clip(self.last_joint_bias, -1.5, 1.5)
        joint_solution[0][5] += self.last_joint_bias

        with self._state_lock:
            prev = self.previous_solution[:6]

        if not utils.is_joint_change_safe(prev, joint_solution[0], self.joint_threshold):
            utils.logger.warning("Joint change unsafe, keeping previous pose!")
            return

        final_target = np.array(joint_solution[0])

        if self.torque_mode:
            self.ur.target_pos = final_target.tolist()
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
            self.previous_solution = np.concatenate(
                [final_target, [self.gripper_solution_width]]
            )

        if self.save_eef:
            tcp_fk = self.ik.forward_kinematics(*joint_solution[0])
            self.last_quat = utils.quat_cal(tcp_fk[:3, :3], self.last_quat)
            solution_eef = np.concatenate([self.last_quat, tcp_fk[:3, 3]])
            self.previous_solution_eef = np.concatenate(
                [solution_eef, [self.gripper_solution_width]]
            )

        utils.logger.debug("Teleop step executed successfully.")

    # ── Main step ─────────────────────────────────────────────────────────

    def step(self, controller_data: list[dict], fine_mode_status: str | None, dt: float = 0.01) -> None:
        self._update_fine_mode(controller_data, fine_mode_status)

        ur_pose = self.capture_joint_pose()
        if self.save_eef:
            self.ur_eef_capture = self.capture_eef_pose(self.last_quat)
        gripper_capture = self.capture_gripper()

        if self.force_mode:
            force = self.capture_tcp_force()
            with self._state_lock:
                self.tcp_force = force

        x = -controller_data[1]["Joystick"][1]
        gripper_state = (x > 0.7) - (x < -0.7)

        self._update_gripper(gripper_state, dt, gripper_capture.item())

        if controller_data[1]["Joystick_Press"] and controller_data[1]["IndexTrigger"] == 1:
            self.reset_sign = True
            self.reset_robot_and_gripper()
            return

        if not self._standby_mode(controller_data):
            with self._state_lock:
                self.state = np.concatenate([ur_pose, gripper_capture])
            self._teleop_mode(controller_data, gripper_state)
