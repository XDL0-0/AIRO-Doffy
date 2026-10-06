"""Composed RealMan teleoperation runtime."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from airo_spatial_algebra.se3 import SE3Container

import doffy_teleop.utils as utils
from doffy_teleop.config import Config
from doffy_teleop.sensors.force_filter import WrenchFilter
from doffy_teleop.visualization.config import VisualizerConfig
from ..robots.backends import RealManBackend
from ..robots.factory import make_robot_backend
from ..control.canfd_loop import CanfdCommandLoop
from ..control.contracts import RealManStateSnapshot
from ..control.realman_qp import RealManRemoteIkSolver
from ..control.input_mapping import InputMappingMixin
from ..control.target_policy import TargetPolicyMixin
from .state import StateLifecycleMixin, _UNCLOSED_REALMAN_TELEOPS


class RealManTeleop(InputMappingMixin, TargetPolicyMixin, StateLifecycleMixin):
    """RealMan controller/hand mapping and cached robot sensor state."""

    def __init__(
        self,
        initial_controller_data: list[dict],
        *,
        cfg: Config | None = None,
        backend: RealManBackend | None = None,
        backend_factory=None,
    ) -> None:
        self.cfg = Config() if cfg is None else cfg
        if self.cfg.ROBOT_TYPE != "realman":
            raise ValueError("realman_teleop.py requires Config.ROBOT_TYPE='realman'.")
        if self.cfg.GRIPPER:
            raise ValueError("Disable Config.GRIPPER for the RealMan-only teleoperation script.")
        if self.cfg.TACTILE_TRANSFER:
            raise ValueError("Disable Config.TACTILE_TRANSFER; this script uses robot force only.")

        created_backend = backend is None
        factory = make_robot_backend if backend_factory is None else backend_factory
        self.backend = factory(self.cfg) if backend is None else backend
        try:
            if self.backend.name != "realman":
                raise TypeError("The selected backend is not a RealMan backend.")
            if not self.backend.supports_force:
                raise RuntimeError("The connected RealMan backend does not expose a force sensor.")

            self.dof = int(self.backend.dof)
            self.wrm_enabled = bool(getattr(self.cfg, "WRM_enable", False))
            if self.wrm_enabled and self.dof != 7:
                raise ValueError("WRM AKM requires a seven-DoF RM75.")
            if self.wrm_enabled and self.cfg.TRACKING_MODE != "controller":
                raise ValueError("WRM AKM currently supports VR controller tracking only.")
            # Redundancy/arm-angle control requires joint CAN-FD targets.  The
            # disabled path retains TELEOP_COMMAND_MODE without modification.
            self.control_mode = (
                "joint" if self.wrm_enabled else self.cfg.TELEOP_COMMAND_MODE
            )
            self.tracking_mode = self.cfg.TRACKING_MODE
            self.tcp_tool = getattr(self.backend, "tcp_tool", self.cfg.TCP_TOOL)
            self.hand = (
                getattr(self.backend, "hand", None)
                if getattr(self.cfg, "BRAINCO_HAND_ENABLE", True)
                else None
            )
            self.joint_threshold = self._joint_threshold_for_dof(self.cfg.MOVE_THRESHOLD)
            self.vr_to_robot_axes = np.asarray(self.cfg.VR_TO_ROBOT_AXES, dtype=float)
            self.vr_to_robot_handedness = float(np.linalg.det(self.vr_to_robot_axes))
            self.tcp_transform = np.asarray(self.cfg.TCP_TRANSFORM, dtype=float)
            self.inv_tcp_transform = np.linalg.inv(self.tcp_transform)
            controller_reference = self._extract_controller_se3(
                initial_controller_data
            )

            initial_joint = self.backend.initial_joint_configuration(self.cfg.INITIAL_JOINT)
            self.initial_joint = np.asarray(initial_joint, dtype=float).copy()
            utils.logger.info(f"Moving RealMan to initial joint configuration: {initial_joint}")
            self.backend.reset(initial_joint)

            joints = np.asarray(
                self.backend.get_joint_configuration(),
                dtype=float,
            )
            tcp_pose = np.asarray(self.backend.get_tcp_pose(), dtype=float)
            raw_wrench = self.backend.get_tcp_force()
            if raw_wrench is None:
                raise RuntimeError(
                    "RealMan force sensor did not return a six-axis wrench during startup."
                )
            raw_wrench = np.asarray(raw_wrench, dtype=float)
            self._validate_sensor_state(joints, tcp_pose)
            self._validate_wrench(raw_wrench)
            self._finish_initialization(
                controller_reference,
                joints,
                tcp_pose,
                raw_wrench,
            )

        except Exception:
            if created_backend:
                self.backend.cleanup()
            raise

    def _finish_initialization(
        self,
        controller_reference: SE3Container,
        joints: np.ndarray,
        tcp_pose: np.ndarray,
        raw_wrench: np.ndarray,
    ) -> None:
        """Initialize local state after the robot connection has been verified."""

        self.wrench_filter = WrenchFilter(
            moving_average_window=self.cfg.FORCE_MOVING_AVERAGE_WINDOW,
            low_pass_alpha=self.cfg.FORCE_LOW_PASS_ALPHA,
        )
        wrench = self.wrench_filter.process(raw_wrench)

        self._sensor_lock = threading.Lock()
        now_ns = time.monotonic_ns()
        self._joints = joints.copy()
        self._tcp_pose = tcp_pose.copy()
        self._wrench = np.asarray(wrench, dtype=float)
        self._state_timestamp_ns = now_ns
        self._force_timestamp_ns = now_ns
        self._action_timestamp_ns = now_ns
        self._state_error = ""
        self._force_error = ""

        self._control_lock = threading.Lock()
        self._input_stale = False
        self._requires_reference = True
        self._grip_active = False
        self._controller_reset_held = False
        self._hand_joystick_motion: str | None = None
        self._hand_joystick_threshold = (
            self.cfg.BRAINCO_HAND_JOYSTICK_THRESHOLD
        )
        self._wrist_joint_bias = 0.0
        self._reset_in_progress = False
        self._reset_requires_grip_release = False
        self._last_joint_target = self._joints.copy()
        self._last_tcp_target = self._tcp_pose.copy()
        self._initial_tcp_pose = self._tcp_pose.copy()
        self._controller_reference = controller_reference
        self._robot_reference = SE3Container.from_homogeneous_matrix(self._tcp_pose)
        self._hand_initialized = False
        self._hand_last_palm: np.ndarray | None = None

        self._position_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=self.cfg.CARTESIAN_POS_FILTER_CUTOFF_HZ,
            dim=3,
        )
        self._rotation_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=self.cfg.CARTESIAN_ROT_FILTER_CUTOFF_HZ,
            dim=3,
        )
        self._joint_filter = utils.TimeAwareLowPassFilter(
            cutoff_hz=self.cfg.HAND_JOINT_FILTER_CUTOFF_HZ,
            dim=self.dof,
        )
        self._seed_joint_filter(self._joints)

        raw_arm = getattr(self.backend.robot, "robot", None)
        if raw_arm is None:
            raise TypeError("RealMan backend does not expose the vendor SDK arm object.")
        self._raw_arm = raw_arm
        self._realman_api = getattr(self.backend.robot, "_api", None)
        self._remote_ik_solver: RealManRemoteIkSolver | None = None
        self._wrm_akm = None
        self._last_remote_ik_status = 0
        self._state_push_callback = None
        self._state_push_active = False
        self._state_push_received = threading.Event()
        self._state_callback_condition = threading.Condition()
        self._accept_state_callbacks = False
        self._state_callbacks_in_flight = 0
        self._lifecycle_lock = threading.Lock()
        self._sdk_worker_threads: list[threading.Thread] = []
        self._started = False
        self._closed = False
        self._quarantined = False

        joint_speed_limits = np.full(
            self.dof,
            self.cfg.REALMAN_MAX_JOINT_SPEED,
            dtype=float,
        )
        joint_acceleration_limits = np.full(
            self.dof,
            self.cfg.REALMAN_MAX_JOINT_ACCELERATION,
            dtype=float,
        )
        linear_speed_limit = float(self.cfg.REALMAN_MAX_LINEAR_SPEED)
        specs = getattr(self.backend.robot, "manipulator_specs", None)
        if specs is not None:
            controller_joint_speeds = np.asarray(
                specs.max_joint_speeds,
                dtype=float,
            )
            if controller_joint_speeds.shape == (self.dof,):
                joint_speed_limits = np.minimum(
                    joint_speed_limits,
                    controller_joint_speeds,
                )
            controller_linear_speed = float(specs.max_linear_speed)
            if controller_linear_speed > 0.0:
                linear_speed_limit = min(
                    linear_speed_limit,
                    controller_linear_speed,
                )
        self._joint_speed_limits = joint_speed_limits

        if self.wrm_enabled:
            from doffy_teleop.control.wrm_akm import Rm75ArmAngleIk

            self._wrm_akm = Rm75ArmAngleIk(
                raw_arm,
                self._realman_api,
                self._joints,
                self.backend.to_robot_tcp_pose(self._tcp_pose),
                lower_limits_radians=getattr(
                    self.backend.robot,
                    "_joint_lower_limits",
                    None,
                ),
                upper_limits_radians=getattr(
                    self.backend.robot,
                    "_joint_upper_limits",
                    None,
                ),
            )
            self._wrm_akm.set_tcp_z_reference()
            utils.logger.info(
                "WRM AKM calibrated: robot elbow arm-angle %.1f deg (high) "
                "-> %.1f deg (horizontal), TCP Z drop up to %.3f m.",
                self._wrm_akm.robot_elbow_high,
                self._wrm_akm.robot_elbow_horizontal,
                self.cfg.WRM_TCP_Z_DROP_M,
            )
        elif self.control_mode == "joint" and self.cfg.REALMAN_QP_IK_ENABLE:
            if RealManRemoteIkSolver.is_available(raw_arm, self._realman_api):
                self._remote_ik_solver = RealManRemoteIkSolver(
                    raw_arm,
                    self._realman_api,
                    dof=self.dof,
                    period_s=1.0 / self.cfg.REALMAN_CTRL_RATE,
                    initial_joints_radians=self._joints,
                    joint_acceleration_limits_radians=(
                        joint_acceleration_limits
                    ),
                    dq_weight=self.cfg.REALMAN_QP_DQ_WEIGHT,
                    limit_holdon=self.cfg.REALMAN_QP_LIMIT_HOLDON,
                    elbow_margin_degrees=self.cfg.REALMAN_QP_ELBOW_MARGIN_DEG,
                )
            else:
                utils.logger.warning(
                    "RealMan teleoperation QP solver is unavailable in this SDK; "
                    "falling back to one-shot inverse kinematics."
                )

        self.canfd = CanfdCommandLoop(
            raw_arm,
            control_mode=self.control_mode,
            dof=self.dof,
            target_hz=self.cfg.REALMAN_CTRL_RATE,
            minimum_hz=self.cfg.REALMAN_MIN_CANFD_RATE,
            rate_check_window=self.cfg.REALMAN_RATE_CHECK_WINDOW,
            maximum_failure_windows=self.cfg.REALMAN_RATE_FAILURE_WINDOWS,
            trajectory_mode=self.cfg.REALMAN_CANFD_TRAJECTORY_MODE,
            radio=self.cfg.REALMAN_CANFD_RADIO,
            joint_speed_limits=joint_speed_limits,
            joint_acceleration_limits=joint_acceleration_limits,
            linear_speed_limit=linear_speed_limit,
            linear_acceleration_limit=(
                self.cfg.REALMAN_MAX_LINEAR_ACCELERATION
            ),
            angular_speed_limit=self.cfg.REALMAN_MAX_ANGULAR_SPEED,
            angular_acceleration_limit=(
                self.cfg.REALMAN_MAX_ANGULAR_ACCELERATION
            ),
            heartbeat_timeout=self.cfg.REALMAN_CANFD_HEARTBEAT_TIMEOUT,
        )
        if self.control_mode == "joint":
            self.canfd.set_joint_target(self._last_joint_target)
            self.canfd.set_joint_target_resolver(
                self._resolve_joint_target,
                continuous=(
                    self._remote_ik_solver is not None
                    or self._wrm_akm is not None
                ),
            )
        else:
            self.canfd.set_tcp_target(self._tool_tcp_to_realman_pose(self._last_tcp_target))
        if not self.cfg.REALMAN_REALTIME_STATE_PUSH:
            self.canfd.set_maintenance_callback(
                self._refresh_sensors,
                self.cfg.REALMAN_SENSOR_RATE,
            )

        if self.cfg.GRAVITY_COMP:
            utils.logger.warning(
                "GRAVITY_COMP is ignored by realman_teleop.py because RealMan "
                "zero_force_data is already controller-compensated."
            )
        if self.cfg.REALMAN_REALTIME_STATE_PUSH:
            sensor_description = (
                f"UDP push {1000.0 / self.cfg.REALMAN_STATE_PUSH_CYCLE_MS:g} Hz"
            )
        else:
            sensor_description = (
                f"polling {self.cfg.REALMAN_SENSOR_RATE:g} Hz"
            )
        utils.logger.info(
            f"RealMan teleop ready - DoF:{self.dof}, mode:{self.control_mode}, "
            f"tracking:{self.tracking_mode}, tool:{self.tcp_tool}, "
            f"CAN-FD:{self.cfg.REALMAN_CTRL_RATE} Hz, "
            "IK:"
            f"{'WRM arm-angle' if self._wrm_akm else ('QP continuous' if self._remote_ik_solver else 'legacy')}, "
            f"sensors:{sensor_description}"
        )
