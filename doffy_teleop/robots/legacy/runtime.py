"""Compatibility implementation of the classic ``RobotTeleop`` runtime.

The class keeps lifecycle/configuration wiring in one small entry module while
state, command, hand, and step behavior live in focused mixins.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np

import doffy_teleop.utils as utils
from airo_spatial_algebra.se3 import SE3Container
from doffy_teleop.config import Config
from doffy_teleop.recording.schema import (
    action_representation,
    build_data_schema,
    should_store_extra_tcp_pose,
    state_representation,
)
from doffy_teleop.sensors.force_filter import WrenchFilter
from ..factory import make_robot_backend
from doffy_teleop.visualization.config import VisualizerConfig

from .control import LegacyControlMixin
from .hand import LegacyHandMixin
from .state import LegacyStateMixin
from .step import LegacyStepMixin


class RobotTeleop(LegacyStateMixin, LegacyControlMixin, LegacyHandMixin, LegacyStepMixin):
    """Main teleop class. The historical name is kept for import compatibility."""

    def __init__(
        self,
        initial_data: list[dict],
        *,
        cfg: Config | None = None,
        visualizer_config: VisualizerConfig | None = None,
        backend: Any | None = None,
        backend_factory: Callable[[Config], Any] | None = None,
    ):
        cfg = Config() if cfg is None else cfg
        viz_cfg = (
            VisualizerConfig()
            if visualizer_config is None
            else visualizer_config
        )
        self.cfg = cfg
        if backend is None:
            factory = make_robot_backend if backend_factory is None else backend_factory
            backend = factory(cfg)
        self.backend = backend

        try:
            self._initialize(initial_data, cfg, viz_cfg)
        except BaseException:
            try:
                self.backend.cleanup()
            except Exception as cleanup_error:
                utils.logger.warning(
                    "Robot backend cleanup after initialization failure failed: %s",
                    cleanup_error,
                )
            raise

    def _initialize(self, initial_data, cfg, viz_cfg) -> None:
        self.ur = self.backend.robot
        self.ik = self.backend.ik_solver
        self.gripper = self.backend.gripper
        self.hand = (
            self.backend.hand
            if getattr(cfg, "BRAINCO_HAND_ENABLE", True)
            else None
        )

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
