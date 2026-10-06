"""RealMan vendor continuous QP inverse-kinematics adapter."""

from __future__ import annotations

from ctypes import c_float
from typing import Any

import numpy as np

import doffy_teleop.utils as utils

class RealManRemoteIkSolver:
    def __init__(
        self,
        arm: Any,
        api: Any,
        *,
        dof: int,
        period_s: float,
        initial_joints_radians: np.ndarray,
        joint_acceleration_limits_radians: np.ndarray,
        dq_weight: float,
        limit_holdon: bool,
        elbow_margin_degrees: float,
    ) -> None:
        required = (
            "rm_algo_ik_remote_init",
            "rm_algo_set_error_weight",
            "rm_algo_set_dq_weight",
            "rm_algo_ik_remote",
        )
        missing = [name for name in required if not callable(getattr(arm, name, None))]
        if missing:
            raise TypeError(
                "RealMan SDK is missing teleoperation solver method(s): "
                + ", ".join(missing)
            )
        matrix_type = getattr(api, "rm_Mat_t", None)
        if matrix_type is None:
            raise TypeError("RealMan SDK does not expose rm_Mat_t.")

        self.arm = arm
        self.api = api
        self.dof = int(dof)
        self._matrix_type = matrix_type
        self._output_type = c_float * self.dof
        self.last_status = 0

        # Targets in this application are absolute poses in the robot work/base
        # frame, hence tool_or_work=1. dT must match the fixed solver cadence.
        self.arm.rm_algo_ik_remote_init(float(period_s), 1)
        self.arm.rm_algo_set_error_weight([1.0] * 6)
        self.arm.rm_algo_set_dq_weight([float(dq_weight)] * self.dof)
        set_joint_max_acc = getattr(self.arm, "rm_algo_set_joint_max_acc", None)
        if callable(set_joint_max_acc):
            acceleration_limits = np.asarray(
                joint_acceleration_limits_radians,
                dtype=float,
            )
            if (
                acceleration_limits.shape != (self.dof,)
                or np.any(acceleration_limits <= 0.0)
            ):
                raise ValueError(
                    "joint_acceleration_limits_radians must be positive with "
                    f"shape ({self.dof},)."
                )
            # RealMan's algorithm API uses RPM/s; the rest of this application
            # uses SI radians/s^2.
            set_joint_max_acc(
                (acceleration_limits * 60.0 / (2.0 * np.pi)).tolist()
            )
        set_limit_holdon = getattr(self.arm, "rm_algo_set_enable_limit_holdon", None)
        if callable(set_limit_holdon):
            set_limit_holdon(int(limit_holdon))

        self._configure_nonzero_elbow_limit(
            np.asarray(initial_joints_radians, dtype=float),
            float(elbow_margin_degrees),
        )

    @staticmethod
    def is_available(arm: Any, api: Any) -> bool:
        return (
            api is not None
            and getattr(api, "rm_Mat_t", None) is not None
            and callable(getattr(arm, "rm_algo_ik_remote_init", None))
            and callable(getattr(arm, "rm_algo_ik_remote", None))
        )

    def _configure_nonzero_elbow_limit(
        self,
        initial_joints_radians: np.ndarray,
        margin_degrees: float,
    ) -> None:
        set_limit = getattr(self.arm, "rm_algo_set_joint_limit_angle", None)
        if not callable(set_limit):
            utils.logger.warning(
                "RealMan SDK cannot configure the QP elbow limit; configure a "
                "nonzero elbow soft limit on the teach pendant."
            )
            return

        elbow_index = 3 if self.dof == 7 else 2
        if initial_joints_radians.shape != (self.dof,):
            raise ValueError(f"Expected {self.dof} initial RealMan joint angles.")
        initial_elbow_degrees = float(np.degrees(initial_joints_radians[elbow_index]))
        if abs(initial_elbow_degrees) <= margin_degrees:
            raise ValueError(
                "Initial RealMan elbow is too close to the QP singular boundary: "
                f"{initial_elbow_degrees:.2f} degrees. Move it beyond "
                f"+/-{margin_degrees:g} degrees before teleoperation."
            )

        dof_type = (
            self.api.rm_dofType_e.DOF_TYPE_7
            if self.dof == 7
            else self.api.rm_dofType_e.DOF_TYPE_6
        )
        joint = (
            self.api.rm_jointType_e.JOINT_Q4
            if self.dof == 7
            else self.api.rm_jointType_e.JOINT_Q3
        )
        if initial_elbow_degrees < 0.0:
            limit_type = self.api.rm_limitType_e.LIMIT_MAX
            limit_angle = -margin_degrees
        else:
            limit_type = self.api.rm_limitType_e.LIMIT_MIN
            limit_angle = margin_degrees
        result = set_limit(dof_type, joint, limit_type, limit_angle)
        if result != 0:
            raise RuntimeError(
                "rm_algo_set_joint_limit_angle failed with RealMan error code "
                f"{result}."
            )

    def solve(
        self,
        tcp_target: np.ndarray,
        current_joints_radians: np.ndarray,
    ) -> np.ndarray | None:
        tcp_target = np.asarray(tcp_target, dtype=float)
        current = np.asarray(current_joints_radians, dtype=float)
        if tcp_target.shape != (4, 4) or not np.all(np.isfinite(tcp_target)):
            raise ValueError("Expected a finite 4x4 QP TCP target.")
        if current.shape != (self.dof,) or not np.all(np.isfinite(current)):
            raise ValueError(f"Expected {self.dof} finite current joint angles.")

        matrix = self._matrix_type(row=4, col=4, data=tcp_target.tolist())
        q_out = self._output_type()
        self.last_status = int(
            self.arm.rm_algo_ik_remote(
                matrix,
                np.degrees(current).tolist(),
                q_out,
            )
        )
        if self.last_status != 0:
            return None
        result = np.radians(np.asarray(list(q_out), dtype=float))
        if result.shape != (self.dof,) or not np.all(np.isfinite(result)):
            return None
        return result
