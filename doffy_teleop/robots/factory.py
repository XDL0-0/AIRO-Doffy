"""Lazy vendor factory for robot backends."""

from __future__ import annotations

import numpy as np

from .backends import RealManBackend, URPositionBackend, URTorqueBackend
from .contracts import RobotBackend
from .gripper import FastRobotiq2F85, NullGripper

def _ur_config_and_ik(robot_type: str, robot_cls):
    if robot_type == "ur3e":
        from ur_analytic_ik import ur3e as ik

        return robot_cls.UR3E_CONFIG, ik
    if robot_type == "ur5e":
        from ur_analytic_ik import ur5e as ik

        return robot_cls.UR5E_CONFIG, ik
    raise ValueError(f"Unsupported UR robot type: {robot_type}")


def make_robot_backend(cfg) -> RobotBackend:
    robot_type = cfg.ROBOT_TYPE.lower()
    robot_ip = cfg.ROBOT_IP
    if robot_ip is None and robot_type in {"ur3e", "ur5e"}:
        robot_ip = cfg.UR_IP
    if robot_ip is None:
        raise ValueError(
            f"ROBOT_IP must be configured when ROBOT_TYPE='{cfg.ROBOT_TYPE}'."
        )

    if robot_type in {"ur3e", "ur5e"}:
        if cfg.GRIPPER:
            gripper = FastRobotiq2F85(robot_ip)
            gripper.open()
        else:
            gripper = NullGripper(cfg.GRIPPER_MAX)
        if cfg.TORQUE_MODE:
            from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque

            robot_config, ik = _ur_config_and_ik(robot_type, URrtdeTorque)
            kwargs = {"initial_joint_configuration": np.asarray(cfg.INITIAL_JOINT, dtype=float)}
            if getattr(cfg, "RUCKIG_ENABLE", False):
                ruckig_params = {
                    "max_vel": cfg.RUCKIG_MAX_VEL,
                    "max_acc": cfg.RUCKIG_MAX_ACC,
                    "max_jerk": cfg.RUCKIG_MAX_JERK,
                }
                kwargs["ruckig_params"] = ruckig_params
            try:
                robot = URrtdeTorque(robot_ip, robot_config, **kwargs)
            except TypeError:
                kwargs.pop("ruckig_params", None)
                robot = URrtdeTorque(robot_ip, robot_config, **kwargs)
            return URTorqueBackend(cfg, robot, ik, gripper)

        from airo_robots.manipulators.hardware.ur_rtde import URrtde

        robot_config, ik = _ur_config_and_ik(robot_type, URrtde)
        robot = URrtde(robot_ip, robot_config)
        return URPositionBackend(cfg, robot, ik, gripper)

    if robot_type == "realman":
        from airo_robots.manipulators.hardware.realman import RealmanControl

        robot = RealmanControl(ip_address=robot_ip, port=cfg.REALMAN_PORT)
        return RealManBackend(cfg, robot)

    raise ValueError(
        f"Unsupported ROBOT_TYPE '{cfg.ROBOT_TYPE}'. Use 'ur3e', 'ur5e', or 'realman'."
    )


def make_robot(ur_ip: str, robot_type: str, torque_mode: bool, initial_joint=None, ruckig_params=None):
    """Backward-compatible factory returning the raw UR object and analytic IK module."""
    del ruckig_params
    if not torque_mode:
        from airo_robots.manipulators.hardware.ur_rtde import URrtde

        robot_cls = URrtde
    else:
        from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque

        robot_cls = URrtdeTorque

    robot_config, ik = _ur_config_and_ik(robot_type, robot_cls)
    kwargs = {}
    if torque_mode:
        kwargs["initial_joint_configuration"] = initial_joint
    return robot_cls(ur_ip, robot_config, **kwargs), ik
