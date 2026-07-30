"""Manually exercise UR force mode with explicit hardware confirmation."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--robot-type", choices=("ur3e", "ur5e"), required=True)
    parser.add_argument("--force-z", type=float, default=8.0)
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="Acknowledge that this command connects to and moves a physical robot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.confirm_hardware:
        raise SystemExit("Refusing hardware access without --confirm-hardware.")

    import numpy as np
    from airo_robots.manipulators import URrtde
    from airo_spatial_algebra.se3 import SE3Container

    from config import Config

    robot_config = URrtde.UR3E_CONFIG if args.robot_type == "ur3e" else URrtde.UR5E_CONFIG
    cfg = Config()
    robot = URrtde(args.robot_ip, robot_config)
    force_mode_started = False
    try:
        robot.move_to_joint_configuration(cfg.INITIAL_JOINT, 0.3).wait()
        logger.info("Initialization complete.")
        task_pose = robot.get_tcp_pose()
        pose = SE3Container.from_homogeneous_matrix(task_pose)
        task_frame = np.concatenate([pose.translation, pose.orientation_as_euler_angles])
        logger.info("Task frame: %s", task_frame)
        logger.info("TCP force: %s", robot.get_tcp_force())
        selection_vector = [1, 1, 0, 0, 0, 0]
        limits = [0.03, 0.03, 0.015, 0.15, 0.15, 0.15]
        robot.rtde_control.forceMode(
            task_frame,
            selection_vector,
            [0, 0, args.force_z, 0, 0, 0],
            2,
            limits,
        )
        force_mode_started = True
        input("Press Enter to stop force mode.")
    finally:
        if force_mode_started:
            robot.rtde_control.forceModeStop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
