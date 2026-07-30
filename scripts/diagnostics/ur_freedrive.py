"""Enter UR teach mode and always leave it during shutdown."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--robot-type", choices=("ur3e", "ur5e"), required=True)
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="Acknowledge that this command controls a physical robot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.confirm_hardware:
        raise SystemExit("Refusing hardware access without --confirm-hardware.")

    from airo_robots.manipulators.hardware.ur_rtde import URrtde

    robot_config = URrtde.UR3E_CONFIG if args.robot_type == "ur3e" else URrtde.UR5E_CONFIG
    robot = URrtde(args.robot_ip, robot_config)
    teach_mode_started = False
    try:
        robot.rtde_control.servoStop()
        robot.rtde_control.teachMode()
        teach_mode_started = True
        input("Freedrive active. Press Enter to stop.")
    finally:
        if teach_mode_started:
            robot.rtde_control.endTeachMode()

    logger.info("Joint configuration: %s", robot.get_joint_configuration())
    logger.info("TCP pose: %s", robot.get_tcp_pose())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
