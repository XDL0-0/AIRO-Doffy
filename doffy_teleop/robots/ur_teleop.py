"""Backward-compatible imports for the renamed robot teleop module."""

from doffy_teleop.robots.teleop import FastRobotiq2F85, RobotTeleop, URTeleop, make_robot

__all__ = ["FastRobotiq2F85", "RobotTeleop", "URTeleop", "make_robot"]
