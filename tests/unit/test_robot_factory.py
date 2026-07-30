"""Tests for dependency-free robot composition routing."""

from __future__ import annotations

import unittest

from airo_doffy.config import RobotConfig, RobotFactory
from airo_doffy.core import LifecycleError, ModelValidationError
from airo_doffy.robots import (
    RealManRobotBackend,
    URRobotBackend,
    create_gripper,
    create_robot_backend,
)
from airo_doffy.robots.grippers import NullGripper, Robotiq2F85Gripper


class RobotFactoryTest(unittest.TestCase):
    def test_routes_vendor_backends_without_connecting(self) -> None:
        ur = create_robot_backend(RobotConfig(robot_type="ur3e"))
        realman = create_robot_backend(RobotConfig(robot_type="realman"))
        self.assertIsInstance(ur, URRobotBackend)
        self.assertIsNone(ur.manipulator)
        self.assertIsInstance(realman, RealManRobotBackend)
        self.assertIsNone(realman.controller)

    def test_config_lazy_factory_can_target_composition_function(self) -> None:
        factory = RobotFactory(
            target="airo_doffy.robots.factory:create_robot_backend"
        )
        backend = factory.create(RobotConfig(robot_type="ur5e"))
        self.assertIsInstance(backend, URRobotBackend)
        self.assertEqual(backend.name, "ur5e")

    def test_gripper_selection_is_separate(self) -> None:
        disabled = create_gripper(RobotConfig(gripper_enabled=False))
        self.assertIsInstance(disabled, NullGripper)
        enabled = create_gripper(
            RobotConfig(
                robot_type="ur3e",
                ip="192.0.2.1",
                gripper_enabled=True,
            )
        )
        self.assertIsInstance(enabled, Robotiq2F85Gripper)
        with self.assertRaises(ModelValidationError):
            create_gripper(
                RobotConfig(
                    robot_type="realman",
                    ip="192.0.2.2",
                    gripper_enabled=True,
                )
            )

    def test_enabled_gripper_requires_deployment_ip(self) -> None:
        with self.assertRaises(LifecycleError):
            create_gripper(RobotConfig(gripper_enabled=True, ip=None))


if __name__ == "__main__":
    unittest.main()
