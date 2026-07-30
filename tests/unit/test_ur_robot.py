"""Hardware-free tests for the UR RTDE adapter."""

from __future__ import annotations

import sys
import unittest

from airo_doffy.config import RobotConfig
from airo_doffy.core import (
    LifecycleError,
    ModelValidationError,
    OptionalDependencyError,
    RobotAction,
    RobotCommandType,
)
from airo_doffy.robots import RobotBackend, URRobotBackend

IDENTITY = (
    (1.0, 0.0, 0.0, 0.1),
    (0.0, 1.0, 0.0, 0.2),
    (0.0, 0.0, 1.0, 0.3),
    (0.0, 0.0, 0.0, 1.0),
)


def action(
    command_type: RobotCommandType,
    values=(),
    *,
    sequence: int = 0,
    duration_s: float | None = None,
) -> RobotAction:
    return RobotAction(
        sequence=sequence,
        source_timestamp_ns=sequence,
        command_type=command_type,
        values=values,
        duration_s=duration_s,
    )


class _Control:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def servoStop(self) -> None:
        self.calls.append("servoStop")

    def stopJ(self) -> None:
        self.calls.append("stopJ")


class _PositionManipulator:
    def __init__(self) -> None:
        self.rtde_control = _Control()
        self.joints = [0.0] * 6
        self.pose = IDENTITY
        self.wrench = [1, 2, 3, 4, 5, 6]
        self.calls: list[tuple] = []
        self.close_count = 0

    def get_joint_configuration(self):
        return self.joints

    def get_tcp_pose(self):
        return self.pose

    def get_tcp_force(self):
        return self.wrench

    def servo_to_joint_configuration(self, target, duration) -> None:
        self.calls.append(("joint", tuple(target), duration))
        self.joints = list(target)

    def servo_to_tcp_pose(self, target, duration) -> None:
        self.calls.append(("tcp", tuple(tuple(row) for row in target), duration))
        self.pose = tuple(tuple(row) for row in target)

    def servo_to_joint_velocity(self, target, duration) -> None:
        self.calls.append(("joint_velocity", tuple(target), duration))

    def servo_to_tcp_velocity(self, target, duration) -> None:
        self.calls.append(("tcp_velocity", tuple(target), duration))

    def close(self) -> None:
        self.close_count += 1


class _TorqueManipulator:
    def __init__(self) -> None:
        self.target_pos = None
        self.disabled = 0
        self.close_count = 0

    def get_cached_joint_configuration(self):
        return [0.0] * 6

    def get_cached_tcp_pose(self):
        return IDENTITY

    def get_cached_tcp_force(self):
        return [0.0] * 6

    def disable_torque_control(self) -> None:
        self.disabled += 1

    def close(self) -> None:
        self.close_count += 1


class URRobotBackendTest(unittest.TestCase):
    def test_import_does_not_load_vendor_sdk(self) -> None:
        self.assertNotIn("airo_robots.manipulators.hardware.ur_rtde", sys.modules)

    def test_position_lifecycle_state_and_actions(self) -> None:
        manipulator = _PositionManipulator()
        backend = URRobotBackend(RobotConfig(robot_type="ur3e"), manipulator=manipulator)
        self.assertIsInstance(backend, RobotBackend)
        self.assertEqual(backend.name, "ur3e")
        with self.assertRaises(LifecycleError):
            backend.read_state()
        backend.start()
        state = backend.read_state()
        self.assertEqual(state.sequence, 0)
        self.assertEqual(state.joints_rad, (0.0,) * 6)
        self.assertEqual(state.tcp_pose, IDENTITY)
        self.assertEqual(state.wrench, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

        backend.apply_action(
            action(
                RobotCommandType.JOINT_POSITION,
                (0.1,) * 6,
                duration_s=0.02,
            )
        )
        pose_values = tuple(value for row in IDENTITY for value in row)
        backend.apply_action(
            action(RobotCommandType.TCP_POSE, pose_values, sequence=1, duration_s=0.03)
        )
        backend.apply_action(
            action(
                RobotCommandType.JOINT_VELOCITY,
                (0.2,) * 6,
                sequence=2,
                duration_s=0.04,
            )
        )
        backend.apply_action(
            action(
                RobotCommandType.TCP_TWIST,
                (0.3,) * 6,
                sequence=3,
                duration_s=0.05,
            )
        )
        backend.apply_action(action(RobotCommandType.HOLD, sequence=4))
        self.assertEqual(manipulator.rtde_control.calls, ["servoStop"])
        backend.apply_action(action(RobotCommandType.STOP, sequence=5))
        self.assertEqual(manipulator.rtde_control.calls, ["servoStop", "stopJ"])
        with self.assertRaises(LifecycleError):
            backend.apply_action(
                action(RobotCommandType.JOINT_POSITION, (0.0,) * 6, sequence=6)
            )
        backend.close()
        backend.close()
        self.assertEqual(manipulator.close_count, 1)

    def test_torque_mode_uses_cached_state_and_shared_target(self) -> None:
        manipulator = _TorqueManipulator()
        ik_calls: list[tuple] = []

        def solve(pose, seed):
            ik_calls.append((pose, seed))
            return (0.6,) * 6

        backend = URRobotBackend(
            RobotConfig(robot_type="ur5e", torque_mode=True),
            manipulator=manipulator,
            inverse_kinematics=solve,
        )
        backend.start()
        self.assertEqual(backend.name, "ur5e_torque")
        self.assertEqual(backend.read_state().wrench, (0.0,) * 6)
        backend.apply_action(action(RobotCommandType.JOINT_POSITION, (0.4,) * 6))
        self.assertEqual(tuple(manipulator.target_pos), (0.4,) * 6)
        backend.apply_action(
            action(RobotCommandType.TCP_POSE, tuple(value for row in IDENTITY for value in row))
        )
        self.assertEqual(tuple(manipulator.target_pos), (0.6,) * 6)
        self.assertEqual(ik_calls[0][1], (0.0,) * 6)
        backend.close()
        self.assertEqual(manipulator.disabled, 1)
        self.assertEqual(manipulator.close_count, 1)

    def test_validation_and_missing_configuration(self) -> None:
        with self.assertRaises(ModelValidationError):
            URRobotBackend(RobotConfig(robot_type="realman"))
        with self.assertRaises(ModelValidationError):
            URRobotBackend(
                RobotConfig(),
                manipulator=object(),
                manipulator_factory=lambda config: object(),
            )
        backend = URRobotBackend(RobotConfig(ip=None))
        with self.assertRaisesRegex(LifecycleError, "robot.ip"):
            backend.start()
        backend.close()

    def test_factory_import_error_is_domain_specific(self) -> None:
        def missing(_config):
            raise OptionalDependencyError("missing sdk")

        backend = URRobotBackend(RobotConfig(ip="192.0.2.1"), manipulator_factory=missing)
        with self.assertRaises(OptionalDependencyError):
            backend.start()
        backend.close()

    def test_velocity_requires_duration(self) -> None:
        backend = URRobotBackend(RobotConfig(), manipulator=_PositionManipulator())
        backend.start()
        with self.assertRaisesRegex(ModelValidationError, "duration_s"):
            backend.apply_action(action(RobotCommandType.JOINT_VELOCITY, (0.0,) * 6))
        backend.close()


if __name__ == "__main__":
    unittest.main()
