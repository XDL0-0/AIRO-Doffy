"""Hardware-free tests for the RealMan RM75 adapter."""

from __future__ import annotations

import math
import sys
import unittest

from airo_doffy.config import RobotConfig
from airo_doffy.core import LifecycleError, ModelValidationError, RobotAction, RobotCommandType
from airo_doffy.robots import RealManRobotBackend, RobotBackend

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
) -> RobotAction:
    return RobotAction(
        sequence=sequence,
        source_timestamp_ns=sequence,
        command_type=command_type,
        values=values,
    )


class _Arm:
    def __init__(self) -> None:
        self.joint_calls: list[tuple] = []
        self.tcp_calls: list[tuple] = []
        self.stop_count = 0
        self.result = 0

    def rm_movej_canfd(self, joints, follow, expand, trajectory_mode, radio):
        self.joint_calls.append((joints, follow, expand, trajectory_mode, radio))
        return self.result

    def rm_movep_canfd(self, pose, follow, trajectory_mode, radio):
        self.tcp_calls.append((pose, follow, trajectory_mode, radio))
        return self.result

    def rm_get_force_data(self):
        return 0, {
            "force_data": [9] * 6,
            "zero_force_data": [1, 2, 3, 4, 5, 6],
        }

    def rm_set_arm_stop(self):
        self.stop_count += 1
        return 0


class _Controller:
    def __init__(self) -> None:
        self.robot = _Arm()
        self.close_count = 0
        self.joint_failures = 0

    def get_joint_configuration(self):
        if self.joint_failures:
            self.joint_failures -= 1
            raise RuntimeError("rm_get_joint_degree failed with error code -2")
        return [0.1] * 7

    def get_tcp_pose(self):
        return IDENTITY

    def close(self) -> None:
        self.close_count += 1


class RealManRobotBackendTest(unittest.TestCase):
    def test_import_does_not_load_vendor_sdk(self) -> None:
        self.assertNotIn("airo_robots.manipulators.hardware.realman", sys.modules)

    def test_lifecycle_state_retry_and_wrench_priority(self) -> None:
        controller = _Controller()
        controller.joint_failures = 2
        sleeps: list[float] = []
        backend = RealManRobotBackend(
            RobotConfig(robot_type="realman"),
            controller=controller,
            sleep=sleeps.append,
        )
        self.assertIsInstance(backend, RobotBackend)
        with self.assertRaises(LifecycleError):
            backend.read_state()
        backend.start()
        state = backend.read_state()
        self.assertEqual(state.joints_rad, (0.1,) * 7)
        self.assertEqual(state.tcp_pose, IDENTITY)
        self.assertEqual(state.wrench, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual(sleeps, [0.05, 0.05])
        backend.close()
        backend.close()
        self.assertEqual(controller.close_count, 1)

    def test_joint_tcp_hold_and_stop_translation(self) -> None:
        controller = _Controller()
        backend = RealManRobotBackend(
            RobotConfig(
                robot_type="realman",
                realman_canfd_trajectory_mode=1,
                realman_canfd_radio=50,
            ),
            controller=controller,
        )
        backend.start()
        backend.apply_action(action(RobotCommandType.JOINT_POSITION, (math.pi / 2,) * 7))
        joints, follow, expand, mode, radio = controller.robot.joint_calls[-1]
        self.assertEqual(joints, [90.0] * 7)
        self.assertEqual((follow, expand, mode, radio), (True, 0, 1, 50))
        backend.apply_action(action(RobotCommandType.HOLD, sequence=1))
        self.assertEqual(len(controller.robot.joint_calls), 2)

        c = math.cos(math.pi / 2)
        s = math.sin(math.pi / 2)
        pose = (
            (c, -s, 0.0, 0.4),
            (s, c, 0.0, 0.5),
            (0.0, 0.0, 1.0, 0.6),
            (0.0, 0.0, 0.0, 1.0),
        )
        backend.apply_action(
            action(
                RobotCommandType.TCP_POSE,
                tuple(value for row in pose for value in row),
                sequence=2,
            )
        )
        values, follow, mode, radio = controller.robot.tcp_calls[-1]
        self.assertEqual(values[:3], [0.4, 0.5, 0.6])
        self.assertAlmostEqual(values[5], math.pi / 2)
        self.assertEqual((follow, mode, radio), (True, 1, 50))
        backend.apply_action(action(RobotCommandType.STOP, sequence=3))
        self.assertEqual(controller.robot.stop_count, 1)
        with self.assertRaises(LifecycleError):
            backend.apply_action(action(RobotCommandType.HOLD, sequence=4))
        backend.close()

    def test_sdk_error_and_unsupported_modes(self) -> None:
        controller = _Controller()
        backend = RealManRobotBackend(
            RobotConfig(robot_type="realman"),
            controller=controller,
        )
        backend.start()
        with self.assertRaisesRegex(LifecycleError, "established setpoint"):
            backend.apply_action(action(RobotCommandType.HOLD))
        with self.assertRaisesRegex(LifecycleError, "velocity modes"):
            backend.apply_action(
                action(RobotCommandType.JOINT_VELOCITY, (0.0,) * 7, sequence=1)
            )
        controller.robot.result = -3
        with self.assertRaisesRegex(RuntimeError, "error code -3"):
            backend.apply_action(
                action(RobotCommandType.JOINT_POSITION, (0.0,) * 7, sequence=2)
            )
        backend.close()

    def test_validation_and_missing_ip(self) -> None:
        with self.assertRaises(ModelValidationError):
            RealManRobotBackend(RobotConfig(robot_type="ur3e"))
        with self.assertRaises(ModelValidationError):
            RealManRobotBackend(
                RobotConfig(robot_type="realman"),
                controller=object(),
                controller_factory=lambda config: object(),
            )
        backend = RealManRobotBackend(RobotConfig(robot_type="realman", ip=None))
        with self.assertRaisesRegex(LifecycleError, "robot.ip"):
            backend.start()
        backend.close()


if __name__ == "__main__":
    unittest.main()
