"""Hardware-free behavior tests for the mock robot backend."""

from __future__ import annotations

import unittest

from airo_doffy.core import (
    ClockDomain,
    LifecycleError,
    ModelValidationError,
    RobotAction,
    RobotCommandType,
    RobotState,
)
from airo_doffy.robots import InjectedRobotError, MockRobotBackend, RobotBackend

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class _Clock:
    def __init__(self) -> None:
        self.value = 100

    def now_ns(self) -> int:
        self.value += 1
        return self.value


def action(
    command_type: RobotCommandType,
    values=(),
    *,
    sequence: int = 0,
    duration_s: float | None = None,
    gripper_width_m: float | None = None,
) -> RobotAction:
    return RobotAction(
        sequence=sequence,
        source_timestamp_ns=sequence,
        command_type=command_type,
        values=values,
        duration_s=duration_s,
        gripper_width_m=gripper_width_m,
    )


class MockRobotBackendTest(unittest.TestCase):
    def test_lifecycle_and_protocol(self) -> None:
        robot = MockRobotBackend()
        self.assertIsInstance(robot, RobotBackend)
        with self.assertRaises(LifecycleError):
            robot.read_state()
        robot.start()
        self.assertTrue(robot.started)
        with self.assertRaises(LifecycleError):
            robot.start()
        robot.close()
        robot.close()
        self.assertTrue(robot.closed)
        with self.assertRaises(LifecycleError):
            robot.read_state()

    def test_configurable_state_and_position_commands(self) -> None:
        clock = _Clock()
        initial = RobotState(
            sequence=7,
            source_timestamp_ns=50,
            clock_domain=ClockDomain.MONOTONIC,
            joints_rad=(1.0,) * 7,
            tcp_pose=IDENTITY,
            gripper_width_m=0.08,
            wrench=(1, 2, 3, 4, 5, 6),
        )
        robot = MockRobotBackend(dof=7, initial_state=initial, clock=clock)
        robot.start()
        joint_action = action(
            RobotCommandType.JOINT_POSITION,
            (0.1,) * 7,
            sequence=1,
            gripper_width_m=0.02,
        )
        robot.apply_action(joint_action)
        state = robot.read_state()
        self.assertEqual(state.sequence, 8)
        self.assertEqual(state.joints_rad, (0.1,) * 7)
        self.assertEqual(state.gripper_width_m, 0.02)
        self.assertEqual(state.wrench, initial.wrench)

        pose = tuple(float(index) for index in range(16))
        tcp_action = action(RobotCommandType.TCP_POSE, pose, sequence=2)
        robot.apply_action(tcp_action)
        self.assertEqual(robot.read_state().tcp_pose[2], (8.0, 9.0, 10.0, 11.0))
        self.assertEqual(robot.captured_actions, (joint_action, tcp_action))
        robot.clear_captured_actions()
        self.assertEqual(robot.captured_actions, ())

    def test_velocity_hold_and_stop(self) -> None:
        robot = MockRobotBackend()
        robot.start()
        robot.apply_action(
            action(RobotCommandType.JOINT_VELOCITY, (2.0,) * 6, duration_s=0.25)
        )
        self.assertEqual(robot.read_state().joints_rad, (0.5,) * 6)
        robot.apply_action(action(RobotCommandType.HOLD, sequence=1))
        self.assertTrue(robot.holding)
        robot.apply_action(action(RobotCommandType.JOINT_POSITION, (1.0,) * 6, sequence=2))
        self.assertFalse(robot.holding)
        robot.apply_action(action(RobotCommandType.STOP, sequence=3))
        self.assertTrue(robot.stopped)
        with self.assertRaises(LifecycleError):
            robot.apply_action(
                action(RobotCommandType.JOINT_POSITION, (0.0,) * 6, sequence=4)
            )

    def test_latency_and_queued_failures(self) -> None:
        sleeps: list[float] = []
        robot = MockRobotBackend(latency_s=0.125, sleep=sleeps.append)
        robot.inject_failure("start", count=2)
        with self.assertRaises(InjectedRobotError):
            robot.start()
        with self.assertRaises(InjectedRobotError):
            robot.start()
        robot.start()
        robot.inject_failure("read_state", RuntimeError("sensor timeout"))
        with self.assertRaisesRegex(RuntimeError, "sensor timeout"):
            robot.read_state()
        robot.inject_failure("apply_action")
        with self.assertRaises(InjectedRobotError):
            robot.apply_action(action(RobotCommandType.HOLD))
        self.assertEqual(robot.captured_actions, ())
        self.assertEqual(sleeps, [0.125] * 5)

    def test_close_failure_still_closes_backend(self) -> None:
        robot = MockRobotBackend()
        robot.start()
        robot.inject_failure("close", RuntimeError("cleanup failure"))
        with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
            robot.close()
        self.assertTrue(robot.closed)
        robot.close()

    def test_validation(self) -> None:
        with self.assertRaises(ModelValidationError):
            MockRobotBackend(dof=5)
        with self.assertRaises(ModelValidationError):
            MockRobotBackend(latency_s=-1)
        robot = MockRobotBackend(dof=6)
        wrong_dof = RobotState(
            sequence=0,
            source_timestamp_ns=0,
            joints_rad=(0.0,) * 7,
            tcp_pose=IDENTITY,
        )
        with self.assertRaisesRegex(ModelValidationError, "expected 6"):
            robot.set_state(wrong_dof)
        with self.assertRaises(ModelValidationError):
            robot.inject_failure("unknown")


if __name__ == "__main__":
    unittest.main()
