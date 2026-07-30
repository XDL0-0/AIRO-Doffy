"""Tests for latest-command scheduling independent of hardware."""

from __future__ import annotations

import threading
import time
import unittest

from airo_doffy.core import LifecycleError, ModelValidationError, RobotAction, RobotCommandType
from airo_doffy.robots import InjectedRobotError, LatestActionExecutor, MockRobotBackend


def action(
    sequence: int,
    command_type: RobotCommandType,
    values=(),
) -> RobotAction:
    return RobotAction(
        sequence=sequence,
        source_timestamp_ns=sequence,
        command_type=command_type,
        values=values,
    )


class LatestActionExecutorTest(unittest.TestCase):
    def test_latest_only_rejection_and_active_command_repetition(self) -> None:
        backend = MockRobotBackend()
        executor = LatestActionExecutor(backend, target_hz=100)
        self.assertTrue(
            executor.submit(action(2, RobotCommandType.JOINT_POSITION, (0.2,) * 6))
        )
        self.assertFalse(
            executor.submit(action(1, RobotCommandType.JOINT_POSITION, (0.1,) * 6))
        )
        executor.start()
        self.assertTrue(executor.execute_once())
        self.assertTrue(executor.execute_once())
        snapshot = executor.snapshot()
        self.assertEqual(snapshot.latest_sequence, 2)
        self.assertEqual(snapshot.applied_count, 2)
        self.assertEqual(snapshot.rejected_count, 1)
        self.assertEqual(len(backend.captured_actions), 2)
        executor.close()

    def test_hold_and_stop_are_not_repeated(self) -> None:
        backend = MockRobotBackend()
        executor = LatestActionExecutor(backend, target_hz=100)
        executor.start()
        executor.submit(action(1, RobotCommandType.HOLD))
        self.assertTrue(executor.execute_once())
        self.assertFalse(executor.execute_once())
        executor.submit(action(2, RobotCommandType.JOINT_POSITION, (0.3,) * 6))
        self.assertTrue(executor.execute_once())
        executor.submit(action(3, RobotCommandType.STOP))
        self.assertTrue(executor.execute_once())
        self.assertFalse(executor.execute_once())
        self.assertTrue(executor.snapshot().terminal_stop)
        with self.assertRaises(LifecycleError):
            executor.submit(action(4, RobotCommandType.HOLD))
        executor.close()

    def test_failure_is_visible_to_health_check(self) -> None:
        backend = MockRobotBackend()
        executor = LatestActionExecutor(backend, target_hz=100)
        executor.start()
        backend.inject_failure("apply_action")
        executor.submit(action(1, RobotCommandType.HOLD))
        with self.assertRaises(InjectedRobotError):
            executor.execute_once()
        with self.assertRaisesRegex(LifecycleError, "executor failed"):
            executor.check_health()
        self.assertIn("InjectedRobotError", executor.snapshot().error)
        executor.close()

    def test_run_loop_is_caller_owned_and_stoppable(self) -> None:
        backend = MockRobotBackend()
        executor = LatestActionExecutor(backend, target_hz=200)
        executor.submit(action(1, RobotCommandType.JOINT_POSITION, (0.1,) * 6))
        executor.start()
        thread = threading.Thread(target=executor.run)
        thread.start()
        deadline = time.monotonic() + 1.0
        while len(backend.captured_actions) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        executor.request_stop()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(len(backend.captured_actions), 2)
        executor.close()

    def test_close_refuses_to_invalidate_running_backend(self) -> None:
        backend = MockRobotBackend()
        executor = LatestActionExecutor(backend, target_hz=10)
        executor.start()
        thread = threading.Thread(target=executor.run)
        thread.start()
        deadline = time.monotonic() + 1.0
        while not executor.snapshot().running and time.monotonic() < deadline:
            time.sleep(0.001)
        with self.assertRaisesRegex(LifecycleError, "join"):
            executor.close()
        self.assertFalse(backend.closed)
        executor.request_stop()
        thread.join(1.0)
        executor.close()

    def test_validation_and_lifecycle(self) -> None:
        backend = MockRobotBackend()
        with self.assertRaises(ModelValidationError):
            LatestActionExecutor(backend, target_hz=0)
        executor = LatestActionExecutor(backend, target_hz=100)
        with self.assertRaises(LifecycleError):
            executor.execute_once()
        executor.start()
        with self.assertRaises(LifecycleError):
            executor.start()
        executor.close()
        executor.close()


if __name__ == "__main__":
    unittest.main()
