"""Hardware-free RealMan high-follow executor tests."""

from __future__ import annotations

import math
import threading
import time
import unittest

from airo_doffy.config import RobotConfig
from airo_doffy.core import LifecycleError, ModelValidationError, RobotAction, RobotCommandType
from airo_doffy.robots import MockRobotBackend, RealManCanfdExecutor


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


class RealManCanfdExecutorTest(unittest.TestCase):
    def config(self, **overrides) -> RobotConfig:
        values = {
            "robot_type": "realman",
            "realman_control_rate_hz": 200,
            "realman_min_canfd_rate_hz": 100,
            "realman_rate_check_window_s": 0.03,
            "realman_rate_failure_windows": 2,
        }
        values.update(overrides)
        return RobotConfig(**values)

    def test_joint_slew_latest_rejection_hold_and_stop(self) -> None:
        backend = MockRobotBackend(dof=7)
        executor = RealManCanfdExecutor(backend, self.config())
        executor.start()
        self.assertTrue(
            executor.submit(action(2, RobotCommandType.JOINT_POSITION, (1.0,) * 7))
        )
        self.assertFalse(
            executor.submit(action(1, RobotCommandType.JOINT_POSITION, (0.5,) * 7))
        )
        executor.execute_once()
        first = backend.captured_actions[-1]
        self.assertEqual(first.values, (0.01,) * 7)
        executor.execute_once()
        second = backend.captured_actions[-1]
        self.assertEqual(second.values, (0.02,) * 7)

        executor.submit(action(3, RobotCommandType.HOLD))
        executor.execute_once()
        held = backend.captured_actions[-1]
        self.assertEqual(held.command_type, RobotCommandType.JOINT_POSITION)
        self.assertEqual(held.values, second.values)
        executor.submit(action(4, RobotCommandType.STOP))
        executor.execute_once()
        self.assertTrue(executor.snapshot().terminal_stop)
        with self.assertRaises(LifecycleError):
            executor.submit(action(5, RobotCommandType.HOLD))
        executor.close()

    def test_tcp_translation_and_rotation_are_slew_limited(self) -> None:
        backend = MockRobotBackend(dof=7)
        executor = RealManCanfdExecutor(backend, self.config(), control_mode="tcp")
        executor.start()
        c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
        target = (
            (c, -s, 0.0, 1.0),
            (s, c, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        executor.submit(
            action(
                1,
                RobotCommandType.TCP_POSE,
                tuple(value for row in target for value in row),
            )
        )
        executor.execute_once()
        sent = backend.captured_actions[-1]
        matrix = tuple(
            tuple(sent.values[row * 4 + column] for column in range(4))
            for row in range(4)
        )
        self.assertAlmostEqual(matrix[0][3], 0.25 / 200)
        self.assertAlmostEqual(matrix[1][0], math.sin(1.0 / 200))
        executor.request_stop()
        executor.close()

    def test_clean_run_reaches_greater_than_100_hz_gate(self) -> None:
        backend = MockRobotBackend(dof=7)
        executor = RealManCanfdExecutor(backend, self.config())
        executor.start()
        thread = threading.Thread(target=executor.run)
        thread.start()
        achieved = executor.wait_until_healthy(timeout=1.0)
        self.assertGreater(achieved, 100)
        executor.request_stop()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(executor.snapshot().ready)
        executor.close()

    def test_slow_backend_fails_timing_gate(self) -> None:
        backend = MockRobotBackend(dof=7, latency_s=0.012)
        executor = RealManCanfdExecutor(
            backend,
            self.config(realman_rate_failure_windows=1),
        )
        executor.start()
        failures: list[Exception] = []

        def run() -> None:
            try:
                executor.run()
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assertGreater(executor.snapshot().sdk_call_overruns, 0)
        with self.assertRaisesRegex(LifecycleError, "failed"):
            executor.wait_until_healthy(timeout=0.1)
        executor.close()

    def test_mode_and_lifecycle_validation(self) -> None:
        backend = MockRobotBackend(dof=6)
        with self.assertRaises(ModelValidationError):
            RealManCanfdExecutor(backend, self.config())
        backend7 = MockRobotBackend(dof=7)
        executor = RealManCanfdExecutor(backend7, self.config(), control_mode="joint")
        with self.assertRaises(ModelValidationError):
            executor.submit(action(1, RobotCommandType.TCP_POSE, (0.0,) * 16))
        with self.assertRaises(LifecycleError):
            executor.execute_once()
        executor.start()
        with self.assertRaises(LifecycleError):
            executor.start()
        executor.close()


if __name__ == "__main__":
    unittest.main()
