"""Tests for fixed, random, periodic, disconnected, and delayed tactile mocks."""

from __future__ import annotations

import math
import unittest

from airo_doffy.core import LifecycleError, ModelValidationError
from airo_doffy.devices.tactile import (
    MockTactileSensor,
    TactileMockMode,
    TactileSensor,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def now_ns(self) -> int:
        return self.value


class MockTactileSensorTest(unittest.TestCase):
    def test_fixed_lifecycle_sequence_and_recalibration(self) -> None:
        values = tuple((1.0, 2.0, 3.0) for _ in range(4))
        sensor = MockTactileSensor(fixed_values=values)
        self.assertIsInstance(sensor, TactileSensor)
        with self.assertRaises(LifecycleError):
            sensor.read_latest()
        sensor.start()
        first = sensor.read_latest()
        second = sensor.read_latest()
        self.assertEqual(first.values, values)
        self.assertEqual((first.sequence, second.sequence), (0, 1))
        sensor.recalibrate()
        self.assertEqual(sensor.recalibration_count, 1)
        sensor.close()
        sensor.close()
        with self.assertRaises(LifecycleError):
            sensor.read_latest()

    def test_random_mode_is_seeded_and_bounded(self) -> None:
        first = MockTactileSensor(mode="random", amplitude=2, random_seed=7)
        second = MockTactileSensor(mode="random", amplitude=2, random_seed=7)
        first.start()
        second.start()
        first_values = first.read_latest().values
        self.assertEqual(first_values, second.read_latest().values)
        self.assertTrue(all(-2 <= value <= 2 for row in first_values for value in row))
        first.close()
        second.close()

    def test_periodic_mode_uses_elapsed_monotonic_time(self) -> None:
        clock = _Clock()
        sensor = MockTactileSensor(
            mode=TactileMockMode.PERIODIC,
            amplitude=3,
            frequency_hz=1,
            clock=clock,
        )
        sensor.start()
        clock.value = 250_000_000
        sample = sensor.read_latest()
        self.assertAlmostEqual(sample.values[0][0], 3.0)
        clock.value = 500_000_000
        self.assertAlmostEqual(sensor.read_latest().values[0][0], 0.0, places=12)
        sensor.close()

    def test_disconnected_and_delayed_modes_return_none(self) -> None:
        disconnected = MockTactileSensor(mode="disconnected")
        disconnected.start()
        self.assertIsNone(disconnected.read_latest())
        disconnected.close()

        clock = _Clock()
        delayed = MockTactileSensor(
            mode="delayed",
            fixed_values=tuple((4.0, 5.0, 6.0) for _ in range(4)),
            connection_delay_s=1.0,
            clock=clock,
        )
        delayed.start()
        clock.value = 999_999_999
        self.assertIsNone(delayed.read_latest())
        clock.value = 1_000_000_000
        self.assertIsNotNone(delayed.read_latest())
        delayed.set_disconnected(True)
        self.assertIsNone(delayed.read_latest())
        delayed.set_disconnected(False)
        self.assertIsNotNone(delayed.read_latest())
        delayed.close()

    def test_validation(self) -> None:
        with self.assertRaises(ModelValidationError):
            MockTactileSensor(mode="unknown")
        with self.assertRaises(ModelValidationError):
            MockTactileSensor(amplitude=math.inf)
        with self.assertRaises(ModelValidationError):
            MockTactileSensor(fixed_values=((1, 2, 3),))


if __name__ == "__main__":
    unittest.main()
