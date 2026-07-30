"""Numerical and compatibility tests for pure wrench filtering."""

from __future__ import annotations

import unittest

import numpy as np

from airo_doffy.devices.wrench import WrenchFilter
from force_filter import WrenchFilter as LegacyWrenchFilter


class WrenchFilterTest(unittest.TestCase):
    def test_short_input_returns_zero_without_initializing(self) -> None:
        wrench_filter = WrenchFilter(moving_average_window=8, low_pass_alpha=0.15)
        self.assertEqual(wrench_filter.process((1, 2, 3)), (0.0,) * 6)
        self.assertFalse(wrench_filter.initialized)

    def test_deadband_subtracts_threshold_and_preserves_order(self) -> None:
        wrench_filter = WrenchFilter(force_deadband=2, torque_deadband=0.5)
        self.assertEqual(
            wrench_filter.process((1, 2, 3, -0.4, -0.5, -1.5)),
            (0.0, 0.0, 1.0, 0.0, 0.0, -1.0),
        )

    def test_moving_average_precedes_low_pass(self) -> None:
        wrench_filter = WrenchFilter(moving_average_window=2, low_pass_alpha=0.25)
        self.assertEqual(wrench_filter.process((1,) * 6), (1.0,) * 6)
        self.assertEqual(wrench_filter.process((3,) * 6), (1.25,) * 6)
        self.assertEqual(wrench_filter.process((5,) * 6), (1.9375,) * 6)
        wrench_filter.reset()
        self.assertEqual(wrench_filter.process((5,) * 6), (5.0,) * 6)

    def test_extra_values_are_truncated(self) -> None:
        wrench_filter = WrenchFilter()
        self.assertEqual(wrench_filter.process(range(10)), tuple(float(i) for i in range(6)))

    def test_root_compatibility_adapter_returns_numpy(self) -> None:
        wrench_filter = LegacyWrenchFilter(
            moving_average_window=2,
            low_pass_alpha=0.5,
            force_deadband=1,
            torque_deadband=0.25,
        )
        first = wrench_filter.process(np.array([2, -2, 0.5, 1, -1, 0.1]))
        second = wrench_filter.process(np.array([4, -4, 1.5, 2, -2, 0.5]))
        self.assertIsInstance(first, np.ndarray)
        np.testing.assert_allclose(first, [1, -1, 0, 0.75, -0.75, 0])
        np.testing.assert_allclose(second, [1.5, -1.5, 0.125, 1, -1, 0.0625])


if __name__ == "__main__":
    unittest.main()
