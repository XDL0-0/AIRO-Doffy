"""Numerical golden tests for the extracted BLE4 signal filter."""

from __future__ import annotations

import unittest

from airo_doffy.config import TactileConfig
from airo_doffy.core import ModelValidationError
from airo_doffy.devices.tactile.filters import Ble4SignalFilter


def uniform(value: float):
    return tuple((value, value, value) for _ in range(4))


class Ble4SignalFilterTest(unittest.TestCase):
    def test_calibration_uses_median_and_robust_deadband(self) -> None:
        signal_filter = Ble4SignalFilter(
            TactileConfig(noise_floor=2.0, deadband_sigma=3.0)
        )
        signal_filter.calibrate((uniform(10), uniform(12), uniform(100)))
        self.assertEqual(signal_filter.baseline, uniform(12))
        expected_deadband = 3.0 * 2.0 * 1.4826
        self.assertAlmostEqual(signal_filter.deadband[0][0], expected_deadband)

    def test_deadband_clip_delta_then_ema_order(self) -> None:
        signal_filter = Ble4SignalFilter(
            TactileConfig(
                filter_alpha=0.5,
                noise_floor=2.0,
                deadband_sigma=1.0,
                max_abs=20.0,
                max_delta=4.0,
            )
        )
        signal_filter.calibrate((uniform(10),))
        first = signal_filter.process(uniform(13))
        self.assertEqual(first, uniform(3))
        second = signal_filter.process(uniform(100))
        self.assertEqual(second, uniform(5))
        third = signal_filter.process(uniform(11))
        self.assertEqual(third, uniform(3))

    def test_baseline_drift_tracks_only_unloaded_samples(self) -> None:
        signal_filter = Ble4SignalFilter(
            TactileConfig(
                baseline_drift_alpha=0.5,
                baseline_drift_threshold=10.0,
                noise_floor=0.1,
            )
        )
        signal_filter.calibrate((uniform(0),))
        signal_filter.process(uniform(2))
        self.assertEqual(signal_filter.baseline, uniform(1))
        signal_filter.process(uniform(100))
        self.assertEqual(signal_filter.baseline, uniform(1))

    def test_kalman_state_and_reset(self) -> None:
        signal_filter = Ble4SignalFilter(
            TactileConfig(
                filter_alpha=1.0,
                use_kalman=True,
                kalman_q=0.02,
                kalman_r=0.02,
                noise_floor=0.1,
            )
        )
        signal_filter.calibrate((uniform(0),))
        self.assertEqual(signal_filter.process(uniform(1)), uniform(1))
        second = signal_filter.process(uniform(3))
        gain = 1.02 / 1.04
        self.assertAlmostEqual(second[0][0], 1 + gain * 2)
        signal_filter.reset()
        self.assertEqual(signal_filter.process(uniform(4)), uniform(4))
        self.assertTrue(signal_filter.calibrated)

    def test_validation(self) -> None:
        signal_filter = Ble4SignalFilter(TactileConfig())
        with self.assertRaises(ModelValidationError):
            signal_filter.calibrate(())
        with self.assertRaises(ModelValidationError):
            signal_filter.process(((1, 2, 3),))


if __name__ == "__main__":
    unittest.main()
