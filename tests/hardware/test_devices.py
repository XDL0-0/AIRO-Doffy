"""Explicitly enabled, read-only smoke checks for supported devices."""

from __future__ import annotations

import os
import time
import unittest

from airo_doffy.config import CameraConfig, RobotConfig, TactileConfig
from airo_doffy.devices.cameras import create_realsense_camera
from airo_doffy.devices.tactile import create_magtouch_ble4
from airo_doffy.robots import create_realman_backend, create_ur_backend


def _wait_for_sample(read, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = read()
        if sample is not None:
            return sample
        time.sleep(0.02)
    return read()


class HardwareDeviceTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("AIRO_DOFFY_TEST_UR_IP"),
        "set AIRO_DOFFY_TEST_UR_IP to enable the UR state check",
    )
    def test_ur_state_read(self) -> None:
        backend = create_ur_backend(
            RobotConfig(
                robot_type=os.getenv("AIRO_DOFFY_TEST_UR_TYPE", "ur3e"),
                ip=os.environ["AIRO_DOFFY_TEST_UR_IP"],
            )
        )
        backend.start()
        try:
            state = backend.read_state()
            self.assertEqual(len(state.joints_rad), 6)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.getenv("AIRO_DOFFY_TEST_REALMAN_IP"),
        "set AIRO_DOFFY_TEST_REALMAN_IP to enable the RealMan state check",
    )
    def test_realman_state_read(self) -> None:
        backend = create_realman_backend(
            RobotConfig(
                robot_type="realman",
                ip=os.environ["AIRO_DOFFY_TEST_REALMAN_IP"],
            )
        )
        backend.start()
        try:
            state = backend.read_state()
            self.assertEqual(len(state.joints_rad), 7)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.getenv("AIRO_DOFFY_TEST_REALSENSE") == "1",
        "set AIRO_DOFFY_TEST_REALSENSE=1 to enable the camera check",
    )
    def test_realsense_frame(self) -> None:
        source = create_realsense_camera(
            CameraConfig(
                resolution=(640, 480),
                fps=30,
                serial_number=os.getenv(
                    "AIRO_DOFFY_TEST_REALSENSE_SERIAL"
                ),
            )
        )
        source.start()
        try:
            frame = _wait_for_sample(source.read_latest, 5.0)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))
        finally:
            source.close()

    @unittest.skipUnless(
        os.getenv("AIRO_DOFFY_TEST_BLE4") == "1",
        "set AIRO_DOFFY_TEST_BLE4=1 to enable the tactile check",
    )
    def test_ble4_sample(self) -> None:
        sensor = create_magtouch_ble4(TactileConfig())
        sensor.start()
        try:
            sample = _wait_for_sample(sensor.read_latest, 10.0)
            self.assertIsNotNone(sample)
            self.assertEqual(len(sample.values), 4)
            self.assertTrue(all(len(taxel) == 3 for taxel in sample.values))
        finally:
            sensor.close()


if __name__ == "__main__":
    unittest.main()
