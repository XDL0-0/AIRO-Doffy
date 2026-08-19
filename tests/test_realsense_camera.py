from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from config import Config
from realsense_camera import RealSenseCameraManager


class FakeDevice:
    def __init__(self, name: str, serial: str) -> None:
        self.name = name
        self.serial = serial

    def get_info(self, field):
        return self.name if str(field).endswith("name") else self.serial


class FakeContext:
    def __init__(self, devices) -> None:
        self.devices = devices

    def query_devices(self):
        return self.devices


class FakePipeline:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeRealsense:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.pipeline = FakePipeline()

    def grab_images(self) -> None:
        pass

    def retrieve_rgb_image(self):
        return np.ones((4, 6, 3), dtype=np.float32)

    def retrieve_depth_map(self):
        return np.full((4, 6), 0.5, dtype=np.float32)


class RealSenseCameraManagerTests(unittest.TestCase):
    def make_manager(self, camera_num=3, depth_mode=True):
        devices = [
            FakeDevice(f"RealSense {idx}", f"serial-{idx}")
            for idx in range(camera_num)
        ]
        created = []

        def create_camera(**kwargs):
            camera = FakeRealsense(**kwargs)
            created.append(camera)
            return camera

        cfg = Config(DEPTH_INFO_ENABLE=depth_mode, REALSENSE_FPS=15)
        with (
            patch("realsense_camera.rs.context", return_value=FakeContext(devices)),
            patch("realsense_camera.Realsense", side_effect=create_camera),
        ):
            manager = RealSenseCameraManager(cfg)
        return manager, created

    def test_initializes_every_detected_camera_without_transport_resources(self) -> None:
        manager, cameras = self.make_manager(camera_num=3)

        self.assertEqual(manager.camera_num, 3)
        self.assertEqual(
            set(manager.camera_list), {"camera_0", "camera_1", "camera_2"}
        )
        self.assertEqual(
            [camera.kwargs["serial_number"] for camera in cameras],
            ["serial-0", "serial-1", "serial-2"],
        )
        self.assertFalse(hasattr(manager, "socket_list"))
        self.assertFalse(hasattr(manager, "signaling_port"))

        manager.close()
        self.assertTrue(all(camera.pipeline.stopped for camera in cameras))

    def test_capture_populates_rgb_depth_and_timestamps(self) -> None:
        manager, cameras = self.make_manager(camera_num=1, depth_mode=True)
        manager.running = True

        def stop_after_frame(_delay):
            manager.running = False

        with patch("realsense_camera.time.sleep", side_effect=stop_after_frame):
            manager._camera_read_thread(cameras[0], 0)

        np.testing.assert_array_equal(
            manager.camera_images["camera_0"],
            np.full((4, 6, 3), 255, dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            manager.depth_images["camera_0"],
            np.full((4, 6), 0.5, dtype=np.float32),
        )
        self.assertGreater(manager.camera_image_timestamps_ns["camera_0"], 0)
        self.assertEqual(
            manager.camera_image_timestamps_ns["camera_0"],
            manager.depth_timestamps_ns["camera_0"],
        )
        manager.close()


if __name__ == "__main__":
    unittest.main()
