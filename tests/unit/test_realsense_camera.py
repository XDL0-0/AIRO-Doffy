"""Hardware-free RealSense discovery, acquisition, and lifecycle tests."""

from __future__ import annotations

import sys
import unittest

from airo_doffy.config import CameraConfig, CameraFactory
from airo_doffy.core import LifecycleError, PixelFormat
from airo_doffy.devices.cameras import (
    CameraSource,
    DepthCameraSource,
    RealSenseCameraSource,
    RealSenseDevice,
    discover_realsense_devices,
)


class _Image:
    def __init__(self, shape, data: bytes) -> None:
        self.shape = shape
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


class _Clock:
    def __init__(self) -> None:
        self.value = 100

    def now_ns(self) -> int:
        self.value += 1
        return self.value


class _Pipeline:
    def __init__(self) -> None:
        self.stop_count = 0

    def stop(self) -> None:
        self.stop_count += 1


class _Camera:
    def __init__(self, *, failures: int = 0) -> None:
        self.pipeline = _Pipeline()
        self.failures = failures
        self.read_count = 0

    def get_rgb_image(self):
        self.read_count += 1
        if self.read_count <= self.failures:
            raise RuntimeError("injected camera read failure")
        return _Image((1, 2, 3), b"\x01\x02\x03\x04\x05\x06")

    def _retrieve_depth_map(self):
        return _Image((1, 2), b"\x01\x00\x02\x00")


class _Device:
    def __init__(self, name: str, serial: str) -> None:
        self.values = {"name": name, "serial": serial}

    def get_info(self, key: str) -> str:
        return self.values[key]


class _Rs:
    class camera_info:
        name = "name"
        serial_number = "serial"

    class _Context:
        def query_devices(self):
            return (_Device("D435", "A"), _Device("D455", "B"))

    @staticmethod
    def context():
        return _Rs._Context()


class RealSenseCameraSourceTest(unittest.TestCase):
    def test_discovery_returns_name_and_serial_without_opening_camera(self) -> None:
        self.assertEqual(
            discover_realsense_devices(_Rs),
            (
                RealSenseDevice("D435", "A"),
                RealSenseDevice("D455", "B"),
            ),
        )

    def test_color_depth_latest_metadata_and_idempotent_close(self) -> None:
        camera = _Camera()
        calls = []

        def factory(config, serial):
            calls.append((config, serial))
            return camera

        config = CameraConfig(
            serial_number="SERIAL",
            depth_enabled=True,
            capture_rate_hz=10,
        )
        source = RealSenseCameraSource(config, camera_factory=factory, clock=_Clock())
        self.assertIsInstance(source, CameraSource)
        self.assertIsInstance(source, DepthCameraSource)
        source.start()
        self.assertTrue(source.wait_for_first_frame(0.5))
        color = source.read_latest()
        depth = source.read_latest_depth()
        self.assertIsNotNone(color)
        self.assertIsNotNone(depth)
        assert color is not None
        assert depth is not None
        self.assertEqual(calls, [(config, "SERIAL")])
        self.assertEqual(color.stream_id, "camera_0")
        self.assertEqual(color.pixel_format, PixelFormat.RGB8)
        self.assertEqual(color.data, b"\x01\x02\x03\x04\x05\x06")
        self.assertEqual(depth.stream_id, "camera_0_depth")
        self.assertEqual(depth.pixel_format, PixelFormat.DEPTH_U16)
        self.assertEqual(depth.source_timestamp_ns, color.source_timestamp_ns)
        self.assertEqual(depth.sequence, color.sequence)
        source.close()
        source.close()
        self.assertEqual(camera.pipeline.stop_count, 1)
        with self.assertRaises(LifecycleError):
            source.read_latest()

    def test_first_device_selection_and_bounded_read_failures(self) -> None:
        camera = _Camera(failures=3)
        selected = []
        config = CameraConfig(
            capture_rate_hz=1000,
            max_consecutive_errors=3,
            retry_delay_s=0.001,
        )
        source = RealSenseCameraSource(
            config,
            camera_factory=lambda _config, serial: selected.append(serial) or camera,
            discoverer=lambda: (RealSenseDevice("D435", "FIRST"),),
        )
        source.start()
        self.assertFalse(source.wait_for_first_frame(0.5))
        self.assertEqual(selected, ["FIRST"])
        self.assertEqual(camera.read_count, 3)
        self.assertIsInstance(source.health_error, RuntimeError)
        source.close()

    def test_factory_constructs_without_importing_optional_sdk(self) -> None:
        factory = CameraFactory(
            target=(
                "airo_doffy.devices.cameras.realsense:"
                "create_realsense_camera"
            )
        )
        source = factory.create(CameraConfig(serial_number="SAFE"))
        self.assertIsInstance(source, RealSenseCameraSource)
        self.assertNotIn("pyrealsense2", sys.modules)
        self.assertNotIn(
            "airo_camera_toolkit.cameras.realsense.realsense",
            sys.modules,
        )
        source.close()


if __name__ == "__main__":
    unittest.main()
