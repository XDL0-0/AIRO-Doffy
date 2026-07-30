"""Tests for static, generated, video, dropped, and delayed mock cameras."""

from __future__ import annotations

import unittest

from airo_doffy.config import CameraConfig, CameraFactory
from airo_doffy.core import LifecycleError, ModelValidationError
from airo_doffy.devices.cameras import CameraSource, MockCameraSource


class _Clock:
    def __init__(self) -> None:
        self.value = 10

    def now_ns(self) -> int:
        self.value += 1
        return self.value


class MockCameraSourceTest(unittest.TestCase):
    def test_static_lifecycle_and_artificial_delay(self) -> None:
        sleeps = []
        source = MockCameraSource(
            CameraConfig(resolution=(2, 1)),
            static_frame=b"\x01\x02\x03\x04\x05\x06",
            artificial_delay_s=0.25,
            clock=_Clock(),
            sleep=sleeps.append,
        )
        self.assertIsInstance(source, CameraSource)
        with self.assertRaises(LifecycleError):
            source.read_latest()
        source.start()
        first = source.read_latest()
        second = source.read_latest()
        self.assertEqual(first.data, b"\x01\x02\x03\x04\x05\x06")
        self.assertEqual(first.sequence, 0)
        self.assertEqual(second.sequence, 1)
        self.assertEqual(sleeps, [0.25, 0.25])
        source.close()
        source.close()
        with self.assertRaises(LifecycleError):
            source.read_latest()

    def test_generated_frames_and_dropped_attempts(self) -> None:
        source = MockCameraSource(
            CameraConfig(resolution=(1, 1)),
            mode="generated",
            drop_every=2,
        )
        source.start()
        self.assertEqual(source.read_latest().data, b"\x00\x00\x00")
        self.assertIsNone(source.read_latest())
        third = source.read_latest()
        self.assertEqual(third.sequence, 2)
        self.assertEqual(third.data, b"\x02\x02\x02")
        source.close()

    def test_in_memory_video_playback_loops(self) -> None:
        source = MockCameraSource(
            CameraConfig(resolution=(1, 1)),
            mode="video",
            video_frames=(b"\x01" * 3, b"\x02" * 3),
        )
        source.start()
        values = [source.read_latest().data[0] for _ in range(3)]
        self.assertEqual(values, [1, 2, 1])
        source.set_disconnected(True)
        self.assertIsNone(source.read_latest())
        source.set_disconnected(False)
        self.assertEqual(source.read_latest().data[0], 2)
        source.close()

    def test_factory_creates_unstarted_static_source(self) -> None:
        factory = CameraFactory(
            target="airo_doffy.devices.cameras.mock:create_mock_camera"
        )
        source = factory.create(CameraConfig(resolution=(1, 1)))
        self.assertIsInstance(source, MockCameraSource)
        source.start()
        self.assertEqual(source.read_latest().data, b"\x00\x00\x00")
        source.close()

    def test_validation(self) -> None:
        config = CameraConfig(resolution=(1, 1))
        with self.assertRaises(ModelValidationError):
            MockCameraSource(config, mode="video")
        with self.assertRaises(ModelValidationError):
            MockCameraSource(config, static_frame=b"short")
        with self.assertRaises(ModelValidationError):
            MockCameraSource(config, drop_every=0)
        with self.assertRaises(ModelValidationError):
            MockCameraSource(config, artificial_delay_s=-1)


if __name__ == "__main__":
    unittest.main()
