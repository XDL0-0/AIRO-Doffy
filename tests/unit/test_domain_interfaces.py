"""Structural checks for hardware-free domain-owned protocols."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from airo_doffy.core import (
    CameraFrame,
    EncodedFrame,
    Observation,
    ProcessedFrame,
    RobotAction,
    RobotState,
    TactileSample,
    VRInputState,
)
from airo_doffy.devices.cameras import CameraSource
from airo_doffy.devices.tactile import TactileSensor
from airo_doffy.devices.vr import VRInputSource
from airo_doffy.recording import EpisodeRecorder
from airo_doffy.robots import RobotBackend
from airo_doffy.streaming.video import (
    FrameProcessor,
    VideoEncoder,
    VideoEncodingPipeline,
    VideoTransport,
)
from airo_doffy.teleop.mappings import TeleopMapping
from airo_doffy.teleop.safety import ActionFilter


class FakeCamera:
    def start(self) -> None:
        pass

    def read_latest(self) -> CameraFrame | None:
        return None

    def close(self) -> None:
        pass


class FakeProcessor:
    def process(self, frame: CameraFrame) -> ProcessedFrame:
        raise NotImplementedError


class FakeEncoder:
    def encode(self, frame: ProcessedFrame) -> EncodedFrame:
        raise NotImplementedError


class FakeVideoTransport:
    def start(self) -> None:
        pass

    def send(self, frame: EncodedFrame) -> None:
        pass

    def close(self) -> None:
        pass


class FakeEncodingPipeline:
    def start(self) -> None:
        pass

    def submit(self, frame: ProcessedFrame) -> bool:
        return True

    def read_latest(self) -> EncodedFrame | None:
        return None

    def close(self) -> None:
        pass


class FakeVR:
    def start(self) -> None:
        pass

    def read_latest(self) -> VRInputState | None:
        return None

    def close(self) -> None:
        pass


class FakeTactile:
    def start(self) -> None:
        pass

    def read_latest(self) -> TactileSample | None:
        return None

    def recalibrate(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeMapping:
    def map_input(
        self,
        vr_input: VRInputState,
        robot_state: RobotState,
        dt_s: float,
    ) -> RobotAction:
        raise NotImplementedError


class FakeFilter:
    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        return action


class FakeRobot:
    name = "fake"
    dof = 6

    def start(self) -> None:
        pass

    def read_state(self) -> RobotState:
        raise NotImplementedError

    def apply_action(self, action: RobotAction) -> None:
        pass

    def close(self) -> None:
        pass


class FakeRecorder:
    def start_episode(self) -> None:
        pass

    def append(self, observation: Observation) -> None:
        pass

    def finish_episode(self) -> int:
        return 0

    def rollback_last_episode(self) -> bool:
        return False

    def close(self) -> None:
        pass


class DomainInterfaceTest(unittest.TestCase):
    def test_structural_protocols_accept_narrow_fakes(self) -> None:
        pairs = (
            (FakeCamera(), CameraSource),
            (FakeProcessor(), FrameProcessor),
            (FakeEncoder(), VideoEncoder),
            (FakeEncodingPipeline(), VideoEncodingPipeline),
            (FakeVideoTransport(), VideoTransport),
            (FakeVR(), VRInputSource),
            (FakeTactile(), TactileSensor),
            (FakeMapping(), TeleopMapping),
            (FakeFilter(), ActionFilter),
            (FakeRobot(), RobotBackend),
            (FakeRecorder(), EpisodeRecorder),
        )
        for instance, protocol in pairs:
            with self.subTest(protocol=protocol.__name__):
                self.assertIsInstance(instance, protocol)

    def test_interface_imports_do_not_load_optional_sdks(self) -> None:
        code = """
import sys
from airo_doffy.devices.cameras import CameraSource
from airo_doffy.devices.tactile import TactileSensor
from airo_doffy.devices.vr import VRInputSource
from airo_doffy.devices.wrench import WrenchSource
from airo_doffy.recording import EpisodeRecorder
from airo_doffy.robots import RobotBackend
from airo_doffy.streaming.video import (
    FrameProcessor, VideoEncoder, VideoEncodingPipeline, VideoTransport,
)
from airo_doffy.teleop.mappings import TeleopMapping
from airo_doffy.teleop.safety import ActionFilter
blocked = (
    "numpy", "torch", "cv2", "scipy", "pyrealsense2",
    "aiortc", "serial", "airo_robots",
)
loaded = [name for name in sys.modules if name.startswith(blocked)]
assert not loaded, loaded
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
