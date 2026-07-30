"""Tests for focused lazy component factories."""

from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass

from airo_doffy.config import (
    CameraConfig,
    CameraFactory,
    EncoderFactory,
    NetworkConfig,
    RecorderFactory,
    RecordingConfig,
    RobotConfig,
    RobotFactory,
    TactileConfig,
    TactileFactory,
    VideoStreamingConfig,
    VideoTransportFactory,
    VisualizationConfig,
    VisualizerFactory,
    VRConfig,
    VRSourceFactory,
)
from airo_doffy.core import ModelValidationError, OptionalDependencyError


class _Lifecycle:
    def start(self) -> None:
        pass

    def close(self) -> None:
        pass


class _Robot(_Lifecycle):
    name = "fake"
    dof = 6

    def read_state(self):
        return None

    def apply_action(self, action) -> None:
        pass


class _Camera(_Lifecycle):
    def read_latest(self):
        return None


class _Encoder:
    def encode(self, frame):
        return frame


class _Transport(_Lifecycle):
    def send(self, frame) -> None:
        pass


class _VRSource(_Lifecycle):
    def read_latest(self):
        return None


class _Tactile(_Lifecycle):
    def read_latest(self):
        return None

    def recalibrate(self) -> None:
        pass


class _Visualizer(_Lifecycle):
    def publish(self, snapshot) -> bool:
        return True


class _Recorder:
    def close(self) -> None:
        pass

    def start_episode(self) -> None:
        pass

    def append(self, observation) -> None:
        pass

    def finish_episode(self) -> int:
        return 0

    def rollback_last_episode(self) -> bool:
        return False


@dataclass
class _Call:
    name: str
    args: tuple[object, ...]


class ConfigFactoryTest(unittest.TestCase):
    module_name = "_airo_doffy_test_factories"

    def setUp(self) -> None:
        self.calls: list[_Call] = []
        module = types.ModuleType(self.module_name)

        def constructor(name, component_type):
            def create(*args):
                self.calls.append(_Call(name, args))
                return component_type()

            return create

        module.make_robot = constructor("robot", _Robot)
        module.make_camera = constructor("camera", _Camera)
        module.make_encoder = constructor("encoder", _Encoder)
        module.make_transport = constructor("transport", _Transport)
        module.make_vr = constructor("vr", _VRSource)
        module.make_tactile = constructor("tactile", _Tactile)
        module.make_recorder = constructor("recorder", _Recorder)
        module.make_visualizer = constructor("visualizer", _Visualizer)
        module.make_lifecycle_only = constructor("lifecycle", _Lifecycle)
        module.make_invalid = lambda config: object()
        module.not_callable = object()
        sys.modules[self.module_name] = module

    def tearDown(self) -> None:
        sys.modules.pop(self.module_name, None)

    def test_each_factory_receives_only_its_declared_sections(self) -> None:
        robot = RobotConfig()
        camera = CameraConfig()
        video = VideoStreamingConfig()
        network = NetworkConfig()
        vr = VRConfig()
        tactile = TactileConfig()
        recording = RecordingConfig()
        visualization = VisualizationConfig()

        RobotFactory(target=f"{self.module_name}:make_robot").create(robot)
        CameraFactory(target=f"{self.module_name}:make_camera").create(camera)
        EncoderFactory(target=f"{self.module_name}:make_encoder").create(video)
        VideoTransportFactory(target=f"{self.module_name}:make_transport").create(video, network)
        VRSourceFactory(target=f"{self.module_name}:make_vr").create(vr, network)
        TactileFactory(target=f"{self.module_name}:make_tactile").create(tactile)
        RecorderFactory(target=f"{self.module_name}:make_recorder").create(recording)
        VisualizerFactory(target=f"{self.module_name}:make_visualizer").create(visualization)

        self.assertEqual(
            self.calls,
            [
                _Call("robot", (robot,)),
                _Call("camera", (camera,)),
                _Call("encoder", (video,)),
                _Call("transport", (video, network)),
                _Call("vr", (vr, network)),
                _Call("tactile", (tactile,)),
                _Call("recorder", (recording,)),
                _Call("visualizer", (visualization,)),
            ],
        )

    def test_factory_does_not_import_target_until_create(self) -> None:
        factory = RobotFactory(target="_missing_vendor_adapter:create")
        self.assertNotIn("_missing_vendor_adapter", sys.modules)
        with self.assertRaises(OptionalDependencyError):
            factory.create(RobotConfig())

    def test_invalid_target_or_component_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ModelValidationError, "module:symbol"):
            RobotFactory(target="invalid").create(RobotConfig())
        with self.assertRaisesRegex(ModelValidationError, "does not exist"):
            RobotFactory(target=f"{self.module_name}:missing").create(RobotConfig())
        with self.assertRaisesRegex(ModelValidationError, "not callable"):
            RobotFactory(target=f"{self.module_name}:not_callable").create(RobotConfig())
        with self.assertRaisesRegex(ModelValidationError, "does not satisfy RobotBackend"):
            RobotFactory(target=f"{self.module_name}:make_invalid").create(RobotConfig())
        with self.assertRaisesRegex(
            ModelValidationError,
            "does not satisfy SnapshotConsumer",
        ):
            VisualizerFactory(
                target=f"{self.module_name}:make_lifecycle_only"
            ).create(VisualizationConfig())


if __name__ == "__main__":
    unittest.main()
