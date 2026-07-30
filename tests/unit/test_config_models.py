"""Validation and legacy-default snapshots for typed configuration models."""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from airo_doffy.config import (
    AiroDoffyConfig,
    CameraConfig,
    CommandTransportConfig,
    RecordingConfig,
    RobotConfig,
    StateTransportConfig,
    TactileConfig,
    TeleopConfig,
    VideoStreamingConfig,
    VisualizationConfig,
    VRConfig,
)
from airo_doffy.core import ModelValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]


class LegacyConfigSnapshotTest(unittest.TestCase):
    def test_selected_legacy_defaults_are_characterized(self) -> None:
        code = textwrap.dedent(
            """
            import importlib.util, json, logging, pathlib, sys, types
            fake_utils = types.ModuleType("utils")
            fake_utils.logger = logging.getLogger("snapshot")
            sys.modules["utils"] = fake_utils
            package = types.ModuleType("airo_spatial_algebra")
            package.__path__ = []
            se3 = types.ModuleType("airo_spatial_algebra.se3")
            se3.SE3Container = type("DummySE3", (), {})
            sys.modules["airo_spatial_algebra"] = package
            sys.modules["airo_spatial_algebra.se3"] = se3
            spec = importlib.util.spec_from_file_location(
                "_legacy_config", pathlib.Path("config.py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            cfg = module.Config()
            viz_spec = importlib.util.spec_from_file_location(
                "_legacy_visualizer_config", pathlib.Path("visualizer_config.py")
            )
            viz_module = importlib.util.module_from_spec(viz_spec)
            sys.modules[viz_spec.name] = viz_module
            viz_spec.loader.exec_module(viz_module)
            viz = viz_module.VisualizerConfig()
            def convert(value):
                return value.tolist() if hasattr(value, "tolist") else value
            names = (
                "ROBOT_TYPE", "ROBOT_IP", "INITIAL_JOINT", "VR_TO_ROBOT_AXES",
                "REALMAN_CTRL_RATE", "REALMAN_MIN_CANFD_RATE", "PC_IP", "VR_IP",
                "TASK_NAME", "DATASET_DIR", "DATA_TYPE", "TRACKING_MODE",
                "REALSENSE_RESOLUTION", "REALSENSE_FPS", "IP_PORT", "POSE_PORT",
                "CONTROL_PORT", "SIGNALING_PORT", "VIDEO_TRANSPORT", "JPEG_QUALITY",
                "HD_CHUNK_SIZE", "UR_CTRL_RATE", "COLLECT_RATE", "GRIPPER",
                "GRIPPER_MAX", "TACTILE_ENABLE", "TACTILE_READER", "TACTILE_SHAPE",
                "FORCE_MOVING_AVERAGE_WINDOW", "FORCE_LOW_PASS_ALPHA",
            )
            result = {name: convert(getattr(cfg, name)) for name in names}
            result["VISUALIZER"] = {
                "ENABLED": viz.ENABLED,
                "HZ": viz.HZ,
                "WINDOW_S": viz.WINDOW_S,
                "FORCE_PANEL_RANGE": viz.FORCE_PANEL_RANGE,
            }
            print(json.dumps(result, sort_keys=True))
            """
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = json.loads(result.stdout)
        self.assertEqual(
            snapshot,
            {
                "COLLECT_RATE": 10,
                "CONTROL_PORT": 8005,
                "DATASET_DIR": "./datasets/pnp_long",
                "DATA_TYPE": "both",
                "FORCE_LOW_PASS_ALPHA": 0.15,
                "FORCE_MOVING_AVERAGE_WINDOW": 8,
                "GRIPPER": False,
                "GRIPPER_MAX": 0.085,
                "HD_CHUNK_SIZE": 60000,
                "INITIAL_JOINT": [1.57, -1.57, 1.57, -1.57, -1.57, 0.0],
                "IP_PORT": 8000,
                "JPEG_QUALITY": 100,
                "PC_IP": "10.10.131.162",
                "POSE_PORT": 8001,
                "REALMAN_CTRL_RATE": 200,
                "REALMAN_MIN_CANFD_RATE": 100.0,
                "REALSENSE_FPS": 60,
                "REALSENSE_RESOLUTION": [640, 480],
                "ROBOT_IP": "10.42.0.162",
                "ROBOT_TYPE": "ur3e",
                "SIGNALING_PORT": 8765,
                "TACTILE_ENABLE": True,
                "TACTILE_READER": "ble4",
                "TACTILE_SHAPE": [4, 3],
                "TASK_NAME": "pick_and_place",
                "TRACKING_MODE": "controller",
                "UR_CTRL_RATE": 60,
                "VIDEO_TRANSPORT": "webrtc",
                "VISUALIZER": {
                    "ENABLED": True,
                    "FORCE_PANEL_RANGE": 30.0,
                    "HZ": 30.0,
                    "WINDOW_S": 8.0,
                },
                "VR_IP": "10.10.130.155",
                "VR_TO_ROBOT_AXES": [
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
            },
        )


class TypedConfigModelTest(unittest.TestCase):
    def test_safe_defaults_preserve_non_address_behavior(self) -> None:
        config = AiroDoffyConfig()
        self.assertIsNone(config.network.pc_ip)
        self.assertIsNone(config.network.vr_ip)
        self.assertIsNone(config.robot.ip)
        self.assertEqual(config.robot.initial_joints_rad, (1.57, -1.57, 1.57, -1.57, -1.57, 0.0))
        self.assertEqual(config.camera, CameraConfig(resolution=(640, 480), fps=60))
        self.assertEqual(config.vr, VRConfig(tracking_mode="controller"))
        self.assertEqual(config.recording.data_type, "both")
        self.assertEqual(config.tactile.shape, (4, 3))
        self.assertEqual(config.tactile.deadband_sigma, 3.0)
        self.assertEqual(config.tactile.max_abs, 20000.0)
        self.assertEqual(config.video.transport, "webrtc")
        self.assertEqual(config.visualization, VisualizationConfig())

    def test_realman_defaults_and_timing_validation(self) -> None:
        robot = RobotConfig(robot_type="realman")
        self.assertEqual(len(robot.initial_joints_rad), 7)
        with self.assertRaises(ModelValidationError):
            RobotConfig(
                robot_type="realman",
                realman_control_rate_hz=100,
                realman_min_canfd_rate_hz=100,
            )
        with self.assertRaises(ModelValidationError):
            RobotConfig(robot_type="realman", torque_mode=True)

    def test_teleop_transform_and_shape_validation(self) -> None:
        config = TeleopConfig(tcp_pose=(1, 2, 3, 0, 0, math.pi / 2))
        transform = config.tcp_transform
        self.assertAlmostEqual(transform[0][0], 0.0)
        self.assertAlmostEqual(transform[1][0], 1.0)
        self.assertEqual((transform[0][3], transform[1][3], transform[2][3]), (1.0, 2.0, 3.0))
        with self.assertRaises(ModelValidationError):
            TeleopConfig(vr_to_robot_axes=((1, 0, 0), (0, 1, 0), (1, 0, 1)))

    def test_section_validation(self) -> None:
        with self.assertRaises(ModelValidationError):
            CameraConfig(resolution=(640, 0))
        with self.assertRaises(ModelValidationError):
            TactileConfig(backend="serial")
        with self.assertRaises(ModelValidationError):
            RecordingConfig(data_type="unknown")
        with self.assertRaises(ModelValidationError):
            VideoStreamingConfig(jpeg_quality=101)
        with self.assertRaises(ModelValidationError):
            StateTransportConfig(transport="tcp")
        with self.assertRaises(ModelValidationError):
            CommandTransportConfig(reliable=False)

    def test_sections_are_frozen(self) -> None:
        config = AiroDoffyConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.robot.ip = "192.0.2.1"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
