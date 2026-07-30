"""Tests for layered, dependency-light configuration loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from airo_doffy.config import (
    AiroDoffyConfig,
    cli_override_mapping,
    config_from_mapping,
    config_to_mapping,
    deep_merge,
    environment_overrides,
    load_config,
    read_yaml,
)
from airo_doffy.core import ModelValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


class ConfigLoaderTest(unittest.TestCase):
    def test_precedence_default_robot_experiment_environment_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layers = (
                ("default.yaml", {"camera": {"fps": 60}, "robot": {"robot_type": "ur3e"}}),
                ("robot.yaml", {"camera": {"fps": 50}, "robot": {"robot_type": "ur5e"}}),
                ("experiment.yaml", {"camera": {"fps": 40}}),
            )
            for name, value in layers:
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            config = load_config(
                root / "default.yaml",
                robot_path=root / "robot.yaml",
                experiment_path=root / "experiment.yaml",
                environment={
                    "AIRO_DOFFY__CAMERA__FPS": "30",
                    "AIRO_DOFFY__CAMERA__DEPTH_ENABLED": "true",
                },
                cli_overrides={"camera.fps": "20"},
            )
        self.assertEqual(config.camera.fps, 20)
        self.assertTrue(config.camera.depth_enabled)
        self.assertEqual(config.robot.robot_type, "ur5e")

    def test_repository_profiles_load_without_optional_yaml_parser(self) -> None:
        default = load_config(CONFIG_ROOT / "default.yaml", environment={})
        ur3e = load_config(
            CONFIG_ROOT / "default.yaml",
            robot_path=CONFIG_ROOT / "robots" / "ur3e.yaml",
            experiment_path=CONFIG_ROOT / "experiments" / "collect_ur3e.yaml",
            environment={},
        )
        realman = load_config(
            CONFIG_ROOT / "default.yaml",
            robot_path=CONFIG_ROOT / "robots" / "realman_rm75.yaml",
            experiment_path=CONFIG_ROOT / "experiments" / "collect_rm75.yaml",
            environment={},
        )
        hand = load_config(
            CONFIG_ROOT / "default.yaml",
            experiment_path=CONFIG_ROOT / "experiments" / "vr_hand_tracking.yaml",
            environment={},
        )
        self.assertEqual(default, AiroDoffyConfig())
        self.assertEqual(ur3e.robot.robot_type, "ur3e")
        self.assertTrue(ur3e.robot.gripper_enabled)
        self.assertEqual(realman.robot.robot_type, "realman")
        self.assertEqual(len(realman.robot.initial_joints_rad), 7)
        self.assertEqual(realman.teleop.vr_to_robot_axes[0], (0.0, 0.0, 1.0))
        self.assertEqual(hand.vr.tracking_mode, "hand")

    def test_mapping_round_trip_and_non_mutating_merge(self) -> None:
        base = {"camera": {"fps": 60, "resolution": [640, 480]}}
        override = {"camera": {"fps": 30}}
        merged = deep_merge(base, override)
        merged["camera"]["resolution"][0] = 1
        self.assertEqual(base["camera"]["resolution"], [640, 480])
        config = config_from_mapping(config_to_mapping(AiroDoffyConfig()))
        self.assertEqual(config, AiroDoffyConfig())

    def test_override_parsing(self) -> None:
        environment = environment_overrides(
            {
                "AIRO_DOFFY__CAMERA__FPS": "24",
                "AIRO_DOFFY__TACTILE__ENABLED": "false",
                "IGNORED": "1",
            }
        )
        cli = cli_override_mapping(
            {
                "camera.resolution": "[1280, 720]",
                "recording.task_name": "unquoted-value",
            }
        )
        self.assertEqual(environment["camera"]["fps"], 24)
        self.assertFalse(environment["tactile"]["enabled"])
        self.assertEqual(cli["camera"]["resolution"], [1280, 720])
        self.assertEqual(cli["recording"]["task_name"], "unquoted-value")

    def test_unknown_and_malformed_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ModelValidationError, "unknown configuration section"):
            config_from_mapping({"robots": {}})
        with self.assertRaisesRegex(ModelValidationError, "unknown field"):
            config_from_mapping({"robot": {"hostname": "example"}})
        with self.assertRaisesRegex(ModelValidationError, "must be a mapping"):
            config_from_mapping({"camera": 60})
        with self.assertRaisesRegex(ModelValidationError, "SECTION__FIELD"):
            environment_overrides({"AIRO_DOFFY__CAMERA": "60"})
        with self.assertRaisesRegex(ModelValidationError, "section.field"):
            cli_override_mapping({"camera": "60"})

    def test_yaml_root_must_be_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "must be a mapping"):
                read_yaml(path)


if __name__ == "__main__":
    unittest.main()
