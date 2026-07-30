"""Recording schema compatibility tests."""

from __future__ import annotations

import unittest

from airo_doffy.core.errors import ModelValidationError
from airo_doffy.recording import build_recording_schema, normalize_data_type


class RecordingSchemaTest(unittest.TestCase):
    def test_legacy_data_type_aliases_preserve_dimensions(self) -> None:
        expected = {
            "qpos": ("qpos", 7, 7, False),
            "joint": ("qpos", 7, 7, False),
            "joint_configuration": ("qpos", 7, 7, False),
            "both": ("both", 7, 7, True),
            "tcp": ("tcp", 8, 8, False),
            "tcp_quat": ("tcp", 8, 8, False),
            "eef": ("tcp", 8, 8, False),
            "delta_tcp": ("delta_tcp", 7, 7, True),
        }
        for value, result in expected.items():
            with self.subTest(data_type=value):
                schema = build_recording_schema(
                    data_type=value,
                    robot_dof=6,
                    camera_count=0,
                    resolution=(640, 480),
                )
                self.assertEqual(
                    (
                        schema.data_type,
                        schema.state_dim,
                        schema.action_dim,
                        schema.stores_tcp_pose,
                    ),
                    result,
                )

    def test_delta_tcp_names_and_timestamp_order_are_stable(self) -> None:
        schema = build_recording_schema(
            data_type="delta_tcp",
            robot_dof=7,
            camera_count=2,
            resolution=(640, 480),
        )
        self.assertEqual(
            schema.action_names,
            (
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_rotvec_x",
                "delta_rotvec_y",
                "delta_rotvec_z",
                "gripper",
            ),
        )
        self.assertEqual(
            schema.timestamp_names,
            (
                "collect",
                "robot_state",
                "robot_action",
                "vr_input",
                "tactile",
                "camera_0",
                "camera_1",
            ),
        )

    def test_hdf5_fields_preserve_paths_shapes_and_dtypes(self) -> None:
        schema = build_recording_schema(
            data_type="both",
            robot_dof=6,
            camera_count=1,
            resolution=(640, 480),
            tactile_shape=(4, 3),
            force_enabled=True,
            torque_enabled=True,
            depth_enabled=True,
        )
        fields = {field.path: field for field in schema.hdf5_fields()}
        self.assertEqual(fields["/observations/qpos"].shape, (7,))
        self.assertEqual(fields["/observations/qpos"].dtype, "float64")
        self.assertEqual(fields["/action"].dtype, "float64")
        self.assertEqual(fields["/extra/timestamps_ns"].dtype, "int64")
        self.assertEqual(fields["/extra/tcp_pose"].shape, (7,))
        self.assertEqual(fields["/observations/force"].shape, (3,))
        self.assertEqual(fields["/observations/torque"].shape, (3,))
        self.assertEqual(fields["/observations/tactile"].shape, (4, 3))
        self.assertEqual(
            fields["/observations/images/camera_0"].shape,
            (480, 640, 3),
        )
        self.assertEqual(
            fields["/observations/depth/camera_0"].shape,
            (480, 640),
        )
        self.assertEqual(
            fields["/observations/depth/camera_0"].dtype,
            "float32",
        )

    def test_lerobot_features_preserve_paths_shapes_and_dtypes(self) -> None:
        schema = build_recording_schema(
            data_type="both",
            robot_dof=6,
            camera_count=1,
            resolution=(640, 480),
            tactile_shape=(4, 3),
            force_enabled=True,
            torque_enabled=True,
            depth_enabled=True,
        )
        features = schema.lerobot_features()
        self.assertEqual(features["action"]["dtype"], "float32")
        self.assertEqual(features["observation.state"]["shape"], (7,))
        self.assertEqual(features["extra.timestamps_ns"]["shape"], (6,))
        self.assertEqual(features["extra.tcp_pose"]["shape"], (7,))
        self.assertEqual(features["observation.images.camera_0"]["dtype"], "video")
        self.assertEqual(
            features["observation.images.camera_0"]["shape"],
            (480, 640, 3),
        )
        self.assertEqual(
            features["observation.depth.camera_0"]["shape"],
            (480, 640, 1),
        )

    def test_invalid_schema_values_are_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_data_type("unknown")
        with self.assertRaises(ModelValidationError):
            build_recording_schema(
                data_type="qpos",
                robot_dof=0,
                camera_count=1,
                resolution=(640, 480),
            )
        with self.assertRaises(ModelValidationError):
            build_recording_schema(
                data_type="qpos",
                robot_dof=6,
                camera_count=-1,
                resolution=(640, 480),
            )


if __name__ == "__main__":
    unittest.main()
