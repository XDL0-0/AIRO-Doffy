"""Tests for dependency-free teleoperation pose transforms."""

from __future__ import annotations

import math
import unittest

from airo_doffy.core import ModelValidationError
from airo_doffy.teleop.transforms import (
    RotationComposition,
    map_relative_pose,
    pose_delta,
    quaternion_xyzw_to_rotation,
    transform,
    validate_transform,
    vr_pose_to_transform,
)

IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
UR_AXES = (
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


def rotation_x(angle: float):
    return (
        (1.0, 0.0, 0.0),
        (0.0, math.cos(angle), -math.sin(angle)),
        (0.0, math.sin(angle), math.cos(angle)),
    )


def rotation_z(angle: float):
    return (
        (math.cos(angle), -math.sin(angle), 0.0),
        (math.sin(angle), math.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )


class TeleopTransformTest(unittest.TestCase):
    def assert_matrix_close(self, actual, expected, places: int = 7) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            for actual_value, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_ur_axis_and_quaternion_mapping_matches_legacy_rules(self) -> None:
        half = math.sin(math.pi / 8)
        cosine = math.cos(math.pi / 8)
        mapped = vr_pose_to_transform(
            (1, 2, 3),
            (0, half, 0, cosine),
            UR_AXES,
        )
        self.assertEqual(
            tuple(row[3] for row in mapped[:3]),
            (-1.0, -3.0, 2.0),
        )
        expected_rotation = quaternion_xyzw_to_rotation((0, 0, -half, cosine))
        self.assert_matrix_close(
            tuple(tuple(row[:3]) for row in mapped[:3]),
            expected_rotation,
        )

    def test_pose_delta_uses_world_translation_and_reference_local_rotation(self) -> None:
        reference = transform(rotation_z(math.pi / 2), (1, 2, 3))
        current_rotation = (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        current = transform(current_rotation, (2, 4, 6))
        delta = pose_delta(reference, current)
        self.assertEqual(delta.translation_m, (1.0, 2.0, 3.0))
        self.assert_matrix_close(delta.rotation, rotation_x(math.pi / 2))

    def test_scaling_and_rotation_composition_are_explicit(self) -> None:
        source_reference = transform(IDENTITY, (0, 0, 0))
        source_current = transform(rotation_z(math.pi / 2), (2, 0, 0))
        robot_reference = transform(rotation_x(math.pi / 2), (0, 1, 0))
        left = map_relative_pose(
            source_reference,
            source_current,
            robot_reference,
            translation_scale=0.5,
            rotation_scale=0.5,
            rotation_composition=RotationComposition.LEFT,
        )
        right = map_relative_pose(
            source_reference,
            source_current,
            robot_reference,
            translation_scale=0.5,
            rotation_scale=0.5,
            rotation_composition=RotationComposition.RIGHT,
        )
        self.assertEqual(tuple(row[3] for row in left[:3]), (1.0, 1.0, 0.0))
        self.assertNotEqual(left, right)

    def test_freeze_rotation_keeps_robot_reference(self) -> None:
        mapped = map_relative_pose(
            transform(IDENTITY, (0, 0, 0)),
            transform(rotation_z(1.0), (1, 0, 0)),
            transform(rotation_x(0.5), (0, 0, 0)),
            freeze_rotation=True,
        )
        self.assert_matrix_close(
            tuple(tuple(row[:3]) for row in mapped[:3]),
            rotation_x(0.5),
        )

    def test_invalid_pose_axis_quaternion_and_scale_are_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            validate_transform(((1, 0),))
        with self.assertRaises(ModelValidationError):
            vr_pose_to_transform(
                (0, 0, 0),
                (0, 0, 0, 1),
                ((1, 0, 0), (0, 1, 0), (1, 0, 1)),
            )
        with self.assertRaises(ModelValidationError):
            quaternion_xyzw_to_rotation((0, 0, 0, 0))
        with self.assertRaises(ModelValidationError):
            map_relative_pose(
                transform(IDENTITY, (0, 0, 0)),
                transform(IDENTITY, (0, 0, 0)),
                transform(IDENTITY, (0, 0, 0)),
                translation_scale=-1,
            )


if __name__ == "__main__":
    unittest.main()
