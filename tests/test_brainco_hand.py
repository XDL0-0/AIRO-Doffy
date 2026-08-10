import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import types
import unittest
from unittest.mock import patch

import numpy as np

from brainco_hand import (
    BrainCoHandDriver,
    BrainCoHandMotionFilter,
    BrainCoHandUnavailableError,
    openxr_thumb_flexion,
    openxr_thumb_opposition_progress,
    openxr_to_brainco_joints,
)
from config import Config


class FakeArm:
    def __init__(self, *, hand_type=2, dof=6, mode=460800):
        self.mode = mode
        self.base_info = {
            "type": hand_type,
            "dof": dof,
            "pos_low": [10, 20, 30, 40, 50, 60],
            "pos_up": [110, 220, 330, 440, 550, 660],
        }
        self.position = [10, 20, 30, 40, 50, 60]
        self.commands = []

    def rm_get_rm_plus_mode(self):
        return 0, self.mode

    def rm_set_rm_plus_mode(self, mode):
        self.mode = mode
        return 0

    def rm_get_rm_plus_base_info(self):
        return 0, self.base_info

    def rm_get_rm_plus_state_info(self):
        return 0, {"pos": self.position}

    def rm_set_hand_follow_pos(self, target, block):
        self.commands.append((target, block))
        return 0


def straight_skeleton():
    bones = np.zeros((26, 3), dtype=float)
    bones[0] = [0.02, -0.02, 0.0]
    bones[1] = [0.02, -0.04, 0.0]
    bones[2:6] = [
        [-0.05, 0.00, 0.0],
        [-0.04, 0.01, 0.0],
        [-0.03, 0.02, 0.0],
        [-0.02, 0.03, 0.0],
    ]
    for start, x in ((6, 0.00), (11, 0.013), (16, 0.026), (21, 0.04)):
        for offset in range(5):
            bones[start + offset] = [x, 0.02 * offset, 0.0]
    return bones


class OpenXrMappingTest(unittest.TestCase):
    def test_straight_hand_maps_to_open_fingers(self):
        bones = straight_skeleton()
        joints = openxr_to_brainco_joints(
            bones,
            thumb_rotate_open_progress=openxr_thumb_opposition_progress(bones),
        )

        np.testing.assert_allclose(joints[:5], np.zeros(5), atol=1e-12)
        self.assertEqual(joints[5], 0.0)

    def test_three_right_angle_bends_close_index(self):
        bones = straight_skeleton()
        bones[6:11] = [
            [0.00, 0.00, 0.0],
            [0.00, 0.02, 0.0],
            [0.02, 0.02, 0.0],
            [0.02, 0.00, 0.0],
            [0.00, 0.00, 0.0],
        ]

        joints = openxr_to_brainco_joints(bones)

        self.assertAlmostEqual(joints[1], 1.0)
        self.assertAlmostEqual(joints[2], 0.0)

    def test_thumb_opposition_progress_maps_to_rotation(self):
        bones = straight_skeleton()
        bones[5] = bones[21].copy()

        joints = openxr_to_brainco_joints(bones)

        self.assertEqual(joints[5], 1.0)

    def test_thumb_flex_closes_for_fingertip_contact_even_when_thumb_is_straight(self):
        bones = straight_skeleton()
        bones[2:6] = [
            [-0.04, 0.00, 0.0],
            [-0.03, 0.02, 0.0],
            [-0.02, 0.04, 0.0],
            [0.00, 0.079, 0.0],
        ]

        joints = openxr_to_brainco_joints(bones)

        self.assertGreater(joints[0], 0.95)

    def test_thumb_tip_trajectory_amplifies_modest_joint_curvature(self):
        bones = straight_skeleton()
        # Three thumb segments at approximately 0, 20 and 40 degrees. This is
        # only a modest human bend but should produce useful Revo2 flexion.
        bones[2:6] = [
            [-0.04000, 0.00000, 0.0],
            [-0.03000, 0.00000, 0.0],
            [-0.02060, 0.00342, 0.0],
            [-0.01294, 0.00985, 0.0],
        ]

        thumb_flex = openxr_thumb_flexion(bones)
        joints = openxr_to_brainco_joints(bones)

        self.assertGreater(thumb_flex, 0.7)
        self.assertAlmostEqual(joints[0], thumb_flex)


class BrainCoDriverTest(unittest.TestCase):
    def test_recognizes_hand_and_maps_normalized_target_to_hardware_limits(self):
        arm = FakeArm()
        driver = BrainCoHandDriver(
            arm,
            retry_delay=0.0,
            mode_settle_delay=0.0,
            max_send_hz=50.0,
            filter_cutoff_hz=0.0,
            dead_zone=0.0,
        )

        sent = driver.send_normalized([0.0, 0.25, 0.5, 0.75, 1.0, 0.5])

        self.assertTrue(sent)
        self.assertEqual(arm.commands[-1][0], [10, 70, 180, 340, 550, 360])
        self.assertTrue(arm.commands[-1][1])

    def test_configures_requested_rm_plus_baudrate(self):
        arm = FakeArm(mode=115200)

        BrainCoHandDriver(
            arm,
            baudrate=460800,
            retry_delay=0.0,
            mode_settle_delay=0.0,
        )

        self.assertEqual(arm.mode, 460800)

    def test_grab_and_release_presets_match_keyboard_teleop(self):
        arm = FakeArm()
        driver = BrainCoHandDriver(
            arm,
            retry_delay=0.0,
            mode_settle_delay=0.0,
            max_send_hz=50.0,
            filter_cutoff_hz=0.0,
            dead_zone=0.0,
        )

        self.assertTrue(driver.request_motion("grab", now=1.0))
        self.assertFalse(driver.advance_motion(now=1.79))
        self.assertTrue(driver.advance_motion(now=1.80))
        self.assertTrue(driver.advance_motion(now=2.60))

        self.assertEqual(
            [command[0] for command in arm.commands],
            [
                [10, 20, 30, 40, 50, 540],
                [38, 200, 300, 400, 500, 540],
                [98, 200, 300, 400, 500, 540],
            ],
        )
        self.assertIsNone(driver.active_motion)

        self.assertTrue(driver.request_motion("release", now=3.0))
        self.assertEqual(arm.commands[-1][0], [10, 20, 30, 40, 50, 60])

    def test_rejects_non_hand_end_effector(self):
        with self.assertRaises(BrainCoHandUnavailableError):
            BrainCoHandDriver(
                FakeArm(hand_type=1),
                retry_delay=0.0,
                mode_settle_delay=0.0,
            )

    def test_thumb_rotation_responds_immediately_after_initial_calibration(self):
        driver = BrainCoHandDriver(
            FakeArm(),
            retry_delay=0.0,
            mode_settle_delay=0.0,
            filter_cutoff_hz=0.0,
            dead_zone=0.0,
            thumb_rotate_progress_range=1.2,
        )
        initial = straight_skeleton()
        moved = initial.copy()
        moved[5, 0] += 0.002

        initial_joints = driver.map_openxr_hand(initial)
        moved_joints = driver.map_openxr_hand(moved)

        self.assertEqual(initial_joints[5], 0.0)
        self.assertGreater(moved_joints[5], 0.04)
        self.assertLess(moved_joints[5], 0.043)


class BrainCoMotionFilterTest(unittest.TestCase):
    def test_dead_zone_rejects_micro_motion_per_joint(self):
        motion_filter = BrainCoHandMotionFilter(
            cutoff_hz=0.0,
            dead_zone=0.05,
            initial=np.zeros(6),
        )

        held = motion_filter.update([0.04, 0.0, 0.0, 0.0, 0.0, 0.0], 0.01)
        accepted = motion_filter.update(
            [0.06, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.01,
        )

        np.testing.assert_allclose(held, np.zeros(6))
        self.assertAlmostEqual(accepted[0], 0.06)
        np.testing.assert_allclose(accepted[1:], np.zeros(5))

    def test_low_pass_filter_uses_elapsed_time(self):
        motion_filter = BrainCoHandMotionFilter(
            cutoff_hz=1.0,
            dead_zone=0.0,
            initial=np.zeros(6),
        )

        filtered = motion_filter.update(np.ones(6), 0.1)

        expected = 1.0 - np.exp(-2.0 * np.pi * 0.1)
        np.testing.assert_allclose(filtered, np.full(6, expected))


class TcpToolConfigTest(unittest.TestCase):
    @staticmethod
    def _fake_spatial_modules():
        package = types.ModuleType("airo_spatial_algebra")
        se3 = types.ModuleType("airo_spatial_algebra.se3")
        se3.SE3Container = object
        return {
            "airo_spatial_algebra": package,
            "airo_spatial_algebra.se3": se3,
        }

    def test_hand_is_kept_for_realman_hand_tracking(self):
        with patch.dict(sys.modules, self._fake_spatial_modules()):
            cfg = Config(TCP_TOOL="hand", TRACKING_MODE="HAND")

        self.assertEqual(cfg.TCP_TOOL, "Hand")
        self.assertFalse(cfg.GRIPPER)

    def test_hand_is_kept_for_realman_controller_tracking(self):
        with patch.dict(sys.modules, self._fake_spatial_modules()):
            cfg = Config(TCP_TOOL="hand", TRACKING_MODE="controller")

        self.assertEqual(cfg.TCP_TOOL, "Hand")
        self.assertFalse(cfg.GRIPPER)

    def test_unsupported_hand_combination_falls_back_to_none(self):
        with patch.dict(sys.modules, self._fake_spatial_modules()):
            cfg = Config(
                ROBOT_TYPE="ur3e",
                ROBOT_IP="192.0.2.1",
                TCP_TOOL="Hand",
                TRACKING_MODE="hand",
            )

        self.assertEqual(cfg.TCP_TOOL, "None")
        self.assertFalse(cfg.GRIPPER)

    def test_gripper_tool_updates_legacy_flag(self):
        with patch.dict(sys.modules, self._fake_spatial_modules()):
            cfg = Config(TCP_TOOL="gripper")

        self.assertEqual(cfg.TCP_TOOL, "Gripper")
        self.assertTrue(cfg.GRIPPER)


if __name__ == "__main__":
    unittest.main()
