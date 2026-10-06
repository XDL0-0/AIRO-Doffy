"""Regression checks for calibration when OpenXR tracking is incomplete."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doffy_teleop.robots.brainco_hand import BrainCoHandDriver, THUMB_ROTATE


class FakeArm:
    def rm_get_rm_plus_mode(self):
        return 0, 460800

    def rm_get_rm_plus_base_info(self):
        return 0, {
            "type": 2,
            "dof": 6,
            "pos_low": [0] * 6,
            "pos_up": [1000] * 6,
        }

    def rm_get_rm_plus_state_info(self):
        return 0, {"pos": [0] * 6}


def open_hand():
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


def incomplete_hand():
    bones = open_hand()
    # Valid thumb/palm geometry reaches calibration; a collapsed index segment
    # is detected only by the subsequent complete motor mapping.
    bones[5, 0] += 0.02
    bones[8] = bones[7]
    return bones


class BrainCoCalibrationTest(unittest.TestCase):
    @staticmethod
    def driver(progress_range=1.2):
        return BrainCoHandDriver(
            FakeArm(),
            retry_delay=0.0,
            mode_settle_delay=0.0,
            thumb_rotate_progress_range=progress_range,
        )

    def test_rejected_first_frame_does_not_calibrate_thumb_rotation(self):
        for progress_range in (0.6, 1.2):
            with self.subTest(progress_range=progress_range):
                driver = self.driver(progress_range)

                with self.assertRaisesRegex(ValueError, "zero-length finger segment"):
                    driver.map_openxr_hand(incomplete_hand())

                self.assertIsNone(driver._thumb_rotate_open_progress)
                self.assertEqual(driver.map_openxr_hand(open_hand())[THUMB_ROTATE], 0.0)

                moved = open_hand()
                moved[5, 0] += 0.012
                # Palm width is 0.04 m, so the movement is 0.3 palm widths.
                self.assertAlmostEqual(
                    driver.map_openxr_hand(moved)[THUMB_ROTATE],
                    0.3 / progress_range,
                )

    def test_rejected_frame_keeps_existing_calibration(self):
        driver = self.driver()
        driver.map_openxr_hand(open_hand())
        baseline = driver._thumb_rotate_open_progress

        with self.assertRaisesRegex(ValueError, "zero-length finger segment"):
            driver.map_openxr_hand(incomplete_hand())

        self.assertEqual(driver._thumb_rotate_open_progress, baseline)
        self.assertEqual(driver.map_openxr_hand(open_hand())[THUMB_ROTATE], 0.0)
        moved = open_hand()
        moved[5, 0] += 0.012
        self.assertAlmostEqual(driver.map_openxr_hand(moved)[THUMB_ROTATE], 0.25)


if __name__ == "__main__":
    unittest.main()
