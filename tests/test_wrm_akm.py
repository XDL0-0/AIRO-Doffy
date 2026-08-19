import json
import time
import unittest

import numpy as np

from wrm_akm import (
    Rm75AkmSettings,
    Rm75ArmAngleIk,
    WrmTrackingSample,
    parse_wrm_unity_packet,
)


class FakeIkParams:
    def __init__(self, q_in, pose, flag):
        self.q_in = list(q_in)
        self.pose = list(pose)
        self.flag = flag


class FakeApi:
    rm_inverse_kinematics_params_t = FakeIkParams


class FakeRm75Arm:
    def __init__(self):
        self.seeds = []
        self.self_collision_enabled = False

    def rm_set_self_collision_enable(self, enabled):
        self.self_collision_enabled = bool(enabled)
        return 0

    def rm_algo_calculate_arm_angle_from_config_rm75(self, joints_deg):
        return 0, 40.0

    def rm_algo_inverse_kinematics_rm75_for_arm_angle(self, params, arm_angle):
        self.seeds.append(np.radians(np.asarray(params.q_in, dtype=float)))
        # Keep the known-safe initial geometry.  The test focuses on API
        # units, seed continuity, and abstract target plumbing.
        output = np.asarray(params.q_in, dtype=float)
        output[2] += 0.001 * (float(arm_angle) - 40.0)
        return 0, output.tolist()


def sample(alpha, confidence=1.0, received_ns=None):
    return WrmTrackingSample(
        frame_id=1,
        timestamp_ns=2,
        received_ns=time.monotonic_ns() if received_ns is None else received_ns,
        elbow_alpha=alpha,
        confidence=confidence,
    )


class WrmPacketTest(unittest.TestCase):
    def test_json_controller_and_elbow_packet(self):
        parsed = parse_wrm_unity_packet(
            json.dumps(
                {
                    "type": "WRM",
                    "frameId": 7,
                    "timestampNs": 9,
                    "position": [1, 2, 3],
                    "rotation": [0, 0, 0, 2],
                    "elbowAlpha": 0.25,
                    "trackingConfidence": 0.8,
                    "gripTrigger": 1,
                }
            ),
            received_ns=11,
        )
        self.assertEqual(parsed.frame_id, 7)
        self.assertEqual(parsed.received_ns, 11)
        self.assertAlmostEqual(parsed.elbow_alpha, 0.25)
        self.assertEqual(parsed.rotation_xyzw, (0.0, 0.0, 0.0, 1.0))
        controller = parsed.as_controller_data()
        self.assertEqual(controller[1]["Position"], (1.0, 2.0, 3.0))
        self.assertEqual(controller[1]["GripTrigger"], 1.0)

    def test_elbow_only_csv(self):
        parsed = parse_wrm_unity_packet("WRM,10,20,0.75,0.9", received_ns=30)
        self.assertFalse(parsed.has_controller_pose)
        self.assertAlmostEqual(parsed.elbow_alpha, 0.75)

    def test_rejects_out_of_range_alpha(self):
        with self.assertRaises(ValueError):
            parse_wrm_unity_packet('{"elbow_alpha":1.1}')


class Rm75ArmAngleIkTest(unittest.TestCase):
    def setUp(self):
        self.initial = np.array([2.2, -0.26, 0.1, -1.17, -0.11, -0.89, -0.5])
        self.arm = FakeRm75Arm()
        self.ik = Rm75ArmAngleIk(
            self.arm,
            FakeApi,
            self.initial,
            np.eye(4),
            settings=Rm75AkmSettings(
                calibration_span_deg=5.0,
                calibration_step_deg=5.0,
                minimum_elbow_height_m=0.20,
                maximum_arm_angle_rate_deg_s=10000.0,
            ),
        )

    def test_enables_controller_self_collision(self):
        self.assertTrue(self.arm.self_collision_enabled)

    def test_inverse_abstract_lerp_and_confidence_freeze(self):
        self.ik.robot_elbow_high = 50.0
        self.ik.robot_elbow_horizontal = -10.0
        self.ik._filtered_target = 50.0
        self.ik._last_target_update_ns = time.monotonic_ns() - 1_000_000_000
        self.assertTrue(self.ik.update_tracking(sample(0.25)))
        self.assertAlmostEqual(self.ik.elbow_target_deg, 35.0)
        frozen = self.ik.elbow_target_deg
        self.assertFalse(self.ik.update_tracking(sample(0.9, confidence=0.1)))
        self.assertEqual(self.ik.elbow_target_deg, frozen)
        visualizer = self.ik.visualizer_state()
        self.assertAlmostEqual(visualizer["elbow_alpha"], 0.9)
        self.assertAlmostEqual(visualizer["confidence"], 0.1)
        self.assertTrue(visualizer["tracking_frozen"])
        self.assertAlmostEqual(visualizer["arm_angle_target_deg"], frozen)

    def test_accepted_previous_solution_is_next_seed(self):
        first = self.ik.solve(np.eye(4))
        self.assertIsNotNone(first)
        accepted = self.initial.copy()
        accepted[6] += 0.03
        self.ik.accept_solution(accepted)
        self.ik.solve(np.eye(4))
        np.testing.assert_allclose(self.arm.seeds[-1], accepted, atol=1e-12)

    def test_stale_tracking_freezes_elbow_only(self):
        stale_ns = time.monotonic_ns() - 1_000_000_000
        before = self.ik.elbow_target_deg
        self.assertFalse(self.ik.update_tracking(sample(1.0, received_ns=stale_ns)))
        self.assertEqual(self.ik.elbow_target_deg, before)
        self.assertTrue(self.ik.tracking_frozen)

    def test_reflected_rotation_matrix_is_rejected(self):
        mirrored = np.eye(4)
        mirrored[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "reflected/mirrored"):
            self.ik.solve(mirrored)

    def test_increasing_progress_lowers_tcp_z_from_clutch_reference(self):
        self.ik.robot_elbow_high = 50.0
        self.ik.robot_elbow_horizontal = -10.0
        self.ik._filtered_target = 35.0  # progress 0.25
        self.ik.set_tcp_z_reference()

        target = np.eye(4)
        target[2, 3] = 0.40
        self.ik._filtered_target = 5.0  # progress 0.75, delta +0.50
        coupled = self.ik.couple_tcp_target_z(target, 0.06)

        self.assertAlmostEqual(coupled[2, 3], 0.37)
        self.assertAlmostEqual(target[2, 3], 0.40)  # input is not mutated
        self.assertAlmostEqual(self.ik.arm_angle_progress, 0.75)

    def test_tcp_z_coupling_is_reversible_and_does_not_accumulate(self):
        self.ik.robot_elbow_high = 50.0
        self.ik.robot_elbow_horizontal = -10.0
        target = np.eye(4)
        target[2, 3] = 0.40

        self.ik._filtered_target = 20.0  # progress 0.50
        self.ik.set_tcp_z_reference()
        np.testing.assert_allclose(
            self.ik.couple_tcp_target_z(target, 0.05),
            target,
        )

        self.ik._filtered_target = -10.0  # progress 1.00
        self.assertAlmostEqual(
            self.ik.couple_tcp_target_z(target, 0.05)[2, 3],
            0.375,
        )

        # Releasing/re-gripping at the current posture establishes a new zero
        # and cannot apply the existing 2.5 cm drop a second time.
        self.ik.set_tcp_z_reference()
        np.testing.assert_allclose(
            self.ik.couple_tcp_target_z(target, 0.05),
            target,
        )

        self.ik._filtered_target = 20.0  # progress decreases by 0.50
        self.assertAlmostEqual(
            self.ik.couple_tcp_target_z(target, 0.05)[2, 3],
            0.425,
        )

    def test_tcp_z_coupling_rejects_invalid_drop(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.ik.couple_tcp_target_z(np.eye(4), -0.01)


if __name__ == "__main__":
    unittest.main()
