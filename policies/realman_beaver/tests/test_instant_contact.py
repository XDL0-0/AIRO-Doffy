"""Tests for the current-frame contact classifier."""

from __future__ import annotations

import unittest

import torch

from policies.realman_beaver.modules.instant_contact import InstantContactMonitor


class InstantContactMonitorTest(unittest.TestCase):
    def test_key4_feature_dim_and_current_frame_only(self) -> None:
        monitor = InstantContactMonitor(use_joints=False)
        distance = torch.full((2, 9, 4, 4), 80.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(2, 9)
        features = monitor.extract_features(distance, status, present)
        self.assertEqual(tuple(features.shape), (2, 40))
        history = distance.unsqueeze(1).repeat(1, 4, 1, 1, 1)
        history[:, :-1] = 400.0
        from_history = monitor.extract_features(
            history, status.unsqueeze(1).repeat(1, 4, 1, 1, 1), present.unsqueeze(1).repeat(1, 4, 1)
        )
        self.assertTrue(torch.allclose(features, from_history, atol=1e-5))

    def test_invalid_zero_is_not_contact_feature(self) -> None:
        monitor = InstantContactMonitor(use_joints=False)
        distance = torch.zeros(1, 9, 4, 4)
        status = torch.full_like(distance, 255.0)
        present = torch.ones(1, 9)
        features = monitor.extract_beaver_features(distance, status, present)
        # Per sensor: 3 mean prox + 3 best prox + near + zero + valid + min. Zero-fraction is index 7.
        zero_fraction = features.reshape(1, 4, 10)[..., 7]
        self.assertTrue(torch.equal(zero_fraction, torch.zeros_like(zero_fraction)))

    def test_joints_require_state_and_change_the_logit(self) -> None:
        monitor = InstantContactMonitor(use_joints=True)
        distance = torch.full((2, 9, 4, 4), 10.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(2, 9)
        with self.assertRaises(ValueError):
            monitor(distance, status, present)
        joints = torch.zeros(2, 7)
        first = monitor(distance, status, present, joints)
        joints[:, 3] = 1.0
        second = monitor(distance, status, present, joints)
        self.assertEqual(tuple(first.shape), (2,))
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
