"""Hardware-free tests for learned WRM_wrap execution monitors."""

from __future__ import annotations

import unittest

import torch

from policies.realman_beaver.configuration import ModelConfig, RealmanBeaverConfig
from policies.realman_beaver.dataset import ObservationNormalizer
from policies.realman_beaver.modeling import build_policy
from policies.realman_beaver.modeling_wrm_wrap_monitor import (
    MonitorWrapBeaverDPPolicy,
)
from policies.realman_beaver.modules.beaver_monitor import (
    BackupBeaverMonitor,
    TemporalBeaverMonitor,
)


def tiny_monitor_config(variant: str) -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant=variant,
            n_obs_steps=2,
            horizon=8,
            n_action_steps=2,
            resize_shape=(64, 64),
            crop_ratio=1.0,
            down_dims=(32, 64),
            n_groups=8,
            diffusion_step_embed_dim=32,
            num_train_timesteps=8,
            num_inference_steps=2,
        )
    )
    config.dataset.image_shape = (3, 64, 64)
    config.validate()
    return config


class TemporalMonitorTest(unittest.TestCase):
    def test_feature_shape_and_all_nine_sensors(self) -> None:
        monitor = TemporalBeaverMonitor()
        distance = torch.full((2, 12, 9, 4, 4), 100.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(2, 12, 9)
        before = monitor.extract_features(distance, status, present)
        distance[:, :, 8] = 0.0
        after = monitor.extract_features(distance, status, present)
        self.assertEqual(tuple(before.shape), (2, 270))
        self.assertFalse(torch.equal(before, after))

    def test_invalid_zero_is_not_contact(self) -> None:
        monitor = TemporalBeaverMonitor()
        distance = torch.zeros(1, 12, 9, 4, 4)
        status = torch.full_like(distance, 255.0)
        present = torch.ones(1, 12, 9)
        frame = monitor.extract_frame_features(distance, status, present)
        # Per-sensor feature order: three proximity, valid, zero, minimum.
        self.assertTrue(torch.equal(frame[..., 3], torch.zeros_like(frame[..., 3])))
        self.assertTrue(torch.equal(frame[..., 4], torch.zeros_like(frame[..., 4])))


class BackupMonitorTest(unittest.TestCase):
    def _exact_monitor(self) -> BackupBeaverMonitor:
        monitor = BackupBeaverMonitor()
        with torch.no_grad():
            for parameter in monitor.parameters():
                parameter.zero_()
            monitor.mlp[0].weight[0].fill_(1.0)
            monitor.mlp[2].weight[:, 0] = 1.0
            monitor.mlp[2].bias.copy_(torch.tensor([-0.5, -1.5]))
        return monitor

    def test_exhaustive_parameter_truth_table(self) -> None:
        monitor = self._exact_monitor()
        patterns = torch.tensor(
            [[(value >> bit) & 1 for bit in range(9)] for value in range(512)],
            dtype=torch.float32,
        )
        key4 = patterns[:, [1, 2, 5, 6]]
        predicted = monitor.mlp(key4) >= 0
        count = key4.sum(dim=-1)
        truth = torch.stack((count >= 1, count >= 2), dim=-1)
        self.assertTrue(torch.equal(predicted, truth))

    def test_other_five_sensors_are_structurally_ignored(self) -> None:
        monitor = BackupBeaverMonitor()
        distance = torch.full((1, 12, 9, 4, 4), 100.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 12, 9)
        baseline = monitor.extract_features(distance, status, present)
        for sensor in (0, 3, 4, 7, 8):
            distance[:, -1, sensor] = 0.0
            status[:, -1, sensor] = 9.0
            present[:, -1, sensor] = 0.0
        changed = monitor.extract_features(distance, status, present)
        self.assertTrue(torch.equal(baseline, changed))


class MonitorPolicyTest(unittest.TestCase):
    def test_builds_both_policy_variants(self) -> None:
        for variant in ("WRM_wrap_monitor", "WRM_wrap_monitor_backup"):
            with self.subTest(variant=variant):
                policy = build_policy(
                    tiny_monitor_config(variant), ObservationNormalizer.identity()
                )
                self.assertIsInstance(policy, MonitorWrapBeaverDPPolicy)
                self.assertEqual(
                    policy.native_policy.config.robot_state_feature.shape, (75,)
                )

    def test_monitor_inherits_parent_wrap_gate(self) -> None:
        """MonitorWrapBeaverDPPolicy uses the parent WRM_wrap gate, not a
        monitor-based override.  Verify that the inherited gate blocks lift
        when sensors report far distances (no wrap progress)."""
        policy = build_policy(
            tiny_monitor_config("WRM_wrap_monitor"), ObservationNormalizer.identity()
        )
        current = torch.tensor([[0.0, 1.4, 1.5, -1.1, -0.2, -1.6, 0.0]])
        action = current + 0.1
        n_sensors = len(policy.config.model.beaver_wrap_sensors)
        # Far sensors → no wrap → lift should be blocked
        sensor_min_far = torch.full((1, n_sensors), 200.0)
        blocked = policy._apply_wrap_lift_gate(
            action,
            {"state": current},
            {"current_sensor_min_mm": sensor_min_far},
        )
        # Lift (J1) held at current joint value
        self.assertEqual(float(blocked[0, 1]), float(current[0, 1]))
        # Non-gated joints (e.g. J2) pass through unchanged
        self.assertAlmostEqual(float(blocked[0, 2]), float(action[0, 2]))


if __name__ == "__main__":
    unittest.main()
