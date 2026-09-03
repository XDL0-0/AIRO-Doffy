"""Tests for WRM_wrap contact-preserving encoder, gates, and stratified batches."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import ModelConfig, RealmanBeaverConfig
from policies.realman_beaver.dataset import (
    BottleStratifiedBatchSampler,
    ObservationNormalizer,
    bottle_id_from_episode,
)
from policies.realman_beaver.modeling import build_policy
from policies.realman_beaver.modeling_wrm_wrap import WrapBeaverDPPolicy
from policies.realman_beaver.modules.wrap_beaver_encoder import WrapBeaverEncoder


def tiny_wrap_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant="WRM_wrap",
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
    config.dataset.stratified_bottle_batch = True
    config.validate()
    return config


class WrapBeaverEncoderTest(unittest.TestCase):
    def test_zero_distance_with_valid_status_is_contact_not_missing(self) -> None:
        encoder = WrapBeaverEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=12
        )
        distance = torch.full((1, 12, 9, 4, 4), 150.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 12, 9)
        distance[0, -1, 1, 0, 0] = 0.0
        features, middle = encoder.preprocess(distance, status, present)
        self.assertTrue(bool(middle["genuine"][0, -1, 0, 0, 0]))
        self.assertTrue(bool(middle["contact_zero"][0, -1, 0, 0, 0]))
        self.assertAlmostEqual(float(features[0, -1, 0, 0, 0, 0]), 1.0, places=5)

    def test_near_threshold_includes_quantized_boundary(self) -> None:
        encoder = WrapBeaverEncoder(
            n_sensors=9,
            sensor_indices=(1, 2, 5, 6),
            history_steps=12,
            near_threshold_mm=10.0,
        )
        distance = torch.full((1, 12, 9, 4, 4), 20.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 12, 9)
        distance[0, -1, 1] = 10.0

        _, middle = encoder.preprocess(distance, status, present)

        self.assertAlmostEqual(float(middle["wrap_progress"][0, -1]), 0.25)

    def test_zero_near_threshold_accepts_only_valid_zero_bin(self) -> None:
        encoder = WrapBeaverEncoder(
            n_sensors=9,
            sensor_indices=(1, 2, 5, 6),
            history_steps=12,
            near_threshold_mm=0.0,
            closing_scale_mm=10.0,
        )
        distance = torch.full((1, 12, 9, 4, 4), 10.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 12, 9)
        distance[0, -1, 1] = 0.0

        _, middle = encoder.preprocess(distance, status, present)

        self.assertAlmostEqual(float(middle["wrap_progress"][0, -1]), 0.25)
        self.assertTrue(torch.isfinite(middle["enclosure"]).all())

    def test_invalid_status_does_not_contribute_proximity(self) -> None:
        encoder = WrapBeaverEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=12
        )
        distance = torch.full((1, 12, 9, 4, 4), 20.0)
        status = torch.full_like(distance, 255.0)
        present = torch.ones(1, 12, 9)
        features, middle = encoder.preprocess(distance, status, present)
        self.assertFalse(bool(middle["genuine"].any()))
        self.assertTrue(torch.allclose(features[..., :3], torch.zeros_like(features[..., :3])))
        self.assertGreaterEqual(float(middle["min_range_mm"][0, -1]), 300.0 - 1e-4)

    def test_encoder_outputs_feature_and_enclosure(self) -> None:
        encoder = WrapBeaverEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=12
        )
        distance = torch.full((2, 2, 12, 9, 4, 4), 80.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(2, 2, 12, 9)
        feature, middle = encoder(
            distance, status, present, return_intermediates=True
        )
        self.assertEqual(tuple(feature.shape), (2, 2, 64))
        self.assertEqual(tuple(middle["current_enclosure"].shape), (2, 2, 4))
        self.assertEqual(tuple(middle["sensor_tokens"].shape), (2, 2, 4, 64))


class WrapBeaverPolicyTest(unittest.TestCase):
    def test_build_and_condition_dim(self) -> None:
        policy = build_policy(tiny_wrap_config(), ObservationNormalizer.identity())
        self.assertIsInstance(policy, WrapBeaverDPPolicy)
        self.assertEqual(
            policy.native_policy.config.robot_state_feature.shape, (75,)
        )

    def test_loss_backprop_and_contact_zero_metric(self) -> None:
        policy = build_policy(tiny_wrap_config(), ObservationNormalizer.identity())
        distance = torch.full((2, 2, 12, 9, 4, 4), 120.0)
        distance[:, :, -1, 1] = 0.0
        status = torch.full_like(distance, 5.0)
        batch = {
            "image": torch.rand(2, 2, 3, 64, 64),
            "state": torch.randn(2, 2, 7),
            "action": torch.randn(2, 8, 7),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "beaver_history_distance": distance,
            "beaver_history_status": status,
            "beaver_history_present": torch.ones(2, 2, 12, 9),
        }
        loss, metrics = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["contact_zero_fraction"], 0.0)
        loss.backward()
        grad_norm = sum(
            parameter.grad.abs().sum()
            for parameter in policy.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(float(grad_norm), 0.0)

    def test_low_wrap_blocks_lift_joint(self) -> None:
        policy = build_policy(tiny_wrap_config(), ObservationNormalizer.identity())
        policy.eval()
        state = torch.tensor([[0.0, 1.46, 1.58, -1.0, -0.2, -1.4, 0.1]])
        observation = {
            "image": torch.rand(3, 64, 64),
            "state": state[0],
            "beaver_distance": torch.full((9, 4, 4), 400.0),
            "beaver_status": torch.full((9, 4, 4), 255.0),
            "beaver_present": torch.ones(9),
        }
        action = policy.select_action(observation)
        self.assertEqual(tuple(action.shape), (1, 7))
        self.assertAlmostEqual(float(action[0, 1]), 1.46, places=4)
        self.assertGreater(policy.last_lift_blocked, 0.5)

    def test_contact_stops_further_closure(self) -> None:
        config = tiny_wrap_config()
        config.model.beaver_wrap_contact_stop_mm = 0.0
        config.validate()
        policy = build_policy(config, ObservationNormalizer.identity())
        policy.eval()
        state = torch.tensor([[0.0, 1.46, 1.58, -1.50, -0.29, -1.91, 0.1]])
        distance = torch.full((9, 4, 4), 400.0)
        status = torch.full((9, 4, 4), 255.0)
        for sensor in (1, 2, 5, 6):
            distance[sensor] = 0.0
            status[sensor] = 5.0
        observation = {
            "image": torch.rand(3, 64, 64),
            "state": state[0],
            "beaver_distance": distance,
            "beaver_status": status,
            "beaver_present": torch.ones(9),
        }
        action = policy.select_action(observation)
        torch.testing.assert_close(action[0, 3:6], state[0, 3:6], atol=1e-5, rtol=0)
        self.assertGreater(policy.last_close_stopped, 0.5)
        self.assertGreater(policy.last_close_stopped_j3, 0.5)
        self.assertGreater(policy.last_close_stopped_j4, 0.5)

    def test_j3_and_j4_stop_independently_from_lift(self) -> None:
        config = tiny_wrap_config()
        config.model.beaver_wrap_near_threshold_mm = 0.0
        config.model.beaver_wrap_lift_min_wrap = 0.25
        config.model.beaver_wrap_stop_close_j3_wrap = 1.0
        config.model.beaver_wrap_stop_close_j4_wrap = 0.5
        config.model.beaver_wrap_contact_stop_mm = 0.0
        config.validate()
        policy = build_policy(config, ObservationNormalizer.identity())

        current = torch.tensor([[0.0, 1.0, 1.5, -1.0, -0.2, -1.6, 0.0]])
        action = current + 0.1
        # One contact in each group: J4 reaches its 0.5 threshold, while J3
        # still needs both of its sensors because its threshold is 1.0. Lift
        # is nevertheless released because its own lift_min=0.25 is reached.
        one_each = policy._apply_wrap_lift_gate(
            action,
            {"state": current},
            {
                "current_wrap_progress": torch.tensor([0.5]),
                "current_sensor_min_mm": torch.tensor([[0.0, 20.0, 0.0, 20.0]]),
            },
        )
        torch.testing.assert_close(one_each[0, 3], action[0, 3])
        torch.testing.assert_close(one_each[0, 4], current[0, 4])
        torch.testing.assert_close(one_each[0, 1], action[0, 1])
        self.assertAlmostEqual(policy.last_close_stopped_j3, 0.0)
        self.assertAlmostEqual(policy.last_close_stopped_j4, 1.0)
        self.assertAlmostEqual(policy.last_lift_blocked, 0.0)

        # Once both groups satisfy their own thresholds, J3/J4 (and the
        # preserved J5 full-closure guard) freeze. Lift remains independent.
        both = policy._apply_wrap_lift_gate(
            action,
            {"state": current},
            {
                "current_wrap_progress": torch.tensor([1.0]),
                "current_sensor_min_mm": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            },
        )
        torch.testing.assert_close(both[0, 3:6], current[0, 3:6])
        torch.testing.assert_close(both[0, 1], action[0, 1])
        self.assertAlmostEqual(policy.last_close_stopped, 1.0)
        self.assertAlmostEqual(policy.last_close_stopped_j3, 1.0)
        self.assertAlmostEqual(policy.last_close_stopped_j4, 1.0)

    def test_large_bottle_pattern_lifts_without_sensor_11(self) -> None:
        config = tiny_wrap_config()
        config.model.beaver_wrap_stop_close_j3_wrap = 0.5
        config.model.beaver_wrap_stop_close_j4_wrap = 0.5
        config.model.beaver_wrap_contact_stop_mm = 10.0
        config.model.beaver_wrap_lift_min_wrap = 0.8
        config.validate()
        policy = build_policy(config, ObservationNormalizer.identity())
        current = torch.tensor([[0.0, 1.0, 1.5, -1.0, -0.2, -1.6, 0.0]])
        action = current + 0.1
        # Bottle-5 geometry at tightness: 01=10, 02=20, 10=10, 11=50.
        gated = policy._apply_wrap_lift_gate(
            action,
            {"state": current},
            {
                "current_sensor_min_mm": torch.tensor([[10.0, 20.0, 10.0, 50.0]]),
            },
        )
        torch.testing.assert_close(gated[0, 1], action[0, 1])
        torch.testing.assert_close(gated[0, 3:6], current[0, 3:6])
        self.assertAlmostEqual(policy.last_jaw_wrap, 1.0)
        self.assertAlmostEqual(policy.last_lift_blocked, 0.0)
        self.assertAlmostEqual(policy.last_close_stopped, 1.0)

    def test_enclosure_hold_delays_stop_and_lift(self) -> None:
        config = tiny_wrap_config()
        config.model.beaver_wrap_stop_close_j3_wrap = 0.5
        config.model.beaver_wrap_stop_close_j4_wrap = 0.5
        config.model.beaver_wrap_contact_stop_mm = 10.0
        config.model.beaver_wrap_lift_min_wrap = 0.8
        config.model.beaver_wrap_stop_hold_frames = 2
        config.model.beaver_wrap_lift_hold_frames = 4
        config.validate()
        policy = build_policy(config, ObservationNormalizer.identity())
        current = torch.tensor([[0.0, 1.0, 1.5, -1.0, -0.2, -1.6, 0.0]])
        action = current + 0.1

        def apply() -> Tensor:
            return policy._apply_wrap_lift_gate(
                action,
                {"state": current},
                {
                    "current_sensor_min_mm": torch.tensor(
                        [[10.0, 10.0, 0.0, 20.0]]
                    ).clone(),
                },
            )

        first = apply()
        torch.testing.assert_close(first[0, 1], current[0, 1])
        torch.testing.assert_close(first[0, 3], action[0, 3])
        self.assertEqual(policy.last_enclosed_hold, 1.0)

        second = apply()
        torch.testing.assert_close(second[0, 1], current[0, 1])
        torch.testing.assert_close(second[0, 3:6], current[0, 3:6])
        self.assertEqual(policy.last_enclosed_hold, 2.0)
        self.assertAlmostEqual(policy.last_close_stopped, 1.0)
        self.assertAlmostEqual(policy.last_lift_blocked, 1.0)

        apply()
        fourth = apply()
        torch.testing.assert_close(fourth[0, 1], action[0, 1])
        self.assertEqual(policy.last_enclosed_hold, 4.0)
        self.assertAlmostEqual(policy.last_lift_blocked, 0.0)

    def test_partial_wrap_does_not_stop_closure(self) -> None:
        policy = build_policy(tiny_wrap_config(), ObservationNormalizer.identity())
        policy.eval()
        distance = torch.full((9, 4, 4), 400.0)
        status = torch.full((9, 4, 4), 255.0)
        for sensor in (1, 2):
            distance[sensor] = 0.0
            status[sensor] = 5.0
        observation = {
            "image": torch.rand(3, 64, 64),
            "state": torch.tensor([0.0, 1.46, 1.58, -1.50, -0.29, -1.91, 0.1]),
            "beaver_distance": distance,
            "beaver_status": status,
            "beaver_present": torch.ones(9),
        }

        policy.select_action(observation)

        self.assertAlmostEqual(policy.last_wrap_progress, 0.5)
        self.assertLess(policy.last_close_stopped, 0.5)

    def test_checkpoint_roundtrip(self) -> None:
        config = tiny_wrap_config()
        policy = build_policy(config, ObservationNormalizer.identity())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrap.pt"
            torch.save(
                {
                    "kind": "WRM_wrap",
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": {},
                },
                path,
            )
            restored = load_policy(path)
        self.assertIsInstance(restored, WrapBeaverDPPolicy)
        self.assertEqual(
            restored.native_policy.config.robot_state_feature.shape, (75,)
        )

    def test_checkpoint_gate_settings_can_be_overridden_for_deployment(self) -> None:
        config = tiny_wrap_config()
        config.model.beaver_wrap_near_threshold_mm = 50.0
        config.model.beaver_wrap_range_scale_mm = 300.0
        config.model.beaver_wrap_lift_min_wrap = 0.5
        config.model.beaver_wrap_stop_close_wrap = 0.5
        config.model.beaver_wrap_contact_stop_mm = 40.0
        policy = build_policy(config, ObservationNormalizer.identity())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrap.pt"
            torch.save(
                {
                    "kind": "WRM_wrap",
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": {},
                },
                path,
            )
            restored = load_policy(
                path,
                wrap_near_threshold_mm=10.0,
                wrap_range_scale_mm=300.0,
                wrap_lift_min_wrap=0.8,
                wrap_stop_close_wrap=1.0,
                wrap_contact_stop_mm=5.0,
                wrap_stop_hold_frames=24,
                wrap_lift_hold_frames=48,
            )

        model = restored.config.model
        self.assertEqual(model.beaver_wrap_near_threshold_mm, 10.0)
        self.assertEqual(model.beaver_wrap_range_scale_mm, 300.0)
        self.assertEqual(model.beaver_wrap_lift_min_wrap, 0.8)
        self.assertEqual(model.beaver_wrap_stop_close_wrap, 1.0)
        self.assertEqual(model.beaver_wrap_stop_close_j3_wrap, 1.0)
        self.assertEqual(model.beaver_wrap_stop_close_j4_wrap, 1.0)
        self.assertEqual(model.beaver_wrap_contact_stop_mm, 5.0)
        self.assertEqual(model.beaver_wrap_stop_hold_frames, 24)
        self.assertEqual(model.beaver_wrap_lift_hold_frames, 48)
        self.assertEqual(restored.beaver_encoder.near_threshold_mm, 10.0)
        self.assertEqual(restored.beaver_encoder.closing_scale_mm, 10.0)

    def test_legacy_checkpoint_preserves_closing_scale_when_near_is_overridden(
        self,
    ) -> None:
        config = tiny_wrap_config()
        config.model.beaver_wrap_near_threshold_mm = 50.0
        policy = build_policy(config, ObservationNormalizer.identity())
        raw_config = config.to_dict()
        for key in (
            "beaver_wrap_closing_scale_mm",
            "beaver_wrap_j3_sensors",
            "beaver_wrap_j4_sensors",
            "beaver_wrap_stop_close_j3_wrap",
            "beaver_wrap_stop_close_j4_wrap",
            "beaver_wrap_stop_hold_frames",
            "beaver_wrap_lift_hold_frames",
        ):
            del raw_config["model"][key]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_wrap.pt"
            torch.save(
                {
                    "kind": "WRM_wrap",
                    "config": raw_config,
                    "model": policy.state_dict(),
                    "ema": {},
                },
                path,
            )
            restored = load_policy(path, wrap_near_threshold_mm=0.0)

        self.assertEqual(restored.config.model.beaver_wrap_near_threshold_mm, 0.0)
        self.assertEqual(restored.config.model.beaver_wrap_closing_scale_mm, 50.0)
        self.assertEqual(restored.beaver_encoder.near_threshold_mm, 0.0)
        self.assertEqual(restored.beaver_encoder.closing_scale_mm, 50.0)


class BottleStratifiedBatchSamplerTest(unittest.TestCase):
    def test_each_batch_contains_every_bottle(self) -> None:
        episodes = np.concatenate(
            [np.full(20, bottle * 25, dtype=np.int64) for bottle in range(5)]
        )
        bottle_ids = bottle_id_from_episode(episodes, 25)
        sampler = BottleStratifiedBatchSampler(bottle_ids, batch_size=10, seed=0)
        batches = list(sampler)
        self.assertGreater(len(batches), 0)
        for batch in batches:
            self.assertEqual(len(batch), 10)
            bottles = {int(bottle_ids[index]) for index in batch}
            self.assertEqual(bottles, {0, 1, 2, 3, 4})

    def test_bottle_ids_follow_episode_blocks(self) -> None:
        ids = bottle_id_from_episode(np.array([0, 24, 25, 49, 100, 124]), 25)
        np.testing.assert_array_equal(ids, np.array([0, 0, 1, 1, 4, 4]))
