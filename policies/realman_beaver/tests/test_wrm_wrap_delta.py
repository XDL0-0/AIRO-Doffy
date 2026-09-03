"""Focused tests for the WRM_wrap_delta output-representation ablation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import Tensor

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import (
    WRAP_DELTA_BEAVER_VARIANT,
    ModelConfig,
    RealmanBeaverConfig,
    load_config,
)
from policies.realman_beaver.dataset import ObservationNormalizer
from policies.realman_beaver.modeling import build_policy
from policies.realman_beaver.modeling_wrm_wrap_delta import (
    WrapDeltaBeaverDPPolicy,
)


def tiny_wrap_delta_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant=WRAP_DELTA_BEAVER_VARIANT,
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


def delta_normalizer() -> ObservationNormalizer:
    return ObservationNormalizer.identity(
        delta_action_statistics={
            "offset": torch.zeros(7),
            "scale": torch.ones(7),
        }
    )


def training_batch(batch_size: int = 2) -> dict[str, Tensor]:
    anchor = torch.linspace(-0.3, 0.3, 7).repeat(batch_size, 1)
    state = torch.stack((anchor - 0.01, anchor), dim=1)
    action = anchor[:, None, :] + torch.randn(batch_size, 8, 7) * 0.05
    distance = torch.full((batch_size, 2, 12, 9, 4, 4), 120.0)
    return {
        "image": torch.rand(batch_size, 2, 3, 64, 64),
        "state": state,
        "action": action,
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
        "beaver_history_distance": distance,
        "beaver_history_status": torch.full_like(distance, 5.0),
        "beaver_history_present": torch.ones(batch_size, 2, 12, 9),
    }


def online_observation(state: Tensor) -> dict[str, Tensor]:
    distance = torch.full((1, 9, 4, 4), 80.0)
    return {
        "image": torch.rand(1, 3, 64, 64),
        "state": state,
        "beaver_distance": distance,
        "beaver_status": torch.full_like(distance, 5.0),
        "beaver_present": torch.ones(1, 9),
    }


class WrapDeltaConfigTest(unittest.TestCase):
    def test_production_config_is_stratified_100_25(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "WRM_wrap_delta.yaml"
        config = load_config(path)
        self.assertEqual(config.model.variant, WRAP_DELTA_BEAVER_VARIANT)
        self.assertTrue(config.dataset.stratified_bottle_batch)
        self.assertEqual(
            tuple(config.dataset.val_episodes or ()),
            tuple(
                episode
                for start in (20, 45, 70, 95, 120)
                for episode in range(start, start + 5)
            ),
        )
        self.assertEqual(config.training.checkpoint_every_steps, 10_000)

    def test_build_requires_relative_action_statistics(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative-action statistics"):
            build_policy(tiny_wrap_delta_config(), ObservationNormalizer.identity())


class WrapDeltaPolicyTest(unittest.TestCase):
    def test_build_and_condition_dim_match_absolute_wrap(self) -> None:
        policy = build_policy(tiny_wrap_delta_config(), delta_normalizer())
        self.assertIsInstance(policy, WrapDeltaBeaverDPPolicy)
        self.assertEqual(policy.native_policy.config.robot_state_feature.shape, (75,))

    def test_target_is_each_action_minus_current_joint(self) -> None:
        policy = build_policy(tiny_wrap_delta_config(), delta_normalizer())
        batch = training_batch(1)
        expected = batch["action"] - batch["state"][:, -1:].expand_as(
            batch["action"]
        )
        torch.testing.assert_close(policy._delta_target(batch), expected)

    def test_loss_backward_and_delta_metric(self) -> None:
        policy = build_policy(tiny_wrap_delta_config(), delta_normalizer())
        loss, metrics = policy.compute_loss(training_batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["delta_action_std"], 0.0)
        loss.backward()
        grad = sum(
            parameter.grad.abs().sum()
            for parameter in policy.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(float(grad), 0.0)

    def test_chunk_reuses_replan_anchor(self) -> None:
        policy = build_policy(tiny_wrap_delta_config(), delta_normalizer())
        policy.eval()
        anchor = torch.full((1, 7), 0.25)
        first = policy.select_action(online_observation(anchor))
        self.assertTrue(policy.last_replanned)
        torch.testing.assert_close(policy._action_anchor, anchor)
        queued = list(policy.native_policy._queues["action"])
        self.assertEqual(len(queued), 1)

        drifted = torch.full((1, 7), 0.75)
        second = policy.select_action(online_observation(drifted))
        self.assertFalse(policy.last_replanned)
        # Joints 0, 2, and 6 are never modified by the wrap/lift gate.
        free = torch.tensor((0, 2, 6))
        expected = anchor + queued[0].clamp(-1.0, 1.0)
        torch.testing.assert_close(second[:, free], expected[:, free])
        self.assertTrue(torch.isfinite(first).all())

    def test_checkpoint_roundtrip_restores_delta_buffers(self) -> None:
        config = tiny_wrap_delta_config()
        policy = build_policy(config, delta_normalizer())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrap_delta.pt"
            torch.save(
                {
                    "kind": WRAP_DELTA_BEAVER_VARIANT,
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": {},
                },
                path,
            )
            restored = load_policy(path)
        self.assertIsInstance(restored, WrapDeltaBeaverDPPolicy)
        torch.testing.assert_close(
            restored.normalizer.delta_action_scale,
            policy.normalizer.delta_action_scale,
        )
        self.assertIsNone(restored._action_anchor)


if __name__ == "__main__":
    unittest.main()
