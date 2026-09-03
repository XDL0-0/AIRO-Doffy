from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import ModelConfig, RealmanBeaverConfig
from policies.realman_beaver.dataset import ObservationNormalizer, RealmanPolicyDataset
from policies.realman_beaver.eval_registry import (
    BEAVER_POLICY_VARIANTS,
    EXPECTED_CHECKPOINT_KINDS,
    SUPPORTED_POLICY_VARIANTS,
)
from policies.realman_beaver.modeling import build_policy
from policies.realman_beaver.modeling_dp_beaver_closure import DPBeaverClosurePolicy
from policies.realman_beaver.modules import ClosureBeaverEncoder


def tiny_closure_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant="dp_beaver_closure",
            closure_joint_mask=(0, 0, 0, 1, 1, 1, 0),
            closure_beaver_encoder_dim=16,
            closure_sensor_hidden_dim=16,
            closure_hidden_dim=32,
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


def closure_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    distance = torch.full((batch_size, 2, 9, 4, 4), 600.0)
    distance[:, 1] -= 50.0
    return {
        "image": torch.rand(batch_size, 2, 3, 64, 64),
        "state": torch.randn(batch_size, 2, 7),
        "action": torch.randn(batch_size, 8, 7),
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
        "beaver_distance": distance,
        "beaver_status": torch.full_like(distance, 5.0),
        "beaver_present": torch.ones(batch_size, 2, 9),
        "grasp_state": torch.tensor(((0.0, 0.0), (0.0, 1.0)))[:batch_size],
    }


class _FakeClosureLeRobotDataset:
    def __init__(
        self, *, delta_timestamps: dict[str, list[float]], **_: object
    ) -> None:
        self.delta_timestamps = delta_timestamps
        self.features = {
            "observation.images.camera_0": {"shape": (3, 64, 64)},
            "observation.state": {"shape": (7,)},
            "action": {"shape": (7,)},
            "observation.beaver.distance_mm": {"shape": (9, 4, 4)},
            "observation.beaver.present": {"shape": (9,)},
            "observation.beaver.target_status": {"shape": (9, 4, 4)},
            "tightness": {"shape": (1,)},
        }
        obs_steps = len(delta_timestamps["observation.state"])
        action_steps = len(delta_timestamps["action"])
        self.item = {
            "observation.images.camera_0": torch.rand(obs_steps, 3, 64, 64),
            "observation.state": torch.rand(obs_steps, 7),
            "action": torch.rand(action_steps, 7),
            "action_is_pad": torch.zeros(action_steps, dtype=torch.bool),
            "observation.beaver.distance_mm": torch.full((obs_steps, 9, 4, 4), 500.0),
            "observation.beaver.present": torch.ones(obs_steps, 9),
            "observation.beaver.target_status": torch.full((obs_steps, 9, 4, 4), 5.0),
            "tightness": torch.tensor((0.0, 1.0)),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        return self.item


class ClosureBeaverEncoderTest(unittest.TestCase):
    def test_masked_mean_and_missing_sensor_support(self) -> None:
        encoder = ClosureBeaverEncoder(
            n_sensors=9,
            sensor_shape=(4, 4),
            distance_max_mm=2550.0,
            valid_statuses=(5, 9),
            hidden_dim=16,
            output_dim=12,
        )
        batch = closure_batch(1)
        output, middle = encoder(
            batch["beaver_distance"][:, 1],
            batch["beaver_distance"][:, 0],
            batch["beaver_status"][:, 1],
            batch["beaver_status"][:, 0],
            batch["beaver_present"][:, 1],
            batch["beaver_present"][:, 0],
            return_intermediates=True,
        )
        self.assertEqual(output.shape, (1, 12))
        self.assertEqual(middle["delta_flat"].shape, (1, 144))
        self.assertTrue(middle["sensor_available"].all())

        missing = closure_batch(1)
        missing["beaver_present"].zero_()
        missing_output, missing_middle = encoder(
            missing["beaver_distance"][:, 1],
            missing["beaver_distance"][:, 0],
            missing["beaver_status"][:, 1],
            missing["beaver_status"][:, 0],
            missing["beaver_present"][:, 1],
            missing["beaver_present"][:, 0],
            return_intermediates=True,
        )
        torch.testing.assert_close(missing_output, torch.zeros_like(missing_output))
        torch.testing.assert_close(
            missing_middle["delta_flat"],
            torch.zeros_like(missing_middle["delta_flat"]),
        )
        self.assertEqual(missing_middle["any_available"].item(), 0.0)


class ClosurePolicyTest(unittest.TestCase):
    def _policy(self) -> DPBeaverClosurePolicy:
        policy = build_policy(tiny_closure_config(), ObservationNormalizer.identity())
        self.assertIsInstance(policy, DPBeaverClosurePolicy)
        return policy

    def test_dataset_uses_only_two_current_frames(self) -> None:
        config = tiny_closure_config()
        with patch(
            "policies.realman_beaver.dataset.LeRobotDataset",
            _FakeClosureLeRobotDataset,
        ):
            dataset = RealmanPolicyDataset(config, [0], stage="policy")
        sample = dataset[0]
        self.assertEqual(sample["state"].shape, (2, 7))
        self.assertEqual(sample["beaver_distance"].shape, (2, 9, 4, 4))
        self.assertEqual(sample["grasp_state"].shape, (2,))
        self.assertNotIn("beaver_history_distance", sample)

    def test_global_dp_is_unmodified_and_loss_logs_required_metrics(self) -> None:
        policy = self._policy()
        self.assertEqual(policy.native_policy.config.robot_state_feature.shape, (7,))
        loss, metrics = policy.compute_loss(closure_batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(metrics),
            {
                "loss",
                "diffusion_loss",
                "grasp_bce",
                "residual_reg",
                "gate_mean",
                "gate_std",
                "closure_residual_magnitude",
                "predicted_grasp_probability",
            },
        )
        loss.backward()
        self.assertGreater(
            policy.closure_residual_head.weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(
            policy.grasp_state_head[0].weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(
            policy.beaver_encoder.sensor_mlp[0].weight.grad.abs().sum().item(), 0.0
        )

    def test_joint_mask_and_all_missing_fallback(self) -> None:
        policy = self._policy()
        with torch.no_grad():
            policy.closure_residual_head.bias.fill_(0.5)
            policy.gate_head.bias.fill_(4.0)
        correction, gate, _, _ = policy._closure_outputs(closure_batch(1))
        torch.testing.assert_close(
            correction[..., :3], torch.zeros_like(correction[..., :3])
        )
        torch.testing.assert_close(
            correction[..., 6], torch.zeros_like(correction[..., 6])
        )
        self.assertGreater(correction[..., 3:6].abs().sum().item(), 0.0)
        self.assertGreater(gate.item(), 0.9)

        missing = closure_batch(1)
        missing["beaver_present"].zero_()
        correction, gate, _, _ = policy._closure_outputs(missing)
        torch.testing.assert_close(correction, torch.zeros_like(correction))
        torch.testing.assert_close(gate, torch.zeros_like(gate))

    def test_checkpoint_round_trip(self) -> None:
        config = tiny_closure_config()
        policy = self._policy()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "dp_beaver_closure.pt"
            torch.save(
                {
                    "kind": "dp_beaver_closure",
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": {},
                    "epoch": 0,
                    "global_step": 1,
                    "metrics": {},
                },
                checkpoint_path,
            )
            restored = load_policy(checkpoint_path)
        self.assertIsInstance(restored, DPBeaverClosurePolicy)
        torch.testing.assert_close(
            restored.closure_joint_mask, policy.closure_joint_mask
        )

    def test_evaluation_registry_marks_policy_as_beaver_aware(self) -> None:
        self.assertIn("dp_beaver_closure", SUPPORTED_POLICY_VARIANTS)
        self.assertIn("dp_beaver_closure", BEAVER_POLICY_VARIANTS)
        self.assertEqual(
            EXPECTED_CHECKPOINT_KINDS["dp_beaver_closure"],
            "dp_beaver_closure",
        )


if __name__ == "__main__":
    unittest.main()
