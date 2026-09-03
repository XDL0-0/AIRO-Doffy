from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import ModelConfig, RealmanBeaverConfig
from policies.realman_beaver.dataset import ObservationNormalizer, RealmanPolicyDataset
from policies.realman_beaver.modeling import AdaptiveBeaverDPPolicy, build_policy
from policies.realman_beaver.modules import AdaptiveBeaverEncoder


def adaptive_statistics() -> dict[str, torch.Tensor]:
    return {
        "p5": torch.tensor((10.0, 20.0, 30.0, 40.0)),
        "p95": torch.tensor((510.0, 520.0, 530.0, 540.0)),
        "median": torch.tensor((210.0, 220.0, 230.0, 240.0)),
        "sensor_indices": torch.tensor((1, 2, 5, 6)),
    }


def tiny_adaptive_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant="WRM_adaptive",
            beaver_adaptive_history_steps=4,
            beaver_adaptive_motion_delta_steps=3,
            beaver_adaptive_lag_steps=(1, 2, 3),
            beaver_adaptive_proximity_scales_mm=(50.0, 100.0),
            beaver_adaptive_feature_dim=32,
            beaver_adaptive_sensor_hidden_dim=32,
            beaver_adaptive_token_dim=16,
            beaver_adaptive_transformer_layers=1,
            beaver_adaptive_attention_heads=4,
            beaver_adaptive_grasp_hidden_dim=16,
            beaver_adaptive_noise_std_mm=0.0,
            beaver_adaptive_pixel_dropout=0.0,
            beaver_adaptive_sensor_dropout=0.0,
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


class _FakeAdaptiveLeRobotDataset:
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
        state_steps = len(delta_timestamps["observation.state"])
        beaver_steps = len(delta_timestamps["observation.beaver.distance_mm"])
        obs_steps = len(delta_timestamps["observation.images.camera_0"])
        action_steps = len(delta_timestamps["action"])
        state = torch.arange(state_steps, dtype=torch.float32)[:, None].repeat(1, 7)
        distance = torch.arange(beaver_steps, dtype=torch.float32)[
            :, None, None, None
        ].repeat(1, 9, 4, 4)
        self.item = {
            "observation.images.camera_0": torch.rand(obs_steps, 3, 64, 64),
            "observation.state": state,
            "action": torch.rand(action_steps, 7),
            "action_is_pad": torch.zeros(action_steps, dtype=torch.bool),
            "observation.beaver.distance_mm": distance + 100.0,
            "observation.beaver.present": torch.ones(beaver_steps, 9),
            "observation.beaver.target_status": torch.full(
                (beaver_steps, 9, 4, 4), 5.0
            ),
            "tightness": torch.tensor((0.0, 1.0)),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        return self.item


def make_encoder() -> AdaptiveBeaverEncoder:
    encoder = AdaptiveBeaverEncoder(
        n_sensors=9,
        sensor_indices=(1, 2, 5, 6),
        history_steps=4,
        lag_steps=(1, 2, 3),
        proximity_scales_mm=(50.0, 100.0),
        sensor_hidden_dim=32,
        token_dim=16,
        transformer_layers=1,
        attention_heads=4,
        output_dim=32,
        noise_std_mm=0.0,
        pixel_dropout=0.0,
        sensor_dropout=0.0,
    )
    statistics = adaptive_statistics()
    encoder.set_normalization_statistics(
        p5=statistics["p5"],
        p95=statistics["p95"],
        median=statistics["median"],
    )
    return encoder


class AdaptiveDatasetTest(unittest.TestCase):
    def test_dataset_aligns_motion_delta_and_nested_beaver_history(self) -> None:
        config = tiny_adaptive_config()
        with patch(
            "policies.realman_beaver.dataset.LeRobotDataset",
            _FakeAdaptiveLeRobotDataset,
        ):
            dataset = RealmanPolicyDataset(config, [0], stage="policy")
        sample = dataset[0]
        self.assertEqual(sample["state"].shape, (2, 7))
        self.assertTrue(torch.all(sample["delta_q"] == 3.0))
        self.assertEqual(sample["beaver_history_distance"].shape, (2, 4, 9, 4, 4))
        self.assertEqual(sample["grasp_state"].shape, (2,))


class AdaptiveEncoderTest(unittest.TestCase):
    def test_uses_key4_geometry_and_ignores_peripheral_noise(self) -> None:
        encoder = make_encoder().eval()
        distance = torch.full((2, 4, 9, 4, 4), 200.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(2, 4, 9)
        feature, middle = encoder(
            distance, status, present, return_intermediates=True
        )
        self.assertEqual(feature.shape, (2, 32))
        self.assertEqual(
            middle["cell_features"].shape,
            (2, 4, 4, 4, encoder.cell_feature_dim),
        )
        self.assertEqual(middle["sensor_attention"].shape, (2, 4))
        torch.testing.assert_close(
            middle["sensor_attention"].sum(dim=-1), torch.ones(2)
        )

        changed = distance.clone()
        changed[:, :, 0] = 2500.0  # unselected sensor 00
        changed_feature = encoder(changed, status, present)
        torch.testing.assert_close(feature, changed_feature)

        changed[:, :, 1] = 50.0  # reliable sensor 01
        selected_feature = encoder(changed, status, present)
        self.assertFalse(torch.equal(feature, selected_feature))

    def test_invalid_pixels_are_masked_and_visible(self) -> None:
        encoder = make_encoder().eval()
        distance = torch.full((1, 4, 9, 4, 4), 200.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 4, 9)
        status[:, :, 1, 0, 0] = 255.0
        _, middle = encoder(distance, status, present, return_intermediates=True)
        self.assertFalse(middle["current_valid"][0, 0, 0, 0])
        self.assertTrue(
            torch.all(middle["proximity"][0, 0, 0, 0] == 0.0)
        )

    def test_invalid_sensors_are_quality_gated_and_all_missing_is_zero(self) -> None:
        encoder = make_encoder().eval()
        torch.nn.init.zeros_(encoder.pool_score.weight)
        torch.nn.init.zeros_(encoder.pool_score.bias)
        distance = torch.full((1, 4, 9, 4, 4), 200.0)
        present = torch.ones(1, 4, 9)
        status = torch.full_like(distance, 255.0)

        missing_feature, missing = encoder(
            distance, status, present, return_intermediates=True
        )
        torch.testing.assert_close(missing_feature, torch.zeros_like(missing_feature))
        torch.testing.assert_close(
            missing["sensor_attention"],
            torch.zeros_like(missing["sensor_attention"]),
        )

        # Sensor 01 has one valid pixel; sensor 02 has a coherent 4x4 field.
        status[:, :, 1, 0, 0] = 5.0
        status[:, :, 2] = 5.0
        _, partial = encoder(distance, status, present, return_intermediates=True)
        torch.testing.assert_close(
            partial["sensor_quality"], torch.tensor(((1.0 / 16.0, 1.0, 0.0, 0.0),))
        )
        self.assertGreater(
            partial["sensor_attention"][0, 1],
            partial["sensor_attention"][0, 0],
        )


class AdaptivePolicyTest(unittest.TestCase):
    def _policy(self) -> AdaptiveBeaverDPPolicy:
        return build_policy(
            tiny_adaptive_config(),
            ObservationNormalizer.identity(
                temporal_beaver_statistics=adaptive_statistics()
            ),
        )

    @staticmethod
    def _batch() -> dict[str, torch.Tensor]:
        return {
            "image": torch.rand(2, 2, 3, 64, 64),
            "state": torch.randn(2, 2, 7),
            "delta_q": torch.randn(2, 2, 7),
            "action": torch.randn(2, 8, 7),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "beaver_history_distance": torch.full((2, 2, 4, 9, 4, 4), 200.0),
            "beaver_history_status": torch.full((2, 2, 4, 9, 4, 4), 5.0),
            "beaver_history_present": torch.ones(2, 2, 4, 9),
            "grasp_state": torch.tensor(((0.0, 0.0), (0.0, 1.0))),
        }

    def test_motion_and_grasp_directly_condition_diffusion_policy(self) -> None:
        policy = self._policy()
        self.assertIsInstance(policy, AdaptiveBeaverDPPolicy)
        self.assertEqual(policy.native_policy.config.robot_state_feature.shape, (47,))
        prepared, grasp_logit, z_beaver, attention = policy._prepare(
            self._batch(), include_action=True, return_auxiliary=True
        )
        self.assertEqual(prepared["observation.state"].shape, (2, 2, 47))
        self.assertEqual(grasp_logit.shape, (2, 2))
        self.assertEqual(z_beaver.shape, (2, 2, 32))
        self.assertEqual(attention.shape, (2, 2, 4))
        loss, metrics = policy.compute_loss(self._batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("sensor_attention_entropy", metrics)
        loss.backward()
        self.assertGreater(
            policy.beaver_encoder.sensor_mlp[0].weight.grad.abs().sum(), 0
        )
        self.assertGreater(policy.grasp_state_head[0].weight.grad.abs().sum(), 0)

    def test_checkpoint_round_trip(self) -> None:
        config = tiny_adaptive_config()
        policy = self._policy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "WRM_adaptive.pt"
            torch.save(
                {
                    "kind": "WRM_adaptive",
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": {},
                    "epoch": 0,
                    "global_step": 1,
                    "metrics": {},
                },
                path,
            )
            restored = load_policy(path)
        self.assertIsInstance(restored, AdaptiveBeaverDPPolicy)
        torch.testing.assert_close(
            restored.beaver_encoder.distance_p5,
            policy.beaver_encoder.distance_p5,
        )


if __name__ == "__main__":
    unittest.main()
