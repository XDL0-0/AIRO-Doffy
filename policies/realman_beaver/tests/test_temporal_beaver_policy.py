from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import ModelConfig, RealmanBeaverConfig
from policies.realman_beaver.dataset import ObservationNormalizer
from policies.realman_beaver.modeling import TemporalBeaverDPPolicy, build_policy
from policies.realman_beaver.modules import TemporalBeaverEncoder


def tiny_temporal_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant="WRM_temporal",
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


def temporal_statistics(
    sensor_indices: tuple[int, ...] = (1, 2, 5, 6),
) -> dict[str, torch.Tensor]:
    count = len(sensor_indices)
    return {
        "p5": torch.full((count,), 100.0),
        "p95": torch.full((count,), 1100.0),
        "median": torch.full((count,), 600.0),
        "sensor_indices": torch.tensor(sensor_indices),
    }


class TemporalBeaverEncoderTest(unittest.TestCase):
    def test_zero_imputation_and_only_genuine_adjacent_deltas(self) -> None:
        encoder = TemporalBeaverEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=12
        )
        encoder.set_normalization_statistics(
            **{key: value for key, value in temporal_statistics().items() if key != "sensor_indices"}
        )
        distance = torch.full((1, 12, 9, 4, 4), 600.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 12, 9)

        # Selected physical sensor 01, pixel (0, 0).
        distance[0, 0, 1, 0, 0] = 0.0
        distance[0, 1, 1, 0, 0] = 300.0
        distance[0, 2, 1, 0, 0] = 500.0
        distance[0, 3, 1, 0, 0] = 700.0
        status[0, 3, 1, 0, 0] = 255.0
        distance[0, 4, 1, 0, 0] = 900.0

        features, middle = encoder.preprocess(distance, status, present)
        pixel = features[0, :, 0, 0, 0]
        self.assertTrue(torch.allclose(pixel[:5, 0], torch.tensor((0.5, 0.2, 0.4, 0.4, 0.8))))
        self.assertTrue(torch.allclose(pixel[:5, 1], torch.tensor((0.0, 0.0, 0.2, 0.0, 0.0))))
        self.assertTrue(torch.equal(pixel[:5, 2].bool(), torch.tensor((False, True, True, False, True))))
        self.assertTrue(torch.equal(pixel[:5, 3].bool(), torch.tensor((True, False, False, False, False))))
        self.assertEqual(middle["filled_distance"][0, 3, 0, 0, 0], 500.0)

    def test_encoder_keeps_four_ordered_sensor_tokens(self) -> None:
        encoder = TemporalBeaverEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=12
        )
        encoder.set_normalization_statistics(
            **{
                key: value
                for key, value in temporal_statistics().items()
                if key != "sensor_indices"
            }
        )
        distance = torch.rand(2, 3, 12, 9, 4, 4) * 1000.0 + 1.0
        status = torch.full_like(distance, 5.0)
        present = torch.ones(2, 3, 12, 9)
        feature, middle = encoder(
            distance, status, present, return_intermediates=True
        )
        self.assertEqual(feature.shape, (2, 3, 64))
        self.assertEqual(middle["sensor_tokens"].shape, (2, 3, 4, 64))
        self.assertEqual(middle["concatenated"].shape, (2, 3, 256))

    def test_encoder_supports_all_nine_sensors(self) -> None:
        indices = tuple(range(9))
        encoder = TemporalBeaverEncoder(
            n_sensors=9, sensor_indices=indices, history_steps=12
        )
        encoder.set_normalization_statistics(
            **{
                key: value
                for key, value in temporal_statistics(indices).items()
                if key != "sensor_indices"
            }
        )
        distance = torch.rand(2, 12, 9, 4, 4) * 1000.0 + 1.0
        status = torch.full_like(distance, 5.0)
        present = torch.ones(2, 12, 9)
        feature, middle = encoder(
            distance, status, present, return_intermediates=True
        )
        self.assertEqual(feature.shape, (2, 64))
        self.assertEqual(middle["sensor_tokens"].shape, (2, 9, 64))
        self.assertEqual(middle["concatenated"].shape, (2, 576))

    def test_encoder_refuses_unfitted_global_distance_fallback(self) -> None:
        encoder = TemporalBeaverEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=12
        )
        distance = torch.full((1, 12, 9, 4, 4), 600.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 12, 9)
        with self.assertRaisesRegex(RuntimeError, "global 2550 mm"):
            encoder(distance, status, present)

    def test_changing_only_sensor_02_changes_the_representation(self) -> None:
        encoder = TemporalBeaverEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=12
        )
        encoder.set_normalization_statistics(
            **{
                key: value
                for key, value in temporal_statistics().items()
                if key != "sensor_indices"
            }
        )
        distance = torch.full((1, 12, 9, 4, 4), 600.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 12, 9)
        baseline = encoder(distance, status, present)
        changed = distance.clone()
        changed[:, 6:, 2] = 900.0
        sensor_02_changed = encoder(changed, status, present)
        self.assertFalse(torch.allclose(baseline, sensor_02_changed))


class TemporalBeaverPolicyTest(unittest.TestCase):
    @staticmethod
    def _batch() -> dict[str, torch.Tensor]:
        return {
            "image": torch.rand(2, 2, 3, 64, 64),
            "state": torch.randn(2, 2, 7),
            "action": torch.randn(2, 8, 7),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "beaver_history_distance": torch.rand(2, 2, 12, 9, 4, 4)
            * 1000.0
            + 100.0,
            "beaver_history_status": torch.full(
                (2, 2, 12, 9, 4, 4), 5.0
            ),
            "beaver_history_present": torch.ones(2, 2, 12, 9),
            "grasp_state": torch.tensor(((0.0, 0.0), (0.0, 1.0))),
        }

    def test_policy_requires_train_split_temporal_statistics(self) -> None:
        with self.assertRaisesRegex(ValueError, "per-sensor robust normalization"):
            build_policy(tiny_temporal_config(), ObservationNormalizer.identity())

    def test_checkpoint_loader_restores_variable_length_statistics(self) -> None:
        config = tiny_temporal_config()
        policy = build_policy(
            config,
            ObservationNormalizer.identity(
                temporal_beaver_statistics=temporal_statistics()
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "temporal.pt"
            torch.save(
                {
                    "kind": "WRM_temporal",
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": {},
                },
                checkpoint_path,
            )
            restored = load_policy(checkpoint_path)

        self.assertTrue(restored.beaver_encoder.normalization_fitted)
        torch.testing.assert_close(
            restored.normalizer.beaver_temporal_p5,
            policy.normalizer.beaver_temporal_p5,
        )
        torch.testing.assert_close(
            restored.beaver_encoder.distance_median,
            policy.beaver_encoder.distance_median,
        )

    def test_native_dp_condition_loss_and_statistics(self) -> None:
        normalizer = ObservationNormalizer.identity(
            temporal_beaver_statistics=temporal_statistics()
        )
        policy = build_policy(tiny_temporal_config(), normalizer)
        self.assertIsInstance(policy, TemporalBeaverDPPolicy)
        self.assertTrue(policy.beaver_encoder.normalization_fitted)
        self.assertEqual(
            policy.native_policy.config.robot_state_feature.shape, (72,)
        )

        batch = {
            "image": torch.rand(2, 2, 3, 64, 64),
            "state": torch.randn(2, 2, 7),
            "action": torch.randn(2, 8, 7),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "beaver_history_distance": torch.rand(2, 2, 12, 9, 4, 4)
            * 1000.0
            + 100.0,
            "beaver_history_status": torch.full(
                (2, 2, 12, 9, 4, 4), 5.0
            ),
            "beaver_history_present": torch.ones(2, 2, 12, 9),
            "grasp_state": torch.tensor(((0.0, 0.0), (0.0, 1.0))),
        }
        self.assertEqual(policy._state(batch).shape, (2, 2, 72))
        loss, metrics = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(metrics),
            {
                "loss",
                "diffusion_loss",
                "grasp_loss",
                "grasp_accuracy",
                "grasp_positive_rate",
                "predicted_grasp_positive_rate",
                "beaver_feature_std",
                "beaver_sensor_01_token_std",
                "beaver_sensor_02_token_std",
                "beaver_sensor_10_token_std",
                "beaver_sensor_11_token_std",
            },
        )
        loss.backward()
        self.assertGreater(policy.beaver_encoder.sensor_embedding.grad.abs().sum(), 0)
        self.assertGreater(policy.grasp_state_head[0].weight.grad.abs().sum(), 0)
        actions = policy.predict_action_chunk(batch)
        self.assertEqual(actions.shape, (2, 2, 7))
        self.assertTrue(torch.isfinite(actions).all())

        restored = build_policy(
            tiny_temporal_config(),
            ObservationNormalizer.identity(
                temporal_beaver_statistics=temporal_statistics()
            ),
        )
        restored.load_state_dict(policy.state_dict())
        self.assertTrue(restored.beaver_encoder.normalization_fitted)
        self.assertTrue(
            torch.equal(
                restored.beaver_encoder.distance_p5,
                policy.beaver_encoder.distance_p5,
            )
        )
        self.assertTrue(
            torch.equal(
                restored.beaver_encoder.sensor_embedding,
                policy.beaver_encoder.sensor_embedding,
            )
        )

    def test_capacity_matched_joint_only_zeros_both_conditions(self) -> None:
        config = tiny_temporal_config()
        config.model.use_visual_condition = False
        config.model.use_beaver_condition = False
        config.model.condition_on_grasp_probability = False
        config.model.beaver_grasp_loss_weight = 0.0
        config.validate()
        policy = build_policy(
            config,
            ObservationNormalizer.identity(
                temporal_beaver_statistics=temporal_statistics()
            ),
        )
        prepared = policy._prepare(self._batch(), include_action=False)
        self.assertEqual(torch.count_nonzero(prepared[config.dataset.image_key]), 0)
        self.assertEqual(torch.count_nonzero(prepared["observation.state"][..., 7:]), 0)

    def test_current_mode_ignores_earlier_history_frames(self) -> None:
        config = tiny_temporal_config()
        config.model.beaver_history_mode = "current"
        policy = build_policy(
            config,
            ObservationNormalizer.identity(
                temporal_beaver_statistics=temporal_statistics()
            ),
        )
        first = self._batch()
        second = {key: value.clone() for key, value in first.items()}
        second["beaver_history_distance"][..., :-1, :, :, :] += 500.0
        torch.testing.assert_close(policy._state(first), policy._state(second))

    def test_online_history_pads_with_earliest_frame(self) -> None:
        policy = build_policy(
            tiny_temporal_config(),
            ObservationNormalizer.identity(
                temporal_beaver_statistics=temporal_statistics()
            ),
        )
        batch = {
            "image": torch.rand(1, 3, 64, 64),
            "state": torch.randn(1, 7),
            "beaver_distance": torch.full((1, 9, 4, 4), 250.0),
            "beaver_status": torch.full((1, 9, 4, 4), 5.0),
            "beaver_present": torch.ones(1, 9),
        }
        first = policy._append_online_history(batch)
        self.assertEqual(first["beaver_history_distance"].shape, (1, 12, 9, 4, 4))
        self.assertTrue(
            torch.equal(
                first["beaver_history_distance"][:, 0],
                first["beaver_history_distance"][:, -1],
            )
        )
        policy._append_online_history(batch)
        self.assertEqual(len(policy._beaver_history), 1)
        second_batch = dict(batch)
        second_batch["beaver_distance"] = torch.full((1, 9, 4, 4), 750.0)
        second = policy._append_online_history(second_batch)
        self.assertTrue(
            torch.equal(
                second["beaver_history_distance"][:, 0],
                batch["beaver_distance"],
            )
        )
        self.assertTrue(
            torch.equal(
                second["beaver_history_distance"][:, -1],
                second_batch["beaver_distance"],
            )
        )


if __name__ == "__main__":
    unittest.main()
