from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import ModelConfig, RealmanBeaverConfig
from policies.realman_beaver.dataset import (
    ObservationNormalizer,
    RealmanPolicyDataset,
    build_delta_pairs,
    resolve_beaver_sensor_indices,
)
from policies.realman_beaver.modeling import DeltaBeaverDPPolicy, build_policy
from policies.realman_beaver.modules import DeltaBeaverEncoder


def tiny_delta_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant="WRM_delta",
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


def delta_statistics() -> dict[str, torch.Tensor]:
    return {
        "mean": torch.tensor((500.0, 600.0, 700.0, 800.0)),
        "std": torch.tensor((100.0, 200.0, 300.0, 400.0)),
        "sensor_indices": torch.tensor((1, 2, 5, 6)),
    }


def make_encoder() -> DeltaBeaverEncoder:
    encoder = DeltaBeaverEncoder(n_sensors=9, sensor_indices=(1, 2, 5, 6))
    statistics = delta_statistics()
    encoder.set_normalization_statistics(mean=statistics["mean"], std=statistics["std"])
    return encoder


def sensor_batch(*leading: int) -> tuple[torch.Tensor, ...]:
    distance = torch.full((*leading, 9, 4, 4), 900.0)
    previous = torch.full_like(distance, 800.0)
    status = torch.full_like(distance, 5.0)
    previous_status = torch.full_like(distance, 5.0)
    present = torch.ones(*leading, 9)
    previous_present = torch.ones_like(present)
    return distance, previous, status, previous_status, present, previous_present


class _FakeDeltaLeRobotDataset:
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
        combined = len(delta_timestamps["observation.state"])
        obs_steps = len(delta_timestamps["observation.images.camera_0"])
        action_steps = len(delta_timestamps["action"])
        state = torch.arange(combined, dtype=torch.float32)[:, None].repeat(1, 7)
        distance = torch.arange(combined, dtype=torch.float32)[:, None, None, None]
        distance = distance.repeat(1, 9, 4, 4) + 100.0
        self.item = {
            "observation.images.camera_0": torch.rand(obs_steps, 3, 64, 64),
            "observation.state": state,
            "action": torch.rand(action_steps, 7),
            "action_is_pad": torch.zeros(action_steps, dtype=torch.bool),
            "observation.beaver.distance_mm": distance,
            "observation.beaver.present": torch.ones(combined, 9),
            "observation.beaver.target_status": torch.full((combined, 9, 4, 4), 5.0),
            "tightness": torch.tensor((0.0, 1.0)),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        return self.item


class DeltaConstructionTest(unittest.TestCase):
    def test_delta_construction_and_episode_boundary(self) -> None:
        combined = torch.arange(8)
        current, previous = build_delta_pairs(combined, n_obs_steps=2, delta_steps=6)
        torch.testing.assert_close(current, torch.tensor((6, 7)))
        torch.testing.assert_close(previous, torch.tensor((0, 1)))
        torch.testing.assert_close(current - previous, torch.tensor((6, 6)))

        # LeRobot clamps pre-episode queries to the earliest frame in the same
        # episode, so no preceding-episode value can enter t-k.
        clamped = torch.tensor((10, 10, 10, 10, 10, 10, 10, 11))
        current, previous = build_delta_pairs(clamped, n_obs_steps=2, delta_steps=6)
        torch.testing.assert_close(current - previous, torch.tensor((0, 1)))
        self.assertFalse(torch.any(previous == 9))

    def test_dataset_builds_exact_six_frame_deltas(self) -> None:
        config = tiny_delta_config()
        with patch(
            "policies.realman_beaver.dataset.LeRobotDataset",
            _FakeDeltaLeRobotDataset,
        ):
            dataset = RealmanPolicyDataset(config, [0], stage="policy")
        sample = dataset[0]
        self.assertEqual(sample["state"].shape, (2, 7))
        self.assertEqual(sample["delta_q"].shape, (2, 7))
        self.assertTrue(torch.all(sample["delta_q"] == 6))
        self.assertEqual(sample["beaver_distance"].shape, (2, 9, 4, 4))
        self.assertEqual(sample["beaver_previous_distance"].shape, (2, 9, 4, 4))
        self.assertTrue(
            torch.all(
                sample["beaver_distance"] - sample["beaver_previous_distance"] == 6
            )
        )
        self.assertEqual(sample["grasp_state"].shape, (2,))
        expected = [step / 24 for step in range(-7, 1)]
        self.assertEqual(
            dataset.dataset.delta_timestamps[config.dataset.state_key], expected
        )

    def test_requested_sensor_names_resolve_to_recorded_slots(self) -> None:
        self.assertEqual(
            resolve_beaver_sensor_indices(
                tiny_delta_config().dataset, ("01", "02", "10", "11")
            ),
            (1, 2, 5, 6),
        )


class DeltaBeaverEncoderTest(unittest.TestCase):
    def test_zero_and_valid_mask_handling(self) -> None:
        encoder = make_encoder()
        values = list(sensor_batch(1))
        distance = values[0]
        status = values[2]
        previous_status = values[3]
        distance[0, 1, 0, 0] = 0.0
        distance[0, 2, 0, 1] = 1234.0
        status[0, 2, 0, 1] = 255.0
        distance[0, 5, 0, 2] = 950.0
        previous_status[0, 5, 0, 2] = 255.0
        features, middle = encoder.preprocess(*values)

        # Selected slot 01: zero is neutral continuous input, invalid, and
        # explicitly represented by zero_mask (never maximum proximity).
        self.assertEqual(features[0, 0, 0, 0, 0].item(), 0.0)
        self.assertEqual(features[0, 0, 0, 0, 1].item(), 0.0)
        self.assertEqual(features[0, 0, 0, 0, 2].item(), 0.0)
        self.assertEqual(features[0, 0, 0, 0, 3].item(), 1.0)

        # Selected slot 02: status 255 preserves the existing invalid semantic.
        self.assertEqual(features[0, 1, 0, 1, 0].item(), 0.0)
        self.assertEqual(features[0, 1, 0, 1, 1].item(), 0.0)
        self.assertEqual(features[0, 1, 0, 1, 2].item(), 0.0)
        self.assertEqual(features[0, 1, 0, 1, 3].item(), 0.0)

        # Delta is used only when both t and t-k are genuine measurements.
        self.assertFalse(middle["delta_valid_mask"][0, 2, 0, 2])
        self.assertEqual(features[0, 2, 0, 2, 1].item(), 0.0)

    def test_identity_preservation_and_output_shapes(self) -> None:
        encoder = make_encoder()
        values = sensor_batch(2, 3)
        z_beaver, middle = encoder(*values, return_intermediates=True)
        self.assertEqual(middle["cell_features"].shape, (2, 3, 4, 4, 4, 4))
        self.assertEqual(middle["sensor_features"].shape, (2, 3, 4, 32))
        self.assertEqual(middle["concatenated"].shape, (2, 3, 128))
        self.assertEqual(z_beaver.shape, (2, 3, 64))

        changed = [value.clone() for value in values]
        changed[0][..., 2, :, :] += 200.0  # physical sensor 02 only
        _, changed_middle = encoder(*changed, return_intermediates=True)
        unchanged_tokens = middle["sensor_features"]
        changed_tokens = changed_middle["sensor_features"]
        self.assertTrue(
            torch.equal(unchanged_tokens[..., 0, :], changed_tokens[..., 0, :])
        )
        self.assertFalse(
            torch.equal(unchanged_tokens[..., 1, :], changed_tokens[..., 1, :])
        )
        self.assertTrue(
            torch.equal(unchanged_tokens[..., 2:, :], changed_tokens[..., 2:, :])
        )
        torch.testing.assert_close(
            changed_middle["concatenated"][..., 32:64], changed_tokens[..., 1, :]
        )


class DeltaPolicyTest(unittest.TestCase):
    def _policy(self) -> DeltaBeaverDPPolicy:
        return build_policy(
            tiny_delta_config(),
            ObservationNormalizer.identity(delta_beaver_statistics=delta_statistics()),
        )

    @staticmethod
    def _batch() -> dict[str, torch.Tensor]:
        values = sensor_batch(2, 2)
        return {
            "image": torch.rand(2, 2, 3, 64, 64),
            "state": torch.randn(2, 2, 7),
            "delta_q": torch.randn(2, 2, 7),
            "action": torch.randn(2, 8, 7),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "beaver_distance": values[0],
            "beaver_previous_distance": values[1],
            "beaver_status": values[2],
            "beaver_previous_status": values[3],
            "beaver_present": values[4],
            "beaver_previous_present": values[5],
            "grasp_state": torch.tensor(((0.0, 0.0), (0.0, 1.0))),
        }

    def test_grasp_head_and_policy_forward_train(self) -> None:
        policy = self._policy()
        self.assertIsInstance(policy, DeltaBeaverDPPolicy)
        self.assertEqual(policy.native_policy.config.robot_state_feature.shape, (71,))
        self.assertEqual(policy.grasp_state_head[0].in_features, 78)
        prepared, grasp_logit, z_beaver = policy._prepare(
            self._batch(), include_action=True, return_auxiliary=True
        )
        self.assertEqual(prepared["observation.state"].shape, (2, 2, 71))
        self.assertEqual(grasp_logit.shape, (2, 2))
        self.assertEqual(z_beaver.shape, (2, 2, 64))

        loss, metrics = policy.compute_loss(self._batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(metrics),
            {"loss", "diffusion_loss", "grasp_loss", "grasp_accuracy", "z_beaver_std"},
        )
        loss.backward()
        self.assertGreater(
            policy.beaver_encoder.sensor_mlp[0].weight.grad.abs().sum(), 0
        )
        self.assertGreater(policy.grasp_state_head[0].weight.grad.abs().sum(), 0)
        actions = policy.predict_action_chunk(self._batch())
        self.assertEqual(actions.shape, (2, 2, 7))
        self.assertTrue(torch.isfinite(actions).all())

    def test_online_history_uses_earliest_until_k_frames_exist(self) -> None:
        policy = self._policy()
        first = {
            "image": torch.rand(1, 3, 64, 64),
            "state": torch.zeros(1, 7),
            "beaver_distance": torch.full((1, 9, 4, 4), 500.0),
            "beaver_status": torch.full((1, 9, 4, 4), 5.0),
            "beaver_present": torch.ones(1, 9),
        }
        initial = policy._append_online_history(first)
        self.assertTrue(torch.equal(initial["delta_q"], torch.zeros(1, 7)))
        self.assertTrue(
            torch.equal(initial["beaver_previous_distance"], first["beaver_distance"])
        )
        for frame_index in range(1, 7):
            frame = {key: value.clone() for key, value in first.items()}
            frame["state"] = torch.full((1, 7), float(frame_index))
            frame["beaver_distance"] = torch.full((1, 9, 4, 4), 500.0 + frame_index)
            latest = policy._append_online_history(frame)
        self.assertTrue(torch.all(latest["delta_q"] == 6.0))
        self.assertTrue(torch.all(latest["beaver_previous_distance"] == 500.0))

    def test_checkpoint_save_load(self) -> None:
        config = tiny_delta_config()
        policy = self._policy()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "WRM_delta.pt"
            torch.save(
                {
                    "kind": "WRM_delta",
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
        self.assertIsInstance(restored, DeltaBeaverDPPolicy)
        torch.testing.assert_close(
            restored.normalizer.beaver_delta_mean,
            policy.normalizer.beaver_delta_mean,
        )
        torch.testing.assert_close(
            restored.beaver_encoder.distance_std,
            policy.beaver_encoder.distance_std,
        )


if __name__ == "__main__":
    unittest.main()
