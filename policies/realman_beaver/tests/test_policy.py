from __future__ import annotations

import unittest

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from policies.realman_beaver.configuration import (
    ModelConfig,
    RDPConfig,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import LatentNormalizer, ObservationNormalizer
from policies.realman_beaver.modeling import LeRobotDPPolicy, RDPPolicy, build_policy


def tiny_config(variant: str) -> RealmanBeaverConfig:
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
        ),
        rdp=RDPConfig(
            action_horizon=8,
            downsample_ratio=2,
            latent_dim=4,
            tokenizer_hidden_dim=16,
            beaver_feature_dim=16,
            slow_observation_stride=2,
            slow_replan_steps=4,
            latent_down_dims=(32, 64),
            latent_kernel_size=3,
            latent_num_inference_steps=2,
        ),
    )
    config.dataset.image_shape = (3, 64, 64)
    config.validate()
    return config


def beaver_fields(
    batch_size: int, horizon: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Realistic Beaver inputs: distance + presence + VL53L7CX status codes.

    Status uses the dominant real-dataset mix (5=valid, 9=weak, 255=no
    target) so the pixel-level mask is actually exercised.
    """
    generator = torch.Generator().manual_seed(seed)
    codes = torch.tensor([5.0, 5.0, 5.0, 5.0, 9.0, 255.0, 255.0])
    status = codes[torch.randint(len(codes), (batch_size, horizon, 9, 4, 4), generator=generator)]
    distance = torch.rand(batch_size, horizon, 9, 4, 4, generator=generator) * 2550.0
    present = torch.ones(batch_size, horizon, 9)
    return {"beaver_distance": distance, "beaver_present": present, "beaver_status": status}


def dp_batch(include_beaver: bool) -> dict[str, torch.Tensor]:
    batch = {
        "image": torch.rand(2, 2, 3, 64, 64),
        "state": torch.randn(2, 2, 7),
        "action": torch.randn(2, 8, 7),
        "action_is_pad": torch.tensor(
            [[False] * 8, [False] * 6 + [True] * 2], dtype=torch.bool
        ),
    }
    if include_beaver:
        batch.update(beaver_fields(2, 2, seed=1))
    return batch


def rdp_batch() -> dict[str, torch.Tensor]:
    batch = dp_batch(False)
    batch.update(beaver_fields(2, 8, seed=2))
    return batch


class PolicyTest(unittest.TestCase):
    def test_original_and_beaver_use_native_lerobot_dp(self) -> None:
        for variant in ("original_dp", "dp_beaver"):
            with self.subTest(variant=variant):
                config = tiny_config(variant)
                policy = build_policy(config, ObservationNormalizer.identity())
                self.assertIsInstance(policy, LeRobotDPPolicy)
                self.assertIsInstance(policy.native_policy, DiffusionPolicy)
                expected_state_dim = 7 if variant == "original_dp" else 160
                self.assertEqual(
                    policy.native_policy.config.input_features[
                        "observation.state"
                    ].shape,
                    (expected_state_dim,),
                )

                batch = dp_batch(include_beaver=variant == "dp_beaver")
                loss, metrics = policy.compute_loss(batch)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(metrics["loss"], 0.0)
                loss.backward()
                actions = policy.predict_action_chunk(batch)
                self.assertEqual(actions.shape, (2, 2, 7))
                self.assertTrue(torch.isfinite(actions).all())

    def test_rdp_has_asymmetric_tokenizer_and_native_slow_dp(self) -> None:
        config = tiny_config("rdp_like")
        policy = build_policy(
            config,
            ObservationNormalizer.identity(),
            LatentNormalizer.identity(config.rdp.latent_dim),
        )
        self.assertIsInstance(policy, RDPPolicy)
        self.assertIsInstance(policy.slow_policy, DiffusionPolicy)

        batch = rdp_batch()
        tokenizer_loss, tokenizer_metrics = policy.tokenizer_loss(batch)
        self.assertTrue(torch.isfinite(tokenizer_loss))
        self.assertIn("reconstruction_loss", tokenizer_metrics)
        tokenizer_loss.backward()

        policy.zero_grad(set_to_none=True)
        policy.freeze_tokenizer()
        latent_loss, _ = policy.latent_loss(batch)
        self.assertTrue(torch.isfinite(latent_loss))
        latent_loss.backward()

        actions = policy.predict_action_chunk(batch)
        self.assertEqual(actions.shape, (2, 8, 7))
        self.assertTrue(torch.isfinite(actions).all())

    def test_online_deployment_shapes(self) -> None:
        for variant in ("original_dp", "rdp_like"):
            with self.subTest(variant=variant):
                config = tiny_config(variant)
                policy = build_policy(
                    config,
                    ObservationNormalizer.identity(),
                    LatentNormalizer.identity(config.rdp.latent_dim),
                ).eval()
                observation = {
                    "image": torch.rand(3, 64, 64),
                    "state": torch.randn(7),
                }
                if variant == "rdp_like":
                    observation.update(
                        {
                            key: value[0, 0]
                            for key, value in beaver_fields(1, 1).items()
                        }
                    )
                policy.reset()
                action = policy.select_action(observation)
                self.assertEqual(action.shape, (1, 7))
                self.assertTrue(torch.isfinite(action).all())


    def test_status_mask_zeroes_invalid_pixels(self) -> None:
        normalizer = ObservationNormalizer.identity()
        distance = torch.tensor(
            [[[100.0, 200.0, 300.0, 400.0]] * 4] * 9, dtype=torch.float32
        ).reshape(1, 9, 4, 4)  # one batch, all 9 sensors
        present = torch.ones(1, 9)
        status = torch.full((1, 9, 4, 4), 5.0)
        status[0, :, 0, 0] = 255.0  # no target
        status[0, :, 1, 1] = 9.0  # weak signal: kept
        status[0, :, 2, 2] = 0.0  # unknown code: filtered

        masked = normalizer.normalize_beaver(distance, present, status)
        expected = normalizer.normalize_beaver(distance, present, None)
        # 5 and 9 pixels keep their normalized value.
        self.assertTrue(torch.equal(masked[0, :, 1, 1], expected[0, :, 1, 1]))
        # 255 and 0 pixels are zeroed.
        self.assertTrue(torch.equal(masked[0, :, 0, 0], torch.zeros(9)))
        self.assertTrue(torch.equal(masked[0, :, 2, 2], torch.zeros(9)))

        # Without status the same pixels are untouched (backward compatible).
        self.assertTrue(torch.equal(
            normalizer.normalize_beaver(distance, present, None)[0, :, 0, 0],
            expected[0, :, 0, 0],
        ))

        # Sensor-level mask still applies: a disconnected sensor reads zero
        # even where the status says valid.
        absent = torch.zeros(1, 9)
        absent[0, 0] = 0.0
        sensor_masked = normalizer.normalize_beaver(distance, absent, status)
        self.assertTrue(torch.equal(sensor_masked[0, 0], torch.zeros(4, 4)))

        # augmented_state flattens the same masked values into state.
        state = torch.zeros(1, 7)
        augmented = normalizer.augmented_state(state, distance, present, status)
        self.assertEqual(augmented.shape, (1, 7 + 9 * 4 * 4 + 9))
        # 255 pixel (flattened index 0) is zero; 9 pixel (index 5) is kept.
        self.assertEqual(augmented[0, 7], 0.0)
        self.assertNotEqual(augmented[0, 7 + 5], 0.0)


if __name__ == "__main__":
    unittest.main()
