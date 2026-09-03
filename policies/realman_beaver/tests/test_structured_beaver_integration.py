from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import (
    STRUCTURED_BEAVER_DP_VARIANTS,
    ModelConfig,
    RealmanBeaverConfig,
    load_config,
)
from policies.realman_beaver.dataset import ObservationNormalizer, RealmanPolicyDataset
from policies.realman_beaver.modeling import StructuredBeaverDPPolicy, build_policy
from policies.realman_beaver.train import ExponentialMovingAverage


def tiny_config(variant: str) -> RealmanBeaverConfig:
    if variant == "dp_beaver_key4":
        beaver_feature_dim = 128
    elif variant == "dp_beaver_key4_pca":
        beaver_feature_dim = 16
    else:
        beaver_feature_dim = 64
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant=variant,
            beaver_feature_dim=beaver_feature_dim,
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


def policy_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "image": torch.rand(batch_size, 2, 3, 64, 64),
        "state": torch.randn(batch_size, 2, 7),
        "action": torch.randn(batch_size, 8, 7),
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
        "beaver_distance": torch.rand(batch_size, 2, 9, 4, 4) * 2550.0,
        "beaver_present": torch.ones(batch_size, 2, 9),
        "beaver_status": torch.full((batch_size, 2, 9, 4, 4), 5.0),
    }


class _FakeLeRobotDataset:
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
        }
        observation_steps = len(delta_timestamps["observation.state"])
        action_steps = len(delta_timestamps["action"])
        self.item = {
            "observation.images.camera_0": torch.rand(observation_steps, 3, 64, 64),
            "observation.state": torch.randn(observation_steps, 7),
            "action": torch.randn(action_steps, 7),
            "action_is_pad": torch.zeros(action_steps, dtype=torch.bool),
            "observation.beaver.distance_mm": torch.rand(observation_steps, 9, 4, 4),
            "observation.beaver.present": torch.ones(observation_steps, 9),
            "observation.beaver.target_status": torch.full(
                (observation_steps, 9, 4, 4), 5.0
            ),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        return self.item


class StructuredBeaverIntegrationTest(unittest.TestCase):
    def test_all_yaml_configs_load(self) -> None:
        config_dir = Path(__file__).parents[1] / "configs"
        names = {
            "original_dp",
            "dp_beaver",
            *STRUCTURED_BEAVER_DP_VARIANTS,
            "rdp_like",
            "fm",
            "fm_beaver",
            "rfm",
        }
        for name in names:
            with self.subTest(name=name):
                config = load_config(config_dir / f"{name}.yaml")
                self.assertEqual(config.model.variant, name)

    def test_structured_config_rejects_nonpositive_encoder_settings(self) -> None:
        config = tiny_config("dp_beaver_enc")
        config.model.beaver_sensor_feature_dim = 0
        with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
            config.validate()

        config = tiny_config("dp_beaver_near")
        config.model.beaver_near_threshold_mm = 0.0
        with self.assertRaisesRegex(ValueError, "threshold"):
            config.validate()

        config = tiny_config("dp_beaver_near_gate")
        config.model.beaver_gate_hidden_dim = 0
        with self.assertRaisesRegex(ValueError, "gate_hidden"):
            config.validate()

    def test_direct_dataset_routes_aligned_beaver_history(self) -> None:
        for variant in STRUCTURED_BEAVER_DP_VARIANTS:
            with self.subTest(variant=variant):
                config = tiny_config(variant)
                with patch(
                    "policies.realman_beaver.dataset.LeRobotDataset",
                    _FakeLeRobotDataset,
                ):
                    dataset = RealmanPolicyDataset(config, [0], stage="policy")
                sample = dataset[0]
                self.assertEqual(
                    set(sample),
                    {
                        "image",
                        "state",
                        "action",
                        "action_is_pad",
                        "beaver_distance",
                        "beaver_present",
                        "beaver_status",
                    },
                )
                self.assertEqual(sample["beaver_distance"].shape, (2, 9, 4, 4))
                self.assertEqual(sample["beaver_status"].shape, (2, 9, 4, 4))
                self.assertEqual(sample["beaver_present"].shape, (2, 9))
                self.assertEqual(
                    dataset.dataset.delta_timestamps[
                        config.dataset.beaver_distance_key
                    ],
                    [-1 / config.dataset.fps, 0.0],
                )

    def test_policy_loss_prediction_dimensions_and_encoder_gradients(self) -> None:
        for variant in STRUCTURED_BEAVER_DP_VARIANTS:
            with self.subTest(variant=variant):
                config = tiny_config(variant)
                policy = build_policy(config, ObservationNormalizer.identity())
                self.assertIsInstance(policy, StructuredBeaverDPPolicy)
                self.assertIsInstance(policy.native_policy, DiffusionPolicy)
                self.assertEqual(
                    policy.native_policy.config.input_features[
                        "observation.state"
                    ].shape,
                    (7 + config.model.beaver_feature_dim,),
                )

                batch = policy_batch()
                self.assertEqual(
                    policy._state(batch).shape,
                    (2, 2, 7 + config.model.beaver_feature_dim),
                )
                loss, metrics = policy.compute_loss(batch)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(metrics["loss"], 0.0)
                loss.backward()

                if variant == "dp_beaver_key4":
                    self._assert_nonzero_gradient(policy, "beaver_encoder.sensor_mlps")
                    self._assert_nonzero_gradient(policy, "beaver_encoder.layer_norm")
                elif variant == "dp_beaver_key4_pca":
                    self._assert_nonzero_gradient(policy, "beaver_encoder.layer_norm")
                    self.assertFalse(
                        any(
                            "pca_" in name
                            for name, _ in policy.beaver_encoder.named_parameters()
                        )
                    )
                else:
                    self._assert_nonzero_gradient(policy, "beaver_encoder.sensor_mlp")
                    self._assert_nonzero_gradient(policy, "beaver_encoder.sensor_embedding")
                    self._assert_nonzero_gradient(policy, "beaver_encoder.fusion_mlp")
                if variant == "dp_beaver_near_gate":
                    self._assert_nonzero_gradient(policy, "beaver_encoder.gate_mlp")

                actions = policy.predict_action_chunk(batch)
                self.assertEqual(actions.shape, (2, 2, 7))
                self.assertTrue(torch.isfinite(actions).all())

    def test_checkpoint_reconstructs_structured_variant_and_encoder(self) -> None:
        config = tiny_config("dp_beaver_near_gate")
        policy = build_policy(config, ObservationNormalizer.identity())
        ema = ExponentialMovingAverage(policy, decay=0.9)
        gate_name = next(name for name in ema.shadow if "gate_mlp" in name)
        ema.shadow[gate_name].fill_(0.125)
        self.assertTrue(any("sensor_mlp" in name for name in ema.shadow))
        self.assertTrue(any("sensor_embedding" in name for name in ema.shadow))
        self.assertTrue(any("fusion_mlp" in name for name in ema.shadow))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(
                {
                    "kind": config.model.variant,
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": ema.state_dict(),
                },
                path,
            )
            restored = load_policy(path, use_ema=True)
        self.assertIsInstance(restored, StructuredBeaverDPPolicy)
        self.assertIsNotNone(restored.beaver_encoder.gate_mlp)
        torch.testing.assert_close(
            dict(restored.named_parameters())[gate_name], ema.shadow[gate_name]
        )
        torch.testing.assert_close(
            restored.beaver_encoder.sensor_embedding,
            ema.shadow["beaver_encoder.sensor_embedding"],
        )

    def test_checkpoint_loader_accepts_pre_temporal_structured_checkpoint(self) -> None:
        config = tiny_config("dp_beaver_near_gate")
        policy = build_policy(config, ObservationNormalizer.identity())
        legacy_state = policy.state_dict()
        for name in ObservationNormalizer._TEMPORAL_BUFFER_NAMES:
            legacy_state.pop(f"normalizer.{name}")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(
                {
                    "kind": config.model.variant,
                    "config": config.to_dict(),
                    "model": legacy_state,
                    "ema": {},
                },
                path,
            )
            restored = load_policy(path)

        self.assertIsInstance(restored, StructuredBeaverDPPolicy)
        self.assertFalse(restored.normalizer.has_temporal_beaver_statistics)

    def test_online_select_action_accepts_one_structured_beaver_frame(self) -> None:
        source = policy_batch(batch_size=1)
        observation = {
            "image": source["image"][0, 0],
            "state": source["state"][0, 0],
            "beaver_distance": source["beaver_distance"][0, 0],
            "beaver_present": source["beaver_present"][0, 0],
            "beaver_status": source["beaver_status"][0, 0],
        }
        for variant in STRUCTURED_BEAVER_DP_VARIANTS:
            with self.subTest(variant=variant):
                policy = build_policy(
                    tiny_config(variant), ObservationNormalizer.identity()
                ).eval()
                policy.reset()
                action = policy.select_action(observation)
                self.assertEqual(action.shape, (1, 7))
                self.assertTrue(torch.isfinite(action).all())

    def test_existing_dp_state_dimensions_remain_unchanged(self) -> None:
        for variant, expected in (("original_dp", 7), ("dp_beaver", 160)):
            with self.subTest(variant=variant):
                policy = build_policy(
                    tiny_config(variant), ObservationNormalizer.identity()
                )
                self.assertEqual(
                    policy.native_policy.config.input_features[
                        "observation.state"
                    ].shape,
                    (expected,),
                )

    def test_ablation_modules_are_only_constructed_when_used(self) -> None:
        enc = build_policy(
            tiny_config("dp_beaver_enc"), ObservationNormalizer.identity()
        ).beaver_encoder
        near = build_policy(
            tiny_config("dp_beaver_near"), ObservationNormalizer.identity()
        ).beaver_encoder
        gated = build_policy(
            tiny_config("dp_beaver_near_gate"), ObservationNormalizer.identity()
        ).beaver_encoder
        key4 = build_policy(
            tiny_config("dp_beaver_key4"), ObservationNormalizer.identity()
        ).beaver_encoder
        pca = build_policy(
            tiny_config("dp_beaver_key4_pca"), ObservationNormalizer.identity()
        ).beaver_encoder
        self.assertFalse(enc.uses_near)
        self.assertFalse(hasattr(enc, "gate_mlp"))
        self.assertTrue(near.uses_near)
        self.assertFalse(hasattr(near, "gate_mlp"))
        self.assertTrue(gated.uses_near)
        self.assertIsNotNone(gated.gate_mlp)
        self.assertFalse(key4.uses_pca)
        self.assertTrue(pca.uses_pca)
        self.assertFalse(any("pca_" in name for name, _ in pca.named_parameters()))

    def _assert_nonzero_gradient(
        self, policy: StructuredBeaverDPPolicy, name_fragment: str
    ) -> None:
        gradients = [
            parameter.grad
            for name, parameter in policy.named_parameters()
            if name_fragment in name
        ]
        self.assertTrue(gradients, f"No parameters matched {name_fragment}")
        self.assertTrue(
            any(
                gradient is not None and torch.count_nonzero(gradient).item() > 0
                for gradient in gradients
            ),
            f"No nonzero gradient reached {name_fragment}",
        )


if __name__ == "__main__":
    unittest.main()
