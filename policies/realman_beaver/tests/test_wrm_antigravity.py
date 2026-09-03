"""Comprehensive test suite for WRM_antigravity policy and encoder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from eval_policy import policy_needs_beaver, validate_deployable_checkpoint
from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import (
    ANTIGRAVITY_BEAVER_VARIANT,
    ModelConfig,
    RealmanBeaverConfig,
    load_config,
)
from policies.realman_beaver.dataset import ObservationNormalizer, RealmanPolicyDataset
from policies.realman_beaver.modeling import AntigravityDPPolicy, build_policy
from policies.realman_beaver.modules import AntigravityBeaverEncoder


def antigravity_statistics() -> dict[str, torch.Tensor]:
    return {
        "p5": torch.tensor((10.0, 15.0, 20.0, 25.0)),
        "p95": torch.tensor((300.0, 320.0, 340.0, 360.0)),
        "median": torch.tensor((100.0, 110.0, 120.0, 130.0)),
        "sensor_indices": torch.tensor((1, 2, 5, 6)),
    }


def tiny_antigravity_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant=ANTIGRAVITY_BEAVER_VARIANT,
            beaver_antigravity_sensors=("01", "02", "10", "11"),
            beaver_antigravity_history_steps=4,
            beaver_antigravity_motion_delta_steps=1,
            beaver_antigravity_motion_delta_long_steps=2,
            beaver_antigravity_lag_steps=(1, 2, 3),
            beaver_antigravity_proximity_scales_mm=(25.0, 75.0),
            beaver_antigravity_spatial_hidden_dim=32,
            beaver_antigravity_token_dim=16,
            beaver_antigravity_temporal_hidden_dim=16,
            beaver_antigravity_transformer_layers=1,
            beaver_antigravity_attention_heads=4,
            beaver_antigravity_feature_dim=16,
            beaver_antigravity_grasp_hidden_dim=16,
            beaver_antigravity_grasp_loss_weight=0.2,
            beaver_antigravity_enclosure_loss_weight=0.1,
            beaver_antigravity_noise_std_mm=0.0,
            beaver_antigravity_pixel_dropout=0.0,
            beaver_antigravity_sensor_dropout=0.0,
            beaver_antigravity_terminal_hold_damping=True,
            beaver_antigravity_hold_threshold=0.85,
            beaver_antigravity_max_damping=0.4,
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


class _FakeAntigravityLeRobotDataset:
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


class TestWRMAntigravity(unittest.TestCase):
    """Test WRM_antigravity policy, encoder, dataset alignment, and checkpoints."""

    def test_configuration_loading_and_parameter_budget(self) -> None:
        """Verify full production config loads and satisfies the <100M parameter limit."""
        stratified_path = Path(
            "policies/realman_beaver/configs/WRM_antigravity.yaml"
        )
        all_train_path = Path(
            "policies/realman_beaver/configs/WRM_antigravity_all_train.yaml"
        )
        self.assertTrue(stratified_path.is_file(), "Missing stratified config")
        self.assertTrue(all_train_path.is_file(), "Missing all-train config")

        config = load_config(stratified_path)
        config.validate()
        self.assertEqual(config.model.variant, "WRM_antigravity")
        self.assertEqual(len(config.dataset.val_episodes), 25)

        all_config = load_config(all_train_path)
        all_config.validate()
        self.assertEqual(all_config.model.variant, "WRM_antigravity")
        self.assertIsNone(all_config.dataset.val_episodes)

        normalizer = ObservationNormalizer.identity(
            config.model.state_dim,
            config.model.action_dim,
            temporal_beaver_statistics=antigravity_statistics(),
        )
        policy = build_policy(config, normalizer)
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(
            f"\n[WRM_antigravity] Trainable Parameters: {trainable_params:,} "
            f"({trainable_params / 1e6:.2f}M) - Budget Limit: < 100M"
        )
        self.assertLess(
            trainable_params,
            100_000_000,
            f"Trainable parameters ({trainable_params}) exceed 100M limit",
        )
        self.assertGreater(trainable_params, 50_000_000)

    def test_encoder_and_loss_forward_backward(self) -> None:
        """Test loss computation and non-zero gradient backpropagation."""
        config = tiny_antigravity_config()
        normalizer = ObservationNormalizer.identity(
            config.model.state_dim,
            config.model.action_dim,
            temporal_beaver_statistics=antigravity_statistics(),
        )
        policy = AntigravityDPPolicy(config, normalizer)
        batch = {
            "image": torch.rand(2, 2, 3, 64, 64),
            "state": torch.rand(2, 2, 7),
            "delta_q": torch.rand(2, 2, 7),
            "delta_q_long": torch.rand(2, 2, 7),
            "beaver_history_distance": torch.rand(2, 2, 4, 9, 4, 4) * 200.0 + 10.0,
            "beaver_history_status": torch.full((2, 2, 4, 9, 4, 4), 5, dtype=torch.int16),
            "beaver_history_present": torch.ones((2, 2, 4, 9), dtype=torch.float32),
            "action": torch.rand(2, 8, 7),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "grasp_state": torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        }
        loss, metrics = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("diffusion_loss", metrics)
        self.assertIn("grasp_loss", metrics)
        self.assertIn("enclosure_loss", metrics)
        self.assertIn("grasp_accuracy", metrics)
        self.assertIn("z_beaver_std", metrics)

        loss.backward()
        for name, param in policy.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad, f"Parameter {name} has no gradient after backward"
                )
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Parameter {name} has non-finite gradients",
                )

    def test_all_three_mandatory_modalities_sensitivity(self) -> None:
        """Prove that changing RGB, joint state, or Beaver distance affects policy outputs."""
        config = tiny_antigravity_config()
        normalizer = ObservationNormalizer.identity(
            config.model.state_dim,
            config.model.action_dim,
            temporal_beaver_statistics=antigravity_statistics(),
        )
        policy = AntigravityDPPolicy(config, normalizer)
        policy.eval()

        base_batch = {
            "image": torch.rand(1, 2, 3, 64, 64),
            "state": torch.rand(1, 2, 7),
            "delta_q": torch.rand(1, 2, 7),
            "delta_q_long": torch.rand(1, 2, 7),
            "beaver_history_distance": torch.rand(1, 2, 4, 9, 4, 4) * 150.0 + 20.0,
            "beaver_history_status": torch.full((1, 2, 4, 9, 4, 4), 5, dtype=torch.int16),
            "beaver_history_present": torch.ones((1, 2, 4, 9), dtype=torch.float32),
            "action": torch.rand(1, 8, 7),
            "action_is_pad": torch.zeros(1, 8, dtype=torch.bool),
            "grasp_state": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }

        with torch.no_grad():
            prep_base, g_base, enc_base, z_base, _ = policy._prepare(
                base_batch, include_action=False, return_auxiliary=True
            )

        # 1. Sensitivity to RGB visual stream
        img_mod_batch = dict(base_batch)
        img_mod_batch["image"] = base_batch["image"] + torch.randn_like(base_batch["image"])
        with torch.no_grad():
            prep_img = policy._prepare(img_mod_batch, include_action=False)
        self.assertFalse(
            torch.allclose(
                prep_base["observation.images.camera_0"],
                prep_img["observation.images.camera_0"],
            ),
            "Policy image conditioning did not change when image inputs changed",
        )

        # 2. Sensitivity to Joint state stream
        state_mod_batch = dict(base_batch)
        state_mod_batch["state"] = base_batch["state"] + 0.5
        with torch.no_grad():
            prep_state, g_state, _, _, _ = policy._prepare(
                state_mod_batch, include_action=False, return_auxiliary=True
            )
        self.assertFalse(
            torch.allclose(prep_base["observation.state"], prep_state["observation.state"]),
            "Policy state conditioning did not change when joint states changed",
        )
        self.assertFalse(
            torch.allclose(g_base, g_state),
            "Grasp state head did not respond to joint state variation",
        )

        # 3. Sensitivity to Beaver proximity stream
        beaver_mod_batch = dict(base_batch)
        beaver_mod_batch["beaver_history_distance"] = base_batch["beaver_history_distance"] + 200.0
        with torch.no_grad():
            prep_beaver, g_beaver, enc_beaver, z_beaver, _ = policy._prepare(
                beaver_mod_batch, include_action=False, return_auxiliary=True
            )
        self.assertFalse(
            torch.allclose(z_base, z_beaver),
            "Antigravity Beaver feature z_beaver did not change when Beaver distances changed",
        )
        self.assertFalse(
            torch.allclose(g_base, g_beaver),
            "Grasp logit did not change when Beaver distances changed",
        )
        self.assertFalse(
            torch.allclose(enc_base, enc_beaver),
            "Enclosure metric did not change when Beaver distances changed",
        )

    def test_beaver_status_masking_and_all_invalid_fallback(self) -> None:
        """Verify explicit status masking (only 5 and 9 valid) and zero-fallback on invalid frames."""
        encoder = AntigravityBeaverEncoder(
            n_sensors=9,
            sensor_indices=(1, 2, 5, 6),
            history_steps=4,
            lag_steps=(1, 2, 3),
            valid_statuses=(5, 9),
            proximity_scales_mm=(25.0, 75.0),
            token_dim=16,
            temporal_hidden_dim=16,
            output_dim=16,
            noise_std_mm=0.0,
            pixel_dropout=0.0,
            sensor_dropout=0.0,
        )
        encoder.set_normalization_statistics(
            p5=torch.tensor([10.0, 10.0, 10.0, 10.0]),
            p95=torch.tensor([300.0, 300.0, 300.0, 300.0]),
            median=torch.tensor([100.0, 100.0, 100.0, 100.0]),
        )
        encoder.eval()

        # Test A: Status filtering: status 0, 1, 2, 3, 4, 6, 7, 8, 255 must be masked
        dist = torch.full((1, 4, 9, 4, 4), 50.0)
        status_invalid = torch.full((1, 4, 9, 4, 4), 255, dtype=torch.int16)
        present = torch.ones((1, 4, 9), dtype=torch.float32)

        feat_invalid, inter_invalid = encoder(
            dist, status_invalid, present, return_intermediates=True
        )
        self.assertTrue(torch.isfinite(feat_invalid).all())
        self.assertTrue(
            torch.all(feat_invalid == 0.0),
            "All-invalid status should cleanly evaluate to zero fallback vector",
        )
        self.assertEqual(inter_invalid["enclosure_score"].item(), 0.0)

        # Test B: Valid status 5 and 9 produce non-zero feature vector
        status_valid = torch.full((1, 4, 9, 4, 4), 5, dtype=torch.int16)
        status_valid[:, :, :, :2, :2] = 9
        feat_valid, inter_valid = encoder(
            dist, status_valid, present, return_intermediates=True
        )
        self.assertTrue(torch.isfinite(feat_valid).all())
        self.assertFalse(
            torch.all(feat_valid == 0.0),
            "Valid statuses 5 and 9 should produce non-zero feature vectors",
        )

        # Test C: Sensor present == 0 is masked
        present_zero = torch.zeros((1, 4, 9), dtype=torch.float32)
        feat_nopresent, _ = encoder(
            dist, status_valid, present_zero, return_intermediates=True
        )
        self.assertTrue(torch.all(feat_nopresent == 0.0))

    def test_history_reset_and_episode_boundary_isolation(self) -> None:
        """Verify that policy.reset() isolates consecutive episodes without observation leakage."""
        config = tiny_antigravity_config()
        normalizer = ObservationNormalizer.identity(
            config.model.state_dim,
            config.model.action_dim,
            temporal_beaver_statistics=antigravity_statistics(),
        )
        policy = AntigravityDPPolicy(config, normalizer)
        policy.eval()

        # Step episode 1
        obs_ep1 = {
            "image": torch.rand(3, 64, 64),
            "state": torch.full((7,), 1.0),
            "beaver_distance": torch.full((9, 4, 4), 50.0),
            "beaver_status": torch.full((9, 4, 4), 5, dtype=torch.int16),
            "beaver_present": torch.ones(9, dtype=torch.float32),
        }
        for _ in range(5):
            _ = policy.select_action(obs_ep1)

        self.assertGreater(len(policy._antigravity_history), 0)

        # Reset for episode 2
        policy.reset()
        self.assertEqual(len(policy._antigravity_history), 0)
        self.assertTrue(policy.last_replanned)
        self.assertIsNone(policy._hold_pose)

        # Step episode 2 with different inputs
        obs_ep2 = {
            "image": torch.rand(3, 64, 64),
            "state": torch.full((7,), -1.0),
            "beaver_distance": torch.full((9, 4, 4), 200.0),
            "beaver_status": torch.full((9, 4, 4), 5, dtype=torch.int16),
            "beaver_present": torch.ones(9, dtype=torch.float32),
        }
        action = policy.select_action(obs_ep2)
        self.assertEqual(tuple(action.shape)[-1], 7)
        self.assertTrue(torch.isfinite(action).all())

    def test_checkpoint_roundtrip_ema_and_normalizer(self) -> None:
        """Verify saving checkpoint with EMA and loading via load_policy()."""
        config = tiny_antigravity_config()
        normalizer = ObservationNormalizer.identity(
            config.model.state_dim,
            config.model.action_dim,
            temporal_beaver_statistics=antigravity_statistics(),
        )
        policy = AntigravityDPPolicy(config, normalizer)
        ema_state = {
            name: param.detach().clone() + 0.01
            for name, param in policy.named_parameters()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "last.pt"
            torch.save(
                {
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": ema_state,
                    "kind": "WRM_antigravity",
                    "global_step": 100000,
                    "epoch": 200,
                },
                ckpt_path,
            )

            loaded = load_policy(ckpt_path, device="cpu", use_ema=True)
            self.assertIsInstance(loaded, AntigravityDPPolicy)
            self.assertEqual(loaded.config.model.variant, "WRM_antigravity")
            self.assertTrue(loaded.normalizer.has_temporal_beaver_statistics)
            self.assertTrue(
                torch.allclose(
                    loaded.normalizer.beaver_temporal_p5,
                    antigravity_statistics()["p5"],
                )
            )

    def test_dataset_item_alignment(self) -> None:
        """Test dataset slicing and multi-delta construction."""
        config = tiny_antigravity_config()
        with patch(
            "policies.realman_beaver.dataset.LeRobotDataset",
            _FakeAntigravityLeRobotDataset,
        ):
            dataset = RealmanPolicyDataset(config, [0], stage="policy")
            sample = dataset[0]
            self.assertEqual(sample["image"].shape, (2, 3, 64, 64))
            self.assertEqual(sample["state"].shape, (2, 7))
            self.assertEqual(sample["delta_q"].shape, (2, 7))
            self.assertEqual(sample["delta_q_long"].shape, (2, 7))
            self.assertEqual(sample["action"].shape, (8, 7))
            self.assertEqual(sample["beaver_history_distance"].shape, (2, 4, 9, 4, 4))
            self.assertEqual(sample["beaver_history_status"].shape, (2, 4, 9, 4, 4))
            self.assertEqual(sample["beaver_history_present"].shape, (2, 4, 9))
            self.assertEqual(sample["grasp_state"].shape, (2,))

    def test_smoke_training_step(self) -> None:
        """Run two actual optimization steps on device/CPU."""
        config = tiny_antigravity_config()
        normalizer = ObservationNormalizer.identity(
            config.model.state_dim,
            config.model.action_dim,
            temporal_beaver_statistics=antigravity_statistics(),
        )
        policy = AntigravityDPPolicy(config, normalizer)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        batch = {
            "image": torch.rand(2, 2, 3, 64, 64),
            "state": torch.rand(2, 2, 7),
            "delta_q": torch.rand(2, 2, 7),
            "delta_q_long": torch.rand(2, 2, 7),
            "beaver_history_distance": torch.rand(2, 2, 4, 9, 4, 4) * 200.0 + 10.0,
            "beaver_history_status": torch.full((2, 2, 4, 9, 4, 4), 5, dtype=torch.int16),
            "beaver_history_present": torch.ones((2, 2, 4, 9), dtype=torch.float32),
            "action": torch.rand(2, 8, 7),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "grasp_state": torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        }
        for _ in range(2):
            optimizer.zero_grad()
            loss, metrics = policy.compute_loss(batch)
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def test_eval_policy_registration(self) -> None:
        """Verify eval_policy registry recognizing WRM_antigravity."""
        self.assertTrue(policy_needs_beaver("WRM_antigravity"))
        summary = {"variant": "WRM_antigravity", "kind": "WRM_antigravity"}
        self.assertEqual(
            validate_deployable_checkpoint(summary), "WRM_antigravity"
        )


if __name__ == "__main__":
    unittest.main()
