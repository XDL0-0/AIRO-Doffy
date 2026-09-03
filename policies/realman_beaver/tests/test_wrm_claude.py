"""Tests for the WRM_claude pose-anchored delta policy.

Covers: config validation + param budget, forward/loss/backward, per-modality
sensitivity and offline ablations, per-pixel Beaver status masking + all-invalid
fallback, history reset + episode-boundary behavior, action shape/finiteness,
checkpoint round trip (EMA + learned statistics), reset/select_action, eval
integration, and a one-step CPU training smoke.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import (
    ModelConfig,
    RealmanBeaverConfig,
    load_config,
)
from policies.realman_beaver.dataset import ObservationNormalizer
from policies.realman_beaver.modeling import ClaudeBeaverDPPolicy, build_policy
from policies.realman_beaver.modules import ContactFieldEncoder
from policies.realman_beaver.offline_metrics import (
    action_smoothness,
    per_bottle_metrics,
    per_joint_trajectory_error,
    tightness_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def tiny_claude_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant="WRM_claude",
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


def claude_statistics() -> dict[str, torch.Tensor]:
    return {
        "p5": torch.full((4,), 100.0),
        "p95": torch.full((4,), 1100.0),
        "median": torch.full((4,), 600.0),
        "sensor_indices": torch.tensor((1, 2, 5, 6)),
    }


def claude_normalizer() -> ObservationNormalizer:
    return ObservationNormalizer.identity(
        temporal_beaver_statistics=claude_statistics(),
        action_delta_statistics={"scale": torch.full((7,), 0.02)},
    )


def training_batch() -> dict[str, torch.Tensor]:
    return {
        "image": torch.rand(2, 2, 3, 64, 64),
        "state": torch.randn(2, 2, 7),
        "action": torch.randn(2, 8, 7),
        "action_delta": torch.randn(2, 8, 7) * 0.02,
        "delta_q": torch.randn(2, 2, 7),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        "beaver_history_distance": torch.rand(2, 2, 8, 9, 4, 4) * 1000.0 + 100.0,
        "beaver_history_status": torch.full((2, 2, 8, 9, 4, 4), 5.0),
        "beaver_history_present": torch.ones(2, 2, 8, 9),
        "grasp_state": torch.tensor(((0.0, 0.0), (0.0, 1.0))),
    }


def online_observation(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "image": torch.rand(batch_size, 3, 64, 64),
        "state": torch.randn(batch_size, 7),
        "beaver_distance": torch.full((batch_size, 9, 4, 4), 250.0),
        "beaver_status": torch.full((batch_size, 9, 4, 4), 5.0),
        "beaver_present": torch.ones(batch_size, 9),
    }


class ContactFieldEncoderTest(unittest.TestCase):
    """Unit tests for per-pixel masking, imputation, and all-invalid fallback."""

    def _encoder(self) -> ContactFieldEncoder:
        encoder = ContactFieldEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=8
        )
        encoder.set_normalization_statistics(
            p5=torch.full((4,), 100.0),
            p95=torch.full((4,), 1100.0),
            median=torch.full((4,), 600.0),
        )
        return encoder

    def test_invalid_pixel_is_masked_with_visible_missingness(self) -> None:
        encoder = self._encoder()
        encoder.eval()  # deterministic: no augmentation noise
        distance = torch.full((1, 8, 9, 4, 4), 600.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 8, 9)

        # Physical sensor 01, pixel (0, 0). The current frame is t7; lag-1 is
        # t6 (masked), lag-3 is t4, lag-6 is t1.
        distance[0, 4, 1, 0, 0] = 300.0
        distance[0, 6, 1, 0, 0] = 500.0
        status[0, 6, 1, 0, 0] = 255.0
        distance[0, 7, 1, 0, 0] = 900.0

        features, middle = encoder.preprocess(distance, status, present)
        self.assertEqual(features.shape, (1, 4, 4, 4, 15))
        pixel = features[0, 0, 0, 0]  # sensor 01, cell (0, 0), current frame
        # Feature layout: [observed_current, sensor_relative, global_relative,
        # proximity x4, lag1, lag3, lag6, lag1_valid, lag3_valid, lag6_valid,
        # valid, raw_zero].
        self.assertAlmostEqual(float(pixel[0]), 0.8, places=6)  # (900-100)/1000
        self.assertAlmostEqual(float(pixel[13]), 1.0, places=6)  # current cell is valid
        self.assertAlmostEqual(float(pixel[14]), 0.0, places=6)  # 900 mm is not a raw zero
        self.assertAlmostEqual(float(pixel[7]), 0.0, places=6)  # lag-1 invalid: t6 masked
        self.assertAlmostEqual(float(pixel[8]), 0.6, places=6)  # lag-3 delta: 0.8-0.2
        self.assertAlmostEqual(float(pixel[9]), 0.3, places=6)  # lag-6 delta: 0.8-0.5
        self.assertAlmostEqual(float(pixel[10]), 0.0, places=6)  # lag-1 pair invalid
        self.assertAlmostEqual(float(pixel[11]), 1.0, places=6)  # lag-3 pair valid
        self.assertAlmostEqual(float(pixel[12]), 1.0, places=6)  # lag-6 pair valid
        self.assertEqual(middle["cell_available"].shape, (1, 4, 16))

        # When the current frame itself is masked, the cell contributes zero
        # geometry (never the per-sensor median) and stays visibly missing.
        status2 = status.clone()
        status2[0, 7, 1, 0, 0] = 255.0
        features2, _ = encoder.preprocess(distance, status2, present)
        pixel2 = features2[0, 0, 0, 0]
        self.assertAlmostEqual(float(pixel2[0]), 0.0, places=6)
        self.assertAlmostEqual(float(pixel2[13]), 0.0, places=6)
        self.assertAlmostEqual(float(pixel2[14]), 0.0, places=6)  # imputed, not a genuine zero

    def test_all_invalid_frames_produce_exact_zero_feature(self) -> None:
        encoder = self._encoder()
        distance = torch.full((2, 3, 8, 9, 4, 4), 600.0)
        status = torch.full_like(distance, 255.0)
        present = torch.ones(2, 3, 8, 9)
        feature = encoder(distance, status, present)
        self.assertEqual(feature.shape, (2, 3, 64))
        self.assertTrue(torch.equal(feature, torch.zeros_like(feature)))
        # The same input with valid statuses produces a non-zero feature.
        status_valid = torch.full_like(distance, 5.0)
        self.assertFalse(torch.allclose(encoder(distance, status_valid, present), feature))

    def test_encoder_refuses_unfitted_statistics(self) -> None:
        encoder = ContactFieldEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=8
        )
        distance = torch.full((1, 8, 9, 4, 4), 600.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 8, 9)
        with self.assertRaisesRegex(RuntimeError, "statistics"):
            encoder(distance, status, present)

    def test_changing_one_sensor_changes_the_representation(self) -> None:
        encoder = self._encoder()
        distance = torch.full((1, 8, 9, 4, 4), 600.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 8, 9)
        baseline = encoder(distance, status, present)
        changed = distance.clone()
        changed[:, 6:, 2] = 900.0  # sensor 02, last two history frames
        self.assertFalse(torch.allclose(baseline, encoder(changed, status, present)))


class ActionDeltaNormalizerTest(unittest.TestCase):
    def test_round_trip_and_safety_clamp(self) -> None:
        normalizer = claude_normalizer()
        self.assertTrue(normalizer.has_action_delta_statistics)
        # Round trip is exact within the per-joint scale.
        x = torch.rand(4, 3, 7) * 0.02
        restored = normalizer.denormalize_action_delta(
            normalizer.normalize_action_delta(x)
        )
        torch.testing.assert_close(restored, x)
        # Out-of-range normalized deltas clamp to the per-joint scale: the
        # deployment safety clamp is exercised, never weakened.
        huge = torch.full((2, 7), 5.0)
        clamped = normalizer.denormalize_action_delta(huge)
        self.assertTrue((clamped.abs() <= 0.02 + 1e-6).all())


class ClaudeBeaverPolicyTest(unittest.TestCase):
    def test_config_validation_and_parameter_budget(self) -> None:
        config = load_config(
            REPO_ROOT / "policies" / "realman_beaver" / "configs" / "WRM_claude.yaml"
        )
        config.validate()
        policy = build_policy(config, claude_normalizer())
        self.assertIsInstance(policy, ClaudeBeaverDPPolicy)
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.assertLess(trainable, 100_000_000)
        self.assertGreater(trainable, 1_000_000)

    def test_requires_learned_statistics(self) -> None:
        with self.assertRaisesRegex(ValueError, "statistics"):
            build_policy(tiny_claude_config(), ObservationNormalizer.identity())
        with self.assertRaisesRegex(ValueError, "action delta"):
            build_policy(
                tiny_claude_config(),
                ObservationNormalizer.identity(
                    temporal_beaver_statistics=claude_statistics()
                ),
            )

    def test_forward_loss_backward_and_all_modality_gradients(self) -> None:
        policy = build_policy(tiny_claude_config(), claude_normalizer())
        # State conditioning is 79-D: [q, dq, contact(64), p_grasp].
        self.assertEqual(
            policy.diffusion_model.config.robot_state_feature.shape, (79,)
        )
        batch = training_batch()
        loss, metrics = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        for key in (
            "loss",
            "diffusion_loss",
            "grasp_loss",
            "smoothness_loss",
            "contact_valid_fraction",
        ):
            self.assertIn(key, metrics)
        self.assertTrue(math.isfinite(metrics["smoothness_loss"]))
        loss.backward()

        named = dict(policy.named_parameters())
        # (1) Vision path.
        self.assertGreater(
            sum(
                p.grad.abs().sum()
                for name, p in named.items()
                if "rgb_encoder" in name and p.grad is not None
            ),
            0,
        )
        # (2) Proprioceptive path (U-Net state conditioning).
        self.assertGreater(
            sum(
                p.grad.abs().sum()
                for name, p in named.items()
                if "unet" in name and p.grad is not None
            ),
            0,
        )
        # (3) Beaver contact path: encoder tokens + FiLM gate.
        self.assertGreater(policy.contact_encoder.sensor_embedding.grad.abs().sum(), 0)
        self.assertGreater(
            sum(
                p.grad.abs().sum()
                for name, p in named.items()
                if "contact_gate" in name and p.grad is not None
            ),
            0,
        )
        self.assertGreater(policy.grasp_state_head[0].weight.grad.abs().sum(), 0)

    def test_offline_zeroing_and_shuffling_ablations(self) -> None:
        policy = build_policy(tiny_claude_config(), claude_normalizer())
        batch = training_batch()
        with torch.no_grad():
            reference, _ = policy.compute_loss(batch)

            image_zeroed = dict(batch)
            image_zeroed["image"] = torch.zeros_like(batch["image"])
            loss_image, _ = policy.compute_loss(image_zeroed)
            self.assertTrue(torch.isfinite(loss_image))
            self.assertFalse(torch.allclose(loss_image, reference))

            beaver_zeroed = dict(batch)
            beaver_zeroed["beaver_history_distance"] = torch.zeros_like(
                batch["beaver_history_distance"]
            )
            loss_beaver, _ = policy.compute_loss(beaver_zeroed)
            self.assertTrue(torch.isfinite(loss_beaver))
            self.assertFalse(torch.allclose(loss_beaver, reference))

            beaver_invalid = dict(batch)
            beaver_invalid["beaver_history_status"] = torch.full_like(
                batch["beaver_history_status"], 255.0
            )
            loss_invalid, metrics_invalid = policy.compute_loss(beaver_invalid)
            self.assertTrue(torch.isfinite(loss_invalid))
            self.assertFalse(torch.allclose(loss_invalid, reference))
            # All-invalid contact input degrades to the identity gate.
            self.assertEqual(metrics_invalid["contact_valid_fraction"], 0.0)

            action_shuffled = dict(batch)
            action_shuffled["action_delta"] = batch["action_delta"].roll(
                shifts=3, dims=1
            )
            loss_shuffled, _ = policy.compute_loss(action_shuffled)
            self.assertTrue(torch.isfinite(loss_shuffled))
            self.assertFalse(torch.allclose(loss_shuffled, reference))

    def test_action_shape_finiteness_and_chunk_round_trip(self) -> None:
        policy = build_policy(tiny_claude_config(), claude_normalizer())
        actions = policy.predict_action_chunk(training_batch())
        self.assertEqual(actions.shape, (2, 2, 7))  # (b, n_action_steps, 7)
        self.assertTrue(torch.isfinite(actions).all())

    def test_checkpoint_round_trip_including_learned_preprocessing(self) -> None:
        config = tiny_claude_config()
        policy = build_policy(config, claude_normalizer())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "wrm_claude.pt"
            torch.save(
                {
                    "kind": "WRM_claude",
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": {},
                },
                checkpoint_path,
            )
            restored = load_policy(checkpoint_path)

        self.assertIsInstance(restored, ClaudeBeaverDPPolicy)
        self.assertTrue(restored.contact_encoder.normalization_fitted)
        torch.testing.assert_close(
            restored.normalizer.beaver_temporal_p5,
            policy.normalizer.beaver_temporal_p5,
        )
        torch.testing.assert_close(
            restored.normalizer.action_delta_scale,
            policy.normalizer.action_delta_scale,
        )
        torch.testing.assert_close(
            restored.contact_encoder.distance_median,
            policy.contact_encoder.distance_median,
        )

    def test_select_action_reset_and_replan_contract(self) -> None:
        policy = build_policy(tiny_claude_config(), claude_normalizer())
        observation = online_observation()
        first = policy.select_action(observation)
        self.assertEqual(first.shape, (1, 7))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(policy.last_replanned)

        # Executed targets stay within the clamped per-joint delta scale of
        # the executed predecessor (simulated closed loop: the robot follows).
        previous = first
        replans = 0
        for _ in range(6):
            step = dict(observation)
            step["state"] = previous.clone()
            target = policy.select_action(step)
            self.assertEqual(target.shape, (1, 7))
            self.assertTrue(torch.isfinite(target).all())
            self.assertLessEqual(float((target - previous).abs().max()), 0.02 + 1e-6)
            if policy.last_replanned:
                replans += 1
            previous = target
        self.assertEqual(replans, 3)  # chunk length 2 over 6 steps

        # reset() clears queues: the next call replans again.
        policy.reset()
        self.assertTrue(policy.last_replanned)
        target = policy.select_action(observation)
        self.assertTrue(policy.last_replanned)

        # A batch-size change forces a reset (episode-boundary isolation).
        policy.select_action(online_observation(1))
        self.assertFalse(policy.last_replanned)
        policy.select_action(online_observation(2))
        self.assertTrue(policy.last_replanned)

    def test_online_history_episode_boundary_and_state_dedup(self) -> None:
        policy = build_policy(tiny_claude_config(), claude_normalizer())
        batch = online_observation()
        first = policy._append_online_history(batch)
        self.assertEqual(first["beaver_history_distance"].shape, (1, 8, 9, 4, 4))
        self.assertEqual(first["delta_q"].shape, (1, 7))
        self.assertTrue(torch.equal(first["delta_q"], torch.zeros_like(first["delta_q"])))
        self.assertTrue(
            torch.equal(
                first["beaver_history_distance"][:, 0],
                first["beaver_history_distance"][:, -1],
            )
        )
        # Feeding the identical frame twice must not double-count it.
        policy._append_online_history(batch)
        self.assertEqual(len(policy._history), 1)
        changed = dict(batch)
        changed["beaver_distance"] = torch.full((1, 9, 4, 4), 750.0)
        changed["state"] = batch["state"] + 0.01
        second = policy._append_online_history(changed)
        self.assertTrue(
            torch.equal(
                second["beaver_history_distance"][:, -1],
                changed["beaver_distance"],
            )
        )
        self.assertFalse(torch.equal(second["delta_q"], torch.zeros(1, 7)))
        policy.reset()
        self.assertEqual(len(policy._history), 0)

    def test_one_step_training_smoke(self) -> None:
        policy = build_policy(tiny_claude_config(), claude_normalizer())
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        before = {name: p.clone() for name, p in policy.named_parameters()}
        loss, _ = policy.compute_loss(training_batch())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        moved = [
            name
            for name, p in policy.named_parameters()
            if not torch.equal(before[name], p)
        ]
        self.assertTrue(moved)
        self.assertTrue(torch.isfinite(loss))


class OfflineMetricsTest(unittest.TestCase):
    def test_trajectory_error_is_in_physical_radians(self) -> None:
        scales = torch.full((7,), 0.02)
        predictions = torch.zeros(2, 16, 7)
        targets = torch.ones(2, 16, 7) * 0.5  # 0.01 rad per joint
        metrics = per_joint_trajectory_error(
            predictions, targets, scales, per_joint=True
        )
        self.assertAlmostEqual(metrics["trajectory_error_rad"], 0.01, places=6)
        for joint in range(7):
            self.assertAlmostEqual(
                metrics[f"trajectory_error_rad_joint_{joint}"], 0.01, places=6
            )

    def test_smoothness_and_chunk_boundary(self) -> None:
        scales = torch.ones(7)
        smooth = torch.zeros(1, 16, 7)  # constant: zero second difference
        metrics = action_smoothness(smooth, scales, n_action_steps=8)
        self.assertEqual(metrics["action_smoothness_rad2"], 0.0)
        self.assertEqual(metrics["chunk_boundary_discontinuity_rad"], 0.0)
        # A kink at the chunk boundary is attributed to the boundary gap, and
        # also registers (correctly) in the interior second differences.
        kinked = torch.zeros(1, 16, 7)
        kinked[0, 8] = 1.0
        metrics = action_smoothness(kinked, scales, n_action_steps=8)
        self.assertEqual(metrics["chunk_boundary_discontinuity_rad"], 1.0)
        self.assertGreater(metrics["action_smoothness_rad2"], 0.0)

    def test_per_bottle_blocks_match_confirmed_split(self) -> None:
        episodes = torch.arange(125)
        values = episodes.float()  # bottle 1 mean 12, bottle 5 mean 112
        metrics = per_bottle_metrics(episodes, values)
        self.assertAlmostEqual(metrics["bottle_1_mean"], 12.0, places=6)
        self.assertAlmostEqual(metrics["bottle_3_mean"], 62.0, places=6)
        self.assertAlmostEqual(metrics["bottle_5_mean"], 112.0, places=6)
        self.assertEqual(metrics["bottle_5_count"], 25.0)

    def test_tightness_metrics_cover_precision_recall_f1_ece(self) -> None:
        labels = torch.tensor((0.0, 0.0, 1.0, 1.0, 1.0, 1.0))
        probabilities = torch.tensor((0.1, 0.4, 0.4, 0.6, 0.8, 0.9))
        metrics = tightness_metrics(probabilities, labels)
        # threshold 0.5: predicted positive = 3 (tp) of which all are positive
        # -> precision 1.0; recall 3/4.
        self.assertAlmostEqual(metrics["tightness_precision"], 1.0, places=6)
        self.assertAlmostEqual(metrics["tightness_recall"], 0.75, places=6)
        self.assertAlmostEqual(metrics["tightness_f1"], 6.0 / 7.0, places=6)
        self.assertGreaterEqual(
            metrics["tightness_expected_calibration_error"], 0.0
        )
        self.assertTrue(
            math.isfinite(metrics["tightness_expected_calibration_error"])
        )


class EvalIntegrationTest(unittest.TestCase):
    @staticmethod
    def _stub_hardware_modules() -> None:
        """RealSense/airo-robots/scipy are not installed in this sandbox.
        eval_policy and config only need these names at import time, so stub
        the hardware packages with a meta-path finder returning Mock modules;
        scipy gets explicit attributes because config uses them directly."""
        import importlib.abc
        import importlib.util
        import sys
        import types

        from unittest.mock import Mock

        class _StubLoader(importlib.abc.Loader):
            def create_module(self, spec):
                module = types.ModuleType(spec.name)
                module.__path__ = []
                module.__getattr__ = lambda name: Mock()
                return module

            def exec_module(self, module):
                pass

        class _StubFinder(importlib.abc.MetaPathFinder):
            _PREFIXES = (
                "airo_robots",
                "airo_camera_toolkit",
                "airo_spatial_algebra",
                "pyrealsense2",
            )

            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] not in self._PREFIXES:
                    return None
                return importlib.util.spec_from_loader(fullname, _StubLoader())

        if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _StubFinder())
        for name in (
            "scipy",
            "scipy.spatial",
            "scipy.spatial.transform",
            "scipy.spatial.distance",
        ):
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module
        sys.modules["scipy.spatial.transform"].Rotation = Mock
        sys.modules["scipy.spatial.distance"].cdist = Mock

    def test_eval_policy_recognizes_wrm_claude(self) -> None:
        self._stub_hardware_modules()
        import eval_config
        import eval_policy

        self.assertIn("WRM_claude", eval_policy.SUPPORTED_POLICY_VARIANTS)
        self.assertIn("WRM_claude", eval_policy.BEAVER_POLICY_VARIANTS)
        self.assertEqual(
            eval_policy.EXPECTED_CHECKPOINT_KINDS["WRM_claude"], "WRM_claude"
        )
        self.assertIn("WRM_claude", eval_config.EvalConfig().POLICIES)
        self.assertEqual(eval_config.EvalConfig().PREDICTION_STEPS["WRM_claude"], 16)
        self.assertEqual(eval_config.EvalConfig().ACTION_STEPS["WRM_claude"], 8)

    def test_mock_observation_end_to_end(self) -> None:
        policy = build_policy(tiny_claude_config(), claude_normalizer())
        # Frame shape the evaluator produces per tick, as _ensure_observation_batch
        # receives it (image without batch dim, state 1-D).
        observation = {
            "image": torch.rand(3, 64, 64),
            "state": torch.randn(7),
            "beaver_distance": torch.full((9, 4, 4), 250.0),
            "beaver_status": torch.full((9, 4, 4), 5.0),
            "beaver_present": torch.ones(9),
        }
        action = policy.select_action(observation)
        self.assertEqual(action.shape, (1, 7))
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(0.0 <= policy.last_grasp_probability <= 1.0)
        self.assertGreaterEqual(policy.last_contact_valid_fraction, 0.0)
        self.assertLessEqual(policy.last_contact_valid_fraction, 1.0)
        self.assertGreaterEqual(policy.last_contact_feature_std, 0.0)
        self.assertGreaterEqual(policy.last_delta_max, 0.0)
        self.assertGreaterEqual(policy.last_accumulated_delta_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
