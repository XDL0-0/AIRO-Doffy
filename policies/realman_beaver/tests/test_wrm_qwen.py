"""Focused tests for the WRM_qwen relative-action temporal Beaver policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import Tensor

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import (
    QWEN_BEAVER_VARIANT,
    ModelConfig,
    RealmanBeaverConfig,
    TrainingConfig,
)
from policies.realman_beaver.dataset import (
    ObservationNormalizer,
    RealmanPolicyDataset,
    fit_delta_action_statistics,
)
from policies.realman_beaver.modeling import QwenBeaverDPPolicy, build_policy

DATASET_ROOT_CANDIDATES = (
    Path("/home/yuyuan/AIRO-Doffy-agent-qwen/datasets/"
         "WRM_grasp_cylinder_different_sizes_lero_tightness"),
    Path("/home/yuyuan/AIRO-Doffy/datasets/"
         "WRM_grasp_cylinder_different_sizes_lero_tightness"),
)
DATASET_ROOT = next(
    (path for path in DATASET_ROOT_CANDIDATES if (path / "meta" / "info.json").exists()),
    None,
)
DATASET_REPO_ID = "WRM_grasp_cylinder_different_sizes_lero_tightness"


def tiny_qwen_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant="WRM_qwen",
            n_obs_steps=2,
            horizon=8,
            n_action_steps=2,
            beaver_history_steps=4,
            qwen_joint_history_steps=3,
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


def qwen_normalizer() -> ObservationNormalizer:
    return ObservationNormalizer.identity(
        temporal_beaver_statistics={
            "p5": torch.full((4,), 100.0),
            "p95": torch.full((4,), 1100.0),
            "median": torch.full((4,), 600.0),
            "sensor_indices": torch.tensor((1, 2, 5, 6)),
        },
        delta_action_statistics={
            "offset": torch.zeros(7),
            "scale": torch.ones(7),
        },
    )


def training_batch(batch_size: int = 2) -> dict[str, Tensor]:
    horizon, history = 8, 4
    return {
        "image": torch.rand(batch_size, 2, 3, 64, 64),
        "state": torch.randn(batch_size, 4, 7),
        "action": torch.randn(batch_size, horizon, 7),
        "action_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool),
        "beaver_history_distance": (
            torch.rand(batch_size, 2, history, 9, 4, 4) * 1000.0 + 100.0
        ),
        "beaver_history_status": torch.full((batch_size, 2, history, 9, 4, 4), 5.0),
        "beaver_history_present": torch.ones(batch_size, 2, history, 9),
        "grasp_state": torch.tensor(((0.0, 0.0), (0.0, 1.0)))[:batch_size],
    }


def online_observation(state: Tensor | None = None) -> dict[str, Tensor]:
    observation = {
        "image": torch.rand(1, 3, 64, 64),
        "state": torch.randn(7) if state is None else state,
        "beaver_distance": torch.full((1, 9, 4, 4), 250.0),
        "beaver_status": torch.full((1, 9, 4, 4), 5.0),
        "beaver_present": torch.ones(1, 9),
    }
    return observation


class QwenConfigTest(unittest.TestCase):
    def test_variant_registered_and_validates(self) -> None:
        from policies.realman_beaver.configuration import SUPPORTED_VARIANTS

        self.assertIn(QWEN_BEAVER_VARIANT, SUPPORTED_VARIANTS)
        tiny_qwen_config()  # raises on any validation regression

    def test_validation_rejects_bad_joint_history(self) -> None:
        for bad in (
            {"qwen_joint_history_steps": 1},
            {"qwen_joint_history_steps": 5, "beaver_history_steps": 4},
        ):
            with self.assertRaisesRegex(ValueError, "joint history|history"):
                config = RealmanBeaverConfig(
                    model=ModelConfig(
                        variant="WRM_qwen",
                        n_obs_steps=2,
                        horizon=8,
                        n_action_steps=2,
                        resize_shape=(64, 64),
                        crop_ratio=1.0,
                        down_dims=(32, 64),
                        n_groups=8,
                        **bad,
                    )
                )
                config.dataset.image_shape = (3, 64, 64)
                config.validate()

    def test_validation_rejects_wrong_feature_dimension(self) -> None:
        config = RealmanBeaverConfig(
            model=ModelConfig(
                variant="WRM_qwen",
                n_obs_steps=2,
                horizon=8,
                n_action_steps=2,
                beaver_temporal_feature_dim=32,
                resize_shape=(64, 64),
                crop_ratio=1.0,
                down_dims=(32, 64),
                n_groups=8,
            )
        )
        config.dataset.image_shape = (3, 64, 64)
        with self.assertRaisesRegex(ValueError, "beaver_temporal_feature_dim"):
            config.validate()

    def test_validation_rejects_wrong_sensor_selection(self) -> None:
        config = RealmanBeaverConfig(
            model=ModelConfig(
                variant="WRM_qwen",
                n_obs_steps=2,
                horizon=8,
                n_action_steps=2,
                beaver_temporal_sensors=("01", "02", "10"),
                resize_shape=(64, 64),
                crop_ratio=1.0,
                down_dims=(32, 64),
                n_groups=8,
            )
        )
        config.dataset.image_shape = (3, 64, 64)
        with self.assertRaisesRegex(ValueError, "four unique names"):
            config.validate()

    def test_full_size_parameter_budget_under_100m(self) -> None:
        config = RealmanBeaverConfig(
            model=ModelConfig(variant="WRM_qwen")
        )
        config.validate()
        policy = build_policy(
            config,
            ObservationNormalizer.identity(
                temporal_beaver_statistics={
                    "p5": torch.full((4,), 100.0),
                    "p95": torch.full((4,), 1100.0),
                    "median": torch.full((4,), 600.0),
                    "sensor_indices": torch.tensor((1, 2, 5, 6)),
                },
                delta_action_statistics={
                    "offset": torch.zeros(7),
                    "scale": torch.full((7,), 0.5),
                },
            ),
        )
        parameters = sum(parameter.numel() for parameter in policy.parameters())
        self.assertLess(parameters, 100_000_000)
        # fp32 deployment weight size stays far below the parameter budget.
        bytes_per_parameter = 4
        self.assertLess(
            sum(
                value.numel() for value in policy.state_dict().values()
            )
            * bytes_per_parameter,
            100_000_000 * bytes_per_parameter,
        )

    def test_dispatch_and_variant_guard(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        self.assertIsInstance(policy, QwenBeaverDPPolicy)
        wrong = tiny_qwen_config()
        wrong.model.variant = "WRM_temporal"
        with self.assertRaisesRegex(ValueError, "requires model.variant"):
            QwenBeaverDPPolicy(wrong, qwen_normalizer())

    def test_constructor_requires_train_split_statistics(self) -> None:
        with self.assertRaisesRegex(ValueError, "per-sensor robust normalization"):
            build_policy(
                tiny_qwen_config(),
                ObservationNormalizer.identity(
                    delta_action_statistics={
                        "offset": torch.zeros(7),
                        "scale": torch.ones(7),
                    }
                ),
            )
        with self.assertRaisesRegex(ValueError, "relative-action"):
            build_policy(
                tiny_qwen_config(),
                ObservationNormalizer.identity(
                    temporal_beaver_statistics={
                        "p5": torch.full((4,), 100.0),
                        "p95": torch.full((4,), 1100.0),
                        "median": torch.full((4,), 600.0),
                        "sensor_indices": torch.tensor((1, 2, 5, 6)),
                    }
                ),
            )
        mismatched = ObservationNormalizer.identity(
            temporal_beaver_statistics={
                "p5": torch.full((4,), 100.0),
                "p95": torch.full((4,), 1100.0),
                "median": torch.full((4,), 600.0),
                "sensor_indices": torch.tensor((0, 1, 2, 3)),
            },
            delta_action_statistics={
                "offset": torch.zeros(7),
                "scale": torch.ones(7),
            },
        )
        with self.assertRaisesRegex(ValueError, "sensor order does not match"):
            build_policy(tiny_qwen_config(), mismatched)


class QwenForwardTest(unittest.TestCase):
    def test_conditioned_state_composition(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        state, phase_logit, beaver_feature, sensor_tokens = policy._state_and_phase(
            batch
        )
        self.assertEqual(state.shape, (2, 2, 86))
        self.assertEqual(phase_logit.shape, (2, 2))
        self.assertEqual(beaver_feature.shape, (2, 2, 64))
        # The first seven columns of each observation time are the
        # normalizer-normalized measured configuration.
        expected_qpos = policy.normalizer.normalize_state(batch["state"][:, -2:])
        torch.testing.assert_close(state[:, :, :7], expected_qpos)
        self.assertGreater(
            sensor_tokens[..., 1, :].abs().sum(), 0
        )  # sensor 02 token participates

    def test_delta_target_uses_current_configuration(self) -> None:
        normalizer = ObservationNormalizer.identity(
            temporal_beaver_statistics={
                "p5": torch.full((4,), 100.0),
                "p95": torch.full((4,), 1100.0),
                "median": torch.full((4,), 600.0),
                "sensor_indices": torch.tensor((1, 2, 5, 6)),
            },
            delta_action_statistics={
                "offset": torch.full((7,), 0.25),
                "scale": torch.full((7,), 0.5),
            },
        )
        policy = build_policy(tiny_qwen_config(), normalizer)
        batch = training_batch()
        target = policy._delta_target(batch)
        expected = (
            batch["action"] - batch["state"][:, -1:, :] - torch.full((7,), 0.25)
        ) / torch.full((7,), 0.5)
        torch.testing.assert_close(target, expected)

    def test_loss_and_backward_gradients(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
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
                "delta_action_std",
                "beaver_sensor_01_token_std",
                "beaver_sensor_02_token_std",
                "beaver_sensor_10_token_std",
                "beaver_sensor_11_token_std",
            },
        )
        loss.backward()
        self.assertGreater(policy.beaver_encoder.sensor_embedding.grad.abs().sum(), 0)
        self.assertGreater(policy.grasp_state_head[0].weight.grad.abs().sum(), 0)
        unet_parameter = next(policy.native_policy.diffusion.unet.parameters())
        self.assertIsNotNone(unet_parameter.grad)
        self.assertGreater(unet_parameter.grad.abs().sum(), 0)
        rgb_parameter = next(
            policy.native_policy.diffusion.rgb_encoder.parameters()
        )
        self.assertIsNotNone(rgb_parameter.grad)

    def test_missing_grasp_label_raises(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        del batch["grasp_state"]
        with self.assertRaisesRegex(KeyError, "grasp_state"):
            policy.compute_loss(batch)

    def test_bad_state_window_raises(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        batch["state"] = batch["state"][:, -2:]  # only two rows, not four
        with self.assertRaisesRegex(ValueError, "four-row joint history"):
            policy.compute_loss(batch)


class QwenModalitySensitivityTest(unittest.TestCase):
    def _loss(self, policy, batch) -> Tensor:
        loss, _ = policy.compute_loss(batch)
        return loss.detach()

    def test_rgb_modality_is_used(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        baseline = self._loss(policy, batch)
        zeroed = dict(batch)
        zeroed["image"] = torch.zeros_like(batch["image"])
        self.assertFalse(torch.allclose(baseline, self._loss(policy, zeroed)))
        gradient_image = batch["image"].clone().requires_grad_(True)
        traced = dict(batch)
        traced["image"] = gradient_image
        loss, _ = policy.compute_loss(traced)
        loss.backward()
        self.assertGreater(gradient_image.grad.abs().sum(), 0)

    def test_joint_modality_is_used(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        baseline = self._loss(policy, batch)
        shifted = dict(batch)
        shifted["state"] = batch["state"] + 0.1
        self.assertFalse(torch.allclose(baseline, self._loss(policy, shifted)))
        gradient_state = batch["state"].clone().requires_grad_(True)
        traced = dict(batch)
        traced["state"] = gradient_state
        loss, _ = policy.compute_loss(traced)
        loss.backward()
        self.assertGreater(gradient_state.grad.abs().sum(), 0)

    def test_beaver_modality_is_used(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        baseline = self._loss(policy, batch)
        perturbed = dict(batch)
        perturbed["beaver_history_distance"] = (
            batch["beaver_history_distance"].flip(2) * 1.2 + 50.0
        )
        self.assertFalse(torch.allclose(baseline, self._loss(policy, perturbed)))

    def test_zeroing_and_shuffling_ablation_changes_output(self) -> None:
        torch.manual_seed(0)
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        policy.eval()
        batch = training_batch()
        with torch.no_grad():
            baseline = policy.predict_action_chunk(batch)
            # Zeroed Beaver readings fall back to the neutral median fill.
            zeroed = dict(batch)
            zeroed["beaver_history_distance"] = torch.zeros_like(
                batch["beaver_history_distance"]
            )
            zeroed["beaver_history_status"] = torch.full_like(
                batch["beaver_history_status"], 255.0
            )
            zeroed_beaver = policy.predict_action_chunk(zeroed)
            # Shuffled joint history across the batch.
            shuffled = dict(batch)
            shuffled["state"] = batch["state"].flip(0)
            shuffled["beaver_history_distance"] = batch[
                "beaver_history_distance"
            ].flip(0)
            shuffled["beaver_history_status"] = batch[
                "beaver_history_status"
            ].flip(0)
            shuffled["beaver_history_present"] = batch[
                "beaver_history_present"
            ].flip(0)
            shuffled["image"] = batch["image"].flip(0)
            shuffled_actions = policy.predict_action_chunk(shuffled)
        self.assertFalse(torch.allclose(baseline, zeroed_beaver))
        self.assertFalse(
            torch.allclose(
                baseline.flip(0), shuffled_actions
            )
        )


class QwenBeaverMaskingTest(unittest.TestCase):
    def _encoder(self):
        from policies.realman_beaver.modules import TemporalBeaverEncoder

        encoder = TemporalBeaverEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=4
        )
        encoder.set_normalization_statistics(
            p5=torch.full((4,), 100.0),
            p95=torch.full((4,), 1100.0),
            median=torch.full((4,), 600.0),
        )
        return encoder

    def test_invalid_pixels_are_masked_and_forward_filled(self) -> None:
        encoder = self._encoder()
        distance = torch.full((1, 4, 9, 4, 4), 600.0)
        status = torch.full((1, 4, 9, 4, 4), 5.0)
        present = torch.ones(1, 4, 9)
        # Sensor 01 pixel (0, 0): invalid (255) first frame, valid later.
        distance[0, 0, 1, 0, 0] = 0.0
        status[0, 0, 1, 0, 0] = 255.0
        distance[0, 1, 1, 0, 0] = 300.0
        distance[0, 2, 1, 0, 0] = 900.0
        distance[0, 3, 1, 0, 0] = 900.0
        feature, _ = encoder(
            distance, status, present, return_intermediates=True
        )
        self.assertTrue(torch.isfinite(feature).all())
        # A fully valid history and one with an invalid frame must differ.
        clean = torch.full((1, 4, 9, 4, 4), 600.0)
        clean_status = torch.full((1, 4, 9, 4, 4), 5.0)
        clean_feature = encoder(clean, clean_status, present)
        self.assertFalse(torch.allclose(feature, clean_feature))

    def test_all_invalid_history_falls_back_to_median(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        batch["beaver_history_distance"] = torch.zeros_like(
            batch["beaver_history_distance"]
        )
        batch["beaver_history_status"] = torch.full_like(
            batch["beaver_history_status"], 255.0
        )
        state, _, beaver_feature, _ = policy._state_and_phase(batch)
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(beaver_feature).all())
        self.assertGreater(beaver_feature.abs().sum(), 0)

    def test_sensor_presence_zeroed_when_disconnected(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        connected = policy._state_and_phase(batch)[2]
        batch["beaver_history_present"] = torch.zeros_like(
            batch["beaver_history_present"]
        )
        disconnected = policy._state_and_phase(batch)[2]
        self.assertFalse(torch.allclose(connected, disconnected))


class QwenOnlineHistoryTest(unittest.TestCase):
    def test_early_ticks_clamp_to_earliest_frame(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = online_observation(state=torch.randn(1, 7))
        first = policy._append_online_history(batch)
        # All four joint rows collapse to the first frame.
        self.assertEqual(first["state"].shape, (1, 4, 7))
        for position in range(4):
            torch.testing.assert_close(
                first["state"][:, position], first["state"][:, -1]
            )
        beaver_window = first["beaver_history_distance"]
        self.assertEqual(beaver_window.shape, (1, 4, 9, 4, 4))
        for position in range(4):
            torch.testing.assert_close(
                beaver_window[:, position], batch["beaver_distance"]
            )

    def test_joint_window_lags_after_history_fills(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        lag = 3  # qwen_joint_history_steps in the tiny config
        built = None
        for tick in range(lag + 2):
            # select_action's _ensure_observation_batch batch-dimensions the
            # observation before _append_online_history sees it.
            state = torch.full((1, 7), float(tick))
            batch = online_observation(state=state)
            if tick > 0:
                batch["beaver_distance"] = torch.full((1, 9, 4, 4), 100.0 + tick)
            built = policy._append_online_history(batch)
        self.assertEqual(len(policy._online_history), lag + 2)
        expected_rows = [0.0, float(lag - 1), float(lag), float(lag + 1)]
        torch.testing.assert_close(
            built["state"][0, :, 0], torch.tensor(expected_rows)
        )

    def test_reset_clears_history_and_anchor(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        for _ in range(4):
            policy.select_action(online_observation())
        self.assertGreater(len(policy._online_history), 0)
        self.assertIsNotNone(policy._action_anchor)
        policy.reset()
        self.assertEqual(len(policy._online_history), 0)
        self.assertIsNone(policy._action_anchor)
        self.assertTrue(policy.last_replanned)
        first = policy._append_online_history(online_observation())
        torch.testing.assert_close(
            first["state"][:, 0], first["state"][:, -1]
        )

    def test_batch_size_change_resets(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        policy.select_action(online_observation())
        self.assertGreater(len(policy._online_history), 0)
        wide = online_observation()
        wide["state"] = torch.randn(2, 7)
        wide["image"] = torch.rand(2, 3, 64, 64)
        wide["beaver_distance"] = torch.full((2, 9, 4, 4), 250.0)
        wide["beaver_status"] = torch.full((2, 9, 4, 4), 5.0)
        wide["beaver_present"] = torch.ones(2, 9)
        policy.select_action(wide)
        self.assertEqual(len(policy._online_history), 1)


class QwenActionSelectionTest(unittest.TestCase):
    def test_chunk_shape_and_finite_actions(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        batch = training_batch()
        actions = policy.predict_action_chunk(batch)
        self.assertEqual(actions.shape, (2, 2, 7))
        self.assertTrue(torch.isfinite(actions).all())

    def test_replan_cadence_follows_action_steps(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        flags = []
        state = torch.randn(7)
        for _ in range(5):
            state = policy.select_action(online_observation(state=state)).squeeze(0)
            flags.append(policy.last_replanned)
        self.assertEqual(flags, [True, False, True, False, True])

    def test_recentering_anchor_is_the_replan_configuration(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        anchor_state = torch.full((1, 7), 0.3)
        first = policy.select_action(
            online_observation(state=anchor_state)
        )
        self.assertTrue(policy.last_replanned)
        torch.testing.assert_close(policy._action_anchor, anchor_state)
        queued = [
            value.clone()
            for value in list(policy.native_policy._queues["action"])
        ]
        self.assertEqual(len(queued), 1)
        # The next executed action re-anchors on the *replan* configuration,
        # not on wherever the arm has drifted in the meantime.
        second = policy.select_action(
            online_observation(state=torch.full((1, 7), 0.9))
        )
        self.assertFalse(policy.last_replanned)
        expected = anchor_state + queued[0].clamp(-1.0, 1.0)
        torch.testing.assert_close(second, expected)
        # Identity delta statistics: the executed action minus the anchor is
        # exactly the (clamped) normalized delta from the chunk.
        residual = first - anchor_state
        torch.testing.assert_close(residual, residual.clamp(-1.0, 1.0))

    def test_select_action_shape_after_reset(self) -> None:
        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        policy.reset()
        action = policy.select_action(online_observation())
        self.assertEqual(action.shape, (1, 7))
        self.assertTrue(torch.isfinite(action).all())


class QwenCheckpointTest(unittest.TestCase):
    def _save(self, path: Path, policy, config, ema_state) -> None:
        torch.save(
            {
                "kind": QWEN_BEAVER_VARIANT,
                "config": config.to_dict(),
                "model": policy.state_dict(),
                "ema": ema_state,
                "epoch": 0,
                "global_step": 1,
                "metrics": {"loss": 1.0},
            },
            path,
        )

    def test_save_load_roundtrip_with_ema_and_buffers(self) -> None:
        from policies.realman_beaver.train import ExponentialMovingAverage

        config = tiny_qwen_config()
        policy = build_policy(config, qwen_normalizer())
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        loss, _ = policy.compute_loss(training_batch())
        loss.backward()
        optimizer.step()
        ema = ExponentialMovingAverage(policy, 0.999)
        ema.update(policy)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "wrm_qwen.pt"
            self._save(checkpoint_path, policy, config, ema.state_dict())
            restored = load_policy(checkpoint_path, use_ema=True)

        self.assertIsInstance(restored, QwenBeaverDPPolicy)
        self.assertTrue(restored.beaver_encoder.normalization_fitted)
        torch.testing.assert_close(
            restored.normalizer.beaver_temporal_p5,
            policy.normalizer.beaver_temporal_p5,
        )
        torch.testing.assert_close(
            restored.normalizer.delta_action_offset,
            policy.normalizer.delta_action_offset,
        )
        parameters = dict(restored.named_parameters())
        name = next(iter(ema.state_dict()))
        torch.testing.assert_close(
            parameters[name].detach(), ema.state_dict()[name]
        )

    def test_loader_rejects_missing_delta_action_buffers(self) -> None:
        config = tiny_qwen_config()
        policy = build_policy(config, qwen_normalizer())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "wrm_qwen_missing.pt"
            state_dict = dict(policy.state_dict())
            del state_dict["normalizer.delta_action_offset"]
            del state_dict["normalizer.delta_action_scale"]
            torch.save(
                {
                    "kind": QWEN_BEAVER_VARIANT,
                    "config": config.to_dict(),
                    "model": state_dict,
                    "ema": {},
                },
                checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "normalization buffer"):
                load_policy(checkpoint_path)


def _eval_policy_importable() -> bool:
    try:
        import eval_policy  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipIf(
    not _eval_policy_importable(),
    "eval_policy hardware dependencies are not importable here",
)
class QwenEvalIntegrationTest(unittest.TestCase):
    def test_eval_variant_recognition(self) -> None:
        from eval_policy import (
            BEAVER_POLICY_VARIANTS,
            EXPECTED_CHECKPOINT_KINDS,
            SUPPORTED_POLICY_VARIANTS,
            policy_needs_beaver,
            validate_deployable_checkpoint,
        )

        self.assertIn("WRM_qwen", SUPPORTED_POLICY_VARIANTS)
        self.assertIn("WRM_qwen", BEAVER_POLICY_VARIANTS)
        self.assertEqual(EXPECTED_CHECKPOINT_KINDS["WRM_qwen"], "WRM_qwen")
        self.assertTrue(policy_needs_beaver("WRM_qwen"))
        variant = validate_deployable_checkpoint(
            {"kind": "WRM_qwen", "variant": "WRM_qwen"}
        )
        self.assertEqual(variant, "WRM_qwen")
        with self.assertRaisesRegex(ValueError, "does not match variant"):
            validate_deployable_checkpoint(
                {"kind": "WRM_temporal", "variant": "WRM_qwen"}
            )

    def test_eval_config_registers_window(self) -> None:
        from eval_config import EvalConfig

        eval_config = EvalConfig()
        self.assertEqual(eval_config.PREDICTION_STEPS["WRM_qwen"], 16)
        self.assertEqual(eval_config.ACTION_STEPS["WRM_qwen"], 8)
        self.assertIn("WRM_qwen", eval_config.POLICIES)

    def test_policy_step_window(self) -> None:
        from eval_policy import policy_step_window

        policy = build_policy(tiny_qwen_config(), qwen_normalizer())
        self.assertEqual(policy_step_window(policy), (8, 2))
        full = build_policy(
            RealmanBeaverConfig(model=ModelConfig(variant="WRM_qwen")),
            ObservationNormalizer.identity(
                temporal_beaver_statistics={
                    "p5": torch.full((4,), 100.0),
                    "p95": torch.full((4,), 1100.0),
                    "median": torch.full((4,), 600.0),
                    "sensor_indices": torch.tensor((1, 2, 5, 6)),
                },
                delta_action_statistics={
                    "offset": torch.zeros(7),
                    "scale": torch.full((7,), 0.5),
                },
            ),
        )
        self.assertEqual(policy_step_window(full), (16, 8))


@unittest.skipIf(DATASET_ROOT is None, "tightness-labelled dataset not available")
class QwenDatasetTest(unittest.TestCase):
    def _config(self) -> RealmanBeaverConfig:
        config = tiny_qwen_config()
        config.dataset.root = str(DATASET_ROOT)
        config.dataset.repo_id = DATASET_REPO_ID
        return config

    def test_dataset_builds_four_row_state_and_history_windows(self) -> None:
        dataset = RealmanPolicyDataset(self._config(), episodes=[0, 1])
        sample = dataset[0]
        self.assertEqual(sample["state"].shape, (4, 7))
        self.assertEqual(sample["image"].shape, (2, 3, 480, 640))
        self.assertEqual(sample["action"].shape, (8, 7))
        self.assertEqual(sample["action_is_pad"].shape, (8,))
        self.assertEqual(sample["beaver_history_distance"].shape, (2, 4, 9, 4, 4))
        self.assertEqual(sample["beaver_history_status"].shape, (2, 4, 9, 4, 4))
        self.assertEqual(sample["beaver_history_present"].shape, (2, 4, 9))
        self.assertEqual(sample["grasp_state"].shape, (2,))
        # Frame 0 clamps every delta row to the earliest frame.
        torch.testing.assert_close(sample["state"][0], sample["state"][-1])

    def test_fit_delta_action_statistics_train_only(self) -> None:
        config = self._config()
        statistics = fit_delta_action_statistics(config, [0, 1])
        minimum = statistics["min"]
        maximum = statistics["max"]
        self.assertEqual(tuple(minimum.shape), (7,))
        self.assertTrue(torch.isfinite(minimum).all())
        self.assertTrue(torch.isfinite(maximum).all())
        self.assertTrue((maximum >= minimum).all())
        with self.assertRaisesRegex(ValueError, "leakage"):
            fit_delta_action_statistics(config, None)

    @unittest.skipUnless(
        torch.cuda.is_available() or True, "CPU smoke is allowed"
    )
    def test_smoke_training_milestone_and_reload(self) -> None:
        from policies.realman_beaver.train import train

        config = self._config()
        config.dataset.val_episodes = (2,)
        config.training = TrainingConfig(
            output_dir="unused",  # replaced below
            device="cpu",
            batch_size=16,
            num_workers=0,
            epochs=1,
            max_steps=3,
            checkpoint_every_steps=2,
            amp=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            config.training.output_dir = str(Path(directory) / "smoke")
            last_checkpoint = train(config)
            self.assertTrue(last_checkpoint.is_file())
            milestone = (
                Path(directory) / "smoke" / "WRM_qwen_step_000002.pt"
            )
            self.assertTrue(milestone.is_file())
            reloaded = load_policy(milestone, device="cpu")
            self.assertIsInstance(reloaded, QwenBeaverDPPolicy)
            action = reloaded.select_action(online_observation())
            self.assertEqual(action.shape, (1, 7))
            self.assertTrue(torch.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
