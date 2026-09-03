from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.codex_policy import (
    CodexTemporalBeaverEncoder,
    WRMCodexPolicy,
)
from policies.realman_beaver.configuration import (
    CODEX_BEAVER_VARIANT,
    ModelConfig,
    RealmanBeaverConfig,
    load_config,
)
from policies.realman_beaver.dataset import ObservationNormalizer, RealmanPolicyDataset
from policies.realman_beaver.eval_registry import (
    EXPECTED_CHECKPOINT_KINDS,
    policy_needs_beaver,
    validate_deployable_checkpoint,
)
from policies.realman_beaver.modeling import build_policy
from policies.realman_beaver.offline_eval_codex import ABLATIONS
from policies.realman_beaver.train import ExponentialMovingAverage


def codex_statistics(
    sensor_indices: tuple[int, ...] = (1, 2, 5, 6),
) -> dict[str, torch.Tensor]:
    sensor_count = len(sensor_indices)
    return {
        "p5": torch.linspace(100.0, 180.0, sensor_count),
        "p95": torch.linspace(900.0, 980.0, sensor_count),
        "median": torch.linspace(400.0, 480.0, sensor_count),
        "sensor_indices": torch.tensor(sensor_indices),
    }


def tiny_codex_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant=CODEX_BEAVER_VARIANT,
            codex_beaver_history_steps=4,
            codex_token_dim=32,
            codex_vision_width=8,
            codex_contact_hidden_dim=32,
            codex_fusion_layers=1,
            codex_sensor_layers=1,
            codex_decoder_layers=1,
            codex_attention_heads=4,
            codex_dropout=0.0,
            codex_plan_ensemble=2,
            n_obs_steps=2,
            horizon=8,
            n_action_steps=2,
            resize_shape=(32, 32),
            down_dims=(32, 64),
            num_train_timesteps=8,
            num_inference_steps=2,
        )
    )
    config.dataset.image_shape = (3, 32, 32)
    config.validate()
    return config


def make_policy() -> WRMCodexPolicy:
    return build_policy(
        tiny_codex_config(),
        ObservationNormalizer.identity(
            temporal_beaver_statistics=codex_statistics()
        ),
    )


def make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "image": torch.rand(batch_size, 2, 3, 32, 32),
        "state": torch.randn(batch_size, 2, 7) * 0.2,
        "action": torch.randn(batch_size, 8, 7) * 0.2,
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
        "beaver_history_distance": torch.rand(
            batch_size, 2, 4, 9, 4, 4
        )
        * 500.0
        + 150.0,
        "beaver_history_status": torch.full(
            (batch_size, 2, 4, 9, 4, 4), 5.0
        ),
        "beaver_history_present": torch.ones(batch_size, 2, 4, 9),
    }


def make_observation(value: float = 250.0) -> dict[str, torch.Tensor]:
    return {
        "image": torch.rand(3, 32, 32),
        "state": torch.randn(7) * 0.1,
        "beaver_distance": torch.full((9, 4, 4), value),
        "beaver_status": torch.full((9, 4, 4), 5.0),
        "beaver_present": torch.ones(9),
    }


class _FakeCodexLeRobotDataset:
    def __init__(
        self, *, delta_timestamps: dict[str, list[float]], **_: object
    ) -> None:
        self.delta_timestamps = delta_timestamps
        self.features = {
            "observation.images.camera_0": {"shape": (3, 32, 32)},
            "observation.state": {"shape": (7,)},
            "action": {"shape": (7,)},
            "observation.beaver.distance_mm": {"shape": (9, 4, 4)},
            "observation.beaver.present": {"shape": (9,)},
            "observation.beaver.target_status": {"shape": (9, 4, 4)},
        }
        obs_steps = len(delta_timestamps["observation.state"])
        action_steps = len(delta_timestamps["action"])
        beaver_steps = len(
            delta_timestamps["observation.beaver.distance_mm"]
        )
        self.item = {
            "observation.images.camera_0": torch.rand(obs_steps, 3, 32, 32),
            "observation.state": torch.rand(obs_steps, 7),
            "action": torch.rand(action_steps, 7),
            "action_is_pad": torch.zeros(action_steps, dtype=torch.bool),
            "observation.beaver.distance_mm": torch.full(
                (beaver_steps, 9, 4, 4), 250.0
            ),
            "observation.beaver.present": torch.ones(beaver_steps, 9),
            "observation.beaver.target_status": torch.full(
                (beaver_steps, 9, 4, 4), 5.0
            ),
            "episode_index": torch.tensor(7),
            "frame_index": torch.tensor(11),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        return self.item


class CodexConfigurationAndDatasetTest(unittest.TestCase):
    def test_configs_validate_split_and_parameter_limit(self) -> None:
        selection = load_config(
            "policies/realman_beaver/configs/WRM_codex_selection.yaml"
        )
        final = load_config("policies/realman_beaver/configs/WRM_codex_all125.yaml")
        self.assertEqual(selection.model.variant, CODEX_BEAVER_VARIANT)
        self.assertEqual(len(selection.dataset.val_episodes or ()), 25)
        self.assertEqual(final.dataset.val_fraction, 0.0)
        policy = build_policy(
            selection,
            ObservationNormalizer.identity(
                temporal_beaver_statistics=codex_statistics()
            ),
        )
        parameters = sum(
            parameter.numel()
            for parameter in policy.parameters()
            if parameter.requires_grad
        )
        self.assertLess(parameters, 100_000_000)
        self.assertGreater(parameters, 1_000_000)

        invalid = tiny_codex_config()
        invalid.model.codex_token_dim = 30
        with self.assertRaisesRegex(ValueError, "divisible"):
            invalid.validate()

    def test_dataset_builds_causal_history_without_tightness_input(self) -> None:
        config = tiny_codex_config()
        with patch(
            "policies.realman_beaver.dataset.LeRobotDataset",
            _FakeCodexLeRobotDataset,
        ):
            dataset = RealmanPolicyDataset(config, [7], stage="policy")
        sample = dataset[0]
        self.assertEqual(sample["beaver_history_distance"].shape, (2, 4, 9, 4, 4))
        self.assertEqual(sample["beaver_history_status"].shape, (2, 4, 9, 4, 4))
        self.assertNotIn("grasp_state", sample)
        self.assertEqual(sample["episode_index"].item(), 7)
        self.assertEqual(sample["frame_index"].item(), 11)
        self.assertEqual(
            dataset.dataset.delta_timestamps[config.dataset.beaver_distance_key],
            [step / config.dataset.fps for step in range(-4, 1)],
        )


class CodexMaskingAndModalityTest(unittest.TestCase):
    def test_per_pixel_status_mask_and_all_invalid_fallback(self) -> None:
        encoder = CodexTemporalBeaverEncoder(
            n_sensors=9,
            sensor_indices=tuple(range(9)),
            history_steps=4,
            valid_statuses=(5, 9),
            hidden_dim=32,
            token_dim=32,
            sensor_layers=1,
            attention_heads=4,
            dropout=0.0,
        ).eval()
        statistics = codex_statistics(tuple(range(9)))
        encoder.set_normalization_statistics(
            p5=statistics["p5"],
            p95=statistics["p95"],
            median=statistics["median"],
        )
        distance = torch.full((1, 4, 9, 4, 4), 250.0)
        status = torch.full_like(distance, 255.0)
        present = torch.ones(1, 4, 9)
        invalid_feature, invalid_middle = encoder(
            distance, status, present, return_intermediates=True
        )
        torch.testing.assert_close(
            invalid_feature, torch.zeros_like(invalid_feature)
        )
        self.assertFalse(invalid_middle["valid"].any())

        changed_invalid = distance.clone()
        changed_invalid[..., 0, 0, 0] = 800.0
        torch.testing.assert_close(
            encoder(changed_invalid, status, present), invalid_feature
        )
        valid_status = status.clone()
        valid_status[..., 0, 0, 0] = 5.0
        valid_feature = encoder(changed_invalid, valid_status, present)
        self.assertFalse(torch.allclose(valid_feature, invalid_feature))
        absent = present.clone()
        absent[..., 0] = 0.0
        torch.testing.assert_close(
            encoder(changed_invalid, valid_status, absent), invalid_feature
        )

    def test_loss_backward_and_all_three_modality_gradients(self) -> None:
        policy = make_policy().train()
        batch = make_batch()
        batch["image"].requires_grad_()
        batch["state"].requires_grad_()
        batch["beaver_history_distance"].requires_grad_()
        loss, metrics = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(metrics),
            {
                "loss",
                "action_loss",
                "velocity_loss",
                "activity_loss",
                "activity_mae",
                "beaver_valid_fraction",
            },
        )
        loss.backward()
        self.assertGreater(batch["image"].grad.abs().sum().item(), 0.0)
        self.assertGreater(batch["state"].grad.abs().sum().item(), 0.0)
        self.assertGreater(
            batch["beaver_history_distance"].grad.abs().sum().item(), 0.0
        )
        self.assertGreater(
            policy.vision_encoder.backbone[0][0].weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(policy.joint_encoder[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            policy.beaver_encoder.frame_mlp[0].weight.grad.abs().sum().item(), 0.0
        )

    def test_offline_zero_and_shuffle_ablation_paths(self) -> None:
        policy = make_policy().eval()
        batch = make_batch()
        complete = policy.predict_action_chunk(batch)
        self.assertEqual(
            set(ABLATIONS),
            {
                "complete",
                "image_zero",
                "image_shuffle",
                "joint_zero",
                "joint_shuffle",
                "beaver_zero",
                "beaver_shuffle",
            },
        )
        for modality in ("image", "joint", "beaver"):
            zeroed = policy.predict_action_chunk(
                batch, ablations={modality: "zero"}
            )
            shuffled = policy.predict_action_chunk(
                batch, ablations={modality: "shuffle"}
            )
            self.assertFalse(torch.allclose(complete, zeroed))
            self.assertFalse(torch.allclose(complete, shuffled))


class CodexPolicyLifecycleTest(unittest.TestCase):
    def test_online_chunk_shape_ensemble_and_reset_isolation(self) -> None:
        policy = make_policy().eval()
        actions = []
        replans = []
        for value in (250.0, 260.0, 270.0, 280.0, 290.0):
            action = policy.select_action(make_observation(value))
            actions.append(action)
            replans.append(policy.last_replanned)
            self.assertEqual(action.shape, (1, 7))
            self.assertTrue(torch.isfinite(action).all())
        self.assertEqual(replans, [True, False, True, False, True])
        self.assertEqual(policy.last_plan_contributors, 2)
        self.assertLessEqual(len(policy._plans), 2)
        self.assertGreater(len(policy._online_history), 1)

        policy.reset()
        self.assertEqual(len(policy._plans), 0)
        self.assertEqual(len(policy._online_history), 0)
        policy.select_action(make_observation(700.0))
        self.assertEqual(len(policy._online_history), 1)
        history = policy._append_online_frame(
            policy._ensure_observation_batch(make_observation(700.0))
        )
        self.assertTrue(
            torch.all(history["beaver_history_distance"] == 700.0)
        )

    def test_checkpoint_round_trip_restores_ema_and_preprocessing(self) -> None:
        policy = make_policy()
        ema = ExponentialMovingAverage(policy, decay=0.5)
        first_name, first_parameter = next(iter(policy.named_parameters()))
        with torch.no_grad():
            first_parameter.add_(0.25)
        ema.update(policy)
        expected_ema = ema.state_dict()[first_name].clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "WRM_codex.pt"
            torch.save(
                {
                    "kind": CODEX_BEAVER_VARIANT,
                    "config": policy.config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": ema.state_dict(),
                    "epoch": 1,
                    "global_step": 2,
                    "metrics": {"loss": 1.0},
                },
                checkpoint,
            )
            restored = load_policy(checkpoint, use_ema=True)
        self.assertIsInstance(restored, WRMCodexPolicy)
        torch.testing.assert_close(
            dict(restored.named_parameters())[first_name], expected_ema
        )
        torch.testing.assert_close(
            restored.normalizer.beaver_temporal_p5,
            policy.normalizer.beaver_temporal_p5,
        )
        self.assertTrue(restored.beaver_encoder.normalization_fitted)

    def test_eval_registry_and_cpu_smoke_optimizer_step(self) -> None:
        self.assertEqual(
            EXPECTED_CHECKPOINT_KINDS[CODEX_BEAVER_VARIANT], CODEX_BEAVER_VARIANT
        )
        self.assertTrue(policy_needs_beaver(CODEX_BEAVER_VARIANT))
        self.assertEqual(
            validate_deployable_checkpoint(
                {"variant": CODEX_BEAVER_VARIANT, "kind": CODEX_BEAVER_VARIANT}
            ),
            CODEX_BEAVER_VARIANT,
        )

        policy = make_policy().train()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        before = policy.residual_head.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy.compute_loss(make_batch())
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, policy.residual_head.weight))


if __name__ == "__main__":
    unittest.main()
