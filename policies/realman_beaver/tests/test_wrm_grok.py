from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from eval_policy import (
    EXPECTED_CHECKPOINT_KINDS,
    policy_needs_beaver,
    validate_deployable_checkpoint,
)
from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import (
    GROK_BEAVER_VARIANT,
    ModelConfig,
    RealmanBeaverConfig,
    load_config,
)
from policies.realman_beaver.dataset import ObservationNormalizer, RealmanPolicyDataset
from policies.realman_beaver.metrics_wrm_grok import (
    action_smoothness,
    chunk_boundary_discontinuity,
    count_trainable_parameters,
    deployment_weight_bytes,
    expected_calibration_error,
    phase_precision_recall_f1,
    trajectory_error_rad,
)
from policies.realman_beaver.modeling import build_policy
from policies.realman_beaver.modeling_wrm_grok import WRMGrokPolicy
from policies.realman_beaver.modules.grok_phase_encoder import (
    ENCLOSURE_DIM,
    PHASE_APPROACH,
    PHASE_HOLD,
    PHASE_WRAP,
    GrokPhaseEncoder,
    derive_phase_labels,
    grok_conditioned_state_dim,
)


def grok_statistics() -> dict[str, torch.Tensor]:
    return {
        "p5": torch.tensor((10.0, 20.0, 30.0, 40.0)),
        "p95": torch.tensor((510.0, 520.0, 530.0, 540.0)),
        "median": torch.tensor((210.0, 220.0, 230.0, 240.0)),
        "sensor_indices": torch.tensor((1, 2, 5, 6)),
    }


def tiny_grok_config() -> RealmanBeaverConfig:
    config = RealmanBeaverConfig(
        model=ModelConfig(
            variant="WRM_grok",
            beaver_grok_history_steps=4,
            beaver_grok_motion_delta_steps=3,
            beaver_grok_noise_std_mm=0.0,
            n_obs_steps=2,
            horizon=8,
            n_action_steps=2,
            resize_shape=(64, 64),
            crop_ratio=1.0,
            down_dims=(32, 64),
            n_groups=8,
            flow_time_embed_dim=32,
            flow_num_inference_steps=2,
        )
    )
    config.dataset.image_shape = (3, 64, 64)
    config.validate()
    return config


def make_encoder() -> GrokPhaseEncoder:
    encoder = GrokPhaseEncoder(
        n_sensors=9,
        sensor_indices=(1, 2, 5, 6),
        history_steps=4,
        near_scales_mm=(50.0, 150.0, 300.0),
        wrap_threshold_mm=150.0,
        noise_std_mm=0.0,
    )
    statistics = grok_statistics()
    encoder.set_normalization_statistics(
        p5=statistics["p5"],
        p95=statistics["p95"],
        median=statistics["median"],
    )
    return encoder


class _FakeGrokLeRobotDataset:
    def __init__(self, *, delta_timestamps: dict[str, list[float]], **_: object) -> None:
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


class GrokConfigTest(unittest.TestCase):
    def test_variant_and_parameter_limit(self) -> None:
        config = load_config("policies/realman_beaver/configs/WRM_grok.yaml")
        self.assertEqual(config.model.variant, GROK_BEAVER_VARIANT)
        config.validate()
        all_train = load_config(
            "policies/realman_beaver/configs/WRM_grok_all_train.yaml"
        )
        self.assertIsNone(all_train.dataset.val_episodes)
        self.assertEqual(all_train.dataset.val_fraction, 0.0)
        self.assertEqual(
            tuple(config.dataset.val_episodes),
            (20, 21, 22, 23, 24, 45, 46, 47, 48, 49, 70, 71, 72, 73, 74, 95, 96, 97, 98, 99, 120, 121, 122, 123, 124),
        )

    def test_rejects_unknown_sensors(self) -> None:
        config = tiny_grok_config()
        config.model.beaver_grok_sensors = ("01", "02", "10", "99")
        with self.assertRaisesRegex(ValueError, "absent"):
            config.validate()


class GrokEncoderTest(unittest.TestCase):
    def test_invalid_pixels_are_masked_and_zero_is_not_contact(self) -> None:
        encoder = make_encoder()
        distance = torch.full((1, 4, 9, 4, 4), 200.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 4, 9)
        distance[0, :, 1, 0, 0] = 0.0
        status[0, :, 1, 1, 1] = 255.0
        present[0, :, 0] = 0.0
        _, middle = encoder(distance, status, present, return_intermediates=True)
        valid = middle["valid"][0, :, 0]
        self.assertFalse(bool(valid[:, 0, 0].any()))
        self.assertFalse(bool(valid[:, 1, 1].any()))
        self.assertTrue(bool(valid[:, 2, 2].all()))

    def test_all_invalid_fallback_is_exactly_zero(self) -> None:
        encoder = make_encoder()
        distance = torch.zeros(2, 4, 9, 4, 4)
        status = torch.full_like(distance, 255.0)
        present = torch.zeros(2, 4, 9)
        feature, middle = encoder(
            distance, status, present, return_intermediates=True
        )
        self.assertEqual(feature.shape[-1], 64)
        self.assertEqual(middle["enclosure"].shape[-1], ENCLOSURE_DIM)
        self.assertTrue(torch.equal(feature, torch.zeros_like(feature)))
        self.assertTrue(
            torch.equal(middle["enclosure"], torch.zeros_like(middle["enclosure"]))
        )
        self.assertTrue(torch.equal(middle["contact_quality"], torch.zeros(2)))

    def test_sensor_02_changes_the_representation(self) -> None:
        encoder = make_encoder()
        distance = torch.full((1, 4, 9, 4, 4), 400.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 4, 9)
        baseline, middle = encoder(
            distance, status, present, return_intermediates=True
        )
        changed = distance.clone()
        changed[:, :, 2] = 80.0
        updated, updated_middle = encoder(
            changed, status, present, return_intermediates=True
        )
        self.assertFalse(torch.allclose(baseline, updated))
        self.assertFalse(
            torch.allclose(middle["enclosure"], updated_middle["enclosure"])
        )
        self.assertGreater(float(updated_middle["enclosure"][0, 10]), 0.0)

    def test_refuses_unfitted_statistics(self) -> None:
        encoder = GrokPhaseEncoder(
            n_sensors=9, sensor_indices=(1, 2, 5, 6), history_steps=4, noise_std_mm=0.0
        )
        with self.assertRaisesRegex(RuntimeError, "2550 mm"):
            encoder(
                torch.full((1, 4, 9, 4, 4), 200.0),
                torch.full((1, 4, 9, 4, 4), 5.0),
                torch.ones(1, 4, 9),
            )

    def test_phase_labels_use_beaver_and_tightness(self) -> None:
        distance = torch.full((1, 2, 4, 9, 4, 4), 800.0)
        status = torch.full_like(distance, 5.0)
        present = torch.ones(1, 2, 4, 9)
        tightness = torch.tensor(((0.0, 1.0),))
        approach_hold = derive_phase_labels(
            tightness,
            distance,
            status,
            present,
            sensor_index=torch.tensor((1, 2, 5, 6)),
            valid_statuses=(5, 9),
            wrap_threshold_mm=150.0,
        )
        self.assertEqual(int(approach_hold[0, 0]), PHASE_APPROACH)
        self.assertEqual(int(approach_hold[0, 1]), PHASE_HOLD)
        near = distance.clone()
        near[0, 0, -1, 1] = 40.0
        near[0, 0, -1, 2] = 40.0
        wrap = derive_phase_labels(
            tightness,
            near,
            status,
            present,
            sensor_index=torch.tensor((1, 2, 5, 6)),
            valid_statuses=(5, 9),
            wrap_threshold_mm=150.0,
        )
        self.assertEqual(int(wrap[0, 0]), PHASE_WRAP)


class GrokPolicyTest(unittest.TestCase):
    def _policy(self) -> WRMGrokPolicy:
        return build_policy(
            tiny_grok_config(),
            ObservationNormalizer.identity(
                temporal_beaver_statistics=grok_statistics()
            ),
        )

    @staticmethod
    def _batch() -> dict[str, torch.Tensor]:
        distance = torch.full((2, 2, 4, 9, 4, 4), 400.0)
        distance[1, :, -1, 1] = 40.0
        distance[1, :, -1, 6] = 40.0
        return {
            "image": torch.rand(2, 2, 3, 64, 64),
            "state": torch.randn(2, 2, 7),
            "delta_q": torch.randn(2, 2, 7),
            "action": torch.randn(2, 8, 7),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "beaver_history_distance": distance,
            "beaver_history_status": torch.full((2, 2, 4, 9, 4, 4), 5.0),
            "beaver_history_present": torch.ones(2, 2, 4, 9),
            "grasp_state": torch.tensor(((0.0, 0.0), (0.0, 1.0))),
        }

    def test_requires_train_split_statistics(self) -> None:
        with self.assertRaisesRegex(ValueError, "robust normalization"):
            build_policy(tiny_grok_config(), ObservationNormalizer.identity())

    def test_conditioned_state_fuses_all_three_modalities(self) -> None:
        policy = self._policy()
        self.assertIsInstance(policy, WRMGrokPolicy)
        expected = grok_conditioned_state_dim(policy.config.model)
        self.assertEqual(expected, 98)
        self.assertEqual(policy.flow.config.robot_state_feature.shape, (98,))
        state, phase_logit, beaver, enclosure, quality = policy._state_and_phase(
            self._batch()
        )
        self.assertEqual(tuple(state.shape), (2, 2, 98))
        self.assertEqual(tuple(phase_logit.shape), (2, 2, 3))
        self.assertEqual(tuple(beaver.shape), (2, 2, 64))
        self.assertEqual(tuple(enclosure.shape), (2, 2, 16))
        self.assertEqual(tuple(quality.shape), (2, 2))

    def test_loss_backward_and_parameter_count(self) -> None:
        policy = self._policy()
        trainable = count_trainable_parameters(policy)
        self.assertLess(trainable, 100_000_000)
        self.assertGreater(trainable, 1_000)
        batch = self._batch()
        for key in (
            "image",
            "state",
            "delta_q",
            "beaver_history_distance",
        ):
            batch[key] = batch[key].detach().requires_grad_(True)
        loss, metrics = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        for name in (
            "flow_matching_loss",
            "phase_loss",
            "smooth_loss",
            "hold_loss",
            "tightness_f1",
            "phase_ece",
        ):
            self.assertIn(name, metrics)
        loss.backward()
        self.assertGreater(batch["image"].grad.abs().sum(), 0)
        self.assertGreater(batch["state"].grad.abs().sum(), 0)
        self.assertGreater(batch["beaver_history_distance"].grad.abs().sum(), 0)
        self.assertGreater(policy.beaver_encoder.frame_mlp[0].weight.grad.abs().sum(), 0)
        self.assertGreater(policy.phase_head[0].weight.grad.abs().sum(), 0)
        self.assertTrue(
            any(
                parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
                for parameter in policy.flow.rgb_encoder.parameters()
            )
        )

    def test_modality_zero_and_shuffle_change_actions(self) -> None:
        policy = self._policy().eval()
        batch = self._batch()
        torch.manual_seed(0)
        baseline = policy.predict_action_chunk(batch)
        self.assertEqual(tuple(baseline.shape), (2, 2, 7))
        self.assertTrue(torch.isfinite(baseline).all())

        zero_image = dict(batch)
        zero_image["image"] = torch.zeros_like(batch["image"])
        torch.manual_seed(0)
        no_image = policy.predict_action_chunk(zero_image)

        zero_state = dict(batch)
        zero_state["state"] = torch.zeros_like(batch["state"])
        zero_state["delta_q"] = torch.zeros_like(batch["delta_q"])
        torch.manual_seed(0)
        no_state = policy.predict_action_chunk(zero_state)

        shuffled = dict(batch)
        shuffled["beaver_history_distance"] = batch["beaver_history_distance"].flip(-3)
        torch.manual_seed(0)
        shuffled_actions = policy.predict_action_chunk(shuffled)

        invalid = dict(batch)
        invalid["beaver_history_distance"] = torch.zeros_like(
            batch["beaver_history_distance"]
        )
        invalid["beaver_history_status"] = torch.full_like(
            batch["beaver_history_status"], 255.0
        )
        invalid["beaver_history_present"] = torch.zeros_like(
            batch["beaver_history_present"]
        )
        torch.manual_seed(0)
        no_beaver = policy.predict_action_chunk(invalid)

        self.assertFalse(torch.allclose(baseline, no_image, atol=1e-5))
        self.assertFalse(torch.allclose(baseline, no_state, atol=1e-5))
        self.assertFalse(torch.allclose(baseline, shuffled_actions, atol=1e-5))
        self.assertFalse(torch.allclose(baseline, no_beaver, atol=1e-5))

    def test_online_reset_and_chunk_selection(self) -> None:
        policy = self._policy().eval()
        observation = {
            "image": torch.rand(3, 64, 64),
            "state": torch.randn(7),
            "beaver_distance": torch.full((9, 4, 4), 250.0),
            "beaver_status": torch.full((9, 4, 4), 5.0),
            "beaver_present": torch.ones(9),
        }
        first = policy.select_action(observation)
        self.assertEqual(tuple(first.shape), (1, 7))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(policy.last_replanned)
        self.assertEqual(len(policy._online_history), 1)
        second = policy.select_action(
            {
                **observation,
                "beaver_distance": torch.full((9, 4, 4), 80.0),
            }
        )
        self.assertEqual(tuple(second.shape), (1, 7))
        self.assertEqual(len(policy._online_history), 2)
        self.assertFalse(policy.last_replanned)
        policy.reset()
        self.assertEqual(len(policy._online_history), 0)
        self.assertIsNone(policy._leftover)
        after_reset = policy.select_action(observation)
        self.assertTrue(policy.last_replanned)
        self.assertTrue(torch.isfinite(after_reset).all())

    def test_checkpoint_round_trip_restores_ema_and_statistics(self) -> None:
        config = tiny_grok_config()
        policy = self._policy()
        ema_payload = {
            name: parameter.detach().clone() + 0.01
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "WRM_grok.pt"
            torch.save(
                {
                    "kind": "WRM_grok",
                    "config": config.to_dict(),
                    "model": policy.state_dict(),
                    "ema": ema_payload,
                    "epoch": 1,
                    "global_step": 10,
                    "metrics": {"loss": 1.0},
                },
                path,
            )
            restored = load_policy(path, use_ema=True)
        self.assertIsInstance(restored, WRMGrokPolicy)
        self.assertTrue(restored.beaver_encoder.normalization_fitted)
        torch.testing.assert_close(
            restored.normalizer.beaver_temporal_p5,
            policy.normalizer.beaver_temporal_p5,
        )
        first_param = next(name for name in ema_payload)
        torch.testing.assert_close(
            dict(restored.named_parameters())[first_param].detach().cpu(),
            ema_payload[first_param].cpu(),
        )

    def test_eval_variant_and_mock_observation(self) -> None:
        self.assertTrue(policy_needs_beaver("WRM_grok"))
        self.assertEqual(
            validate_deployable_checkpoint(
                {"variant": "WRM_grok", "kind": EXPECTED_CHECKPOINT_KINDS["WRM_grok"]}
            ),
            "WRM_grok",
        )
        policy = self._policy().eval()
        observation = {
            "image": torch.rand(1, 3, 64, 64),
            "state": torch.randn(1, 7),
            "beaver_distance": torch.rand(1, 9, 4, 4) * 200 + 50,
            "beaver_status": torch.full((1, 9, 4, 4), 5.0),
            "beaver_present": torch.ones(1, 9),
        }
        action = policy.select_action(observation)
        self.assertEqual(tuple(action.shape), (1, 7))
        self.assertTrue(torch.isfinite(action).all())

    def test_smoke_optimizer_step(self) -> None:
        policy = self._policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        loss, _ = policy.compute_loss(self._batch())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        again, _ = policy.compute_loss(self._batch())
        self.assertTrue(torch.isfinite(again))

    def test_dataset_history_and_delta_alignment(self) -> None:
        config = tiny_grok_config()
        with patch(
            "policies.realman_beaver.dataset.LeRobotDataset",
            _FakeGrokLeRobotDataset,
        ):
            dataset = RealmanPolicyDataset(config, episodes=[0], stage="policy")
        sample = dataset[0]
        self.assertEqual(tuple(sample["state"].shape), (2, 7))
        self.assertEqual(tuple(sample["delta_q"].shape), (2, 7))
        self.assertEqual(tuple(sample["beaver_history_distance"].shape), (2, 4, 9, 4, 4))
        self.assertEqual(tuple(sample["grasp_state"].shape), (2,))
        self.assertEqual(tuple(sample["action"].shape), (8, 7))


class GrokMetricsTest(unittest.TestCase):
    def test_trajectory_smoothness_and_calibration(self) -> None:
        target = torch.zeros(2, 8, 7)
        predicted = torch.zeros(2, 8, 7)
        predicted[:, 4:] = 0.2
        errors = trajectory_error_rad(predicted, target)
        self.assertGreater(errors["traj_mae_rad"], 0.0)
        self.assertIn("traj_mae_joint_2_rad", errors)
        actions = torch.zeros(1, 8, 7)
        actions[0, 4] = 1.0
        smooth = action_smoothness(actions)
        boundary = chunk_boundary_discontinuity(actions, n_action_steps=4)
        self.assertGreater(smooth["smoothness_max_rad"], 0.0)
        self.assertGreater(boundary["replan_boundary_jump_rad"], 0.0)
        logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]])
        labels = torch.tensor([[0, 2]])
        metrics = phase_precision_recall_f1(logits, labels)
        self.assertGreater(metrics["tightness_f1"], 0.9)
        probs = torch.softmax(logits, dim=-1)
        self.assertLess(expected_calibration_error(probs, labels), 0.2)

    def test_production_parameter_count_under_limit(self) -> None:
        config = load_config("policies/realman_beaver/configs/WRM_grok.yaml")
        policy = build_policy(
            config,
            ObservationNormalizer.identity(
                temporal_beaver_statistics=grok_statistics()
            ),
        )
        trainable = count_trainable_parameters(policy)
        self.assertLess(trainable, 100_000_000)
        self.assertGreater(trainable, 10_000_000)
        self.assertGreater(deployment_weight_bytes(policy), 1_000_000)


if __name__ == "__main__":
    unittest.main()
