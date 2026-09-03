from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from eval_policy import (
    EXPECTED_CHECKPOINT_KINDS,
    MonitorWindow,
    PolicyEvaluator,
    _infer_policy_name,
    _resolve_policy_selections,
    policy_needs_beaver,
    policy_step_window,
    select_action_with_latency,
    validate_deployable_checkpoint,
)
from policies.realman_beaver.checkpoint import configure_deployment_steps
from policies.realman_beaver.configuration import RealmanBeaverConfig


class MonitorWindowTest(unittest.TestCase):
    def test_window_is_initialized_with_content_and_explicit_size(self) -> None:
        monitor = MonitorWindow(dof=7, distance_max_mm=2550.0, fps=24)
        with (
            patch("eval_policy.cv2.namedWindow") as named_window,
            patch("eval_policy.cv2.imshow") as imshow,
            patch("eval_policy.cv2.resizeWindow") as resize_window,
            patch("eval_policy.cv2.waitKey", return_value=-1),
        ):
            self.assertTrue(monitor._check_enabled())

        named_window.assert_called_once()
        displayed = imshow.call_args.args[1]
        self.assertEqual(displayed.shape, (720, 1120, 3))
        resize_window.assert_called_once_with("Policy Eval Monitor", 1120, 720)

    def test_large_camera_frame_is_fitted_into_the_dashboard(self) -> None:
        monitor = MonitorWindow(dof=7, distance_max_mm=2550.0, fps=24)
        canvas = monitor._compose(
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            None,
            {"state": "READY", "beaver_ok": None},
        )
        self.assertEqual(canvas.shape, (720, 1120, 3))

    def test_waiting_preview_reads_and_displays_live_camera(self) -> None:
        evaluator = PolicyEvaluator.__new__(PolicyEvaluator)
        evaluator.monitor = SimpleNamespace(show=Mock(), pump_keys=Mock())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        evaluator.cameras = SimpleNamespace(
            get_images=Mock(return_value={"camera_0": frame})
        )
        evaluator.camera_name = "camera_0"
        evaluator.backend = SimpleNamespace(
            get_joint_configuration=Mock(return_value=np.zeros(7))
        )
        evaluator.beaver = None
        evaluator._last_beaver_np = None
        evaluator.cfg = SimpleNamespace(FPS=24, STALE_AFTER_S=0.5)
        evaluator.name = "dp_beaver"
        evaluator._monitor_preview_warning_shown = False

        evaluator._show_monitor_preview("READY", 0)

        displayed = evaluator.monitor.show.call_args.kwargs
        self.assertIs(displayed["camera"], frame)
        self.assertEqual(displayed["status"]["state"], "READY")
        self.assertEqual(displayed["status"]["episode"], 0)

    def test_image_tensor_converts_normalized_float_camera_frame(self) -> None:
        evaluator = PolicyEvaluator.__new__(PolicyEvaluator)
        frame = np.full((2, 3, 3), 0.5, dtype=np.float32)
        evaluator.cameras = SimpleNamespace(
            get_images=Mock(return_value={"camera_0": frame})
        )
        evaluator.camera_name = "camera_0"
        evaluator.device = "cpu"

        tensor = evaluator._image_tensor()

        stored = evaluator._last_video_frame["camera_0"]
        self.assertEqual(stored.dtype, np.uint8)
        self.assertTrue(stored.flags.c_contiguous)
        self.assertTrue(np.all(stored == 127))
        self.assertEqual(tuple(tensor.shape), (1, 3, 2, 3))
        self.assertTrue(torch.allclose(tensor, torch.full_like(tensor, 127 / 255)))


class _QueuedPolicy:
    def __init__(self, chunks: list[list[float]]) -> None:
        self._chunks = iter(chunks)
        self._queue: list[float] = []
        self.replan_count = 0
        self.last_replanned = False
        self.last_chunk_step = 0

    def select_action(self, _observation) -> torch.Tensor:
        if not self._queue:
            self._queue = list(next(self._chunks))
            self.replan_count += 1
            self.last_replanned = True
            self.last_chunk_step = 0
        else:
            self.last_replanned = False
            self.last_chunk_step += 1
        return torch.tensor([[self._queue.pop(0)]])


class InferenceLatencyTest(unittest.TestCase):
    def test_discards_only_the_new_chunks_stale_prefix(self) -> None:
        policy = _QueuedPolicy([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]])

        first = select_action_with_latency(policy, {}, latency_steps=2)
        second = select_action_with_latency(policy, {}, latency_steps=2)
        third = select_action_with_latency(policy, {}, latency_steps=2)
        fourth = select_action_with_latency(policy, {}, latency_steps=2)

        self.assertEqual(
            [first.item(), second.item(), third.item(), fourth.item()],
            [2, 3, 4, 5],
        )
        self.assertEqual(policy.replan_count, 1)
        self.assertEqual(policy.last_latency_steps, 0)

    def test_rejects_latency_that_crosses_a_chunk_boundary(self) -> None:
        policy = _QueuedPolicy([[0, 1], [10, 11]])
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            select_action_with_latency(policy, {}, latency_steps=2)

    def test_zero_latency_returns_first_prediction(self) -> None:
        policy = _QueuedPolicy([[7, 8]])
        action = select_action_with_latency(policy, {}, latency_steps=0)
        self.assertEqual(action.item(), 7)
        self.assertTrue(policy.last_replanned)
        self.assertEqual(policy.last_latency_steps, 0)


class PolicyCompatibilityTest(unittest.TestCase):
    def test_custom_labels_keep_icra_seeds_distinct(self) -> None:
        selections = _resolve_policy_selections(
            [
                "joint_only_seed42=/models/joint_only/seed_42/last.pt",
                "joint_only_seed43=/models/joint_only/seed_43/last.pt",
            ],
            {"WRM_temporal": "/models/default/last.pt"},
        )
        self.assertEqual(
            selections,
            {
                "joint_only_seed42": "/models/joint_only/seed_42/last.pt",
                "joint_only_seed43": "/models/joint_only/seed_43/last.pt",
            },
        )

    def test_custom_label_rejects_output_path_traversal(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Invalid policy label"):
            _resolve_policy_selections(
                ["../escape=/models/last.pt"],
                {"WRM_temporal": "/models/default/last.pt"},
            )

    def test_bare_temporal_checkpoint_path_uses_registered_variant(self) -> None:
        checkpoint = (
            "policies/downloaded/dp_beaver_temporal/checkpoints/"
            "WRM_temporal_step_075000.pt"
        )
        policies = {
            "WRM_temporal": (
                "policies/downloaded/dp_beaver_temporal/checkpoints/"
                "WRM_temporal_step_100000.pt"
            ),
            "WRM_delta": "policies/downloaded/WRM_delta/checkpoints/last.pt",
        }
        self.assertEqual(_infer_policy_name(checkpoint, policies), "WRM_temporal")

    def test_checkpoint_kind_mapping_accepts_all_policies(self) -> None:
        for variant, kind in EXPECTED_CHECKPOINT_KINDS.items():
            with self.subTest(variant=variant):
                self.assertEqual(
                    validate_deployable_checkpoint(
                        {"variant": variant, "kind": kind}
                    ),
                    variant,
                )

    def test_checkpoint_kind_mapping_rejects_non_deployable_kinds(self) -> None:
        with self.assertRaisesRegex(ValueError, "tokenizer-only"):
            validate_deployable_checkpoint(
                {"variant": "rfm", "kind": "tokenizer"}
            )
        with self.assertRaisesRegex(ValueError, "expected 'latent_fm'"):
            validate_deployable_checkpoint(
                {"variant": "rfm", "kind": "latent_dp"}
            )

    def test_beaver_requirements_match_policy_inputs(self) -> None:
        expected = {
            "original_dp": False,
            "dp_beaver": True,
            "dp_beaver_enc": True,
            "dp_beaver_near": True,
            "dp_beaver_near_gate": True,
            "dp_beaver_key4": True,
            "dp_beaver_key4_pca": True,
            "rdp_like": True,
            "fm": False,
            "fm_beaver": True,
            "rfm": True,
        }
        self.assertEqual(
            {variant: policy_needs_beaver(variant) for variant in expected},
            expected,
        )

    def test_execution_windows_cover_direct_and_reactive_policies(self) -> None:
        for variant in (
            "original_dp",
            "dp_beaver",
            "dp_beaver_enc",
            "dp_beaver_near",
            "dp_beaver_near_gate",
            "dp_beaver_key4",
            "dp_beaver_key4_pca",
            "fm",
            "fm_beaver",
        ):
            policy = self._policy(variant, action_steps=8)
            with self.subTest(variant=variant):
                self.assertEqual(policy_step_window(policy), (16, 8))

        rdp = self._policy("rdp_like", rdp_steps=6)
        rfm = self._policy("rfm", rfm_steps=7)
        self.assertEqual(policy_step_window(rdp), (32, 6))
        self.assertEqual(policy_step_window(rfm), (32, 7))

    def test_deployment_steps_override_direct_and_reactive_configs(self) -> None:
        direct = RealmanBeaverConfig()
        self.assertEqual(
            configure_deployment_steps(
                direct, prediction_steps=8, action_steps=4
            ),
            (8, 4),
        )
        self.assertEqual((direct.model.horizon, direct.model.n_action_steps), (8, 4))

        reactive = RealmanBeaverConfig()
        reactive.model.variant = "rdp_like"
        self.assertEqual(
            configure_deployment_steps(
                reactive, prediction_steps=16, action_steps=4
            ),
            (16, 4),
        )
        self.assertEqual(
            (reactive.rdp.action_horizon, reactive.rdp.slow_replan_steps),
            (16, 4),
        )

    @staticmethod
    def _policy(
        variant: str,
        *,
        action_steps: int = 8,
        rdp_steps: int = 8,
        rfm_steps: int = 8,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(
                    variant=variant,
                    horizon=16,
                    n_action_steps=action_steps,
                ),
                rdp=SimpleNamespace(
                    action_horizon=32, slow_replan_steps=rdp_steps
                ),
                rfm=SimpleNamespace(
                    action_horizon=32, slow_replan_steps=rfm_steps
                ),
            )
        )

if __name__ == "__main__":
    unittest.main()
