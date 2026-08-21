from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from eval_policy import (
    EXPECTED_CHECKPOINT_KINDS,
    configure_policy_execution_window,
    policy_needs_beaver,
    select_action_with_latency,
    validate_deployable_checkpoint,
)


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


class SixPolicyCompatibilityTest(unittest.TestCase):
    def test_checkpoint_kind_mapping_accepts_all_six_policies(self) -> None:
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
        eval_cfg = SimpleNamespace(
            RDP_SLOW_REPLAN_STEPS=6,
            RFM_SLOW_REPLAN_STEPS=7,
        )
        for variant in ("original_dp", "dp_beaver", "fm", "fm_beaver"):
            policy = self._policy(variant, action_steps=8)
            with self.subTest(variant=variant):
                self.assertEqual(
                    configure_policy_execution_window(policy, eval_cfg), 8
                )

        rdp = self._policy("rdp_like", rdp_steps=3)
        rfm = self._policy("rfm", rfm_steps=4)
        self.assertEqual(configure_policy_execution_window(rdp, eval_cfg), 6)
        self.assertEqual(rdp.config.rdp.slow_replan_steps, 6)
        self.assertEqual(configure_policy_execution_window(rfm, eval_cfg), 7)
        self.assertEqual(rfm.config.rfm.slow_replan_steps, 7)

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
                    n_action_steps=action_steps,
                ),
                rdp=SimpleNamespace(slow_replan_steps=rdp_steps),
                rfm=SimpleNamespace(slow_replan_steps=rfm_steps),
            )
        )

if __name__ == "__main__":
    unittest.main()
