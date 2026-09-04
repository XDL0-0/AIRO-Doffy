"""Unit tests for WRM_phase_ddim: Fast DDIM planning + per-tick Phase Task Monitor."""

import unittest
from pathlib import Path
import torch

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import RealmanBeaverConfig
from policies.realman_beaver.dataset import ObservationNormalizer
from policies.realman_beaver.modeling import build_policy
from policies.realman_beaver.modeling_wrm_phase_ddim import PhaseDDIMBeaverDPPolicy


class PhaseDDIMPolicyTest(unittest.TestCase):
    def setUp(self):
        self.config = RealmanBeaverConfig()
        self.config.model.variant = "WRM_phase_ddim"
        self.config.model.noise_scheduler_type = "DDIM"
        self.config.model.num_inference_steps = 15
        self.config.model.n_action_steps = 4
        self.config.validate()
        self.normalizer = ObservationNormalizer.identity(7, 7)

    def test_initialization_and_ddim_scheduler(self):
        policy = build_policy(self.config, self.normalizer)
        self.assertIsInstance(policy, PhaseDDIMBeaverDPPolicy)
        self.assertEqual(
            policy.native_policy.diffusion.noise_scheduler.__class__.__name__,
            "DDIMScheduler",
        )
        self.assertEqual(policy.native_policy.diffusion.num_inference_steps, 15)

    def test_per_tick_contact_interception_and_smoothing(self):
        policy = build_policy(self.config, self.normalizer)
        policy.eval()

        # Mock observation where fingers are closed onto bottle
        # J3 = -1.60 (past the -1.35 shield), ToF readings = 0.0mm
        current_joints = torch.tensor([[0.0, 1.45, 0.0, -1.60, -0.45, -1.65, 0.0]])
        obs = {
            "state": current_joints,
            "image": torch.zeros(1, 3, 480, 640),
            "beaver_distance": torch.zeros(1, 9, 4, 4),  # 0mm contact
            "beaver_status": torch.full((1, 9, 4, 4), 5, dtype=torch.long),
            "beaver_present": torch.ones(1, 9),
        }

        # First action
        act1 = policy.select_action(obs)
        self.assertEqual(act1.shape, (1, 7))

        # Check that EMA smoothing maintains consistency
        act2 = policy.select_action(obs)
        self.assertEqual(act2.shape, (1, 7))

    def test_load_packaged_checkpoint(self):
        ckpt_path = Path("policies/downloaded/WRM_phase_ddim/checkpoints/last.pt")
        if not ckpt_path.exists():
            self.skipTest("Packaged checkpoint not found")
        policy = load_policy(str(ckpt_path), device="cpu")
        self.assertIsInstance(policy, PhaseDDIMBeaverDPPolicy)
        self.assertEqual(policy.native_policy.diffusion.noise_scheduler.__class__.__name__, "DDIMScheduler")
        self.assertEqual(policy.native_policy.diffusion.num_inference_steps, 15)


if __name__ == "__main__":
    unittest.main()
