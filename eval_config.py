"""Evaluation configuration for the trained RealMan-Beaver policies.

Hardware (robot IP, Realsense, Beaver USB, safety thresholds, initial pose)
is inherited from :mod:`config` so there is a single source of truth for the
robot stack. This file only configures the evaluation itself: which trained
policies to run, at what rate, and what to record.

Usage:
    python eval_policy.py                      # evaluate all policies in EVAL.POLICIES
    python eval_policy.py --policy dp_beaver   # only one policy
    python eval_policy.py --checkpoint-root policies/output/WRM_grasp_cylinder_all
    python eval_policy.py --episodes 5 --fps 24
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import Config


@dataclass
class EvalConfig:
    # ── Policies to evaluate ─────────────────────────────────────────────
    # name -> checkpoint path. `last.pt` is the final EMA checkpoint written
    # by policies/realman_beaver/train.py; numbered snapshots (e.g.
    # dp_beaver_step_100000.pt) can be substituted per run via --checkpoint.
    POLICIES: dict[str, str] = field(
        default_factory=lambda: {
            name: f"policies/output/WRM_grasp_cylinder_lero/{name}/last.pt"
            for name in (
                "original_dp",
                "dp_beaver",
                "rdp_like",
                "fm",
                "fm_beaver",
                "rfm",
            )
        }
    )
    # Prefer the EMA weights saved in each checkpoint.
    USE_EMA: bool = True
    DEVICE: str = "cuda:0"

    # ── Control loop ─────────────────────────────────────────────────────
    # The dataset was recorded at 24 Hz and the policies were trained for it.
    # Keep this equal to the dataset fps unless you retrained at another rate.
    FPS: int = 24
    # Number of freshly predicted action steps to discard after each replan.
    # RDP uses 4 at 24 Hz: inference takes about 1/6 s, so action[0:4] is
    # already stale when the new chunk becomes available. Set 0 to disable.
    INFERENCE_LATENCY_STEPS: int = 0
    # Episodes per policy. 0 = loop until Ctrl-C.
    EPISODES: int = 1
    # RDP-like only: how many fast control ticks each latent chunk is used
    # for before the slow LDP samples a fresh one. 8 ticks at 24 Hz ~= 3 Hz,
    # aligned with the DP variants' n_action_steps=8 receding horizon.
    # Overrides the value baked into the checkpoint at load time.
    RDP_SLOW_REPLAN_STEPS: int | None = 8
    # RFM equivalent of RDP_SLOW_REPLAN_STEPS.
    RFM_SLOW_REPLAN_STEPS: int | None = 8
    # Max control steps per episode (~25 s per episode at 24 Hz).
    MAX_STEPS: int = 1500

    # ── Safety ───────────────────────────────────────────────────────────
    # Per-joint max |target - current| before a command is dropped. Inherited
    # from Config.MOVE_THRESHOLD (resized to the robot's DoF at runtime).
    # A violation is counted in the episode log — a policy that hits this
    # repeatedly is producing absurd targets.
    MOVE_THRESHOLD: np.ndarray | None = None
    # Per-tick joint motion cap in rad: |delta| <= MAX_JOINT_DELTA, so the
    # executed trajectory never exceeds REALMAN_MAX_JOINT_SPEED. None disables
    # the clamp (not recommended).
    MAX_JOINT_DELTA: float | None = None

    # ── Recording ────────────────────────────────────────────────────────
    # Runs are written to OUTPUT_DIR/<timestamp>/<policy>/episode_<i>/
    OUTPUT_DIR: str = "policies/output/eval"
    SAVE_VIDEO: bool = True          # camera feed, mp4/h264 at FPS
    SAVE_LOG: bool = True            # per-step JSONL (joints, action, flags)
    # Live cv2 monitor window: camera + rolling joint plot + Beaver heatmaps.
    # Auto-disables when no display is available.
    MONITOR: bool = True
    # Prompt for a success/failure verdict after each episode (y/n, Enter =
    # skip). The verdict is stored in the episode and run summaries.
    ASK_SUCCESS: bool = True
    # If the camera stream or Beaver falls behind this long, the step is
    # flagged stale in the log (Beaver data is masked out by `present=0`).
    STALE_AFTER_S: float = 0.5

    # ── Hardware (validated against Config at startup) ───────────────────
    ROBOT_TYPE: str = "realman"      # eval_policy.py only supports the RM75

    def __post_init__(self) -> None:
        if self.ROBOT_TYPE.lower() != "realman":
            raise ValueError(
                f"eval_policy.py supports ROBOT_TYPE='realman', got "
                f"'{self.ROBOT_TYPE}'."
            )
        if self.FPS <= 0:
            raise ValueError(f"FPS must be positive, got {self.FPS}")
        if self.INFERENCE_LATENCY_STEPS < 0:
            raise ValueError(
                "INFERENCE_LATENCY_STEPS cannot be negative, got "
                f"{self.INFERENCE_LATENCY_STEPS}"
            )
        if self.EPISODES < 0:
            raise ValueError(f"EPISODES cannot be negative, got {self.EPISODES}")
        if self.MAX_STEPS <= 0:
            raise ValueError(f"MAX_STEPS must be positive, got {self.MAX_STEPS}")
        if self.STALE_AFTER_S <= 0:
            raise ValueError(
                f"STALE_AFTER_S must be positive, got {self.STALE_AFTER_S}"
            )
        if self.MAX_JOINT_DELTA is not None and self.MAX_JOINT_DELTA <= 0:
            raise ValueError(
                f"MAX_JOINT_DELTA must be positive, got {self.MAX_JOINT_DELTA}"
            )
        for name, value in (
            ("RDP_SLOW_REPLAN_STEPS", self.RDP_SLOW_REPLAN_STEPS),
            ("RFM_SLOW_REPLAN_STEPS", self.RFM_SLOW_REPLAN_STEPS),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")
        hw = Config()
        if hw.ROBOT_TYPE.lower() != self.ROBOT_TYPE:
            raise ValueError(
                "eval_config.ROBOT_TYPE must match config.Config.ROBOT_TYPE "
                f"('{self.ROBOT_TYPE}' vs '{hw.ROBOT_TYPE}')."
            )
        if self.MOVE_THRESHOLD is None:
            self.MOVE_THRESHOLD = np.asarray(hw.MOVE_THRESHOLD, dtype=float)
        else:
            self.MOVE_THRESHOLD = np.asarray(self.MOVE_THRESHOLD, dtype=float)
        if self.MAX_JOINT_DELTA is None:
            # Same per-joint speed limit the VR teleop pipeline enforced
            # while the training data was recorded.
            self.MAX_JOINT_DELTA = hw.REALMAN_MAX_JOINT_SPEED / self.FPS
