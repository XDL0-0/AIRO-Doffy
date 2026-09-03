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
    # Locally available policies. Downloaded runs are mirrored through public
    # Hugging Face model repositories. Cluster identifiers
    # intentionally stay out of the public/local policy names; provenance
    # remains in each model card.
    POLICIES: dict[str, str] = field(
        default_factory=lambda: {
            "dp_beaver_enc": (
                "policies/downloaded/dp_beaver_enc/checkpoints/last.pt"
            ),
            "dp_beaver_near": (
                "policies/downloaded/dp_beaver_near/checkpoints/last.pt"
            ),
            "dp_beaver_near_gate": (
                "policies/downloaded/dp_beaver_near_gate/checkpoints/last.pt"
            ),
            "dp_beaver_closure": (
                "policies/downloaded/dp_beaver_closure/checkpoints/last.pt"
            ),
            "dp_beaver_key4": (
                "policies/downloaded/dp_beaver_key4/checkpoints/"
                "dp_beaver_key4_step_100000.pt"
            ),
            "dp_beaver_key4_pca": (
                "policies/downloaded/dp_beaver_key4_PCA/checkpoints/"
                "dp_beaver_key4_pca_step_100000.pt"
            ),
            "WRM_temporal": (
                "policies/downloaded/dp_beaver_temporal/checkpoints/"
                "WRM_temporal_step_100000.pt"
            ),
            "WRM_adaptive_all125": (
                "policies/downloaded/WRM_adaptive-all125/checkpoints/last.pt"
            ),
            "WRM_delta_all125": (
                "policies/downloaded/WRM_delta-all125/checkpoints/last.pt"
            ),
            "WRM_delta": "policies/downloaded/WRM_delta/checkpoints/last.pt",
            "WRM_antigravity": (
                "policies/downloaded/WRM_antigravity/checkpoints/last.pt"
            ),
            "WRM_grok": "policies/downloaded/WRM_grok/checkpoints/last.pt",
            "WRM_codex": "policies/downloaded/WRM_codex/checkpoints/last.pt",
            "WRM_claude": "policies/downloaded/WRM_claude/checkpoints/last.pt",
            "WRM_qwen": "policies/downloaded/WRM_qwen/checkpoints/last.pt",
            "WRM_wrap": "policies/downloaded/WRM_wrap/checkpoints/last.pt",
            "WRM_wrap_monitor": (
                "policies/downloaded/WRM_wrap_monitor/checkpoints/last.pt"
            ),
            "WRM_wrap_monitor_backup": (
                "policies/downloaded/WRM_wrap_monitor_backup/checkpoints/last.pt"
            ),
            "WRM_lobo_monitor": (
                "policies/downloaded/WRM_lobo_monitor/checkpoints/last.pt"
            ),
            "WRM_wrap_delta": (
                "policies/downloaded/WRM_wrap_delta/checkpoints/last.pt"
            ),
            "rdp_like_key4_all125": (
                "policies/downloaded/rdp_like_key4-all125/checkpoints/last.pt"
            ),
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
    INFERENCE_LATENCY_STEPS: int = 2

    # WRM_wrap deployment-only gate overrides. Checkpoints serialize their
    # training config, so changing ModelConfig defaults or WRM_wrap.yaml does
    # not affect an existing checkpoint. These values are deliberately kept
    # in the eval config and applied before the policy/encoder is constructed.
    WRAP_NEAR_THRESHOLD_MM: float | None = 10.0
    WRAP_RANGE_SCALE_MM: float | None = 300.0
    WRAP_LIFT_MIN_WRAP: float | None = 0.8
    # Any-of-pair: 01 (J3) and 10 (J4) are the reliable enclosure sensors.
    # Requiring 2/2 at 0 mm blocked large-bottle lift (sensor 11 blind) and
    # froze small-bottle wrap on the first 10 mm reading.
    WRAP_STOP_CLOSE_J3_WRAP: float | None = 0.5
    WRAP_STOP_CLOSE_J4_WRAP: float | None = 0.5
    # Deprecated shared override; when set it applies to both J3 and J4.
    WRAP_STOP_CLOSE_WRAP: float | None = 0.5
    WRAP_CONTACT_STOP_MM: float | None = 10.0
    WRAP_STOP_HOLD_FRAMES: int | None = 24
    WRAP_LIFT_HOLD_FRAMES: int | None = 48

    # ── Prediction / execution window ───────────────────────────────────
    # These are inference-time overrides. Edit the number for the policy you
    # evaluate here; values stored in the training checkpoint are overridden.
    #
    # PREDICTION_STEPS: action targets generated per inference.
    # ACTION_STEPS: leading targets actually executed before
    #               throwing away the remainder and predicting again.
    #
    # Example: PREDICTION_STEPS=16 and ACTION_STEPS=4 means predict actions
    # [t..t+15], execute only [t..t+3], then replan from a fresh observation.
    # ACTION_STEPS must not exceed PREDICTION_STEPS. For direct DP/FM, the
    # usable upper bound is PREDICTION_STEPS - n_obs_steps + 1.
    # With the current U-Nets, direct prediction steps must be a multiple of 8;
    # RDP/RFM prediction steps must be a multiple of 16.
    PREDICTION_STEPS: dict[str, int] = field(
        default_factory=lambda: {
            "original_dp": 16,
            "dp_beaver": 16,
            "dp_beaver_closure": 16,
            "dp_beaver_enc": 16,
            "dp_beaver_near": 16,
            "dp_beaver_near_gate": 16,
            "dp_beaver_key4": 16,
            "dp_beaver_key4_pca": 16,
            "WRM_temporal": 16,
            "WRM_delta": 16,
            "WRM_adaptive": 16,
            "WRM_antigravity": 16,
            "WRM_grok": 16,
            "WRM_codex": 16,
            "WRM_claude": 16,
            "WRM_qwen": 16,
            "WRM_wrap": 16,
            "WRM_wrap_delta": 16,
            "WRM_wrap_monitor": 16,
            "WRM_wrap_monitor_backup": 16,
            "WRM_lobo_monitor": 16,
            # Registered run names; checkpoint loading still dispatches by the
            # internal WRM_delta/WRM_adaptive variant above.
            "WRM_delta_all125": 16,
            "WRM_adaptive_all125": 16,
            "fm": 16,
            "fm_beaver": 16,
            # Reactive policies predict a longer decoded action trajectory.
            "rdp_like": 32,
            # Registered run name; checkpoint loading dispatches through its
            # internal rdp_like variant.
            "rdp_like_key4_all125": 32,
            "rfm": 32,
        }
    )
    ACTION_STEPS: dict[str, int] = field(
        default_factory=lambda: {
            "original_dp": 8,
            "dp_beaver": 8,
            "dp_beaver_closure": 8,
            "dp_beaver_enc": 8,
            "dp_beaver_near": 8,
            "dp_beaver_near_gate": 8,
            "dp_beaver_key4": 8,
            "dp_beaver_key4_pca": 8,
            "WRM_temporal": 8,
            "WRM_delta": 8,
            "WRM_adaptive": 8,
            "WRM_antigravity": 8,
            "WRM_grok": 8,
            "WRM_codex": 4,
            "WRM_claude": 8,
            "WRM_qwen": 8,
            "WRM_wrap": 8,
            "WRM_wrap_delta": 8,
            "WRM_wrap_monitor": 8,
            "WRM_wrap_monitor_backup": 8,
            "WRM_lobo_monitor": 8,
            "WRM_delta_all125": 8,
            "WRM_adaptive_all125": 8,
            "fm": 8,
            "fm_beaver": 8,
            "rdp_like": 8,
            "rdp_like_key4_all125": 8,
            "rfm": 8,
        }
    )

    # Episodes per policy. 0 = loop until Ctrl-C.
    EPISODES: int = 1
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
    SAVE_VIDEO: bool = True  # camera feed, mp4/h264 at FPS
    SAVE_LOG: bool = True  # per-step JSONL (joints, action, flags)
    # Live cv2 monitor window: camera + rolling joint plot + Beaver heatmaps.
    # Auto-disables when no display is available.
    MONITOR: bool = True
    # Prompt for a success/failure verdict after each episode (y/n, Enter =
    # skip). The verdict is stored in the episode and run summaries.
    ASK_SUCCESS: bool = True

    # ── Pause / freedrive ───────────────────────────────────────────────
    # Enter during an episode pauses the run; with PAUSE_FREEDRIVE the robot
    # then enters drag-teach (freedrive) so the operator can reposition it
    # by hand. Enter again exits freedrive and resumes the episode from the
    # current joint configuration (the policy queues are reset so it
    # replans from the new pose). Backends without freedrive fall back to
    # holding the pose.
    PAUSE_FREEDRIVE: bool = True
    # RealMan drag-teach sensitivity (0..100) applied when pausing; None
    # keeps the SDK default.
    FREEDRIVE_SENSITIVITY: int | None = 99
    # If the camera stream or Beaver falls behind this long, the step is
    # flagged stale in the log (Beaver data is masked out by `present=0`).
    STALE_AFTER_S: float = 0.5

    # ── Hardware (validated against Config at startup) ───────────────────
    ROBOT_TYPE: str = "realman"  # eval_policy.py only supports the RM75

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
        for name, value in (("WRAP_RANGE_SCALE_MM", self.WRAP_RANGE_SCALE_MM),):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")
        if self.WRAP_NEAR_THRESHOLD_MM is not None and self.WRAP_NEAR_THRESHOLD_MM < 0:
            raise ValueError(
                "WRAP_NEAR_THRESHOLD_MM must be non-negative when set, got "
                f"{self.WRAP_NEAR_THRESHOLD_MM}"
            )
        if self.WRAP_CONTACT_STOP_MM is not None and self.WRAP_CONTACT_STOP_MM < 0:
            raise ValueError(
                "WRAP_CONTACT_STOP_MM must be non-negative when set, got "
                f"{self.WRAP_CONTACT_STOP_MM}"
            )
        for name, value in (
            ("WRAP_LIFT_MIN_WRAP", self.WRAP_LIFT_MIN_WRAP),
            ("WRAP_STOP_CLOSE_J3_WRAP", self.WRAP_STOP_CLOSE_J3_WRAP),
            ("WRAP_STOP_CLOSE_J4_WRAP", self.WRAP_STOP_CLOSE_J4_WRAP),
            ("WRAP_STOP_CLOSE_WRAP", self.WRAP_STOP_CLOSE_WRAP),
        ):
            if value is not None and not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1] when set, got {value}")
        for name, value in (
            ("WRAP_STOP_HOLD_FRAMES", self.WRAP_STOP_HOLD_FRAMES),
            ("WRAP_LIFT_HOLD_FRAMES", self.WRAP_LIFT_HOLD_FRAMES),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer when set")
        if (
            self.WRAP_STOP_HOLD_FRAMES is not None
            and self.WRAP_LIFT_HOLD_FRAMES is not None
            and self.WRAP_LIFT_HOLD_FRAMES < self.WRAP_STOP_HOLD_FRAMES
        ):
            raise ValueError(
                "WRAP_LIFT_HOLD_FRAMES must be >= WRAP_STOP_HOLD_FRAMES"
            )
        if self.MAX_JOINT_DELTA is not None and self.MAX_JOINT_DELTA <= 0:
            raise ValueError(
                f"MAX_JOINT_DELTA must be positive, got {self.MAX_JOINT_DELTA}"
            )
        if (
            self.FREEDRIVE_SENSITIVITY is not None
            and not 0 <= self.FREEDRIVE_SENSITIVITY <= 100
        ):
            raise ValueError(
                "FREEDRIVE_SENSITIVITY must be in 0..100 or None, got "
                f"{self.FREEDRIVE_SENSITIVITY}"
            )
        prediction_variants = set(self.PREDICTION_STEPS)
        action_variants = set(self.ACTION_STEPS)
        if prediction_variants != action_variants:
            raise ValueError(
                "PREDICTION_STEPS and ACTION_STEPS must contain the same "
                f"policy names; only in prediction={sorted(prediction_variants - action_variants)}, "
                f"only in action={sorted(action_variants - prediction_variants)}"
            )
        missing_variants = set(self.POLICIES) - prediction_variants
        if missing_variants:
            raise ValueError(
                "Missing PREDICTION_STEPS/ACTION_STEPS entries for policies: "
                f"{sorted(missing_variants)}"
            )
        for variant in sorted(prediction_variants):
            prediction_steps = self.PREDICTION_STEPS[variant]
            action_steps = self.ACTION_STEPS[variant]
            if prediction_steps <= 0 or action_steps <= 0:
                raise ValueError(
                    f"{variant}: prediction/action steps must be positive, got "
                    f"{prediction_steps}/{action_steps}"
                )
            if action_steps > prediction_steps:
                raise ValueError(
                    f"{variant}: ACTION_STEPS ({action_steps}) cannot exceed "
                    f"PREDICTION_STEPS ({prediction_steps})"
                )
            if variant in {"rdp_like", "rfm"}:
                if prediction_steps % 16:
                    raise ValueError(
                        f"{variant}: PREDICTION_STEPS must be a multiple of 16, "
                        f"got {prediction_steps}"
                    )
            else:
                if prediction_steps % 8:
                    raise ValueError(
                        f"{variant}: PREDICTION_STEPS must be a multiple of 8, "
                        f"got {prediction_steps}"
                    )
                # All current direct checkpoints use n_obs_steps=2.
                if action_steps > prediction_steps - 1:
                    raise ValueError(
                        f"{variant}: ACTION_STEPS ({action_steps}) exceeds the "
                        f"usable direct-policy window ({prediction_steps - 1})"
                    )
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
