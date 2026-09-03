"""Typed configuration for the diffusion and flow-matching baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

STRUCTURED_BEAVER_DP_VARIANTS = frozenset(
    {
        "dp_beaver_enc",
        "dp_beaver_near",
        "dp_beaver_near_gate",
        "dp_beaver_key4",
        "dp_beaver_key4_pca",
    }
)
TEMPORAL_BEAVER_VARIANT = "WRM_temporal"
DELTA_BEAVER_VARIANT = "WRM_delta"
ADAPTIVE_BEAVER_VARIANT = "WRM_adaptive"
ANTIGRAVITY_BEAVER_VARIANT = "WRM_antigravity"
GROK_BEAVER_VARIANT = "WRM_grok"
CODEX_BEAVER_VARIANT = "WRM_codex"
CLAUDE_BEAVER_VARIANT = "WRM_claude"
QWEN_BEAVER_VARIANT = "WRM_qwen"
WRAP_BEAVER_VARIANT = "WRM_wrap"
WRAP_DELTA_BEAVER_VARIANT = "WRM_wrap_delta"
WRAP_MONITOR_BEAVER_VARIANT = "WRM_wrap_monitor"
WRAP_MONITOR_BACKUP_BEAVER_VARIANT = "WRM_wrap_monitor_backup"
LOBO_MONITOR_BEAVER_VARIANT = "WRM_lobo_monitor"
WRAP_MONITOR_BEAVER_VARIANTS = frozenset(
    {
        WRAP_MONITOR_BEAVER_VARIANT,
        WRAP_MONITOR_BACKUP_BEAVER_VARIANT,
        LOBO_MONITOR_BEAVER_VARIANT,
    }
)
WRAP_BEAVER_VARIANTS = frozenset(
    {
        WRAP_BEAVER_VARIANT,
        WRAP_DELTA_BEAVER_VARIANT,
        *WRAP_MONITOR_BEAVER_VARIANTS,
    }
)
RELATIVE_ACTION_VARIANTS = frozenset(
    {QWEN_BEAVER_VARIANT, WRAP_DELTA_BEAVER_VARIANT}
)
BEAVER_CLOSURE_VARIANT = "dp_beaver_closure"
KEY4_BEAVER_DP_VARIANTS = frozenset({"dp_beaver_key4", "dp_beaver_key4_pca"})
COMPETITION_BEAVER_VARIANTS = frozenset(
    {
        ANTIGRAVITY_BEAVER_VARIANT,
        GROK_BEAVER_VARIANT,
        CODEX_BEAVER_VARIANT,
        CLAUDE_BEAVER_VARIANT,
        QWEN_BEAVER_VARIANT,
    }
)
HISTORY_BEAVER_VARIANTS = frozenset(
    {
        TEMPORAL_BEAVER_VARIANT,
        ADAPTIVE_BEAVER_VARIANT,
        *WRAP_BEAVER_VARIANTS,
        *COMPETITION_BEAVER_VARIANTS,
    }
)
MOTION_DELTA_VARIANTS = frozenset(
    {
        DELTA_BEAVER_VARIANT,
        ADAPTIVE_BEAVER_VARIANT,
        ANTIGRAVITY_BEAVER_VARIANT,
        GROK_BEAVER_VARIANT,
        CLAUDE_BEAVER_VARIANT,
    }
)
GRASP_STATE_VARIANTS = frozenset(
    {
        BEAVER_CLOSURE_VARIANT,
        TEMPORAL_BEAVER_VARIANT,
        DELTA_BEAVER_VARIANT,
        ADAPTIVE_BEAVER_VARIANT,
        ANTIGRAVITY_BEAVER_VARIANT,
        GROK_BEAVER_VARIANT,
        CLAUDE_BEAVER_VARIANT,
        QWEN_BEAVER_VARIANT,
    }
)

SUPPORTED_VARIANTS = {
    "original_dp",
    "dp_beaver",
    BEAVER_CLOSURE_VARIANT,
    *STRUCTURED_BEAVER_DP_VARIANTS,
    TEMPORAL_BEAVER_VARIANT,
    DELTA_BEAVER_VARIANT,
    ADAPTIVE_BEAVER_VARIANT,
    *COMPETITION_BEAVER_VARIANTS,
    *WRAP_BEAVER_VARIANTS,
    "rdp_like",
    "fm",
    "fm_beaver",
    "rfm",
}


@dataclass
class DatasetConfig:
    root: str = "/home/yuyuan/AIRO-Doffy/datasets/WRM_grasp_lero"
    repo_id: str = "WRM_grasp_lero"
    video_backend: str = "pyav"
    fps: int = 24
    image_key: str = "observation.images.camera_0"
    image_shape: tuple[int, int, int] = (3, 480, 640)
    state_key: str = "observation.state"
    action_key: str = "action"
    beaver_distance_key: str = "observation.beaver.distance_mm"
    beaver_present_key: str = "observation.beaver.present"
    beaver_status_key: str = "observation.beaver.target_status"
    grasp_state_key: str = "tightness"
    # Stable physical sensor order from Config.BEAVER_SENSOR_LAYOUT:
    # (0,0)..(0,4),(1,0)..(1,3). Selected temporal sensors are resolved
    # through these names rather than by interpreting names as tensor indices.
    beaver_sensor_layout: tuple[str, ...] = (
        "00",
        "01",
        "02",
        "03",
        "04",
        "10",
        "11",
        "12",
        "13",
    )
    # VL53L7CX target status codes that carry a usable distance reading.
    # 5 = valid, 9 = weak-signal. Everything else (255 = no target, etc.)
    # is filtered out pixel-wise during normalization.
    beaver_valid_statuses: tuple[int, ...] = (5, 9)
    val_fraction: float = 0.1
    split_seed: int = 42
    # Explicit zero-based LeRobot episode indices take precedence over the
    # seeded val_fraction split when provided.
    val_episodes: tuple[int, ...] | None = None
    normalization_source: str = "parquet"
    distance_max_mm: float = 2550.0
    normalization_floor: float = 1e-4
    # When true, each training batch draws equally from every bottle block
    # of ``episodes_per_bottle`` consecutive episodes (0-24, 25-49, ...).
    stratified_bottle_batch: bool = False
    episodes_per_bottle: int = 25

    def __post_init__(self) -> None:
        self.image_shape = tuple(self.image_shape)
        self.beaver_valid_statuses = tuple(self.beaver_valid_statuses)
        self.beaver_sensor_layout = tuple(
            str(name) for name in self.beaver_sensor_layout
        )
        if self.val_episodes is not None:
            self.val_episodes = tuple(self.val_episodes)


@dataclass
class ModelConfig:
    variant: str = "original_dp"
    state_dim: int = 7
    action_dim: int = 7
    beaver_shape: tuple[int, int, int] = (9, 4, 4)

    # Structured Beaver DP settings. These fields are only consumed by the
    # additive structured variants; the existing flat and reactive
    # Beaver paths retain their original representations.
    beaver_feature_dim: int = 64
    beaver_sensor_hidden_dim: int = 64
    beaver_sensor_feature_dim: int = 32
    beaver_near_threshold_mm: float = 300.0
    beaver_gate_hidden_dim: int = 32
    # Physical sensor slots 01, 02, 10, and 11 from Config.BEAVER_SENSOR_IDS.
    beaver_key_sensor_indices: tuple[int, ...] = (1, 2, 5, 6)
    # Number of fixed principal components retained independently per Key4
    # sensor by dp_beaver_key4_pca (4 sensors x 4 components = 16 features).
    beaver_pca_components: int = 4

    # Beaver-aware closure residual. The global branch remains an unmodified
    # RGB + q Diffusion Policy; these settings apply only to the lightweight
    # closed-loop residual branch.
    closure_joint_mask: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
    closure_beaver_encoder_dim: int = 64
    closure_sensor_hidden_dim: int = 64
    closure_hidden_dim: int = 128
    closure_grasp_loss_weight: float = 0.2
    closure_residual_loss_weight: float = 0.05
    closure_residual_scale: float = 0.15

    # WRM_temporal settings. Beaver has an independent 12-frame history for
    # every native DP observation time; image/qpos horizons stay unchanged.
    beaver_history_steps: int = 12
    beaver_temporal_feature_dim: int = 64
    beaver_frame_hidden_dim: int = 64
    beaver_frame_feature_dim: int = 32
    beaver_temporal_hidden_dim: int = 64
    beaver_grasp_loss_weight: float = 0.2
    beaver_temporal_sensors: tuple[str, ...] = ("01", "02", "10", "11")
    # Capacity-matched ICRA policy ablations. Disabled modalities are replaced
    # by zeros after their normal transforms, so the native DP architecture and
    # parameter count are identical across the modality factorial.
    use_visual_condition: bool = True
    use_beaver_condition: bool = True
    beaver_history_mode: str = "temporal"
    condition_on_grasp_probability: bool = True

    # WRM_delta settings. This variant uses only the current frame and a
    # configurable short-lag frame; it does not add a temporal network.
    beaver_delta_steps: int = 6
    beaver_delta_feature_dim: int = 64
    beaver_delta_sensor_hidden_dim: int = 64
    beaver_delta_sensor_feature_dim: int = 32
    beaver_delta_fusion_hidden_dim: int = 128
    beaver_delta_grasp_hidden_dim: int = 64
    beaver_delta_sensors: tuple[str, ...] = ("01", "02", "10", "11")
    lambda_grasp: float = 0.2

    # WRM_adaptive uses the four reliable near-field Beaver sensors as a
    # continuous object-relative contact field. It explicitly conditions
    # action generation on both short-term robot motion and the learned grasp
    # probability; no object-size class or task ID is used. The five noisy /
    # frequently invalid peripheral sensors are deliberately excluded.
    beaver_adaptive_history_steps: int = 12
    beaver_adaptive_motion_delta_steps: int = 6
    beaver_adaptive_feature_dim: int = 96
    beaver_adaptive_sensor_hidden_dim: int = 128
    beaver_adaptive_token_dim: int = 64
    beaver_adaptive_transformer_layers: int = 2
    beaver_adaptive_attention_heads: int = 4
    beaver_adaptive_grasp_hidden_dim: int = 128
    beaver_adaptive_grasp_loss_weight: float = 0.5
    beaver_adaptive_sensors: tuple[str, ...] = ("01", "02", "10", "11")
    beaver_adaptive_lag_steps: tuple[int, ...] = (1, 3, 6, 11)
    beaver_adaptive_proximity_scales_mm: tuple[float, ...] = (
        50.0,
        100.0,
        200.0,
        400.0,
    )
    beaver_adaptive_noise_std_mm: float = 5.0
    beaver_adaptive_pixel_dropout: float = 0.05
    beaver_adaptive_sensor_dropout: float = 0.05

    # WRM_antigravity: Key4 multi-scale geometry, temporal flux and
    # cross-sensor attention.
    beaver_antigravity_sensors: tuple[str, ...] = ("01", "02", "10", "11")
    beaver_antigravity_history_steps: int = 12
    beaver_antigravity_motion_delta_steps: int = 1
    beaver_antigravity_motion_delta_long_steps: int = 6
    beaver_antigravity_lag_steps: tuple[int, ...] = (1, 3, 6, 11)
    beaver_antigravity_proximity_scales_mm: tuple[float, ...] = (
        25.0,
        75.0,
        150.0,
        300.0,
    )
    beaver_antigravity_spatial_hidden_dim: int = 128
    beaver_antigravity_token_dim: int = 64
    beaver_antigravity_temporal_hidden_dim: int = 64
    beaver_antigravity_transformer_layers: int = 2
    beaver_antigravity_attention_heads: int = 4
    beaver_antigravity_feature_dim: int = 64
    beaver_antigravity_grasp_hidden_dim: int = 64
    beaver_antigravity_grasp_loss_weight: float = 0.2
    beaver_antigravity_enclosure_loss_weight: float = 0.1
    beaver_antigravity_noise_std_mm: float = 5.0
    beaver_antigravity_pixel_dropout: float = 0.05
    beaver_antigravity_sensor_dropout: float = 0.05
    beaver_antigravity_terminal_hold_damping: bool = True
    beaver_antigravity_hold_threshold: float = 0.85
    beaver_antigravity_max_damping: float = 0.4

    # WRM_grok: Key4 phase-gated flow matching with enclosure geometry.
    beaver_grok_history_steps: int = 12
    beaver_grok_motion_delta_steps: int = 6
    beaver_grok_feature_dim: int = 64
    beaver_grok_enclosure_dim: int = 16
    beaver_grok_frame_hidden_dim: int = 64
    beaver_grok_frame_feature_dim: int = 32
    beaver_grok_temporal_hidden_dim: int = 64
    beaver_grok_phase_hidden_dim: int = 64
    beaver_grok_wrap_threshold_mm: float = 150.0
    beaver_grok_min_near_sensors: int = 2
    beaver_grok_phase_loss_weight: float = 0.3
    beaver_grok_smooth_loss_weight: float = 0.05
    beaver_grok_hold_loss_weight: float = 0.2
    beaver_grok_overlap_blend_steps: int = 4
    beaver_grok_noise_std_mm: float = 3.0
    beaver_grok_sensors: tuple[str, ...] = ("01", "02", "10", "11")
    beaver_grok_near_scales_mm: tuple[float, ...] = (50.0, 150.0, 300.0)

    # WRM_codex: compact deterministic residual chunk policy.
    codex_beaver_sensors: tuple[str, ...] = ("01", "02", "10", "11")
    codex_beaver_history_steps: int = 12
    codex_token_dim: int = 128
    codex_vision_width: int = 32
    codex_contact_hidden_dim: int = 128
    codex_fusion_layers: int = 2
    codex_sensor_layers: int = 1
    codex_decoder_layers: int = 2
    codex_attention_heads: int = 4
    codex_dropout: float = 0.1
    codex_residual_scale: float = 2.0
    codex_activity_scale_rad: float = 0.02
    codex_activity_loss_weight: float = 0.2
    codex_velocity_loss_weight: float = 0.1
    codex_plan_ensemble: int = 3
    codex_plan_decay: float = 0.7

    # WRM_claude: Key4 contact-field encoder and pose-anchored deltas.
    claude_motion_delta_steps: int = 1
    claude_history_steps: int = 8
    claude_feature_dim: int = 64
    claude_sensor_hidden_dim: int = 48
    claude_token_dim: int = 48
    claude_transformer_layers: int = 2
    claude_attention_heads: int = 4
    claude_grasp_hidden_dim: int = 64
    claude_grasp_loss_weight: float = 0.3
    claude_smoothness_loss_weight: float = 0.02
    claude_sensors: tuple[str, ...] = ("01", "02", "10", "11")
    claude_lag_steps: tuple[int, ...] = (1, 3, 6)
    claude_proximity_scales_mm: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0)
    claude_noise_std_mm: float = 5.0
    claude_pixel_dropout: float = 0.05
    claude_sensor_dropout: float = 0.05

    # WRM_qwen: temporal Key4 encoder plus joint-motion conditioning and
    # re-anchored relative action diffusion.
    qwen_joint_history_steps: int = 12
    qwen_grasp_hidden_dim: int = 64
    qwen_grasp_loss_weight: float = 0.2

    # WRM_wrap: contact-preserving temporal Key4 + wrap/lift execution gate.
    beaver_wrap_history_steps: int = 12
    beaver_wrap_feature_dim: int = 64
    beaver_wrap_enclosure_dim: int = 4
    beaver_wrap_frame_hidden_dim: int = 64
    beaver_wrap_frame_feature_dim: int = 32
    beaver_wrap_temporal_hidden_dim: int = 64
    beaver_wrap_sensors: tuple[str, ...] = ("01", "02", "10", "11")
    # The first pair observes the J3 closure path and the second pair observes
    # the J4 closure path.  Keep these as names so the mapping stays tied to
    # physical sensors rather than to their raw tensor slots.
    beaver_wrap_j3_sensors: tuple[str, ...] = ("01", "02")
    beaver_wrap_j4_sensors: tuple[str, ...] = ("10", "11")
    beaver_wrap_proximity_scales_mm: tuple[float, ...] = (50.0, 150.0, 300.0)
    beaver_wrap_near_threshold_mm: float = 10.0
    # Kept separate from the binary near threshold so near=0 remains valid.
    beaver_wrap_closing_scale_mm: float = 10.0
    beaver_wrap_range_scale_mm: float = 300.0
    beaver_wrap_lift_min_wrap: float = 0.8
    # New deployments can stop J3 and J4 independently.  None preserves the
    # legacy single-threshold value below for old checkpoints/configs.
    beaver_wrap_stop_close_j3_wrap: float | None = None
    beaver_wrap_stop_close_j4_wrap: float | None = None
    # Deprecated compatibility field.  It is used as the fallback for both
    # joint-specific thresholds when they are not supplied.
    beaver_wrap_stop_close_wrap: float = 1.0
    beaver_wrap_contact_stop_mm: float = 5.0
    # Consecutive frames both jaws stay enclosed before freeze / lift.
    # 0 keeps the instantaneous threshold tests. 24 Hz demos plateau about
    # 1 s after 01 and 10 enter the 10 mm bin, and lift about 1 s later.
    beaver_wrap_stop_hold_frames: int = 0
    beaver_wrap_lift_hold_frames: int = 0

    # Beaver-only execution monitors. The primary monitor consumes an
    # all-nine-sensor temporal window; the backup monitor is intentionally
    # restricted to current-frame Key4 zero-contact bits. State decisions use
    # the fixed logit boundary 0, so deployment has no monitor threshold knobs.
    beaver_monitor_hidden_dims: tuple[int, ...] = (128, 64)
    beaver_monitor_lag_steps: tuple[int, ...] = (0, 1, 3, 6, 11)
    beaver_monitor_proximity_scales_mm: tuple[float, ...] = (50.0, 150.0, 300.0)
    beaver_monitor_range_scale_mm: float = 300.0
    beaver_monitor_dropout: float = 0.05
    beaver_monitor_backup_hidden_dim: int = 16

    # Shared visual encoder and conditional 1D U-Net settings.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] = (128, 128)
    crop_ratio: float = 0.9
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    # Diffusion Policy settings used by the original three baselines.
    diffusion_step_embed_dim: int = 128
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample_range: float = 1.0

    # Conditional flow-matching settings used by the three new baselines.
    flow_time_embed_dim: int = 128
    flow_time_embedding_scale: float = 100.0
    flow_num_inference_steps: int = 10

    def __post_init__(self) -> None:
        self.beaver_shape = tuple(self.beaver_shape)
        self.resize_shape = tuple(self.resize_shape)
        self.down_dims = tuple(self.down_dims)
        self.beaver_key_sensor_indices = tuple(self.beaver_key_sensor_indices)
        self.closure_joint_mask = tuple(
            float(value) for value in self.closure_joint_mask
        )
        self.beaver_temporal_sensors = tuple(
            str(name) for name in self.beaver_temporal_sensors
        )
        self.beaver_delta_sensors = tuple(
            str(name) for name in self.beaver_delta_sensors
        )
        self.beaver_adaptive_sensors = tuple(
            str(name) for name in self.beaver_adaptive_sensors
        )
        self.beaver_adaptive_lag_steps = tuple(self.beaver_adaptive_lag_steps)
        self.beaver_adaptive_proximity_scales_mm = tuple(
            self.beaver_adaptive_proximity_scales_mm
        )
        self.beaver_antigravity_sensors = tuple(
            str(name) for name in self.beaver_antigravity_sensors
        )
        self.beaver_antigravity_lag_steps = tuple(self.beaver_antigravity_lag_steps)
        self.beaver_antigravity_proximity_scales_mm = tuple(
            self.beaver_antigravity_proximity_scales_mm
        )
        self.beaver_grok_sensors = tuple(str(name) for name in self.beaver_grok_sensors)
        self.beaver_grok_near_scales_mm = tuple(self.beaver_grok_near_scales_mm)
        self.codex_beaver_sensors = tuple(
            str(name) for name in self.codex_beaver_sensors
        )
        self.claude_sensors = tuple(str(name) for name in self.claude_sensors)
        self.claude_lag_steps = tuple(self.claude_lag_steps)
        self.claude_proximity_scales_mm = tuple(self.claude_proximity_scales_mm)
        self.beaver_wrap_sensors = tuple(str(name) for name in self.beaver_wrap_sensors)
        self.beaver_wrap_j3_sensors = tuple(
            str(name) for name in self.beaver_wrap_j3_sensors
        )
        self.beaver_wrap_j4_sensors = tuple(
            str(name) for name in self.beaver_wrap_j4_sensors
        )
        self.beaver_wrap_proximity_scales_mm = tuple(
            self.beaver_wrap_proximity_scales_mm
        )
        self.beaver_monitor_hidden_dims = tuple(self.beaver_monitor_hidden_dims)
        self.beaver_monitor_lag_steps = tuple(self.beaver_monitor_lag_steps)
        self.beaver_monitor_proximity_scales_mm = tuple(
            self.beaver_monitor_proximity_scales_mm
        )


@dataclass
class RDPConfig:
    """Asymmetric-tokenizer and slow latent-DP settings."""

    action_horizon: int = 32
    downsample_ratio: int = 2
    latent_dim: int = 4
    tokenizer_hidden_dim: int = 32
    tokenizer_layers: int = 1
    beaver_feature_dim: int = 32
    # Optional physical sensor subset for the fast Beaver-conditioned decoder.
    # None preserves the original all-nine-sensor RDP-like baseline.
    beaver_sensors: tuple[str, ...] | None = None
    kl_weight: float = 1e-6
    slow_observation_stride: int = 2
    slow_replan_steps: int = 8
    latent_down_dims: tuple[int, ...] = (256, 512, 1024)
    latent_kernel_size: int = 3
    latent_num_inference_steps: int = 100
    latent_noise_scheduler_type: str = "DDPM"
    tokenizer_epochs: int = 601
    tokenizer_max_steps: int | None = None
    tokenizer_batch_size: int = 64
    tokenizer_learning_rate: float = 1e-3
    tokenizer_weight_decay: float = 1e-4
    latent_epochs: int = 401
    latent_max_steps: int | None = None
    latent_batch_size: int = 64
    latent_resume_from: str | None = None
    tokenizer_checkpoint: str | None = None

    def __post_init__(self) -> None:
        self.latent_down_dims = tuple(self.latent_down_dims)
        if self.beaver_sensors is not None:
            self.beaver_sensors = tuple(str(name) for name in self.beaver_sensors)

    @property
    def latent_horizon(self) -> int:
        return self.action_horizon // self.downsample_ratio


@dataclass
class RFMConfig:
    """Asymmetric-tokenizer and slow latent-flow-matching settings."""

    action_horizon: int = 32
    # RDP reference: horizon 32 / downsample 2 = 16 latent tokens, each
    # token covering 2 actions (RDP at/downsampled_input_h = 16).
    downsample_ratio: int = 2
    # RDP reference: n_latent_dims = 4.
    latent_dim: int = 4
    # RDP reference: rnn_latent_dims = 32.
    tokenizer_hidden_dim: int = 32
    tokenizer_layers: int = 1
    # Beaver encoder output; matched to the tokenizer's 32-D hidden size.
    beaver_feature_dim: int = 32
    kl_weight: float = 1e-6
    slow_observation_stride: int = 2  # 24 Hz source -> 12 Hz observations
    # How many fast control ticks each latent chunk is used for before the
    # slow latent FM samples a fresh one. 8 ticks at 24 Hz ~= 3 Hz, matching
    # the FM variants' n_action_steps=8 receding horizon. At downsample 2, one
    # replan consumes tokens 0-3 of the 16-token chunk.
    slow_replan_steps: int = 8

    # Conditional 1D U-Net and flow integration settings for latent sequences.
    latent_down_dims: tuple[int, ...] = (256, 512, 1024)
    latent_kernel_size: int = 3
    latent_num_inference_steps: int = 10

    # The tokenizer and slow flow stages retain the original training lengths.
    # batch size 64, with max_train_steps null (every epoch runs the full
    # dataloader).
    tokenizer_epochs: int = 601
    tokenizer_max_steps: int | None = None
    tokenizer_batch_size: int = 64
    tokenizer_learning_rate: float = 1e-3
    tokenizer_weight_decay: float = 1e-4
    latent_epochs: int = 401
    latent_max_steps: int | None = None
    latent_batch_size: int = 64
    # Resume the latent-FM stage from a "latent_fm" checkpoint and keep
    # training. latent_epochs is the new TOTAL epoch count (resume starts
    # after the checkpoint's epoch), so extend it to the desired final step
    # count, e.g. latent_epochs=582 to reach ~100k steps at 172 steps/epoch.
    latent_resume_from: str | None = None
    tokenizer_checkpoint: str | None = None

    def __post_init__(self) -> None:
        self.latent_down_dims = tuple(self.latent_down_dims)

    @property
    def latent_horizon(self) -> int:
        return self.action_horizon // self.downsample_ratio


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/realman_beaver"
    device: str = "auto"
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 200
    max_steps: int | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple[float, float] = (0.95, 0.999)
    eps: float = 1e-8
    warmup_steps: int = 500
    gradient_clip_norm: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999
    log_every_steps: int = 20
    checkpoint_every_steps: int = 25_000
    resume_from: str | None = None
    # W&B visualization. Direct policy trainers use shared metric names
    # (train_loss/val_loss/lr/epoch/global_step) so runs can be overlaid.
    # Reactive trainers retain stage-local metrics for their two optimizers.
    # Leave None to train without W&B.
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)


@dataclass
class RealmanBeaverConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rdp: RDPConfig = field(default_factory=RDPConfig)
    rfm: RFMConfig = field(default_factory=RFMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RealmanBeaverConfig:
        allowed = {"dataset", "model", "rdp", "rfm", "training"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        config = cls(
            dataset=DatasetConfig(**raw.get("dataset", {})),
            model=ModelConfig(**raw.get("model", {})),
            rdp=RDPConfig(**raw.get("rdp", {})),
            rfm=RFMConfig(**raw.get("rfm", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        dataset, model, rdp, rfm, training = (
            self.dataset,
            self.model,
            self.rdp,
            self.rfm,
            self.training,
        )
        if model.variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"model.variant must be one of {sorted(SUPPORTED_VARIANTS)}"
            )
        if dataset.fps <= 0 or not 0.0 <= dataset.val_fraction < 1.0:
            raise ValueError(
                "dataset fps must be positive and val_fraction must be in [0, 1)"
            )
        if dataset.val_episodes is not None:
            if any(episode < 0 for episode in dataset.val_episodes):
                raise ValueError(
                    "dataset.val_episodes must contain non-negative indices"
                )
            if len(set(dataset.val_episodes)) != len(dataset.val_episodes):
                raise ValueError("dataset.val_episodes must not contain duplicates")
        if len(dataset.image_shape) != 3 or dataset.image_shape[0] != 3:
            raise ValueError(
                "dataset.image_shape must be channel-first RGB, for example (3, 480, 640)"
            )
        if dataset.normalization_source not in {"parquet", "metadata"}:
            raise ValueError("normalization_source must be 'parquet' or 'metadata'")
        if dataset.distance_max_mm <= 0 or dataset.normalization_floor <= 0:
            raise ValueError("distance_max_mm and normalization_floor must be positive")
        if (model.state_dim, model.action_dim) != (7, 7):
            raise ValueError("the WRM policy expects 7-D state and 7-D action")
        if model.beaver_shape != (9, 4, 4):
            raise ValueError("the Beaver distance feature must have shape (9, 4, 4)")
        if len(dataset.beaver_sensor_layout) != model.beaver_shape[0]:
            raise ValueError(
                "dataset.beaver_sensor_layout must name every Beaver tensor slot"
            )
        if len(set(dataset.beaver_sensor_layout)) != len(dataset.beaver_sensor_layout):
            raise ValueError("dataset.beaver_sensor_layout must contain unique names")
        if model.n_obs_steps <= 0 or model.horizon <= 0:
            raise ValueError("n_obs_steps and horizon must be positive")
        if not 1 <= model.n_action_steps <= model.horizon - model.n_obs_steps + 1:
            raise ValueError(
                "n_action_steps exceeds the LeRobot receding-horizon window"
            )
        if model.horizon % (2 ** len(model.down_dims)):
            raise ValueError(
                "policy horizon must be divisible by the U-Net downsampling factor"
            )
        if model.variant in {
            "original_dp",
            "dp_beaver",
            BEAVER_CLOSURE_VARIANT,
            *STRUCTURED_BEAVER_DP_VARIANTS,
            TEMPORAL_BEAVER_VARIANT,
            DELTA_BEAVER_VARIANT,
            ADAPTIVE_BEAVER_VARIANT,
            ANTIGRAVITY_BEAVER_VARIANT,
            CODEX_BEAVER_VARIANT,
            CLAUDE_BEAVER_VARIANT,
            QWEN_BEAVER_VARIANT,
            *WRAP_BEAVER_VARIANTS,
            "rdp_like",
        }:
            if model.noise_scheduler_type not in {"DDPM", "DDIM"}:
                raise ValueError("noise_scheduler_type must be DDPM or DDIM")
            if model.prediction_type not in {"epsilon", "sample"}:
                raise ValueError("prediction_type must be epsilon or sample")
            if not 1 <= model.num_inference_steps <= model.num_train_timesteps:
                raise ValueError(
                    "num_inference_steps must be within the diffusion training schedule"
                )
        if model.variant == BEAVER_CLOSURE_VARIANT:
            if model.n_obs_steps < 2:
                raise ValueError("dp_beaver_closure requires n_obs_steps >= 2")
            dimensions = (
                model.closure_beaver_encoder_dim,
                model.closure_sensor_hidden_dim,
                model.closure_hidden_dim,
            )
            if any(value <= 0 for value in dimensions):
                raise ValueError("closure encoder dimensions must be positive")
            if len(model.closure_joint_mask) != model.action_dim:
                raise ValueError("closure_joint_mask must contain one value per joint")
            if any(value < 0.0 or value > 1.0 for value in model.closure_joint_mask):
                raise ValueError("closure_joint_mask values must be in [0, 1]")
            if not any(model.closure_joint_mask):
                raise ValueError("closure_joint_mask must enable at least one joint")
            if model.closure_residual_scale <= 0:
                raise ValueError("closure_residual_scale must be positive")
            if (
                min(
                    model.closure_grasp_loss_weight,
                    model.closure_residual_loss_weight,
                )
                < 0
            ):
                raise ValueError("closure auxiliary-loss weights must be non-negative")
            if not dataset.beaver_valid_statuses:
                raise ValueError("dp_beaver_closure requires beaver_valid_statuses")
        if model.variant in STRUCTURED_BEAVER_DP_VARIANTS:
            if (
                model.beaver_feature_dim <= 0
                or model.beaver_sensor_hidden_dim <= 0
                or model.beaver_sensor_feature_dim <= 0
            ):
                raise ValueError(
                    "structured Beaver feature and sensor dimensions must be positive"
                )
            if not dataset.beaver_valid_statuses:
                raise ValueError(
                    "structured Beaver policies require beaver_valid_statuses"
                )
        if model.variant == TEMPORAL_BEAVER_VARIANT:
            if (
                model.beaver_history_steps <= 0
                or model.beaver_temporal_feature_dim <= 0
                or model.beaver_frame_hidden_dim <= 0
                or model.beaver_frame_feature_dim <= 0
                or model.beaver_temporal_hidden_dim <= 0
            ):
                raise ValueError(
                    "temporal Beaver history and dimensions must be positive"
                )
            if model.beaver_grasp_loss_weight < 0:
                raise ValueError("beaver_grasp_loss_weight must be non-negative")
            if model.beaver_temporal_feature_dim != 64:
                raise ValueError("WRM_temporal requires beaver_temporal_feature_dim=64")
            selected = model.beaver_temporal_sensors
            if not selected or len(set(selected)) != len(selected):
                raise ValueError(
                    "beaver_temporal_sensors must contain at least one unique name"
                )
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "beaver_temporal_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_temporal requires beaver_valid_statuses")
            if model.beaver_history_mode not in {"temporal", "current"}:
                raise ValueError(
                    "beaver_history_mode must be 'temporal' or 'current'"
                )
            if not model.use_beaver_condition and model.beaver_grasp_loss_weight:
                raise ValueError(
                    "use_beaver_condition=false requires "
                    "beaver_grasp_loss_weight=0 for a clean modality ablation"
                )
            if (
                not model.use_beaver_condition
                and model.condition_on_grasp_probability
            ):
                raise ValueError(
                    "use_beaver_condition=false requires "
                    "condition_on_grasp_probability=false"
                )
        if model.variant in WRAP_BEAVER_VARIANTS:
            if (
                model.beaver_wrap_history_steps <= 0
                or model.beaver_wrap_feature_dim != 64
                or model.beaver_wrap_enclosure_dim != 4
                or model.beaver_wrap_frame_hidden_dim <= 0
                or model.beaver_wrap_frame_feature_dim <= 0
                or model.beaver_wrap_temporal_hidden_dim <= 0
            ):
                raise ValueError("WRM_wrap history and feature dimensions are invalid")
            selected = model.beaver_wrap_sensors
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError(
                    "beaver_wrap_sensors must contain exactly four unique names"
                )
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "beaver_wrap_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            selected_set = set(selected)
            for joint_name, joint_sensors in (
                ("J3", model.beaver_wrap_j3_sensors),
                ("J4", model.beaver_wrap_j4_sensors),
            ):
                if not joint_sensors or len(set(joint_sensors)) != len(joint_sensors):
                    raise ValueError(
                        f"beaver_wrap_{joint_name.lower()}_sensors must be non-empty "
                        "and contain unique names"
                    )
                unknown = sorted(set(joint_sensors) - selected_set)
                if unknown:
                    raise ValueError(
                        f"beaver_wrap_{joint_name.lower()}_sensors must be a subset "
                        f"of beaver_wrap_sensors; unknown: {unknown}"
                    )
            if set(model.beaver_wrap_j3_sensors) & set(model.beaver_wrap_j4_sensors):
                raise ValueError("J3 and J4 wrap sensor groups must be disjoint")
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_wrap requires beaver_valid_statuses")
            if model.beaver_wrap_near_threshold_mm < 0:
                raise ValueError(
                    "beaver_wrap_near_threshold_mm must be non-negative"
                )
            if model.beaver_wrap_closing_scale_mm <= 0:
                raise ValueError(
                    "beaver_wrap_closing_scale_mm must be positive"
                )
            if model.beaver_wrap_range_scale_mm <= 0:
                raise ValueError("beaver_wrap_range_scale_mm must be positive")
            if not 0.0 < model.beaver_wrap_lift_min_wrap <= 1.0:
                raise ValueError("beaver_wrap_lift_min_wrap must be in (0, 1]")
            for name, value in (
                ("beaver_wrap_stop_close_j3_wrap", model.beaver_wrap_stop_close_j3_wrap),
                ("beaver_wrap_stop_close_j4_wrap", model.beaver_wrap_stop_close_j4_wrap),
                ("beaver_wrap_stop_close_wrap", model.beaver_wrap_stop_close_wrap),
            ):
                if value is not None and not 0.0 < value <= 1.0:
                    raise ValueError(f"{name} must be in (0, 1]")
            if model.beaver_wrap_contact_stop_mm < 0:
                raise ValueError(
                    "beaver_wrap_contact_stop_mm must be non-negative"
                )
            for name, value in (
                ("beaver_wrap_stop_hold_frames", model.beaver_wrap_stop_hold_frames),
                ("beaver_wrap_lift_hold_frames", model.beaver_wrap_lift_hold_frames),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer")
            if model.beaver_wrap_lift_hold_frames < model.beaver_wrap_stop_hold_frames:
                raise ValueError(
                    "beaver_wrap_lift_hold_frames must be >= "
                    "beaver_wrap_stop_hold_frames"
                )
            if any(scale <= 0 for scale in model.beaver_wrap_proximity_scales_mm):
                raise ValueError("beaver_wrap_proximity_scales_mm must be positive")
            if model.variant in WRAP_MONITOR_BEAVER_VARIANTS:
                if not model.beaver_monitor_hidden_dims or any(
                    width <= 0 for width in model.beaver_monitor_hidden_dims
                ):
                    raise ValueError("beaver_monitor_hidden_dims must be positive")
                lags = model.beaver_monitor_lag_steps
                if not lags or lags[0] != 0 or any(lag < 0 for lag in lags):
                    raise ValueError(
                        "beaver_monitor_lag_steps must start at 0 and be non-negative"
                    )
                if max(lags) >= model.beaver_wrap_history_steps:
                    raise ValueError(
                        "beaver_monitor_lag_steps must fit beaver_wrap_history_steps"
                    )
                if any(
                    scale <= 0
                    for scale in model.beaver_monitor_proximity_scales_mm
                ):
                    raise ValueError(
                        "beaver_monitor_proximity_scales_mm must be positive"
                    )
                if model.beaver_monitor_range_scale_mm <= 0:
                    raise ValueError("beaver_monitor_range_scale_mm must be positive")
                if not 0.0 <= model.beaver_monitor_dropout < 1.0:
                    raise ValueError("beaver_monitor_dropout must be in [0, 1)")
                if model.beaver_monitor_backup_hidden_dim <= 0:
                    raise ValueError(
                        "beaver_monitor_backup_hidden_dim must be positive"
                    )
            if dataset.stratified_bottle_batch:
                if dataset.episodes_per_bottle <= 0:
                    raise ValueError("episodes_per_bottle must be positive")
                if training.batch_size < 5:
                    raise ValueError(
                        "WRM_wrap stratified batches need batch_size >= 5 bottles"
                    )
        if model.variant == DELTA_BEAVER_VARIANT:
            dimensions = {
                "beaver_delta_steps": model.beaver_delta_steps,
                "beaver_delta_feature_dim": model.beaver_delta_feature_dim,
                "beaver_delta_sensor_hidden_dim": model.beaver_delta_sensor_hidden_dim,
                "beaver_delta_sensor_feature_dim": model.beaver_delta_sensor_feature_dim,
                "beaver_delta_fusion_hidden_dim": model.beaver_delta_fusion_hidden_dim,
                "beaver_delta_grasp_hidden_dim": model.beaver_delta_grasp_hidden_dim,
            }
            if any(value <= 0 for value in dimensions.values()):
                raise ValueError(f"WRM_delta dimensions must be positive: {dimensions}")
            if model.beaver_delta_feature_dim != 64:
                raise ValueError("WRM_delta requires beaver_delta_feature_dim=64")
            selected = model.beaver_delta_sensors
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError(
                    "beaver_delta_sensors must contain exactly four unique names"
                )
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "beaver_delta_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_delta requires beaver_valid_statuses")
            if model.lambda_grasp < 0:
                raise ValueError("lambda_grasp must be non-negative")
        if model.variant == ADAPTIVE_BEAVER_VARIANT:
            dimensions = {
                "beaver_adaptive_history_steps": model.beaver_adaptive_history_steps,
                "beaver_adaptive_motion_delta_steps": (
                    model.beaver_adaptive_motion_delta_steps
                ),
                "beaver_adaptive_feature_dim": model.beaver_adaptive_feature_dim,
                "beaver_adaptive_sensor_hidden_dim": (
                    model.beaver_adaptive_sensor_hidden_dim
                ),
                "beaver_adaptive_token_dim": model.beaver_adaptive_token_dim,
                "beaver_adaptive_transformer_layers": (
                    model.beaver_adaptive_transformer_layers
                ),
                "beaver_adaptive_attention_heads": (
                    model.beaver_adaptive_attention_heads
                ),
                "beaver_adaptive_grasp_hidden_dim": (
                    model.beaver_adaptive_grasp_hidden_dim
                ),
            }
            if any(value <= 0 for value in dimensions.values()):
                raise ValueError(
                    f"WRM_adaptive dimensions must be positive: {dimensions}"
                )
            selected = model.beaver_adaptive_sensors
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError(
                    "beaver_adaptive_sensors must name four unique reliable sensors"
                )
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "beaver_adaptive_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            if not model.beaver_adaptive_lag_steps or any(
                lag <= 0 or lag >= model.beaver_adaptive_history_steps
                for lag in model.beaver_adaptive_lag_steps
            ):
                raise ValueError(
                    "adaptive lag steps must be positive and shorter than history"
                )
            if len(set(model.beaver_adaptive_lag_steps)) != len(
                model.beaver_adaptive_lag_steps
            ):
                raise ValueError("adaptive lag steps must be unique")
            if not model.beaver_adaptive_proximity_scales_mm or any(
                scale <= 0 for scale in model.beaver_adaptive_proximity_scales_mm
            ):
                raise ValueError("adaptive proximity scales must be positive")
            if model.beaver_adaptive_token_dim % model.beaver_adaptive_attention_heads:
                raise ValueError(
                    "adaptive token dimension must be divisible by attention heads"
                )
            if model.beaver_adaptive_grasp_loss_weight < 0:
                raise ValueError("adaptive grasp loss weight must be non-negative")
            if model.beaver_adaptive_noise_std_mm < 0:
                raise ValueError("adaptive distance noise cannot be negative")
            for name, probability in {
                "pixel_dropout": model.beaver_adaptive_pixel_dropout,
                "sensor_dropout": model.beaver_adaptive_sensor_dropout,
            }.items():
                if not 0.0 <= probability < 1.0:
                    raise ValueError(f"adaptive {name} must be in [0, 1)")
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_adaptive requires beaver_valid_statuses")
        if model.variant == ANTIGRAVITY_BEAVER_VARIANT:
            dimensions = (
                model.beaver_antigravity_history_steps,
                model.beaver_antigravity_motion_delta_steps,
                model.beaver_antigravity_motion_delta_long_steps,
                model.beaver_antigravity_spatial_hidden_dim,
                model.beaver_antigravity_token_dim,
                model.beaver_antigravity_temporal_hidden_dim,
                model.beaver_antigravity_transformer_layers,
                model.beaver_antigravity_attention_heads,
                model.beaver_antigravity_feature_dim,
                model.beaver_antigravity_grasp_hidden_dim,
            )
            if any(value <= 0 for value in dimensions):
                raise ValueError("WRM_antigravity dimensions must be positive")
            selected = model.beaver_antigravity_sensors
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError(
                    "beaver_antigravity_sensors must name four unique reliable sensors"
                )
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "beaver_antigravity_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            if not model.beaver_antigravity_lag_steps or any(
                lag <= 0 or lag >= model.beaver_antigravity_history_steps
                for lag in model.beaver_antigravity_lag_steps
            ):
                raise ValueError(
                    "antigravity lag steps must be positive and shorter than history"
                )
            if len(set(model.beaver_antigravity_lag_steps)) != len(
                model.beaver_antigravity_lag_steps
            ):
                raise ValueError("antigravity lag steps must be unique")
            if not model.beaver_antigravity_proximity_scales_mm or any(
                scale <= 0 for scale in model.beaver_antigravity_proximity_scales_mm
            ):
                raise ValueError("antigravity proximity scales must be positive")
            if (
                model.beaver_antigravity_token_dim
                % model.beaver_antigravity_attention_heads
            ):
                raise ValueError(
                    "antigravity token dimension must be divisible by attention heads"
                )
            for weight in (
                model.beaver_antigravity_grasp_loss_weight,
                model.beaver_antigravity_enclosure_loss_weight,
                model.beaver_antigravity_noise_std_mm,
            ):
                if weight < 0:
                    raise ValueError("antigravity losses/noise must be non-negative")
            for probability in (
                model.beaver_antigravity_pixel_dropout,
                model.beaver_antigravity_sensor_dropout,
            ):
                if not 0.0 <= probability < 1.0:
                    raise ValueError("antigravity dropout must be in [0, 1)")
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_antigravity requires beaver_valid_statuses")
        if model.variant == GROK_BEAVER_VARIANT:
            dimensions = (
                model.beaver_grok_history_steps,
                model.beaver_grok_motion_delta_steps,
                model.beaver_grok_feature_dim,
                model.beaver_grok_enclosure_dim,
                model.beaver_grok_frame_hidden_dim,
                model.beaver_grok_frame_feature_dim,
                model.beaver_grok_temporal_hidden_dim,
                model.beaver_grok_phase_hidden_dim,
                model.beaver_grok_min_near_sensors,
            )
            if any(value <= 0 for value in dimensions):
                raise ValueError("WRM_grok dimensions must be positive")
            if model.beaver_grok_enclosure_dim != 16:
                raise ValueError("WRM_grok requires beaver_grok_enclosure_dim=16")
            selected = model.beaver_grok_sensors
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError("beaver_grok_sensors must name four unique sensors")
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "beaver_grok_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            if model.beaver_grok_wrap_threshold_mm <= 0 or any(
                scale <= 0 for scale in model.beaver_grok_near_scales_mm
            ):
                raise ValueError("WRM_grok distance scales must be positive")
            for value in (
                model.beaver_grok_phase_loss_weight,
                model.beaver_grok_smooth_loss_weight,
                model.beaver_grok_hold_loss_weight,
                model.beaver_grok_noise_std_mm,
            ):
                if value < 0:
                    raise ValueError("WRM_grok losses/noise must be non-negative")
            if model.beaver_grok_overlap_blend_steps < 0:
                raise ValueError("beaver_grok_overlap_blend_steps cannot be negative")
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_grok requires beaver_valid_statuses")
        if model.variant == CODEX_BEAVER_VARIANT:
            dimensions = (
                model.codex_beaver_history_steps,
                model.codex_token_dim,
                model.codex_vision_width,
                model.codex_contact_hidden_dim,
                model.codex_fusion_layers,
                model.codex_sensor_layers,
                model.codex_decoder_layers,
                model.codex_attention_heads,
                model.codex_plan_ensemble,
            )
            if any(value <= 0 for value in dimensions):
                raise ValueError("WRM_codex dimensions must be positive")
            selected = model.codex_beaver_sensors
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError(
                    "codex_beaver_sensors must name exactly four unique reliable sensors"
                )
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "codex_beaver_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            if model.codex_token_dim % model.codex_attention_heads:
                raise ValueError(
                    "codex_token_dim must be divisible by codex_attention_heads"
                )
            if not 0.0 <= model.codex_dropout < 1.0:
                raise ValueError("codex_dropout must be in [0, 1)")
            if any(
                value <= 0
                for value in (
                    model.codex_residual_scale,
                    model.codex_activity_scale_rad,
                    model.codex_plan_decay,
                )
            ):
                raise ValueError("WRM_codex scales must be positive")
            if (
                min(
                    model.codex_activity_loss_weight,
                    model.codex_velocity_loss_weight,
                )
                < 0
            ):
                raise ValueError("WRM_codex loss weights must be non-negative")
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_codex requires beaver_valid_statuses")
        if model.variant == CLAUDE_BEAVER_VARIANT:
            if model.n_obs_steps != 2:
                raise ValueError("WRM_claude requires n_obs_steps=2")
            dimensions = (
                model.claude_motion_delta_steps,
                model.claude_history_steps,
                model.claude_feature_dim,
                model.claude_sensor_hidden_dim,
                model.claude_token_dim,
                model.claude_transformer_layers,
                model.claude_attention_heads,
                model.claude_grasp_hidden_dim,
            )
            if any(value <= 0 for value in dimensions):
                raise ValueError("WRM_claude dimensions must be positive")
            if model.claude_feature_dim != 64:
                raise ValueError("WRM_claude requires claude_feature_dim=64")
            selected = model.claude_sensors
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError(
                    "claude_sensors must name four unique reliable sensors"
                )
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "claude_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            if not model.claude_lag_steps or any(
                lag <= 0 or lag >= model.claude_history_steps
                for lag in model.claude_lag_steps
            ):
                raise ValueError(
                    "claude lag steps must be positive and shorter than history"
                )
            if len(set(model.claude_lag_steps)) != len(model.claude_lag_steps):
                raise ValueError("claude lag steps must be unique")
            if not model.claude_proximity_scales_mm or any(
                scale <= 0 for scale in model.claude_proximity_scales_mm
            ):
                raise ValueError("claude proximity scales must be positive")
            if model.claude_token_dim % model.claude_attention_heads:
                raise ValueError(
                    "claude token dimension must be divisible by attention heads"
                )
            if (
                min(
                    model.claude_grasp_loss_weight,
                    model.claude_smoothness_loss_weight,
                    model.claude_noise_std_mm,
                )
                < 0
            ):
                raise ValueError("claude losses/noise must be non-negative")
            for probability in (
                model.claude_pixel_dropout,
                model.claude_sensor_dropout,
            ):
                if not 0.0 <= probability < 1.0:
                    raise ValueError("claude dropout must be in [0, 1)")
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_claude requires beaver_valid_statuses")
        if model.variant == QWEN_BEAVER_VARIANT:
            dimensions = (
                model.beaver_history_steps,
                model.beaver_temporal_feature_dim,
                model.beaver_frame_hidden_dim,
                model.beaver_frame_feature_dim,
                model.beaver_temporal_hidden_dim,
                model.qwen_joint_history_steps,
                model.qwen_grasp_hidden_dim,
            )
            if any(value <= 0 for value in dimensions):
                raise ValueError(
                    "WRM_qwen Beaver history, joint history, and phase-head "
                    "dimensions must be positive"
                )
            if model.qwen_joint_history_steps < 2:
                raise ValueError("qwen_joint_history_steps must be at least two")
            if model.qwen_joint_history_steps > model.beaver_history_steps:
                raise ValueError(
                    "qwen_joint_history_steps cannot exceed Beaver history length"
                )
            if model.beaver_temporal_feature_dim != 64:
                raise ValueError("WRM_qwen requires beaver_temporal_feature_dim=64")
            if model.qwen_grasp_loss_weight < 0:
                raise ValueError("qwen_grasp_loss_weight must be non-negative")
            selected = model.beaver_temporal_sensors
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError(
                    "beaver_temporal_sensors must contain exactly four unique names"
                )
            unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
            if unknown:
                raise ValueError(
                    "beaver_temporal_sensors contains names absent from "
                    f"dataset.beaver_sensor_layout: {unknown}"
                )
            if not dataset.beaver_valid_statuses:
                raise ValueError("WRM_qwen requires beaver_valid_statuses")
        if model.variant in KEY4_BEAVER_DP_VARIANTS:
            if len(model.beaver_key_sensor_indices) != 4:
                raise ValueError("Key4 policies require exactly four sensor indices")
            if len(set(model.beaver_key_sensor_indices)) != 4 or any(
                index < 0 or index >= model.beaver_shape[0]
                for index in model.beaver_key_sensor_indices
            ):
                raise ValueError(
                    "beaver_key_sensor_indices must contain four unique valid slots"
                )
            expected_feature_dim = (
                4 * model.beaver_sensor_feature_dim
                if model.variant == "dp_beaver_key4"
                else 4 * model.beaver_pca_components
            )
            if model.beaver_feature_dim != expected_feature_dim:
                raise ValueError(
                    f"{model.variant} requires beaver_feature_dim="
                    f"{expected_feature_dim}, got {model.beaver_feature_dim}"
                )
            if not 1 <= model.beaver_pca_components <= 32:
                raise ValueError("beaver_pca_components must be in [1, 32]")
        if (
            model.variant in {"dp_beaver_near", "dp_beaver_near_gate"}
            and model.beaver_near_threshold_mm <= 0
        ):
            raise ValueError("beaver_near_threshold_mm must be positive")
        if model.variant == "dp_beaver_near_gate" and model.beaver_gate_hidden_dim <= 0:
            raise ValueError("beaver_gate_hidden_dim must be positive")
        if model.variant in {"fm", "fm_beaver", "rfm", GROK_BEAVER_VARIANT}:
            if model.flow_time_embed_dim <= 0 or model.flow_time_embedding_scale <= 0:
                raise ValueError(
                    "flow time embedding dimensions and scale must be positive"
                )
            if model.flow_num_inference_steps <= 0:
                raise ValueError("flow_num_inference_steps must be positive")
        if model.variant == "rdp_like":
            self._validate_reactive(rdp, "RDP")
            if rdp.beaver_sensors is not None:
                selected = rdp.beaver_sensors
                if not selected or len(set(selected)) != len(selected):
                    raise ValueError(
                        "rdp.beaver_sensors must contain unique physical sensor names"
                    )
                unknown = sorted(set(selected) - set(dataset.beaver_sensor_layout))
                if unknown:
                    raise ValueError(
                        "rdp.beaver_sensors contains names absent from "
                        f"dataset.beaver_sensor_layout: {unknown}"
                    )
            if rdp.latent_noise_scheduler_type not in {"DDPM", "DDIM"}:
                raise ValueError("latent_noise_scheduler_type must be DDPM or DDIM")
        if model.variant == "rfm":
            self._validate_reactive(rfm, "RFM")
            if rfm.latent_num_inference_steps <= 0:
                raise ValueError("latent_num_inference_steps must be positive")
        if training.batch_size <= 0 or training.epochs <= 0:
            raise ValueError("training batch_size and epochs must be positive")
        if training.max_steps is not None and training.max_steps <= 0:
            raise ValueError("training.max_steps must be positive when set")
        if training.checkpoint_every_steps <= 0:
            raise ValueError("training.checkpoint_every_steps must be positive")

    @staticmethod
    def _validate_reactive(config: RDPConfig | RFMConfig, name: str) -> None:
        if config.action_horizon % config.downsample_ratio:
            raise ValueError(
                f"{name} action_horizon must be divisible by downsample_ratio"
            )
        if config.downsample_ratio <= 0 or config.downsample_ratio & (
            config.downsample_ratio - 1
        ):
            raise ValueError(f"{name} downsample_ratio must be a positive power of two")
        if config.latent_horizon % (2 ** len(config.latent_down_dims)):
            raise ValueError(
                "latent horizon must be divisible by the latent U-Net downsampling factor"
            )
        if not 1 <= config.slow_replan_steps <= config.action_horizon:
            raise ValueError("slow_replan_steps must be in [1, action_horizon]")
        if config.kl_weight < 0 or config.latent_dim <= 0:
            raise ValueError("kl_weight must be non-negative and latent_dim positive")


def load_config(path: str | Path) -> RealmanBeaverConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError(f"Config must contain a mapping: {config_path}")
    return RealmanBeaverConfig.from_dict(raw)
