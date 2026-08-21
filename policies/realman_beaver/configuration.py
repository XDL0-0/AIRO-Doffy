"""Typed configuration for the diffusion and flow-matching baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_VARIANTS = {
    "original_dp",
    "dp_beaver",
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

    def __post_init__(self) -> None:
        self.image_shape = tuple(self.image_shape)
        self.beaver_valid_statuses = tuple(self.beaver_valid_statuses)
        if self.val_episodes is not None:
            self.val_episodes = tuple(self.val_episodes)


@dataclass
class ModelConfig:
    variant: str = "original_dp"
    state_dim: int = 7
    action_dim: int = 7
    beaver_shape: tuple[int, int, int] = (9, 4, 4)

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


@dataclass
class RDPConfig:
    """Asymmetric-tokenizer and slow latent-DP settings."""

    action_horizon: int = 32
    downsample_ratio: int = 2
    latent_dim: int = 4
    tokenizer_hidden_dim: int = 32
    tokenizer_layers: int = 1
    beaver_feature_dim: int = 32
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
    # W&B visualization. When set, wandb.init(project=wandb_project) runs
    # every stage's step/epoch metrics through one run, with keys prefixed
    # by the stage kind (tokenizer/latent_fm/fm/...). Leave None
    # to train without W&B.
    wandb_project: str | None = None
    wandb_run_name: str | None = None

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
                raise ValueError("dataset.val_episodes must contain non-negative indices")
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
        if model.variant in {"original_dp", "dp_beaver", "rdp_like"}:
            if model.noise_scheduler_type not in {"DDPM", "DDIM"}:
                raise ValueError("noise_scheduler_type must be DDPM or DDIM")
            if model.prediction_type not in {"epsilon", "sample"}:
                raise ValueError("prediction_type must be epsilon or sample")
            if not 1 <= model.num_inference_steps <= model.num_train_timesteps:
                raise ValueError(
                    "num_inference_steps must be within the diffusion training schedule"
                )
        if model.variant in {"fm", "fm_beaver", "rfm"}:
            if model.flow_time_embed_dim <= 0 or model.flow_time_embedding_scale <= 0:
                raise ValueError(
                    "flow time embedding dimensions and scale must be positive"
                )
            if model.flow_num_inference_steps <= 0:
                raise ValueError("flow_num_inference_steps must be positive")
        if model.variant == "rdp_like":
            self._validate_reactive(rdp, "RDP")
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
