"""Configuration for the local ForceFlow++ policy.

This follows LeRobot's policy layout: a ``PreTrainedConfig`` subclass registered
under a policy type string, plus feature-delta properties used by the training
pipeline to sample temporal observations and action chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("forceflowpp")
@dataclass
class ForceFlowPPConfig(PreTrainedConfig):
    """Force/tactile-conditioned flow matching policy.

    The first implementation intentionally targets the three-month ForceFlow++
    version: AdaLN-Zero modulation, flow matching, and a contact-aware adaptive
    prior. ReGuide, Joint LAM, and residual correction are left out of this
    policy.
    """

    # I/O temporal structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # Flow matching.
    sigma_min: float = 0.0
    num_integration_steps: int = 10
    integration_method: str = "euler"
    timestep_sampling_strategy: str = "uniform"
    timestep_sampling_alpha: float = 1.5
    timestep_sampling_beta: float = 1.0
    timestep_sampling_s: float = 0.999

    # Observation encoders.
    hidden_dim: int = 512
    image_embedding_dim: int = 256
    image_resize_shape: tuple[int, int] | None = (224, 224)
    state_feature_dim: int = 256
    force_tactile_feature_dim: int = 512
    encoder_dropout: float = 0.1
    use_force: bool = True
    use_torque: bool = True
    use_tactile: bool = True
    force_key: str = "observation.force"
    torque_key: str = "observation.torque"
    tactile_key: str = "observation.tactile"

    # DiT backbone.
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    timestep_embed_dim: int = 256
    use_positional_encoding: bool = True
    use_adaln_modulation: bool = True

    # Contact-aware prior.
    use_adaptive_prior: bool = True
    num_contact_modes: int = 4
    free_force_threshold_n: float = 0.5
    heavy_force_threshold_n: float = 2.0
    transition_force_delta_n: float = 1.0
    prior_aux_loss_weight: float = 0.1
    prior_kl_weight: float = 1e-4
    prior_std_init: float = 1.0
    prior_std_floor: float = 0.03

    # Training.
    do_mask_loss_for_padding: bool = False
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    drop_n_last_frames: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1
        self._validate()

    def _validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive.")
        if self.n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive.")
        max_action_steps = self.horizon - self.n_obs_steps + 1
        if self.n_action_steps > max_action_steps:
            raise ValueError(
                "n_action_steps must be <= horizon - n_obs_steps + 1 "
                f"({max_action_steps}), got {self.n_action_steps}."
            )
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads.")
        if self.image_embedding_dim <= 0 or self.image_embedding_dim % 8 != 0:
            raise ValueError("image_embedding_dim must be positive and divisible by 8.")
        if self.state_feature_dim <= 0 or self.force_tactile_feature_dim <= 0:
            raise ValueError("encoder feature dimensions must be positive.")
        if self.num_contact_modes < 1:
            raise ValueError("num_contact_modes must be at least 1.")
        if self.integration_method != "euler":
            raise ValueError("Only Euler integration is implemented for ForceFlow++ v1.")
        if self.timestep_sampling_strategy not in {"uniform", "beta"}:
            raise ValueError("timestep_sampling_strategy must be 'uniform' or 'beta'.")
        if not (0.0 <= self.sigma_min < 1.0):
            raise ValueError("sigma_min must be in [0, 1).")
        if self.prior_std_init <= 0 or self.prior_std_floor <= 0:
            raise ValueError("prior std values must be positive.")

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError("ForceFlow++ requires an 'action' output feature.")
        if not self.image_features and self.robot_state_feature is None:
            raise ValueError("ForceFlow++ requires at least an image feature or observation.state.")
        if self.robot_state_feature is not None and self.robot_state_feature.shape[0] <= 0:
            raise ValueError("observation.state must have at least one dimension.")

        optional_vector_keys = [self.force_key, self.torque_key, self.tactile_key]
        for key in optional_vector_keys:
            feature = (self.input_features or {}).get(key)
            if feature is not None and feature.type is FeatureType.VISUAL:
                raise ValueError(f"{key} must be a vector feature, not VISUAL.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def action_dim(self) -> int:
        if self.action_feature is None:
            raise ValueError("Missing action feature.")
        return int(self.action_feature.shape[0])

    @property
    def state_dim(self) -> int:
        feature = (self.input_features or {}).get(OBS_STATE)
        return int(feature.shape[0]) if feature is not None else 0

    @property
    def has_force_feature(self) -> bool:
        return self.use_force and self.force_key in (self.input_features or {})

    @property
    def has_torque_feature(self) -> bool:
        return self.use_torque and self.torque_key in (self.input_features or {})

    @property
    def has_tactile_feature(self) -> bool:
        return self.use_tactile and self.tactile_key in (self.input_features or {})

    @property
    def action_feature_name(self) -> str:
        return ACTION
