"""Diffusion and conditional flow-matching policies for Realman and Beaver."""

from __future__ import annotations

from collections import deque

import einops
import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import (
    DiffusionConfig as LeRobotUnetConfig,
)
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionPolicy,
    DiffusionRgbEncoder,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from policies.realman_beaver.configuration import (
    ADAPTIVE_BEAVER_VARIANT,
    ANTIGRAVITY_BEAVER_VARIANT,
    BEAVER_CLOSURE_VARIANT,
    CLAUDE_BEAVER_VARIANT,
    CODEX_BEAVER_VARIANT,
    DELTA_BEAVER_VARIANT,
    GROK_BEAVER_VARIANT,
    KEY4_BEAVER_DP_VARIANTS,
    QWEN_BEAVER_VARIANT,
    STRUCTURED_BEAVER_DP_VARIANTS,
    TEMPORAL_BEAVER_VARIANT,
    WRAP_BEAVER_VARIANT,
    WRAP_BEAVER_VARIANTS,
    WRAP_DELTA_BEAVER_VARIANT,
    WRAP_MONITOR_BEAVER_VARIANTS,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import (
    LatentNormalizer,
    ObservationNormalizer,
    resolve_beaver_sensor_indices,
)
from policies.realman_beaver.modeling_wrm_antigravity import AntigravityDPPolicy
from policies.realman_beaver.modeling_wrm_claude import ClaudeBeaverDPPolicy
from policies.realman_beaver.modeling_wrm_qwen import QwenBeaverDPPolicy
from policies.realman_beaver.modules import (
    AdaptiveBeaverEncoder,
    AsymmetricBeaverTokenizer,
    DeltaBeaverEncoder,
    Key4BeaverEncoder,
    StructuredBeaverEncoder,
    TemporalBeaverEncoder,
)


def build_native_diffusion_config(
    config: RealmanBeaverConfig, *, latent_actions: bool = False
) -> LeRobotUnetConfig:
    """Translate the original baseline config into LeRobot's native DP config."""
    dataset, model, rdp = config.dataset, config.model, config.rdp
    if latent_actions:
        state_dim = model.state_dim
        action_dim = rdp.latent_dim
        horizon = rdp.latent_horizon
        n_action_steps = horizon - model.n_obs_steps + 1
        down_dims = rdp.latent_down_dims
        kernel_size = rdp.latent_kernel_size
        scheduler_type = rdp.latent_noise_scheduler_type
        inference_steps = rdp.latent_num_inference_steps
    else:
        if model.variant == "dp_beaver":
            state_dim = model.state_dim + 153
        elif model.variant in STRUCTURED_BEAVER_DP_VARIANTS:
            state_dim = model.state_dim + model.beaver_feature_dim
        elif model.variant == TEMPORAL_BEAVER_VARIANT:
            state_dim = model.state_dim + model.beaver_temporal_feature_dim + 1
        elif model.variant in WRAP_BEAVER_VARIANTS:
            state_dim = (
                model.state_dim
                + model.beaver_wrap_feature_dim
                + model.beaver_wrap_enclosure_dim
            )
        elif model.variant == DELTA_BEAVER_VARIANT:
            state_dim = model.state_dim + model.beaver_delta_feature_dim
        elif model.variant == ADAPTIVE_BEAVER_VARIANT:
            state_dim = 2 * model.state_dim + model.beaver_adaptive_feature_dim + 1
        elif model.variant == ANTIGRAVITY_BEAVER_VARIANT:
            state_dim = 2 * model.state_dim + model.beaver_antigravity_feature_dim + 2
        elif model.variant == CLAUDE_BEAVER_VARIANT:
            state_dim = 2 * model.state_dim + model.claude_feature_dim + 1
        elif model.variant == QWEN_BEAVER_VARIANT:
            state_dim = 3 * model.state_dim + model.beaver_temporal_feature_dim + 1
        else:
            state_dim = model.state_dim
        action_dim = model.action_dim
        horizon = model.horizon
        n_action_steps = model.n_action_steps
        down_dims = model.down_dims
        kernel_size = model.kernel_size
        scheduler_type = model.noise_scheduler_type
        inference_steps = model.num_inference_steps

    return LeRobotUnetConfig(
        n_obs_steps=model.n_obs_steps,
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
            dataset.image_key: PolicyFeature(FeatureType.VISUAL, dataset.image_shape),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,))},
        device="cpu",
        use_amp=False,
        push_to_hub=False,
        horizon=horizon,
        n_action_steps=n_action_steps,
        drop_n_last_frames=horizon - n_action_steps - model.n_obs_steps + 1,
        vision_backbone=model.vision_backbone,
        resize_shape=model.resize_shape,
        crop_ratio=model.crop_ratio,
        crop_is_random=True,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=model.n_groups,
        diffusion_step_embed_dim=model.diffusion_step_embed_dim,
        use_film_scale_modulation=True,
        noise_scheduler_type=scheduler_type,
        num_train_timesteps=model.num_train_timesteps,
        beta_schedule=model.beta_schedule,
        prediction_type=model.prediction_type,
        clip_sample=True,
        clip_sample_range=model.clip_sample_range,
        num_inference_steps=inference_steps,
        do_mask_loss_for_padding=True,
    )


def build_flow_network_config(
    config: RealmanBeaverConfig, *, latent_actions: bool = False
) -> LeRobotUnetConfig:
    """Build the LeRobot visual encoder/U-Net shape config used by flow matching.

    LeRobot exposes these reusable network blocks under its diffusion module, but
    this package does not construct a DiffusionPolicy or a diffusion scheduler.
    """
    dataset, model, rfm = config.dataset, config.model, config.rfm
    if latent_actions:
        state_dim = model.state_dim
        action_dim = rfm.latent_dim
        horizon = rfm.latent_horizon
        n_action_steps = horizon - model.n_obs_steps + 1
        down_dims = rfm.latent_down_dims
        kernel_size = rfm.latent_kernel_size
    else:
        if model.variant == "fm_beaver":
            state_dim = model.state_dim + 153
        elif model.variant == GROK_BEAVER_VARIANT:
            from policies.realman_beaver.modules.grok_phase_encoder import (
                grok_conditioned_state_dim,
            )

            state_dim = grok_conditioned_state_dim(model)
        else:
            state_dim = model.state_dim
        action_dim = model.action_dim
        horizon = model.horizon
        n_action_steps = model.n_action_steps
        down_dims = model.down_dims
        kernel_size = model.kernel_size

    return LeRobotUnetConfig(
        n_obs_steps=model.n_obs_steps,
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
            dataset.image_key: PolicyFeature(FeatureType.VISUAL, dataset.image_shape),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,))},
        device="cpu",
        use_amp=False,
        push_to_hub=False,
        horizon=horizon,
        n_action_steps=n_action_steps,
        drop_n_last_frames=horizon - n_action_steps - model.n_obs_steps + 1,
        vision_backbone=model.vision_backbone,
        resize_shape=model.resize_shape,
        crop_ratio=model.crop_ratio,
        crop_is_random=True,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=model.n_groups,
        diffusion_step_embed_dim=model.flow_time_embed_dim,
        use_film_scale_modulation=True,
    )


class FlowMatchingModel(nn.Module):
    """Conditional velocity field trained with linear optimal-transport paths."""

    def __init__(
        self,
        network_config: LeRobotUnetConfig,
        *,
        num_inference_steps: int,
        time_embedding_scale: float,
        clip_sample_range: float,
    ) -> None:
        super().__init__()
        self.config = network_config
        self.num_inference_steps = num_inference_steps
        self.time_embedding_scale = time_embedding_scale
        self.clip_sample_range = clip_sample_range

        num_images = len(network_config.image_features)
        global_cond_dim = network_config.robot_state_feature.shape[0]
        self.rgb_encoder = DiffusionRgbEncoder(network_config)
        global_cond_dim += self.rgb_encoder.feature_dim * num_images
        self.velocity_field = DiffusionConditionalUnet1d(
            network_config,
            global_cond_dim=global_cond_dim * network_config.n_obs_steps,
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        image_features = self.rgb_encoder(
            einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
        )
        image_features = einops.rearrange(
            image_features,
            "(b s n) ... -> b s (n ...)",
            b=batch_size,
            s=n_obs_steps,
        )
        return torch.cat((batch[OBS_STATE], image_features), dim=-1).flatten(
            start_dim=1
        )

    def _network_time(self, time: Tensor) -> Tensor:
        return time * self.time_embedding_scale

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        trajectory = batch[ACTION]
        noise = torch.randn_like(trajectory)
        time = torch.rand(
            trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype
        )
        path_time = time[:, None, None]
        interpolated = (1.0 - path_time) * noise + path_time * trajectory
        target_velocity = trajectory - noise
        condition = self._prepare_global_conditioning(batch)
        predicted_velocity = self.velocity_field(
            interpolated, self._network_time(time), global_cond=condition
        )
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        valid = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(loss)
        return (loss * valid).sum() / valid.sum().clamp_min(1)

    @torch.no_grad()
    def conditional_sample(
        self,
        batch_size: int,
        *,
        global_cond: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        parameter = next(self.parameters())
        sample = noise
        if sample is None:
            sample = torch.randn(
                batch_size,
                self.config.horizon,
                self.config.action_feature.shape[0],
                device=parameter.device,
                dtype=parameter.dtype,
            )
        step_size = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            time = torch.full(
                (batch_size,),
                step * step_size,
                device=sample.device,
                dtype=sample.dtype,
            )
            velocity = self.velocity_field(
                sample, self._network_time(time), global_cond=global_cond
            )
            sample = sample + step_size * velocity
        return sample.clamp(-self.clip_sample_range, self.clip_sample_range)

    @torch.no_grad()
    def generate_actions(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(
                f"Expected {self.config.n_obs_steps} observations, got {n_obs_steps}"
            )
        trajectory = self.conditional_sample(
            batch_size,
            global_cond=self._prepare_global_conditioning(batch),
            noise=noise,
        )
        start = n_obs_steps - 1
        return trajectory[:, start : start + self.config.n_action_steps]


def build_tokenizer(config: RealmanBeaverConfig) -> AsymmetricBeaverTokenizer:
    model = config.model
    reactive = config.rdp if model.variant == "rdp_like" else config.rfm
    sensor_indices = None
    if model.variant == "rdp_like" and config.rdp.beaver_sensors is not None:
        sensor_indices = resolve_beaver_sensor_indices(
            config.dataset, config.rdp.beaver_sensors
        )
    return AsymmetricBeaverTokenizer(
        action_dim=model.action_dim,
        latent_dim=reactive.latent_dim,
        action_horizon=reactive.action_horizon,
        downsample_ratio=reactive.downsample_ratio,
        hidden_dim=reactive.tokenizer_hidden_dim,
        gru_layers=reactive.tokenizer_layers,
        n_sensors=model.beaver_shape[0],
        sensor_indices=sensor_indices,
        beaver_feature_dim=reactive.beaver_feature_dim,
    )


class LeRobotDPPolicy(nn.Module):
    """Original LeRobot DP, optionally with status-masked Beaver values."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant not in {"original_dp", "dp_beaver"}:
            raise ValueError("LeRobotDPPolicy supports original_dp and dp_beaver")
        self.config = config
        self.normalizer = normalizer
        self.use_beaver = config.model.variant == "dp_beaver"
        self.native_policy = DiffusionPolicy(build_native_diffusion_config(config))

    def _state(self, batch: dict[str, Tensor]) -> Tensor:
        if self.use_beaver:
            return self.normalizer.augmented_state(
                batch["state"],
                batch["beaver_distance"],
                batch["beaver_present"],
                batch["beaver_status"],
            )
        return self.normalizer.normalize_state(batch["state"])

    def _prepare(
        self, batch: dict[str, Tensor], include_action: bool
    ) -> dict[str, Tensor]:
        prepared = {
            OBS_STATE: self._state(batch),
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        if include_action:
            prepared[ACTION] = self.normalizer.normalize_action(batch["action"])
            prepared["action_is_pad"] = batch["action_is_pad"]
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        loss, _ = self.native_policy(self._prepare(batch, include_action=True))
        return loss, {"loss": float(loss.detach())}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._prepare(batch, include_action=False)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        normalized = self.native_policy.diffusion.generate_actions(prepared)
        return self.normalizer.denormalize_action(normalized)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        batch = _ensure_observation_batch(observation)
        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        normalized = self.native_policy.select_action(
            self._prepare(batch, include_action=False)
        )
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)
        return self.normalizer.denormalize_action(normalized)

    def reset(self) -> None:
        self.last_replanned = True
        self.last_chunk_step = 0
        self._chunk_len = 0
        self.native_policy.reset()


class StructuredBeaverDPPolicy(nn.Module):
    """Native LeRobot DP conditioned on a trainable structured Beaver feature."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant not in STRUCTURED_BEAVER_DP_VARIANTS:
            raise ValueError(
                "StructuredBeaverDPPolicy requires a structured Beaver DP variant"
            )
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        if model.variant in KEY4_BEAVER_DP_VARIANTS:
            self.beaver_encoder = Key4BeaverEncoder(
                variant=model.variant,
                n_sensors=model.beaver_shape[0],
                key_sensor_indices=model.beaver_key_sensor_indices,
                valid_statuses=dataset.beaver_valid_statuses,
                output_dim=model.beaver_feature_dim,
                sensor_hidden_dim=model.beaver_sensor_hidden_dim,
                sensor_feature_dim=model.beaver_sensor_feature_dim,
                near_threshold_mm=model.beaver_near_threshold_mm,
                pca_components=model.beaver_pca_components,
            )
        else:
            self.beaver_encoder = StructuredBeaverEncoder(
                variant=model.variant,
                n_sensors=model.beaver_shape[0],
                distance_max_mm=dataset.distance_max_mm,
                valid_statuses=dataset.beaver_valid_statuses,
                output_dim=model.beaver_feature_dim,
                sensor_hidden_dim=model.beaver_sensor_hidden_dim,
                sensor_feature_dim=model.beaver_sensor_feature_dim,
                near_threshold_mm=model.beaver_near_threshold_mm,
                gate_hidden_dim=model.beaver_gate_hidden_dim,
            )
        self.native_policy = DiffusionPolicy(build_native_diffusion_config(config))

    def _state(self, batch: dict[str, Tensor]) -> Tensor:
        beaver_feature = self.beaver_encoder(
            batch["beaver_distance"],
            batch["beaver_status"],
            batch["beaver_present"],
        )
        return torch.cat(
            (self.normalizer.normalize_state(batch["state"]), beaver_feature), dim=-1
        )

    def _prepare(
        self, batch: dict[str, Tensor], include_action: bool
    ) -> dict[str, Tensor]:
        prepared = {
            OBS_STATE: self._state(batch),
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        if include_action:
            prepared[ACTION] = self.normalizer.normalize_action(batch["action"])
            prepared["action_is_pad"] = batch["action_is_pad"]
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        loss, _ = self.native_policy(self._prepare(batch, include_action=True))
        return loss, {"loss": float(loss.detach())}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._prepare(batch, include_action=False)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        normalized = self.native_policy.diffusion.generate_actions(prepared)
        return self.normalizer.denormalize_action(normalized)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        batch = _ensure_observation_batch(observation)
        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        normalized = self.native_policy.select_action(
            self._prepare(batch, include_action=False)
        )
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)
        return self.normalizer.denormalize_action(normalized)

    def reset(self) -> None:
        self.last_replanned = True
        self.last_chunk_step = 0
        self._chunk_len = 0
        self.native_policy.reset()


class TemporalBeaverDPPolicy(nn.Module):
    """Native DP with causal four-sensor temporal Beaver conditioning."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != TEMPORAL_BEAVER_VARIANT:
            raise ValueError(
                f"TemporalBeaverDPPolicy requires model.variant="
                f"{TEMPORAL_BEAVER_VARIANT}"
            )
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.beaver_temporal_sensors
        )
        self.beaver_encoder = TemporalBeaverEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_indices=sensor_indices,
            history_steps=model.beaver_history_steps,
            valid_statuses=dataset.beaver_valid_statuses,
            frame_hidden_dim=model.beaver_frame_hidden_dim,
            frame_feature_dim=model.beaver_frame_feature_dim,
            temporal_hidden_dim=model.beaver_temporal_hidden_dim,
            output_dim=model.beaver_temporal_feature_dim,
        )
        if not normalizer.has_temporal_beaver_statistics:
            raise ValueError(
                "WRM_temporal requires per-sensor robust normalization statistics "
                "fitted from explicit training episodes"
            )
        fitted_indices = tuple(
            int(index) for index in normalizer.beaver_temporal_sensor_indices.tolist()
        )
        if fitted_indices != tuple(sensor_indices):
            raise ValueError(
                "Temporal Beaver normalizer sensor order does not match "
                f"configured sensors: {fitted_indices} != {tuple(sensor_indices)}"
            )
        self.beaver_encoder.set_normalization_statistics(
            p5=normalizer.beaver_temporal_p5,
            p95=normalizer.beaver_temporal_p95,
            median=normalizer.beaver_temporal_median,
        )
        self.grasp_state_head = nn.Sequential(
            nn.Linear(model.beaver_temporal_feature_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.native_policy = DiffusionPolicy(build_native_diffusion_config(config))
        self.reset()

    @torch.no_grad()
    def set_temporal_statistics(
        self, *, p5: Tensor, p95: Tensor, median: Tensor
    ) -> None:
        """Install train-only robust Beaver statistics in checkpoint buffers."""
        self.normalizer.set_temporal_beaver_statistics(
            {
                "p5": p5,
                "p95": p95,
                "median": median,
                "sensor_indices": self.beaver_encoder.sensor_index,
            }
        )
        self.beaver_encoder.set_normalization_statistics(p5=p5, p95=p95, median=median)

    def _state_and_grasp_logit(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        required = {
            "beaver_history_distance",
            "beaver_history_status",
            "beaver_history_present",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(
                f"Temporal Beaver batch is missing history fields: {sorted(missing)}"
            )
        history_distance = batch["beaver_history_distance"]
        history_status = batch["beaver_history_status"]
        history_present = batch["beaver_history_present"]
        if self.config.model.beaver_history_mode == "current":
            history_distance = history_distance[..., -1:, :, :, :].expand_as(
                history_distance
            )
            history_status = history_status[..., -1:, :, :, :].expand_as(
                history_status
            )
            history_present = history_present[..., -1:, :].expand_as(
                history_present
            )
        beaver_feature, encoder_intermediates = self.beaver_encoder(
            history_distance,
            history_status,
            history_present,
            return_intermediates=True,
        )
        sensor_tokens = encoder_intermediates["sensor_tokens"]
        grasp_logit = self.grasp_state_head(beaver_feature).squeeze(-1)
        grasp_probability = grasp_logit.sigmoid()
        self.last_grasp_probability = float(grasp_probability.detach().mean())
        self.last_beaver_feature_mean = float(beaver_feature.detach().mean())
        self.last_beaver_feature_std = float(
            beaver_feature.detach().std(unbiased=False)
        )
        self.last_sensor_token_std = {
            sensor_name: float(
                sensor_tokens[..., sensor_position, :].detach().std(unbiased=False)
            )
            for sensor_position, sensor_name in enumerate(
                self.config.model.beaver_temporal_sensors
            )
        }
        if not self.config.model.use_beaver_condition:
            beaver_feature = torch.zeros_like(beaver_feature)
        grasp_condition = grasp_probability.unsqueeze(-1)
        if (
            not self.config.model.use_beaver_condition
            or not self.config.model.condition_on_grasp_probability
        ):
            grasp_condition = torch.zeros_like(grasp_condition)
        conditioned_state = torch.cat(
            (
                self.normalizer.normalize_state(batch["state"]),
                beaver_feature,
                grasp_condition,
            ),
            dim=-1,
        )
        return conditioned_state, grasp_logit, beaver_feature, sensor_tokens

    def _state(self, batch: dict[str, Tensor]) -> Tensor:
        state, _, _, _ = self._state_and_grasp_logit(batch)
        return state

    def _prepare(
        self,
        batch: dict[str, Tensor],
        include_action: bool,
        *,
        return_grasp_logit: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        state, grasp_logit, beaver_feature, sensor_tokens = self._state_and_grasp_logit(
            batch
        )
        image = self.normalizer.normalize_image(batch["image"])
        if not self.config.model.use_visual_condition:
            image = torch.zeros_like(image)
        prepared = {
            OBS_STATE: state,
            self.config.dataset.image_key: image,
        }
        if include_action:
            prepared[ACTION] = self.normalizer.normalize_action(batch["action"])
            prepared["action_is_pad"] = batch["action_is_pad"]
        if return_grasp_logit:
            return prepared, grasp_logit, beaver_feature, sensor_tokens
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if "grasp_state" not in batch:
            raise KeyError("Temporal Beaver training batch is missing grasp_state")
        prepared, grasp_logit, beaver_feature, sensor_tokens = self._prepare(
            batch, include_action=True, return_grasp_logit=True
        )
        diffusion_loss, _ = self.native_policy(prepared)
        grasp_target = batch["grasp_state"].to(
            device=grasp_logit.device, dtype=grasp_logit.dtype
        )
        if grasp_target.shape == (*grasp_logit.shape, 1):
            grasp_target = grasp_target.squeeze(-1)
        if grasp_target.shape != grasp_logit.shape:
            raise ValueError(
                f"grasp_state shape {tuple(grasp_target.shape)} must match temporal "
                f"observations {tuple(grasp_logit.shape)}"
            )
        grasp_loss = F.binary_cross_entropy_with_logits(grasp_logit, grasp_target)
        total_loss = (
            diffusion_loss + self.config.model.beaver_grasp_loss_weight * grasp_loss
        )
        with torch.no_grad():
            grasp_probability = grasp_logit.sigmoid()
            grasp_accuracy = (
                ((grasp_probability >= 0.5) == (grasp_target >= 0.5)).float().mean()
            )
        return total_loss, {
            "loss": float(total_loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "grasp_loss": float(grasp_loss.detach()),
            "grasp_accuracy": float(grasp_accuracy),
            "grasp_positive_rate": float((grasp_target >= 0.5).float().mean()),
            "predicted_grasp_positive_rate": float(
                (grasp_probability >= 0.5).float().mean()
            ),
            "beaver_feature_std": float(beaver_feature.detach().std(unbiased=False)),
            **{
                f"beaver_sensor_{sensor_name}_token_std": float(
                    sensor_tokens[..., sensor_position, :].detach().std(unbiased=False)
                )
                for sensor_position, sensor_name in enumerate(
                    self.config.model.beaver_temporal_sensors
                )
            },
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._prepare(batch, include_action=False)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        normalized = self.native_policy.diffusion.generate_actions(prepared)
        return self.normalizer.denormalize_action(normalized)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    def _append_online_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        signature = tuple(
            batch[key].data_ptr()
            for key in ("beaver_distance", "beaver_status", "beaver_present")
        )
        frame = {
            "distance": batch["beaver_distance"],
            "status": batch["beaver_status"],
            "present": batch["beaver_present"],
        }
        if signature != self._last_online_frame_signature:
            self._beaver_history.append(frame)
            self._last_online_frame_signature = signature
        frames = list(self._beaver_history)
        padded_frames = [frames[0]] * (
            self.config.model.beaver_history_steps - len(frames)
        ) + frames
        temporal_batch = dict(batch)
        for output_key, frame_key in (
            ("beaver_history_distance", "distance"),
            ("beaver_history_status", "status"),
            ("beaver_history_present", "present"),
        ):
            temporal_batch[output_key] = torch.stack(
                [history_frame[frame_key] for history_frame in padded_frames],
                dim=1,
            )
        return temporal_batch

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        temporal_batch = self._append_online_history(batch)

        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        prepared = self._prepare(temporal_batch, include_action=False)
        normalized = self.native_policy.select_action(prepared)
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)
        return self.normalizer.denormalize_action(normalized)

    def reset(self) -> None:
        self._beaver_history: deque[dict[str, Tensor]] = deque(
            maxlen=self.config.model.beaver_history_steps
        )
        self._batch_size: int | None = None
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_grasp_probability = 0.0
        self.last_beaver_feature_mean = 0.0
        self.last_beaver_feature_std = 0.0
        self.last_sensor_token_std = {
            sensor_name: 0.0
            for sensor_name in self.config.model.beaver_temporal_sensors
        }
        self._chunk_len = 0
        self.native_policy.reset()


class DeltaBeaverDPPolicy(nn.Module):
    """Single native Diffusion Policy conditioned on current and t-k changes."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != DELTA_BEAVER_VARIANT:
            raise ValueError(f"DeltaBeaverDPPolicy requires {DELTA_BEAVER_VARIANT}")
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.beaver_delta_sensors
        )
        if not normalizer.has_delta_beaver_statistics:
            raise ValueError(
                "WRM_delta requires per-sensor training-set normalization statistics"
            )
        fitted_indices = tuple(
            int(index) for index in normalizer.beaver_delta_sensor_indices.tolist()
        )
        if fitted_indices != tuple(sensor_indices):
            raise ValueError(
                "WRM_delta normalizer sensor order does not match configured sensors: "
                f"{fitted_indices} != {tuple(sensor_indices)}"
            )
        self.beaver_encoder = DeltaBeaverEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_indices=sensor_indices,
            valid_statuses=dataset.beaver_valid_statuses,
            sensor_hidden_dim=model.beaver_delta_sensor_hidden_dim,
            sensor_feature_dim=model.beaver_delta_sensor_feature_dim,
            fusion_hidden_dim=model.beaver_delta_fusion_hidden_dim,
            output_dim=model.beaver_delta_feature_dim,
        )
        self.beaver_encoder.set_normalization_statistics(
            mean=normalizer.beaver_delta_mean,
            std=normalizer.beaver_delta_std,
        )
        grasp_input_dim = model.beaver_delta_feature_dim + 2 * model.state_dim
        self.grasp_state_head = nn.Sequential(
            nn.Linear(grasp_input_dim, model.beaver_delta_grasp_hidden_dim),
            nn.SiLU(),
            nn.Linear(model.beaver_delta_grasp_hidden_dim, 1),
        )
        self.native_policy = DiffusionPolicy(build_native_diffusion_config(config))
        self.reset()

    def _state_and_grasp_logit(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        required = {
            "delta_q",
            "beaver_distance",
            "beaver_previous_distance",
            "beaver_status",
            "beaver_previous_status",
            "beaver_present",
            "beaver_previous_present",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"WRM_delta batch is missing fields: {sorted(missing)}")
        z_beaver = self.beaver_encoder(
            batch["beaver_distance"],
            batch["beaver_previous_distance"],
            batch["beaver_status"],
            batch["beaver_previous_status"],
            batch["beaver_present"],
            batch["beaver_previous_present"],
        )
        normalized_qpos = self.normalizer.normalize_state(batch["state"])
        normalized_delta_q = batch["delta_q"] / self.normalizer.state_scale
        grasp_input = torch.cat((z_beaver, normalized_qpos, normalized_delta_q), dim=-1)
        grasp_logit = self.grasp_state_head(grasp_input).squeeze(-1)
        self.last_grasp_probability = float(grasp_logit.detach().sigmoid().mean())
        self.last_z_beaver_std = float(z_beaver.detach().std(unbiased=False))
        conditioned_state = torch.cat((normalized_qpos, z_beaver), dim=-1)
        return conditioned_state, grasp_logit, z_beaver

    def _prepare(
        self,
        batch: dict[str, Tensor],
        include_action: bool,
        *,
        return_auxiliary: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], Tensor, Tensor]:
        state, grasp_logit, z_beaver = self._state_and_grasp_logit(batch)
        prepared = {
            OBS_STATE: state,
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        if include_action:
            prepared[ACTION] = self.normalizer.normalize_action(batch["action"])
            prepared["action_is_pad"] = batch["action_is_pad"]
        if return_auxiliary:
            return prepared, grasp_logit, z_beaver
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if "grasp_state" not in batch:
            raise KeyError("WRM_delta training batch is missing grasp_state")
        prepared, grasp_logit, z_beaver = self._prepare(
            batch, include_action=True, return_auxiliary=True
        )
        diffusion_loss, _ = self.native_policy(prepared)
        grasp_target = batch["grasp_state"].to(
            device=grasp_logit.device, dtype=grasp_logit.dtype
        )
        if grasp_target.shape == (*grasp_logit.shape, 1):
            grasp_target = grasp_target.squeeze(-1)
        if grasp_target.shape != grasp_logit.shape:
            raise ValueError(
                f"grasp_state shape {tuple(grasp_target.shape)} must match "
                f"grasp logits {tuple(grasp_logit.shape)}"
            )
        grasp_loss = F.binary_cross_entropy_with_logits(grasp_logit, grasp_target)
        total_loss = diffusion_loss + self.config.model.lambda_grasp * grasp_loss
        with torch.no_grad():
            grasp_accuracy = (
                ((grasp_logit.sigmoid() >= 0.5) == (grasp_target >= 0.5)).float().mean()
            )
        return total_loss, {
            "loss": float(total_loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "grasp_loss": float(grasp_loss.detach()),
            "grasp_accuracy": float(grasp_accuracy),
            "z_beaver_std": float(z_beaver.detach().std(unbiased=False)),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._prepare(batch, include_action=False)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        normalized = self.native_policy.diffusion.generate_actions(prepared)
        return self.normalizer.denormalize_action(normalized)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    def _append_online_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        signature = tuple(
            batch[key].data_ptr()
            for key in ("state", "beaver_distance", "beaver_status", "beaver_present")
        )
        current = {
            "state": batch["state"],
            "distance": batch["beaver_distance"],
            "status": batch["beaver_status"],
            "present": batch["beaver_present"],
        }
        if signature != self._last_online_frame_signature:
            self._delta_history.append(current)
            self._last_online_frame_signature = signature
        frames = list(self._delta_history)
        previous = (
            frames[0]
            if len(frames) <= self.config.model.beaver_delta_steps
            else frames[-self.config.model.beaver_delta_steps - 1]
        )
        delta_batch = dict(batch)
        delta_batch["delta_q"] = batch["state"] - previous["state"]
        delta_batch["beaver_previous_distance"] = previous["distance"]
        delta_batch["beaver_previous_status"] = previous["status"]
        delta_batch["beaver_previous_present"] = previous["present"]
        return delta_batch

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        delta_batch = self._append_online_history(batch)
        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        normalized = self.native_policy.select_action(
            self._prepare(delta_batch, include_action=False)
        )
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)
        return self.normalizer.denormalize_action(normalized)

    def reset(self) -> None:
        self._delta_history: deque[dict[str, Tensor]] = deque(
            maxlen=self.config.model.beaver_delta_steps + 1
        )
        self._batch_size: int | None = None
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self._chunk_len = 0
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_grasp_probability = 0.0
        self.last_z_beaver_std = 0.0
        self.native_policy.reset()


class AdaptiveBeaverDPPolicy(nn.Module):
    """Diffusion Policy conditioned on size-agnostic Beaver contact geometry."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != ADAPTIVE_BEAVER_VARIANT:
            raise ValueError(
                f"AdaptiveBeaverDPPolicy requires {ADAPTIVE_BEAVER_VARIANT}"
            )
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.beaver_adaptive_sensors
        )
        if not normalizer.has_temporal_beaver_statistics:
            raise ValueError(
                "WRM_adaptive requires robust training-set Beaver statistics"
            )
        fitted_indices = tuple(
            int(index) for index in normalizer.beaver_temporal_sensor_indices.tolist()
        )
        if fitted_indices != tuple(sensor_indices):
            raise ValueError(
                "WRM_adaptive normalizer sensor order does not match config: "
                f"{fitted_indices} != {tuple(sensor_indices)}"
            )
        self.beaver_encoder = AdaptiveBeaverEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_indices=sensor_indices,
            history_steps=model.beaver_adaptive_history_steps,
            lag_steps=model.beaver_adaptive_lag_steps,
            valid_statuses=dataset.beaver_valid_statuses,
            proximity_scales_mm=model.beaver_adaptive_proximity_scales_mm,
            sensor_hidden_dim=model.beaver_adaptive_sensor_hidden_dim,
            token_dim=model.beaver_adaptive_token_dim,
            transformer_layers=model.beaver_adaptive_transformer_layers,
            attention_heads=model.beaver_adaptive_attention_heads,
            output_dim=model.beaver_adaptive_feature_dim,
            noise_std_mm=model.beaver_adaptive_noise_std_mm,
            pixel_dropout=model.beaver_adaptive_pixel_dropout,
            sensor_dropout=model.beaver_adaptive_sensor_dropout,
        )
        self.beaver_encoder.set_normalization_statistics(
            p5=normalizer.beaver_temporal_p5,
            p95=normalizer.beaver_temporal_p95,
            median=normalizer.beaver_temporal_median,
        )
        grasp_input_dim = model.beaver_adaptive_feature_dim + 2 * model.state_dim
        self.grasp_state_head = nn.Sequential(
            nn.Linear(grasp_input_dim, model.beaver_adaptive_grasp_hidden_dim),
            nn.SiLU(),
            nn.Linear(model.beaver_adaptive_grasp_hidden_dim, 1),
        )
        self.native_policy = DiffusionPolicy(build_native_diffusion_config(config))
        self.reset()

    def _state_and_grasp_logit(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        required = {
            "delta_q",
            "beaver_history_distance",
            "beaver_history_status",
            "beaver_history_present",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"WRM_adaptive batch is missing fields: {sorted(missing)}")
        z_beaver, intermediates = self.beaver_encoder(
            batch["beaver_history_distance"],
            batch["beaver_history_status"],
            batch["beaver_history_present"],
            return_intermediates=True,
        )
        normalized_qpos = self.normalizer.normalize_state(batch["state"])
        normalized_delta_q = batch["delta_q"] / self.normalizer.state_scale
        grasp_input = torch.cat((z_beaver, normalized_qpos, normalized_delta_q), dim=-1)
        grasp_logit = self.grasp_state_head(grasp_input).squeeze(-1)
        grasp_probability = grasp_logit.sigmoid()
        # Unlike WRM_delta, both motion delta and inferred contact state directly
        # condition the action generator, so they can break replan limit cycles.
        conditioned_state = torch.cat(
            (
                normalized_qpos,
                normalized_delta_q,
                z_beaver,
                grasp_probability.unsqueeze(-1),
            ),
            dim=-1,
        )
        attention = intermediates["sensor_attention"]
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(
            dim=-1
        )
        proximity = intermediates["proximity"]
        self.last_grasp_probability = float(grasp_probability.detach().mean())
        self.last_z_beaver_std = float(z_beaver.detach().std(unbiased=False))
        self.last_sensor_attention_entropy = float(entropy.detach().mean())
        self.last_near_field_fraction = float(
            (proximity[..., 0] > torch.exp(torch.tensor(-1.0, device=proximity.device)))
            .float()
            .mean()
        )
        self.last_sensor_attention = {
            sensor_name: float(attention[..., sensor_position].detach().mean())
            for sensor_position, sensor_name in enumerate(
                self.config.model.beaver_adaptive_sensors
            )
        }
        return conditioned_state, grasp_logit, z_beaver, attention

    def _prepare(
        self,
        batch: dict[str, Tensor],
        include_action: bool,
        *,
        return_auxiliary: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        state, grasp_logit, z_beaver, attention = self._state_and_grasp_logit(batch)
        prepared = {
            OBS_STATE: state,
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        if include_action:
            prepared[ACTION] = self.normalizer.normalize_action(batch["action"])
            prepared["action_is_pad"] = batch["action_is_pad"]
        if return_auxiliary:
            return prepared, grasp_logit, z_beaver, attention
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if "grasp_state" not in batch:
            raise KeyError("WRM_adaptive training batch is missing grasp_state")
        prepared, grasp_logit, z_beaver, attention = self._prepare(
            batch, include_action=True, return_auxiliary=True
        )
        diffusion_loss, _ = self.native_policy(prepared)
        grasp_target = batch["grasp_state"].to(
            device=grasp_logit.device, dtype=grasp_logit.dtype
        )
        if grasp_target.shape == (*grasp_logit.shape, 1):
            grasp_target = grasp_target.squeeze(-1)
        if grasp_target.shape != grasp_logit.shape:
            raise ValueError(
                f"grasp_state shape {tuple(grasp_target.shape)} must match "
                f"grasp logits {tuple(grasp_logit.shape)}"
            )
        grasp_loss = F.binary_cross_entropy_with_logits(grasp_logit, grasp_target)
        total_loss = (
            diffusion_loss
            + self.config.model.beaver_adaptive_grasp_loss_weight * grasp_loss
        )
        with torch.no_grad():
            probability = grasp_logit.sigmoid()
            accuracy = ((probability >= 0.5) == (grasp_target >= 0.5)).float().mean()
            entropy = (
                -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log())
                .sum(dim=-1)
                .mean()
            )
        return total_loss, {
            "loss": float(total_loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "grasp_loss": float(grasp_loss.detach()),
            "grasp_accuracy": float(accuracy),
            "grasp_positive_rate": float((grasp_target >= 0.5).float().mean()),
            "predicted_grasp_positive_rate": float((probability >= 0.5).float().mean()),
            "z_beaver_std": float(z_beaver.detach().std(unbiased=False)),
            "sensor_attention_entropy": float(entropy),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._prepare(batch, include_action=False)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        normalized = self.native_policy.diffusion.generate_actions(prepared)
        return self.normalizer.denormalize_action(normalized)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    def _append_online_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        signature = tuple(
            batch[key].data_ptr()
            for key in ("state", "beaver_distance", "beaver_status", "beaver_present")
        )
        current = {
            "state": batch["state"],
            "distance": batch["beaver_distance"],
            "status": batch["beaver_status"],
            "present": batch["beaver_present"],
        }
        if signature != self._last_online_frame_signature:
            self._adaptive_history.append(current)
            self._last_online_frame_signature = signature
        frames = list(self._adaptive_history)
        history_steps = self.config.model.beaver_adaptive_history_steps
        padded = [frames[0]] * (history_steps - len(frames)) + frames[-history_steps:]
        delta_steps = self.config.model.beaver_adaptive_motion_delta_steps
        previous = frames[0] if len(frames) <= delta_steps else frames[-delta_steps - 1]
        adaptive_batch = dict(batch)
        adaptive_batch["delta_q"] = batch["state"] - previous["state"]
        for output_key, frame_key in (
            ("beaver_history_distance", "distance"),
            ("beaver_history_status", "status"),
            ("beaver_history_present", "present"),
        ):
            adaptive_batch[output_key] = torch.stack(
                [history_frame[frame_key] for history_frame in padded], dim=1
            )
        return adaptive_batch

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        adaptive_batch = self._append_online_history(batch)
        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        normalized = self.native_policy.select_action(
            self._prepare(adaptive_batch, include_action=False)
        )
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)
        return self.normalizer.denormalize_action(normalized)

    def reset(self) -> None:
        history_length = max(
            self.config.model.beaver_adaptive_history_steps,
            self.config.model.beaver_adaptive_motion_delta_steps + 1,
        )
        self._adaptive_history: deque[dict[str, Tensor]] = deque(maxlen=history_length)
        self._batch_size: int | None = None
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self._chunk_len = 0
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_grasp_probability = 0.0
        self.last_z_beaver_std = 0.0
        self.last_sensor_attention_entropy = 0.0
        self.last_near_field_fraction = 0.0
        self.last_sensor_attention = {
            sensor_name: 0.0
            for sensor_name in self.config.model.beaver_adaptive_sensors
        }
        self.native_policy.reset()


class FMPolicy(nn.Module):
    """Flow matching with visual/state conditioning and optional Beaver input."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant not in {"fm", "fm_beaver"}:
            raise ValueError("FMPolicy supports fm and fm_beaver")
        self.config = config
        self.normalizer = normalizer
        self.use_beaver = config.model.variant == "fm_beaver"
        self.flow = FlowMatchingModel(
            build_flow_network_config(config),
            num_inference_steps=config.model.flow_num_inference_steps,
            time_embedding_scale=config.model.flow_time_embedding_scale,
            clip_sample_range=config.model.clip_sample_range,
        )
        self.reset()

    def _state(self, batch: dict[str, Tensor]) -> Tensor:
        if self.use_beaver:
            return self.normalizer.augmented_state(
                batch["state"],
                batch["beaver_distance"],
                batch["beaver_present"],
                batch["beaver_status"],
            )
        return self.normalizer.normalize_state(batch["state"])

    def _prepare(
        self, batch: dict[str, Tensor], include_action: bool
    ) -> dict[str, Tensor]:
        prepared = {
            OBS_STATE: self._state(batch),
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        if include_action:
            prepared[ACTION] = self.normalizer.normalize_action(batch["action"])
            prepared["action_is_pad"] = batch["action_is_pad"]
        return prepared

    def _add_image_axis(self, prepared: dict[str, Tensor]) -> dict[str, Tensor]:
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        prepared = self._add_image_axis(self._prepare(batch, include_action=True))
        loss = self.flow.compute_loss(prepared)
        return loss, {
            "loss": float(loss.detach()),
            "flow_matching_loss": float(loss.detach()),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._add_image_axis(self._prepare(batch, include_action=False))
        normalized = self.flow.generate_actions(prepared)
        return self.normalizer.denormalize_action(normalized)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        prepared = self._prepare(batch, include_action=False)
        self._observation_queue.append(prepared)
        while len(self._observation_queue) < self._observation_queue.maxlen:
            self._observation_queue.append(prepared)

        replanned = not self._action_queue
        if replanned:
            history = {
                key: torch.stack(
                    [frame[key] for frame in self._observation_queue], dim=1
                )
                for key in (OBS_STATE, self.config.dataset.image_key)
            }
            normalized = self.flow.generate_actions(self._add_image_axis(history))
            self._action_queue.extend(normalized.transpose(0, 1))
            self._chunk_len = len(self._action_queue)

        action = self._action_queue.popleft()
        self.last_replanned = replanned
        self.last_chunk_step = self._chunk_len - len(self._action_queue) - 1
        return self.normalizer.denormalize_action(action)

    def reset(self) -> None:
        self._observation_queue: deque[dict[str, Tensor]] = deque(
            maxlen=self.config.model.n_obs_steps
        )
        self._action_queue: deque[Tensor] = deque(
            maxlen=self.config.model.n_action_steps
        )
        self._batch_size: int | None = None
        self._chunk_len = 0
        self.last_replanned = True
        self.last_chunk_step = 0


class RFMPolicy(nn.Module):
    """Slow visual latent flow plus fast causal Beaver-conditioned decoder."""

    def __init__(
        self,
        config: RealmanBeaverConfig,
        normalizer: ObservationNormalizer,
        latent_normalizer: LatentNormalizer,
        tokenizer: AsymmetricBeaverTokenizer | None = None,
    ) -> None:
        super().__init__()
        if config.model.variant != "rfm":
            raise ValueError("RFMPolicy requires model.variant=rfm")
        self.config = config
        self.normalizer = normalizer
        self.latent_normalizer = latent_normalizer
        self.tokenizer = tokenizer if tokenizer is not None else build_tokenizer(config)
        self.slow_flow = FlowMatchingModel(
            build_flow_network_config(config, latent_actions=True),
            num_inference_steps=config.rfm.latent_num_inference_steps,
            time_embedding_scale=config.model.flow_time_embedding_scale,
            clip_sample_range=config.model.clip_sample_range,
        )
        self.reset()

    def freeze_tokenizer(self) -> None:
        self.tokenizer.eval()
        self.tokenizer.requires_grad_(False)

    def tokenizer_loss(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        action = self.normalizer.normalize_action(batch["action"])
        present = batch["beaver_present"]
        distance = self.normalizer.normalize_beaver(
            batch["beaver_distance"], present, batch["beaver_status"]
        )
        return self.tokenizer.compute_loss(
            action,
            batch["action_is_pad"],
            distance,
            present,
            self.config.rfm.kl_weight,
        )

    def _prepare_slow(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        prepared = {
            OBS_STATE: self.normalizer.normalize_state(batch["state"]),
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        return prepared

    def latent_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        with torch.no_grad():
            action = self.normalizer.normalize_action(batch["action"])
            latent, _, _ = self.tokenizer.encode(action, sample=False)
            latent = self.latent_normalizer.normalize(latent)
        prepared = self._prepare_slow(batch)
        prepared[ACTION] = latent
        prepared["action_is_pad"] = (
            batch["action_is_pad"]
            .reshape(
                action.shape[0],
                self.config.rfm.latent_horizon,
                self.config.rfm.downsample_ratio,
            )
            .any(dim=-1)
        )
        loss = self.slow_flow.compute_loss(prepared)
        return loss, {
            "loss": float(loss.detach()),
            "latent_flow_matching_loss": float(loss.detach()),
        }

    @torch.no_grad()
    def sample_latent(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._prepare_slow(batch)
        condition = self.slow_flow._prepare_global_conditioning(prepared)
        latent = self.slow_flow.conditional_sample(
            prepared[OBS_STATE].shape[0], global_cond=condition
        )
        return self.latent_normalizer.denormalize(latent)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        latent = self.sample_latent(batch)
        present = batch["beaver_present"]
        distance = self.normalizer.normalize_beaver(
            batch["beaver_distance"], present, batch["beaver_status"]
        )
        normalized_action, _ = self.tokenizer.decode(latent, distance, present)
        return self.normalizer.denormalize_action(normalized_action)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)[:, : self.config.rfm.slow_replan_steps]

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        """Run one fast step, replanning the latent chunk at the configured rate."""
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        current = {"image": batch["image"], "state": batch["state"]}
        self._observation_queue.append(current)
        while len(self._observation_queue) < self._observation_queue.maxlen:
            self._observation_queue.append(current)

        replanned = (
            self._slow_latent is None
            or self._chunk_step >= self.config.rfm.slow_replan_steps
        )
        if replanned:
            history = list(self._observation_queue)
            stride = self.config.rfm.slow_observation_stride
            count = self.config.model.n_obs_steps
            indices = [
                len(history) - 1 - (count - 1 - index) * stride
                for index in range(count)
            ]
            slow_batch = {
                key: torch.stack([history[index][key] for index in indices], dim=1)
                for key in ("image", "state")
            }
            self._slow_latent = self.sample_latent(slow_batch)
            self._decoder_hidden = None
            self._chunk_step = 0

        token_index = self._chunk_step // self.config.rfm.downsample_ratio
        present = batch["beaver_present"]
        distance = self.normalizer.normalize_beaver(
            batch["beaver_distance"], present, batch["beaver_status"]
        )
        normalized_action, self._decoder_hidden = self.tokenizer.decode_step(
            self._slow_latent[:, token_index], distance, present, self._decoder_hidden
        )
        self.last_replanned = replanned
        self.last_chunk_step = self._chunk_step
        self.last_token_index = token_index
        self._chunk_step += 1
        return self.normalizer.denormalize_action(normalized_action)

    def reset(self) -> None:
        history_length = (
            self.config.model.n_obs_steps - 1
        ) * self.config.rfm.slow_observation_stride + 1
        self._observation_queue: deque[dict[str, Tensor]] = deque(maxlen=history_length)
        self._slow_latent: Tensor | None = None
        self._decoder_hidden: Tensor | None = None
        self._chunk_step = 0
        self._batch_size: int | None = None
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_token_index = 0


class RDPPolicy(nn.Module):
    """Original slow visual latent DP plus fast Beaver-conditioned decoder."""

    def __init__(
        self,
        config: RealmanBeaverConfig,
        normalizer: ObservationNormalizer,
        latent_normalizer: LatentNormalizer,
        tokenizer: AsymmetricBeaverTokenizer | None = None,
    ) -> None:
        super().__init__()
        if config.model.variant != "rdp_like":
            raise ValueError("RDPPolicy requires model.variant=rdp_like")
        self.config = config
        self.normalizer = normalizer
        self.latent_normalizer = latent_normalizer
        self.tokenizer = tokenizer if tokenizer is not None else build_tokenizer(config)
        self.slow_policy = DiffusionPolicy(
            build_native_diffusion_config(config, latent_actions=True)
        )
        self.reset()

    def freeze_tokenizer(self) -> None:
        self.tokenizer.eval()
        self.tokenizer.requires_grad_(False)

    def tokenizer_loss(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        action = self.normalizer.normalize_action(batch["action"])
        present = batch["beaver_present"]
        distance = self.normalizer.normalize_beaver(
            batch["beaver_distance"], present, batch["beaver_status"]
        )
        return self.tokenizer.compute_loss(
            action,
            batch["action_is_pad"],
            distance,
            present,
            self.config.rdp.kl_weight,
        )

    def _prepare_slow(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            OBS_STATE: self.normalizer.normalize_state(batch["state"]),
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }

    def latent_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        with torch.no_grad():
            action = self.normalizer.normalize_action(batch["action"])
            latent, _, _ = self.tokenizer.encode(action, sample=False)
            latent = self.latent_normalizer.normalize(latent)
        prepared = self._prepare_slow(batch)
        prepared[ACTION] = latent
        prepared["action_is_pad"] = (
            batch["action_is_pad"]
            .reshape(
                action.shape[0],
                self.config.rdp.latent_horizon,
                self.config.rdp.downsample_ratio,
            )
            .any(dim=-1)
        )
        loss, _ = self.slow_policy(prepared)
        return loss, {
            "loss": float(loss.detach()),
            "latent_dp_loss": float(loss.detach()),
        }

    @torch.no_grad()
    def sample_latent(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._prepare_slow(batch)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        condition = self.slow_policy.diffusion._prepare_global_conditioning(prepared)
        latent = self.slow_policy.diffusion.conditional_sample(
            prepared[OBS_STATE].shape[0], global_cond=condition
        )
        return self.latent_normalizer.denormalize(latent)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        latent = self.sample_latent(batch)
        present = batch["beaver_present"]
        distance = self.normalizer.normalize_beaver(
            batch["beaver_distance"], present, batch["beaver_status"]
        )
        normalized_action, _ = self.tokenizer.decode(latent, distance, present)
        return self.normalizer.denormalize_action(normalized_action)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)[:, : self.config.rdp.slow_replan_steps]

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        current = {"image": batch["image"], "state": batch["state"]}
        self._observation_queue.append(current)
        while len(self._observation_queue) < self._observation_queue.maxlen:
            self._observation_queue.append(current)

        replanned = (
            self._slow_latent is None
            or self._chunk_step >= self.config.rdp.slow_replan_steps
        )
        if replanned:
            history = list(self._observation_queue)
            stride = self.config.rdp.slow_observation_stride
            count = self.config.model.n_obs_steps
            indices = [
                len(history) - 1 - (count - 1 - index) * stride
                for index in range(count)
            ]
            slow_batch = {
                key: torch.stack([history[index][key] for index in indices], dim=1)
                for key in ("image", "state")
            }
            self._slow_latent = self.sample_latent(slow_batch)
            self._decoder_hidden = None
            self._chunk_step = 0

        token_index = self._chunk_step // self.config.rdp.downsample_ratio
        present = batch["beaver_present"]
        distance = self.normalizer.normalize_beaver(
            batch["beaver_distance"], present, batch["beaver_status"]
        )
        normalized_action, self._decoder_hidden = self.tokenizer.decode_step(
            self._slow_latent[:, token_index], distance, present, self._decoder_hidden
        )
        self.last_replanned = replanned
        self.last_chunk_step = self._chunk_step
        self.last_token_index = token_index
        self._chunk_step += 1
        return self.normalizer.denormalize_action(normalized_action)

    def reset(self) -> None:
        history_length = (
            self.config.model.n_obs_steps - 1
        ) * self.config.rdp.slow_observation_stride + 1
        self._observation_queue: deque[dict[str, Tensor]] = deque(maxlen=history_length)
        self._slow_latent: Tensor | None = None
        self._decoder_hidden: Tensor | None = None
        self._chunk_step = 0
        self._batch_size: int | None = None
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_token_index = 0
        self.slow_policy.reset()


def _ensure_observation_batch(observation: dict[str, Tensor]) -> dict[str, Tensor]:
    required = {"image": 4, "state": 2}
    if "beaver_distance" in observation:
        required.update({"beaver_distance": 4, "beaver_present": 2, "beaver_status": 4})
    batch: dict[str, Tensor] = {}
    for key, batched_ndim in required.items():
        if key not in observation:
            raise KeyError(f"Deployment observation is missing {key}")
        value = observation[key]
        if value.ndim == batched_ndim - 1:
            value = value.unsqueeze(0)
        if value.ndim != batched_ndim:
            raise ValueError(
                f"{key} must have {batched_ndim - 1} or {batched_ndim} dims"
            )
        batch[key] = value
    return batch


def build_policy(
    config: RealmanBeaverConfig,
    normalizer: ObservationNormalizer,
    latent_normalizer: LatentNormalizer | None = None,
):
    if config.model.variant in {"original_dp", "dp_beaver"}:
        return LeRobotDPPolicy(config, normalizer)
    if config.model.variant == BEAVER_CLOSURE_VARIANT:
        from policies.realman_beaver.modeling_dp_beaver_closure import (
            DPBeaverClosurePolicy,
        )

        return DPBeaverClosurePolicy(config, normalizer)
    if config.model.variant in STRUCTURED_BEAVER_DP_VARIANTS:
        return StructuredBeaverDPPolicy(config, normalizer)
    if config.model.variant == TEMPORAL_BEAVER_VARIANT:
        return TemporalBeaverDPPolicy(config, normalizer)
    if config.model.variant == WRAP_BEAVER_VARIANT:
        from policies.realman_beaver.modeling_wrm_wrap import WrapBeaverDPPolicy

        return WrapBeaverDPPolicy(config, normalizer)
    if config.model.variant in WRAP_MONITOR_BEAVER_VARIANTS:
        from policies.realman_beaver.modeling_wrm_wrap_monitor import (
            MonitorWrapBeaverDPPolicy,
        )

        return MonitorWrapBeaverDPPolicy(config, normalizer)
    if config.model.variant == WRAP_DELTA_BEAVER_VARIANT:
        from policies.realman_beaver.modeling_wrm_wrap_delta import (
            WrapDeltaBeaverDPPolicy,
        )

        return WrapDeltaBeaverDPPolicy(config, normalizer)
    if config.model.variant == DELTA_BEAVER_VARIANT:
        return DeltaBeaverDPPolicy(config, normalizer)
    if config.model.variant == ADAPTIVE_BEAVER_VARIANT:
        return AdaptiveBeaverDPPolicy(config, normalizer)
    if config.model.variant == ANTIGRAVITY_BEAVER_VARIANT:
        return AntigravityDPPolicy(config, normalizer)
    if config.model.variant == GROK_BEAVER_VARIANT:
        from policies.realman_beaver.modeling_wrm_grok import WRMGrokPolicy

        return WRMGrokPolicy(config, normalizer)
    if config.model.variant == CODEX_BEAVER_VARIANT:
        from policies.realman_beaver.codex_policy import WRMCodexPolicy

        return WRMCodexPolicy(config, normalizer)
    if config.model.variant == CLAUDE_BEAVER_VARIANT:
        return ClaudeBeaverDPPolicy(config, normalizer)
    if config.model.variant == QWEN_BEAVER_VARIANT:
        return QwenBeaverDPPolicy(config, normalizer)
    if config.model.variant == "rdp_like":
        latent_normalizer = latent_normalizer or LatentNormalizer.identity(
            config.rdp.latent_dim
        )
        return RDPPolicy(config, normalizer, latent_normalizer)
    if config.model.variant in {"fm", "fm_beaver"}:
        return FMPolicy(config, normalizer)
    if config.model.variant == "rfm":
        latent_normalizer = latent_normalizer or LatentNormalizer.identity(
            config.rfm.latent_dim
        )
        return RFMPolicy(config, normalizer, latent_normalizer)
    raise ValueError(f"Unsupported policy variant: {config.model.variant}")
