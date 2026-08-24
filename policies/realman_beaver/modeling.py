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
    STRUCTURED_BEAVER_DP_VARIANTS,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import LatentNormalizer, ObservationNormalizer
from policies.realman_beaver.modules import (
    AsymmetricBeaverTokenizer,
    StructuredBeaverEncoder,
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
        state_dim = model.state_dim + (153 if model.variant == "fm_beaver" else 0)
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
    return AsymmetricBeaverTokenizer(
        action_dim=model.action_dim,
        latent_dim=reactive.latent_dim,
        action_horizon=reactive.action_horizon,
        downsample_ratio=reactive.downsample_ratio,
        hidden_dim=reactive.tokenizer_hidden_dim,
        gru_layers=reactive.tokenizer_layers,
        n_sensors=model.beaver_shape[0],
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
) -> LeRobotDPPolicy | StructuredBeaverDPPolicy | RDPPolicy | FMPolicy | RFMPolicy:
    if config.model.variant in {"original_dp", "dp_beaver"}:
        return LeRobotDPPolicy(config, normalizer)
    if config.model.variant in STRUCTURED_BEAVER_DP_VARIANTS:
        return StructuredBeaverDPPolicy(config, normalizer)
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
