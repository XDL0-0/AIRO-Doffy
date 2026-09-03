"""WRM_claude: pose-anchored delta diffusion with Key4 contact gating."""

from __future__ import annotations

from collections import deque

import einops
import torch
import torch.nn.functional as F
from lerobot.policies.diffusion.configuration_diffusion import (
    DiffusionConfig as LeRobotUnetConfig,
)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from policies.realman_beaver.configuration import (
    CLAUDE_BEAVER_VARIANT,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import (
    ObservationNormalizer,
    resolve_beaver_sensor_indices,
)
from policies.realman_beaver.modules.claude_contact_encoder import ContactFieldEncoder


class ClaudeDeltaDiffusionModel(DiffusionModel):
    """Diffusion model with contact-gated vision and an x0-smoothness penalty.

    Subclasses LeRobot's DiffusionModel so the native sampling path
    (``generate_actions`` -> ``conditional_sample``) is reused unchanged.
    Two overrides differentiate it from WRM_temporal's plain native model:

    1. ``_prepare_global_conditioning`` splits the contact-field block out of
       the state conditioning and applies a bias-free FiLM gate to the visual
       features. An all-invalid contact field encodes to exactly zero, which
       maps to scale=1/shift=0 (identity), so a disconnected or stale Beaver
       can never distort the visual pathway.
    2. ``compute_loss`` adds a small penalty on the second differences of the
       closed-form x0 estimate, encouraging within-chunk action smoothness in
       the epsilon parameterization.
    """

    def __init__(
        self,
        config: LeRobotUnetConfig,
        *,
        state_dim: int,
        contact_feature_dim: int,
        smoothness_weight: float = 0.0,
    ) -> None:
        super().__init__(config)
        if smoothness_weight < 0:
            raise ValueError("smoothness_weight must be non-negative")
        self.state_dim = int(state_dim)
        self.contact_feature_dim = int(contact_feature_dim)
        self.smoothness_weight = float(smoothness_weight)
        self.last_smoothness_loss = 0.0
        image_feature_dim = self.rgb_encoder.feature_dim
        # Bias-free on purpose: contact == 0 (all-invalid Beaver frame) must
        # produce scale=1, shift=0 so the visual path is left untouched.
        self.contact_gate = nn.Sequential(
            nn.Linear(contact_feature_dim, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, 2 * image_feature_dim, bias=False),
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        state = batch[OBS_STATE]
        contact_start = 2 * self.state_dim
        contact = state[
            ..., contact_start : contact_start + self.contact_feature_dim
        ]
        image_features = self.rgb_encoder(
            einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
        )
        image_features = einops.rearrange(
            image_features,
            "(b s n) ... -> b s (n ...)",
            b=batch_size,
            s=n_obs_steps,
        )
        gate = self.contact_gate(contact)  # (b, s, 2 * feature_dim)
        scale, shift = gate.chunk(2, dim=-1)
        image_features = image_features * (1.0 + torch.tanh(scale)) + shift
        return torch.cat((state, image_features), dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        # Mirrors lerobot DiffusionModel.compute_loss, with the smoothness
        # penalty appended. Kept explicit so the penalty sees the raw
        # noised trajectory and epsilon prediction.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, eps, timesteps
        )
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}"
            )

        loss = F.mse_loss(pred, target, reduction="none")
        in_episode_bound = ~batch["action_is_pad"]
        loss = loss * in_episode_bound.unsqueeze(-1)
        diffusion_loss = loss.mean()

        self.last_smoothness_loss = 0.0
        if self.smoothness_weight > 0 and self.config.prediction_type == "epsilon":
            x0 = self._predict_x0(noisy_trajectory, pred, timesteps)
            acceleration = x0[:, 2:] - 2.0 * x0[:, 1:-1] + x0[:, :-2]
            smooth = acceleration.square().mean(dim=-1)  # (b, horizon - 2)
            pad = batch["action_is_pad"]
            valid = ~(pad[:, 2:] | pad[:, 1:-1] | pad[:, :-2])
            smooth_loss = (smooth * valid).sum() / valid.sum().clamp_min(1)
            self.last_smoothness_loss = float(smooth_loss.detach())
            return diffusion_loss + self.smoothness_weight * smooth_loss
        return diffusion_loss

    def _predict_x0(
        self, noisy: Tensor, eps_pred: Tensor, timesteps: Tensor
    ) -> Tensor:
        alphas = self.noise_scheduler.alphas_cumprod.to(
            device=noisy.device, dtype=noisy.dtype
        )
        alpha_t = alphas[timesteps]  # (b,)
        sqrt_alpha = alpha_t.sqrt().clamp_min(1e-4)[:, None, None]
        sqrt_one_minus = (1.0 - alpha_t).clamp_min(0.0).sqrt()[:, None, None]
        return (noisy - sqrt_one_minus * eps_pred) / sqrt_alpha

class ClaudeBeaverDPPolicy(nn.Module):
    """Pose-anchored delta-action DP with contact-gated vision.

    The policy samples normalized joint-delta chunks (delta ~ 0 means "hold")
    anchored to the measured configuration at replan time. Each executed
    target is ``anchor + cumsum(clamped deltas)``, so replan-boundary
    discontinuity is structurally bounded and closed-loop replanning re-
    anchors on the true pose. The grasp head conditions actions only through
    the learned probability; it never gates the loss.
    """

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != CLAUDE_BEAVER_VARIANT:
            raise ValueError(
                f"ClaudeBeaverDPPolicy requires {CLAUDE_BEAVER_VARIANT}"
            )
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.claude_sensors
        )
        if not normalizer.has_temporal_beaver_statistics:
            raise ValueError(
                "WRM_claude requires robust training-set Beaver statistics"
            )
        fitted_indices = tuple(
            int(index)
            for index in normalizer.beaver_temporal_sensor_indices.tolist()
        )
        if fitted_indices != tuple(sensor_indices):
            raise ValueError(
                "WRM_claude normalizer sensor order does not match config: "
                f"{fitted_indices} != {tuple(sensor_indices)}"
            )
        if not normalizer.has_action_delta_statistics:
            raise ValueError(
                "WRM_claude requires train-split per-joint action delta scales"
            )
        self.contact_encoder = ContactFieldEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_indices=sensor_indices,
            history_steps=model.claude_history_steps,
            lag_steps=model.claude_lag_steps,
            valid_statuses=dataset.beaver_valid_statuses,
            proximity_scales_mm=model.claude_proximity_scales_mm,
            sensor_hidden_dim=model.claude_sensor_hidden_dim,
            token_dim=model.claude_token_dim,
            transformer_layers=model.claude_transformer_layers,
            attention_heads=model.claude_attention_heads,
            output_dim=model.claude_feature_dim,
            noise_std_mm=model.claude_noise_std_mm,
            pixel_dropout=model.claude_pixel_dropout,
            sensor_dropout=model.claude_sensor_dropout,
        )
        self.contact_encoder.set_normalization_statistics(
            p5=normalizer.beaver_temporal_p5,
            p95=normalizer.beaver_temporal_p95,
            median=normalizer.beaver_temporal_median,
        )
        self.grasp_state_head = nn.Sequential(
            nn.Linear(model.claude_feature_dim, model.claude_grasp_hidden_dim),
            nn.SiLU(),
            nn.Linear(model.claude_grasp_hidden_dim, 1),
        )
        from policies.realman_beaver.modeling import build_native_diffusion_config
        self.diffusion_model = ClaudeDeltaDiffusionModel(
            build_native_diffusion_config(config),
            state_dim=model.state_dim,
            contact_feature_dim=model.claude_feature_dim,
            smoothness_weight=model.claude_smoothness_loss_weight,
        )
        self.reset()

    @torch.no_grad()
    def set_claude_statistics(
        self,
        *,
        p5: Tensor,
        p95: Tensor,
        median: Tensor,
        action_delta_scale: Tensor,
    ) -> None:
        """Install train-split Beaver and delta-scale statistics at train time."""
        self.normalizer.set_temporal_beaver_statistics(
            {
                "p5": p5,
                "p95": p95,
                "median": median,
                "sensor_indices": self.contact_encoder.sensor_index,
            }
        )
        self.normalizer.set_action_delta_statistics({"scale": action_delta_scale})
        self.contact_encoder.set_normalization_statistics(p5=p5, p95=p95, median=median)

    def _state_and_grasp_logit(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        required = {
            "delta_q",
            "beaver_history_distance",
            "beaver_history_status",
            "beaver_history_present",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"WRM_claude batch is missing fields: {sorted(missing)}")
        contact, intermediates = self.contact_encoder(
            batch["beaver_history_distance"],
            batch["beaver_history_status"],
            batch["beaver_history_present"],
            return_intermediates=True,
        )
        normalized_qpos = self.normalizer.normalize_state(batch["state"])
        normalized_delta_q = batch["delta_q"] / self.normalizer.state_scale
        grasp_logit = self.grasp_state_head(contact).squeeze(-1)
        grasp_probability = grasp_logit.sigmoid()
        self.last_grasp_probability = float(grasp_probability.detach().mean())
        self.last_contact_feature_std = float(
            contact.detach().std(unbiased=False)
        )
        self.last_contact_valid_fraction = float(
            intermediates["cell_available"].float().mean()
        )
        conditioned_state = torch.cat(
            (
                normalized_qpos,
                normalized_delta_q,
                contact,
                grasp_probability.unsqueeze(-1),
            ),
            dim=-1,
        )
        return conditioned_state, grasp_logit, contact, intermediates

    def _prepare(
        self,
        batch: dict[str, Tensor],
        include_action: bool,
        *,
        return_auxiliary: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], Tensor, Tensor, dict]:
        state, grasp_logit, contact, intermediates = self._state_and_grasp_logit(
            batch
        )
        prepared = {
            OBS_STATE: state,
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
            OBS_IMAGES: torch.stack(
                [self.normalizer.normalize_image(batch["image"])], dim=-4
            ),
        }
        if include_action:
            prepared[ACTION] = self.normalizer.normalize_action_delta(
                batch["action_delta"]
            )
            prepared["action_is_pad"] = batch["action_is_pad"]
        if return_auxiliary:
            return prepared, grasp_logit, contact, intermediates
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if "grasp_state" not in batch:
            raise KeyError("WRM_claude training batch is missing grasp_state")
        if "action_delta" not in batch:
            raise KeyError("WRM_claude training batch is missing action_delta")
        prepared, grasp_logit, contact, intermediates = self._prepare(
            batch, include_action=True, return_auxiliary=True
        )
        diffusion_loss = self.diffusion_model.compute_loss(prepared)
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
            diffusion_loss + self.config.model.claude_grasp_loss_weight * grasp_loss
        )
        with torch.no_grad():
            probability = grasp_logit.sigmoid()
            accuracy = (
                ((probability >= 0.5) == (grasp_target >= 0.5)).float().mean()
            )
        return total_loss, {
            "loss": float(total_loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "smoothness_loss": self.diffusion_model.last_smoothness_loss,
            "grasp_loss": float(grasp_loss.detach()),
            "grasp_accuracy": float(accuracy),
            "grasp_positive_rate": float((grasp_target >= 0.5).float().mean()),
            "predicted_grasp_positive_rate": float(
                (probability >= 0.5).float().mean()
            ),
            "contact_feature_std": float(
                contact.detach().std(unbiased=False)
            ),
            "contact_valid_fraction": float(
                intermediates["cell_available"].float().mean()
            ),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._prepare(batch, include_action=False)
        normalized = self.diffusion_model.generate_actions(prepared)
        return self.normalizer.denormalize_action_delta(normalized)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    def _append_online_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        signature = tuple(
            batch[key].data_ptr()
            for key in ("state", "beaver_distance", "beaver_status", "beaver_present")
        )
        frame = {
            "state": batch["state"],
            "distance": batch["beaver_distance"],
            "status": batch["beaver_status"],
            "present": batch["beaver_present"],
        }
        if signature != self._last_online_frame_signature:
            self._history.append(frame)
            self._last_online_frame_signature = signature
        frames = list(self._history)
        history_steps = self.config.model.claude_history_steps
        padded = [frames[0]] * (history_steps - len(frames)) + frames[-history_steps:]
        delta_steps = self.config.model.claude_motion_delta_steps
        previous = (
            frames[0] if len(frames) <= delta_steps else frames[-delta_steps - 1]
        )
        claude_batch = dict(batch)
        claude_batch["delta_q"] = batch["state"] - previous["state"]
        for output_key, frame_key in (
            ("beaver_history_distance", "distance"),
            ("beaver_history_status", "status"),
            ("beaver_history_present", "present"),
        ):
            claude_batch[output_key] = torch.stack(
                [history_frame[frame_key] for history_frame in padded], dim=1
            )
        return claude_batch

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        from policies.realman_beaver.modeling import _ensure_observation_batch
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        claude_batch = self._append_online_history(batch)
        current_state = batch["state"]
        prepared = self._prepare(claude_batch, include_action=False)

        # Deployment observations arrive one tick at a time; the policy owns
        # the n_obs_steps conditioning queue (same pattern as FMPolicy).
        frame = {
            OBS_STATE: prepared[OBS_STATE],
            self.config.dataset.image_key: prepared[
                self.config.dataset.image_key
            ],
        }
        self._conditioning_queue.append(frame)
        while len(self._conditioning_queue) < self._conditioning_queue.maxlen:
            self._conditioning_queue.append(frame)

        replanned = not self._action_queue
        if replanned:
            frames = list(self._conditioning_queue)
            replan_batch = {
                OBS_STATE: torch.stack([f[OBS_STATE] for f in frames], dim=1),
                self.config.dataset.image_key: torch.stack(
                    [f[self.config.dataset.image_key] for f in frames], dim=1
                ),
            }
            replan_batch[OBS_IMAGES] = replan_batch[
                self.config.dataset.image_key
            ].unsqueeze(-4)
            normalized = self.diffusion_model.generate_actions(replan_batch)
            deltas = self.normalizer.denormalize_action_delta(normalized)
            self._anchor = current_state.clone()
            self._accumulated = torch.zeros_like(current_state)
            self._action_queue.extend(deltas.transpose(0, 1))
            self._chunk_len = len(self._action_queue)

        delta = self._action_queue.popleft()
        self._accumulated = self._accumulated + delta
        target = self._anchor + self._accumulated
        self.last_replanned = replanned
        self.last_chunk_step = self._chunk_len - len(self._action_queue) - 1
        self.last_latency_steps = 0
        self.last_delta_max = float(delta.abs().max())
        self.last_accumulated_delta_norm = float(self._accumulated.abs().max())
        return target

    def reset(self) -> None:
        history_length = max(
            self.config.model.claude_history_steps,
            self.config.model.claude_motion_delta_steps + 1,
        )
        self._history: deque[dict[str, Tensor]] = deque(maxlen=history_length)
        self._conditioning_queue: deque[dict[str, Tensor]] = deque(
            maxlen=self.config.model.n_obs_steps
        )
        self._action_queue: deque[Tensor] = deque(
            maxlen=self.config.model.n_action_steps
        )
        self._batch_size: int | None = None
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self._anchor: Tensor | None = None
        self._accumulated: Tensor | None = None
        self._chunk_len = 0
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_latency_steps = 0
        self.last_grasp_probability = 0.0
        self.last_contact_feature_std = 0.0
        self.last_contact_valid_fraction = 0.0
        self.last_delta_max = 0.0
        self.last_accumulated_delta_norm = 0.0

