"""WRM_antigravity: Key4 spatio-temporal contact-field diffusion policy."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from policies.realman_beaver.configuration import (
    ANTIGRAVITY_BEAVER_VARIANT,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import (
    ObservationNormalizer,
    resolve_beaver_sensor_indices,
)
from policies.realman_beaver.modules.antigravity_beaver_encoder import (
    AntigravityBeaverEncoder,
)


class AntigravityDPPolicy(nn.Module):
    """Diffusion Policy with Spatio-Temporal Kinematic Beaver Cross-Attention.

    WRM_antigravity model:
      - 12-frame temporal Beaver history on key contact sensors 01, 02, 10, 11
      - Multi-scale spatial curvature, multi-lag temporal contact flux, spatial gradients
      - Causal GRU + Inter-sensor Kinematic Cross-Attention Transformer
      - Velocity-anchored joint conditioning (q_t, Delta q_t)
      - Dual auxiliary grasp phase logit and contact enclosure metric
      - Closed-loop contact-adaptive terminal hold damping at replan boundaries
    """

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != ANTIGRAVITY_BEAVER_VARIANT:
            raise ValueError(
                f"AntigravityDPPolicy requires {ANTIGRAVITY_BEAVER_VARIANT}"
            )
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.beaver_antigravity_sensors
        )
        if not normalizer.has_temporal_beaver_statistics:
            raise ValueError(
                "WRM_antigravity requires robust training-set Beaver statistics"
            )
        fitted_indices = tuple(
            int(index)
            for index in normalizer.beaver_temporal_sensor_indices.tolist()
        )
        if fitted_indices != tuple(sensor_indices):
            raise ValueError(
                "WRM_antigravity normalizer sensor order does not match config: "
                f"{fitted_indices} != {tuple(sensor_indices)}"
            )

        self.beaver_encoder = AntigravityBeaverEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_indices=sensor_indices,
            history_steps=model.beaver_antigravity_history_steps,
            lag_steps=model.beaver_antigravity_lag_steps,
            valid_statuses=dataset.beaver_valid_statuses,
            proximity_scales_mm=model.beaver_antigravity_proximity_scales_mm,
            spatial_hidden_dim=model.beaver_antigravity_spatial_hidden_dim,
            token_dim=model.beaver_antigravity_token_dim,
            temporal_hidden_dim=model.beaver_antigravity_temporal_hidden_dim,
            transformer_layers=model.beaver_antigravity_transformer_layers,
            attention_heads=model.beaver_antigravity_attention_heads,
            output_dim=model.beaver_antigravity_feature_dim,
            noise_std_mm=model.beaver_antigravity_noise_std_mm,
            pixel_dropout=model.beaver_antigravity_pixel_dropout,
            sensor_dropout=model.beaver_antigravity_sensor_dropout,
        )
        self.beaver_encoder.set_normalization_statistics(
            p5=normalizer.beaver_temporal_p5,
            p95=normalizer.beaver_temporal_p95,
            median=normalizer.beaver_temporal_median,
        )

        grasp_input_dim = model.beaver_antigravity_feature_dim + 2 * model.state_dim
        self.grasp_state_head = nn.Sequential(
            nn.Linear(grasp_input_dim, model.beaver_antigravity_grasp_hidden_dim),
            nn.SiLU(),
            nn.Linear(model.beaver_antigravity_grasp_hidden_dim, 1),
        )

        from policies.realman_beaver.modeling import build_native_diffusion_config
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        self.native_policy = DiffusionPolicy(build_native_diffusion_config(config))
        self.reset()

    @torch.no_grad()
    def set_temporal_statistics(
        self, *, p5: Tensor, p95: Tensor, median: Tensor
    ) -> None:
        self.normalizer.set_temporal_beaver_statistics(
            {
                "p5": p5,
                "p95": p95,
                "median": median,
                "sensor_indices": self.beaver_encoder.sensor_index,
            }
        )
        self.beaver_encoder.set_normalization_statistics(p5=p5, p95=p95, median=median)

    def _state_and_auxiliaries(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        required = {
            "delta_q",
            "beaver_history_distance",
            "beaver_history_status",
            "beaver_history_present",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"WRM_antigravity batch is missing fields: {sorted(missing)}")

        z_beaver, intermediates = self.beaver_encoder(
            batch["beaver_history_distance"],
            batch["beaver_history_status"],
            batch["beaver_history_present"],
            return_intermediates=True,
        )
        normalized_qpos = self.normalizer.normalize_state(batch["state"])
        normalized_delta_q = batch["delta_q"] / self.normalizer.state_scale

        if "delta_q_long" in batch:
            normalized_delta_q_long = batch["delta_q_long"] / self.normalizer.state_scale
        else:
            normalized_delta_q_long = normalized_delta_q

        grasp_input = torch.cat(
            (z_beaver, normalized_qpos, normalized_delta_q_long), dim=-1
        )
        grasp_logit = self.grasp_state_head(grasp_input).squeeze(-1)
        grasp_probability = grasp_logit.sigmoid()
        enclosure_score = intermediates["enclosure_score"].squeeze(-1)

        conditioned_state = torch.cat(
            (
                normalized_qpos,
                normalized_delta_q,
                z_beaver,
                grasp_probability.unsqueeze(-1),
                enclosure_score.unsqueeze(-1),
            ),
            dim=-1,
        )

        attention = intermediates["sensor_attention"]
        entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=-1)

        self.last_grasp_probability = float(grasp_probability.detach().mean())
        self.last_enclosure_score = float(enclosure_score.detach().mean())
        self.last_z_beaver_std = float(z_beaver.detach().std(unbiased=False))
        self.last_sensor_attention_entropy = float(entropy.detach().mean())
        self.last_near_field_fraction = float(intermediates["near_field_fraction"].detach())
        self.last_sensor_attention = {
            sensor_name: float(
                attention[..., sensor_position].detach().mean()
            )
            for sensor_position, sensor_name in enumerate(
                self.config.model.beaver_antigravity_sensors
            )
        }

        return conditioned_state, grasp_logit, enclosure_score, z_beaver, intermediates

    def _prepare(
        self,
        batch: dict[str, Tensor],
        include_action: bool,
        *,
        return_auxiliary: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        state, grasp_logit, enclosure_score, z_beaver, intermediates = self._state_and_auxiliaries(batch)
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
            return prepared, grasp_logit, enclosure_score, z_beaver, intermediates
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if "grasp_state" not in batch:
            raise KeyError("WRM_antigravity training batch is missing grasp_state")
        prepared, grasp_logit, enclosure_score, z_beaver, intermediates = self._prepare(
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

        smooth_target = grasp_target * 0.96 + 0.02
        grasp_loss = F.binary_cross_entropy_with_logits(grasp_logit, smooth_target)

        enclosure_target = (grasp_target >= 0.5).float()
        # The encoder already applies sigmoid. BCE on probabilities is not AMP-safe.
        with torch.autocast(
            device_type=enclosure_score.device.type, enabled=False
        ):
            enclosure_loss = F.binary_cross_entropy(
                enclosure_score.float().clamp(1e-6, 1.0 - 1e-6),
                enclosure_target.float(),
            )

        total_loss = (
            diffusion_loss
            + self.config.model.beaver_antigravity_grasp_loss_weight * grasp_loss
            + self.config.model.beaver_antigravity_enclosure_loss_weight * enclosure_loss
        )

        with torch.no_grad():
            probability = grasp_logit.sigmoid()
            accuracy = (
                ((probability >= 0.5) == (grasp_target >= 0.5)).float().mean()
            )
            attention = intermediates["sensor_attention"]
            entropy = -(
                attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()

        return total_loss, {
            "loss": float(total_loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "grasp_loss": float(grasp_loss.detach()),
            "enclosure_loss": float(enclosure_loss.detach()),
            "grasp_accuracy": float(accuracy),
            "grasp_positive_rate": float((grasp_target >= 0.5).float().mean()),
            "predicted_grasp_positive_rate": float(
                (probability >= 0.5).float().mean()
            ),
            "z_beaver_std": float(z_beaver.detach().std(unbiased=False)),
            "sensor_attention_entropy": float(entropy),
            "near_field_fraction": float(intermediates["near_field_fraction"]),
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
            self._antigravity_history.append(current)
            self._last_online_frame_signature = signature
        frames = list(self._antigravity_history)
        history_steps = self.config.model.beaver_antigravity_history_steps
        padded = [frames[0]] * (history_steps - len(frames)) + frames[-history_steps:]

        delta_short = self.config.model.beaver_antigravity_motion_delta_steps
        delta_long = self.config.model.beaver_antigravity_motion_delta_long_steps
        prev_short = frames[0] if len(frames) <= delta_short else frames[-delta_short - 1]
        prev_long = frames[0] if len(frames) <= delta_long else frames[-delta_long - 1]

        antigravity_batch = dict(batch)
        antigravity_batch["delta_q"] = batch["state"] - prev_short["state"]
        antigravity_batch["delta_q_long"] = batch["state"] - prev_long["state"]
        for output_key, frame_key in (
            ("beaver_history_distance", "distance"),
            ("beaver_history_status", "status"),
            ("beaver_history_present", "present"),
        ):
            antigravity_batch[output_key] = torch.stack(
                [history_frame[frame_key] for history_frame in padded], dim=1
            )
        return antigravity_batch

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        from policies.realman_beaver.modeling import _ensure_observation_batch
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        antigravity_batch = self._append_online_history(batch)
        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        normalized = self.native_policy.select_action(
            self._prepare(antigravity_batch, include_action=False)
        )
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)

        action = self.normalizer.denormalize_action(normalized)

        if self.config.model.beaver_antigravity_terminal_hold_damping:
            hold_thresh = self.config.model.beaver_antigravity_hold_threshold
            if self.last_grasp_probability > hold_thresh and self.last_near_field_fraction > 0.2:
                if self._hold_pose is None:
                    self._hold_pose = batch["state"].clone()
                else:
                    self._hold_pose = 0.95 * self._hold_pose + 0.05 * batch["state"]

                excess = (self.last_grasp_probability - hold_thresh) / max(1.0 - hold_thresh, 1e-4)
                beta = min(excess * self.config.model.beaver_antigravity_max_damping, self.config.model.beaver_antigravity_max_damping)
                action = (1.0 - beta) * action + beta * self._hold_pose
            else:
                self._hold_pose = None

        return action

    def reset(self) -> None:
        history_length = max(
            self.config.model.beaver_antigravity_history_steps,
            self.config.model.beaver_antigravity_motion_delta_long_steps + 1,
        )
        self._antigravity_history: deque[dict[str, Tensor]] = deque(
            maxlen=history_length
        )
        self._batch_size: int | None = None
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self._chunk_len = 0
        self._hold_pose: Tensor | None = None
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_grasp_probability = 0.0
        self.last_enclosure_score = 0.0
        self.last_z_beaver_std = 0.0
        self.last_sensor_attention_entropy = 0.0
        self.last_near_field_fraction = 0.0
        self.last_sensor_attention = {
            sensor_name: 0.0
            for sensor_name in self.config.model.beaver_antigravity_sensors
        }
        self.native_policy.reset()

