"""WRM_grok: phase-gated flow matching with enclosure Beaver and overlap blending."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from policies.realman_beaver.configuration import (
    GROK_BEAVER_VARIANT,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import (
    ObservationNormalizer,
    resolve_beaver_sensor_indices,
)
from policies.realman_beaver.metrics_wrm_grok import (
    expected_calibration_error,
    phase_precision_recall_f1,
)
from policies.realman_beaver.modules.grok_phase_encoder import (
    ENCLOSURE_DIM,
    PHASE_HOLD,
    GrokPhaseEncoder,
    derive_phase_labels,
    grok_conditioned_state_dim,
)


def _blend_overlap(
    new_window: Tensor, leftover: Tensor | None, blend_steps: int
) -> tuple[Tensor, float]:
    """Cosine-style mix of a previous unused tail into a newly sampled prefix."""
    if leftover is None or leftover.shape[1] == 0 or blend_steps <= 0:
        return new_window, 0.0
    overlap = min(int(blend_steps), leftover.shape[1], new_window.shape[1])
    if overlap <= 0:
        return new_window, 0.0
    weights = torch.linspace(
        1.0 / (overlap + 1),
        overlap / (overlap + 1),
        overlap,
        device=new_window.device,
        dtype=new_window.dtype,
    ).view(1, overlap, 1)
    blended = new_window.clone()
    blended[:, :overlap] = (
        weights * new_window[:, :overlap] + (1.0 - weights) * leftover[:, :overlap]
    )
    mix = float((new_window[:, :overlap] - leftover[:, :overlap]).abs().mean().item())
    return blended, mix


class WRMGrokPolicy(nn.Module):
    """Closed-loop flow policy with phase-gated Key4 enclosure fusion."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != GROK_BEAVER_VARIANT:
            raise ValueError(
                f"WRMGrokPolicy requires model.variant={GROK_BEAVER_VARIANT}"
            )
        from policies.realman_beaver.modeling import (
            FlowMatchingModel,
            build_flow_network_config,
        )

        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.beaver_grok_sensors
        )
        if not normalizer.has_temporal_beaver_statistics:
            raise ValueError(
                "WRM_grok requires per-sensor robust normalization statistics "
                "fitted from explicit training episodes"
            )
        fitted_indices = tuple(
            int(index) for index in normalizer.beaver_temporal_sensor_indices.tolist()
        )
        if fitted_indices != tuple(sensor_indices):
            raise ValueError(
                "WRM_grok normalizer sensor order does not match config: "
                f"{fitted_indices} != {tuple(sensor_indices)}"
            )
        self.beaver_encoder = GrokPhaseEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_indices=sensor_indices,
            history_steps=model.beaver_grok_history_steps,
            valid_statuses=dataset.beaver_valid_statuses,
            frame_hidden_dim=model.beaver_grok_frame_hidden_dim,
            frame_feature_dim=model.beaver_grok_frame_feature_dim,
            temporal_hidden_dim=model.beaver_grok_temporal_hidden_dim,
            output_dim=model.beaver_grok_feature_dim,
            enclosure_dim=model.beaver_grok_enclosure_dim,
            near_scales_mm=model.beaver_grok_near_scales_mm,
            wrap_threshold_mm=model.beaver_grok_wrap_threshold_mm,
            noise_std_mm=model.beaver_grok_noise_std_mm,
        )
        self.beaver_encoder.set_normalization_statistics(
            p5=normalizer.beaver_temporal_p5,
            p95=normalizer.beaver_temporal_p95,
            median=normalizer.beaver_temporal_median,
        )
        phase_input_dim = (
            model.beaver_grok_feature_dim
            + model.beaver_grok_enclosure_dim
            + 2 * model.state_dim
        )
        self.phase_head = nn.Sequential(
            nn.Linear(phase_input_dim, model.beaver_grok_phase_hidden_dim),
            nn.SiLU(),
            nn.Linear(model.beaver_grok_phase_hidden_dim, 3),
        )
        self.phase_gate = nn.Linear(3, 2 * model.beaver_grok_feature_dim)
        self.flow = FlowMatchingModel(
            build_flow_network_config(config),
            num_inference_steps=model.flow_num_inference_steps,
            time_embedding_scale=model.flow_time_embedding_scale,
            clip_sample_range=model.clip_sample_range,
        )
        self.reset()

    def _state_and_phase(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        required = {
            "beaver_history_distance",
            "beaver_history_status",
            "beaver_history_present",
            "delta_q",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"WRM_grok batch is missing fields: {sorted(missing)}")
        beaver_feature, intermediates = self.beaver_encoder(
            batch["beaver_history_distance"],
            batch["beaver_history_status"],
            batch["beaver_history_present"],
            return_intermediates=True,
        )
        enclosure = intermediates["enclosure"]
        contact_quality = intermediates["contact_quality"]
        normalized_qpos = self.normalizer.normalize_state(batch["state"])
        normalized_delta_q = batch["delta_q"] / self.normalizer.state_scale
        phase_input = torch.cat(
            (beaver_feature, enclosure, normalized_qpos, normalized_delta_q), dim=-1
        )
        phase_logit = self.phase_head(phase_input)
        phase_prob = torch.softmax(phase_logit, dim=-1)
        gate = self.phase_gate(phase_prob)
        scale, shift = gate.chunk(2, dim=-1)
        gated_beaver = beaver_feature * (1.0 + torch.tanh(scale)) + shift
        gated_beaver = torch.where(
            intermediates["all_invalid"].unsqueeze(-1),
            torch.zeros_like(gated_beaver),
            gated_beaver,
        )
        conditioned_state = torch.cat(
            (
                normalized_qpos,
                normalized_delta_q,
                gated_beaver,
                enclosure,
                phase_prob,
                contact_quality.unsqueeze(-1),
            ),
            dim=-1,
        )
        expected = grok_conditioned_state_dim(self.config.model)
        if conditioned_state.shape[-1] != expected:
            raise RuntimeError(
                f"WRM_grok state dim {conditioned_state.shape[-1]} != {expected}"
            )
        if enclosure.shape[-1] != ENCLOSURE_DIM:
            raise RuntimeError(
                f"WRM_grok enclosure dim {enclosure.shape[-1]} != {ENCLOSURE_DIM}"
            )
        hold_prob = phase_prob[..., PHASE_HOLD]
        self.last_phase_probability = {
            "approach": float(phase_prob[..., 0].detach().mean()),
            "wrap": float(phase_prob[..., 1].detach().mean()),
            "hold": float(hold_prob.detach().mean()),
        }
        self.last_grasp_probability = float(hold_prob.detach().mean())
        self.last_contact_quality = float(contact_quality.detach().mean())
        self.last_enclosure_score = float(enclosure[..., 10].detach().mean())
        self.last_beaver_feature_std = float(
            beaver_feature.detach().std(unbiased=False)
        )
        return (
            conditioned_state,
            phase_logit,
            beaver_feature,
            enclosure,
            contact_quality,
        )

    def _prepare(
        self,
        batch: dict[str, Tensor],
        include_action: bool,
        *,
        return_auxiliary: bool = False,
    ):
        state, phase_logit, beaver_feature, enclosure, contact_quality = (
            self._state_and_phase(batch)
        )
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
            return prepared, phase_logit, beaver_feature, enclosure, contact_quality
        return prepared

    def _add_image_axis(self, prepared: dict[str, Tensor]) -> dict[str, Tensor]:
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        return prepared

    def _phase_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if "grasp_state" not in batch:
            raise KeyError("WRM_grok training batch is missing grasp_state")
        tightness = batch["grasp_state"]
        if tightness.ndim == 3 and tightness.shape[-1] == 1:
            tightness = tightness.squeeze(-1)
        return derive_phase_labels(
            tightness,
            batch["beaver_history_distance"],
            batch["beaver_history_status"],
            batch["beaver_history_present"],
            sensor_index=self.beaver_encoder.sensor_index,
            valid_statuses=self.config.dataset.beaver_valid_statuses,
            wrap_threshold_mm=self.config.model.beaver_grok_wrap_threshold_mm,
            min_near_sensors=self.config.model.beaver_grok_min_near_sensors,
        )

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        prepared, phase_logit, beaver_feature, enclosure, contact_quality = (
            self._prepare(batch, include_action=True, return_auxiliary=True)
        )
        prepared = self._add_image_axis(prepared)
        trajectory = prepared[ACTION]
        noise = torch.randn_like(trajectory)
        time = torch.rand(
            trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype
        )
        path_time = time[:, None, None]
        interpolated = (1.0 - path_time) * noise + path_time * trajectory
        target_velocity = trajectory - noise
        condition = self.flow._prepare_global_conditioning(prepared)
        predicted_velocity = self.flow.velocity_field(
            interpolated, self.flow._network_time(time), global_cond=condition
        )
        fm_error = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        pad = prepared["action_is_pad"].bool()
        valid = (~pad).unsqueeze(-1).expand_as(fm_error)
        flow_loss = (fm_error * valid).sum() / valid.sum().clamp_min(1)
        implied = interpolated + (1.0 - path_time) * predicted_velocity
        implied_rad = self.normalizer.denormalize_action(implied)
        phase_target = self._phase_targets(batch)
        if phase_target.shape != phase_logit.shape[:-1]:
            raise ValueError(
                f"phase label shape {tuple(phase_target.shape)} must match "
                f"{tuple(phase_logit.shape[:-1])}"
            )
        phase_loss = F.cross_entropy(
            phase_logit.reshape(-1, 3), phase_target.reshape(-1)
        )
        delta = implied_rad[:, 1:] - implied_rad[:, :-1]
        smooth_valid = (~pad[:, 1:] & ~pad[:, :-1]).unsqueeze(-1)
        smooth_loss = (delta.pow(2) * smooth_valid).sum() / smooth_valid.sum().clamp_min(
            1
        )
        current_rad = batch["state"][:, -1:, :]
        hold_mask = (phase_target[:, -1] == PHASE_HOLD).to(dtype=implied_rad.dtype)
        hold_err = (implied_rad - current_rad).pow(2)
        hold_valid = (~pad).unsqueeze(-1) * hold_mask[:, None, None]
        hold_loss = (hold_err * hold_valid).sum() / hold_valid.sum().clamp_min(1)
        model = self.config.model
        total = (
            flow_loss
            + model.beaver_grok_phase_loss_weight * phase_loss
            + model.beaver_grok_smooth_loss_weight * smooth_loss
            + model.beaver_grok_hold_loss_weight * hold_loss
        )
        with torch.no_grad():
            phase_prob = torch.softmax(phase_logit, dim=-1)
            phase_metrics = phase_precision_recall_f1(phase_logit, phase_target)
            ece = expected_calibration_error(phase_prob, phase_target)
        return total, {
            "loss": float(total.detach()),
            "flow_matching_loss": float(flow_loss.detach()),
            "phase_loss": float(phase_loss.detach()),
            "smooth_loss": float(smooth_loss.detach()),
            "hold_loss": float(hold_loss.detach()),
            "phase_ece": ece,
            "contact_quality": float(contact_quality.detach().mean()),
            "enclosure_score": float(enclosure[..., 10].detach().mean()),
            "beaver_feature_std": float(
                beaver_feature.detach().std(unbiased=False)
            ),
            **phase_metrics,
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._add_image_axis(self._prepare(batch, include_action=False))
        normalized = self.flow.generate_actions(prepared)
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
            self._online_history.append(current)
            self._last_online_frame_signature = signature
        frames = list(self._online_history)
        history_steps = self.config.model.beaver_grok_history_steps
        padded = [frames[0]] * (history_steps - len(frames)) + frames[-history_steps:]
        delta_steps = self.config.model.beaver_grok_motion_delta_steps
        previous = frames[0] if len(frames) <= delta_steps else frames[-delta_steps - 1]
        grok_batch = dict(batch)
        grok_batch["delta_q"] = batch["state"] - previous["state"]
        for output_key, frame_key in (
            ("beaver_history_distance", "distance"),
            ("beaver_history_status", "status"),
            ("beaver_history_present", "present"),
        ):
            grok_batch[output_key] = torch.stack(
                [history_frame[frame_key] for history_frame in padded], dim=1
            )
        return grok_batch

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        from policies.realman_beaver.modeling import _ensure_observation_batch

        batch = _ensure_observation_batch(observation)
        missing = {"beaver_distance", "beaver_status", "beaver_present"} - batch.keys()
        if missing:
            raise KeyError(
                f"WRM_grok deployment observation is missing Beaver fields: "
                f"{sorted(missing)}"
            )
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        grok_batch = self._append_online_history(batch)
        prepared = self._prepare(grok_batch, include_action=False)
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
            global_cond = self.flow._prepare_global_conditioning(
                self._add_image_axis(history)
            )
            full = self.flow.conditional_sample(batch_size, global_cond=global_cond)
            start = self.config.model.n_obs_steps - 1
            window = full[:, start:]
            window, mix = _blend_overlap(
                window,
                self._leftover,
                self.config.model.beaver_grok_overlap_blend_steps,
            )
            self.last_overlap_blend = mix
            execute = window[:, : self.config.model.n_action_steps]
            leftover_start = self.config.model.n_action_steps
            self._leftover = (
                window[:, leftover_start:]
                if leftover_start < window.shape[1]
                else None
            )
            self._action_queue.extend(execute.transpose(0, 1))
            self._chunk_len = len(self._action_queue)

        action = self._action_queue.popleft()
        self.last_replanned = replanned
        self.last_chunk_step = self._chunk_len - len(self._action_queue) - 1
        if not replanned:
            self.last_overlap_blend = 0.0
        return self.normalizer.denormalize_action(action)

    def reset(self) -> None:
        history_length = max(
            self.config.model.beaver_grok_history_steps,
            self.config.model.beaver_grok_motion_delta_steps + 1,
        )
        self._online_history: deque[dict[str, Tensor]] = deque(maxlen=history_length)
        self._observation_queue: deque[dict[str, Tensor]] = deque(
            maxlen=self.config.model.n_obs_steps
        )
        self._action_queue: deque[Tensor] = deque(
            maxlen=self.config.model.n_action_steps
        )
        self._leftover: Tensor | None = None
        self._batch_size: int | None = None
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self._chunk_len = 0
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_overlap_blend = 0.0
        self.last_grasp_probability = 0.0
        self.last_contact_quality = 0.0
        self.last_enclosure_score = 0.0
        self.last_beaver_feature_std = 0.0
        self.last_phase_probability = {
            "approach": 0.0,
            "wrap": 0.0,
            "hold": 0.0,
        }
