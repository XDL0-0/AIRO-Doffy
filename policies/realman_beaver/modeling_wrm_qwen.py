"""WRM_qwen: relative-action temporal Beaver diffusion policy."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from policies.realman_beaver.configuration import (
    QWEN_BEAVER_VARIANT,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import (
    ObservationNormalizer,
    resolve_beaver_sensor_indices,
)
from policies.realman_beaver.modules.beaver_encoder import TemporalBeaverEncoder


class QwenBeaverDPPolicy(nn.Module):
    """Native DP that diffuses action-minus-state chunks with motion conditioning.

    WRM_qwen keeps the WRM_temporal four-sensor (01/02/10/11) 12-frame causal
    Beaver trunk and the scratch ResNet18 RGB encoder, and adds two mechanisms
    targeting the dominant rollout failures:

    * **Relative-action target.** The diffusion output is ``action - q(t)``,
      normalized with train-only per-joint min/max, and every replan re-anchors
      the chunk on the latest *measured* joint configuration. Drift therefore
      cannot accumulate across replans, and a chunk whose first delta is ~0
      starts exactly where the arm is, removing the replan-boundary discontinuity.
    * **Joint-history conditioning.** The U-Net conditions on q, the 1-frame
      delta, and the L-frame delta (``qwen_joint_history_steps``), plus the
      learned pre-grasp/hold phase probability from the tightness auxiliary.
      Explicit velocity information lets the policy distinguish approach motion
      from the stationary hold phase instead of extrapolating a 13.5 s
      trajectory, which is what breaks rollouts that run past the demo horizon.
    """

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != QWEN_BEAVER_VARIANT:
            raise ValueError(
                f"QwenBeaverDPPolicy requires model.variant={QWEN_BEAVER_VARIANT}"
            )
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.beaver_temporal_sensors
        )
        if not normalizer.has_temporal_beaver_statistics:
            raise ValueError(
                "WRM_qwen requires per-sensor robust normalization statistics "
                "fitted from explicit training episodes"
            )
        fitted_indices = tuple(
            int(index) for index in normalizer.beaver_temporal_sensor_indices.tolist()
        )
        if fitted_indices != tuple(sensor_indices):
            raise ValueError(
                "WRM_qwen normalizer sensor order does not match configured "
                f"sensors: {fitted_indices} != {tuple(sensor_indices)}"
            )
        if not normalizer.has_delta_action_statistics:
            raise ValueError(
                "WRM_qwen requires train-split relative-action (action - state) "
                "normalization statistics"
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
        self.beaver_encoder.set_normalization_statistics(
            p5=normalizer.beaver_temporal_p5,
            p95=normalizer.beaver_temporal_p95,
            median=normalizer.beaver_temporal_median,
        )
        grasp_input_dim = model.beaver_temporal_feature_dim + 3 * model.state_dim
        self.grasp_state_head = nn.Sequential(
            nn.Linear(grasp_input_dim, model.qwen_grasp_hidden_dim),
            nn.SiLU(),
            nn.Linear(model.qwen_grasp_hidden_dim, 1),
        )
        from policies.realman_beaver.modeling import build_native_diffusion_config
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
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

    @staticmethod
    def _joint_deltas(state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split the four-row joint window into q, 1-frame, and L-frame deltas.

        ``state`` is ``(B, 4, 7)`` with rows ``[t-L-1, t-2, t-1, t]``; the
        returned tensors are indexed over the two native DP observation times
        ``(t-1, t)``. The L-frame delta references the oldest window row, so
        its span is L at the first observation time and L+1 at the second —
        one frame of difference inside a 13-frame window is within the
        intended scale of the feature.
        """
        if state.ndim != 3 or state.shape[1] != 4:
            raise ValueError(
                "WRM_qwen expects a four-row joint history "
                f"([t-L-1, t-2, t-1, t]), got shape {tuple(state.shape)}"
            )
        qpos = state[:, -2:]
        delta_1 = state[:, -2:] - state[:, -3:-1]
        delta_lag = state[:, -2:] - state[:, :1]
        return qpos, delta_1, delta_lag

    def _state_and_phase(
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
                f"WRM_qwen batch is missing Beaver history fields: {sorted(missing)}"
            )
        # Training windows are (B, n_obs, H, S, 4, 4); online is (B, H, S, 4, 4).
        per_observation_windows = batch["beaver_history_distance"].ndim == 6
        beaver_feature, encoder_intermediates = self.beaver_encoder(
            batch["beaver_history_distance"],
            batch["beaver_history_status"],
            batch["beaver_history_present"],
            return_intermediates=True,
        )
        sensor_tokens = encoder_intermediates["sensor_tokens"]
        qpos, delta_1, delta_lag = self._joint_deltas(batch["state"])
        normalized_qpos = self.normalizer.normalize_state(qpos)
        normalized_delta_1 = delta_1 / self.normalizer.state_scale
        normalized_delta_lag = delta_lag / self.normalizer.state_scale
        # Online, the encoder sees a single history window ending at the
        # current tick; share it with both native DP observation times.
        if beaver_feature.dim() == 2:
            beaver_feature = beaver_feature.unsqueeze(1).expand(-1, 2, -1)
        phase_input = torch.cat(
            (beaver_feature, normalized_qpos, normalized_delta_1, normalized_delta_lag),
            dim=-1,
        )
        phase_logit = self.grasp_state_head(phase_input).squeeze(-1)
        phase_probability = phase_logit.sigmoid()
        conditioned_state = torch.cat(
            (
                normalized_qpos,
                normalized_delta_1,
                normalized_delta_lag,
                beaver_feature,
                phase_probability.unsqueeze(-1),
            ),
            dim=-1,
        )
        self.last_phase_probability = float(phase_probability.detach().mean())
        self.last_beaver_feature_mean = float(beaver_feature.detach().mean())
        self.last_beaver_feature_std = float(beaver_feature.detach().std(unbiased=False))
        self.last_sensor_token_std = {
            sensor_name: float(
                sensor_tokens[..., sensor_position, :].detach().std(unbiased=False)
            )
            for sensor_position, sensor_name in enumerate(
                self.config.model.beaver_temporal_sensors
            )
        }
        # Training batches carry one conditioned frame per native DP
        # observation time; the online batch carries the single current frame.
        if per_observation_windows:
            return conditioned_state, phase_logit, beaver_feature, sensor_tokens
        return (
            conditioned_state[:, -1],
            phase_logit[:, -1],
            beaver_feature[:, -1],
            sensor_tokens,
        )

    def _delta_target(self, batch: dict[str, Tensor]) -> Tensor:
        anchor = batch["state"][:, -1]
        return self.normalizer.normalize_delta_action(
            batch["action"] - anchor.unsqueeze(1)
        )

    def _prepare(
        self,
        batch: dict[str, Tensor],
        include_action: bool,
        *,
        return_phase: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        state, phase_logit, beaver_feature, sensor_tokens = self._state_and_phase(
            batch
        )
        prepared = {
            OBS_STATE: state,
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        if include_action:
            prepared[ACTION] = self._delta_target(batch)
            prepared["action_is_pad"] = batch["action_is_pad"]
        if return_phase:
            return prepared, phase_logit, beaver_feature, sensor_tokens
        return prepared

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if "grasp_state" not in batch:
            raise KeyError("WRM_qwen training batch is missing grasp_state")
        prepared, phase_logit, beaver_feature, sensor_tokens = self._prepare(
            batch, include_action=True, return_phase=True
        )
        diffusion_loss, _ = self.native_policy(prepared)
        grasp_target = batch["grasp_state"].to(
            device=phase_logit.device, dtype=phase_logit.dtype
        )
        if grasp_target.shape == (*phase_logit.shape, 1):
            grasp_target = grasp_target.squeeze(-1)
        if grasp_target.shape != phase_logit.shape:
            raise ValueError(
                f"grasp_state shape {tuple(grasp_target.shape)} must match "
                f"observation times {tuple(phase_logit.shape)}"
            )
        grasp_loss = F.binary_cross_entropy_with_logits(phase_logit, grasp_target)
        total_loss = (
            diffusion_loss + self.config.model.qwen_grasp_loss_weight * grasp_loss
        )
        with torch.no_grad():
            phase_probability = phase_logit.sigmoid()
            phase_accuracy = (
                ((phase_probability >= 0.5) == (grasp_target >= 0.5)).float().mean()
            )
            normalized_delta = prepared[ACTION]
        return total_loss, {
            "loss": float(total_loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "grasp_loss": float(grasp_loss.detach()),
            "grasp_accuracy": float(phase_accuracy),
            "grasp_positive_rate": float((grasp_target >= 0.5).float().mean()),
            "predicted_grasp_positive_rate": float(
                (phase_probability >= 0.5).float().mean()
            ),
            "beaver_feature_std": float(beaver_feature.detach().std(unbiased=False)),
            "delta_action_std": float(normalized_delta.detach().std(unbiased=False)),
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
        delta = self.normalizer.denormalize_delta_action(normalized)
        anchor = batch["state"][:, -1]
        return anchor.unsqueeze(1) + delta

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    def _append_online_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        lag = self.config.model.qwen_joint_history_steps
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
        count = len(frames)

        def joint_row(offset: int) -> Tensor:
            index = count - 1 - offset
            return frames[0]["state"] if index < 0 else frames[index]["state"]

        # Rows [t-L-1, t-2, t-1, t], clamped to the earliest available frame
        # exactly like LeRobot clamps training-time delta timestamps.
        state_rows = torch.stack(
            (joint_row(lag + 1), joint_row(2), joint_row(1), joint_row(0)), dim=1
        )
        history_steps = self.config.model.beaver_history_steps
        padded = [frames[0]] * (history_steps - count) + frames[-history_steps:]
        online_batch = dict(batch)
        online_batch["state"] = state_rows
        for output_key, frame_key in (
            ("beaver_history_distance", "distance"),
            ("beaver_history_status", "status"),
            ("beaver_history_present", "present"),
        ):
            online_batch[output_key] = torch.stack(
                [history_frame[frame_key] for history_frame in padded], dim=1
            )
        return online_batch

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        from policies.realman_beaver.modeling import _ensure_observation_batch
        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        online_batch = self._append_online_history(batch)

        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        if replanned:
            # The chunk is a set of displacements from the configuration
            # measured at this tick; hold that anchor until the next replan.
            self._action_anchor = batch["state"].detach()
        prepared = self._prepare(online_batch, include_action=False)
        normalized = self.native_policy.select_action(prepared)
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)
        return self._action_anchor + self.normalizer.denormalize_delta_action(normalized)

    def reset(self) -> None:
        history_length = max(
            self.config.model.beaver_history_steps,
            self.config.model.qwen_joint_history_steps + 2,
        )
        self._online_history: deque[dict[str, Tensor]] = deque(maxlen=history_length)
        self._batch_size: int | None = None
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self._action_anchor: Tensor | None = None
        self._chunk_len = 0
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_phase_probability = 0.0
        self.last_beaver_feature_mean = 0.0
        self.last_beaver_feature_std = 0.0
        self.last_sensor_token_std = {
            sensor_name: 0.0
            for sensor_name in self.config.model.beaver_temporal_sensors
        }
        self.native_policy.reset()

