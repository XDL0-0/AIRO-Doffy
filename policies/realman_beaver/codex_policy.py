"""WRM_codex: masked temporal contact fusion with residual plan ensembling."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from policies.realman_beaver.configuration import (
    CODEX_BEAVER_VARIANT,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import (
    ObservationNormalizer,
    resolve_beaver_sensor_indices,
)


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )


class CodexVisionEncoder(nn.Module):
    """Small scratch-trained RGB encoder; no external or pretrained weights."""

    def __init__(self, width: int, output_dim: int) -> None:
        super().__init__()
        channels = (width, 2 * width, 4 * width, 8 * width)
        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(_ConvBlock(in_channels, out_channels, stride=2))
            in_channels = out_channels
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.projection(self.pool(self.backbone(image)))


class CodexTemporalBeaverEncoder(nn.Module):
    """Causal per-pixel Beaver encoder with explicit validity and weak flags.

    Invalid geometry is zero before any learned layer. Sensors are pooled with
    validity-quality-weighted attention, and a completely invalid history has
    an exactly zero output even though the learned layers contain biases.
    """

    def __init__(
        self,
        *,
        n_sensors: int,
        sensor_indices: Sequence[int],
        history_steps: int,
        valid_statuses: Sequence[int],
        hidden_dim: int,
        token_dim: int,
        sensor_layers: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in sensor_indices)
        statuses = tuple(int(status) for status in valid_statuses)
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("WRM_codex sensor indices must be non-empty and unique")
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError("WRM_codex sensor index is outside the Beaver layout")
        if history_steps <= 0 or hidden_dim <= 0 or token_dim <= 0:
            raise ValueError("WRM_codex Beaver dimensions must be positive")
        if not statuses:
            raise ValueError("WRM_codex requires at least one valid status")

        self.n_sensors = int(n_sensors)
        self.n_selected_sensors = len(indices)
        self.history_steps = int(history_steps)
        self.token_dim = int(token_dim)
        self.register_buffer("sensor_index", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "valid_status_values", torch.tensor(statuses), persistent=False
        )
        self.register_buffer("distance_p5", torch.zeros(len(indices)))
        self.register_buffer("distance_p95", torch.full((len(indices),), 2550.0))
        self.register_buffer("distance_median", torch.full((len(indices),), 1275.0))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        self.register_buffer(
            "proximity_scales_mm",
            torch.tensor((50.0, 100.0, 200.0, 400.0), dtype=torch.float32),
        )

        # normalized range, four proximity scales, adjacent delta, validity,
        # delta validity, weak-status flag, and raw-zero flag.
        self.cell_feature_dim = 10
        self.frame_mlp = nn.Sequential(
            nn.Linear(16 * self.cell_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.sensor_embedding = nn.Parameter(
            torch.randn(len(indices), token_dim) * 0.02
        )
        self.temporal_gru = nn.GRU(token_dim, token_dim, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=2 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.sensor_transformer = nn.TransformerEncoder(
            layer,
            num_layers=sensor_layers,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(token_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(2 * token_dim, token_dim),
            nn.SiLU(),
            nn.LayerNorm(token_dim),
        )

    @torch.no_grad()
    def set_normalization_statistics(
        self, *, p5: Tensor, p95: Tensor, median: Tensor
    ) -> None:
        converted: dict[str, Tensor] = {}
        for name, value in {"p5": p5, "p95": p95, "median": median}.items():
            tensor = torch.as_tensor(
                value, dtype=self.distance_p5.dtype, device=self.distance_p5.device
            ).flatten()
            if tensor.shape != (self.n_selected_sensors,):
                raise ValueError(
                    f"WRM_codex Beaver {name} must have shape "
                    f"({self.n_selected_sensors},), got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"WRM_codex Beaver {name} must be finite")
            converted[name] = tensor
        if torch.any(converted["p95"] <= converted["p5"]):
            raise ValueError("WRM_codex Beaver p95 must exceed p5 per sensor")
        self.distance_p5.copy_(converted["p5"])
        self.distance_p95.copy_(converted["p95"])
        self.distance_median.copy_(converted["median"])
        self.normalization_fitted.fill_(True)

    def _select(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if distance.ndim < 4 or distance.shape[-2:] != (4, 4):
            raise ValueError(
                "WRM_codex Beaver distance must end in [history, sensor, 4, 4]"
            )
        if distance.shape[-4] != self.history_steps:
            raise ValueError(
                f"Expected {self.history_steps} Beaver frames, got "
                f"{distance.shape[-4]}"
            )
        if status.shape != distance.shape or present.shape != distance.shape[:-2]:
            raise ValueError("WRM_codex Beaver distance/status/present shapes differ")
        sensor_count = distance.shape[-3]
        if sensor_count == self.n_sensors:
            index = self.sensor_index
            return (
                distance.index_select(-3, index),
                status.index_select(-3, index),
                present.index_select(-1, index),
            )
        if sensor_count == self.n_selected_sensors:
            return distance, status, present
        raise ValueError(
            "WRM_codex Beaver input does not contain the configured sensor layout"
        )

    def preprocess(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not bool(self.normalization_fitted.item()):
            raise RuntimeError(
                "WRM_codex Beaver train-split normalization statistics are not fitted"
            )
        distance, status, present = self._select(distance, status, present)
        statuses = self.valid_status_values.to(device=status.device, dtype=status.dtype)
        status_valid = (status.unsqueeze(-1) == statuses).any(dim=-1)
        raw_zero = distance.eq(0)
        valid = (
            status_valid
            & present.bool().unsqueeze(-1).unsqueeze(-1)
            & torch.isfinite(distance)
            & ~raw_zero
        )

        statistic_shape = [1] * (distance.ndim - 3) + [
            self.n_selected_sensors,
            1,
            1,
        ]
        p5 = self.distance_p5.to(distance.dtype).view(*statistic_shape)
        p95 = self.distance_p95.to(distance.dtype).view(*statistic_shape)
        normalized = ((distance - p5) / (p95 - p5)).clamp(0.0, 1.0)
        observed = torch.where(valid, normalized, torch.zeros_like(normalized))

        first_delta = torch.zeros_like(observed[..., :1, :, :, :])
        delta_valid_tail = valid[..., 1:, :, :, :] & valid[..., :-1, :, :, :]
        delta_tail = torch.where(
            delta_valid_tail,
            normalized[..., 1:, :, :, :] - normalized[..., :-1, :, :, :],
            torch.zeros_like(normalized[..., 1:, :, :, :]),
        )
        delta = torch.cat((first_delta, delta_tail), dim=-4)
        delta_valid = torch.cat(
            (torch.zeros_like(valid[..., :1, :, :, :]), delta_valid_tail), dim=-4
        )

        scale_shape = [1] * distance.ndim + [len(self.proximity_scales_mm)]
        scales = self.proximity_scales_mm.to(distance.dtype).view(*scale_shape)
        proximity = torch.exp(-distance.clamp_min(0.0).unsqueeze(-1) / scales)
        proximity = proximity * valid.to(distance.dtype).unsqueeze(-1)
        weak = (status == 9) & valid
        features = torch.cat(
            (
                observed.unsqueeze(-1),
                proximity,
                delta.unsqueeze(-1),
                valid.to(distance.dtype).unsqueeze(-1),
                delta_valid.to(distance.dtype).unsqueeze(-1),
                weak.to(distance.dtype).unsqueeze(-1),
                raw_zero.to(distance.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        return features, {
            "selected_distance": distance,
            "valid": valid,
            "delta_valid": delta_valid,
            "weak": weak,
            "raw_zero": raw_zero,
            "observed_normalized": observed,
            "proximity": proximity,
            "sensor_quality": valid.to(distance.dtype).mean(dim=(-4, -2, -1)),
            "sensor_available": valid.any(dim=(-4, -2, -1)),
        }

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        *,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features, intermediates = self.preprocess(distance, status, present)
        frame_vectors = features.flatten(start_dim=-3)
        frame_tokens = self.frame_mlp(frame_vectors)
        embedding_shape = [1] * (frame_tokens.ndim - 2) + [
            self.n_selected_sensors,
            self.token_dim,
        ]
        frame_tokens = frame_tokens + self.sensor_embedding.view(*embedding_shape)
        sensor_sequences = frame_tokens.transpose(-3, -2)
        leading_shape = sensor_sequences.shape[:-3]
        gru_input = sensor_sequences.reshape(-1, self.history_steps, self.token_dim)
        _, hidden = self.temporal_gru(gru_input)
        sensor_tokens = hidden[-1].reshape(
            *leading_shape, self.n_selected_sensors, self.token_dim
        )

        available = intermediates["sensor_available"].reshape(
            -1, self.n_selected_sensors
        )
        all_missing = ~available.any(dim=-1, keepdim=True)
        key_padding_mask = ~available
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[:, 0] &= ~all_missing.squeeze(-1)
        encoded = self.sensor_transformer(
            sensor_tokens.reshape(-1, self.n_selected_sensors, self.token_dim),
            src_key_padding_mask=key_padding_mask,
        )
        encoded = encoded * available.to(encoded.dtype).unsqueeze(-1)

        quality = intermediates["sensor_quality"].reshape(
            -1, self.n_selected_sensors
        )
        scores = self.pool_score(encoded).squeeze(-1)
        scores = scores + quality.clamp_min(1e-6).log()
        scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
        scores = torch.where(all_missing, torch.zeros_like(scores), scores)
        attention = scores.softmax(dim=-1)
        attention = torch.where(all_missing, torch.zeros_like(attention), attention)
        pooled = (attention.unsqueeze(-1) * encoded).sum(dim=-2)
        denominator = available.to(encoded.dtype).sum(dim=-1, keepdim=True).clamp_min(1)
        mean = encoded.sum(dim=-2) / denominator
        contact = self.output(torch.cat((pooled, mean), dim=-1))
        contact = contact * (~all_missing).to(contact.dtype)
        contact = contact.reshape(*leading_shape, self.token_dim)
        if return_intermediates:
            intermediates.update(
                {
                    "frame_tokens": frame_tokens,
                    "sensor_tokens": sensor_tokens,
                    "encoded_sensor_tokens": encoded.reshape(
                        *leading_shape, self.n_selected_sensors, self.token_dim
                    ),
                    "sensor_attention": attention.reshape(
                        *leading_shape, self.n_selected_sensors
                    ),
                }
            )
            return contact, intermediates
        return contact


class WRMCodexPolicy(nn.Module):
    """Deterministic RGB/joint/contact policy with overlapping residual plans."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != CODEX_BEAVER_VARIANT:
            raise ValueError(f"WRMCodexPolicy requires {CODEX_BEAVER_VARIANT}")
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.codex_beaver_sensors
        )
        if not normalizer.has_temporal_beaver_statistics:
            raise ValueError(
                "WRM_codex requires robust Beaver statistics fitted on the "
                "explicit training episodes"
            )
        fitted_indices = tuple(
            int(index)
            for index in normalizer.beaver_temporal_sensor_indices.tolist()
        )
        if fitted_indices != tuple(sensor_indices):
            raise ValueError(
                "WRM_codex normalizer sensor order does not match config: "
                f"{fitted_indices} != {tuple(sensor_indices)}"
            )

        token_dim = model.codex_token_dim
        self.vision_encoder = CodexVisionEncoder(model.codex_vision_width, token_dim)
        self.joint_encoder = nn.Sequential(
            nn.Linear(2 * model.state_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.beaver_encoder = CodexTemporalBeaverEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_indices=sensor_indices,
            history_steps=model.codex_beaver_history_steps,
            valid_statuses=dataset.beaver_valid_statuses,
            hidden_dim=model.codex_contact_hidden_dim,
            token_dim=token_dim,
            sensor_layers=model.codex_sensor_layers,
            attention_heads=model.codex_attention_heads,
            dropout=model.codex_dropout,
        )
        self.beaver_encoder.set_normalization_statistics(
            p5=normalizer.beaver_temporal_p5,
            p95=normalizer.beaver_temporal_p95,
            median=normalizer.beaver_temporal_median,
        )

        self.modality_embedding = nn.Parameter(torch.randn(3, token_dim) * 0.02)
        self.observation_embedding = nn.Parameter(
            torch.randn(model.n_obs_steps, token_dim) * 0.02
        )
        self.context_token = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=model.codex_attention_heads,
            dim_feedforward=4 * token_dim,
            dropout=model.codex_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer,
            num_layers=model.codex_fusion_layers,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=model.codex_attention_heads,
            dim_feedforward=4 * token_dim,
            dropout=model.codex_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=model.codex_decoder_layers,
            norm=nn.LayerNorm(token_dim),
        )
        self.action_queries = nn.Parameter(torch.randn(model.horizon, token_dim) * 0.02)
        self.residual_head = nn.Linear(token_dim, model.action_dim)
        self.activity_head = nn.Linear(token_dim, 1)
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
        self.beaver_encoder.set_normalization_statistics(
            p5=p5, p95=p95, median=median
        )

    @staticmethod
    def _ablate(token: Tensor, mode: str | None) -> Tensor:
        if mode is None or mode == "complete":
            return token
        if mode == "zero":
            return torch.zeros_like(token)
        if mode == "shuffle":
            if token.shape[0] > 1:
                return token.roll(1, dims=0)
            if token.shape[1] > 1:
                return token.flip(1)
            return torch.zeros_like(token)
        raise ValueError(f"Unknown WRM_codex ablation mode: {mode}")

    def _encode(
        self,
        batch: Mapping[str, Tensor],
        ablations: Mapping[str, str] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        required = {
            "image",
            "state",
            "beaver_history_distance",
            "beaver_history_status",
            "beaver_history_present",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"WRM_codex batch is missing fields: {sorted(missing)}")
        image, state = batch["image"], batch["state"]
        if image.ndim != 5 or state.ndim != 3:
            raise ValueError("WRM_codex image/state must be [B,O,C,H,W] and [B,O,7]")
        if image.shape[:2] != state.shape[:2]:
            raise ValueError("WRM_codex image/state observation axes do not match")
        batch_size, obs_steps = state.shape[:2]
        if obs_steps != self.config.model.n_obs_steps:
            raise ValueError(
                f"Expected {self.config.model.n_obs_steps} observations, got {obs_steps}"
            )
        normalized_image = self.normalizer.normalize_image(image)
        resize_shape = tuple(self.config.model.resize_shape)
        flat_image = normalized_image.flatten(0, 1)
        if tuple(flat_image.shape[-2:]) != resize_shape:
            flat_image = F.interpolate(
                flat_image, size=resize_shape, mode="bilinear", align_corners=False
            )
        image_token = self.vision_encoder(flat_image).reshape(
            batch_size, obs_steps, -1
        )

        normalized_state = self.normalizer.normalize_state(state)
        state_delta = torch.cat(
            (
                torch.zeros_like(state[:, :1]),
                state[:, 1:] - state[:, :-1],
            ),
            dim=1,
        ) / self.normalizer.state_scale
        joint_token = self.joint_encoder(
            torch.cat((normalized_state, state_delta), dim=-1)
        )
        beaver_token, beaver_intermediates = self.beaver_encoder(
            batch["beaver_history_distance"],
            batch["beaver_history_status"],
            batch["beaver_history_present"],
            return_intermediates=True,
        )

        ablations = ablations or {}
        image_token = self._ablate(image_token, ablations.get("image"))
        joint_token = self._ablate(joint_token, ablations.get("joint"))
        beaver_token = self._ablate(beaver_token, ablations.get("beaver"))
        tokens = torch.stack((image_token, joint_token, beaver_token), dim=2)
        tokens = (
            tokens
            + self.modality_embedding.view(1, 1, 3, -1)
            + self.observation_embedding.view(1, obs_steps, 1, -1)
        ).flatten(1, 2)
        context = self.context_token.view(1, 1, -1).expand(batch_size, 1, -1)
        memory = self.fusion(torch.cat((context, tokens), dim=1))
        return memory, beaver_intermediates

    def forward(
        self,
        batch: Mapping[str, Tensor],
        *,
        ablations: Mapping[str, str] | None = None,
    ) -> dict[str, Tensor]:
        memory, beaver_intermediates = self._encode(batch, ablations)
        batch_size = memory.shape[0]
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        residual = self.residual_head(decoded).tanh()
        activity_logit = self.activity_head(decoded).squeeze(-1)
        activity = activity_logit.sigmoid()
        current_state = batch["state"][:, -1]
        residual_scale = (
            self.normalizer.action_scale * self.config.model.codex_residual_scale
        )
        action = current_state[:, None] + (
            activity.unsqueeze(-1) * residual * residual_scale
        )
        return {
            "action": action,
            "residual": residual,
            "activity": activity,
            "activity_logit": activity_logit,
            "beaver_valid": beaver_intermediates["valid"],
            "beaver_sensor_attention": beaver_intermediates["sensor_attention"],
        }

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        prediction = self.forward(batch)
        target = batch["action"]
        pad = batch["action_is_pad"].bool()
        if target.shape != prediction["action"].shape:
            raise ValueError(
                f"WRM_codex target shape {tuple(target.shape)} differs from "
                f"prediction {tuple(prediction['action'].shape)}"
            )
        valid = ~pad
        scale = self.normalizer.action_scale.clamp_min(
            self.config.dataset.normalization_floor
        )
        action_error = F.smooth_l1_loss(
            prediction["action"] / scale,
            target / scale,
            reduction="none",
        )
        action_loss = (
            action_error * valid.unsqueeze(-1)
        ).sum() / valid.sum().clamp_min(1) / target.shape[-1]

        target_delta = (target - batch["state"][:, -1, None]).abs().amax(dim=-1)
        activity_target = (
            target_delta / self.config.model.codex_activity_scale_rad
        ).clamp(0.0, 1.0)
        activity_error = F.binary_cross_entropy_with_logits(
            prediction["activity_logit"], activity_target, reduction="none"
        )
        activity_loss = (activity_error * valid).sum() / valid.sum().clamp_min(1)

        velocity_valid = valid[:, 1:] & valid[:, :-1]
        predicted_velocity = (
            prediction["action"][:, 1:] - prediction["action"][:, :-1]
        ) / scale
        target_velocity = (target[:, 1:] - target[:, :-1]) / scale
        velocity_error = F.smooth_l1_loss(
            predicted_velocity, target_velocity, reduction="none"
        )
        velocity_loss = (
            velocity_error * velocity_valid.unsqueeze(-1)
        ).sum() / velocity_valid.sum().clamp_min(1) / target.shape[-1]

        total = (
            action_loss
            + self.config.model.codex_activity_loss_weight * activity_loss
            + self.config.model.codex_velocity_loss_weight * velocity_loss
        )
        with torch.no_grad():
            activity_mae = (
                (prediction["activity"] - activity_target).abs() * valid
            ).sum() / valid.sum().clamp_min(1)
            valid_fraction = prediction["beaver_valid"].float().mean()
        return total, {
            "loss": float(total.detach()),
            "action_loss": float(action_loss.detach()),
            "velocity_loss": float(velocity_loss.detach()),
            "activity_loss": float(activity_loss.detach()),
            "activity_mae": float(activity_mae),
            "beaver_valid_fraction": float(valid_fraction),
        }

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        ablations: Mapping[str, str] | None = None,
    ) -> Tensor:
        action = self.forward(batch, ablations=ablations)["action"]
        start = self.config.model.n_obs_steps - 1
        return action[:, start : start + self.config.model.n_action_steps]

    @torch.no_grad()
    def predict_actions(
        self,
        batch: dict[str, Tensor],
        *,
        ablations: Mapping[str, str] | None = None,
    ) -> Tensor:
        return self.predict_action_chunk(batch, ablations=ablations)

    @staticmethod
    def _ensure_observation_batch(
        observation: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        dimensions = {
            "image": 4,
            "state": 2,
            "beaver_distance": 4,
            "beaver_present": 2,
            "beaver_status": 4,
        }
        result: dict[str, Tensor] = {}
        for key, batched_ndim in dimensions.items():
            if key not in observation:
                raise KeyError(f"WRM_codex deployment observation is missing {key}")
            value = observation[key]
            if value.ndim == batched_ndim - 1:
                value = value.unsqueeze(0)
            if value.ndim != batched_ndim:
                raise ValueError(
                    f"WRM_codex {key} must have {batched_ndim - 1} or "
                    f"{batched_ndim} dimensions"
                )
            result[key] = value
        return result

    def _append_online_frame(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        signature = tuple(
            batch[key].data_ptr()
            for key in (
                "image",
                "state",
                "beaver_distance",
                "beaver_status",
                "beaver_present",
            )
        )
        frame = {
            "image": batch["image"],
            "state": batch["state"],
            "distance": batch["beaver_distance"],
            "status": batch["beaver_status"],
            "present": batch["beaver_present"],
        }
        if signature != self._last_online_frame_signature:
            self._online_history.append(frame)
            self._last_online_frame_signature = signature
        frames = list(self._online_history)
        required = (
            self.config.model.codex_beaver_history_steps
            + self.config.model.n_obs_steps
            - 1
        )
        padded = [frames[0]] * (required - len(frames)) + frames[-required:]
        obs_steps = self.config.model.n_obs_steps
        history_steps = self.config.model.codex_beaver_history_steps
        online = {
            "image": torch.stack(
                [frame["image"] for frame in padded[-obs_steps:]], dim=1
            ),
            "state": torch.stack(
                [frame["state"] for frame in padded[-obs_steps:]], dim=1
            ),
        }
        for output_key, frame_key in (
            ("beaver_history_distance", "distance"),
            ("beaver_history_status", "status"),
            ("beaver_history_present", "present"),
        ):
            windows = [
                torch.stack(
                    [history_frame[frame_key] for history_frame in padded[start : start + history_steps]],
                    dim=1,
                )
                for start in range(obs_steps)
            ]
            online[output_key] = torch.stack(windows, dim=1)
        return online

    @torch.no_grad()
    def _new_plan(self, batch: dict[str, Tensor]) -> Tensor:
        prediction = self.forward(batch)
        start = self.config.model.n_obs_steps - 1
        self.last_activity_probability = float(
            prediction["activity"][:, start].detach().mean()
        )
        self.last_beaver_valid_fraction = float(
            prediction["beaver_valid"][:, -1].float().mean()
        )
        attention = prediction["beaver_sensor_attention"][:, -1]
        self.last_sensor_attention = {
            name: float(attention[:, position].detach().mean())
            for position, name in enumerate(self.config.model.codex_beaver_sensors)
        }
        return prediction["action"][:, start:]

    @torch.no_grad()
    def select_action(self, observation: Mapping[str, Tensor]) -> Tensor:
        batch = self._ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        online = self._append_online_frame(batch)
        replanned = not self._plans or (
            self._steps_since_replan >= self.config.model.n_action_steps
        )
        if replanned:
            self._plans.append((self._control_step, self._new_plan(online)))
            self._steps_since_replan = 0

        candidates: list[Tensor] = []
        weights: list[float] = []
        newest = len(self._plans) - 1
        for plan_index, (start_step, plan) in enumerate(self._plans):
            offset = self._control_step - start_step
            if 0 <= offset < plan.shape[1]:
                candidates.append(plan[:, offset])
                age = newest - plan_index
                weights.append(math.exp(-self.config.model.codex_plan_decay * age))
        if not candidates:
            raise RuntimeError("WRM_codex has no action for the current control step")
        stacked = torch.stack(candidates, dim=0)
        weight = torch.tensor(
            weights, device=stacked.device, dtype=stacked.dtype
        ).view(-1, 1, 1)
        action = (stacked * weight).sum(dim=0) / weight.sum()
        self.last_plan_contributors = len(candidates)
        self.last_plan_disagreement = float(
            stacked.std(dim=0, unbiased=False).abs().amax().detach()
        )
        self.last_replanned = replanned
        self.last_chunk_step = self._steps_since_replan
        self._steps_since_replan += 1
        self._control_step += 1
        return action

    def reset(self) -> None:
        history_length = (
            self.config.model.codex_beaver_history_steps
            + self.config.model.n_obs_steps
            - 1
        )
        self._online_history: deque[dict[str, Tensor]] = deque(maxlen=history_length)
        self._plans: deque[tuple[int, Tensor]] = deque(
            maxlen=self.config.model.codex_plan_ensemble
        )
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self._batch_size: int | None = None
        self._control_step = 0
        self._steps_since_replan = 0
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_activity_probability = 0.0
        self.last_beaver_valid_fraction = 0.0
        self.last_plan_contributors = 0
        self.last_plan_disagreement = 0.0
        self.last_sensor_attention = {
            name: 0.0 for name in self.config.model.codex_beaver_sensors
        }


__all__ = [
    "CodexTemporalBeaverEncoder",
    "CodexVisionEncoder",
    "WRMCodexPolicy",
]
