"""Spatio-temporal kinematic Beaver contact-field encoder for WRM_antigravity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AntigravityBeaverEncoder(nn.Module):
    """Spatio-temporal kinematic contact-field encoder for WRM_antigravity.

    Fuses four inner-contact physical sensors (01, 02, 10, 11) using:
      1. Per-pixel VL53L7CX validity masking (status 5 and 9 valid, all else masked),
         causal forward-fill imputation, and train-split robust percentile scaling;
      2. Multi-scale exponential proximity fields (25, 75, 150, 300 mm) capturing
         sub-centimetre contact resolution;
      3. Multi-baseline temporal contact flux (1, 3, 6, 11 frames) capturing approach
         and contact velocity dynamics;
      4. Spatial 4x4 surface curvature and gradient features capturing cylinder tangency;
      5. Sensor-relative and global-relative spatial deviations;
      6. Per-sensor causal GRU temporal sequence modeling over the 12-frame horizon;
      7. Inter-sensor kinematic cross-attention transformer across the 4 arm links;
      8. Contact-quality weighted attention pooling with guaranteed zero-fallback on
         all-invalid frames.
    """

    def __init__(
        self,
        *,
        n_sensors: int = 9,
        sensor_indices: Sequence[int] = (1, 2, 5, 6),
        history_steps: int = 12,
        lag_steps: Sequence[int] = (1, 3, 6, 11),
        valid_statuses: Sequence[int] = (5, 9),
        proximity_scales_mm: Sequence[float] = (25.0, 75.0, 150.0, 300.0),
        spatial_hidden_dim: int = 128,
        token_dim: int = 64,
        temporal_hidden_dim: int = 64,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        output_dim: int = 64,
        noise_std_mm: float = 5.0,
        pixel_dropout: float = 0.05,
        sensor_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in sensor_indices)
        lags = tuple(int(lag) for lag in lag_steps)
        statuses = tuple(int(status) for status in valid_statuses)
        scales = tuple(float(scale) for scale in proximity_scales_mm)

        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Antigravity Beaver sensor indices must be non-empty and unique")
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError(f"Antigravity Beaver sensor index is out of range [0, {n_sensors})")
        if not statuses:
            raise ValueError("valid_statuses must not be empty")
        if not lags or any(lag <= 0 or lag >= history_steps for lag in lags):
            raise ValueError("lag_steps must be positive and shorter than history_steps")
        if len(set(lags)) != len(lags):
            raise ValueError("lag_steps must be unique")
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("proximity scales must be positive")
        if output_dim <= 0 or spatial_hidden_dim <= 0 or token_dim <= 0 or temporal_hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if token_dim % attention_heads != 0:
            raise ValueError(f"token_dim ({token_dim}) must be divisible by attention_heads ({attention_heads})")

        self.n_sensors = int(n_sensors)
        self.n_selected_sensors = len(indices)
        self.history_steps = int(history_steps)
        self.lag_steps = lags
        self.output_dim = int(output_dim)
        self.token_dim = int(token_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.noise_std_mm = float(noise_std_mm)
        self.pixel_dropout = float(pixel_dropout)
        self.sensor_dropout = float(sensor_dropout)

        self.register_buffer("sensor_index", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "valid_status_values", torch.tensor(statuses), persistent=False
        )
        self.register_buffer(
            "proximity_scales_mm", torch.tensor(scales, dtype=torch.float32)
        )

        # Robust normalization buffers (P5, P95, Median per sensor) fitted from train split only
        self.register_buffer("distance_p5", torch.zeros(len(indices)))
        self.register_buffer("distance_p95", torch.full((len(indices),), 2550.0))
        self.register_buffer("distance_median", torch.full((len(indices),), 1275.0))
        self.register_buffer("normalization_fitted", torch.tensor(False))

        # Per-cell channels:
        #  1: Normalized distance
        #  len(scales): Multi-scale exponential proximity fields (4)
        #  len(lags): Multi-lag temporal flux deltas (4)
        #  len(lags): Multi-lag delta validity flags (4)
        #  2: Spatial gradient dx, dy (2)
        #  2: Sensor-relative and global-relative spatial deviations (2)
        #  1: Instantaneous genuine validity flag (1)
        #  1: Raw zero flag (1)
        # Total per cell = 1 + 4 + 4 + 4 + 2 + 2 + 1 + 1 = 19
        self.cell_feature_dim = 1 + len(scales) + 2 * len(lags) + 2 + 2 + 1 + 1
        self.frame_input_dim = 16 * self.cell_feature_dim  # 16 cells * 19 = 304

        # 1. Per-sensor spatial frame encoder
        self.spatial_mlp = nn.Sequential(
            nn.Linear(self.frame_input_dim, spatial_hidden_dim),
            nn.LayerNorm(spatial_hidden_dim),
            nn.SiLU(),
            nn.Linear(spatial_hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.sensor_embedding = nn.Parameter(
            torch.randn(len(indices), token_dim) * 0.02
        )

        # 2. Per-sensor temporal causal GRU
        self.temporal_gru = nn.GRU(
            input_size=token_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )

        # Kinematic order positional embedding along the 4 arm link positions
        self.kinematic_pos_embedding = nn.Parameter(
            torch.randn(len(indices), temporal_hidden_dim) * 0.02
        )

        # 3. Inter-sensor kinematic cross-attention transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_hidden_dim,
            nhead=attention_heads,
            dim_feedforward=2 * temporal_hidden_dim,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.kinematic_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(temporal_hidden_dim),
        )

        # 4. Contact quality attention score & fusion
        self.pool_score = nn.Linear(temporal_hidden_dim, 1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * temporal_hidden_dim, 2 * temporal_hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * temporal_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # 5. Auxiliary contact enclosure head
        self.enclosure_head = nn.Sequential(
            nn.Linear(len(indices) * temporal_hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    @torch.no_grad()
    def set_normalization_statistics(
        self, *, p5: Tensor, p95: Tensor, median: Tensor
    ) -> None:
        """Install train-split robust percentiles into persistent buffers."""
        converted: dict[str, Tensor] = {}
        for name, value in {"p5": p5, "p95": p95, "median": median}.items():
            tensor = torch.as_tensor(
                value, dtype=self.distance_p5.dtype, device=self.distance_p5.device
            ).flatten()
            if tensor.shape != (self.n_selected_sensors,):
                raise ValueError(
                    f"Antigravity Beaver {name} must have shape ({self.n_selected_sensors},), got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Antigravity Beaver {name} must be finite")
            converted[name] = tensor
        if torch.any(converted["p95"] <= converted["p5"]):
            raise ValueError("Antigravity Beaver p95 must exceed p5 per sensor")
        self.distance_p5.copy_(converted["p5"])
        self.distance_p95.copy_(converted["p95"])
        self.distance_median.copy_(converted["median"])
        self.normalization_fitted.fill_(True)

    @torch.no_grad()
    def set_temporal_statistics(
        self, *, p5: Tensor, p95: Tensor, median: Tensor
    ) -> None:
        """Alias for compatibility with standard training loops."""
        self.set_normalization_statistics(p5=p5, p95=p95, median=median)

    def _select_inputs(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if distance.ndim < 4 or distance.shape[-2:] != (4, 4):
            raise ValueError(
                f"Antigravity Beaver distance must end in [history, sensor, 4, 4], got {tuple(distance.shape)}"
            )
        if distance.shape[-4] != self.history_steps:
            raise ValueError(
                f"Expected {self.history_steps} Beaver history frames, got {distance.shape[-4]}"
            )
        if status.shape != distance.shape:
            raise ValueError(
                f"Status shape {tuple(status.shape)} must match distance shape {tuple(distance.shape)}"
            )
        if present.shape != distance.shape[:-2]:
            raise ValueError(
                f"Present shape {tuple(present.shape)} must match distance shape {tuple(distance.shape[:-2])}"
            )

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
            f"Sensor count must be {self.n_sensors} or {self.n_selected_sensors}, got {sensor_count}"
        )

    def _augment(self, distance: Tensor, genuine: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training:
            return distance, genuine
        leading = distance.shape[:-4]
        if self.noise_std_mm > 0.0:
            # Consistent per-cell sensor bias across the 12-frame history
            bias = torch.randn(
                *leading,
                1,
                self.n_selected_sensors,
                4,
                4,
                device=distance.device,
                dtype=distance.dtype,
            )
            distance = torch.where(
                genuine, distance + bias * self.noise_std_mm, distance
            )
        if self.pixel_dropout > 0.0:
            pixel_drop = torch.rand(
                *leading,
                1,
                self.n_selected_sensors,
                4,
                4,
                device=distance.device,
            ) < self.pixel_dropout
            genuine = genuine & ~pixel_drop
        if self.sensor_dropout > 0.0:
            sensor_drop = torch.rand(
                *leading,
                1,
                self.n_selected_sensors,
                1,
                1,
                device=distance.device,
            ) < self.sensor_dropout
            genuine = genuine & ~sensor_drop
        return distance, genuine

    def _compute_spatial_gradients(self, normalized_dist: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Compute horizontal (dx) and vertical (dy) spatial surface curvature on 4x4 grid."""
        leading = normalized_dist.shape[:-2]
        flat_grid = normalized_dist.reshape(-1, 1, 4, 4)
        padded = F.pad(flat_grid, (1, 1, 1, 1), mode="replicate")
        dx = (padded[:, 0, 1:5, 2:6] - padded[:, 0, 1:5, 0:4]) * 0.5
        dy = (padded[:, 0, 2:6, 1:5] - padded[:, 0, 0:4, 1:5]) * 0.5
        dx = dx.reshape(*leading, 4, 4) * valid_mask
        dy = dy.reshape(*leading, 4, 4) * valid_mask
        return dx, dy

    def preprocess(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, dict[str, Any]]:
        """Preprocess 12-frame raw Beaver data into multi-channel physical tokens."""
        if not bool(self.normalization_fitted.item()):
            raise RuntimeError("Antigravity Beaver normalization statistics are not fitted")

        distance, status, present = self._select_inputs(distance, status, present)
        statuses = self.valid_status_values.to(device=status.device, dtype=status.dtype)
        status_valid = (status.unsqueeze(-1) == statuses).any(dim=-1)
        raw_zero = distance.eq(0)
        sensor_present = present.bool().unsqueeze(-1).unsqueeze(-1)
        genuine = (
            status_valid
            & sensor_present
            & torch.isfinite(distance)
            & ~raw_zero
        )
        distance, genuine = self._augment(distance, genuine)

        statistic_shape = [1] * (distance.ndim - 3) + [self.n_selected_sensors, 1, 1]
        frame_stat_shape = [1] * (distance.ndim - 4) + [self.n_selected_sensors, 1, 1]
        p5 = self.distance_p5.to(distance.dtype).view(*statistic_shape)
        p95 = self.distance_p95.to(distance.dtype).view(*statistic_shape)
        median = self.distance_median.to(distance.dtype).view(*frame_stat_shape)
        scale = (p95 - p5).clamp_min(1e-4)

        # Causal forward-fill across the 12-frame history per pixel
        previous = median.expand(*distance.shape[:-4], self.n_selected_sensors, 4, 4)
        filled_frames: list[Tensor] = []
        for t in range(self.history_steps):
            meas = distance[..., t, :, :, :]
            v_now = genuine[..., t, :, :, :]
            previous = torch.where(v_now, meas, previous)
            filled_frames.append(previous)
        filled_distance = torch.stack(filled_frames, dim=-4)

        normalized_distance = ((filled_distance - p5) / scale).clamp(0.0, 1.0)
        current_valid = genuine[..., -1, :, :, :]
        current_float = current_valid.to(distance.dtype)

        # Multi-scale exponential proximity fields
        scale_shape = [1] * filled_distance.ndim + [len(self.proximity_scales_mm)]
        scales = self.proximity_scales_mm.to(distance.dtype).view(*scale_shape)
        proximity_all = torch.exp(-filled_distance.clamp_min(0.0).unsqueeze(-1) / scales)
        proximity_all = proximity_all * genuine.to(distance.dtype).unsqueeze(-1)

        # Multi-lag temporal contact flux
        delta_features: list[Tensor] = []
        delta_valid_features: list[Tensor] = []
        for lag in self.lag_steps:
            prev_t_idx = [max(0, t - lag) for t in range(self.history_steps)]
            prev_norm = torch.stack([normalized_distance[..., p_idx, :, :, :] for p_idx in prev_t_idx], dim=-4)
            prev_gen = torch.stack([genuine[..., p_idx, :, :, :] for p_idx in prev_t_idx], dim=-4)
            d_valid = genuine & prev_gen
            d_val = torch.where(d_valid, normalized_distance - prev_norm, torch.zeros_like(normalized_distance))
            delta_features.append(d_val)
            delta_valid_features.append(d_valid.to(distance.dtype))

        deltas_stacked = torch.stack(delta_features, dim=-1)  # (..., history, sensor, 4, 4, len(lags))
        deltas_valid_stacked = torch.stack(delta_valid_features, dim=-1)

        # Spatial surface gradients on current and historical normalized distances
        dx_list, dy_list = [], []
        for t in range(self.history_steps):
            dx_t, dy_t = self._compute_spatial_gradients(
                normalized_distance[..., t, :, :, :], genuine[..., t, :, :, :].to(distance.dtype)
            )
            dx_list.append(dx_t)
            dy_list.append(dy_t)
        dx_stacked = torch.stack(dx_list, dim=-4)
        dy_stacked = torch.stack(dy_list, dim=-4)

        # Sensor-relative and global-relative spatial deviations
        valid_float = genuine.to(distance.dtype)
        obs_norm = torch.where(genuine, normalized_distance, torch.zeros_like(normalized_distance))
        sensor_cnt = valid_float.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        sensor_mean = obs_norm.sum(dim=(-2, -1), keepdim=True) / sensor_cnt
        sensor_rel = torch.where(genuine, normalized_distance - sensor_mean, torch.zeros_like(normalized_distance))

        global_cnt = valid_float.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0)
        global_mean = obs_norm.sum(dim=(-3, -2, -1), keepdim=True) / global_cnt
        global_rel = torch.where(genuine, normalized_distance - global_mean, torch.zeros_like(normalized_distance))

        # Assemble per-cell feature tensors: (..., history, sensor, 4, 4, 19)
        cell_features = torch.cat(
            (
                obs_norm.unsqueeze(-1),
                proximity_all,
                deltas_stacked,
                deltas_valid_stacked,
                dx_stacked.unsqueeze(-1),
                dy_stacked.unsqueeze(-1),
                sensor_rel.unsqueeze(-1),
                global_rel.unsqueeze(-1),
                valid_float.unsqueeze(-1),
                raw_zero.to(distance.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )

        sensor_available = current_valid.any(dim=(-2, -1))  # (..., 4)
        sensor_quality = torch.sqrt(
            current_float.mean(dim=(-2, -1))
            * valid_float.mean(dim=(-4, -2, -1))
        )  # (..., 4)

        near_fraction = (proximity_all[..., -1, :, :, :, 0] > 0.3678).float().mean()

        return cell_features, {
            "selected_distance": distance,
            "genuine": genuine,
            "current_valid": current_valid,
            "proximity_all": proximity_all,
            "sensor_available": sensor_available,
            "sensor_quality": sensor_quality,
            "near_field_fraction": near_fraction,
        }

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        *,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Encode 12-frame Beaver history into a 64-D contact state representation."""
        cell_features, intermediates = self.preprocess(distance, status, present)
        leading_shape = cell_features.shape[:-5]
        flat_cells = cell_features.flatten(start_dim=-3)  # (..., history, sensor, 304)
        flat_cells = flat_cells.transpose(-3, -2)         # (..., sensor, history, 304)

        # 1. Per-sensor spatial frame MLP
        spatial_tokens = self.spatial_mlp(flat_cells)     # (..., sensor, history, token_dim)
        emb_shape = [1] * len(leading_shape) + [self.n_selected_sensors, 1, self.token_dim]
        spatial_tokens = spatial_tokens + self.sensor_embedding.view(*emb_shape)

        # 2. Per-sensor causal GRU over history
        flat_gru_in = spatial_tokens.reshape(-1, self.history_steps, self.token_dim)
        _, hidden = self.temporal_gru(flat_gru_in)
        temporal_sensor_tokens = hidden[-1].reshape(
            *leading_shape, self.n_selected_sensors, self.temporal_hidden_dim
        )

        # 3. Add kinematic positional order embedding
        pos_emb_shape = [1] * len(leading_shape) + [self.n_selected_sensors, self.temporal_hidden_dim]
        kinematic_tokens = temporal_sensor_tokens + self.kinematic_pos_embedding.view(*pos_emb_shape)

        # 4. Inter-sensor kinematic cross-attention transformer
        available = intermediates["sensor_available"].reshape(-1, self.n_selected_sensors)
        all_missing = ~available.any(dim=-1, keepdim=True)
        key_padding_mask = ~available
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[:, 0] &= ~all_missing.squeeze(-1)

        flat_tokens = kinematic_tokens.reshape(-1, self.n_selected_sensors, self.temporal_hidden_dim)
        encoded_tokens = self.kinematic_transformer(
            flat_tokens, src_key_padding_mask=key_padding_mask
        )
        available_float = available.to(encoded_tokens.dtype).unsqueeze(-1)
        encoded_tokens = encoded_tokens * available_float

        # 5. Quality-weighted attention pooling
        quality = intermediates["sensor_quality"].reshape(-1, self.n_selected_sensors)
        score = self.pool_score(encoded_tokens).squeeze(-1)
        score = score + quality.clamp_min(1e-6).log()
        score = score.masked_fill(~available, torch.finfo(score.dtype).min)
        score = torch.where(all_missing, torch.zeros_like(score), score)
        attention = score.softmax(dim=-1)
        attention = torch.where(all_missing, torch.zeros_like(attention), attention)

        attn_pooled = (attention.unsqueeze(-1) * encoded_tokens).sum(dim=-2)
        mean_pooled = (encoded_tokens * available_float).sum(dim=-2) / available_float.sum(dim=-2).clamp_min(1.0)
        max_pooled = encoded_tokens.masked_fill((~available).unsqueeze(-1), torch.finfo(encoded_tokens.dtype).min).amax(dim=-2)
        max_pooled = torch.where(all_missing, torch.zeros_like(max_pooled), max_pooled)

        pooled_cat = torch.cat((attn_pooled, mean_pooled, max_pooled), dim=-1)
        z_beaver = self.fusion_mlp(pooled_cat)
        z_beaver = z_beaver * (~all_missing).to(z_beaver.dtype)
        z_beaver = z_beaver.reshape(*leading_shape, self.output_dim)

        # 6. Auxiliary enclosure score
        flat_all_sensors = encoded_tokens.flatten(start_dim=-2)
        enclosure_score = self.enclosure_head(flat_all_sensors).sigmoid()
        enclosure_score = enclosure_score * (~all_missing).to(enclosure_score.dtype)
        enclosure_score = enclosure_score.reshape(*leading_shape, 1)

        if return_intermediates:
            intermediates.update(
                {
                    "cell_features": cell_features,
                    "spatial_tokens": spatial_tokens,
                    "temporal_sensor_tokens": temporal_sensor_tokens,
                    "encoded_tokens": encoded_tokens.reshape(*leading_shape, self.n_selected_sensors, self.temporal_hidden_dim),
                    "sensor_attention": attention.reshape(*leading_shape, self.n_selected_sensors),
                    "enclosure_score": enclosure_score,
                }
            )
            return z_beaver, intermediates
        return z_beaver


__all__ = ["AntigravityBeaverEncoder"]
