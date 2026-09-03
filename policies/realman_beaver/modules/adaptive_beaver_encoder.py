"""Size-agnostic, reliable-sensor Beaver contact-field encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class AdaptiveBeaverEncoder(nn.Module):
    """Encode continuous object-relative geometry from reliable Beaver sensors.

    The representation deliberately avoids an object-size label.  It combines
    absolute range, within-sensor and cross-sensor relative geometry, multiple
    physical proximity scales, and changes over several temporal baselines.
    A small Transformer lets the four reliable near-field sensors exchange
    information before attention pooling produces the conditioning feature.
    """

    def __init__(
        self,
        *,
        n_sensors: int,
        sensor_indices: Sequence[int],
        history_steps: int,
        lag_steps: Sequence[int],
        valid_statuses: Sequence[int] = (5, 9),
        proximity_scales_mm: Sequence[float] = (50.0, 100.0, 200.0, 400.0),
        sensor_hidden_dim: int = 128,
        token_dim: int = 64,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        output_dim: int = 96,
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
            raise ValueError("Adaptive Beaver sensor indices must be non-empty/unique")
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError("Adaptive Beaver sensor index is out of range")
        if not statuses:
            raise ValueError("valid_statuses must not be empty")
        if not lags or any(lag <= 0 or lag >= history_steps for lag in lags):
            raise ValueError("lag_steps must be positive and shorter than history")
        if len(set(lags)) != len(lags):
            raise ValueError("lag_steps must be unique")
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("proximity scales must be positive")
        dimensions = {
            "history_steps": history_steps,
            "sensor_hidden_dim": sensor_hidden_dim,
            "token_dim": token_dim,
            "transformer_layers": transformer_layers,
            "attention_heads": attention_heads,
            "output_dim": output_dim,
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError(f"encoder dimensions must be positive: {dimensions}")
        if token_dim % attention_heads:
            raise ValueError("token_dim must be divisible by attention_heads")
        if noise_std_mm < 0:
            raise ValueError("noise_std_mm cannot be negative")
        if not 0.0 <= pixel_dropout < 1.0:
            raise ValueError("pixel_dropout must be in [0, 1)")
        if not 0.0 <= sensor_dropout < 1.0:
            raise ValueError("sensor_dropout must be in [0, 1)")

        self.n_sensors = int(n_sensors)
        self.n_selected_sensors = len(indices)
        self.history_steps = int(history_steps)
        self.lag_steps = lags
        self.output_dim = int(output_dim)
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
        self.register_buffer("distance_p5", torch.zeros(len(indices)))
        self.register_buffer("distance_p95", torch.full((len(indices),), 2550.0))
        self.register_buffer("distance_median", torch.full((len(indices),), 1275.0))
        self.register_buffer("normalization_fitted", torch.tensor(False))

        # Current range + sensor-relative + global-relative + multi-scale
        # proximity + multi-lag deltas + their validity + valid + raw-zero.
        self.cell_feature_dim = 5 + len(scales) + 2 * len(lags)
        self.sensor_mlp = nn.Sequential(
            nn.Linear(16 * self.cell_feature_dim, sensor_hidden_dim),
            nn.SiLU(),
            nn.Linear(sensor_hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.sensor_embedding = nn.Parameter(
            torch.randn(len(indices), token_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=2 * token_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sensor_transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, norm=nn.LayerNorm(token_dim)
        )
        self.pool_score = nn.Linear(token_dim, 1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * token_dim, 2 * token_dim),
            nn.SiLU(),
            nn.Linear(2 * token_dim, output_dim),
            nn.LayerNorm(output_dim),
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
                    f"Adaptive Beaver {name} must have shape "
                    f"({self.n_selected_sensors},)"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Adaptive Beaver {name} must be finite")
            converted[name] = tensor
        if torch.any(converted["p95"] <= converted["p5"]):
            raise ValueError("Adaptive Beaver p95 must exceed p5 per sensor")
        self.distance_p5.copy_(converted["p5"])
        self.distance_p95.copy_(converted["p95"])
        self.distance_median.copy_(converted["median"])
        self.normalization_fitted.fill_(True)

    def _select(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if distance.ndim < 4 or distance.shape[-2:] != (4, 4):
            raise ValueError(
                "Adaptive Beaver distance must end in [history, sensor, 4, 4]"
            )
        if distance.shape[-4] != self.history_steps:
            raise ValueError(
                f"Expected {self.history_steps} history frames, "
                f"got {distance.shape[-4]}"
            )
        if status.shape != distance.shape or present.shape != distance.shape[:-2]:
            raise ValueError("Adaptive Beaver distance/status/present shapes differ")
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
        raise ValueError("Adaptive Beaver inputs do not contain configured sensors")

    def _augment(self, distance: Tensor, genuine: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training:
            return distance, genuine
        leading = distance.shape[:-4]
        if self.noise_std_mm:
            # One calibration bias per sample/observation/sensor/cell is kept
            # constant through history so temporal changes remain physical.
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
        if self.pixel_dropout:
            pixel_drop = torch.rand(
                *leading,
                1,
                self.n_selected_sensors,
                4,
                4,
                device=distance.device,
            ) < self.pixel_dropout
            genuine = genuine & ~pixel_drop
        if self.sensor_dropout:
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

    def preprocess(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not bool(self.normalization_fitted.item()):
            raise RuntimeError("Adaptive Beaver normalization statistics are not fitted")
        distance, status, present = self._select(distance, status, present)
        statuses = self.valid_status_values.to(device=status.device, dtype=status.dtype)
        status_valid = (status.unsqueeze(-1) == statuses).any(dim=-1)
        raw_zero = distance.eq(0)
        genuine = (
            status_valid
            & present.bool().unsqueeze(-1).unsqueeze(-1)
            & torch.isfinite(distance)
            & ~raw_zero
        )
        distance, genuine = self._augment(distance, genuine)

        statistic_shape = [1] * (distance.ndim - 3) + [self.n_selected_sensors, 1, 1]
        p5 = self.distance_p5.to(distance.dtype).view(*statistic_shape)
        p95 = self.distance_p95.to(distance.dtype).view(*statistic_shape)
        median = self.distance_median.to(distance.dtype).view(*statistic_shape)
        normalized = ((distance - p5) / (p95 - p5)).clamp(0.0, 1.0)
        median_normalized = ((median - p5) / (p95 - p5)).clamp(0.0, 1.0)

        current_valid = genuine[..., -1, :, :, :]
        current_raw = distance[..., -1, :, :, :]
        current_normalized = torch.where(
            current_valid,
            normalized[..., -1, :, :, :],
            median_normalized.squeeze(-4),
        )
        observed_current = torch.where(
            current_valid, current_normalized, torch.zeros_like(current_normalized)
        )
        current_float = current_valid.to(distance.dtype)
        sensor_count = current_float.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        sensor_mean = observed_current.sum(dim=(-2, -1), keepdim=True) / sensor_count
        sensor_relative = torch.where(
            current_valid,
            current_normalized - sensor_mean,
            torch.zeros_like(current_normalized),
        )
        global_count = current_float.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0)
        global_mean = observed_current.sum(
            dim=(-3, -2, -1), keepdim=True
        ) / global_count
        global_relative = torch.where(
            current_valid,
            current_normalized - global_mean,
            torch.zeros_like(current_normalized),
        )

        scale_shape = [1] * current_raw.ndim + [len(self.proximity_scales_mm)]
        scales = self.proximity_scales_mm.to(current_raw.dtype).view(*scale_shape)
        proximity = torch.exp(-current_raw.clamp_min(0.0).unsqueeze(-1) / scales)
        proximity = proximity * current_float.unsqueeze(-1)

        delta_features: list[Tensor] = []
        delta_valid_features: list[Tensor] = []
        for lag in self.lag_steps:
            previous_valid = genuine[..., -1 - lag, :, :, :]
            delta_valid = current_valid & previous_valid
            delta = torch.where(
                delta_valid,
                current_normalized - normalized[..., -1 - lag, :, :, :],
                torch.zeros_like(current_normalized),
            )
            delta_features.append(delta)
            delta_valid_features.append(delta_valid.to(distance.dtype))

        cell_features = torch.cat(
            (
                # Invalid cells must not inject the per-sensor median into the
                # policy.  Their geometry is strictly zero; the validity and
                # raw-zero channels below retain the missingness information.
                observed_current.unsqueeze(-1),
                sensor_relative.unsqueeze(-1),
                global_relative.unsqueeze(-1),
                proximity,
                torch.stack(delta_features, dim=-1),
                torch.stack(delta_valid_features, dim=-1),
                current_float.unsqueeze(-1),
                raw_zero[..., -1, :, :, :].to(distance.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        return cell_features, {
            "selected_distance": distance,
            "current_valid": current_valid,
            "current_normalized": current_normalized,
            "sensor_relative": sensor_relative,
            "global_relative": global_relative,
            "proximity": proximity,
            "delta_features": torch.stack(delta_features, dim=-1),
            "delta_valid": torch.stack(delta_valid_features, dim=-1).bool(),
            "sensor_available": current_valid.any(dim=(-2, -1)),
            # A sensor with one sporadically valid cell should not compete on
            # equal footing with a spatially and temporally coherent field.
            "sensor_quality": torch.sqrt(
                current_float.mean(dim=(-2, -1))
                * genuine.to(distance.dtype).mean(dim=(-4, -2, -1))
            ),
        }

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        *,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        cell_features, intermediates = self.preprocess(distance, status, present)
        sensor_inputs = cell_features.flatten(start_dim=-3)
        sensor_tokens = self.sensor_mlp(sensor_inputs) + self.sensor_embedding
        leading_shape = sensor_tokens.shape[:-2]
        flat_tokens = sensor_tokens.reshape(
            -1, self.n_selected_sensors, sensor_tokens.shape[-1]
        )
        available = intermediates["sensor_available"].reshape(
            -1, self.n_selected_sensors
        )
        all_missing = ~available.any(dim=-1, keepdim=True)
        # PyTorch attention cannot consume a row where every key is padded.
        # Unmask one temporary token for that row and zero the complete result
        # below, so an all-invalid Beaver frame has exactly zero influence.
        key_padding_mask = ~available
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[:, 0] &= ~all_missing.squeeze(-1)
        encoded = self.sensor_transformer(
            flat_tokens, src_key_padding_mask=key_padding_mask
        )

        available_float = available.to(encoded.dtype).unsqueeze(-1)
        encoded = encoded * available_float
        quality = intermediates["sensor_quality"].reshape(
            -1, self.n_selected_sensors
        )
        score = self.pool_score(encoded).squeeze(-1)
        score = score + quality.clamp_min(1e-6).log()
        score = score.masked_fill(~available, torch.finfo(score.dtype).min)
        score = torch.where(all_missing, torch.zeros_like(score), score)
        attention = score.softmax(dim=-1)
        attention = torch.where(all_missing, torch.zeros_like(attention), attention)
        pooled = (attention.unsqueeze(-1) * encoded).sum(dim=-2)
        denominator = available_float.sum(dim=-2).clamp_min(1.0)
        mean = (encoded * available_float).sum(dim=-2) / denominator
        maximum = encoded.masked_fill(
            (~available).unsqueeze(-1), torch.finfo(encoded.dtype).min
        ).amax(dim=-2)
        maximum = torch.where(all_missing, torch.zeros_like(maximum), maximum)
        feature = self.fusion_mlp(torch.cat((pooled, mean, maximum), dim=-1))
        feature = feature * (~all_missing).to(feature.dtype)
        feature = feature.reshape(*leading_shape, self.output_dim)
        if return_intermediates:
            intermediates.update(
                {
                    "cell_features": cell_features,
                    "sensor_tokens": sensor_tokens,
                    "encoded_sensor_tokens": encoded.reshape(
                        *leading_shape, self.n_selected_sensors, encoded.shape[-1]
                    ),
                    "sensor_attention": attention.reshape(
                        *leading_shape, self.n_selected_sensors
                    ),
                }
            )
            return feature, intermediates
        return feature


__all__ = ["AdaptiveBeaverEncoder"]
