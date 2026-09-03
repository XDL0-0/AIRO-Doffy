"""Contact-preserving Key4 temporal encoder for wrap-then-lift gating."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class WrapBeaverEncoder(nn.Module):
    """Encode 12-frame Key4 history without discarding near-field zeros.

    Valid status-5/9 pixels with a quantized 0 mm reading are treated as
    contact, not as missing values. Invalid pixels contribute zero
    proximity instead of a train-split median fill, so absolute millimetre
    templates cannot leak bottle identity into the feature.
    """

    def __init__(
        self,
        *,
        n_sensors: int,
        sensor_indices: Sequence[int],
        history_steps: int = 12,
        valid_statuses: Sequence[int] = (5, 9),
        proximity_scales_mm: Sequence[float] = (50.0, 150.0, 300.0),
        near_threshold_mm: float = 50.0,
        closing_scale_mm: float = 10.0,
        range_scale_mm: float = 300.0,
        frame_hidden_dim: int = 64,
        frame_feature_dim: int = 32,
        temporal_hidden_dim: int = 64,
        output_dim: int = 64,
        enclosure_dim: int = 4,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in sensor_indices)
        statuses = tuple(int(status) for status in valid_statuses)
        scales = tuple(float(scale) for scale in proximity_scales_mm)
        if len(indices) != 4 or len(set(indices)) != 4:
            raise ValueError("Wrap Beaver encoding requires exactly four unique sensors")
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError(f"Wrap sensor indices must be within [0, {n_sensors})")
        if not statuses:
            raise ValueError("valid_statuses must contain at least one status code")
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("proximity_scales_mm must be positive")
        if history_steps <= 0 or output_dim != 64 or enclosure_dim != 4:
            raise ValueError("history_steps must be positive and output/enclosure dims fixed")
        if near_threshold_mm < 0:
            raise ValueError("near_threshold_mm must be non-negative")
        if closing_scale_mm <= 0 or range_scale_mm <= 0:
            raise ValueError("closing_scale_mm and range_scale_mm must be positive")

        self.n_sensors = int(n_sensors)
        self.n_selected_sensors = len(indices)
        self.history_steps = int(history_steps)
        self.output_dim = int(output_dim)
        self.enclosure_dim = int(enclosure_dim)
        self.frame_feature_dim = int(frame_feature_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.near_threshold_mm = float(near_threshold_mm)
        self.closing_scale_mm = float(closing_scale_mm)
        self.range_scale_mm = float(range_scale_mm)
        self.register_buffer("sensor_index", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "valid_status_values", torch.tensor(statuses), persistent=False
        )
        self.register_buffer(
            "proximity_scales_mm", torch.tensor(scales, dtype=torch.float32)
        )

        cell_channels = len(scales) + 3  # proximities + genuine + contact_zero + delta
        frame_input_dim = 16 * cell_channels
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_input_dim, frame_hidden_dim),
            nn.SiLU(),
            nn.Linear(frame_hidden_dim, frame_feature_dim),
            nn.SiLU(),
        )
        self.sensor_embedding = nn.Parameter(
            torch.randn(4, frame_feature_dim) * 0.02
        )
        self.temporal_gru = nn.GRU(
            input_size=frame_feature_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * temporal_hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _select_inputs(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if distance.ndim < 4 or distance.shape[-2:] != (4, 4):
            raise ValueError(
                "Wrap distance must end in [history, sensors, 4, 4], "
                f"got {tuple(distance.shape)}"
            )
        if status.shape != distance.shape:
            raise ValueError("Wrap status shape must match distance shape")
        if present.shape != distance.shape[:-2]:
            raise ValueError(
                f"Expected wrap present {tuple(distance.shape[:-2])}, got "
                f"{tuple(present.shape)}"
            )
        if distance.shape[-4] != self.history_steps:
            raise ValueError(
                f"Expected {self.history_steps} Beaver history frames, got "
                f"{distance.shape[-4]}"
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
            "Wrap Beaver sensor axis must contain either the complete "
            f"{self.n_sensors}-sensor layout or the resolved "
            f"{self.n_selected_sensors}-sensor subset, got {sensor_count}"
        )

    def preprocess(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return cell features ``(..., T, 4, 4, 4, C)`` and enclosure stats."""
        distance, status, present = self._select_inputs(distance, status, present)
        status_values = self.valid_status_values.to(
            device=status.device, dtype=status.dtype
        )
        genuine = (status.unsqueeze(-1) == status_values).any(dim=-1) & present.bool().unsqueeze(
            -1
        ).unsqueeze(-1)
        finite = torch.isfinite(distance)
        genuine = genuine & finite
        contact_zero = genuine & distance.eq(0)
        # Quantized 0 mm is contact, not missing. Invalid cells stay unused.
        range_mm = distance.clamp_min(0.0)
        genuine_f = genuine.to(dtype=distance.dtype)
        scales = self.proximity_scales_mm.to(device=distance.device, dtype=distance.dtype)
        proximity = torch.exp(
            -range_mm.unsqueeze(-1) / scales.view(*((1,) * range_mm.ndim), -1)
        ) * genuine_f.unsqueeze(-1)

        both_genuine = genuine[..., 1:, :, :, :] & genuine[..., :-1, :, :, :]
        prox50 = proximity[..., 0]
        adjacent_delta = torch.where(
            both_genuine,
            prox50[..., 1:, :, :, :] - prox50[..., :-1, :, :, :],
            torch.zeros_like(prox50[..., 1:, :, :, :]),
        )
        delta = torch.cat(
            (torch.zeros_like(prox50[..., :1, :, :, :]), adjacent_delta), dim=-4
        )
        features = torch.cat(
            (
                proximity,
                genuine_f.unsqueeze(-1),
                contact_zero.to(distance.dtype).unsqueeze(-1),
                delta.unsqueeze(-1),
            ),
            dim=-1,
        )

        huge = range_mm.new_full((), 1.0e6)
        masked_range = torch.where(genuine, range_mm, huge)
        sensor_min = masked_range.amin(dim=(-2, -1))
        sensor_valid = genuine.any(dim=(-2, -1))
        sensor_min = torch.where(
            sensor_valid, sensor_min, sensor_min.new_full((), self.range_scale_mm)
        )
        # Beaver distances are transported in 10 mm bins.  Include the
        # configured boundary so a 10 mm threshold accepts both 0 and 10 mm.
        wrap_flag = sensor_min <= self.near_threshold_mm
        wrap_progress = wrap_flag.to(distance.dtype).mean(dim=-1)
        any_sensor = sensor_valid.any(dim=-1)
        min_range = sensor_min.min(dim=-1).values
        min_range = torch.where(
            any_sensor, min_range, min_range.new_full((), self.range_scale_mm)
        )
        prev_min = min_range[..., :-1]
        curr_min = min_range[..., 1:]
        closing = torch.cat(
            (torch.zeros_like(min_range[..., :1]), prev_min - curr_min), dim=-1
        )
        closing_norm = (closing / self.closing_scale_mm).clamp(-1.0, 1.0)
        valid_fraction = genuine_f.mean(dim=(-3, -2, -1))
        min_range_norm = (min_range / self.range_scale_mm).clamp(0.0, 1.0)
        enclosure = torch.stack(
            (min_range_norm, wrap_progress, closing_norm, valid_fraction), dim=-1
        )
        return features, {
            "genuine": genuine,
            "contact_zero": contact_zero,
            "enclosure": enclosure,
            "min_range_mm": min_range,
            "wrap_progress": wrap_progress,
            "sensor_min_mm": sensor_min,
        }

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        frame_features, intermediates = self.preprocess(distance, status, present)
        frame_vectors = frame_features.flatten(start_dim=-3).transpose(-3, -2)
        encoded_frames = self.frame_mlp(frame_vectors)
        embedding_shape = [1] * (encoded_frames.ndim - 3) + [4, 1, -1]
        encoded_frames = encoded_frames + self.sensor_embedding.view(*embedding_shape)
        leading_shape = encoded_frames.shape[:-3]
        gru_input = encoded_frames.reshape(
            -1, self.history_steps, self.frame_feature_dim
        )
        _, hidden = self.temporal_gru(gru_input)
        sensor_tokens = hidden[-1].reshape(
            *leading_shape, self.n_selected_sensors, self.temporal_hidden_dim
        )
        feature = self.fusion_mlp(sensor_tokens.flatten(start_dim=-2))
        enclosure = intermediates["enclosure"][..., -1, :]
        min_range_mm = intermediates["min_range_mm"][..., -1]
        wrap_progress = intermediates["wrap_progress"][..., -1]
        sensor_min_mm = intermediates["sensor_min_mm"][..., -1, :]
        if return_intermediates:
            intermediates.update(
                {
                    "frame_features": frame_features,
                    "encoded_frames": encoded_frames,
                    "sensor_tokens": sensor_tokens,
                    "current_enclosure": enclosure,
                    "current_min_range_mm": min_range_mm,
                    "current_wrap_progress": wrap_progress,
                    "current_sensor_min_mm": sensor_min_mm,
                }
            )
            return feature, intermediates
        return feature
