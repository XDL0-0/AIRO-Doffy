"""Enclosure-aware temporal Beaver encoder and contact-phase labels for WRM_grok.

The encoder keeps only the four reliable near-field sensors (01/02/10/11).
Invalid pixels are masked rather than treated as contact: status must be 5 or 9,
the sensor must be present, and a raw zero is not a valid range. All-invalid
Key4 input produces an exactly zero Beaver feature and enclosure vector.

Unlike WRM_temporal's independent per-sensor GRUs, this encoder also emits an
explicit 16-D enclosure geometry vector (cross-sensor min-range, wrap count,
closing rate). Phase labels are derived from tightness plus this contact field
so wrap versus approach cannot be inferred from tightness alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

PHASE_APPROACH = 0
PHASE_WRAP = 1
PHASE_HOLD = 2
PHASE_NAMES = ("approach", "wrap", "hold")
ENCLOSURE_DIM = 16
PHASE_DIM = 3
CONTACT_QUALITY_DIM = 1


def grok_conditioned_state_dim(model) -> int:
    """Robot state seen by the flow U-Net: q, Δq, gated Beaver, enclosure, phase."""
    return (
        2 * int(model.state_dim)
        + int(model.beaver_grok_feature_dim)
        + int(model.beaver_grok_enclosure_dim)
        + PHASE_DIM
        + CONTACT_QUALITY_DIM
    )


def derive_phase_labels(
    tightness: Tensor,
    distance: Tensor,
    status: Tensor,
    present: Tensor,
    *,
    sensor_index: Tensor,
    valid_statuses: Sequence[int],
    wrap_threshold_mm: float,
    min_near_sensors: int = 2,
) -> Tensor:
    """Build approach/wrap/hold labels from tightness and current-frame contact.

    Hold is tightness >= 0.5. Wrap is the remaining frames where at least
    ``min_near_sensors`` Key4 sensors have a valid reading closer than
    ``wrap_threshold_mm``. Everything else is approach. Tightness is a
    training-only label; the deployed policy never receives it.
    """
    if tightness.ndim == distance.ndim:
        tightness = tightness.squeeze(-1)
    current_distance, current_status, current_present = _current_key4(
        distance, status, present, sensor_index
    )
    genuine = _genuine_mask(
        current_distance, current_status, current_present, valid_statuses
    )
    inf = torch.finfo(current_distance.dtype).max
    min_mm = current_distance.masked_fill(~genuine, inf).amin(dim=(-2, -1))
    near = torch.isfinite(min_mm) & (min_mm < float(wrap_threshold_mm))
    enclosure_count = near.to(dtype=torch.int64).sum(dim=-1)
    hold = tightness >= 0.5
    wrap = (~hold) & (enclosure_count >= int(min_near_sensors))
    labels = torch.zeros_like(enclosure_count, dtype=torch.long)
    labels = torch.where(wrap, torch.full_like(labels, PHASE_WRAP), labels)
    labels = torch.where(hold, torch.full_like(labels, PHASE_HOLD), labels)
    return labels


def _current_key4(
    distance: Tensor,
    status: Tensor,
    present: Tensor,
    sensor_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    current_distance = distance[..., -1, :, :, :]
    current_status = status[..., -1, :, :, :]
    current_present = present[..., -1, :]
    index = sensor_index.to(device=distance.device)
    if current_distance.shape[-3] == index.numel():
        return current_distance, current_status, current_present
    return (
        current_distance.index_select(-3, index),
        current_status.index_select(-3, index),
        current_present.index_select(-1, index),
    )


def _genuine_mask(
    distance: Tensor,
    status: Tensor,
    present: Tensor,
    valid_statuses: Sequence[int],
) -> Tensor:
    status_values = torch.as_tensor(
        tuple(int(code) for code in valid_statuses),
        device=status.device,
        dtype=status.dtype,
    )
    status_is_valid = (status.unsqueeze(-1) == status_values).any(dim=-1)
    sensor_present = present.bool().unsqueeze(-1).unsqueeze(-1)
    return (
        status_is_valid
        & sensor_present
        & torch.isfinite(distance)
        & ~distance.eq(0)
    )


class GrokPhaseEncoder(nn.Module):
    """Temporal Key4 encoder with explicit enclosure geometry."""

    def __init__(
        self,
        *,
        n_sensors: int,
        sensor_indices: Sequence[int],
        history_steps: int = 12,
        valid_statuses: Sequence[int] = (5, 9),
        frame_hidden_dim: int = 64,
        frame_feature_dim: int = 32,
        temporal_hidden_dim: int = 64,
        output_dim: int = 64,
        enclosure_dim: int = ENCLOSURE_DIM,
        near_scales_mm: Sequence[float] = (50.0, 150.0, 300.0),
        wrap_threshold_mm: float = 150.0,
        noise_std_mm: float = 3.0,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in sensor_indices)
        statuses = tuple(int(status) for status in valid_statuses)
        scales = tuple(float(scale) for scale in near_scales_mm)
        if len(indices) != 4 or len(set(indices)) != 4:
            raise ValueError("WRM_grok requires exactly four unique Key4 sensors")
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError(f"Key4 sensor indices must be within [0, {n_sensors})")
        if not statuses:
            raise ValueError("valid_statuses must contain at least one status code")
        if history_steps <= 0 or output_dim <= 0 or enclosure_dim != ENCLOSURE_DIM:
            raise ValueError("history/output must be positive and enclosure_dim=16")
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("near_scales_mm must be positive")
        if wrap_threshold_mm <= 0:
            raise ValueError("wrap_threshold_mm must be positive")
        if noise_std_mm < 0:
            raise ValueError("noise_std_mm cannot be negative")

        self.n_sensors = int(n_sensors)
        self.n_selected_sensors = len(indices)
        self.history_steps = int(history_steps)
        self.output_dim = int(output_dim)
        self.enclosure_dim = int(enclosure_dim)
        self.frame_feature_dim = int(frame_feature_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.wrap_threshold_mm = float(wrap_threshold_mm)
        self.noise_std_mm = float(noise_std_mm)
        self.register_buffer("sensor_index", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "valid_status_values", torch.tensor(statuses), persistent=False
        )
        self.register_buffer(
            "near_scales_mm", torch.tensor(scales, dtype=torch.float32)
        )
        self.register_buffer("distance_p5", torch.zeros(4))
        self.register_buffer("distance_p95", torch.full((4,), 2550.0))
        self.register_buffer("distance_median", torch.full((4,), 1275.0))
        self.register_buffer("normalization_fitted", torch.tensor(False))

        cell_channels = 4 + len(scales)
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

    @torch.no_grad()
    def set_normalization_statistics(
        self, *, p5: Tensor, p95: Tensor, median: Tensor
    ) -> None:
        values = {"p5": p5, "p95": p95, "median": median}
        converted: dict[str, Tensor] = {}
        for name, value in values.items():
            tensor = torch.as_tensor(
                value, device=self.distance_p5.device, dtype=self.distance_p5.dtype
            )
            if tensor.shape != (self.n_selected_sensors,):
                raise ValueError(
                    f"WRM_grok {name} must have shape "
                    f"({self.n_selected_sensors},), got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"WRM_grok {name} must be finite")
            converted[name] = tensor
        if torch.any(converted["p95"] <= converted["p5"]):
            raise ValueError("WRM_grok p95 must be greater than p5 per sensor")
        self.distance_p5.copy_(converted["p5"])
        self.distance_p95.copy_(converted["p95"])
        self.distance_median.copy_(converted["median"])
        self.normalization_fitted.fill_(True)

    def _select_inputs(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if distance.ndim < 4 or distance.shape[-2:] != (4, 4):
            raise ValueError(
                "WRM_grok distance must end in (..., history, sensors, 4, 4), "
                f"got {tuple(distance.shape)}"
            )
        if status.shape != distance.shape:
            raise ValueError("WRM_grok status shape must match distance shape")
        if present.shape != distance.shape[:-2]:
            raise ValueError(
                f"Expected present {tuple(distance.shape[:-2])}, got {tuple(present.shape)}"
            )
        if distance.shape[-4] != self.history_steps:
            raise ValueError(
                f"Expected {self.history_steps} Beaver history frames, "
                f"got {distance.shape[-4]}"
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
            "WRM_grok sensor axis must contain the full 9-sensor layout or the "
            f"resolved {self.n_selected_sensors}-sensor Key4 subset, got {sensor_count}"
        )

    def preprocess(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not bool(self.normalization_fitted.item()):
            raise RuntimeError(
                "WRM_grok robust normalization statistics are not fitted; "
                "refusing to fall back to global 2550 mm normalization"
            )
        distance, status, present = self._select_inputs(distance, status, present)
        if self.training and self.noise_std_mm > 0:
            noise = torch.randn_like(distance) * self.noise_std_mm
            distance = (distance + noise).clamp_min(0.0)
        genuine = _genuine_mask(
            distance, status, present, tuple(self.valid_status_values.tolist())
        )
        zero_flag = distance.eq(0)
        statistic_shape = [1] * (distance.ndim - 3) + [4, 1, 1]
        frame_statistic_shape = [1] * (distance.ndim - 4) + [4, 1, 1]
        median = self.distance_median.to(distance.dtype).view(*frame_statistic_shape)
        previous = median.expand(*distance.shape[:-4], 4, 4, 4)
        filled_frames: list[Tensor] = []
        for time_index in range(self.history_steps):
            measurement = distance[..., time_index, :, :, :]
            valid_now = genuine[..., time_index, :, :, :]
            previous = torch.where(valid_now, measurement, previous)
            filled_frames.append(previous)
        filled_distance = torch.stack(filled_frames, dim=-4)

        p5 = self.distance_p5.to(distance.dtype).view(*statistic_shape)
        p95 = self.distance_p95.to(distance.dtype).view(*statistic_shape)
        scale = (p95 - p5).clamp_min(1e-4)
        normalized_distance = ((filled_distance - p5) / scale).clamp(0.0, 1.0)
        measured_normalized = ((distance - p5) / scale).clamp(0.0, 1.0)
        delta_first = torch.zeros_like(measured_normalized[..., :1, :, :, :])
        consecutive_genuine = genuine[..., 1:, :, :, :] & genuine[..., :-1, :, :, :]
        adjacent_delta = torch.where(
            consecutive_genuine,
            measured_normalized[..., 1:, :, :, :]
            - measured_normalized[..., :-1, :, :, :],
            torch.zeros_like(measured_normalized[..., 1:, :, :, :]),
        )
        delta_distance = torch.cat((delta_first, adjacent_delta), dim=-4)
        proximity = []
        for scale_mm in self.near_scales_mm.tolist():
            near = (1.0 - (filled_distance / scale_mm).clamp(0.0, 1.0)) * genuine.to(
                distance.dtype
            )
            proximity.append(near)
        cell_features = torch.stack(
            (
                normalized_distance,
                delta_distance,
                genuine.to(distance.dtype),
                zero_flag.to(distance.dtype),
                *proximity,
            ),
            dim=-1,
        )
        return cell_features, {
            "selected_distance": distance,
            "valid": genuine,
            "zero_flag": zero_flag,
            "filled_distance": filled_distance,
            "normalized_distance": normalized_distance,
            "delta_distance": delta_distance,
        }

    def _enclosure(
        self, filled_distance: Tensor, genuine: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(..., 16)`` enclosure geometry and a scalar contact quality."""
        last_filled = filled_distance[..., -1, :, :, :]
        last_genuine = genuine[..., -1, :, :, :]
        first_filled = filled_distance[..., 0, :, :, :]
        first_genuine = genuine[..., 0, :, :, :]
        inf = torch.finfo(filled_distance.dtype).max
        last_min = last_filled.masked_fill(~last_genuine, inf).amin(dim=(-2, -1))
        first_min = first_filled.masked_fill(~first_genuine, inf).amin(dim=(-2, -1))
        any_valid = last_genuine.any(dim=(-2, -1))
        median = self.distance_median.to(filled_distance.dtype)
        last_min = torch.where(any_valid, last_min, median)
        p5 = self.distance_p5.to(filled_distance.dtype)
        p95 = self.distance_p95.to(filled_distance.dtype)
        scale = (p95 - p5).clamp_min(1e-4)
        min_norm = ((last_min - p5) / scale).clamp(0.0, 1.0) * any_valid.to(
            filled_distance.dtype
        )
        valid_frac = last_genuine.to(filled_distance.dtype).mean(dim=(-2, -1))
        contact_quality = valid_frac.mean(dim=-1)
        all_invalid = contact_quality <= 0
        enclosure_frac = (
            any_valid & (last_min < self.wrap_threshold_mm)
        ).to(filled_distance.dtype).mean(dim=-1)
        first_ok = first_genuine.any(dim=(-2, -1))
        last_ok = any_valid
        pair_ok = first_ok & last_ok
        first_min = torch.where(first_ok, first_min, median)
        closing = ((first_min - last_min) / scale).clamp(-1.0, 1.0)
        closing = torch.where(
            pair_ok, closing, torch.zeros_like(closing)
        ).mean(dim=-1)
        near50 = (
            (last_filled < self.near_scales_mm[0]).to(filled_distance.dtype)
            * last_genuine.to(filled_distance.dtype)
        ).mean(dim=(-3, -2, -1))
        valid_mean = last_filled.masked_fill(~last_genuine, 0.0).sum(
            dim=(-2, -1)
        ) / last_genuine.to(filled_distance.dtype).sum(dim=(-2, -1)).clamp_min(1.0)
        mean_norm = ((valid_mean - p5) / scale).clamp(0.0, 1.0) * any_valid.to(
            filled_distance.dtype
        )
        mean_norm = mean_norm.mean(dim=-1)
        valid_sensor_frac = any_valid.to(filled_distance.dtype).mean(dim=-1)
        enclosure = torch.cat(
            (
                min_norm,
                valid_frac,
                (min_norm[..., 0] - min_norm[..., 3]).unsqueeze(-1),
                (min_norm[..., 1] - min_norm[..., 2]).unsqueeze(-1),
                enclosure_frac.unsqueeze(-1),
                closing.unsqueeze(-1),
                contact_quality.unsqueeze(-1),
                near50.unsqueeze(-1),
                mean_norm.unsqueeze(-1),
                valid_sensor_frac.unsqueeze(-1),
            ),
            dim=-1,
        )
        enclosure = torch.where(
            all_invalid.unsqueeze(-1), torch.zeros_like(enclosure), enclosure
        )
        contact_quality = torch.where(
            all_invalid, torch.zeros_like(contact_quality), contact_quality
        )
        return enclosure, contact_quality

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        cell_features, intermediates = self.preprocess(distance, status, present)
        frame_vectors = cell_features.flatten(start_dim=-3).transpose(-3, -2)
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
        concatenated = sensor_tokens.flatten(start_dim=-2)
        beaver_feature = self.fusion_mlp(concatenated)
        enclosure, contact_quality = self._enclosure(
            intermediates["filled_distance"], intermediates["valid"]
        )
        all_invalid = contact_quality <= 0
        beaver_feature = torch.where(
            all_invalid.unsqueeze(-1), torch.zeros_like(beaver_feature), beaver_feature
        )
        intermediates.update(
            {
                "frame_features": cell_features,
                "encoded_frames": encoded_frames,
                "sensor_tokens": sensor_tokens,
                "concatenated": concatenated,
                "enclosure": enclosure,
                "contact_quality": contact_quality,
                "all_invalid": all_invalid,
            }
        )
        if return_intermediates:
            return beaver_feature, intermediates
        return beaver_feature
