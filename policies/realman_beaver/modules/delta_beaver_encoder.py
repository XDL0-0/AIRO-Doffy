"""Current-plus-short-delta Beaver encoder for WRM_delta."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class DeltaBeaverEncoder(nn.Module):
    """Encode four fixed-order sensors without temporal recurrence or pooling."""

    def __init__(
        self,
        *,
        n_sensors: int,
        sensor_indices: Sequence[int],
        valid_statuses: Sequence[int] = (5, 9),
        sensor_hidden_dim: int = 64,
        sensor_feature_dim: int = 32,
        fusion_hidden_dim: int = 128,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in sensor_indices)
        statuses = tuple(int(status) for status in valid_statuses)
        if len(indices) != 4 or len(set(indices)) != 4:
            raise ValueError("WRM_delta requires exactly four unique sensors")
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError(f"sensor indices must be within [0, {n_sensors})")
        if not statuses:
            raise ValueError("valid_statuses must not be empty")
        dimensions = {
            "sensor_hidden_dim": sensor_hidden_dim,
            "sensor_feature_dim": sensor_feature_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
            "output_dim": output_dim,
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError(f"encoder dimensions must be positive: {dimensions}")
        if output_dim != 64:
            raise ValueError("WRM_delta Beaver output_dim must be 64")

        self.n_sensors = int(n_sensors)
        self.n_selected_sensors = 4
        self.sensor_feature_dim = int(sensor_feature_dim)
        self.output_dim = int(output_dim)
        self.register_buffer("sensor_index", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "valid_status_values", torch.tensor(statuses), persistent=False
        )
        self.register_buffer("distance_mean", torch.zeros(4))
        self.register_buffer("distance_std", torch.ones(4))
        self.register_buffer("normalization_fitted", torch.tensor(False))

        # 16 pixels x (current, t-k delta, valid mask, zero mask).
        self.sensor_mlp = nn.Sequential(
            nn.Linear(16 * 4, sensor_hidden_dim),
            nn.SiLU(),
            nn.Linear(sensor_hidden_dim, sensor_feature_dim),
            nn.SiLU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * sensor_feature_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    @torch.no_grad()
    def set_normalization_statistics(self, *, mean: Tensor, std: Tensor) -> None:
        mean = torch.as_tensor(
            mean, dtype=self.distance_mean.dtype, device=self.distance_mean.device
        ).flatten()
        std = torch.as_tensor(
            std, dtype=self.distance_std.dtype, device=self.distance_std.device
        ).flatten()
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError("WRM_delta mean/std must each have shape (4,)")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("WRM_delta mean/std must be finite")
        if torch.any(std <= 0):
            raise ValueError("WRM_delta std must be positive")
        self.distance_mean.copy_(mean)
        self.distance_std.copy_(std)
        self.normalization_fitted.fill_(True)

    def _select(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if distance.ndim < 3 or distance.shape[-2:] != (4, 4):
            raise ValueError(
                "Beaver distance must end in [sensor, 4, 4], got "
                f"{tuple(distance.shape)}"
            )
        if status.shape != distance.shape or present.shape != distance.shape[:-2]:
            raise ValueError("Beaver distance/status/present shapes do not match")
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
            f"Beaver sensor axis must contain {self.n_sensors} or 4 sensors, "
            f"got {sensor_count}"
        )

    def _genuine(self, distance: Tensor, status: Tensor, present: Tensor) -> Tensor:
        statuses = self.valid_status_values.to(device=status.device, dtype=status.dtype)
        status_valid = (status.unsqueeze(-1) == statuses).any(dim=-1)
        return (
            status_valid
            & present.bool().unsqueeze(-1).unsqueeze(-1)
            & torch.isfinite(distance)
            & distance.ne(0)
        )

    def preprocess(
        self,
        distance: Tensor,
        previous_distance: Tensor,
        status: Tensor,
        previous_status: Tensor,
        present: Tensor,
        previous_present: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not bool(self.normalization_fitted.item()):
            raise RuntimeError(
                "WRM_delta per-sensor training statistics are not fitted"
            )
        distance, status, present = self._select(distance, status, present)
        previous_distance, previous_status, previous_present = self._select(
            previous_distance, previous_status, previous_present
        )
        current_valid = self._genuine(distance, status, present)
        previous_valid = self._genuine(
            previous_distance, previous_status, previous_present
        )
        zero_mask = distance.eq(0)
        statistic_shape = [1] * (distance.ndim - 3) + [4, 1, 1]
        mean = self.distance_mean.to(distance.dtype).view(*statistic_shape)
        std = self.distance_std.to(distance.dtype).view(*statistic_shape)
        normalized_distance = torch.where(
            current_valid, (distance - mean) / std, torch.zeros_like(distance)
        )
        raw_delta = distance - previous_distance
        delta_valid = current_valid & previous_valid
        normalized_delta = torch.where(
            delta_valid, raw_delta / std, torch.zeros_like(raw_delta)
        )
        features = torch.stack(
            (
                normalized_distance,
                normalized_delta,
                current_valid.to(distance.dtype),
                zero_mask.to(distance.dtype),
            ),
            dim=-1,
        )
        return features, {
            "selected_distance": distance,
            "selected_previous_distance": previous_distance,
            "delta_distance": raw_delta,
            "normalized_delta": normalized_delta,
            "valid_mask": current_valid,
            "delta_valid_mask": delta_valid,
            "zero_mask": zero_mask,
        }

    def forward(
        self,
        distance: Tensor,
        previous_distance: Tensor,
        status: Tensor,
        previous_status: Tensor,
        present: Tensor,
        previous_present: Tensor,
        *,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features, intermediates = self.preprocess(
            distance,
            previous_distance,
            status,
            previous_status,
            present,
            previous_present,
        )
        sensor_inputs = features.flatten(start_dim=-3)
        sensor_features = self.sensor_mlp(sensor_inputs)
        concatenated = sensor_features.flatten(start_dim=-2)
        z_beaver = self.fusion_mlp(concatenated)
        if return_intermediates:
            intermediates.update(
                {
                    "cell_features": features,
                    "sensor_features": sensor_features,
                    "concatenated": concatenated,
                }
            )
            return z_beaver, intermediates
        return z_beaver


__all__ = ["DeltaBeaverEncoder"]
