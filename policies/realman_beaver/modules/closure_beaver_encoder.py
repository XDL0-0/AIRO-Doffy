"""Minimal current/delta Beaver encoder for the closure-residual policy."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ClosureBeaverEncoder(nn.Module):
    """Encode sensors independently and combine them with a masked mean.

    Distances are scaled only by the sensor's configured measurement range;
    there is deliberately no learned-phase or hand-designed near-distance
    threshold in this encoder.
    """

    def __init__(
        self,
        *,
        n_sensors: int,
        sensor_shape: tuple[int, int],
        distance_max_mm: float,
        valid_statuses: tuple[int, ...],
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        if n_sensors <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("closure Beaver dimensions must be positive")
        if len(sensor_shape) != 2 or any(size <= 0 for size in sensor_shape):
            raise ValueError("sensor_shape must contain two positive dimensions")
        if distance_max_mm <= 0 or not valid_statuses:
            raise ValueError("distance range and valid statuses are required")

        self.n_sensors = int(n_sensors)
        self.sensor_shape = tuple(sensor_shape)
        self.distance_max_mm = float(distance_max_mm)
        self.valid_statuses = tuple(int(value) for value in valid_statuses)
        cells = self.sensor_shape[0] * self.sensor_shape[1]
        # B_t, dB_t, current-valid, delta-valid, current-present, previous-present.
        sensor_input_dim = 4 * cells + 2
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
        )
        self.sensor_identity = nn.Parameter(torch.zeros(n_sensors, output_dim))
        nn.init.normal_(self.sensor_identity, std=0.02)

    def _valid_mask(self, distance: Tensor, status: Tensor, present: Tensor) -> Tensor:
        status_valid = torch.zeros_like(status, dtype=torch.bool)
        for code in self.valid_statuses:
            status_valid |= status == code
        return (
            status_valid & (present[..., :, None, None] > 0) & torch.isfinite(distance)
        )

    def forward(
        self,
        current_distance: Tensor,
        previous_distance: Tensor,
        current_status: Tensor,
        previous_status: Tensor,
        current_present: Tensor,
        previous_present: Tensor,
        *,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        expected_distance = (self.n_sensors, *self.sensor_shape)
        if current_distance.shape[-3:] != expected_distance:
            raise ValueError(
                "closure Beaver distance must end in "
                f"{expected_distance}, got {tuple(current_distance.shape)}"
            )
        if previous_distance.shape != current_distance.shape:
            raise ValueError("current and previous Beaver distance shapes must match")
        if (
            current_status.shape != current_distance.shape
            or previous_status.shape != current_distance.shape
        ):
            raise ValueError("Beaver status and distance shapes must match")
        if (
            current_present.shape != current_distance.shape[:-2]
            or previous_present.shape != current_present.shape
        ):
            raise ValueError("Beaver present masks must end at the sensor axis")

        current_valid = self._valid_mask(
            current_distance, current_status, current_present
        )
        previous_valid = self._valid_mask(
            previous_distance, previous_status, previous_present
        )
        delta_valid = current_valid & previous_valid

        current = (
            torch.nan_to_num(current_distance, nan=0.0).clamp(0.0, self.distance_max_mm)
            / self.distance_max_mm
        )
        previous = (
            torch.nan_to_num(previous_distance, nan=0.0).clamp(
                0.0, self.distance_max_mm
            )
            / self.distance_max_mm
        )
        current = torch.where(current_valid, current, torch.zeros_like(current))
        delta = torch.where(
            delta_valid,
            (current - previous).clamp(-1.0, 1.0),
            torch.zeros_like(current),
        )

        sensor_input = torch.cat(
            (
                current.flatten(start_dim=-2),
                delta.flatten(start_dim=-2),
                current_valid.to(current.dtype).flatten(start_dim=-2),
                delta_valid.to(current.dtype).flatten(start_dim=-2),
                current_present.clamp(0.0, 1.0).unsqueeze(-1),
                previous_present.clamp(0.0, 1.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        sensor_embedding = self.sensor_mlp(sensor_input) + self.sensor_identity
        sensor_available = current_valid.flatten(start_dim=-2).any(dim=-1)
        availability = sensor_available.to(sensor_embedding.dtype).unsqueeze(-1)
        pooled = (sensor_embedding * availability).sum(dim=-2) / availability.sum(
            dim=-2
        ).clamp_min(1.0)
        any_available = sensor_available.any(dim=-1, keepdim=True).to(pooled.dtype)
        pooled = pooled * any_available
        delta_flat = delta.flatten(start_dim=-3)

        if not return_intermediates:
            return pooled
        return pooled, {
            "delta_flat": delta_flat,
            "current_valid": current_valid,
            "delta_valid": delta_valid,
            "sensor_available": sensor_available,
            "any_available": any_available,
            "sensor_embedding": sensor_embedding,
        }


__all__ = ["ClosureBeaverEncoder"]
