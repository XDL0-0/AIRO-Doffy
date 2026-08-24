"""Structured sensor-wise encoder for the new Beaver DP variants."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn


class StructuredBeaverEncoder(nn.Module):
    """Encode Beaver cells per physical sensor, then aggregate sensor tokens.

    The sensor axis is retained until after a single shared MLP has processed
    every sensor.  ``dp_beaver_near`` adds cell-wise near-field proximity and
    ``dp_beaver_near_gate`` additionally learns independent sigmoid gates for
    the sensor tokens.
    """

    SUPPORTED_VARIANTS = frozenset(
        {"dp_beaver_enc", "dp_beaver_near", "dp_beaver_near_gate"}
    )

    def __init__(
        self,
        variant: str,
        n_sensors: int = 9,
        distance_max_mm: float = 2550.0,
        valid_statuses: Sequence[int] = (5, 9),
        output_dim: int = 64,
        sensor_hidden_dim: int = 64,
        sensor_feature_dim: int = 32,
        near_threshold_mm: float = 300.0,
        gate_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported structured Beaver variant {variant!r}; expected one of "
                f"{sorted(self.SUPPORTED_VARIANTS)}"
            )
        dimensions = {
            "n_sensors": n_sensors,
            "output_dim": output_dim,
            "sensor_hidden_dim": sensor_hidden_dim,
            "sensor_feature_dim": sensor_feature_dim,
            "gate_hidden_dim": gate_hidden_dim,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if distance_max_mm <= 0:
            raise ValueError(f"distance_max_mm must be positive, got {distance_max_mm}")
        statuses = tuple(valid_statuses)
        if not statuses:
            raise ValueError("valid_statuses must contain at least one status code")

        self.variant = variant
        self.n_sensors = n_sensors
        self.output_dim = output_dim
        self.sensor_feature_dim = sensor_feature_dim
        self.uses_near = variant in {"dp_beaver_near", "dp_beaver_near_gate"}
        self.uses_gate = variant == "dp_beaver_near_gate"
        self.register_buffer("distance_max_mm", torch.tensor(float(distance_max_mm)))
        self.register_buffer(
            "valid_status_values", torch.tensor(statuses), persistent=False
        )

        if self.uses_near:
            if near_threshold_mm <= 0:
                raise ValueError(
                    "near_threshold_mm must be positive for near-field variants, "
                    f"got {near_threshold_mm}"
                )
            self.register_buffer(
                "near_threshold_mm", torch.tensor(float(near_threshold_mm))
            )

        sensor_input_dim = 49 if self.uses_near else 33
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_input_dim, sensor_hidden_dim),
            nn.SiLU(),
            nn.Linear(sensor_hidden_dim, sensor_hidden_dim),
            nn.SiLU(),
            nn.Linear(sensor_hidden_dim, sensor_feature_dim),
            nn.SiLU(),
        )
        self.sensor_embedding = nn.Parameter(
            torch.randn(n_sensors, sensor_feature_dim) * 0.02
        )

        if self.uses_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(sensor_feature_dim, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(),
            )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * sensor_feature_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def _validate_inputs(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> None:
        expected_tail = (self.n_sensors, 4, 4)
        if distance.shape[-3:] != expected_tail:
            raise ValueError(
                f"Expected distance (..., {self.n_sensors}, 4, 4), got "
                f"{tuple(distance.shape)}"
            )
        if status.shape != distance.shape:
            raise ValueError(
                f"Status shape {tuple(status.shape)} must match distance shape "
                f"{tuple(distance.shape)}"
            )
        expected_present_shape = distance.shape[:-2]
        if present.shape != expected_present_shape:
            raise ValueError(
                f"Expected present {tuple(expected_present_shape)}, got "
                f"{tuple(present.shape)}"
            )

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Encode ``(..., sensors, 4, 4)`` inputs into ``(..., output_dim)``.

        With ``return_intermediates=True``, the second return value exposes the
        cell validity, MLP input, and final local tokens.  Gated variants also
        expose sigmoid gates both before and after the presence mask.
        """

        self._validate_inputs(distance, status, present)
        present_feature = present.to(dtype=distance.dtype).clamp(0.0, 1.0)
        status_values = self.valid_status_values.to(dtype=status.dtype)
        status_is_valid = (status.unsqueeze(-1) == status_values).any(dim=-1)
        valid_cell_bool = status_is_valid & present_feature.bool().unsqueeze(
            -1
        ).unsqueeze(-1)
        valid_cell = valid_cell_bool.to(dtype=distance.dtype)

        distance_global = (distance / self.distance_max_mm).clamp(0.0, 1.0)
        distance_global = distance_global * valid_cell
        sensor_parts = [distance_global.flatten(start_dim=-2)]

        near: Tensor | None = None
        if self.uses_near:
            near = 1.0 - (distance / self.near_threshold_mm).clamp(0.0, 1.0)
            near = near * valid_cell
            sensor_parts.append(near.flatten(start_dim=-2))

        sensor_parts.extend(
            (
                valid_cell.flatten(start_dim=-2),
                present_feature.unsqueeze(-1),
            )
        )
        sensor_input = torch.cat(sensor_parts, dim=-1)
        sensor_tokens = self.sensor_mlp(sensor_input) + self.sensor_embedding
        sensor_tokens = sensor_tokens * present_feature.unsqueeze(-1)

        intermediates: dict[str, Any] = {
            "distance_global": distance_global,
            "valid_cell": valid_cell,
            "sensor_input": sensor_input,
            "sensor_tokens": sensor_tokens,
        }
        if near is not None:
            intermediates["near"] = near

        if self.uses_gate:
            raw_gate = self.gate_mlp(sensor_tokens)
            effective_gate = raw_gate * present_feature.unsqueeze(-1)
            gated_tokens = effective_gate * sensor_tokens
            pooled_mean = gated_tokens.sum(dim=-2) / (effective_gate.sum(dim=-2) + 1e-6)
            pooled_max = gated_tokens.max(dim=-2).values
            intermediates.update(
                {
                    "raw_gate": raw_gate,
                    "effective_gate": effective_gate,
                }
            )
        else:
            pooled_mean = sensor_tokens.mean(dim=-2)
            pooled_max = sensor_tokens.max(dim=-2).values

        feature = self.fusion_mlp(torch.cat((pooled_mean, pooled_max), dim=-1))
        if return_intermediates:
            return feature, intermediates
        return feature


__all__ = ["StructuredBeaverEncoder"]
