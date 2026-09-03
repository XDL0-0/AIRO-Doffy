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


class Key4BeaverEncoder(nn.Module):
    """Topology-preserving encoder for physical sensors 01/02/10/11.

    ``dp_beaver_key4`` uses four independent MLPs, concatenates their tokens in
    physical slot order, and applies LayerNorm. ``dp_beaver_key4_pca`` applies
    a fixed, train-split-only PCA independently to every sensor before the same
    concatenate-and-normalize operation. PCA statistics are persistent buffers,
    so inference reconstructs the exact training transform from the checkpoint.
    """

    SUPPORTED_VARIANTS = frozenset(
        {"dp_beaver_key4", "dp_beaver_key4_pca"}
    )

    def __init__(
        self,
        variant: str,
        *,
        n_sensors: int = 9,
        key_sensor_indices: Sequence[int] = (1, 2, 5, 6),
        valid_statuses: Sequence[int] = (5, 9),
        output_dim: int | None = None,
        sensor_hidden_dim: int = 64,
        sensor_feature_dim: int = 32,
        near_threshold_mm: float = 300.0,
        pca_components: int = 4,
    ) -> None:
        super().__init__()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported Key4 Beaver variant {variant!r}; expected one of "
                f"{sorted(self.SUPPORTED_VARIANTS)}"
            )
        indices = tuple(int(index) for index in key_sensor_indices)
        if len(indices) != 4 or len(set(indices)) != 4:
            raise ValueError("Key4 requires four unique sensor indices")
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError(f"Key4 sensor indices must be within [0, {n_sensors})")
        statuses = tuple(valid_statuses)
        if not statuses:
            raise ValueError("valid_statuses must contain at least one status code")
        if near_threshold_mm <= 0:
            raise ValueError("near_threshold_mm must be positive")
        if not 1 <= pca_components <= 32:
            raise ValueError("pca_components must be in [1, 32]")

        self.variant = variant
        self.n_sensors = n_sensors
        self.key_sensor_indices = indices
        self.pca_components = pca_components
        self.uses_pca = variant == "dp_beaver_key4_pca"
        expected_output_dim = (
            len(indices) * pca_components
            if self.uses_pca
            else len(indices) * sensor_feature_dim
        )
        output_dim = expected_output_dim if output_dim is None else output_dim
        if output_dim != expected_output_dim:
            raise ValueError(
                f"{variant} output_dim must be {expected_output_dim}, got {output_dim}"
            )
        self.output_dim = output_dim
        self.register_buffer("key_sensor_index", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "valid_status_values", torch.tensor(statuses), persistent=False
        )
        self.register_buffer("near_threshold_mm", torch.tensor(float(near_threshold_mm)))

        if self.uses_pca:
            self.register_buffer("pca_mean", torch.zeros(4, 32))
            self.register_buffer("pca_scale", torch.ones(4, 32))
            initial_components = torch.zeros(4, pca_components, 32)
            initial_components[:, :, :pca_components] = torch.eye(pca_components)
            self.register_buffer("pca_basis", initial_components)
            self.register_buffer("pca_explained_variance_ratio", torch.zeros(4, pca_components))
            self.register_buffer("pca_fitted", torch.tensor(False))
        else:
            self.sensor_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(33, sensor_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(sensor_hidden_dim, sensor_feature_dim),
                        nn.SiLU(),
                    )
                    for _ in indices
                ]
            )
        self.layer_norm = nn.LayerNorm(output_dim)

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
            raise ValueError("status shape must match distance shape")
        if present.shape != distance.shape[:-2]:
            raise ValueError(
                f"Expected present {tuple(distance.shape[:-2])}, got "
                f"{tuple(present.shape)}"
            )

    def key4_inputs(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return proximity, validity, presence, and 32-D PCA input."""
        self._validate_inputs(distance, status, present)
        index = self.key_sensor_index
        key_distance = distance.index_select(-3, index)
        key_status = status.index_select(-3, index)
        key_present = present.index_select(-1, index).to(distance.dtype).clamp(0.0, 1.0)
        status_values = self.valid_status_values.to(dtype=status.dtype)
        validity = (key_status.unsqueeze(-1) == status_values).any(dim=-1)
        validity = validity & key_present.bool().unsqueeze(-1).unsqueeze(-1)
        valid_feature = validity.to(distance.dtype)
        proximity = 1.0 - (key_distance / self.near_threshold_mm).clamp(0.0, 1.0)
        proximity = proximity * valid_feature
        pca_input = torch.cat(
            (
                proximity.flatten(start_dim=-2),
                valid_feature.flatten(start_dim=-2),
            ),
            dim=-1,
        )
        return proximity, valid_feature, key_present, pca_input

    @torch.no_grad()
    def set_pca_statistics(
        self,
        *,
        mean: Tensor,
        scale: Tensor,
        basis: Tensor,
        explained_variance_ratio: Tensor,
    ) -> None:
        if not self.uses_pca:
            raise RuntimeError("PCA statistics only apply to dp_beaver_key4_pca")
        expected = {
            "mean": self.pca_mean.shape,
            "scale": self.pca_scale.shape,
            "basis": self.pca_basis.shape,
            "explained_variance_ratio": self.pca_explained_variance_ratio.shape,
        }
        supplied = {
            "mean": mean,
            "scale": scale,
            "basis": basis,
            "explained_variance_ratio": explained_variance_ratio,
        }
        for name, value in supplied.items():
            if value.shape != expected[name]:
                raise ValueError(
                    f"PCA {name} shape must be {tuple(expected[name])}, got "
                    f"{tuple(value.shape)}"
                )
        if torch.any(scale <= 0):
            raise ValueError("PCA scales must be positive")
        self.pca_mean.copy_(mean)
        self.pca_scale.copy_(scale)
        self.pca_basis.copy_(basis)
        self.pca_explained_variance_ratio.copy_(explained_variance_ratio)
        self.pca_fitted.fill_(True)

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        proximity, validity, key_present, pca_input = self.key4_inputs(
            distance, status, present
        )
        intermediates: dict[str, Any] = {
            "key_sensor_indices": self.key_sensor_index,
            "proximity": proximity,
            "valid_cell": validity,
            "pca_input": pca_input,
        }
        if self.uses_pca:
            standardized = (pca_input - self.pca_mean) / self.pca_scale
            sensor_tokens = torch.einsum("...si,ski->...sk", standardized, self.pca_basis)
            sensor_tokens = sensor_tokens * key_present.unsqueeze(-1)
            intermediates["standardized"] = standardized
        else:
            learned_input = torch.cat((pca_input, key_present.unsqueeze(-1)), dim=-1)
            sensor_tokens = torch.stack(
                [
                    mlp(learned_input[..., sensor_id, :])
                    * key_present[..., sensor_id, None]
                    for sensor_id, mlp in enumerate(self.sensor_mlps)
                ],
                dim=-2,
            )
            intermediates["sensor_input"] = learned_input
        concatenated = sensor_tokens.flatten(start_dim=-2)
        feature = self.layer_norm(concatenated)
        intermediates.update(
            {"sensor_tokens": sensor_tokens, "concatenated": concatenated}
        )
        if return_intermediates:
            return feature, intermediates
        return feature


class TemporalBeaverEncoder(nn.Module):
    """Encode a short, ordered history from four physical Beaver sensors.

    Each frame/pixel is represented by normalized distance, genuine temporal
    delta, validity, and a raw-zero flag. Invalid and zero distances are
    causally forward-filled per pixel, falling back to the train-split sensor
    median at the beginning of a history. Robust normalization statistics are
    persistent buffers so checkpoint inference uses the exact training
    transform.
    """

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
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in sensor_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(
                "Temporal Beaver encoding requires at least one unique sensor"
            )
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError(
                f"Temporal sensor indices must be within [0, {n_sensors})"
            )
        statuses = tuple(int(status) for status in valid_statuses)
        if not statuses:
            raise ValueError("valid_statuses must contain at least one status code")
        dimensions = {
            "n_sensors": n_sensors,
            "history_steps": history_steps,
            "frame_hidden_dim": frame_hidden_dim,
            "frame_feature_dim": frame_feature_dim,
            "temporal_hidden_dim": temporal_hidden_dim,
            "output_dim": output_dim,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if output_dim != 64:
            raise ValueError(f"Temporal Beaver output_dim must be 64, got {output_dim}")

        self.n_sensors = n_sensors
        self.n_selected_sensors = len(indices)
        self.history_steps = history_steps
        self.output_dim = output_dim
        self.frame_feature_dim = frame_feature_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.register_buffer(
            "sensor_index", torch.tensor(indices, dtype=torch.long)
        )
        self.register_buffer(
            "valid_status_values", torch.tensor(statuses), persistent=False
        )

        # Safe neutral defaults make shape/unit tests possible before fitting;
        # production training replaces all three from the train split.
        self.register_buffer("distance_p5", torch.zeros(self.n_selected_sensors))
        self.register_buffer(
            "distance_p95",
            torch.full((self.n_selected_sensors,), 2550.0),
        )
        self.register_buffer(
            "distance_median",
            torch.full((self.n_selected_sensors,), 1275.0),
        )
        self.register_buffer("normalization_fitted", torch.tensor(False))

        self.frame_mlp = nn.Sequential(
            nn.Linear(64, frame_hidden_dim),
            nn.SiLU(),
            nn.Linear(frame_hidden_dim, frame_feature_dim),
            nn.SiLU(),
        )
        self.sensor_embedding = nn.Parameter(
            torch.randn(self.n_selected_sensors, frame_feature_dim) * 0.02
        )
        self.temporal_gru = nn.GRU(
            input_size=frame_feature_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.n_selected_sensors * temporal_hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
        )

    @torch.no_grad()
    def set_normalization_statistics(
        self, *, p5: Tensor, p95: Tensor, median: Tensor
    ) -> None:
        """Install per-selected-sensor train-split robust statistics."""
        values = {"p5": p5, "p95": p95, "median": median}
        converted: dict[str, Tensor] = {}
        for name, value in values.items():
            tensor = torch.as_tensor(
                value, device=self.distance_p5.device, dtype=self.distance_p5.dtype
            )
            if tensor.shape != (self.n_selected_sensors,):
                raise ValueError(
                    f"Temporal Beaver {name} must have shape "
                    f"({self.n_selected_sensors},), got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Temporal Beaver {name} must be finite")
            converted[name] = tensor
        if torch.any(converted["p95"] <= converted["p5"]):
            raise ValueError("Temporal Beaver p95 must be greater than p5 per sensor")
        self.distance_p5.copy_(converted["p5"])
        self.distance_p95.copy_(converted["p95"])
        self.distance_median.copy_(converted["median"])
        self.normalization_fitted.fill_(True)

    @torch.no_grad()
    def set_temporal_statistics(
        self, *, p5: Tensor, p95: Tensor, median: Tensor
    ) -> None:
        """Compatibility alias for callers that name the transform by policy."""
        self.set_normalization_statistics(p5=p5, p95=p95, median=median)

    def _select_inputs(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if distance.ndim < 4 or distance.shape[-2:] != (4, 4):
            raise ValueError(
                "Temporal distance must have shape (..., history, sensors, 4, 4), "
                f"got {tuple(distance.shape)}"
            )
        if status.shape != distance.shape:
            raise ValueError("Temporal status shape must match distance shape")
        if present.shape != distance.shape[:-2]:
            raise ValueError(
                f"Expected temporal present {tuple(distance.shape[:-2])}, got "
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
            "Temporal Beaver sensor axis must contain either the complete "
            f"{self.n_sensors}-sensor layout or the resolved "
            f"{self.n_selected_sensors}-sensor subset, got {sensor_count}"
        )

    def preprocess(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return ``(..., T, sensors, 4, 4, 4)`` temporal frame features.

        The final dimension is ordered as normalized distance, delta distance,
        genuine-valid mask, and zero-value flag.
        """
        if not bool(self.normalization_fitted.item()):
            raise RuntimeError(
                "Temporal Beaver robust normalization statistics are not fitted; "
                "refusing to fall back to global 2550 mm normalization"
            )
        distance, status, present = self._select_inputs(distance, status, present)
        status_values = self.valid_status_values.to(device=status.device, dtype=status.dtype)
        status_is_valid = (status.unsqueeze(-1) == status_values).any(dim=-1)
        sensor_is_present = present.bool().unsqueeze(-1).unsqueeze(-1)
        zero_flag_bool = distance.eq(0)
        genuine = (
            status_is_valid
            & sensor_is_present
            & torch.isfinite(distance)
            & ~zero_flag_bool
        )

        statistic_shape = [1] * (distance.ndim - 3) + [
            self.n_selected_sensors,
            1,
            1,
        ]
        frame_statistic_shape = [1] * (distance.ndim - 4) + [
            self.n_selected_sensors,
            1,
            1,
        ]
        median = self.distance_median.to(distance.dtype).view(
            *frame_statistic_shape
        )
        previous = median.expand(
            *distance.shape[:-4], self.n_selected_sensors, 4, 4
        )
        filled_frames: list[Tensor] = []
        for time_index in range(self.history_steps):
            measurement = distance[..., time_index, :, :, :]
            valid_now = genuine[..., time_index, :, :, :]
            previous = torch.where(valid_now, measurement, previous)
            filled_frames.append(previous)
        filled_distance = torch.stack(filled_frames, dim=-4)

        p5 = self.distance_p5.to(distance.dtype).view(*statistic_shape)
        p95 = self.distance_p95.to(distance.dtype).view(*statistic_shape)
        scale = p95 - p5
        normalized_distance = ((filled_distance - p5) / scale).clamp(0.0, 1.0)
        measured_normalized = ((distance - p5) / scale).clamp(0.0, 1.0)

        delta_first = torch.zeros_like(measured_normalized[..., :1, :, :, :])
        consecutive_genuine = (
            genuine[..., 1:, :, :, :] & genuine[..., :-1, :, :, :]
        )
        adjacent_delta = torch.where(
            consecutive_genuine,
            measured_normalized[..., 1:, :, :, :]
            - measured_normalized[..., :-1, :, :, :],
            torch.zeros_like(measured_normalized[..., 1:, :, :, :]),
        )
        delta_distance = torch.cat((delta_first, adjacent_delta), dim=-4)
        features = torch.stack(
            (
                normalized_distance,
                delta_distance,
                genuine.to(distance.dtype),
                zero_flag_bool.to(distance.dtype),
            ),
            dim=-1,
        )
        return features, {
            "selected_distance": distance,
            "valid": genuine,
            "zero_flag": zero_flag_bool,
            "filled_distance": filled_distance,
            "normalized_distance": normalized_distance,
            "delta_distance": delta_distance,
        }

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        frame_features, intermediates = self.preprocess(distance, status, present)
        # (..., time, sensor, row, col, channel) -> (..., sensor, time, 64)
        frame_vectors = frame_features.flatten(start_dim=-3).transpose(-3, -2)
        encoded_frames = self.frame_mlp(frame_vectors)
        embedding_shape = [1] * (encoded_frames.ndim - 3) + [
            self.n_selected_sensors,
            1,
            -1,
        ]
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
        feature = self.fusion_mlp(concatenated)
        if return_intermediates:
            intermediates.update(
                {
                    "frame_features": frame_features,
                    "encoded_frames": encoded_frames,
                    "sensor_tokens": sensor_tokens,
                    "concatenated": concatenated,
                }
            )
            return feature, intermediates
        return feature


__all__ = [
    "Key4BeaverEncoder",
    "StructuredBeaverEncoder",
    "TemporalBeaverEncoder",
]
