"""Small Beaver-only state monitors used by WRM_wrap deployment policies."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _as_history(value: Tensor, *, pixel_value: bool) -> Tensor:
    """Collapse an optional native observation axis and retain B,H,..."""
    expected = 5 if pixel_value else 3
    if value.ndim == expected - 1:
        return value.unsqueeze(1)
    if value.ndim == expected:
        return value
    if value.ndim == expected + 1:
        return value[:, -1]
    kind = "distance/status" if pixel_value else "present"
    raise ValueError(
        f"Monitor {kind} must have {expected - 1}, {expected}, or "
        f"{expected + 1} dims, "
        f"got shape {tuple(value.shape)}"
    )


class TemporalBeaverMonitor(nn.Module):
    """All-nine-sensor temporal feature MLP for lift/contact state.

    Output order is ``[lift_logit, contact_logit]``.  The extractor is kept
    deterministic and bounded: the MLP learns event boundaries, not sensor
    validity conventions or distance units.
    """

    outputs = ("lift_state", "contact_state")

    def __init__(
        self,
        *,
        n_sensors: int = 9,
        history_steps: int = 12,
        lag_steps: tuple[int, ...] = (0, 1, 3, 6, 11),
        valid_statuses: tuple[int, ...] = (5, 9),
        proximity_scales_mm: tuple[float, ...] = (50.0, 150.0, 300.0),
        range_scale_mm: float = 300.0,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if not lag_steps or lag_steps[0] != 0 or max(lag_steps) >= history_steps:
            raise ValueError("lag_steps must start at 0 and fit inside history_steps")
        self.n_sensors = int(n_sensors)
        self.history_steps = int(history_steps)
        self.lag_steps = tuple(int(lag) for lag in lag_steps)
        self.range_scale_mm = float(range_scale_mm)
        self.register_buffer(
            "valid_statuses", torch.tensor(valid_statuses, dtype=torch.long)
        )
        self.register_buffer(
            "proximity_scales_mm",
            torch.tensor(proximity_scales_mm, dtype=torch.float32),
        )
        # Per sensor: proximity at each scale, valid fraction, exact-zero
        # fraction, and valid minimum range.
        per_sensor = len(proximity_scales_mm) + 3
        input_dim = len(self.lag_steps) * self.n_sensors * per_sensor
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous, int(width)),
                    nn.LayerNorm(int(width)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                )
            )
            previous = int(width)
        layers.append(nn.Linear(previous, 2))
        self.mlp = nn.Sequential(*layers)

    def extract_features(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> Tensor:
        distance = _as_history(distance, pixel_value=True)
        status = _as_history(status, pixel_value=True)
        present = _as_history(present, pixel_value=False)
        if distance.shape != status.shape:
            raise ValueError("Monitor distance and status shapes must match")
        if distance.shape[:3] != present.shape:
            raise ValueError("Monitor present shape must match B,H,S")
        if distance.shape[2] != self.n_sensors:
            raise ValueError(
                f"Expected {self.n_sensors} Beaver sensors, got {distance.shape[2]}"
            )
        if distance.shape[1] < self.history_steps:
            raise ValueError(
                f"Expected at least {self.history_steps} history frames, "
                f"got {distance.shape[1]}"
            )

        frame_features = self.extract_frame_features(distance, status, present)
        indices = torch.tensor(
            [frame_features.shape[1] - 1 - lag for lag in self.lag_steps],
            device=frame_features.device,
        )
        features = frame_features.index_select(1, indices)
        return features.flatten(start_dim=1)

    def extract_frame_features(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> Tensor:
        """Return bounded B,H,S,F features before temporal lag selection."""
        distance = _as_history(distance, pixel_value=True).float()
        status = _as_history(status, pixel_value=True).long()
        present = _as_history(present, pixel_value=False) > 0.5
        if distance.shape != status.shape or distance.shape[:3] != present.shape:
            raise ValueError("Monitor distance/status/present shapes do not align")
        status_valid = (status.unsqueeze(-1) == self.valid_statuses).any(dim=-1)
        genuine = status_valid & present[..., None, None]
        finite = torch.isfinite(distance)
        genuine = genuine & finite
        safe_distance = torch.where(genuine, distance.clamp_min(0.0), 0.0)

        proximity = torch.exp(
            -safe_distance.unsqueeze(-1)
            / self.proximity_scales_mm.to(distance).view(1, 1, 1, 1, 1, -1)
        )
        proximity = (proximity * genuine.unsqueeze(-1)).mean(dim=(-3, -2))
        valid_fraction = genuine.float().mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        zero_fraction = (
            genuine & (safe_distance <= 0.0)
        ).float().mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        invalid_fill = torch.full_like(distance, self.range_scale_mm)
        minimum = torch.where(genuine, distance, invalid_fill).amin(
            dim=(-2, -1), keepdim=False
        )
        minimum = (
            minimum.clamp(0.0, self.range_scale_mm) / self.range_scale_mm
        ).unsqueeze(-1)
        features = torch.cat(
            (proximity, valid_fraction, zero_fraction, minimum), dim=-1
        )
        return features

    def forward(self, distance: Tensor, status: Tensor, present: Tensor) -> Tensor:
        return self.mlp(self.extract_features(distance, status, present))


class BackupBeaverMonitor(nn.Module):
    """Key4-only MLP distilled from near=0/lift=.25/stop=.5 rules.

    The other five sensors are absent from the MLP input by construction.
    Consequently their distances, status values, and presence bits cannot
    influence either state even outside the synthetic training distribution.
    """

    outputs = ("lift_state", "contact_state")

    def __init__(
        self,
        *,
        sensor_indices: tuple[int, ...] = (1, 2, 5, 6),
        valid_statuses: tuple[int, ...] = (5, 9),
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if len(sensor_indices) != 4 or len(set(sensor_indices)) != 4:
            raise ValueError("Backup monitor requires four unique Key4 sensors")
        self.register_buffer(
            "sensor_indices", torch.tensor(sensor_indices, dtype=torch.long)
        )
        self.register_buffer(
            "valid_statuses", torch.tensor(valid_statuses, dtype=torch.long)
        )
        self.mlp = nn.Sequential(
            nn.Linear(4, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def extract_features(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> Tensor:
        distance = _as_history(distance, pixel_value=True)[:, -1]
        status = _as_history(status, pixel_value=True)[:, -1].long()
        present = _as_history(present, pixel_value=False)[:, -1] > 0.5
        distance = distance.index_select(1, self.sensor_indices)
        status = status.index_select(1, self.sensor_indices)
        present = present.index_select(1, self.sensor_indices)
        valid = (status.unsqueeze(-1) == self.valid_statuses).any(dim=-1)
        valid = valid & present[..., None, None] & torch.isfinite(distance)
        contact = (valid & (distance <= 0.0)).any(dim=(-2, -1))
        return contact.float()

    def forward(self, distance: Tensor, status: Tensor, present: Tensor) -> Tensor:
        return self.mlp(self.extract_features(distance, status, present))


def monitor_states(logits: Tensor) -> Tensor:
    """Fixed, parameter-free decision boundary for both monitor policies."""
    if logits.shape[-1] != 2:
        raise ValueError(f"Expected two monitor logits, got {tuple(logits.shape)}")
    return logits >= 0.0
