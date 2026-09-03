"""Current-frame contact classifier for wrap execution.

Lift is not an independent event. If contact is right, J1 may move; J3/J4/J5
freeze. Features are the current Beaver frame, optionally concatenated with
the current joint configuration. There are no lags, deltas, or hold counters.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from policies.realman_beaver.modules.beaver_monitor import _as_history

KEY4_INDICES = (1, 2, 5, 6)
DEFAULT_CLOSURE_JOINTS = (3, 4, 5)
PER_SENSOR_FEATURES = 10  # 3 mean prox + 3 best prox + near + zero + valid + min


def _current_beaver(
    distance: Tensor, status: Tensor, present: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    distance = _as_history(distance, pixel_value=True)[:, -1].float()
    status = _as_history(status, pixel_value=True)[:, -1].long()
    present = _as_history(present, pixel_value=False)[:, -1] > 0.5
    return distance, status, present


class InstantContactMonitor(nn.Module):
    """Single-logit contact head over the current Beaver frame."""

    def __init__(
        self,
        *,
        sensor_indices: Sequence[int] = KEY4_INDICES,
        n_sensors: int = 9,
        use_joints: bool = False,
        joint_indices: Sequence[int] | None = DEFAULT_CLOSURE_JOINTS,
        n_joints: int | None = None,
        valid_statuses: Sequence[int] = (5, 9),
        proximity_scales_mm: Sequence[float] = (50.0, 150.0, 300.0),
        best_proximity_scales_mm: Sequence[float] = (20.0, 50.0, 150.0),
        near_threshold_mm: float = 10.0,
        range_scale_mm: float = 300.0,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in sensor_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("sensor_indices must contain unique sensors")
        if any(index < 0 or index >= n_sensors for index in indices):
            raise ValueError(f"sensor_indices must lie in [0, {n_sensors})")
        self.n_sensors = int(n_sensors)
        self.use_joints = bool(use_joints)
        self.near_threshold_mm = float(near_threshold_mm)
        if self.use_joints:
            if joint_indices is not None:
                j_idx = tuple(int(i) for i in joint_indices)
                self.register_buffer("joint_indices", torch.tensor(j_idx, dtype=torch.long))
                self.n_joints = len(j_idx)
            else:
                self.joint_indices = None
                self.n_joints = int(n_joints if n_joints is not None else 7)
            if self.n_joints <= 0:
                raise ValueError("n_joints must be positive when use_joints is set")
        else:
            self.joint_indices = None
            self.n_joints = 0

        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.range_scale_mm = float(range_scale_mm)
        self.register_buffer(
            "sensor_indices", torch.tensor(indices, dtype=torch.long)
        )
        self.register_buffer(
            "valid_statuses",
            torch.tensor(tuple(int(status) for status in valid_statuses), dtype=torch.long),
        )
        self.register_buffer(
            "proximity_scales_mm",
            torch.tensor(
                tuple(float(scale) for scale in proximity_scales_mm),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "best_proximity_scales_mm",
            torch.tensor(
                tuple(float(scale) for scale in best_proximity_scales_mm),
                dtype=torch.float32,
            ),
        )
        beaver_dim = len(indices) * PER_SENSOR_FEATURES
        joint_dim = self.n_joints if self.use_joints else 0
        self.input_dim = beaver_dim + joint_dim
        if self.use_joints:
            self.register_buffer("joint_offset", torch.zeros(self.n_joints))
            self.register_buffer("joint_scale", torch.ones(self.n_joints))
        layers: list[nn.Module] = []
        previous = self.input_dim
        for width in self.hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                )
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.mlp = nn.Sequential(*layers)

    def set_joint_statistics(self, offset: Tensor, scale: Tensor) -> None:
        if not self.use_joints:
            raise ValueError("joint statistics require use_joints=True")
        if offset.shape != (self.n_joints,) or scale.shape != (self.n_joints,):
            raise ValueError(
                f"joint statistics must have shape ({self.n_joints},)"
            )
        if torch.any(scale <= 0):
            raise ValueError("joint_scale must be positive")
        self.joint_offset.copy_(offset.float())
        self.joint_scale.copy_(scale.float())

    def extract_beaver_features(
        self, distance: Tensor, status: Tensor, present: Tensor
    ) -> Tensor:
        distance, status, present = _current_beaver(distance, status, present)
        if distance.shape[1] != self.n_sensors:
            raise ValueError(
                f"Expected {self.n_sensors} Beaver sensors, got {distance.shape[1]}"
            )
        if distance.shape != status.shape or distance.shape[:2] != present.shape:
            raise ValueError("distance/status/present shapes do not align")
        distance = distance.index_select(1, self.sensor_indices)
        status = status.index_select(1, self.sensor_indices)
        present = present.index_select(1, self.sensor_indices)
        genuine = (status.unsqueeze(-1) == self.valid_statuses).any(dim=-1)
        genuine = genuine & present[..., None, None] & torch.isfinite(distance)
        safe = torch.where(genuine, distance.clamp_min(0.0), 0.0)

        # 1. Mean proximity across 16 pixels (3 features)
        proximity = torch.exp(
            -safe.unsqueeze(-1)
            / self.proximity_scales_mm.to(distance).view(1, 1, 1, 1, -1)
        )
        mean_proximity = (proximity * genuine.unsqueeze(-1)).mean(dim=(-3, -2))

        # 2. Best-pixel / min-distance proximity (3 features)
        invalid_fill = torch.full_like(distance, self.range_scale_mm)
        minimum = torch.where(genuine, distance, invalid_fill).amin(dim=(-2, -1))
        best_proximity = torch.exp(
            -minimum.unsqueeze(-1)
            / self.best_proximity_scales_mm.to(distance).view(1, 1, -1)
        )

        # 3. Near fraction <= near_threshold_mm (1 feature)
        near_fraction = (genuine & (safe <= self.near_threshold_mm)).float().mean(dim=(-2, -1)).unsqueeze(-1)

        # 4. Zero fraction <= 0mm (1 feature)
        zero_fraction = (genuine & (safe <= 0.0)).float().mean(dim=(-2, -1)).unsqueeze(-1)

        # 5. Valid fraction (1 feature)
        valid_fraction = genuine.float().mean(dim=(-2, -1)).unsqueeze(-1)

        # 6. Normalized minimum distance (1 feature)
        norm_minimum = (minimum.clamp(0.0, self.range_scale_mm) / self.range_scale_mm).unsqueeze(-1)

        return torch.cat(
            (mean_proximity, best_proximity, near_fraction, zero_fraction, valid_fraction, norm_minimum),
            dim=-1,
        ).flatten(start_dim=1)

    def extract_features(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        joints: Tensor | None = None,
    ) -> Tensor:
        beaver = self.extract_beaver_features(distance, status, present)
        if not self.use_joints:
            return beaver
        if joints is None:
            raise ValueError("InstantContactMonitor(use_joints=True) needs joints")
        if joints.ndim == 3:
            joints = joints[:, -1]
        if self.joint_indices is not None and joints.shape[-1] > self.n_joints:
            joints = joints.index_select(1, self.joint_indices)
        if joints.ndim != 2 or joints.shape[-1] != self.n_joints:
            raise ValueError(
                f"joints must have shape (B, {self.n_joints}), got {tuple(joints.shape)}"
            )
        if beaver.shape[0] != joints.shape[0]:
            raise ValueError("Beaver and joint batch sizes must match")
        normalized = (joints.float() - self.joint_offset) / self.joint_scale
        return torch.cat((beaver, normalized), dim=-1)

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        joints: Tensor | None = None,
    ) -> Tensor:
        return self.mlp(
            self.extract_features(distance, status, present, joints)
        ).squeeze(-1)

    def contact_state(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        joints: Tensor | None = None,
    ) -> Tensor:
        return self.forward(distance, status, present, joints) >= 0.0


def current_joint_tensor(state: Tensor) -> Tensor:
    """Collapse an observation-horizon joint tensor to the current frame."""
    if state.ndim == 3:
        return state[:, -1]
    if state.ndim == 2:
        return state
    raise ValueError(f"joint state must be B,J or B,T,J, got {tuple(state.shape)}")


def current_beaver_frame(
    distance: Tensor, status: Tensor, present: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep a trailing history axis of 1 for the shared frame extractor."""
    return (
        _as_history(distance, pixel_value=True),
        _as_history(status, pixel_value=True),
        _as_history(present, pixel_value=False),
    )
