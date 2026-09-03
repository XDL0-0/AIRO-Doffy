"""Phase Task Monitor for Realman Beaver wrap execution.

Dual-head architecture:
1. Contact Stop Head: Predicts when firm grasp contact is established,
   stopping finger closure to prevent OVER-GRASP (crushing).
2. Lift Ready Head: Predicts when the grasp is stable and lift should initiate,
   unblocking and executing J1 lift.

Physics-informed safety guardrails:
- Anti-Undergrasp Shield: Prevents premature stopping in mid-air (J3 > -1.35 rad).
- Dual-Jaw Asymmetry: Tracks J3 and J4 jaws independently.
- Deceleration / Impedance Cue: Uses finite-difference finger velocities.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from policies.realman_beaver.modules.beaver_monitor import _as_history

KEY4_INDICES = (1, 2, 5, 6)  # 01, 02 on J3 jaw; 10, 11 on J4 jaw
PER_SENSOR_FEATURES = 10


def _current_beaver(
    distance: Tensor, status: Tensor, present: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    distance = _as_history(distance, pixel_value=True)[:, -1].float()
    status = _as_history(status, pixel_value=True)[:, -1].long()
    present = _as_history(present, pixel_value=False)[:, -1] > 0.5
    return distance, status, present


class PhaseTaskMonitor(nn.Module):
    """Dual-head Phase Task Monitor predicting Contact Stop and Lift Ready."""

    def __init__(
        self,
        *,
        sensor_indices: Sequence[int] = KEY4_INDICES,
        n_sensors: int = 9,
        closure_joint_indices: Sequence[int] = (3, 4, 5),
        lift_joint_index: int = 1,
        valid_statuses: Sequence[int] = (5, 9),
        proximity_scales_mm: Sequence[float] = (50.0, 150.0, 300.0),
        best_proximity_scales_mm: Sequence[float] = (20.0, 50.0, 150.0),
        range_scale_mm: float = 300.0,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        indices = tuple(int(i) for i in sensor_indices)
        self.sensor_indices_tuple = indices
        self.register_buffer("sensor_indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "closure_joint_indices",
            torch.tensor(tuple(int(i) for i in closure_joint_indices), dtype=torch.long),
        )
        self.lift_joint_index = int(lift_joint_index)
        self.n_sensors = int(n_sensors)
        self.range_scale_mm = float(range_scale_mm)

        self.register_buffer(
            "valid_statuses",
            torch.tensor(tuple(int(s) for s in valid_statuses), dtype=torch.long),
        )
        self.register_buffer(
            "proximity_scales_mm",
            torch.tensor(tuple(float(s) for s in proximity_scales_mm), dtype=torch.float32),
        )
        self.register_buffer(
            "best_proximity_scales_mm",
            torch.tensor(tuple(float(s) for s in best_proximity_scales_mm), dtype=torch.float32),
        )

        # Feature dimensions:
        # Per Key4 sensor: 10 features (3 mean prox + 3 min prox + near + zero + valid + norm_min) = 40
        # Per-jaw summary: 5 features (min_j3, min_j4, j3_contact, j4_contact, dual_contact) = 5
        # Joint positions: J1, J3, J4, J5 = 4
        # Joint deltas: dJ3, dJ4, dJ5 = 3
        # Total = 52 dims
        self.input_dim = len(indices) * PER_SENSOR_FEATURES + 5 + 4 + 3

        self.register_buffer("joint_offset", torch.zeros(7))
        self.register_buffer("joint_scale", torch.ones(7))

        # Shared representation trunk
        trunk_layers: list[nn.Module] = []
        prev = self.input_dim
        for h in hidden_dims:
            trunk_layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)

        # Head 1: Contact Stop (tightness / physical grasp established)
        self.head_contact = nn.Linear(prev, 1)

        # Head 2: Lift Ready (transport / ready to pull up J1)
        self.head_lift = nn.Linear(prev, 1)

        # Runtime online history for joint velocity calculation & debounce
        self._history_joints: deque[Tensor] = deque(maxlen=5)
        self._contact_hold_count = 0

    def set_joint_statistics(self, offset: Tensor, scale: Tensor) -> None:
        self.joint_offset.copy_(offset)
        self.joint_scale.copy_(scale.clamp_min(1e-3))

    def extract_features(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        joints: Tensor,
        delta_joints: Tensor | None = None,
    ) -> Tensor:
        curr_dist, curr_stat, curr_pres = _current_beaver(distance, status, present)
        batch_size = curr_dist.shape[0]

        # 1. Key4 per-sensor features
        parts: list[Tensor] = []
        sensor_mins: list[Tensor] = []
        for idx in self.sensor_indices:
            dist = curr_dist[:, idx]  # (B, 4, 4)
            stat = curr_stat[:, idx]
            pres = curr_pres[:, idx, None, None]

            genuine = torch.isin(stat, self.valid_statuses) & pres & torch.isfinite(dist)
            valid_frac = genuine.float().mean(dim=(-2, -1)).unsqueeze(-1)  # (B, 1)

            masked_mean = torch.where(genuine, dist, torch.full_like(dist, self.range_scale_mm))
            mean_dist = masked_mean.mean(dim=(-2, -1)).unsqueeze(-1)  # (B, 1)
            mean_prox = torch.exp(
                -mean_dist / self.proximity_scales_mm.view(1, -1)
            )  # (B, 3)

            masked_min = torch.where(genuine, dist, torch.full_like(dist, 1e6))
            min_dist = masked_min.amin(dim=(-2, -1)).unsqueeze(-1)  # (B, 1)
            min_dist = torch.where(valid_frac > 0, min_dist, torch.full_like(min_dist, self.range_scale_mm))
            sensor_mins.append(min_dist)

            best_prox = torch.exp(
                -min_dist / self.best_proximity_scales_mm.view(1, -1)
            )  # (B, 3)

            near = (min_dist <= 10.0).float()
            zero = (min_dist <= 0.5).float()
            norm_min = (min_dist / self.range_scale_mm).clamp(0.0, 1.0)

            parts.extend([mean_prox, best_prox, near, zero, valid_frac, norm_min])

        # 2. Per-jaw summaries (01,02 on J3; 10,11 on J4)
        min_01, min_02, min_10, min_11 = sensor_mins
        min_j3 = torch.minimum(min_01, min_02)
        min_j4 = torch.minimum(min_10, min_11)
        j3_contact = (min_j3 <= 10.0).float()
        j4_contact = (min_j4 <= 10.0).float()
        dual_contact = (j3_contact > 0.5) & (j4_contact > 0.5)

        jaw_feats = torch.cat(
            [
                (min_j3 / self.range_scale_mm).clamp(0.0, 1.0),
                (min_j4 / self.range_scale_mm).clamp(0.0, 1.0),
                j3_contact,
                j4_contact,
                dual_contact.float(),
            ],
            dim=-1,
        )
        parts.append(jaw_feats)

        # 3. Kinematic positions (J1, J3, J4, J5) normalized
        norm_joints = (joints - self.joint_offset) / self.joint_scale
        selected_joints = torch.cat(
            [
                norm_joints[:, [self.lift_joint_index]],
                norm_joints[:, self.closure_joint_indices],
            ],
            dim=-1,
        )
        parts.append(selected_joints)

        # 4. Kinematic velocity / differential (dJ3, dJ4, dJ5)
        if delta_joints is None:
            delta_joints = torch.zeros(batch_size, 3, device=joints.device, dtype=joints.dtype)
        parts.append(delta_joints * 10.0)  # scale for neural input

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        joints: Tensor,
        delta_joints: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = self.extract_features(distance, status, present, joints, delta_joints)
        feat = self.trunk(x)
        logit_contact = self.head_contact(feat).squeeze(-1)
        logit_lift = self.head_lift(feat).squeeze(-1)
        return logit_contact, logit_lift

    def predict_phase_state(
        self,
        distance: Tensor,
        status: Tensor,
        present: Tensor,
        joints: Tensor,
        delta_joints: Tensor | None = None,
        *,
        contact_threshold: float = 0.5,
        lift_threshold: float = 0.5,
        anti_undergrasp_j3_limit: float = -1.35,
    ) -> dict[str, Tensor]:
        logit_c, logit_l = self.forward(distance, status, present, joints, delta_joints)
        prob_c = torch.sigmoid(logit_c)
        prob_l = torch.sigmoid(logit_l)

        # Anti-Undergrasp Shield:
        # If J3 > -1.35 rad (gripper is wide open), it is physically IMPOSSIBLE
        # to have contact with any cylinder. Force contact_stop = False and lift = False!
        j3 = joints[:, 3] if joints.ndim == 2 else joints[3]
        in_open_zone = j3 > anti_undergrasp_j3_limit

        contact_stop = (prob_c >= contact_threshold) & (~in_open_zone)
        lift_ready = (prob_l >= lift_threshold) & (~in_open_zone)

        return {
            "logit_contact": logit_c,
            "logit_lift": logit_l,
            "prob_contact": prob_c,
            "prob_lift": prob_l,
            "contact_stop": contact_stop,
            "lift_ready": lift_ready,
        }

    def reset(self) -> None:
        self._history_joints.clear()
        self._contact_hold_count = 0
