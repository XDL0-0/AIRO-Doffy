"""Contact phase labeling utilities for ForceFlow++."""

from __future__ import annotations

import torch
from torch import Tensor

CONTACT_PHASE_NAMES = ("free", "light", "heavy", "transition")


def force_sequence_to_phase(
    force: Tensor | None,
    free_threshold_n: float,
    heavy_threshold_n: float,
    transition_delta_n: float,
    num_modes: int = 4,
) -> Tensor | None:
    """Convert a force observation/history into contact phase labels.

    Labels are:
      0. free space
      1. light contact
      2. heavy contact
      3. transition, if enabled by ``num_modes >= 4``
    """
    if force is None:
        return None
    if force.ndim < 2:
        raise ValueError(f"force tensor must have at least 2 dims, got {tuple(force.shape)}")

    if force.ndim == 2:
        current = force[..., :3]
        previous = torch.zeros_like(current)
    else:
        current = force[:, -1, ..., :3].reshape(force.shape[0], -1)
        current = current[:, :3]
        if force.shape[1] > 1:
            previous = force[:, -2, ..., :3].reshape(force.shape[0], -1)
            previous = previous[:, :3]
        else:
            previous = torch.zeros_like(current)

    force_mag = torch.linalg.norm(current, dim=-1)
    delta_mag = torch.linalg.norm(current - previous, dim=-1)

    labels = torch.zeros(force_mag.shape, dtype=torch.long, device=force.device)
    if num_modes >= 2:
        labels = torch.where(force_mag >= free_threshold_n, torch.ones_like(labels), labels)
    if num_modes >= 3:
        labels = torch.where(force_mag >= heavy_threshold_n, torch.full_like(labels, 2), labels)
    if num_modes >= 4:
        labels = torch.where(delta_mag >= transition_delta_n, torch.full_like(labels, 3), labels)

    return labels.clamp(max=num_modes - 1)

