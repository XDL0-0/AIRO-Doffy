"""Contact-aware action prior library for ForceFlow++."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ContactPriorLibrary(nn.Module):
    """A small differentiable Gaussian prior library over action chunks.

    The library stores one diagonal Gaussian per contact mode. At runtime the
    prior selector predicts soft mode weights; v1 samples from the moment-mixed
    Gaussian to keep the path differentiable and stable.
    """

    def __init__(
        self,
        num_modes: int,
        horizon: int,
        action_dim: int,
        std_init: float = 1.0,
        std_floor: float = 0.03,
    ) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.std_floor = float(std_floor)

        self.mean = nn.Parameter(torch.zeros(self.num_modes, self.horizon, self.action_dim))
        self.log_std = nn.Parameter(
            torch.full(
                (self.num_modes, self.horizon, self.action_dim),
                math.log(float(std_init)),
            )
        )

    def soft_parameters(self, weights: Tensor) -> tuple[Tensor, Tensor]:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mean = torch.einsum("bk,kha->bha", weights, self.mean)
        std = torch.einsum("bk,kha->bha", weights, self.log_std.exp())
        std = std.clamp_min(self.std_floor)
        return mean, std

    def sample(self, weights: Tensor, like: Tensor | None = None) -> Tensor:
        mean, std = self.soft_parameters(weights)
        eps = torch.randn_like(mean if like is None else like)
        return mean + std * eps

    def kl_to_standard_normal(self) -> Tensor:
        var = self.log_std.mul(2).exp()
        return 0.5 * (var + self.mean.square() - 1.0 - var.log()).mean()


def uniform_prior_weights(batch_size: int, num_modes: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.full((batch_size, num_modes), 1.0 / num_modes, device=device, dtype=dtype)
