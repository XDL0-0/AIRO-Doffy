"""Flow matching objective for ForceFlow++."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from lerobot.utils.constants import ACTION


class ForceFlowMatchingObjective(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        if self.config.timestep_sampling_strategy == "uniform":
            return torch.rand(batch_size, device=device)
        beta_dist = torch.distributions.Beta(
            self.config.timestep_sampling_alpha,
            self.config.timestep_sampling_beta,
        )
        u = beta_dist.sample((batch_size,)).to(device)
        return self.config.timestep_sampling_s * (1.0 - u)

    def _action_chunk(self, actions: Tensor) -> Tensor:
        if actions.ndim == 2:
            return actions.unsqueeze(1).expand(-1, self.config.horizon, -1)
        if actions.shape[1] > self.config.horizon:
            return actions[:, : self.config.horizon]
        if actions.shape[1] < self.config.horizon:
            pad = actions[:, -1:].expand(-1, self.config.horizon - actions.shape[1], -1)
            return torch.cat([actions, pad], dim=1)
        return actions

    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        encoded: dict[str, Tensor | None],
        z0: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        target_actions = self._action_chunk(batch[ACTION])

        batch_size = target_actions.shape[0]
        t = self._sample_timesteps(batch_size, target_actions.device)
        t_view = t.view(-1, 1, 1)
        x_t = (1.0 - t_view) * z0 + t_view * target_actions
        if self.config.sigma_min:
            x_t = x_t + self.config.sigma_min * torch.randn_like(x_t)

        target_velocity = target_actions - z0
        pred_velocity = model(
            x_t,
            t,
            encoded["context_tokens"],
            encoded["conditioning_vec"],
        )
        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_mask = ~batch["action_is_pad"][:, : loss.shape[1]]
            loss = loss * valid_mask.unsqueeze(-1)
        fm_loss = loss.mean()
        return fm_loss, {"fm_loss": float(fm_loss.detach().cpu())}

    @torch.no_grad()
    def conditional_sample(
        self,
        model: nn.Module,
        encoded: dict[str, Tensor | None],
        z0: Tensor,
    ) -> Tensor:
        x = z0
        time_grid = torch.linspace(
            0.0,
            1.0,
            self.config.num_integration_steps + 1,
            device=x.device,
            dtype=x.dtype,
        )
        for i in range(len(time_grid) - 1):
            t = torch.full((x.shape[0],), time_grid[i], device=x.device, dtype=x.dtype)
            dt = time_grid[i + 1] - time_grid[i]
            velocity = model(
                x,
                t,
                encoded["context_tokens"],
                encoded["conditioning_vec"],
            )
            x = x + dt * velocity
        return x
