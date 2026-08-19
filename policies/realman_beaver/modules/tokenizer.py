"""Asymmetric action/Beaver tokenizer used by the RDP-like policy."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BeaverFrameEncoder(nn.Module):
    """Encode each 9 x 4 x 4 Beaver frame while preserving sensor identity."""

    def __init__(self, n_sensors: int, output_dim: int) -> None:
        super().__init__()
        sensor_dim = max(16, output_dim // 2)
        self.n_sensors = n_sensors
        self.sensor_mlp = nn.Sequential(
            nn.Linear(17, sensor_dim),
            nn.SiLU(),
            nn.Linear(sensor_dim, sensor_dim),
        )
        self.sensor_embedding = nn.Parameter(torch.randn(n_sensors, sensor_dim) * 0.02)
        self.output = nn.Sequential(
            nn.Linear(n_sensors * sensor_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, distance: Tensor, present: Tensor) -> Tensor:
        if distance.shape[-3:] != (self.n_sensors, 4, 4):
            raise ValueError(
                f"Expected Beaver distance (..., {self.n_sensors}, 4, 4), got {distance.shape}"
            )
        if present.shape != distance.shape[:-2]:
            raise ValueError(
                f"Present mask {present.shape} does not match distance {distance.shape}"
            )
        mask = present.to(distance.dtype).clamp(0.0, 1.0)
        sensor_input = torch.cat(
            (distance.flatten(start_dim=-2), mask.unsqueeze(-1)), dim=-1
        )
        encoded = self.sensor_mlp(sensor_input) + self.sensor_embedding
        encoded = encoded * mask.unsqueeze(-1)
        return self.output(encoded.flatten(start_dim=-2))


class AsymmetricBeaverTokenizer(nn.Module):
    """Encode actions only, then decode them from latent tokens and Beaver frames.

    This preserves the key asymmetry of RDP's Action Tokenizer: the encoder cannot
    hide sensor observations in the latent, while the causal GRU decoder can react
    to the newest Beaver measurement at every control step.
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        action_horizon: int,
        downsample_ratio: int,
        hidden_dim: int,
        gru_layers: int,
        n_sensors: int,
        beaver_feature_dim: int,
    ) -> None:
        super().__init__()
        if downsample_ratio <= 0 or downsample_ratio & (downsample_ratio - 1):
            raise ValueError("downsample_ratio must be a positive power of two")
        if action_horizon % downsample_ratio:
            raise ValueError("action_horizon must be divisible by downsample_ratio")
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.action_horizon = action_horizon
        self.downsample_ratio = downsample_ratio

        encoder: list[nn.Module] = [
            nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(int(math.log2(downsample_ratio))):
            encoder.extend(
                (
                    nn.Conv1d(
                        hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GroupNorm(1, hidden_dim),
                    nn.SiLU(),
                )
            )
        self.action_encoder = nn.Sequential(*encoder)
        self.latent_mean = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.latent_logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.beaver_encoder = BeaverFrameEncoder(n_sensors, beaver_feature_dim)
        self.decoder = nn.GRU(
            latent_dim + beaver_feature_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    @property
    def latent_horizon(self) -> int:
        return self.action_horizon // self.downsample_ratio

    def encode(
        self, action: Tensor, sample: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        if action.shape[-2:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"Expected action (..., {self.action_horizon}, {self.action_dim}), got {action.shape}"
            )
        features = self.action_encoder(action.transpose(1, 2))
        mean = self.latent_mean(features).transpose(1, 2)
        logvar = self.latent_logvar(features).transpose(1, 2).clamp(-10.0, 10.0)
        latent = (
            mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean
        )
        return latent, mean, logvar

    def decode(
        self,
        latent: Tensor,
        beaver_distance: Tensor,
        beaver_present: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if latent.shape[1:] != (self.latent_horizon, self.latent_dim):
            raise ValueError(
                f"Expected latent (B, {self.latent_horizon}, {self.latent_dim}), got {latent.shape}"
            )
        expanded = latent.repeat_interleave(self.downsample_ratio, dim=1)
        beaver_features = self.beaver_encoder(beaver_distance, beaver_present)
        if beaver_features.shape[:2] != expanded.shape[:2]:
            raise ValueError("Beaver sequence must have one frame per decoded action")
        decoded, hidden = self.decoder(
            torch.cat((expanded, beaver_features), dim=-1), hidden
        )
        return self.action_head(decoded), hidden

    def decode_step(
        self,
        latent: Tensor,
        beaver_distance: Tensor,
        beaver_present: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent (B, {self.latent_dim}), got {latent.shape}"
            )
        beaver_features = self.beaver_encoder(beaver_distance, beaver_present)
        decoded, hidden = self.decoder(
            torch.cat((latent, beaver_features), dim=-1).unsqueeze(1), hidden
        )
        return self.action_head(decoded[:, 0]), hidden

    def compute_loss(
        self,
        action: Tensor,
        action_is_pad: Tensor,
        beaver_distance: Tensor,
        beaver_present: Tensor,
        kl_weight: float,
    ) -> tuple[Tensor, dict[str, float]]:
        latent, mean, logvar = self.encode(action, sample=self.training)
        reconstruction, _ = self.decode(latent, beaver_distance, beaver_present)
        valid = (~action_is_pad).to(action.dtype)
        reconstruction_per_step = F.l1_loss(
            reconstruction, action, reduction="none"
        ).mean(-1)
        reconstruction_loss = (
            reconstruction_per_step * valid
        ).sum() / valid.sum().clamp_min(1.0)

        latent_valid = valid.reshape(
            action.shape[0], self.latent_horizon, self.downsample_ratio
        ).amax(dim=-1)
        kl_per_token = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean(-1)
        kl_loss = (kl_per_token * latent_valid).sum() / latent_valid.sum().clamp_min(
            1.0
        )
        loss = reconstruction_loss + kl_weight * kl_loss
        return loss, {
            "loss": float(loss.detach()),
            "reconstruction_loss": float(reconstruction_loss.detach()),
            "kl_loss": float(kl_loss.detach()),
        }
