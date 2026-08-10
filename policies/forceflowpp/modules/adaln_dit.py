"""AdaLN-Zero DiT backbone for ForceFlow++."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class ForceAdaLNDiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero and optional context cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: int,
        dropout: float,
        use_adaln_modulation: bool = True,
    ) -> None:
        super().__init__()
        self.use_adaln_modulation = use_adaln_modulation
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 9 * hidden_dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, context: Tensor, features: Tensor) -> Tensor:
        if not self.use_adaln_modulation:
            self_out, _ = self.self_attn(self.norm_self(x), self.norm_self(x), self.norm_self(x), need_weights=False)
            x = x + self_out
            cross_out, _ = self.cross_attn(self.norm_cross(x), context, context, need_weights=False)
            x = x + cross_out
            return x + self.mlp(self.norm_mlp(x))

        chunks = self.adaLN_modulation(features).chunk(9, dim=-1)
        (
            shift_self,
            scale_self,
            gate_self,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = chunks

        self_in = modulate(self.norm_self(x), shift_self.unsqueeze(1), scale_self.unsqueeze(1))
        self_out, _ = self.self_attn(self_in, self_in, self_in, need_weights=False)
        x = x + gate_self.unsqueeze(1) * self_out

        cross_in = modulate(self.norm_cross(x), shift_cross.unsqueeze(1), scale_cross.unsqueeze(1))
        cross_out, _ = self.cross_attn(cross_in, context, context, need_weights=False)
        x = x + gate_cross.unsqueeze(1) * cross_out

        mlp_in = modulate(self.norm_mlp(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class ForceFlowPPDiT(nn.Module):
    """Action-token DiT predicting flow-matching velocity fields."""

    def __init__(self, config, conditioning_dim: int) -> None:
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        self.timestep_embed_dim = config.timestep_embed_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, 2 * config.timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * config.timestep_embed_dim, config.timestep_embed_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Linear(self.action_dim, config.hidden_dim)
        self.pos_embedding = (
            nn.Parameter(torch.empty(1, config.horizon, config.hidden_dim).normal_(std=0.02))
            if config.use_positional_encoding
            else None
        )

        block_cond_dim = config.timestep_embed_dim + conditioning_dim
        self.blocks = nn.ModuleList(
            [
                ForceAdaLNDiTBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    cond_dim=block_cond_dim,
                    dropout=config.dropout,
                    use_adaln_modulation=config.use_adaln_modulation,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_proj = nn.Linear(config.hidden_dim, self.action_dim)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        context_tokens: Tensor,
        conditioning_vec: Tensor,
    ) -> Tensor:
        time_features = self.time_mlp(timestep)
        block_features = torch.cat([time_features, conditioning_vec], dim=-1)

        hidden = self.input_proj(x)
        if self.pos_embedding is not None:
            hidden = hidden + self.pos_embedding[:, : hidden.shape[1]]

        for block in self.blocks:
            hidden = block(hidden, context_tokens, block_features)
        return self.output_proj(hidden)
