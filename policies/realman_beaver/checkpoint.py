"""Checkpoint loading helpers for inference and resumed training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from policies.realman_beaver.configuration import RealmanBeaverConfig
from policies.realman_beaver.dataset import LatentNormalizer, ObservationNormalizer
from policies.realman_beaver.modeling import build_policy


def load_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
):
    """Load a trained policy, preferring EMA parameters when available."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("kind") == "tokenizer":
        raise ValueError(
            "A tokenizer-only checkpoint is not deployable; use the final RDP last.pt"
        )
    config = RealmanBeaverConfig.from_dict(checkpoint["config"])
    normalizer = ObservationNormalizer.identity(
        config.model.state_dim, config.model.action_dim
    )
    latent_normalizer = LatentNormalizer.identity(config.rdp.latent_dim)
    policy = build_policy(config, normalizer, latent_normalizer)
    policy.load_state_dict(checkpoint["model"])
    if use_ema and checkpoint.get("ema"):
        parameters = dict(policy.named_parameters())
        for name, value in checkpoint["ema"].items():
            if name in parameters:
                parameters[name].data.copy_(value)
    policy.to(device).eval()
    policy.reset()
    return policy


def checkpoint_summary(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return {
        "variant": checkpoint["config"]["model"]["variant"],
        "kind": checkpoint.get("kind", "unknown"),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "metrics": checkpoint.get("metrics", {}),
    }
