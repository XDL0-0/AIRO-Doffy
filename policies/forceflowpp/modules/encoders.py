"""Observation encoders for ForceFlow++."""

from __future__ import annotations

from math import prod

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from lerobot.configs.types import FeatureType
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


def _feature_dim(feature) -> int:
    return int(prod(tuple(feature.shape)))


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.GELU(),
    )


class SmallImageEncoder(nn.Module):
    """Compact CNN encoder used to avoid external pretrained downloads."""

    def __init__(self, out_dim: int, resize_shape: tuple[int, int] | None) -> None:
        super().__init__()
        self.resize_shape = resize_shape
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, images: Tensor) -> Tensor:
        if self.resize_shape is not None and tuple(images.shape[-2:]) != tuple(self.resize_shape):
            images = F.interpolate(images, size=self.resize_shape, mode="bilinear", align_corners=False)
        if images.dtype != torch.float32:
            images = images.float()
        if images.numel() and images.detach().amax() > 2:
            images = images / 255.0
        return self.net(images).flatten(start_dim=1)


class ForceFlowPPObservationEncoder(nn.Module):
    """Encode visual/state/force/tactile observations into DiT conditioning."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.input_features = config.input_features or {}
        self.image_features = config.image_features
        self.n_obs_steps = config.n_obs_steps
        self.hidden_dim = config.hidden_dim

        self.state_dim = _feature_dim(self.input_features[OBS_STATE]) if OBS_STATE in self.input_features else 0
        self.force_dim = _feature_dim(self.input_features[config.force_key]) if config.has_force_feature else 0
        self.torque_dim = _feature_dim(self.input_features[config.torque_key]) if config.has_torque_feature else 0
        self.tactile_dim = _feature_dim(self.input_features[config.tactile_key]) if config.has_tactile_feature else 0
        self.ft_input_dim = self.force_dim + self.torque_dim + self.tactile_dim

        state_in = max(1, self.state_dim * self.n_obs_steps)
        ft_in = max(1, self.ft_input_dim * self.n_obs_steps)
        self.state_encoder = _make_mlp(
            state_in,
            config.state_feature_dim,
            config.state_feature_dim,
            config.encoder_dropout,
        )
        self.ft_encoder = _make_mlp(
            ft_in,
            config.force_tactile_feature_dim,
            config.force_tactile_feature_dim,
            config.encoder_dropout,
        )
        self.state_to_token = nn.Linear(config.state_feature_dim, config.hidden_dim)
        self.ft_to_token = nn.Linear(config.force_tactile_feature_dim, config.hidden_dim)

        self.image_encoder = None
        self.image_to_token = None
        if self.image_features:
            self.image_encoder = SmallImageEncoder(config.image_embedding_dim, config.image_resize_shape)
            self.image_to_token = nn.Linear(config.image_embedding_dim, config.hidden_dim)

        self.conditioning_dim = (
            config.state_feature_dim
            + config.force_tactile_feature_dim
            + config.hidden_dim
        )

    def _batch_info(self, batch: dict[str, Tensor]) -> tuple[int, torch.device, torch.dtype]:
        for value in batch.values():
            if isinstance(value, Tensor):
                return int(value.shape[0]), value.device, value.dtype if value.is_floating_point() else torch.float32
        raise ValueError("Cannot infer batch size from an empty batch.")

    def _zeros(self, batch_size: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.n_obs_steps, dim, device=device, dtype=dtype)

    def _zero_flat(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, 1, device=device, dtype=dtype)

    def _as_sequence(self, tensor: Tensor, feature_key: str) -> Tensor:
        feature = self.input_features[feature_key]
        feature_ndim = len(tuple(feature.shape))
        if tensor.ndim == feature_ndim + 1:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim == feature_ndim + 2:
            if tensor.shape[1] != self.n_obs_steps:
                if tensor.shape[1] == 1:
                    tensor = tensor.expand(-1, self.n_obs_steps, *tensor.shape[2:])
                else:
                    tensor = tensor[:, -self.n_obs_steps :]
            return tensor.reshape(tensor.shape[0], self.n_obs_steps, -1).float()
        raise ValueError(
            f"{feature_key} has shape {tuple(tensor.shape)}, expected batch or sequence "
            f"with feature shape {tuple(feature.shape)}."
        )

    def _vector_sequence(
        self,
        batch: dict[str, Tensor],
        feature_key: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if feature_key not in self.input_features:
            return self._zeros(batch_size, 0, device, dtype)
        if feature_key not in batch:
            return self._zeros(batch_size, _feature_dim(self.input_features[feature_key]), device, dtype)
        return self._as_sequence(batch[feature_key], feature_key).to(device=device, dtype=dtype)

    def _image_tokens(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor]:
        if self.image_encoder is None or OBS_IMAGES not in batch:
            visual_global = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            return None, visual_global

        images = batch[OBS_IMAGES]
        if images.ndim == 5:
            images = images.unsqueeze(1)
        if images.ndim != 6:
            raise ValueError(f"{OBS_IMAGES} must be [B,S,N,C,H,W] or [B,N,C,H,W], got {tuple(images.shape)}")
        if images.shape[1] != self.n_obs_steps:
            if images.shape[1] == 1:
                images = images.expand(-1, self.n_obs_steps, *images.shape[2:])
            else:
                images = images[:, -self.n_obs_steps :]

        b, s, n, c, h, w = images.shape
        flat_images = images.reshape(b * s * n, c, h, w)
        encoded = self.image_encoder(flat_images)
        tokens = self.image_to_token(encoded).reshape(b, s * n, self.hidden_dim)
        visual_global = tokens.mean(dim=1)
        return tokens.to(dtype=dtype), visual_global.to(dtype=dtype)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        batch_size, device, dtype = self._batch_info(batch)

        if self.state_dim:
            state_seq = self._vector_sequence(batch, OBS_STATE, batch_size, device, dtype)
            state_input = state_seq.flatten(start_dim=1)
        else:
            state_input = self._zero_flat(batch_size, device, dtype)
        state_feat = self.state_encoder(state_input)

        ft_parts: list[Tensor] = []
        if self.force_dim:
            ft_parts.append(self._vector_sequence(batch, self.config.force_key, batch_size, device, dtype))
        if self.torque_dim:
            ft_parts.append(self._vector_sequence(batch, self.config.torque_key, batch_size, device, dtype))
        if self.tactile_dim:
            ft_parts.append(self._vector_sequence(batch, self.config.tactile_key, batch_size, device, dtype))
        if ft_parts:
            ft_seq = torch.cat(ft_parts, dim=-1)
            ft_input = ft_seq.flatten(start_dim=1)
        else:
            ft_input = self._zero_flat(batch_size, device, dtype)
        ft_feat = self.ft_encoder(ft_input)

        visual_tokens, visual_global = self._image_tokens(batch, batch_size, device, dtype)

        context_tokens = [
            self.state_to_token(state_feat).unsqueeze(1),
            self.ft_to_token(ft_feat).unsqueeze(1),
        ]
        if visual_tokens is not None:
            context_tokens.append(visual_tokens)
        context = torch.cat(context_tokens, dim=1)

        conditioning_vec = torch.cat([state_feat, visual_global, ft_feat], dim=-1)
        force_seq = None
        if self.force_dim and self.config.force_key in batch:
            force_seq = self._as_sequence(batch[self.config.force_key], self.config.force_key)

        return {
            "conditioning_vec": conditioning_vec,
            "context_tokens": context,
            "ft_feat": ft_feat,
            "force_sequence": force_seq,
        }
