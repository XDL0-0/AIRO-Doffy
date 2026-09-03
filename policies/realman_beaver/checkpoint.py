"""Checkpoint loading helpers for inference and resumed training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from policies.realman_beaver.configuration import (
    CLAUDE_BEAVER_VARIANT,
    CODEX_BEAVER_VARIANT,
    DELTA_BEAVER_VARIANT,
    HISTORY_BEAVER_VARIANTS,
    RELATIVE_ACTION_VARIANTS,
    WRAP_BEAVER_VARIANTS,
    WRAP_BEAVER_VARIANT,
    WRAP_DELTA_BEAVER_VARIANT,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import LatentNormalizer, ObservationNormalizer
from policies.realman_beaver.modeling import build_policy


def _temporal_statistics_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Recover history-policy normalizer buffers before model construction."""
    buffer_names = {
        "p5": "normalizer.beaver_temporal_p5",
        "p95": "normalizer.beaver_temporal_p95",
        "median": "normalizer.beaver_temporal_median",
        "sensor_indices": "normalizer.beaver_temporal_sensor_indices",
    }
    missing = [path for path in buffer_names.values() if path not in state_dict]
    if missing:
        raise ValueError(
            f"history-conditioned checkpoint is missing normalization buffers: {missing}"
        )
    statistics = {
        name: state_dict[path].detach().cpu() for name, path in buffer_names.items()
    }
    if not statistics["p5"].numel():
        raise ValueError("history-conditioned checkpoint has empty normalization buffers")
    return statistics


def _delta_statistics_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Recover WRM_delta train-split normalization buffers before model build."""
    buffer_names = {
        "mean": "normalizer.beaver_delta_mean",
        "std": "normalizer.beaver_delta_std",
        "sensor_indices": "normalizer.beaver_delta_sensor_indices",
    }
    missing = [path for path in buffer_names.values() if path not in state_dict]
    if missing:
        raise ValueError(
            f"WRM_delta checkpoint is missing normalization buffers: {missing}"
        )
    statistics = {
        name: state_dict[path].detach().cpu() for name, path in buffer_names.items()
    }
    if not statistics["mean"].numel():
        raise ValueError("WRM_delta checkpoint has empty normalization buffers")
    return statistics


def _action_delta_statistics_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Recover WRM_claude train-split action-delta scales before model build."""
    path = "normalizer.action_delta_scale"
    if path not in state_dict:
        raise ValueError(
            f"WRM_claude checkpoint is missing normalization buffer: {path}"
        )
    scale = state_dict[path].detach().cpu()
    if not scale.numel():
        raise ValueError("WRM_claude checkpoint has empty action delta scales")
    return {"scale": scale}


def _relative_action_statistics_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Recover train-split relative-action buffers before model construction."""
    offset_path = "normalizer.delta_action_offset"
    scale_path = "normalizer.delta_action_scale"
    missing = [path for path in (offset_path, scale_path) if path not in state_dict]
    if missing:
        raise ValueError(
            f"relative-action checkpoint is missing normalization buffers: {missing}"
        )
    delta_action = {
        "offset": state_dict[offset_path].detach().cpu(),
        "scale": state_dict[scale_path].detach().cpu(),
    }
    if not delta_action["offset"].numel():
        raise ValueError("checkpoint has empty relative-action buffers")
    return delta_action


def configure_deployment_steps(
    config: RealmanBeaverConfig,
    *,
    prediction_steps: int | None,
    action_steps: int | None,
) -> tuple[int, int]:
    """Apply evaluation-time prediction/action horizons before model build."""
    variant = config.model.variant
    if variant == "rdp_like":
        window = config.rdp
        if prediction_steps is not None:
            window.action_horizon = int(prediction_steps)
        if action_steps is not None:
            window.slow_replan_steps = int(action_steps)
        configured = (window.action_horizon, window.slow_replan_steps)
    elif variant == "rfm":
        window = config.rfm
        if prediction_steps is not None:
            window.action_horizon = int(prediction_steps)
        if action_steps is not None:
            window.slow_replan_steps = int(action_steps)
        configured = (window.action_horizon, window.slow_replan_steps)
    else:
        if (
            variant == CODEX_BEAVER_VARIANT
            and prediction_steps is not None
            and int(prediction_steps) != config.model.horizon
        ):
            raise ValueError(
                "WRM_codex uses learned horizon queries; prediction_steps must "
                f"remain {config.model.horizon} for this checkpoint"
            )
        if prediction_steps is not None:
            config.model.horizon = int(prediction_steps)
        if action_steps is not None:
            config.model.n_action_steps = int(action_steps)
        configured = (config.model.horizon, config.model.n_action_steps)
    # Re-run all shape/window checks after applying the deployment override.
    config.validate()
    return configured


def configure_wrap_deployment(
    config: RealmanBeaverConfig,
    *,
    near_threshold_mm: float | None = None,
    range_scale_mm: float | None = None,
    lift_min_wrap: float | None = None,
    stop_close_j3_wrap: float | None = None,
    stop_close_j4_wrap: float | None = None,
    stop_close_wrap: float | None = None,
    contact_stop_mm: float | None = None,
    stop_hold_frames: int | None = None,
    lift_hold_frames: int | None = None,
) -> None:
    """Override serialized WRM_wrap gate settings before model construction."""
    overrides = {
        "beaver_wrap_near_threshold_mm": near_threshold_mm,
        "beaver_wrap_range_scale_mm": range_scale_mm,
        "beaver_wrap_lift_min_wrap": lift_min_wrap,
        "beaver_wrap_stop_close_j3_wrap": stop_close_j3_wrap,
        "beaver_wrap_stop_close_j4_wrap": stop_close_j4_wrap,
        "beaver_wrap_stop_close_wrap": stop_close_wrap,
        "beaver_wrap_contact_stop_mm": contact_stop_mm,
        "beaver_wrap_stop_hold_frames": stop_hold_frames,
        "beaver_wrap_lift_hold_frames": lift_hold_frames,
    }
    # The old shared flag remains a convenient alias.  It must override both
    # per-joint values unless an explicit joint-specific value is supplied.
    if stop_close_wrap is not None:
        if stop_close_j3_wrap is None:
            overrides["beaver_wrap_stop_close_j3_wrap"] = stop_close_wrap
        if stop_close_j4_wrap is None:
            overrides["beaver_wrap_stop_close_j4_wrap"] = stop_close_wrap
    requested = {name: value for name, value in overrides.items() if value is not None}
    if not requested:
        return
    if config.model.variant not in WRAP_BEAVER_VARIANTS:
        raise ValueError(
            "WRM_wrap gate overrides require a WRM_wrap variant checkpoint, "
            f"got '{config.model.variant}'"
        )
    integer_fields = {
        "beaver_wrap_stop_hold_frames",
        "beaver_wrap_lift_hold_frames",
    }
    for name, value in requested.items():
        setattr(
            config.model,
            name,
            int(value) if name in integer_fields else float(value),
        )
    config.validate()


def load_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
    *,
    prediction_steps: int | None = None,
    action_steps: int | None = None,
    wrap_near_threshold_mm: float | None = None,
    wrap_range_scale_mm: float | None = None,
    wrap_lift_min_wrap: float | None = None,
    wrap_stop_close_j3_wrap: float | None = None,
    wrap_stop_close_j4_wrap: float | None = None,
    wrap_stop_close_wrap: float | None = None,
    wrap_contact_stop_mm: float | None = None,
    wrap_stop_hold_frames: int | None = None,
    wrap_lift_hold_frames: int | None = None,
):
    """Load a trained policy, preferring EMA parameters when available.

    Deployment overrides are applied to the serialized config before
    constructing the policy, so they reach the actual sampler, queues, and
    WRM_wrap encoder rather than only changing logs or dataclass defaults.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("kind") == "tokenizer":
        raise ValueError(
            "A tokenizer-only checkpoint is not deployable; use the final reactive-policy last.pt"
        )
    raw_config = checkpoint["config"]
    raw_model = raw_config.get("model", {})
    if (
        raw_model.get("variant") in WRAP_BEAVER_VARIANTS
        and "beaver_wrap_closing_scale_mm" not in raw_model
    ):
        # Legacy WRM_wrap reused its near threshold as the closing-rate
        # denominator. Preserve that learned input scaling even when a
        # deployment override changes the binary near threshold.
        raw_config = {
            **raw_config,
            "model": {
                **raw_model,
                "beaver_wrap_closing_scale_mm": raw_model.get(
                    "beaver_wrap_near_threshold_mm", 50.0
                ),
            },
        }
    config = RealmanBeaverConfig.from_dict(raw_config)
    configure_wrap_deployment(
        config,
        near_threshold_mm=wrap_near_threshold_mm,
        range_scale_mm=wrap_range_scale_mm,
        lift_min_wrap=wrap_lift_min_wrap,
        stop_close_j3_wrap=wrap_stop_close_j3_wrap,
        stop_close_j4_wrap=wrap_stop_close_j4_wrap,
        stop_close_wrap=wrap_stop_close_wrap,
        contact_stop_mm=wrap_contact_stop_mm,
        stop_hold_frames=wrap_stop_hold_frames,
        lift_hold_frames=wrap_lift_hold_frames,
    )
    configure_deployment_steps(
        config,
        prediction_steps=prediction_steps,
        action_steps=action_steps,
    )
    temporal_statistics = None
    delta_statistics = None
    action_delta_statistics = None
    delta_action_statistics = None
    if (
        config.model.variant in HISTORY_BEAVER_VARIANTS
        and config.model.variant not in WRAP_BEAVER_VARIANTS
    ):
        temporal_statistics = _temporal_statistics_from_state_dict(checkpoint["model"])
    if config.model.variant == DELTA_BEAVER_VARIANT:
        delta_statistics = _delta_statistics_from_state_dict(checkpoint["model"])
    if config.model.variant == CLAUDE_BEAVER_VARIANT:
        action_delta_statistics = _action_delta_statistics_from_state_dict(
            checkpoint["model"]
        )
    if config.model.variant in RELATIVE_ACTION_VARIANTS:
        delta_action_statistics = _relative_action_statistics_from_state_dict(
            checkpoint["model"]
        )
    normalizer = ObservationNormalizer.identity(
        config.model.state_dim,
        config.model.action_dim,
        temporal_beaver_statistics=temporal_statistics,
        delta_beaver_statistics=delta_statistics,
        action_delta_statistics=action_delta_statistics,
        delta_action_statistics=delta_action_statistics,
    )
    reactive = config.rdp if config.model.variant == "rdp_like" else config.rfm
    latent_normalizer = LatentNormalizer.identity(reactive.latent_dim)
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
    # Eval inspects every registered run before constructing the first policy.
    # Memory-map the ~1.25 GB checkpoints so metadata validation does not read
    # three complete model states into host memory in succession.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    return {
        "variant": checkpoint["config"]["model"]["variant"],
        "kind": checkpoint.get("kind", "unknown"),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "metrics": checkpoint.get("metrics", {}),
    }
