"""Offline metrics for WRM_grok: trajectory error, smoothness, phase calibration."""

from __future__ import annotations

import torch
from torch import Tensor

from policies.realman_beaver.modules.grok_phase_encoder import (
    PHASE_HOLD,
    PHASE_NAMES,
)


def _valid_mask(pad: Tensor | None, reference: Tensor) -> Tensor:
    if pad is None:
        return torch.ones(reference.shape[:2], dtype=torch.bool, device=reference.device)
    mask = pad.bool()
    if mask.ndim == reference.ndim:
        mask = ~mask.any(dim=-1)
    else:
        mask = ~mask
    return mask


def trajectory_error_rad(
    predicted: Tensor,
    target: Tensor,
    pad: Tensor | None = None,
) -> dict[str, float]:
    """Per-joint and total MAE in physical radians (unnormalized joint space)."""
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted/target shapes must match, got {tuple(predicted.shape)} "
            f"vs {tuple(target.shape)}"
        )
    valid = _valid_mask(pad, predicted).unsqueeze(-1)
    abs_err = (predicted - target).abs() * valid
    denom = valid.sum().clamp_min(1)
    per_joint = abs_err.sum(dim=(0, 1)) / valid.sum(dim=(0, 1)).clamp_min(1)
    metrics = {
        "traj_mae_rad": float((abs_err.sum() / denom).item()),
        "traj_rmse_rad": float(
            ((abs_err.pow(2).sum() / denom).sqrt()).item()
        ),
    }
    for index, value in enumerate(per_joint.tolist()):
        metrics[f"traj_mae_joint_{index}_rad"] = float(value)
    return metrics


def action_smoothness(
    actions: Tensor, pad: Tensor | None = None
) -> dict[str, float]:
    """Mean absolute adjacent-step joint change, plus max jump."""
    delta = (actions[:, 1:] - actions[:, :-1]).abs()
    if pad is None:
        valid = torch.ones(delta.shape[:2], dtype=torch.bool, device=actions.device)
    else:
        valid = ~(pad[:, 1:].bool() | pad[:, :-1].bool())
    valid_exp = valid.unsqueeze(-1)
    denom = valid_exp.sum().clamp_min(1)
    mean = (delta * valid_exp).sum() / denom
    peak = torch.where(valid_exp, delta, torch.zeros_like(delta)).amax()
    return {
        "smoothness_mae_rad": float(mean.item()),
        "smoothness_max_rad": float(peak.item()),
    }


def chunk_boundary_discontinuity(
    actions: Tensor,
    n_action_steps: int,
    pad: Tensor | None = None,
) -> dict[str, float]:
    """Compare within-chunk jumps to jumps exactly at replan boundaries."""
    if n_action_steps <= 0:
        raise ValueError("n_action_steps must be positive")
    delta = (actions[:, 1:] - actions[:, :-1]).abs().amax(dim=-1)
    if pad is None:
        valid = torch.ones_like(delta, dtype=torch.bool)
    else:
        valid = ~(pad[:, 1:].bool() | pad[:, :-1].bool())
    horizon = actions.shape[1]
    boundary = torch.zeros_like(delta, dtype=torch.bool)
    for index in range(n_action_steps - 1, horizon - 1, n_action_steps):
        boundary[:, index] = True
    inside = valid & ~boundary
    edge = valid & boundary
    inside_mean = delta[inside].mean() if inside.any() else delta.new_zeros(())
    edge_mean = delta[edge].mean() if edge.any() else delta.new_zeros(())
    return {
        "within_chunk_jump_rad": float(inside_mean.item()),
        "replan_boundary_jump_rad": float(edge_mean.item()),
        "replan_minus_within_rad": float((edge_mean - inside_mean).item()),
    }


def phase_precision_recall_f1(
    logits: Tensor, target: Tensor
) -> dict[str, float]:
    """Per-class and macro precision/recall/F1 for the 3-way phase head."""
    pred = logits.argmax(dim=-1)
    target = target.reshape_as(pred)
    metrics: dict[str, float] = {}
    precisions = []
    recalls = []
    f1s = []
    for class_id, name in enumerate(PHASE_NAMES):
        predicted_pos = pred == class_id
        true_pos = target == class_id
        tp = (predicted_pos & true_pos).sum().clamp_min(0).to(dtype=torch.float32)
        fp = (predicted_pos & ~true_pos).sum().to(dtype=torch.float32)
        fn = (~predicted_pos & true_pos).sum().to(dtype=torch.float32)
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
        metrics[f"phase_{name}_precision"] = float(precision.item())
        metrics[f"phase_{name}_recall"] = float(recall.item())
        metrics[f"phase_{name}_f1"] = float(f1.item())
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    metrics["phase_macro_precision"] = float(torch.stack(precisions).mean().item())
    metrics["phase_macro_recall"] = float(torch.stack(recalls).mean().item())
    metrics["phase_macro_f1"] = float(torch.stack(f1s).mean().item())
    hold_pred = pred == PHASE_HOLD
    hold_true = target == PHASE_HOLD
    tp = (hold_pred & hold_true).sum().to(dtype=torch.float32)
    fp = (hold_pred & ~hold_true).sum().to(dtype=torch.float32)
    fn = (~hold_pred & hold_true).sum().to(dtype=torch.float32)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    metrics["tightness_precision"] = float(precision.item())
    metrics["tightness_recall"] = float(recall.item())
    metrics["tightness_f1"] = float(f1.item())
    return metrics


def expected_calibration_error(
    probabilities: Tensor,
    target: Tensor,
    *,
    n_bins: int = 10,
) -> float:
    """ECE of the hold-class probability against hold labels."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    hold_prob = probabilities[..., PHASE_HOLD].reshape(-1)
    hold_true = (target.reshape(-1) == PHASE_HOLD).to(dtype=hold_prob.dtype)
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=hold_prob.device)
    ece = hold_prob.new_zeros(())
    total = hold_prob.numel()
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (hold_prob >= lower) & (hold_prob < upper)
        if not bool(in_bin.any()):
            continue
        conf = hold_prob[in_bin].mean()
        acc = hold_true[in_bin].mean()
        ece = ece + (in_bin.sum().to(dtype=hold_prob.dtype) / total) * (acc - conf).abs()
    return float(ece.item())


def count_trainable_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def deployment_weight_bytes(module) -> int:
    total = 0
    for tensor in module.state_dict().values():
        total += tensor.numel() * tensor.element_size()
    return total
