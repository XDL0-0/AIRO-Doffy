"""Offline metrics for the WRM policies.

Pure functions so they are usable from the GPUlab training job and from any
checkpoint evaluation without robot hardware. All trajectory errors are
reported in physical radians by undoing the per-joint action scale.

Metrics implemented per the competition spec:

- total and per-joint trajectory error in physical radians;
- action smoothness and chunk-boundary discontinuity;
- per-bottle metrics from confirmed episode blocks;
- tightness precision / recall / F1 / expected calibration error;
- helper to run complete-model vs modality-zero ablations on a fixed batch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

# Confirmed bottle blocks of WRM_grasp_cylinder_different_sizes_lero_tightness.
BOTTLE_BLOCKS: dict[int, range] = {
    1: range(0, 25),
    2: range(25, 50),
    3: range(50, 75),
    4: range(75, 100),
    5: range(100, 125),
}


def per_joint_trajectory_error(
    predictions: Tensor,
    targets: Tensor,
    action_scales: Tensor,
    *,
    per_joint: bool = False,
) -> dict[str, float]:
    """Absolute error in radians: |predictions - targets| * scale per joint.

    predictions/targets: (..., joints); action_scales: (joints,).
    """
    error = (predictions - targets).abs() * action_scales
    total = float(error.mean())
    if not per_joint:
        return {"trajectory_error_rad": total}
    per_joint_values = error.flatten(0, -2).mean(dim=0)
    return {
        "trajectory_error_rad": total,
        **{
            f"trajectory_error_rad_joint_{joint}": float(value)
            for joint, value in enumerate(per_joint_values)
        },
    }


def action_smoothness(
    actions: Tensor,
    action_scales: Tensor,
    n_action_steps: int,
) -> dict[str, float]:
    """Second differences (jerk proxy) in radians, plus chunk-boundary gaps.

    actions: (..., horizon, joints). The chunk boundary sits between index
    n_action_steps - 1 and n_action_steps: the discontinuity there is the
    replan-boundary artifact measured in the rollouts.
    """
    physical = actions * action_scales
    second = physical[..., 2:, :] - 2 * physical[..., 1:-1, :] + physical[..., :-2, :]
    smoothness = float(second.abs().mean())
    if physical.shape[-2] <= n_action_steps:
        boundary = float("nan")
    else:
        boundary = float(
            (physical[..., n_action_steps, :] - physical[..., n_action_steps - 1, :])
            .abs()
            .mean()
        )
    return {
        "action_smoothness_rad2": smoothness,
        "chunk_boundary_discontinuity_rad": boundary,
    }


def per_bottle_metrics(
    episode_index: Tensor,
    metric_values: Tensor,
    *,
    reduce: str = "mean",
) -> dict[str, float]:
    """Aggregate per-episode metric values into confirmed per-bottle blocks.

    episode_index: (n_episodes,) episode ids; metric_values: (n_episodes,).
    """
    result: dict[str, float] = {}
    for bottle, block in BOTTLE_BLOCKS.items():
        mask = torch.isin(episode_index, torch.tensor(list(block)))
        values = metric_values[mask]
        if values.numel() == 0:
            result[f"bottle_{bottle}_count"] = 0.0
            result[f"bottle_{bottle}_{reduce}"] = float("nan")
            continue
        reduced = float(values.mean() if reduce == "mean" else values.max())
        result[f"bottle_{bottle}_count"] = float(values.numel())
        result[f"bottle_{bottle}_{reduce}"] = reduced
    return result


def tightness_metrics(
    probabilities: Tensor,
    labels: Tensor,
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, float]:
    """Precision / recall / F1 / expected calibration error for the grasp head."""
    predictions = probabilities >= threshold
    labels_bool = labels >= 0.5
    true_positive = (predictions & labels_bool).sum().float()
    false_positive = (predictions & ~labels_bool).sum().float()
    false_negative = (~predictions & labels_bool).sum().float()
    precision = float(true_positive / (true_positive + false_positive).clamp_min(1))
    recall = float(true_positive / (true_positive + false_negative).clamp_min(1))
    f1 = float(
        2 * true_positive
        / (2 * true_positive + false_positive + false_negative).clamp_min(1)
    )
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    bin_indices = (probabilities * n_bins).long().clamp(0, n_bins - 1)
    bin_true = torch.zeros(n_bins)
    bin_count = torch.zeros(n_bins)
    bin_true.scatter_add_(0, bin_indices, labels_bool.float())
    bin_count.scatter_add_(0, bin_indices, torch.ones_like(probabilities))
    expected_calibration_error = float(
        ((bin_true / bin_count.clamp_min(1) - probabilities.mean()) ** 2)
        .mul(bin_count / bin_count.sum().clamp_min(1))
        .sum()
        .sqrt()
        if bin_count.sum() > 0
        else float("nan")
    )
    return {
        "tightness_precision": precision,
        "tightness_recall": recall,
        "tightness_f1": f1,
        "tightness_expected_calibration_error": expected_calibration_error,
    }


def ablation_table(
    compute_loss: Callable[[dict[str, Tensor]], float],
    batch: dict[str, Tensor],
    *,
    variant: str,
) -> dict[str, float]:
    """Complete-model objective vs zeroed image / joints / Beaver ablations.

    Zeroing (not shuffling) keeps input statistics fixed; a real fused model
    must change its loss when any mandatory modality is destroyed.
    """
    losses = {"complete": compute_loss(batch)}

    image_zeroed = dict(batch)
    image_zeroed["image"] = torch.zeros_like(batch["image"])
    losses["image_zeroed"] = compute_loss(image_zeroed)

    state_zeroed = dict(batch)
    state_zeroed["state"] = torch.zeros_like(batch["state"])
    losses["state_zeroed"] = compute_loss(state_zeroed)

    beaver_zeroed = dict(batch)
    beaver_zeroed["beaver_history_distance"] = torch.zeros_like(
        batch["beaver_history_distance"]
    )
    losses["beaver_zeroed"] = compute_loss(beaver_zeroed)

    shuffled = dict(batch)
    shuffled["action_delta"] = batch["action_delta"].roll(shifts=3, dims=1)
    losses["action_shuffled"] = compute_loss(shuffled)
    losses["variant"] = float(hash(variant) % 1_000_000)  # provenance marker
    return losses


def expected_calibration_error_from_counts(
    bin_true: Tensor, bin_count: Tensor
) -> float:
    fraction_positive = bin_true / bin_count.clamp_min(1)
    fraction_in_bin = bin_count / bin_count.sum().clamp_min(1)
    deviation = (fraction_positive - fraction_positive.mean()).abs()
    return float((deviation * fraction_in_bin).sum())


def report_unit_interval(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


__all__ = [
    "BOTTLE_BLOCKS",
    "ablation_table",
    "action_smoothness",
    "expected_calibration_error_from_counts",
    "per_bottle_metrics",
    "per_joint_trajectory_error",
    "tightness_metrics",
]
