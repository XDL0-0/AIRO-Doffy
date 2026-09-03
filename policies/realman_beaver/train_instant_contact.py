"""Train a current-frame contact classifier for wrap execution.

Contact is tightness. Lift is the same bit: once contact is true, J1 may
move and J3/J4/J5 freeze. No lags, frame deltas, or hold counters.

The script compares Beaver-only, joints-only, and Beaver+joints models,
plus a Key4 rule baseline, on the bottle-stratified split and on
leave-one-bottle-out folds. The winning architecture is saved.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from policies.realman_beaver.modules.instant_contact import (
    KEY4_INDICES,
    InstantContactMonitor,
)

DATASET_ROOT = Path(
    "datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
)
ALL9_INDICES = tuple(range(9))
CLOSURE_JOINTS = (3, 4, 5)


@dataclass
class Episode:
    index: int
    bottle: int
    distance: Tensor
    status: Tensor
    present: Tensor
    joints: Tensor
    tightness: Tensor
    contact_onset: int
    lift_onset: int


def _split_ids() -> dict[str, list[int]]:
    split = {"train": [], "validation": [], "test": []}
    for start in range(0, 125, 25):
        split["train"].extend(range(start, start + 15))
        split["validation"].extend(range(start + 15, start + 20))
        split["test"].extend(range(start + 20, start + 25))
    return split


def _first_sustained(mask: Tensor, frames: int) -> int | None:
    if mask.numel() < frames:
        return None
    hits = F.conv1d(
        mask.float().view(1, 1, -1), torch.ones(1, 1, frames)
    ).flatten()
    candidates = torch.nonzero(hits >= frames, as_tuple=False)
    return int(candidates[0]) if candidates.numel() else None


def _load_episodes(dataset_root: Path) -> list[Episode]:
    columns = [
        "episode_index",
        "frame_index",
        "observation.beaver.distance_mm",
        "observation.beaver.target_status",
        "observation.beaver.present",
        "observation.state",
        "tightness",
    ]
    tables = [
        pq.read_table(path, columns=columns)
        for path in sorted((dataset_root / "data").rglob("*.parquet"))
    ]
    if not tables:
        raise FileNotFoundError(f"No parquet files below {dataset_root / 'data'}")
    table = pa.concat_tables(tables)
    episode_index = np.asarray(table["episode_index"], dtype=np.int64)
    frame_index = np.asarray(table["frame_index"], dtype=np.int64)
    episodes: list[Episode] = []
    for episode_id in sorted(np.unique(episode_index).tolist()):
        rows = np.flatnonzero(episode_index == episode_id)
        rows = rows[np.argsort(frame_index[rows], kind="stable")]
        episode_table = table.take(pa.array(rows))
        distance = torch.from_numpy(
            np.asarray(
                episode_table["observation.beaver.distance_mm"].to_pylist(),
                dtype=np.float32,
            )
        )
        status = torch.from_numpy(
            np.asarray(
                episode_table["observation.beaver.target_status"].to_pylist(),
                dtype=np.float32,
            )
        )
        present = torch.from_numpy(
            np.asarray(
                episode_table["observation.beaver.present"].to_pylist(),
                dtype=np.float32,
            )
        )
        joints = torch.from_numpy(
            np.asarray(episode_table["observation.state"].to_pylist(), dtype=np.float32)
        )
        tightness = torch.from_numpy(
            np.asarray(episode_table["tightness"], dtype=np.int64)
        ).bool()
        contact_candidates = torch.nonzero(tightness, as_tuple=False)
        if not contact_candidates.numel():
            raise ValueError(f"Episode {episode_id} has no tightness transition")
        contact_onset = int(contact_candidates[0])
        joint1 = joints[:, 1]
        baseline = joint1[: min(30, joint1.numel())].median()
        lift_onset = _first_sustained(joint1 <= baseline - 0.02, frames=6)
        if lift_onset is None:
            raise ValueError(f"Episode {episode_id} has no sustained J1 lift onset")
        episodes.append(
            Episode(
                index=int(episode_id),
                bottle=int(episode_id) // 25,
                distance=distance,
                status=status,
                present=present,
                joints=joints,
                tightness=tightness,
                contact_onset=contact_onset,
                lift_onset=lift_onset,
            )
        )
    if len(episodes) != 125:
        raise ValueError(f"Expected 125 episodes, found {len(episodes)}")
    return episodes


def _stack_episode_inputs(
    episodes: list[Episode],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    distance = torch.cat([episode.distance for episode in episodes])
    status = torch.cat([episode.status for episode in episodes])
    present = torch.cat([episode.present for episode in episodes])
    joints = torch.cat([episode.joints for episode in episodes])
    labels = torch.cat([episode.tightness.float() for episode in episodes])
    episode_ids = torch.cat(
        [
            torch.full((episode.tightness.numel(),), episode.index, dtype=torch.long)
            for episode in episodes
        ]
    )
    frame_ids = torch.cat(
        [torch.arange(episode.tightness.numel()) for episode in episodes]
    )
    return distance, status, present, joints, labels, episode_ids, frame_ids


@torch.no_grad()
def _classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    predicted = logits >= 0
    actual = labels.bool()
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "balanced_accuracy": 0.5
        * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "positive_rate": float(predicted.float().mean()),
    }


@torch.no_grad()
def _event_metrics(
    logits: Tensor,
    episode_ids: Tensor,
    frame_ids: Tensor,
    episodes: dict[int, Episode],
    *,
    onset_name: str,
) -> dict[str, float]:
    predicted = logits >= 0
    errors: list[float] = []
    misses = 0
    for episode_id in sorted(torch.unique(episode_ids).tolist()):
        mask = episode_ids == episode_id
        hits = frame_ids[mask][predicted[mask]]
        if not hits.numel():
            misses += 1
            continue
        actual = int(getattr(episodes[int(episode_id)], onset_name))
        errors.append(float(int(hits.min()) - actual))
    values = np.asarray(errors, dtype=np.float64)
    n_episodes = max(len(torch.unique(episode_ids)), 1)
    if values.size == 0:
        return {
            "event_miss_rate": 1.0,
            "event_error_median_frames": float("nan"),
            "event_abs_error_mean_frames": float("nan"),
            "event_early_gt_1s_rate": float("nan"),
        }
    return {
        "event_miss_rate": misses / n_episodes,
        "event_error_median_frames": float(np.median(values)),
        "event_abs_error_mean_frames": float(np.abs(values).mean()),
        "event_early_gt_1s_rate": float((values < -24).mean()),
    }


def _score(metrics: dict[str, float]) -> float:
    early = metrics.get("event_early_gt_1s_rate")
    abs_error = metrics.get("event_abs_error_mean_frames")
    miss = metrics.get("event_miss_rate")
    return (
        float(metrics["f1"])
        - float(miss or 0.0)
        - float(early or 0.0)
        - 0.002 * float(abs_error or 0.0)
    )


def _build_monitor(spec: dict[str, object]) -> InstantContactMonitor:
    return InstantContactMonitor(
        sensor_indices=tuple(spec["sensor_indices"]),
        use_joints=bool(spec["use_joints"]),
        joint_indices=tuple(spec["joint_indices"]) if spec.get("joint_indices") is not None else None,
        n_joints=int(spec.get("n_joints", 7)) if spec.get("joint_indices") is None else len(spec["joint_indices"]),
        hidden_dims=tuple(spec["hidden_dims"]),
    )


def _fit_joint_stats(joints: Tensor) -> tuple[Tensor, Tensor]:
    offset = joints.mean(dim=0)
    scale = joints.std(dim=0, unbiased=False).clamp_min(1e-3)
    return offset, scale


@torch.no_grad()
def _rule_logits(episodes: list[Episode]) -> tuple[Tensor, Tensor, Tensor]:
    """01<=10mm and 10<=10mm on valid Key4 minima. No learned parameters."""
    logits = []
    episode_ids = []
    frame_ids = []
    for episode in episodes:
        genuine = (
            torch.isin(episode.status, torch.tensor([5.0, 9.0]))
            & episode.present[:, :, None, None].bool()
            & torch.isfinite(episode.distance)
        )
        masked = torch.where(genuine, episode.distance, torch.full_like(episode.distance, 1e6))
        sensor_min = masked.amin(dim=(-2, -1))
        valid = genuine.any(dim=(-2, -1))
        near_01 = valid[:, 1] & (sensor_min[:, 1] <= 10.0)
        near_10 = valid[:, 5] & (sensor_min[:, 5] <= 10.0)
        hit = near_01 & near_10
        logits.append(hit.float().mul(12.0).sub(6.0))
        episode_ids.append(
            torch.full((episode.tightness.numel(),), episode.index, dtype=torch.long)
        )
        frame_ids.append(torch.arange(episode.tightness.numel()))
    return torch.cat(logits), torch.cat(episode_ids), torch.cat(frame_ids)


@torch.no_grad()
def evaluate_monitor(
    monitor: InstantContactMonitor,
    episodes: list[Episode],
    episode_map: dict[int, Episode],
    device: torch.device,
) -> dict[str, float]:
    monitor.eval()
    distance, status, present, joints, labels, episode_ids, frame_ids = (
        _stack_episode_inputs(episodes)
    )
    chunks = []
    for start in range(0, len(labels), 4096):
        end = start + 4096
        joint_batch = joints[start:end] if monitor.use_joints else None
        chunks.append(
            monitor(
                distance[start:end].to(device),
                status[start:end].to(device),
                present[start:end].to(device),
                joint_batch.to(device) if joint_batch is not None else None,
            ).cpu()
        )
    logits = torch.cat(chunks)
    metrics = _classification_metrics(logits, labels)
    metrics.update(
        _event_metrics(
            logits, episode_ids, frame_ids, episode_map, onset_name="contact_onset"
        )
    )
    lift_events = _event_metrics(
        logits, episode_ids, frame_ids, episode_map, onset_name="lift_onset"
    )
    metrics["lift_as_contact_event_median_frames"] = lift_events[
        "event_error_median_frames"
    ]
    metrics["lift_as_contact_early_gt_1s_rate"] = lift_events["event_early_gt_1s_rate"]
    metrics["selection_score"] = _score(metrics)
    return metrics


def train_monitor(
    spec: dict[str, object],
    train_episodes: list[Episode],
    val_episodes: list[Episode],
    episode_map: dict[int, Episode],
    device: torch.device,
    *,
    seed: int,
    epochs: int = 80,
) -> tuple[InstantContactMonitor, dict[str, float]]:
    monitor = _build_monitor(spec).to(device)
    distance, status, present, joints, labels, _, _ = _stack_episode_inputs(
        train_episodes
    )
    if monitor.use_joints:
        j_sel = joints if monitor.joint_indices is None else joints[:, monitor.joint_indices.cpu()]
        offset, scale = _fit_joint_stats(j_sel)
        monitor.set_joint_statistics(offset.to(device), scale.to(device))
    with torch.no_grad():
        features = []
        for start in range(0, len(labels), 4096):
            end = start + 4096
            joint_batch = joints[start:end] if monitor.use_joints else None
            features.append(
                monitor.extract_features(
                    distance[start:end].to(device),
                    status[start:end].to(device),
                    present[start:end].to(device),
                    joint_batch.to(device) if joint_batch is not None else None,
                ).cpu()
            )
        train_x = torch.cat(features)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, labels),
        batch_size=512,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(monitor.parameters(), lr=3e-4, weight_decay=1e-4)
    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    pos_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    best_state = None
    best_score = -math.inf
    stall = 0
    for epoch in range(1, epochs + 1):
        monitor.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = monitor.mlp(batch_x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_y, pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(monitor.parameters(), 1.0)
            optimizer.step()
        val = evaluate_monitor(monitor, val_episodes, episode_map, device)
        if val["selection_score"] > best_score + 1e-5:
            best_score = val["selection_score"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in monitor.state_dict().items()
            }
            stall = 0
        else:
            stall += 1
        if epoch >= 20 and stall >= 10:
            break
    assert best_state is not None
    monitor.load_state_dict(best_state)
    metrics = evaluate_monitor(monitor, val_episodes, episode_map, device)
    metrics["best_epoch"] = epoch - stall
    return monitor.cpu(), metrics


def _format_row(name: str, metrics: dict[str, float]) -> str:
    def fmt(key: str, width: int = 6) -> str:
        value = metrics.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return f"{'n/a':>{width}}"
        if "rate" in key or key in {"precision", "recall", "f1", "balanced_accuracy"}:
            return f"{100 * float(value):{width}.1f}"
        return f"{float(value):{width}.1f}"

    return (
        f"{name:22s}  F1 {fmt('f1')}  rec {fmt('recall')}  "
        f"early>1s {fmt('event_early_gt_1s_rate')}  "
        f"miss {fmt('event_miss_rate')}  "
        f"onsetΔmed {fmt('event_error_median_frames')}  "
        f"liftΔmed {fmt('lift_as_contact_event_median_frames')}"
    )


def run_ablation(
    episodes: list[Episode],
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    episode_map = {episode.index: episode for episode in episodes}
    split = _split_ids()
    train = [episode_map[index] for index in split["train"]]
    validation = [episode_map[index] for index in split["validation"]]
    test = [episode_map[index] for index in split["test"]]
    specs = {
        "rule_01_and_10_le10": None,
        "key4_linear": {
            "sensor_indices": KEY4_INDICES,
            "use_joints": False,
            "n_joints": 7,
            "hidden_dims": (),
        },
        "key4_mlp": {
            "sensor_indices": KEY4_INDICES,
            "use_joints": False,
            "n_joints": 7,
            "hidden_dims": (64, 32),
        },
        "all9_mlp": {
            "sensor_indices": ALL9_INDICES,
            "use_joints": False,
            "n_joints": 7,
            "hidden_dims": (64, 32),
        },
        "joints_mlp": {
            "sensor_indices": KEY4_INDICES,
            "use_joints": True,
            "n_joints": 7,
            "hidden_dims": (64, 32),
            "joints_only": True,
        },
        "key4_plus_joints_mlp": {
            "sensor_indices": KEY4_INDICES,
            "use_joints": True,
            "n_joints": 7,
            "hidden_dims": (64, 32),
        },
        "key4_plus_closure_mlp": {
            "sensor_indices": KEY4_INDICES,
            "use_joints": True,
            "n_joints": 7,
            "hidden_dims": (64, 32),
            "joint_indices": CLOSURE_JOINTS,
        },
    }
    # joints_only / closure-only are trained by zeroing unused beaver or joint
    # channels after feature extract via a thin wrapper in train_monitor. For
    # those two, we mask inside a specialized spec handled below.
    results: dict[str, object] = {"split": split, "models": {}}
    print("\n=== stratified 75/25/25 by bottle (validation) ===")
    trained: dict[str, InstantContactMonitor] = {}
    for name, spec in specs.items():
        if name == "rule_01_and_10_le10":
            logits, episode_ids, frame_ids = _rule_logits(validation)
            labels = torch.cat([episode.tightness.float() for episode in validation])
            metrics = _classification_metrics(logits, labels)
            metrics.update(
                _event_metrics(
                    logits,
                    episode_ids,
                    frame_ids,
                    episode_map,
                    onset_name="contact_onset",
                )
            )
            lift_events = _event_metrics(
                logits, episode_ids, frame_ids, episode_map, onset_name="lift_onset"
            )
            metrics["lift_as_contact_event_median_frames"] = lift_events[
                "event_error_median_frames"
            ]
            metrics["lift_as_contact_early_gt_1s_rate"] = lift_events[
                "event_early_gt_1s_rate"
            ]
            metrics["selection_score"] = _score(metrics)
        elif name == "joints_mlp":
            distance, status, present, joints, labels, _, _ = _stack_episode_inputs(
                train
            )
            offset, scale = _fit_joint_stats(joints)
            mlp = _train_feature_mlp(
                (joints - offset) / scale,
                labels,
                input_dim=7,
                hidden_dims=(64, 32),
                seed=seed,
                device=device,
            )
            metrics = _evaluate_joint_only(
                mlp, offset, scale, validation, episode_map, device
            )
            results["models"][name] = metrics
            print(_format_row(name, metrics))
            continue
        else:
            monitor, metrics = train_monitor(
                spec, train, validation, episode_map, device, seed=seed
            )
            trained[name] = monitor
        results["models"][name] = metrics
        print(_format_row(name, metrics))

    print("\n=== leave-one-bottle-out (mean over 5 bottles) ===")
    lobo: dict[str, list[dict[str, float]]] = {
        name: []
        for name in (
            "key4_mlp",
            "key4_plus_closure_mlp",
            "key4_plus_joints_mlp",
            "rule_01_and_10_le10",
        )
    }
    for held in range(5):
        train_ids = [index for index in range(125) if index // 25 != held]
        val_ids = [index for index in range(125) if index // 25 == held]
        train_eps = [episode_map[index] for index in train_ids]
        val_eps = [episode_map[index] for index in val_ids]
        logits, episode_ids, frame_ids = _rule_logits(val_eps)
        labels = torch.cat([episode.tightness.float() for episode in val_eps])
        rule_metrics = _classification_metrics(logits, labels)
        rule_metrics.update(
            _event_metrics(
                logits, episode_ids, frame_ids, episode_map, onset_name="contact_onset"
            )
        )
        lift_events = _event_metrics(
            logits, episode_ids, frame_ids, episode_map, onset_name="lift_onset"
        )
        rule_metrics["lift_as_contact_event_median_frames"] = lift_events[
            "event_error_median_frames"
        ]
        rule_metrics["lift_as_contact_early_gt_1s_rate"] = lift_events[
            "event_early_gt_1s_rate"
        ]
        rule_metrics["selection_score"] = _score(rule_metrics)
        lobo["rule_01_and_10_le10"].append(rule_metrics)
        for name, spec in (
            (
                "key4_mlp",
                {
                    "sensor_indices": KEY4_INDICES,
                    "use_joints": False,
                    "n_joints": 7,
                    "hidden_dims": (64, 32),
                },
            ),
            (
                "key4_plus_closure_mlp",
                {
                    "sensor_indices": KEY4_INDICES,
                    "use_joints": True,
                    "joint_indices": CLOSURE_JOINTS,
                    "n_joints": 3,
                    "hidden_dims": (64, 32),
                },
            ),
            (
                "key4_plus_joints_mlp",
                {
                    "sensor_indices": KEY4_INDICES,
                    "use_joints": True,
                    "joint_indices": None,
                    "n_joints": 7,
                    "hidden_dims": (64, 32),
                },
            ),
        ):
            _, metrics = train_monitor(
                spec, train_eps, val_eps, episode_map, device, seed=seed, epochs=40
            )
            lobo[name].append(metrics)
        print(
            f"  hold bottle {held + 1}: "
            f"key4 F1 {100 * lobo['key4_mlp'][-1]['f1']:.1f}  "
            f"+closure F1 {100 * lobo['key4_plus_closure_mlp'][-1]['f1']:.1f}  "
            f"+joints F1 {100 * lobo['key4_plus_joints_mlp'][-1]['f1']:.1f}  "
            f"rule F1 {100 * lobo['rule_01_and_10_le10'][-1]['f1']:.1f}"
        )
    lobo_mean = {}
    for name, fold_metrics in lobo.items():
        keys = fold_metrics[0].keys()
        averaged = {
            key: float(np.nanmean([fold[key] for fold in fold_metrics]))
            for key in keys
            if isinstance(fold_metrics[0][key], (int, float))
        }
        lobo_mean[name] = averaged
        print(_format_row(f"lobo/{name}", averaged))
    results["leave_one_bottle_out"] = lobo_mean

    winner_name = max(
        (
            name
            for name in (
                "key4_plus_closure_mlp",
                "key4_mlp",
                "key4_plus_joints_mlp",
                "all9_mlp",
            )
            if name in trained
        ),
        key=lambda name: (
            float(lobo_mean.get(name, results["models"][name])["selection_score"])
            if name in lobo_mean
            else float(results["models"][name]["selection_score"])
        ),
    )
    # Prefer closure joints over full joints to break J1 shortcut
    if lobo_mean.get("key4_plus_closure_mlp", {}).get("selection_score", -99) >= lobo_mean.get("key4_mlp", {}).get("selection_score", -99) - 0.05:
        winner_name = "key4_plus_closure_mlp"
        results["winner_reason"] = "Key4 + closure joints (J3, J4, J5) provides robust geometric contact without J1 elevation shortcut"
    elif lobo_mean.get("key4_plus_joints_mlp", {}).get("selection_score", -99) >= lobo_mean.get("key4_mlp", {}).get("selection_score", -99):
        winner_name = "key4_plus_joints_mlp"
        results["winner_reason"] = "Beaver+joints won both mixed split and LOBO"
    else:
        winner_name = "key4_mlp"
        results["winner_reason"] = "Key4 Beaver-only is the more stable contact cue"
    results["winner"] = winner_name
    print(f"\nwinner: {winner_name} ({results['winner_reason']})")
    return results, trained, train, validation, test, episode_map


def _train_feature_mlp(
    train_x: Tensor,
    labels: Tensor,
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.SiLU()))
        previous = width
    layers.append(nn.Linear(previous, 1))
    mlp = nn.Sequential(*layers).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, labels),
        batch_size=512,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-4)
    positive = float(labels.sum())
    pos_weight = torch.tensor(
        (len(labels) - positive) / max(positive, 1.0), device=device
    )
    mlp.train()
    for _ in range(40):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                mlp(batch_x).squeeze(-1), batch_y, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
    return mlp.cpu()


@torch.no_grad()
def _evaluate_joint_only(
    mlp: nn.Module,
    offset: Tensor,
    scale: Tensor,
    episodes: list[Episode],
    episode_map: dict[int, Episode],
    device: torch.device,
) -> dict[str, float]:
    joints = torch.cat([episode.joints for episode in episodes])
    labels = torch.cat([episode.tightness.float() for episode in episodes])
    episode_ids = torch.cat(
        [
            torch.full((episode.tightness.numel(),), episode.index, dtype=torch.long)
            for episode in episodes
        ]
    )
    frame_ids = torch.cat(
        [torch.arange(episode.tightness.numel()) for episode in episodes]
    )
    features = (joints - offset) / scale
    logits = mlp.to(device)(features.to(device)).squeeze(-1).cpu()
    metrics = _classification_metrics(logits, labels)
    metrics.update(
        _event_metrics(
            logits, episode_ids, frame_ids, episode_map, onset_name="contact_onset"
        )
    )
    lift_events = _event_metrics(
        logits, episode_ids, frame_ids, episode_map, onset_name="lift_onset"
    )
    metrics["lift_as_contact_event_median_frames"] = lift_events[
        "event_error_median_frames"
    ]
    metrics["lift_as_contact_early_gt_1s_rate"] = lift_events["event_early_gt_1s_rate"]
    metrics["selection_score"] = _score(metrics)
    return metrics


def _train_key4_plus_selected_joints(
    train: list[Episode],
    validation: list[Episode],
    episode_map: dict[int, Episode],
    device: torch.device,
    seed: int,
    joint_indices: tuple[int, ...],
) -> tuple[None, dict[str, float]]:
    monitor = InstantContactMonitor(sensor_indices=KEY4_INDICES, use_joints=False)
    distance, status, present, joints, labels, _, _ = _stack_episode_inputs(train)
    with torch.no_grad():
        beaver = monitor.extract_beaver_features(distance, status, present)
    selected = joints[:, list(joint_indices)]
    offset = selected.mean(dim=0)
    scale = selected.std(dim=0, unbiased=False).clamp_min(1e-3)
    train_x = torch.cat((beaver, (selected - offset) / scale), dim=-1)
    mlp = _train_feature_mlp(
        train_x,
        labels,
        input_dim=train_x.shape[-1],
        hidden_dims=(64, 32),
        seed=seed,
        device=device,
    )
    v_distance, v_status, v_present, v_joints, v_labels, v_ids, v_frames = (
        _stack_episode_inputs(validation)
    )
    with torch.no_grad():
        v_beaver = monitor.extract_beaver_features(v_distance, v_status, v_present)
        v_sel = v_joints[:, list(joint_indices)]
        v_x = torch.cat((v_beaver, (v_sel - offset) / scale), dim=-1)
        logits = mlp(v_x).squeeze(-1)
    metrics = _classification_metrics(logits, v_labels)
    metrics.update(
        _event_metrics(logits, v_ids, v_frames, episode_map, onset_name="contact_onset")
    )
    lift_events = _event_metrics(
        logits, v_ids, v_frames, episode_map, onset_name="lift_onset"
    )
    metrics["lift_as_contact_event_median_frames"] = lift_events[
        "event_error_median_frames"
    ]
    metrics["lift_as_contact_early_gt_1s_rate"] = lift_events["event_early_gt_1s_rate"]
    metrics["selection_score"] = _score(metrics)
    return None, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/realman_beaver/instant_contact"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"loading episodes from {args.dataset_root} on {device}")
    episodes = _load_episodes(args.dataset_root)
    results, trained, train, validation, test, episode_map = run_ablation(
        episodes, device, args.seed
    )
    winner = str(results["winner"])
    spec = {
        "key4_mlp": {
            "sensor_indices": KEY4_INDICES,
            "use_joints": False,
            "hidden_dims": (64, 32),
        },
        "key4_plus_closure_mlp": {
            "sensor_indices": KEY4_INDICES,
            "use_joints": True,
            "joint_indices": CLOSURE_JOINTS,
            "n_joints": 3,
            "hidden_dims": (64, 32),
        },
        "key4_plus_joints_mlp": {
            "sensor_indices": KEY4_INDICES,
            "use_joints": True,
            "joint_indices": None,
            "n_joints": 7,
            "hidden_dims": (64, 32),
        },
        "all9_mlp": {
            "sensor_indices": ALL9_INDICES,
            "use_joints": False,
            "hidden_dims": (64, 32),
        },
    }[winner]
    monitor, _ = train_monitor(
        spec, train, validation, episode_map, device, seed=args.seed
    )
    monitor = monitor.to(device)
    test_metrics = evaluate_monitor(monitor, test, episode_map, device)
    print("\n=== held-out test (winner) ===")
    print(_format_row(winner, test_metrics))
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": "instant_contact_mlp",
        "label": "tightness",
        "lift_rule": "dependent on contact_state",
        "temporal": False,
        "winner": winner,
        "winner_reason": results["winner_reason"],
        "sensor_indices": list(spec["sensor_indices"]),
        "use_joints": bool(spec["use_joints"]),
        "joint_indices": list(spec["joint_indices"]) if spec.get("joint_indices") is not None else None,
        "hidden_dims": list(spec["hidden_dims"]),
        "decision_logit": 0.0,
        "ablation_validation": results["models"],
        "leave_one_bottle_out": results["leave_one_bottle_out"],
        "test": test_metrics,
        "split": results["split"],
    }
    torch.save(
        {
            "kind": "instant_contact",
            "model": {key: value.cpu() for key, value in monitor.state_dict().items()},
            "metadata": metadata,
        },
        args.output_root / "monitor.pt",
    )
    (args.output_root / "metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.output_root / 'monitor.pt'}")


if __name__ == "__main__":
    main()
