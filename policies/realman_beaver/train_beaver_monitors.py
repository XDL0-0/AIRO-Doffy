"""Train the primary and strict-backup Beaver execution monitors.

The script deliberately bypasses image decoding and the 1.2 GB diffusion
model. It trains only the two small MLPs, writes episode-level split/event
manifests, and verifies the backup monitor exhaustively before saving it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from policies.realman_beaver.modules.beaver_monitor import (
    BackupBeaverMonitor,
    TemporalBeaverMonitor,
)


@dataclass(frozen=True)
class Episode:
    index: int
    bottle: int
    distance: Tensor
    status: Tensor
    present: Tensor
    tightness: Tensor
    joint1: Tensor
    contact_onset: int
    lift_onset: int


def _tensor_column(table, name: str, dtype=np.float32) -> Tensor:
    values = np.asarray(table[name].to_pylist(), dtype=dtype)
    return torch.from_numpy(values)


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
    import pyarrow as pa

    table = pa.concat_tables(tables)
    episode_index = np.asarray(table["episode_index"], dtype=np.int64)
    frame_index = np.asarray(table["frame_index"], dtype=np.int64)
    episodes: list[Episode] = []
    for episode_id in sorted(np.unique(episode_index).tolist()):
        rows = np.flatnonzero(episode_index == episode_id)
        rows = rows[np.argsort(frame_index[rows], kind="stable")]
        episode_table = table.take(pa.array(rows))
        distance = _tensor_column(
            episode_table, "observation.beaver.distance_mm"
        )
        status = _tensor_column(
            episode_table, "observation.beaver.target_status"
        )
        present = _tensor_column(episode_table, "observation.beaver.present")
        state = _tensor_column(episode_table, "observation.state")
        tightness = _tensor_column(episode_table, "tightness", np.int64).bool()
        contact_candidates = torch.nonzero(tightness, as_tuple=False)
        if not contact_candidates.numel():
            raise ValueError(f"Episode {episode_id} has no tightness transition")
        contact_onset = int(contact_candidates[0])
        joint1 = state[:, 1]
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
                tightness=tightness,
                joint1=joint1,
                contact_onset=contact_onset,
                lift_onset=lift_onset,
            )
        )
    if len(episodes) != 125:
        raise ValueError(f"Expected 125 episodes, found {len(episodes)}")
    return episodes


def _split_ids() -> dict[str, list[int]]:
    split = {"train": [], "validation": [], "test": []}
    for start in range(0, 125, 25):
        split["train"].extend(range(start, start + 15))
        split["validation"].extend(range(start + 15, start + 20))
        split["test"].extend(range(start + 20, start + 25))
    return split


def _frame_features(
    monitor: TemporalBeaverMonitor, episode: Episode
) -> Tensor:
    with torch.no_grad():
        return monitor.extract_frame_features(
            episode.distance.unsqueeze(0),
            episode.status.unsqueeze(0),
            episode.present.unsqueeze(0),
        )[0]


def _examples(
    monitor: TemporalBeaverMonitor,
    episodes: Iterable[Episode],
    *,
    strides: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    features, labels, episode_ids, frame_ids = [], [], [], []
    for episode in episodes:
        frame = _frame_features(monitor, episode)
        count = frame.shape[0]
        lift = torch.arange(count) >= episode.lift_onset
        target = torch.stack((lift, episode.tightness), dim=-1).float()
        for stride in strides:
            lag_indices = torch.tensor(monitor.lag_steps) * int(stride)
            source = torch.arange(count).unsqueeze(1) - lag_indices.unsqueeze(0)
            source.clamp_(min=0)
            temporal = frame[source].flatten(start_dim=1)
            features.append(temporal)
            labels.append(target)
            episode_ids.append(torch.full((count,), episode.index, dtype=torch.long))
            frame_ids.append(torch.arange(count))
    return (
        torch.cat(features),
        torch.cat(labels),
        torch.cat(episode_ids),
        torch.cat(frame_ids),
    )


def _classification_metrics(logits: Tensor, target: Tensor) -> dict[str, float]:
    state = logits >= 0
    truth = target.bool()
    result: dict[str, float] = {}
    for column, name in enumerate(("lift", "contact")):
        pred, actual = state[:, column], truth[:, column]
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        tn = int((~pred & ~actual).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        result[f"{name}_precision"] = precision
        result[f"{name}_recall"] = recall
        result[f"{name}_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
        result[f"{name}_balanced_accuracy"] = 0.5 * (
            tp / max(tp + fn, 1) + tn / max(tn + fp, 1)
        )
    return result


def _event_metrics(
    logits: Tensor,
    episode_ids: Tensor,
    frame_ids: Tensor,
    episodes: dict[int, Episode],
) -> dict[str, float | list[float]]:
    result: dict[str, float | list[float]] = {}
    states = logits >= 0
    for column, name, onset_name in (
        (0, "lift", "lift_onset"),
        (1, "contact", "contact_onset"),
    ):
        errors: list[float] = []
        misses = 0
        for episode_id in sorted(torch.unique(episode_ids).tolist()):
            mask = episode_ids == episode_id
            frames = frame_ids[mask]
            predictions = states[mask, column]
            hits = frames[predictions]
            if not hits.numel():
                misses += 1
                continue
            predicted = int(hits.min())
            actual = int(getattr(episodes[int(episode_id)], onset_name))
            errors.append(float(predicted - actual))
        values = np.asarray(errors, dtype=np.float64)
        result[f"{name}_event_miss_rate"] = misses / max(
            len(torch.unique(episode_ids)), 1
        )
        result[f"{name}_event_error_median_frames"] = float(np.median(values))
        result[f"{name}_event_abs_error_mean_frames"] = float(
            np.abs(values).mean()
        )
        result[f"{name}_event_error_q10_q90_frames"] = [
            float(value) for value in np.quantile(values, [0.1, 0.9])
        ]
        result[f"{name}_event_early_gt_1s_rate"] = float((values < -24).mean())
    return result


@torch.no_grad()
def _evaluate_primary(
    monitor: TemporalBeaverMonitor,
    features: Tensor,
    labels: Tensor,
    episode_ids: Tensor,
    frame_ids: Tensor,
    episode_map: dict[int, Episode],
    device: torch.device,
) -> dict[str, object]:
    monitor.eval()
    chunks = []
    for start in range(0, len(features), 4096):
        chunks.append(monitor.mlp(features[start : start + 4096].to(device)).cpu())
    logits = torch.cat(chunks)
    loss = float(F.binary_cross_entropy_with_logits(logits, labels))
    return {
        "loss": loss,
        **_classification_metrics(logits, labels),
        **_event_metrics(logits, episode_ids, frame_ids, episode_map),
    }


def _train_primary(
    episodes: list[Episode], output: Path, device: torch.device, seed: int
) -> tuple[TemporalBeaverMonitor, dict[str, object]]:
    split = _split_ids()
    episode_map = {episode.index: episode for episode in episodes}
    monitor = TemporalBeaverMonitor().to(device)
    base_monitor = TemporalBeaverMonitor()
    data: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
    for name, ids in split.items():
        selected = [episode_map[index] for index in ids]
        # Stride 2 is training-only rate augmentation: a deployment tick at
        # ~12 Hz sees approximately twice the physical duration of 24 Hz data.
        data[name] = _examples(
            base_monitor,
            selected,
            strides=(1, 2) if name == "train" else (1,),
        )
    train_x, train_y, _, _ = data["train"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=512,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(monitor.parameters(), lr=3e-4, weight_decay=1e-5)
    positive = train_y.sum(dim=0)
    negative = len(train_y) - positive
    positive_weight = (negative / positive.clamp_min(1)).to(device)
    best_state: dict[str, Tensor] | None = None
    best_score = -math.inf
    best_epoch = 0
    epochs_without_gain = 0
    history = []
    for epoch in range(1, 121):
        monitor.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = monitor.mlp(batch_x)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_y, pos_weight=positive_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(monitor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val = _evaluate_primary(
            monitor, *data["validation"], episode_map, device
        )
        score = (
            float(val["lift_f1"])
            + float(val["contact_f1"])
            - 0.002
            * (
                float(val["lift_event_abs_error_mean_frames"])
                + float(val["contact_event_abs_error_mean_frames"])
            )
            - float(val["lift_event_early_gt_1s_rate"])
            - float(val["contact_event_early_gt_1s_rate"])
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            **val,
        }
        history.append(record)
        print(json.dumps({"primary": record}, sort_keys=True), flush=True)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in monitor.state_dict().items()
            }
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        if epoch >= 25 and epochs_without_gain >= 15:
            break
    assert best_state is not None
    monitor.load_state_dict(best_state)
    validation = _evaluate_primary(
        monitor, *data["validation"], episode_map, device
    )
    # Keep the exact validation-selected weights. A fresh random restart on
    # train+validation is a different model and its test metrics would not be
    # justified by the selection run. The held-out test remains untouched
    # until this one final evaluation.
    test = _evaluate_primary(
        monitor, *data["test"], episode_map, device
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "architecture": "all9_temporal_feature_mlp",
        "output_order": ["lift_state", "contact_state"],
        "decision_logit": 0.0,
        "history_steps": monitor.history_steps,
        "lag_steps": list(monitor.lag_steps),
        "rate_augmentation_strides": [1, 2],
        "hidden_dims": [128, 64],
        "split": split,
        "best_epoch": best_epoch,
        "selection_validation": validation,
        "production_train_episodes": split["train"],
        "test": test,
        "history": history,
    }
    torch.save(
        {
            "kind": "beaver_monitor",
            "variant": "WRM_wrap_monitor",
            "model": {
                key: value.cpu()
                for key, value in monitor.state_dict().items()
            },
            "metadata": metadata,
        },
        output / "monitor.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return monitor.cpu(), metadata


def _train_backup(output: Path, device: torch.device, seed: int) -> dict[str, object]:
    monitor = BackupBeaverMonitor().to(device)
    patterns = torch.tensor(
        [[(value >> bit) & 1 for bit in range(9)] for value in range(512)],
        dtype=torch.float32,
        device=device,
    )
    key4 = patterns[:, [1, 2, 5, 6]]
    count = key4.sum(dim=-1)
    labels = torch.stack((count >= 1, count >= 2), dim=-1)
    target_logits = labels.float().mul(12.0).sub(6.0)
    optimizer = torch.optim.AdamW(monitor.parameters(), lr=2e-3, weight_decay=0.0)
    for step in range(1, 5001):
        optimizer.zero_grad(set_to_none=True)
        logits = monitor.mlp(key4)
        loss = F.mse_loss(logits, target_logits)
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            exact = bool(torch.equal(logits >= 0, labels))
            print(
                json.dumps(
                    {"backup": {"step": step, "loss": float(loss), "exact": exact}}
                ),
                flush=True,
            )
            if exact and float(loss) < 0.01:
                break
    with torch.no_grad():
        logits = monitor.mlp(key4)
        predicted = logits >= 0
    mismatches = int((predicted != labels).any(dim=-1).sum())
    if mismatches:
        raise RuntimeError(
            f"Backup monitor failed exhaustive 512-pattern test: {mismatches} mismatches"
        )
    nuisance_pairs = 0
    for key_pattern in range(16):
        mask = torch.ones(512, dtype=torch.bool, device=device)
        for local, global_index in enumerate((1, 2, 5, 6)):
            mask &= patterns[:, global_index].bool() == bool(
                (key_pattern >> local) & 1
            )
        group = logits[mask]
        nuisance_pairs += int(group.shape[0])
        if not torch.equal(group >= 0, (group[0] >= 0).expand_as(group)):
            raise RuntimeError("Backup monitor depends on a masked nuisance sensor")
    margin = float(logits.abs().min())
    metadata: dict[str, object] = {
        "architecture": "key4_current_contact_mlp",
        "output_order": ["lift_state", "contact_state"],
        "decision_logit": 0.0,
        "input_sensor_indices": [1, 2, 5, 6],
        "ignored_sensor_indices": [0, 3, 4, 7, 8],
        "distilled_parameters": {
            "near_mm": 0.0,
            "closing_scale_mm": 50.0,
            "lift_min_wrap": 0.25,
            "stop_close_wrap": 0.5,
            "contact_stop_mm": 0.0,
        },
        "truth": {
            "lift_state": "Key4 valid exact-zero contact count >= 1",
            "contact_state": "Key4 valid exact-zero contact count >= 2",
        },
        "exhaustive_patterns": 512,
        "nuisance_invariance_patterns": nuisance_pairs,
        "mismatches": mismatches,
        "minimum_absolute_logit_margin": margin,
        "steps": step,
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "beaver_monitor",
            "variant": "WRM_wrap_monitor_backup",
            "model": {key: value.cpu() for key, value in monitor.state_dict().items()},
            "metadata": metadata,
        },
        output / "monitor.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def _write_manifests(
    episodes: list[Episode], dataset_root: Path, output_root: Path
) -> None:
    events = [
        {
            "episode": episode.index,
            "bottle": episode.bottle,
            "frames": int(episode.tightness.numel()),
            "contact_onset": episode.contact_onset,
            "lift_onset": episode.lift_onset,
            "contact_to_lift_frames": episode.lift_onset - episode.contact_onset,
        }
        for episode in episodes
    ]
    parquet_hash = hashlib.sha256()
    for path in sorted((dataset_root / "data").rglob("*.parquet")):
        parquet_hash.update(path.name.encode())
        parquet_hash.update(str(path.stat().st_size).encode())
    manifest = {
        "dataset_root": str(dataset_root),
        "dataset_fingerprint": parquet_hash.hexdigest(),
        "fps": 24,
        "split": _split_ids(),
        "label_definition": {
            "contact_state": "dataset tightness",
            "lift_state": "J1 <= median(first 30 frames)-0.02 rad for 6 frames",
        },
        "events": events,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    episodes = _load_episodes(args.dataset_root)
    _write_manifests(episodes, args.dataset_root, args.output_root)
    _, primary = _train_primary(
        episodes, args.output_root / "WRM_wrap_monitor", device, args.seed
    )
    backup = _train_backup(
        args.output_root / "WRM_wrap_monitor_backup", device, args.seed
    )
    summary = {
        "primary_test": primary["test"],
        "backup": backup,
    }
    test = primary["test"]
    acceptance = {
        "lift_f1>=0.97": float(test["lift_f1"]) >= 0.97,
        "contact_f1>=0.90": float(test["contact_f1"]) >= 0.90,
        "lift_event_mae<=5_frames": (
            float(test["lift_event_abs_error_mean_frames"]) <= 5.0
        ),
        "contact_event_mae<=15_frames": (
            float(test["contact_event_abs_error_mean_frames"]) <= 15.0
        ),
        "no_event_misses": (
            float(test["lift_event_miss_rate"]) == 0.0
            and float(test["contact_event_miss_rate"]) == 0.0
        ),
        "backup_exhaustive_exact": int(backup["mismatches"]) == 0,
    }
    summary["acceptance"] = acceptance
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    failed = [name for name, passed in acceptance.items() if not passed]
    if failed:
        raise RuntimeError(f"Monitor acceptance gates failed: {failed}")
    print(json.dumps({"complete": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
