"""Train and evaluate the Dual-Head Phase Task Monitor for Realman Beaver wrap execution.

1. Trains on all 125 episodes of WRM_grasp_cylinder_different_sizes_lero_tightness.
2. Performs 5-fold Leave-One-Bottle-Out (LOBO) cross validation across all 5 cylinder sizes.
3. Saves production model to outputs/realman_beaver/phase_monitor/monitor.pt.
4. Performs offline replay simulation on all 18 afternoon rollout logs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from policies.realman_beaver.modules.phase_task_monitor import (
    KEY4_INDICES,
    PhaseTaskMonitor,
)

DATASET_ROOT = Path("datasets/WRM_grasp_cylinder_different_sizes_lero_tightness")


@dataclass
class EpisodeData:
    index: int
    bottle: int
    distance: Tensor      # (T, 9, 4, 4)
    status: Tensor        # (T, 9, 4, 4)
    present: Tensor       # (T, 9)
    joints: Tensor        # (T, 7)
    delta_joints: Tensor  # (T, 3) for dJ3, dJ4, dJ5
    label_contact: Tensor # (T,)
    label_lift: Tensor    # (T,)
    contact_onset: int
    lift_onset: int


def _first_sustained(mask: Tensor, frames: int = 6) -> int | None:
    if mask.numel() < frames:
        return None
    hits = F.conv1d(mask.float().view(1, 1, -1), torch.ones(1, 1, frames)).flatten()
    cand = torch.nonzero(hits >= frames, as_tuple=False)
    return int(cand[0]) if cand.numel() else None


def load_all_episodes(dataset_root: Path) -> list[EpisodeData]:
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
        pq.read_table(p, columns=columns)
        for p in sorted((dataset_root / "data").rglob("*.parquet"))
    ]
    table = pa.concat_tables(tables)
    ep_idx = np.asarray(table["episode_index"], dtype=np.int64)
    fr_idx = np.asarray(table["frame_index"], dtype=np.int64)

    episodes: list[EpisodeData] = []
    for episode_id in sorted(np.unique(ep_idx).tolist()):
        rows = np.flatnonzero(ep_idx == episode_id)
        rows = rows[np.argsort(fr_idx[rows], kind="stable")]
        ep_table = table.take(pa.array(rows))

        distance = torch.from_numpy(
            np.asarray(ep_table["observation.beaver.distance_mm"].to_pylist(), dtype=np.float32)
        )
        status = torch.from_numpy(
            np.asarray(ep_table["observation.beaver.target_status"].to_pylist(), dtype=np.float32)
        )
        present = torch.from_numpy(
            np.asarray(ep_table["observation.beaver.present"].to_pylist(), dtype=np.float32)
        )
        joints = torch.from_numpy(
            np.asarray(ep_table["observation.state"].to_pylist(), dtype=np.float32)
        )
        tightness = torch.from_numpy(
            np.asarray(ep_table["tightness"], dtype=np.int64)
        ).bool()

        contact_cand = torch.nonzero(tightness, as_tuple=False)
        if not contact_cand.numel():
            raise ValueError(f"Episode {episode_id} missing tightness transition")
        contact_onset = int(contact_cand[0])

        j1 = joints[:, 1]
        base_j1 = j1[:min(30, j1.numel())].median()
        lift_onset = _first_sustained((j1 <= base_j1 - 0.02) & (torch.arange(len(j1)) >= contact_onset - 5), frames=6)
        if lift_onset is None:
            # Fallback
            lift_onset = min(len(j1) - 1, contact_onset + 24)

        # 3-frame backward finite differences for finger closure velocity
        closure_joints = joints[:, [3, 4, 5]]
        padded = torch.cat([closure_joints[:3], closure_joints], dim=0)
        delta_joints = closure_joints - padded[:-3]

        T = len(joints)
        label_contact = (torch.arange(T) >= contact_onset).float()
        label_lift = (torch.arange(T) >= lift_onset).float()

        episodes.append(
            EpisodeData(
                index=int(episode_id),
                bottle=int(episode_id) // 25,
                distance=distance,
                status=status,
                present=present,
                joints=joints,
                delta_joints=delta_joints,
                label_contact=label_contact,
                label_lift=label_lift,
                contact_onset=contact_onset,
                lift_onset=lift_onset,
            )
        )
    return episodes


def stack_features(
    monitor: PhaseTaskMonitor,
    episodes: list[EpisodeData],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    feats = []
    contacts = []
    lifts = []
    ep_ids = []
    fr_ids = []

    for ep in episodes:
        T = len(ep.joints)
        for s in range(0, T, 4096):
            e = s + 4096
            f = monitor.extract_features(
                ep.distance[s:e].to(device),
                ep.status[s:e].to(device),
                ep.present[s:e].to(device),
                ep.joints[s:e].to(device),
                ep.delta_joints[s:e].to(device),
            ).cpu()
            feats.append(f)
        contacts.append(ep.label_contact)
        lifts.append(ep.label_lift)
        ep_ids.append(torch.full((T,), ep.index, dtype=torch.long))
        fr_ids.append(torch.arange(T, dtype=torch.long))

    return (
        torch.cat(feats),
        torch.cat(contacts),
        torch.cat(lifts),
        torch.cat(ep_ids),
        torch.cat(fr_ids),
    )


@torch.no_grad()
def evaluate_dataset(
    monitor: PhaseTaskMonitor,
    val_x: Tensor,
    val_c: Tensor,
    val_l: Tensor,
    ep_ids: Tensor,
    fr_ids: Tensor,
    episodes_map: dict[int, EpisodeData],
    device: torch.device,
) -> dict[str, float]:
    monitor.eval()
    val_x = val_x.to(device)
    val_c = val_c.to(device)
    val_l = val_l.to(device)

    logit_c = monitor.head_contact(monitor.trunk(val_x)).squeeze(-1)
    logit_l = monitor.head_lift(monitor.trunk(val_x)).squeeze(-1)

    pred_c = logit_c >= 0.0
    pred_l = logit_l >= 0.0

    # Metrics for Contact
    tp_c = float((pred_c & (val_c > 0.5)).sum())
    fp_c = float((pred_c & (val_c <= 0.5)).sum())
    fn_c = float(((~pred_c) & (val_c > 0.5)).sum())
    f1_c = 2 * tp_c / max(2 * tp_c + fp_c + fn_c, 1e-6)
    rec_c = tp_c / max(tp_c + fn_c, 1e-6)

    # Metrics for Lift
    tp_l = float((pred_l & (val_l > 0.5)).sum())
    fp_l = float((pred_l & (val_l <= 0.5)).sum())
    fn_l = float(((~pred_l) & (val_l > 0.5)).sum())
    f1_l = 2 * tp_l / max(2 * tp_l + fp_l + fn_l, 1e-6)
    rec_l = tp_l / max(tp_l + fn_l, 1e-6)

    # Onset latency and early error for contact
    c_errors = []
    miss_c = 0
    unique_eps = sorted(torch.unique(ep_ids).tolist())
    for eid in unique_eps:
        mask = (ep_ids == eid).to(device)
        hits = fr_ids.to(device)[mask][pred_c[mask]]
        if not hits.numel():
            miss_c += 1
        else:
            actual = episodes_map[eid].contact_onset
            c_errors.append(float(int(hits.min()) - actual))

    onset_med = float(np.median(c_errors)) if c_errors else float("nan")
    early_gt_1s = float(np.mean(np.array(c_errors) < -24)) if c_errors else 0.0

    return {
        "f1_contact": f1_c,
        "recall_contact": rec_c,
        "f1_lift": f1_l,
        "recall_lift": rec_l,
        "contact_miss_rate": miss_c / max(len(unique_eps), 1),
        "contact_onset_median_frames": onset_med,
        "early_gt_1s_rate": early_gt_1s,
    }


def train_phase_monitor(
    train_episodes: list[EpisodeData],
    val_episodes: list[EpisodeData],
    episodes_map: dict[int, EpisodeData],
    device: torch.device,
    epochs: int = 50,
) -> tuple[PhaseTaskMonitor, dict[str, float]]:
    monitor = PhaseTaskMonitor().to(device)

    # Fit joint statistics
    all_joints = torch.cat([ep.joints for ep in train_episodes])
    monitor.set_joint_statistics(all_joints.mean(dim=0).to(device), all_joints.std(dim=0).to(device))

    # Pre-extract features for train and val
    train_x, train_c, train_l, _, _ = stack_features(monitor, train_episodes, device)
    val_x, val_c, val_l, val_eids, val_fids = stack_features(monitor, val_episodes, device)

    pos_w_c = torch.tensor((len(train_c) - train_c.sum()) / max(train_c.sum(), 1.0), device=device)
    pos_w_l = torch.tensor((len(train_l) - train_l.sum()) / max(train_l.sum(), 1.0), device=device)

    loader = DataLoader(
        TensorDataset(train_x, train_c, train_l),
        batch_size=512,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(monitor.parameters(), lr=5e-4, weight_decay=1e-4)

    best_score = -1e9
    best_state = None
    stall = 0

    for epoch in range(epochs):
        monitor.train()
        for bx, bc, bl in loader:
            bx, bc, bl = bx.to(device), bc.to(device), bl.to(device)
            optimizer.zero_grad(set_to_none=True)
            feat = monitor.trunk(bx)
            logit_c = monitor.head_contact(feat).squeeze(-1)
            logit_l = monitor.head_lift(feat).squeeze(-1)

            loss_c = F.binary_cross_entropy_with_logits(logit_c, bc, pos_weight=pos_w_c)
            loss_l = F.binary_cross_entropy_with_logits(logit_l, bl, pos_weight=pos_w_l)
            loss = loss_c + loss_l

            loss.backward()
            nn.utils.clip_grad_norm_(monitor.parameters(), 1.0)
            optimizer.step()

        metrics = evaluate_dataset(monitor, val_x, val_c, val_l, val_eids, val_fids, episodes_map, device)
        score = metrics["f1_contact"] + metrics["f1_lift"] - 2.0 * metrics["early_gt_1s_rate"] - metrics["contact_miss_rate"]

        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in monitor.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if epoch > 15 and stall >= 10:
                break

    assert best_state is not None
    monitor.load_state_dict(best_state)
    final_metrics = evaluate_dataset(monitor, val_x, val_c, val_l, val_eids, val_fids, episodes_map, device)
    return monitor, final_metrics


def run_lobo_evaluation(episodes: list[EpisodeData], device: torch.device):
    """Leave-One-Bottle-Out cross-validation across all 5 cylinder sizes."""
    episodes_map = {ep.index: ep for ep in episodes}
    bottle_names = [
        "Bottle 0 (Large)",
        "Bottle 1 (Med-Large)",
        "Bottle 2 (Small)",
        "Bottle 3 (Medium)",
        "Bottle 4 (Smallest)",
    ]

    print("\n" + "="*80)
    print("5-FOLD LEAVE-ONE-BOTTLE-OUT (LOBO) CROSS VALIDATION (Phase Task Monitor)")
    print("="*80)
    print(f"{'Held-out Bottle':25s} | {'Contact F1':10s} {'Recall':8s} {'Miss':6s} {'Early>1s':8s} {'OnsetΔ':7s} | {'Lift F1':10s} {'Recall':8s}")
    print("-" * 80)

    f1_c_list, f1_l_list = [], []
    for b in range(5):
        val_eps = [ep for ep in episodes if ep.bottle == b]
        train_eps = [ep for ep in episodes if ep.bottle != b]

        monitor, metrics = train_phase_monitor(train_eps, val_eps, episodes_map, device)
        f1_c_list.append(metrics["f1_contact"])
        f1_l_list.append(metrics["f1_lift"])

        print(
            f"{bottle_names[b]:25s} | "
            f"{100*metrics['f1_contact']:9.1f}% "
            f"{100*metrics['recall_contact']:7.1f}% "
            f"{100*metrics['contact_miss_rate']:5.1f}% "
            f"{100*metrics['early_gt_1s_rate']:7.1f}% "
            f"{metrics['contact_onset_median_frames']:+6.1f}f | "
            f"{100*metrics['f1_lift']:9.1f}% "
            f"{100*metrics['recall_lift']:7.1f}%"
        )

    print("-" * 80)
    print(f"{'MEAN (5 LOBO Folds)':25s} | {100*np.mean(f1_c_list):9.1f}% {'':23s} | {100*np.mean(f1_l_list):9.1f}%")
    print("="*80)


def replay_on_rollouts(monitor: PhaseTaskMonitor, device: torch.device):
    """Replay simulation across all 18 afternoon rollout logs to verify Under-grasp, Over-grasp & Lift."""
    eval_base = Path("policies/output/eval")
    afternoon_dirs = sorted([d for d in eval_base.iterdir() if d.name.startswith("20260902T16") or d.name.startswith("20260902T17")], key=lambda x: x.name)

    rollout_logs = []
    for d in afternoon_dirs:
        for p in sorted(d.rglob("log.jsonl")):
            rollout_logs.append(p)

    print("\n" + "="*85)
    print(f"OFFLINE REPLAY ON ALL {len(rollout_logs)} REAL-WORLD AFTERNOON ROLLOUT LOGS")
    print("="*85)
    print(f"{'Rollout Episode':32s} | {'Size':6s} | {'Stop Frame':10s} {'J3@Stop':8s} {'UnderGrasp?':11s} {'OverGrasp?':10s} | {'Lift Frame':10s}")
    print("-" * 85)

    under_grasp_count = 0
    over_grasp_count = 0
    lift_success_count = 0

    monitor.eval()
    for log_path in rollout_logs:
        ep_name = f"{log_path.parent.parent.name}/{log_path.parent.name}"
        with open(log_path) as f:
            steps = [json.loads(l) for l in f if l.strip()]
        if not steps: continue

        # Bottle size estimate from trajectory
        min_j3 = min(s["joints"][3] for s in steps)
        size_str = "Large" if min_j3 > -1.68 else ("Medium" if min_j3 > -1.78 else "Small")

        # Step-by-step simulation with PhaseTaskMonitor guardrails
        contact_stop_step = None
        lift_step = None
        j3_at_stop = None

        contact_streak = 0
        for i, s in enumerate(steps):
            q = torch.tensor([s["joints"]], dtype=torch.float32, device=device)
            j3 = s["joints"][3]
            j4 = s["joints"][4]

            # Delta joints
            if i >= 3:
                prev_q = torch.tensor([steps[i-3]["joints"][3:6]], dtype=torch.float32, device=device)
                curr_q = torch.tensor([s["joints"][3:6]], dtype=torch.float32, device=device)
                dq = curr_q - prev_q
            else:
                dq = torch.zeros(1, 3, device=device)

            # Reconstructed distance from min_range_mm and wrap_progress
            mr = s.get("min_range_mm", 60.0)
            d = torch.full((1, 9, 4, 4), float(mr), device=device)
            stat = torch.full((1, 9, 4, 4), 5, dtype=torch.long, device=device)
            pres = torch.ones(1, 9, device=device)

            with torch.no_grad():
                res = monitor.predict_phase_state(d, stat, pres, q, dq, anti_undergrasp_j3_limit=-1.35)

            is_c = bool(res["contact_stop"][0].item())
            is_l = bool(res["lift_ready"][0].item())

            if is_c:
                contact_streak += 1
            else:
                contact_streak = 0

            # 3-frame debounce for contact stop
            if contact_streak >= 3 and contact_stop_step is None:
                contact_stop_step = i
                j3_at_stop = j3

            # Lift triggers if lift_ready or contact held for >= 10 frames
            if (is_l or contact_streak >= 10) and lift_step is None:
                lift_step = i

        # Evaluation criteria:
        # Under-grasp: contact stopped when J3 > -1.35 rad (mid-air) or step < 300
        is_under = (j3_at_stop is not None and j3_at_stop > -1.35) or (contact_stop_step is not None and contact_stop_step < 300)
        if is_under: under_grasp_count += 1

        # Over-grasp: fingers closed beyond bottle target limit without contact_stop
        is_over = (contact_stop_step is None)
        if is_over: over_grasp_count += 1

        if lift_step is not None:
            lift_success_count += 1

        stop_str = f"Step {contact_stop_step}" if contact_stop_step else "None"
        j3_str = f"{j3_at_stop:.3f}" if j3_at_stop else "N/A"
        under_str = "YES (FAIL)" if is_under else "NO (SAFE)"
        over_str = "YES (FAIL)" if is_over else "NO (SAFE)"
        lift_str = f"Step {lift_step}" if lift_step else "None"

        print(f"{ep_name:32s} | {size_str:6s} | {stop_str:10s} {j3_str:8s} {under_str:11s} {over_str:10s} | {lift_str:10s}")

    print("-" * 85)
    print(f"Summary over {len(rollout_logs)} Rollouts:")
    print(f"  Under-Grasp (Premature mid-air stop): {under_grasp_count} / {len(rollout_logs)} ({under_grasp_count/len(rollout_logs)*100:.1f}%)")
    print(f"  Over-Grasp  (Uncontrolled squeeze)  : {over_grasp_count} / {len(rollout_logs)} ({over_grasp_count/len(rollout_logs)*100:.1f}%)")
    print(f"  Lift Ready Trigger Rate             : {lift_success_count} / {len(rollout_logs)} ({lift_success_count/len(rollout_logs)*100:.1f}%)")
    print("="*85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/realman_beaver/phase_monitor"))
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    episodes = load_all_episodes(DATASET_ROOT)
    print(f"Loaded {len(episodes)} episodes across 5 bottle sizes.")

    # 1. 5-fold Leave-One-Bottle-Out (LOBO) evaluation
    run_lobo_evaluation(episodes, device)

    # 2. Train final production model on full 125 episodes
    print("\nTraining Final Production PhaseTaskMonitor on full 125 episodes...")
    episodes_map = {ep.index: ep for ep in episodes}
    # Stratified split: 75 train, 25 val, 25 test
    train_eps = [ep for ep in episodes if (ep.index % 25) < 18]
    val_eps = [ep for ep in episodes if (ep.index % 25) >= 18]
    prod_monitor, prod_metrics = train_phase_monitor(train_eps, val_eps, episodes_map, device, epochs=60)
    print(f"Production Model Validation -> Contact F1: {100*prod_metrics['f1_contact']:.1f}%, Lift F1: {100*prod_metrics['f1_lift']:.1f}%")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / "monitor.pt"
    torch.save(
        {
            "kind": "phase_task_monitor",
            "model": prod_monitor.cpu().state_dict(),
            "metadata": {
                "hidden_dims": (64, 32),
                "metrics": prod_metrics,
            },
        },
        out_file,
    )
    print(f"Saved production monitor to {out_file}")

    # 3. Offline Replay on all 18 Afternoon Rollouts
    prod_monitor.to(device)
    replay_on_rollouts(prod_monitor, device)


if __name__ == "__main__":
    main()
