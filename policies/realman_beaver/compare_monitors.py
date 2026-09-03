"""Head-to-head comparison between:
1. Model A: WRM_wrap_monitor_backup (Exact distillation of near=0, lift_min=0.25, stop_close=0.5, contact_stop=0)
2. Model B: PhaseTaskMonitor (Tactile Key4 + Kinematic J3,J4,J5,J1 + Differential dJ + Anti-Undergrasp Shield)

Evaluates on:
- All 125 dataset episodes across 5 bottle sizes (Large to Smallest)
- All 18 real-world rollout logs from 2026-09-02 afternoon
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from policies.realman_beaver.modules.beaver_monitor import BackupBeaverMonitor
from policies.realman_beaver.modules.phase_task_monitor import PhaseTaskMonitor

DATASET_ROOT = Path("datasets/WRM_grasp_cylinder_different_sizes_lero_tightness")
BACKUP_CKPT = Path("policies/downloaded/WRM_wrap_monitor_backup/monitor.pt")
PHASE_CKPT = Path("outputs/realman_beaver/phase_monitor/monitor.pt")


def load_dataset():
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
        for p in sorted((DATASET_ROOT / "data").rglob("*.parquet"))
    ]
    table = pa.concat_tables(tables)
    ep_idx = np.asarray(table["episode_index"], dtype=np.int64)
    fr_idx = np.asarray(table["frame_index"], dtype=np.int64)

    episodes = []
    for episode_id in sorted(np.unique(ep_idx).tolist()):
        rows = np.flatnonzero(ep_idx == episode_id)
        rows = rows[np.argsort(fr_idx[rows], kind="stable")]
        ep_table = table.take(pa.array(rows))

        dist = torch.from_numpy(np.asarray(ep_table["observation.beaver.distance_mm"].to_pylist(), dtype=np.float32))
        stat = torch.from_numpy(np.asarray(ep_table["observation.beaver.target_status"].to_pylist(), dtype=np.float32))
        pres = torch.from_numpy(np.asarray(ep_table["observation.beaver.present"].to_pylist(), dtype=np.float32))
        joints = torch.from_numpy(np.asarray(ep_table["observation.state"].to_pylist(), dtype=np.float32))
        tightness = torch.from_numpy(np.asarray(ep_table["tightness"], dtype=np.int64)).bool()

        contact_cand = torch.nonzero(tightness, as_tuple=False)
        c_onset = int(contact_cand[0]) if contact_cand.numel() else 0

        j1 = joints[:, 1]
        base_j1 = j1[:min(30, j1.numel())].median()
        hits = F.conv1d(((j1 <= base_j1 - 0.02) & (torch.arange(len(j1)) >= c_onset - 5)).float().view(1, 1, -1), torch.ones(1, 1, 6)).flatten()
        cand = torch.nonzero(hits >= 6, as_tuple=False)
        l_onset = int(cand[0]) if cand.numel() else min(len(j1) - 1, c_onset + 24)

        closure_joints = joints[:, [3, 4, 5]]
        padded = torch.cat([closure_joints[:3], closure_joints], dim=0)
        d_joints = closure_joints - padded[:-3]

        episodes.append({
            "id": int(episode_id),
            "bottle": int(episode_id) // 25,
            "dist": dist,
            "stat": stat,
            "pres": pres,
            "joints": joints,
            "d_joints": d_joints,
            "c_onset": c_onset,
            "l_onset": l_onset,
            "tightness": tightness,
        })
    return episodes


def eval_model_on_dataset(model_name: str, model, episodes: list[dict], device: torch.device):
    bottle_names = [
        "Bottle 0 (Large)",
        "Bottle 1 (Med-Large)",
        "Bottle 2 (Small)",
        "Bottle 3 (Medium)",
        "Bottle 4 (Smallest)",
    ]
    results = {}

    for b in range(5):
        b_eps = [ep for ep in episodes if ep["bottle"] == b]
        tp_c, fp_c, fn_c = 0, 0, 0
        tp_l, fp_l, fn_l = 0, 0, 0
        c_errors = []
        miss_c = 0

        for ep in b_eps:
            T = len(ep["joints"])
            d = ep["dist"].to(device)
            s = ep["stat"].to(device)
            p = ep["pres"].to(device)
            q = ep["joints"].to(device)
            dq = ep["d_joints"].to(device)

            with torch.no_grad():
                if model_name == "backup":
                    logits = model(d, s, p)
                    pred_c = logits[:, 1] >= 0.0
                    pred_l = logits[:, 0] >= 0.0
                else:
                    res = model.predict_phase_state(d, s, p, q, dq, anti_undergrasp_j3_limit=-1.35)
                    pred_c = res["contact_stop"]
                    pred_l = res["lift_ready"]

            gt_c = (torch.arange(T, device=device) >= ep["c_onset"])
            gt_l = (torch.arange(T, device=device) >= ep["l_onset"])

            tp_c += int((pred_c & gt_c).sum())
            fp_c += int((pred_c & (~gt_c)).sum())
            fn_c += int(((~pred_c) & gt_c).sum())

            tp_l += int((pred_l & gt_l).sum())
            fp_l += int((pred_l & (~gt_l)).sum())
            fn_l += int(((~pred_l) & gt_l).sum())

            hits = torch.nonzero(pred_c, as_tuple=False)
            if not hits.numel():
                miss_c += 1
            else:
                c_errors.append(int(hits[0]) - ep["c_onset"])

        f1_c = 2 * tp_c / max(2 * tp_c + fp_c + fn_c, 1)
        rec_c = tp_c / max(tp_c + fn_c, 1)
        f1_l = 2 * tp_l / max(2 * tp_l + fp_l + fn_l, 1)
        rec_l = tp_l / max(tp_l + fn_l, 1)
        med_err = np.median(c_errors) if c_errors else float("nan")
        early_gt_1s = np.mean(np.array(c_errors) < -24) if c_errors else 0.0

        results[b] = {
            "name": bottle_names[b],
            "f1_c": f1_c,
            "rec_c": rec_c,
            "f1_l": f1_l,
            "rec_l": rec_l,
            "miss_c": miss_c / len(b_eps),
            "med_err": med_err,
            "early_gt_1s": early_gt_1s,
        }

    return results


def eval_on_rollouts(model_name: str, model, device: torch.device):
    eval_base = Path("policies/output/eval")
    afternoon_dirs = sorted([d for d in eval_base.iterdir() if d.name.startswith("20260902T16") or d.name.startswith("20260902T17")], key=lambda x: x.name)

    rollout_logs = []
    for d in afternoon_dirs:
        for p in sorted(d.rglob("log.jsonl")):
            rollout_logs.append(p)

    outcomes = []
    for log_path in rollout_logs:
        ep_name = f"{log_path.parent.parent.name}/{log_path.parent.name}"
        with open(log_path) as f:
            steps = [json.loads(l) for l in f if l.strip()]
        if not steps: continue

        min_j3 = min(s["joints"][3] for s in steps)
        size_str = "Large" if min_j3 > -1.68 else ("Medium" if min_j3 > -1.78 else "Small")

        contact_stop_step = None
        lift_step = None
        j3_at_stop = None

        streak = 0
        for i, s in enumerate(steps):
            q = torch.tensor([s["joints"]], dtype=torch.float32, device=device)
            if i >= 3:
                curr_q = torch.tensor([s["joints"][3:6]], dtype=torch.float32, device=device)
                prev_q = torch.tensor([steps[i-3]["joints"][3:6]], dtype=torch.float32, device=device)
                dq = curr_q - prev_q
            else:
                dq = torch.zeros(1, 3, device=device)

            mr = s.get("min_range_mm", 60.0)
            d = torch.full((1, 9, 4, 4), float(mr), device=device)
            stat = torch.full((1, 9, 4, 4), 5, dtype=torch.long, device=device)
            pres = torch.ones(1, 9, device=device)

            with torch.no_grad():
                if model_name == "backup":
                    logits = model(d, stat, pres)
                    is_c = bool((logits[:, 1] >= 0.0)[0].item())
                    is_l = bool((logits[:, 0] >= 0.0)[0].item())
                else:
                    res = model.predict_phase_state(d, stat, pres, q, dq, anti_undergrasp_j3_limit=-1.35)
                    is_c = bool(res["contact_stop"][0].item())
                    is_l = bool(res["lift_ready"][0].item())

            if is_c:
                streak += 1
            else:
                streak = 0

            # 3-frame debounce
            if streak >= 3 and contact_stop_step is None:
                contact_stop_step = i
                j3_at_stop = s["joints"][3]

            if (is_l or streak >= 10) and lift_step is None:
                lift_step = i

        is_under = (j3_at_stop is not None and j3_at_stop > -1.35) or (contact_stop_step is not None and contact_stop_step < 300)
        is_over = (contact_stop_step is None)
        outcomes.append({
            "ep_name": ep_name,
            "size": size_str,
            "stop_step": contact_stop_step,
            "j3_stop": j3_at_stop,
            "lift_step": lift_step,
            "is_under": is_under,
            "is_over": is_over,
        })

    return outcomes


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Model A (Backup Monitor)
    backup_data = torch.load(BACKUP_CKPT, map_location=device)
    backup_model = BackupBeaverMonitor(
        sensor_indices=(1, 2, 5, 6),
        valid_statuses=(5, 9),
        hidden_dim=16,
    ).to(device)
    backup_model.load_state_dict(backup_data["model"])
    backup_model.eval()

    # 2. Load Model B (Phase Task Monitor)
    phase_data = torch.load(PHASE_CKPT, map_location=device)
    phase_model = PhaseTaskMonitor().to(device)
    phase_model.load_state_dict(phase_data["model"])
    phase_model.eval()

    # Load 125 dataset episodes
    episodes = load_dataset()

    print("\n" + "="*95)
    print("DATASET EVALUATION: WRM_wrap_monitor_backup VS PhaseTaskMonitor (125 Episodes)")
    print("="*95)
    res_backup = eval_model_on_dataset("backup", backup_model, episodes, device)
    res_phase = eval_model_on_dataset("phase", phase_model, episodes, device)

    print(f"{'Bottle Size':22s} | {'--- Model A: Backup Monitor ---':33s} | {'--- Model B: Phase Task Monitor ---':33s}")
    print(f"{'':22s} | {'Contact F1':11s} {'Miss':6s} {'Early>1s':9s} {'Lift F1':8s} | {'Contact F1':11s} {'Miss':6s} {'Early>1s':9s} {'Lift F1':8s}")
    print("-" * 95)

    for b in range(5):
        b_name = res_backup[b]["name"]
        a = res_backup[b]
        m = res_phase[b]
        print(
            f"{b_name:22s} | "
            f"{100*a['f1_c']:9.1f}% "
            f"{100*a['miss_c']:5.1f}% "
            f"{100*a['early_gt_1s']:7.1f}% "
            f"{100*a['f1_l']:7.1f}% | "
            f"{100*m['f1_c']:9.1f}% "
            f"{100*m['miss_c']:5.1f}% "
            f"{100*m['early_gt_1s']:7.1f}% "
            f"{100*m['f1_l']:7.1f}%"
        )

    mean_a_f1_c = np.mean([res_backup[b]["f1_c"] for b in range(5)])
    mean_a_f1_l = np.mean([res_backup[b]["f1_l"] for b in range(5)])
    mean_m_f1_c = np.mean([res_phase[b]["f1_c"] for b in range(5)])
    mean_m_f1_l = np.mean([res_phase[b]["f1_l"] for b in range(5)])

    print("-" * 95)
    print(f"{'OVERALL MEAN':22s} | {100*mean_a_f1_c:9.1f}% {'':14s} {100*mean_a_f1_l:7.1f}% | {100*mean_m_f1_c:9.1f}% {'':14s} {100*mean_m_f1_l:7.1f}%")
    print("=" * 95)

    # Rollouts comparison
    print("\n" + "="*95)
    print("ROLLOUT SIMULATION: WRM_wrap_monitor_backup VS PhaseTaskMonitor (18 Real Afternoon Rollouts)")
    print("="*95)
    out_a = eval_on_rollouts("backup", backup_model, device)
    out_b = eval_on_rollouts("phase", phase_model, device)

    print(f"{'Rollout Episode':32s} | {'Size':6s} | {'Model A (Backup) Stop':23s} | {'Model B (Phase) Stop':23s}")
    print("-" * 95)

    for i in range(len(out_a)):
        oa = out_a[i]
        ob = out_b[i]
        ep_str = oa["ep_name"]
        sz_str = oa["size"]

        sa_str = f"Step {oa['stop_step']} (J3={oa['j3_stop']:.2f})" if oa["stop_step"] else "None (OverGrasp)"
        if oa["is_under"]: sa_str = f"Step {oa['stop_step']} (UnderGrasp)"

        sb_str = f"Step {ob['stop_step']} (J3={ob['j3_stop']:.2f})" if ob["stop_step"] else "None (OverGrasp)"
        if ob["is_under"]: sb_str = f"Step {ob['stop_step']} (UnderGrasp)"

        print(f"{ep_str:32s} | {sz_str:6s} | {sa_str:23s} | {sb_str:23s}")

    under_a = sum(x["is_under"] for x in out_a)
    over_a = sum(x["is_over"] for x in out_a)
    lift_a = sum(x["lift_step"] is not None for x in out_a)

    under_b = sum(x["is_under"] for x in out_b)
    over_b = sum(x["is_over"] for x in out_b)
    lift_b = sum(x["lift_step"] is not None for x in out_b)

    print("-" * 95)
    print(f"Summary over 18 Rollouts:")
    print(f"  Under-Grasp (Premature mid-air stop): Model A = {under_a}/18 ({under_a/18*100:.1f}%) | Model B = {under_b}/18 ({under_b/18*100:.1f}%)")
    print(f"  Over-Grasp  (Uncontrolled squeeze)  : Model A = {over_a}/18 ({over_a/18*100:.1f}%) | Model B = {over_b}/18 ({over_b/18*100:.1f}%)")
    print(f"  Lift Triggered Rate                 : Model A = {lift_a}/18 ({lift_a/18*100:.1f}%) | Model B = {lift_b}/18 ({lift_b/18*100:.1f}%)")
    print("=" * 95)


if __name__ == "__main__":
    main()
