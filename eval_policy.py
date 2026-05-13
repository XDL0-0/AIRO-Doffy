"""Offline policy evaluation against a lerobot dataset.

Loads a trained policy and a recorded dataset, runs the policy on dataset
observations (open-loop), compares predicted actions to ground truth, and
produces per-joint trajectory plots, error analysis, and a quantitative score.

Usage:
    # Evaluate using the policy's training dataset (auto-resolved from train_config.json)
    python eval_policy.py --policy IXDLI/WIPE

    # Evaluate on a specific local or HF dataset
    python eval_policy.py \\
        --policy IXDLI/WIPE \\
        --dataset ./datasets/WipeBoard_lero

    # Evaluate specific episodes
    python eval_policy.py \\
        --policy ./checkpoints/my_policy \\
        --dataset username/my_dataset \\
        --episodes 0 1 2 --no-show

    # Skip plots entirely
    python eval_policy.py \\
        --policy username/my_policy \\
        --dataset ./datasets/task_lero --no-plot
"""

import time
import argparse
import json
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import utils
from inference import load_pretrained_policy, _resolve_model_dir


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_path):
    """Load a lerobot dataset from a local directory or HuggingFace repo."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    local_path = Path(dataset_path).resolve()
    if local_path.exists():
        repo_id = local_path.name
        dataset = LeRobotDataset(repo_id=repo_id, root=local_path)
    else:
        utils.logger.info(f"Local path not found, trying HuggingFace: {dataset_path}")
        dataset = LeRobotDataset(repo_id=dataset_path)

    return dataset


def resolve_training_dataset(policy_path):
    """Extract the training dataset repo_id from the policy's train_config.json."""
    model_dir = _resolve_model_dir(policy_path)
    train_cfg_path = model_dir / "train_config.json"

    if not train_cfg_path.exists():
        return None

    with open(train_cfg_path) as f:
        train_cfg = json.load(f)

    ds_cfg = train_cfg.get("dataset", {})
    return ds_cfg.get("repo_id")


# ---------------------------------------------------------------------------
# Video frame reader (cv2 fallback for torchcodec issues)
# ---------------------------------------------------------------------------

class VideoFrameReader:
    """Read individual frames from mp4 files using pyav (supports AV1 / h264 / etc.)."""

    def __init__(self, dataset_root, video_path_template, video_keys, chunks_size=1000):
        import av as _av  # noqa: F401 – verify pyav is installed
        self.root = Path(dataset_root)
        self.template = video_path_template
        self.video_keys = video_keys
        self.chunks_size = chunks_size
        self._frame_cache = {}

    def _video_path(self, video_key, episode_idx):
        chunk_idx = episode_idx // self.chunks_size
        file_idx = episode_idx
        return self.root / self.template.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )

    def _load_all_frames(self, video_key, episode_idx):
        """Decode all frames of an episode video once and cache them."""
        import av

        cache_key = (video_key, episode_idx)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        path = self._video_path(video_key, episode_idx)
        container = av.open(str(path))
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()

        self._frame_cache[cache_key] = frames
        return frames

    def get_frame(self, video_key, episode_idx, frame_idx):
        """Read a single RGB frame (H, W, 3) uint8."""
        frames = self._load_all_frames(video_key, episode_idx)
        if frame_idx >= len(frames):
            raise RuntimeError(
                f"Frame {frame_idx} out of range (video has {len(frames)} frames) "
                f"for {video_key} episode {episode_idx}"
            )
        return frames[frame_idx]

    def close(self):
        self._frame_cache.clear()


# ---------------------------------------------------------------------------
# Feature mapping
# ---------------------------------------------------------------------------

def build_feature_map(policy_config, dataset_features):
    """Build a mapping from policy input feature names to dataset feature names.

    Handles common renaming patterns, e.g.:
      policy expects "observation.image" but dataset has "observation.images.camera_0"
    """
    fmap = {}
    policy_features = policy_config.input_features

    for pkey in policy_features:
        if pkey in dataset_features:
            fmap[pkey] = pkey
            continue

        # observation.image -> observation.images.camera_0
        if "image" in pkey and "images" not in pkey:
            candidates = [
                k for k in dataset_features
                if k.startswith("observation.images.")
            ]
            if candidates:
                fmap[pkey] = sorted(candidates)[0]
                utils.logger.info(f"Feature map: '{pkey}' -> '{fmap[pkey]}'")
                continue

        # observation.images.camera_X -> observation.image (reverse)
        if "images" in pkey:
            if "observation.image" in dataset_features:
                fmap[pkey] = "observation.image"
                utils.logger.info(f"Feature map: '{pkey}' -> '{fmap[pkey]}'")
                continue

        fmap[pkey] = pkey

    return fmap


def check_compatibility(policy_config, dataset):
    """Verify that dataset features are dimensionally compatible with the policy.

    Raises ValueError with a helpful message when there is a mismatch.
    """
    ds_features = dataset.meta.features
    issues = []

    for pkey, pft in policy_config.input_features.items():
        pshape = tuple(pft.shape) if hasattr(pft, "shape") else tuple(pft.get("shape", []))
        ft_type = str(pft.type) if hasattr(pft, "type") else str(pft.get("type", ""))

        if "VISUAL" in ft_type:
            continue

        if pkey in ds_features:
            ds_shape = tuple(ds_features[pkey].get("shape", ds_features[pkey].get("shape", [])))
            if pshape and ds_shape and pshape != ds_shape:
                issues.append(
                    f"  {pkey}: policy expects {pshape}, dataset has {ds_shape}"
                )

    for pkey, pft in policy_config.output_features.items():
        pshape = tuple(pft.shape) if hasattr(pft, "shape") else tuple(pft.get("shape", []))
        if pkey in ds_features:
            ds_shape = tuple(ds_features[pkey].get("shape", ds_features[pkey].get("shape", [])))
            if pshape and ds_shape and pshape != ds_shape:
                issues.append(
                    f"  {pkey}: policy expects {pshape}, dataset has {ds_shape}"
                )

    if issues:
        msg = (
            "Feature dimension mismatch between policy and dataset:\n"
            + "\n".join(issues)
            + "\n\nThis dataset is not compatible with this policy."
            + "\nTry omitting --dataset to auto-use the training dataset, or"
            + "\nspecify the correct dataset with --dataset <repo_id>."
        )
        raise ValueError(msg)

    utils.logger.info("Feature compatibility check passed.")


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

def _get_episode_ranges(dataset):
    """Return dict {episode_idx: (from_global, to_global)} for all episodes."""
    if hasattr(dataset, "episode_data_index"):
        ranges = {}
        for ep in range(dataset.num_episodes):
            f = dataset.episode_data_index["from"][ep].item()
            t = dataset.episode_data_index["to"][ep].item()
            ranges[ep] = (f, t)
        return ranges

    ep_col = np.array([
        t.item() if isinstance(t, torch.Tensor) else t
        for t in dataset.hf_dataset["episode_index"]
    ])
    ranges = {}
    for ep in range(dataset.num_episodes):
        indices = np.where(ep_col == ep)[0]
        ranges[ep] = (int(indices[0]), int(indices[-1]) + 1)
    return ranges


def evaluate_episode(policy, dataset, episode_idx, device, video_reader,
                     feature_map, policy_config):
    """Run policy open-loop on a single episode.

    For each timestep the *ground-truth* observation from the dataset is fed
    into the policy; the returned action is compared to the ground-truth
    action recorded during data collection.
    """
    ep_ranges = _get_episode_ranges(dataset)
    from_idx, to_idx = ep_ranges[episode_idx]

    policy.reset()
    gt_actions, pred_actions, states = [], [], []

    hf = dataset.hf_dataset
    input_features = policy_config.input_features

    for global_idx in range(from_idx, to_idx):
        item = hf[global_idx]
        frame_idx_in_ep = global_idx - from_idx

        obs = {}
        for policy_key, ft in input_features.items():
            ds_key = feature_map.get(policy_key, policy_key)
            ft_type = ft.type if hasattr(ft, "type") else str(ft.get("type", ""))

            is_visual = ("VISUAL" in str(ft_type))

            if is_visual and video_reader is not None and ds_key in video_reader.video_keys:
                frame = video_reader.get_frame(ds_key, episode_idx, frame_idx_in_ep)
                target_shape = ft.shape if hasattr(ft, "shape") else ft.get("shape")
                if target_shape is not None and len(target_shape) == 3:
                    _, th, tw = target_shape
                    from PIL import Image
                    frame = np.array(Image.fromarray(frame).resize((tw, th)))
                # (H, W, 3) -> (1, 3, H, W) float [0, 1]
                t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs[policy_key] = t.to(device)

            elif ds_key in item:
                value = item[ds_key]
                if isinstance(value, torch.Tensor):
                    obs[policy_key] = value.unsqueeze(0).to(device)
                elif isinstance(value, np.ndarray):
                    obs[policy_key] = (
                        torch.from_numpy(value.copy()).float().unsqueeze(0).to(device)
                    )

        action_val = item["action"]
        state_val = item["observation.state"]
        gt_actions.append(
            action_val.numpy() if isinstance(action_val, torch.Tensor)
            else np.asarray(action_val, dtype=np.float32)
        )
        states.append(
            state_val.numpy() if isinstance(state_val, torch.Tensor)
            else np.asarray(state_val, dtype=np.float32)
        )

        with torch.no_grad():
            pred = policy.select_action(obs)
        pred_actions.append(pred.squeeze(0).cpu().numpy())

    return np.array(gt_actions), np.array(pred_actions), np.array(states)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt, pred):
    """Compute evaluation metrics."""
    diff = gt - pred

    mse_j = np.mean(diff ** 2, axis=0)
    mae_j = np.mean(np.abs(diff), axis=0)
    rmse_j = np.sqrt(mse_j)

    ss_res = np.sum(diff ** 2, axis=0)
    ss_tot = np.sum((gt - gt.mean(axis=0)) ** 2, axis=0) + 1e-12
    r2_j = 1.0 - ss_res / ss_tot

    dot = np.sum(gt * pred, axis=0)
    cos_j = dot / (np.linalg.norm(gt, axis=0) * np.linalg.norm(pred, axis=0) + 1e-12)

    action_range = gt.max(axis=0) - gt.min(axis=0) + 1e-12
    nrmse_j = rmse_j / action_range
    score_j = np.clip(100.0 * (1.0 - nrmse_j), 0, 100)

    return {
        "mse_j": mse_j, "mae_j": mae_j, "rmse_j": rmse_j,
        "r2_j": r2_j, "cos_j": cos_j, "score_j": score_j,
        "mse": float(mse_j.mean()), "mae": float(mae_j.mean()),
        "rmse": float(rmse_j.mean()), "r2": float(r2_j.mean()),
        "cos": float(cos_j.mean()), "score": float(score_j.mean()),
    }


def _dim_names(n_dims):
    names = [f"Dim_{i}" for i in range(n_dims)]
    return names


def print_metrics(metrics, episode_idx, n_dims):
    names = _dim_names(n_dims)
    lines = [
        "",
        "=" * 78,
        f"  Episode {episode_idx}",
        "=" * 78,
        f"  {'':10s} {'MSE':>10s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s} {'Score':>8s}",
        f"  {'-' * 68}",
    ]

    for i, name in enumerate(names):
        lines.append(
            f"  {name:10s}"
            f" {metrics['mse_j'][i]:10.6f}"
            f" {metrics['mae_j'][i]:10.6f}"
            f" {metrics['rmse_j'][i]:10.6f}"
            f" {metrics['r2_j'][i]:10.4f}"
            f" {metrics['score_j'][i]:8.1f}"
        )

    lines.extend(
        [
            f"  {'-' * 68}",
            (
        f"  {'Overall':10s}"
        f" {metrics['mse']:10.6f}"
        f" {metrics['mae']:10.6f}"
        f" {metrics['rmse']:10.6f}"
        f" {metrics['r2']:10.4f}"
        f" {metrics['score']:8.1f}"
            ),
            "=" * 78,
            "",
        ]
    )
    utils.logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_GT_CLR = "#2196F3"
_PRED_CLR = "#FF5722"


def plot_trajectories(gt, pred, metrics, episode_idx, save_dir=None):
    n_steps, n_dims = gt.shape
    t = np.arange(n_steps)
    names = _dim_names(n_dims)

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 2.6 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    fig.suptitle(
        f"Episode {episode_idx} — Action Trajectories\n"
        f"Score: {metrics['score']:.1f}/100   "
        f"RMSE: {metrics['rmse']:.4f}   "
        f"R²: {metrics['r2']:.3f}",
        fontsize=13, fontweight="bold",
    )

    for i, ax in enumerate(axes):
        ax.plot(t, gt[:, i], label="Ground Truth", color=_GT_CLR, lw=1.6)
        ax.plot(t, pred[:, i], label="Predicted", color=_PRED_CLR, lw=1.6, alpha=0.85)
        ax.fill_between(t, gt[:, i], pred[:, i], color=_PRED_CLR, alpha=0.12)

        name = names[i] if i < len(names) else f"Dim {i}"
        ax.set_ylabel(name, fontsize=10)
        ax.text(
            0.99, 0.92,
            f"Score {metrics['score_j'][i]:.1f}  R² {metrics['r2_j'][i]:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.6),
        )
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="upper left", fontsize=9)

    axes[-1].set_xlabel("Timestep", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        fig.savefig(Path(save_dir) / f"ep{episode_idx}_trajectories.png", dpi=150)
    return fig


def plot_error(gt, pred, metrics, episode_idx, save_dir=None):
    n_steps, n_dims = gt.shape
    t = np.arange(n_steps)
    errors = np.abs(gt - pred)
    names = _dim_names(n_dims)
    cmap = plt.cm.tab10

    fig, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"Episode {episode_idx} — Error Analysis   "
        f"(MAE: {metrics['mae']:.4f})",
        fontsize=13, fontweight="bold",
    )

    for i in range(n_dims):
        c = cmap(i / max(n_dims - 1, 1))
        label = names[i] if i < len(names) else f"Dim {i}"
        ax_t.plot(t, errors[:, i], label=label, lw=1.0, alpha=0.85, color=c)
        ax_h.hist(errors[:, i], bins=40, alpha=0.45, label=label, color=c)

    ax_t.set_xlabel("Timestep")
    ax_t.set_ylabel("Absolute Error")
    ax_t.set_title("Error over Time")
    ax_t.legend(fontsize=8, ncol=2)
    ax_t.grid(True, alpha=0.25)

    ax_h.set_xlabel("Absolute Error")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Error Distribution")
    ax_h.legend(fontsize=8, ncol=2)
    ax_h.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_dir:
        fig.savefig(Path(save_dir) / f"ep{episode_idx}_error.png", dpi=150)
    return fig


def plot_summary(all_metrics, episode_indices, save_dir=None):
    if len(all_metrics) < 2:
        return None

    scores = [m["score"] for m in all_metrics]
    rmses = [m["rmse"] for m in all_metrics]
    r2s = [m["r2"] for m in all_metrics]
    x = np.arange(len(all_metrics))
    labels = [str(ep) for ep in episode_indices[:len(all_metrics)]]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Summary Across Episodes", fontsize=13, fontweight="bold")

    for ax, vals, ylabel, title, clr in [
        (a1, scores, "Score (0–100)", "Score", "#4CAF50"),
        (a2, rmses, "RMSE", "RMSE", "#FF9800"),
        (a3, r2s, "R²", "R²", "#2196F3"),
    ]:
        ax.bar(x, vals, color=clr, alpha=0.8, tick_label=labels)
        mean_v = np.mean(vals)
        ax.axhline(mean_v, color="red", ls="--", lw=1, label=f"Mean: {mean_v:.3f}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_dir:
        fig.savefig(Path(save_dir) / "summary.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluation of a lerobot policy against a dataset",
    )
    parser.add_argument(
        "--policy", required=True,
        help="HuggingFace repo_id or local checkpoint path",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="HuggingFace repo_id or local lerobot dataset path. "
             "If omitted, the training dataset is resolved from train_config.json.",
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    parser.add_argument(
        "--episodes", type=int, nargs="*", default=None,
        help="Episode indices to evaluate (default: all)",
    )
    parser.add_argument(
        "--save-dir", default="./eval_results",
        help="Directory for saved plots (default: ./eval_results)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip all plots")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save plot files but do not open a GUI window",
    )
    args = parser.parse_args()

    if not args.no_plot and not args.no_show:
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            utils.logger.warning("TkAgg not available, plots will be saved only.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- resolve dataset -----------------------------------------------------
    dataset_id = args.dataset
    if dataset_id is None:
        utils.logger.info("No --dataset given, resolving from train_config.json ...")
        dataset_id = resolve_training_dataset(args.policy)
        if dataset_id is None:
            parser.error(
                "Could not resolve training dataset from policy checkpoint. "
                "Please specify --dataset explicitly."
            )
        utils.logger.info(f"Resolved training dataset: {dataset_id}")

    utils.logger.info("=== Offline Policy Evaluation ===")
    utils.logger.info(f"Policy:  {args.policy}")
    utils.logger.info(f"Dataset: {dataset_id}")
    utils.logger.info(f"Device:  {device}")

    # ---- load ----------------------------------------------------------------
    utils.logger.info("Loading dataset...")
    dataset = load_dataset(dataset_id)
    utils.logger.info(
        f"Dataset loaded: {dataset.num_episodes} episodes, {len(dataset)} frames"
    )

    utils.logger.info("Loading policy...")
    policy = load_pretrained_policy(args.policy, device)
    policy_config = policy.config

    # ---- compatibility check & feature mapping ---------------------------------
    check_compatibility(policy_config, dataset)

    ds_features = set(dataset.meta.features.keys())
    feature_map = build_feature_map(policy_config, ds_features)
    utils.logger.info(f"Feature map: {feature_map}")

    # ---- video reader --------------------------------------------------------
    video_reader = None
    video_keys = dataset.meta.video_keys
    if video_keys:
        video_reader = VideoFrameReader(
            dataset_root=dataset.root,
            video_path_template=dataset.meta.video_path,
            video_keys=video_keys,
            chunks_size=dataset.meta.chunks_size,
        )
        utils.logger.info(f"Video reader ready: {video_keys}")

    # ---- episodes to evaluate ------------------------------------------------
    episode_indices = (
        args.episodes if args.episodes is not None
        else list(range(dataset.num_episodes))
    )
    utils.logger.info(f"Episodes to evaluate: {episode_indices}")

    save_dir = None
    if not args.no_plot:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # ---- evaluate per episode ------------------------------------------------
    all_metrics = []

    for ep in episode_indices:
        if ep >= dataset.num_episodes:
            utils.logger.warning(f"Episode {ep} out of range — skipping.")
            continue

        utils.logger.info(f"Evaluating episode {ep} ...")
        t0 = time.time()

        gt, pred, states = evaluate_episode(
            policy, dataset, ep, device,
            video_reader, feature_map, policy_config,
        )
        m = compute_metrics(gt, pred)
        all_metrics.append(m)

        elapsed = time.time() - t0
        utils.logger.info(
            f"Episode {ep}: Score={m['score']:.1f}  "
            f"RMSE={m['rmse']:.4f}  R²={m['r2']:.3f}  "
            f"({elapsed:.1f}s)"
        )
        print_metrics(m, ep, gt.shape[1])

        if not args.no_plot:
            plot_trajectories(gt, pred, m, ep, save_dir)
            plot_error(gt, pred, m, ep, save_dir)

    if video_reader:
        video_reader.close()

    # ---- overall summary -----------------------------------------------------
    if not all_metrics:
        utils.logger.warning("No episodes evaluated.")
        return

    if len(all_metrics) > 1:
        avg_s = np.mean([m["score"] for m in all_metrics])
        avg_r = np.mean([m["rmse"] for m in all_metrics])
        avg_r2 = np.mean([m["r2"] for m in all_metrics])

        utils.logger.info(
            "\n".join(
                [
                    "",
                    "#" * 78,
                    f"  OVERALL  ({len(all_metrics)} episodes)",
                    "#" * 78,
                    f"  Avg Score : {avg_s:.1f} / 100",
                    f"  Avg RMSE  : {avg_r:.6f}",
                    f"  Avg R²    : {avg_r2:.4f}",
                    "#" * 78,
                    "",
                ]
            )
        )

        if not args.no_plot:
            plot_summary(all_metrics, episode_indices, save_dir)
    else:
        utils.logger.info(f"Final Score: {all_metrics[0]['score']:.1f} / 100")

    if not args.no_plot:
        utils.logger.info(f"Plots saved to {save_dir}/")
        if not args.no_show:
            plt.show()

    utils.logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
