"""Evaluate a LeRobot policy on a LeRobot dataset and plot each episode.

Usage:
    python -m scripts.dataset.evaluate_policy \
        --policy IXDLI/dp_ppfine_h16_a8_v2 \
        --dataset IXDLI/ppfine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils
from inference import (
    _resolve_model_dir,
    load_policy_processor,
    load_pretrained_policy,
)


JOINT_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "gripper",
]


def resolve_training_dataset(policy_path: str) -> str | None:
    model_dir = _resolve_model_dir(policy_path)
    train_cfg_path = model_dir / "train_config.json"
    if not train_cfg_path.exists():
        return None

    with open(train_cfg_path, "r") as f:
        train_cfg = json.load(f)
    return train_cfg.get("dataset", {}).get("repo_id")


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def is_visual_feature(feature) -> bool:
    ft_type = feature.type if hasattr(feature, "type") else feature.get("type", "")
    return "VISUAL" in str(ft_type).upper() or "IMAGE" in str(ft_type).upper()


def feature_shape(feature) -> tuple[int, ...]:
    if hasattr(feature, "shape"):
        return tuple(feature.shape)
    return tuple(feature.get("shape", ()))


def image_to_tensor(image: np.ndarray, feature, device: str, add_batch: bool) -> torch.Tensor:
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in {1, 3}:
        chw = img
    elif img.ndim == 3:
        chw = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(f"Expected image with 3 dimensions, got shape {img.shape}")

    shape = feature_shape(feature)
    if len(shape) == 3 and tuple(chw.shape) != shape:
        from PIL import Image

        channels, height, width = shape
        hwc = np.transpose(chw, (1, 2, 0))
        hwc = np.asarray(Image.fromarray(hwc.astype(np.uint8)).resize((width, height)))
        chw = np.transpose(hwc, (2, 0, 1))
        if channels == 1 and chw.shape[0] != 1:
            chw = chw[:1]

    tensor = torch.from_numpy(chw.copy()).float()
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    if add_batch:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def value_to_tensor(value, device: str, add_batch: bool) -> torch.Tensor:
    tensor = value.float() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if add_batch:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def names_for_dims(n_dims: int) -> list[str]:
    if n_dims <= len(JOINT_NAMES):
        return JOINT_NAMES[:n_dims]
    return [f"joint_{i}" for i in range(n_dims)]


def print_action_stats(label: str, actions: np.ndarray) -> None:
    print(f"{label}: min={actions.min()}, max={actions.max()}, mean={actions.mean()}")
    for dim_idx, name in enumerate(names_for_dims(actions.shape[1])):
        values = actions[:, dim_idx]
        print(
            f"{label} {dim_idx} ({name}): "
            f"min={values.min()}, max={values.max()}, mean={values.mean()}"
        )


def print_action_error_metrics(gt_actions: np.ndarray, pred_actions: np.ndarray) -> None:
    n_dims = common_dim(gt_actions, pred_actions)
    gt = gt_actions[:, :n_dims]
    pred = pred_actions[:, :n_dims]
    err = pred - gt

    pos_mae = pos_rmse = pos_range = None
    if n_dims >= 3:
        pos_err = err[:, :3]
        pos_gt = gt[:, :3]
        pos_mae = np.mean(np.abs(pos_err), axis=0)
        pos_rmse = np.sqrt(np.mean(pos_err**2, axis=0))
        pos_range = pos_gt.max(axis=0) - pos_gt.min(axis=0)

    rot_mae = rot_rmse = rot_range = None
    if n_dims >= 6:
        rot_err = err[:, 3:6]
        rot_gt = gt[:, 3:6]
        rot_mae = np.mean(np.abs(rot_err), axis=0)
        rot_rmse = np.sqrt(np.mean(rot_err**2, axis=0))
        rot_range = rot_gt.max(axis=0) - rot_gt.min(axis=0)

    grip_mae = np.mean(np.abs(err[:, 6])) if n_dims >= 7 else None

    if pos_mae is not None:
        print("pos_mae:", pos_mae)
    if rot_mae is not None:
        print("rot_mae:", rot_mae)
    if grip_mae is not None:
        print("grip_mae:", grip_mae)
    if pos_rmse is not None:
        print("pos_rmse:", pos_rmse)
    if rot_rmse is not None:
        print("rot_rmse:", rot_rmse)
    if pos_range is not None:
        print("pos range:", pos_range)
    if rot_range is not None:
        print("rot range:", rot_range)
    if pos_mae is not None and pos_range is not None:
        print("pos relative mae:", pos_mae / (pos_range + 1e-8))
    if rot_mae is not None and rot_range is not None:
        print("rot relative mae:", rot_mae / (rot_range + 1e-8))


def processor_attr_names(policy) -> list[str]:
    tokens = ("preprocessor", "postprocessor", "processor", "normalizer", "unnormalizer")
    names = []
    for name in dir(policy):
        if name.startswith("_"):
            continue
        if not any(token in name.lower() for token in tokens):
            continue
        try:
            value = getattr(policy, name)
        except Exception:
            continue
        if value is not None:
            names.append(name)
    return names


def print_checkpoint_diagnostics(policy, model_dir: Path) -> None:
    def files(pattern: str) -> list[str]:
        return [str(path) for path in sorted(model_dir.glob(pattern))]

    attrs = processor_attr_names(policy)
    print("checkpoint model_dir:", model_dir)
    print("checkpoint has model weights:", bool(files("*.safetensors") + files("pytorch_model*.bin")))
    print("checkpoint has train_config:", bool(files("train_config.json")), files("train_config.json"))
    print("checkpoint has policy_preprocessor.json:", bool(files("policy_preprocessor.json")))
    print("checkpoint has policy_postprocessor.json:", bool(files("policy_postprocessor.json")))
    print("checkpoint has policy_preprocessor normalizer:", bool(files("policy_preprocessor*normalizer*")))
    print("checkpoint has policy_postprocessor unnormalizer:", bool(files("policy_postprocessor*unnormalizer*")))
    print("policy processor attrs:", attrs or "none detected")
    print("policy has preprocessor:", any("preprocessor" in name.lower() for name in attrs))
    print("policy has postprocessor:", any("postprocessor" in name.lower() for name in attrs))
    print("policy has unnormalizer:", any("unnormalizer" in name.lower() for name in attrs))


def apply_policy_timing_overrides(
    policy,
    horizon: int | None,
    action_horizon: int | None,
    n_obs_steps: int | None,
) -> None:
    overrides = {
        "horizon": horizon,
        "n_action_steps": action_horizon,
        "n_obs_steps": n_obs_steps,
    }
    for name, value in overrides.items():
        if value is None:
            continue
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        if not hasattr(policy.config, name):
            raise ValueError(f"Policy config does not support {name}")
        setattr(policy.config, name, value)

    if hasattr(policy, "diffusion") and hasattr(policy.diffusion, "config"):
        for name, value in overrides.items():
            if value is not None and hasattr(policy.diffusion.config, name):
                setattr(policy.diffusion.config, name, value)

    if hasattr(policy.config, "horizon") and hasattr(policy.config, "n_action_steps") and hasattr(policy.config, "n_obs_steps"):
        max_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
        if policy.config.n_action_steps > max_action_steps:
            raise ValueError(
                "action_horizon must be <= horizon - n_obs_steps + 1 "
                f"({max_action_steps}), got {policy.config.n_action_steps}"
            )

    policy.reset()
    utils.logger.info(
        "Policy timing: "
        f"horizon={getattr(policy.config, 'horizon', 'n/a')}, "
        f"action_horizon={getattr(policy.config, 'n_action_steps', 'n/a')}, "
        f"n_obs_steps={getattr(policy.config, 'n_obs_steps', 'n/a')}"
    )


def build_feature_map(policy_config, dataset_features: set[str]) -> dict[str, str]:
    feature_map = {}
    for policy_key in policy_config.input_features:
        if policy_key in dataset_features:
            feature_map[policy_key] = policy_key
        elif policy_key == "observation.image":
            image_keys = sorted(k for k in dataset_features if k.startswith("observation.images."))
            feature_map[policy_key] = image_keys[0] if image_keys else policy_key
        elif policy_key.startswith("observation.images.") and "observation.image" in dataset_features:
            feature_map[policy_key] = "observation.image"
        else:
            feature_map[policy_key] = policy_key
    return feature_map


class PyAvVideoReader:
    def __init__(self, dataset_root, dataset_meta, video_keys):
        self.root = Path(dataset_root)
        self.meta = dataset_meta
        self.video_keys = set(video_keys)
        self._cache = {}

    def _video_path(self, video_key: str, episode_idx: int) -> Path:
        if hasattr(self.meta, "get_video_file_path"):
            return self.root / self.meta.get_video_file_path(episode_idx, video_key)

        chunks_size = getattr(self.meta, "chunks_size", 1000)
        chunk_idx = episode_idx // chunks_size
        template = getattr(
            self.meta,
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        return self.root / template.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=episode_idx,
        )

    def _episode_start_frame(self, video_key: str, episode_idx: int) -> int:
        episodes = getattr(self.meta, "episodes", None)
        if episodes is None or episode_idx >= len(episodes):
            return 0

        episode = episodes[episode_idx]
        timestamp_key = f"videos/{video_key}/from_timestamp"
        if timestamp_key not in episode:
            return 0

        features = getattr(self.meta, "features", {})
        feature = features.get(video_key, {}) if hasattr(features, "get") else {}
        info = feature.get("info", {}) if hasattr(feature, "get") else {}
        fps = info.get("video.fps", getattr(self.meta, "fps", 0))
        return int(round(float(episode[timestamp_key]) * float(fps))) if fps else 0

    def _frames(self, video_key: str, episode_idx: int) -> list[np.ndarray]:
        import av

        video_path = self._video_path(video_key, episode_idx)
        cache_key = (video_key, video_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        container = av.open(str(video_path))
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        container.close()
        self._cache[cache_key] = frames
        return frames

    def get_frame(self, video_key: str, episode_idx: int, frame_idx: int) -> np.ndarray:
        frames = self._frames(video_key, episode_idx)
        video_frame_idx = self._episode_start_frame(video_key, episode_idx) + frame_idx
        return frames[min(video_frame_idx, len(frames) - 1)]

    def close(self) -> None:
        self._cache.clear()


class LeRobotEpisodeSource:
    def __init__(self, dataset_ref: str):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        local_path = Path(dataset_ref).expanduser().resolve()
        if local_path.exists():
            try:
                self.dataset = LeRobotDataset(repo_id=local_path.name, root=local_path.parent)
            except Exception:
                self.dataset = LeRobotDataset(repo_id=local_path.name, root=local_path)
        else:
            utils.logger.info(f"Local dataset not found, trying Hugging Face: {dataset_ref}")
            self.dataset = LeRobotDataset(repo_id=dataset_ref)

        self.num_episodes = self.dataset.num_episodes
        self.features = set(self.dataset.meta.features.keys())
        self.video_reader = None
        if self.dataset.meta.video_keys:
            self.video_reader = PyAvVideoReader(
                dataset_root=self.dataset.root,
                dataset_meta=self.dataset.meta,
                video_keys=self.dataset.meta.video_keys,
            )

    def episode_ranges(self) -> dict[int, tuple[int, int]]:
        if hasattr(self.dataset, "episode_data_index"):
            return {
                ep: (
                    self.dataset.episode_data_index["from"][ep].item(),
                    self.dataset.episode_data_index["to"][ep].item(),
                )
                for ep in range(self.dataset.num_episodes)
            }

        ep_col = np.asarray([to_numpy(v).item() for v in self.dataset.hf_dataset["episode_index"]])
        ranges = {}
        for ep in range(self.dataset.num_episodes):
            indices = np.where(ep_col == ep)[0]
            ranges[ep] = (int(indices[0]), int(indices[-1]) + 1)
        return ranges

    def iter_episode(self, episode_idx: int):
        from_idx, to_idx = self.episode_ranges()[episode_idx]
        for global_idx in range(from_idx, to_idx):
            yield self.dataset.hf_dataset[global_idx], global_idx - from_idx

    def close(self) -> None:
        if self.video_reader is not None:
            self.video_reader.close()

    def reset_for_new_eval_loop(self) -> None:
        if self.video_reader is not None:
            self.video_reader.close()


def build_observation(
    item,
    frame_idx: int,
    episode_idx: int,
    source: LeRobotEpisodeSource,
    feature_map: dict[str, str],
    policy_config,
    device: str,
    add_batch: bool,
) -> dict[str, torch.Tensor]:
    obs = {}
    for policy_key, feature in policy_config.input_features.items():
        dataset_key = feature_map.get(policy_key, policy_key)

        if (
            source.video_reader is not None
            and is_visual_feature(feature)
            and dataset_key in source.video_reader.video_keys
        ):
            image = source.video_reader.get_frame(dataset_key, episode_idx, frame_idx)
            obs[policy_key] = image_to_tensor(image, feature, device, add_batch)
        elif dataset_key in item and is_visual_feature(feature):
            obs[policy_key] = image_to_tensor(to_numpy(item[dataset_key]), feature, device, add_batch)
        elif dataset_key in item:
            obs[policy_key] = value_to_tensor(item[dataset_key], device, add_batch)

    return obs


def evaluate_episode(
    policy,
    source: LeRobotEpisodeSource,
    episode_idx: int,
    feature_map: dict[str, str],
    policy_config,
    device: str,
    preprocessor=None,
    postprocessor=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    policy.reset()
    states, gt_actions, pred_actions = [], [], []
    action_path = "select_action"

    for item, frame_idx in source.iter_episode(episode_idx):
        obs = build_observation(
            item=item,
            frame_idx=frame_idx,
            episode_idx=episode_idx,
            source=source,
            feature_map=feature_map,
            policy_config=policy_config,
            device=device,
            add_batch=preprocessor is None,
        )
        if preprocessor is not None:
            obs = preprocessor(obs)
            action_path = "policy_preprocessor -> select_action"

        with torch.no_grad():
            action = policy.select_action(obs)
            if postprocessor is not None:
                action = postprocessor(action)
                action_path += " -> policy_postprocessor"

        states.append(to_numpy(item["observation.state"]).astype(np.float32))
        gt_actions.append(to_numpy(item["action"]).astype(np.float32))
        pred_actions.append(to_numpy(action.squeeze(0)).astype(np.float32))

    return np.asarray(states), np.asarray(gt_actions), np.asarray(pred_actions), action_path


def common_dim(*arrays: np.ndarray) -> int:
    return min(arr.shape[1] for arr in arrays if arr.ndim == 2)


def plot_episode(states, gt_actions, pred_actions, episode_idx: int, save_dir: Path | None):
    import matplotlib.pyplot as plt

    n_dims = common_dim(states, gt_actions, pred_actions)
    names = names_for_dims(n_dims)
    steps = np.arange(states.shape[0])
    fig, axes = plt.subplots(n_dims, 1, figsize=(14, max(3.0, 2.25 * n_dims)), sharex=True)
    axes = [axes] if n_dims == 1 else axes

    fig.suptitle(f"Episode {episode_idx}: original action, policy action")
    for dim_idx, ax in enumerate(axes):
        ax.plot(steps, gt_actions[:, dim_idx], label="original action", linewidth=1.4)
        ax.plot(steps, pred_actions[:, dim_idx], label="policy action", linewidth=1.4)
        ax.set_ylabel(names[dim_idx])
        ax.grid(True, alpha=0.25)
        if dim_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("timestep")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_dir is not None:
        fig.savefig(save_dir / f"episode_{episode_idx:06d}_policy_vs_dataset.png", dpi=150)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot policy on a LeRobot dataset.")
    parser.add_argument("--policy", required=True, help="Local policy path or Hugging Face policy repo id")
    parser.add_argument("--dataset", default=None, help="Local or Hugging Face LeRobot dataset")
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    parser.add_argument("--episodes", nargs="+", type=int, default=None)
    parser.add_argument("--save-dir", default="./eval_dataset_plots")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--horizon", type=int, default=None, help="Override policy prediction horizon")
    parser.add_argument("--action_horizon", type=int, default=None, help="Override policy action horizon")
    parser.add_argument("--n_obs_steps", type=int, default=None, help="Override number of observation steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    dataset_ref = args.dataset or resolve_training_dataset(args.policy)
    if dataset_ref is None:
        raise ValueError("No --dataset given and no dataset repo_id found in policy train_config.json")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    utils.logger.info("=== LeRobot Dataset Policy Eval ===")
    utils.logger.info(f"Policy : {args.policy}")
    utils.logger.info(f"Dataset: {dataset_ref}")
    utils.logger.info(f"Device : {device}")

    source = LeRobotEpisodeSource(dataset_ref)
    policy = load_pretrained_policy(args.policy, device)
    apply_policy_timing_overrides(policy, args.horizon, args.action_horizon, args.n_obs_steps)
    model_dir = _resolve_model_dir(args.policy)
    preprocessor = load_policy_processor(args.policy, "policy_preprocessor.json", device=device)
    postprocessor = load_policy_processor(args.policy, "policy_postprocessor.json")

    print_checkpoint_diagnostics(policy, model_dir)
    print("loaded policy_preprocessor:", preprocessor is not None)
    print("loaded policy_postprocessor:", postprocessor is not None)

    feature_map = build_feature_map(policy.config, source.features)
    utils.logger.info(f"Feature map: {feature_map}")

    episode_indices = args.episodes if args.episodes is not None else list(range(source.num_episodes))
    save_dir = None if args.no_save else Path(args.save_dir)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for episode_idx in episode_indices:
        if episode_idx < 0 or episode_idx >= source.num_episodes:
            utils.logger.warning(f"Episode {episode_idx} out of range, skipping.")
            continue

        utils.logger.info(f"Evaluating episode {episode_idx} ...")
        states, gt_actions, pred_actions, action_path = evaluate_episode(
            policy=policy,
            source=source,
            episode_idx=episode_idx,
            feature_map=feature_map,
            policy_config=policy.config,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        utils.logger.info(
            f"Episode {episode_idx}: states={states.shape}, "
            f"gt_actions={gt_actions.shape}, pred_actions={pred_actions.shape}"
        )
        print("action scale path:", action_path)
        print_action_stats("raw gt action", gt_actions)
        print_action_stats("raw pred action", pred_actions)
        print_action_error_metrics(gt_actions, pred_actions)

        plot_episode(states, gt_actions, pred_actions, episode_idx, save_dir)
        if args.no_show:
            plt.close("all")
        else:
            plt.show()

        if args.episodes is None and episode_idx != episode_indices[-1]:
            answer = input("Press Enter to test next episode, or type q then Enter to stop: ")
            if answer.strip().lower() in {"q", "quit", "stop", "exit"}:
                break

    source.close()
    if save_dir is not None:
        utils.logger.info(f"Plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
