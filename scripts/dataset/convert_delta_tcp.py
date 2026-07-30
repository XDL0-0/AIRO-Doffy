#!/usr/bin/env python
"""Convert a LeRobot dataset so action is a delta TCP pose.

The source dataset must contain ``extra.tcp_pose`` with this convention:
``[qx, qy, qz, qw, x, y, z]``.

The output action is:
``[dx, dy, dz, drotvec_x, drotvec_y, drotvec_z, gripper]``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


TCP_POSE_KEY = "extra.tcp_pose"
ACTION_KEY = "action"
STATE_KEY = "observation.state"
ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rotvec_x",
    "delta_rotvec_y",
    "delta_rotvec_z",
    "gripper",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_to_rotvec_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(quat)
    if quat[3] < 0:
        quat = -quat

    xyz = quat[:3]
    w = np.clip(quat[3], -1.0, 1.0)
    sin_half_angle = np.linalg.norm(xyz)
    if sin_half_angle < 1e-12:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * np.arctan2(sin_half_angle, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return xyz / sin_half_angle * angle


def delta_tcp_action(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    gripper: float,
) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)

    current_quat = normalize_quat_xyzw(current_pose[:4])
    target_quat = normalize_quat_xyzw(target_pose[:4])
    delta_quat = quat_multiply_xyzw(target_quat, quat_conjugate_xyzw(current_quat))
    delta_rotvec = quat_to_rotvec_xyzw(delta_quat)
    delta_xyz = target_pose[4:7] - current_pose[4:7]

    return np.asarray([*delta_xyz, *delta_rotvec, gripper], dtype=np.float32)


def default_output_repo_id(source_repo_id: str) -> str:
    if "/" in source_repo_id:
        owner, name = source_repo_id.rsplit("/", 1)
        return f"{owner}/{name}_deltaTCP"
    return f"{Path(source_repo_id).name}_deltaTCP"


def user_features(features: dict[str, dict]) -> dict[str, dict]:
    return {
        key: copy.deepcopy(feature)
        for key, feature in features.items()
        if key not in DEFAULT_FEATURES
    }


def build_output_features(source: LeRobotDataset) -> dict[str, dict]:
    features = user_features(source.features)
    if TCP_POSE_KEY not in features:
        raise KeyError(
            f"Source dataset is missing '{TCP_POSE_KEY}'. "
            "Record with DATA_TYPE='both' so TCP poses are saved in extra."
        )

    features[ACTION_KEY] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ACTION_NAMES,
    }
    return features


def get_episode_ranges(dataset: LeRobotDataset) -> list[tuple[int, int]]:
    episodes = dataset.meta.episodes
    if episodes is None:
        return [(0, len(dataset))]

    if hasattr(episodes, "iterrows"):
        ranges: list[tuple[int, int]] = []
        for _, row in episodes.iterrows():
            if "dataset_from_index" in row and "dataset_to_index" in row:
                ranges.append((int(row["dataset_from_index"]), int(row["dataset_to_index"])))
            elif "from_index" in row and "to_index" in row:
                ranges.append((int(row["from_index"]), int(row["to_index"])))
            elif "length" in row:
                start = ranges[-1][1] if ranges else 0
                ranges.append((start, start + int(row["length"])))
            else:
                raise KeyError(f"Cannot infer episode range from columns: {list(row.index)}")
        return ranges

    ranges = []
    start = 0
    for episode in episodes:
        if "dataset_from_index" in episode and "dataset_to_index" in episode:
            ranges.append((int(episode["dataset_from_index"]), int(episode["dataset_to_index"])))
        elif "from_index" in episode and "to_index" in episode:
            ranges.append((int(episode["from_index"]), int(episode["to_index"])))
        elif "length" in episode:
            end = start + int(episode["length"])
            ranges.append((start, end))
            start = end
        else:
            raise KeyError(f"Cannot infer episode range from episode metadata: {episode}")
    return ranges


def raw_feature_array(dataset: LeRobotDataset, key: str, start: int, end: int) -> np.ndarray:
    raw_dataset = dataset.hf_dataset.select_columns([key])
    values = [as_numpy(raw_dataset[idx][key]) for idx in range(start, end)]
    return np.asarray(values)


def episode_task(dataset: LeRobotDataset, episode_index: int, fallback_item: dict) -> str:
    episodes = dataset.meta.episodes
    if episodes is not None:
        episode = episodes.iloc[episode_index] if hasattr(episodes, "iloc") else episodes[episode_index]
        tasks = episode.get("tasks") if hasattr(episode, "get") else None
        if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks) > 0:
            return str(tasks[0])
        if isinstance(tasks, str):
            return tasks
    return str(fallback_item.get("task", "delta tcp dataset"))


def gripper_values(
    dataset: LeRobotDataset,
    source: str,
    start: int,
    end: int,
) -> np.ndarray:
    if source == "zero":
        return np.zeros(end - start, dtype=np.float32)

    key = ACTION_KEY if source == "action" else STATE_KEY
    if key not in dataset.features:
        if source == "action" and STATE_KEY in dataset.features:
            logger.warning("Source has no action feature; using observation.state for gripper.")
            key = STATE_KEY
        else:
            logger.warning("No %s feature found for gripper; using zeros.", key)
            return np.zeros(end - start, dtype=np.float32)

    values = raw_feature_array(dataset, key, start, end)
    if values.ndim == 1:
        return values.astype(np.float32)
    return values[:, -1].astype(np.float32)


def compute_episode_actions(
    dataset: LeRobotDataset,
    start: int,
    end: int,
    lookahead_frames: int,
    gripper_source: str,
    gripper_timing: str,
) -> np.ndarray:
    tcp_poses = raw_feature_array(dataset, TCP_POSE_KEY, start, end)
    if tcp_poses.ndim != 2 or tcp_poses.shape[1] != 7:
        raise ValueError(f"Expected '{TCP_POSE_KEY}' shape (N, 7), got {tcp_poses.shape}")

    grippers = gripper_values(dataset, gripper_source, start, end)
    actions = []
    last = len(tcp_poses) - 1
    for idx, pose in enumerate(tcp_poses):
        target_idx = min(idx + lookahead_frames, last)
        gripper_idx = idx if gripper_timing == "current" else target_idx
        actions.append(delta_tcp_action(pose, tcp_poses[target_idx], grippers[gripper_idx]))
    return np.stack(actions)


def image_to_hwc_uint8(value: Any, feature: dict) -> np.ndarray:
    arr = as_numpy(value)
    shape = tuple(feature["shape"])
    if arr.ndim == 3 and len(shape) == 3 and arr.shape[0] == shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.5:
        arr = arr * 255.0
    return np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)


def feature_value_for_writer(value: Any, feature: dict) -> Any:
    dtype = feature["dtype"]
    if dtype in {"image", "video"}:
        return image_to_hwc_uint8(value, feature)

    arr = as_numpy(value)
    if dtype.startswith("float"):
        return np.asarray(arr, dtype=np.float32)
    if dtype.startswith("int"):
        return np.asarray(arr, dtype=np.int64)
    return arr


def convert_dataset(args: argparse.Namespace) -> None:
    output_repo_id = args.output_repo_id or default_output_repo_id(args.repo_id)
    output_root = Path(args.output_root).expanduser() if args.output_root else None

    if output_root and output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    logger.info("Loading source dataset: %s", args.repo_id)
    source = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.input_root,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        download_videos=not args.no_download_videos,
        video_backend=args.video_backend,
    )

    if args.lookahead_seconds is not None:
        lookahead_frames = max(1, round(float(args.lookahead_seconds) * source.fps))
    else:
        lookahead_frames = args.lookahead_frames
    logger.info("Delta lookahead: %d frame(s) at %s FPS", lookahead_frames, source.fps)

    output_features = build_output_features(source)
    use_videos = any(feature["dtype"] == "video" for feature in output_features.values())
    robot_type = source.meta.info.get("robot_type", "unknown")

    logger.info("Creating output dataset: %s", output_repo_id)
    output = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_root,
        fps=source.fps,
        robot_type=robot_type,
        features=output_features,
        use_videos=use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
    )

    feature_keys = [key for key in output_features if key != ACTION_KEY]
    episode_ranges = get_episode_ranges(source)

    try:
        for episode_index, (start, end) in enumerate(episode_ranges):
            logger.info(
                "Episode %d/%d: frames %d:%d",
                episode_index + 1,
                len(episode_ranges),
                start,
                end,
            )
            actions = compute_episode_actions(
                source,
                start,
                end,
                lookahead_frames,
                args.gripper_source,
                args.gripper_timing,
            )

            first_item = source[start]
            task = episode_task(source, episode_index, first_item)

            for offset, frame_idx in enumerate(range(start, end)):
                item = first_item if offset == 0 else source[frame_idx]
                frame = {
                    key: feature_value_for_writer(item[key], output_features[key])
                    for key in feature_keys
                }
                frame[ACTION_KEY] = actions[offset]
                frame["task"] = task
                output.add_frame(frame)

            output.save_episode()

        output.finalize()
        logger.info("Wrote converted dataset to %s", output.root)

        if args.push_to_hub:
            logger.info("Uploading %s to the Hugging Face Hub", output_repo_id)
            output.push_to_hub(
                private=args.private,
                push_videos=not args.no_push_videos,
                tags=args.tags,
                license=args.license,
            )
            logger.info("Upload complete: %s", output_repo_id)
    finally:
        output.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot HF dataset action to delta TCP pose."
    )
    parser.add_argument("--repo-id", required=True, help="Source HF dataset repo id, e.g. IXDLI/pnp_long_lero")
    parser.add_argument("--input-root", default=None, help="Optional local source dataset root/cache target")
    parser.add_argument("--revision", default=None, help="Source dataset revision")
    parser.add_argument("--output-repo-id", default=None, help="Defaults to <source_dataset_name>_deltaTCP")
    parser.add_argument("--output-root", default=None, help="Optional local output root")
    parser.add_argument("--overwrite", action="store_true", help="Replace --output-root if it already exists")
    parser.add_argument(
        "--lookahead-seconds",
        type=float,
        default=None,
        help="Seconds between current TCP pose and target TCP pose. Overrides --lookahead-frames.",
    )
    parser.add_argument(
        "--lookahead-frames",
        type=int,
        default=1,
        help="Exact frame offset between current and target TCP pose. Default: 1 for next-frame deltas.",
    )
    parser.add_argument(
        "--gripper-source",
        choices=["action", "state", "zero"],
        default="action",
        help="Where to copy the gripper value from. It is not differenced.",
    )
    parser.add_argument(
        "--gripper-timing",
        choices=["target", "current"],
        default="target",
        help="Use the target frame gripper by default, matching the TCP delta target.",
    )
    parser.add_argument("--video-backend", default=None, help="Video decoder backend for reading source videos")
    parser.add_argument("--vcodec", default="libsvtav1", help="Output video codec: libsvtav1, h264, hevc, auto")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=0)
    parser.add_argument("--force-cache-sync", action="store_true")
    parser.add_argument("--no-download-videos", action="store_true")
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", default=True)
    parser.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    parser.add_argument("--private", action="store_true", help="Upload as a private dataset")
    parser.add_argument("--no-push-videos", action="store_true", help="Skip videos during upload")
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--tags", nargs="*", default=["delta-tcp", "lerobot"])
    args = parser.parse_args()

    if args.lookahead_frames < 1:
        raise ValueError("--lookahead-frames must be >= 1")
    if args.lookahead_seconds is not None and args.lookahead_seconds <= 0:
        raise ValueError("--lookahead-seconds must be > 0")
    return args


if __name__ == "__main__":
    convert_dataset(parse_args())
