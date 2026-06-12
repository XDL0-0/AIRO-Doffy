#!/usr/bin/env python
"""Copy a LeRobot dataset while smoothing its action feature with Savitzky-Golay.

The output dataset keeps the source observations, videos, metadata features, and
tasks, but replaces ``action`` with a per-episode Savitzky-Golay filtered action.

By default the output Hugging Face repo id is
``<source_dataset_name>_SD<window_length><polyorder>``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_tool.convert_delta_tcp_dataset import (
    ACTION_KEY,
    episode_task,
    feature_value_for_writer,
    get_episode_ranges,
    raw_feature_array,
    user_features,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def default_output_repo_id(source_repo_id: str, window_length: int, polyorder: int) -> str:
    suffix = f"_SD{window_length}{polyorder}"
    if "/" in source_repo_id and not Path(source_repo_id).is_absolute():
        owner, name = source_repo_id.rsplit("/", 1)
        return f"{owner}/{name}{suffix}"
    return f"{Path(source_repo_id).name}{suffix}"


def build_output_features(source: LeRobotDataset) -> dict[str, dict]:
    features = user_features(source.features)
    if ACTION_KEY not in features:
        raise KeyError(f"Source dataset is missing '{ACTION_KEY}'.")
    return features


def effective_window_length(num_frames: int, window_length: int) -> int | None:
    window = min(window_length, num_frames)
    if window % 2 == 0:
        window -= 1
    return window if window >= 1 else None


def smooth_episode_actions(
    actions: np.ndarray,
    window_length: int,
    polyorder: int,
    mode: str,
    skip_last_action_dim: bool,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    num_frames = actions.shape[0]
    window = effective_window_length(num_frames, window_length)

    if window is None or window <= polyorder:
        logger.warning(
            "Episode has %d frame(s), too short for window_length=%d/polyorder=%d; keeping actions unchanged.",
            num_frames,
            window_length,
            polyorder,
        )
        return actions

    if actions.ndim == 1:
        return savgol_filter(actions, window_length=window, polyorder=polyorder, mode=mode).astype(np.float32)

    smooth_dims = actions.shape[1]
    if skip_last_action_dim and smooth_dims > 1:
        smooth_dims -= 1

    smoothed = actions.copy()
    smoothed[:, :smooth_dims] = savgol_filter(
        actions[:, :smooth_dims],
        window_length=window,
        polyorder=polyorder,
        axis=0,
        mode=mode,
    ).astype(np.float32)
    return smoothed


def frame_value(item: dict[str, Any], key: str, feature: dict) -> Any:
    return feature_value_for_writer(item[key], feature)


def output_dataset_root(output_repo_id: str, output_root: str | None) -> Path:
    if output_root:
        return Path(output_root).expanduser()
    return HF_LEROBOT_HOME / output_repo_id


def prepare_output_root(root: Path, overwrite: bool) -> None:
    if not root.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"{root} already exists. Use --overwrite to replace the existing SG dataset."
        )
    logger.info("Removing existing output dataset: %s", root)
    shutil.rmtree(root)


def smooth_dataset(args: argparse.Namespace) -> None:
    output_repo_id = args.output_repo_id or default_output_repo_id(
        args.repo_id,
        args.window_length,
        args.polyorder,
    )
    output_root = output_dataset_root(output_repo_id, args.output_root)
    prepare_output_root(output_root, args.overwrite)

    logger.info("Loading source dataset: %s", args.repo_id)
    source = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.input_root,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        download_videos=not args.no_download_videos,
        video_backend=args.video_backend,
    )

    output_features = build_output_features(source)
    use_videos = any(feature["dtype"] == "video" for feature in output_features.values())
    robot_type = source.meta.info.get("robot_type", "unknown")

    logger.info(
        "Creating SG-smoothed dataset: %s (window_length=%d, polyorder=%d)",
        output_repo_id,
        args.window_length,
        args.polyorder,
    )
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
    finalized = False

    try:
        for episode_index, (start, end) in enumerate(episode_ranges):
            logger.info(
                "Episode %d/%d: frames %d:%d",
                episode_index + 1,
                len(episode_ranges),
                start,
                end,
            )

            actions = raw_feature_array(source, ACTION_KEY, start, end)
            smoothed_actions = smooth_episode_actions(
                actions,
                args.window_length,
                args.polyorder,
                args.mode,
                args.skip_last_action_dim,
            )

            first_item = source[start]
            task = episode_task(source, episode_index, first_item)

            for offset, frame_idx in enumerate(range(start, end)):
                item = first_item if offset == 0 else source[frame_idx]
                frame = {
                    key: frame_value(item, key, output_features[key])
                    for key in feature_keys
                }
                frame[ACTION_KEY] = np.asarray(smoothed_actions[offset], dtype=np.float32)
                frame["task"] = task
                output.add_frame(frame)

            output.save_episode()

        output.finalize()
        finalized = True
        logger.info("Wrote SG-smoothed dataset to %s", output.root)

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
        if not finalized:
            output.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smooth a LeRobot dataset action feature with a Savitzky-Golay filter."
    )
    parser.add_argument("--repo-id", required=True, help="Source HF dataset repo id, e.g. IXDLI/pnp_long_lero")
    parser.add_argument("--input-root", default=None, help="Optional local source dataset root/cache target")
    parser.add_argument("--revision", default=None, help="Source dataset revision")
    parser.add_argument(
        "--output-repo-id",
        default=None,
        help="Defaults to <source_dataset_name>_SD<window_length><polyorder>, e.g. dataset_SD31",
    )
    parser.add_argument("--output-root", default=None, help="Optional local output root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output dataset if it already exists locally or in the LeRobot cache",
    )
    parser.add_argument("--window-length", type=int, default=3, help="Savitzky-Golay window length")
    parser.add_argument("--polyorder", type=int, default=1, help="Savitzky-Golay polynomial order")
    parser.add_argument(
        "--mode",
        choices=["interp", "mirror", "nearest", "constant", "wrap"],
        default="interp",
        help="Edge handling mode passed to scipy.signal.savgol_filter",
    )
    parser.add_argument(
        "--skip-last-action-dim",
        action="store_true",
        help="Leave the last action dimension unchanged, useful when it is a gripper command.",
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
    parser.add_argument("--tags", nargs="*", default=["savitzky-golay", "lerobot"])
    args = parser.parse_args()

    if args.window_length < 1:
        raise ValueError("--window-length must be >= 1")
    if args.window_length % 2 == 0:
        raise ValueError("--window-length must be odd")
    if args.polyorder < 0:
        raise ValueError("--polyorder must be >= 0")
    if args.polyorder >= args.window_length:
        raise ValueError("--polyorder must be < --window-length")
    return args


if __name__ == "__main__":
    smooth_dataset(parse_args())
