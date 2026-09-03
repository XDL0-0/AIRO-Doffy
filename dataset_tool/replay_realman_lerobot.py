"""Safely replay joint or TCP trajectories from a RealMan LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger(__name__)
REALMAN_DOF = 7
DEFAULT_DATASET_DIR = "./datasets/WRM_grasp_cylinder_different_sizes_lero"


@dataclass(frozen=True)
class EpisodeTrajectory:
    episode_index: int
    targets: np.ndarray
    fps: float
    control_mode: str = "joint"

    @property
    def duration_s(self) -> float:
        return len(self.targets) / self.fps

    @property
    def joints(self) -> np.ndarray:
        if self.control_mode != "joint":
            raise ValueError(
                "This episode trajectory contains TCP targets, not joints."
            )
        return self.targets

    @property
    def maximum_joint_speed(self) -> float:
        if self.control_mode != "joint" or len(self.targets) < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(self.targets, axis=0))) * self.fps)

    @property
    def maximum_linear_speed(self) -> float:
        if self.control_mode != "tcp" or len(self.targets) < 2:
            return 0.0
        translations = self.targets[:, 4:7]
        return float(
            np.max(np.linalg.norm(np.diff(translations, axis=0), axis=1)) * self.fps
        )

    @property
    def maximum_angular_speed(self) -> float:
        if self.control_mode != "tcp" or len(self.targets) < 2:
            return 0.0
        quaternions = self.targets[:, :4]
        quaternions = quaternions / np.linalg.norm(
            quaternions,
            axis=1,
            keepdims=True,
        )
        adjacent_dots = np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1))
        angles = 2.0 * np.arccos(np.clip(adjacent_dots, 0.0, 1.0))
        return float(np.max(angles) * self.fps)


class RealManLeRobotDataset:
    """Read and validate joint or TCP data from a local LeRobot v3 dataset."""

    def __init__(
        self,
        root: str | Path,
        *,
        control_mode: str = "joint",
        source: str = "observation.state",
    ) -> None:
        if control_mode not in {"joint", "tcp"}:
            raise ValueError("control_mode must be 'joint' or 'tcp'.")
        if source not in {"action", "observation.state"}:
            raise ValueError("source must be 'action' or 'observation.state'.")
        self.root = Path(root).expanduser().resolve()
        self.control_mode = control_mode
        self.source = "extra.tcp_pose" if control_mode == "tcp" else source
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"Missing LeRobot metadata: {info_path}. Pass the dataset root "
                "that contains meta/info.json."
            )
        self.info = json.loads(info_path.read_text())
        self.fps = float(self.info.get("fps", 0.0))
        self.total_episodes = int(self.info.get("total_episodes", 0))
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path = self.info.get(
            "data_path",
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        )
        self.video_path = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        self._episode_table_cache: pd.DataFrame | None = None
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        if self.fps <= 0.0:
            raise ValueError(f"Dataset fps must be positive, got {self.fps}.")
        if self.total_episodes <= 0:
            raise ValueError("Dataset contains no episodes.")
        robot_type = str(self.info.get("robot_type", "")).lower()
        if robot_type and robot_type != "realman":
            raise ValueError(f"Dataset robot_type is '{robot_type}', not 'realman'.")

        feature = self.info.get("features", {}).get(self.source)
        if not isinstance(feature, dict):
            # This is an invalid metadata value/schema, not a caller type error.
            raise ValueError(  # noqa: TRY004
                f"Dataset does not contain feature '{self.source}'."
            )
        shape = tuple(feature.get("shape", ()))
        if shape != (REALMAN_DOF,):
            representation = (
                "[qx,qy,qz,qw,x,y,z] TCP"
                if self.control_mode == "tcp"
                else "seven-joint RealMan"
            )
            raise ValueError(
                f"Feature '{self.source}' has shape {shape}; expected "
                f"({REALMAN_DOF},) for a {representation} trajectory."
            )
        names = feature.get("names")
        if names is not None and len(names) != REALMAN_DOF:
            raise ValueError(
                f"Feature '{self.source}' has {len(names)} names; expected "
                f"{REALMAN_DOF}."
            )

    def episode_indices(
        self,
        *,
        episodes: list[int] | None = None,
        from_episode: int = 0,
        to_episode: int | None = None,
    ) -> list[int]:
        if episodes is not None:
            result = episodes
        else:
            stop = self.total_episodes if to_episode is None else to_episode
            result = list(range(from_episode, stop))
        if not result:
            raise ValueError("No episodes selected.")
        invalid = [idx for idx in result if idx < 0 or idx >= self.total_episodes]
        if invalid:
            raise IndexError(
                f"Episode index/indices {invalid} are outside "
                f"[0, {self.total_episodes - 1}]."
            )
        return result

    def load_episode(self, episode_index: int) -> EpisodeTrajectory:
        if not 0 <= episode_index < self.total_episodes:
            raise IndexError(
                f"Episode {episode_index} is outside [0, {self.total_episodes - 1}]."
            )
        frames = self._load_episode_frames(episode_index)
        if "frame_index" in frames:
            frames = frames.sort_values("frame_index")
        targets = self._extract_vectors(frames, self.source)
        if targets.shape[0] == 0:
            raise ValueError(f"Episode {episode_index} contains no frames.")
        if targets.shape[1:] != (7,) or not np.all(np.isfinite(targets)):
            raise ValueError(
                f"Episode {episode_index} has invalid '{self.source}' data with "
                f"shape {targets.shape}."
            )
        if self.control_mode == "tcp":
            quaternion_norms = np.linalg.norm(targets[:, :4], axis=1)
            if np.any(quaternion_norms < 1e-6):
                raise ValueError(
                    f"Episode {episode_index} contains an invalid zero TCP quaternion."
                )
        return EpisodeTrajectory(
            episode_index,
            targets,
            self.fps,
            self.control_mode,
        )

    @property
    def video_keys(self) -> list[str]:
        """Return the dataset features backed by per-episode video files."""
        return sorted(
            key
            for key, feature in self.info.get("features", {}).items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        )

    def episode_video_path(self, episode_index: int, video_key: str) -> Path:
        """Resolve one episode video using LeRobot v3 episode metadata.

        v3 packs many episodes into each mp4. ``file_index`` is the packed
        video file from ``meta/episodes``, not ``episode_index % chunks_size``.
        """
        if video_key not in self.video_keys:
            raise KeyError(f"Dataset does not contain video feature '{video_key}'.")
        if not 0 <= episode_index < self.total_episodes:
            raise IndexError(
                f"Episode {episode_index} is outside [0, {self.total_episodes - 1}]."
            )
        chunk_index, file_index = self._legacy_chunk_file(episode_index)
        row = self._episode_row(episode_index)
        if row is not None:
            packed_chunk = _optional_int(row, f"videos/{video_key}/chunk_index")
            packed_file = _optional_int(row, f"videos/{video_key}/file_index")
            if packed_chunk is not None and packed_file is not None:
                chunk_index, file_index = packed_chunk, packed_file
        return self.root / self.video_path.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_index=episode_index,
        )

    def episode_video_from_timestamp(self, episode_index: int, video_key: str) -> float:
        """Return the packed-video timestamp of this episode's first frame."""
        row = self._episode_row(episode_index)
        if row is None:
            return 0.0
        timestamp = _optional_float(row, f"videos/{video_key}/from_timestamp")
        return 0.0 if timestamp is None else timestamp

    def load_first_camera_frames(self, episode_index: int) -> dict[str, np.ndarray]:
        """Decode the first BGR frame of every camera video in an episode."""
        frames: dict[str, np.ndarray] = {}
        for video_key in self.video_keys:
            path = self.episode_video_path(episode_index, video_key)
            if not path.is_file():
                logger.warning(
                    "Episode %d camera video is missing for '%s': %s",
                    episode_index,
                    video_key,
                    path,
                )
                continue
            frame = read_first_video_frame(
                path,
                timestamp_s=self.episode_video_from_timestamp(episode_index, video_key),
            )
            if frame is None:
                logger.warning(
                    "Could not decode the first frame of episode %d camera '%s': %s",
                    episode_index,
                    video_key,
                    path,
                )
                continue
            frames[video_key] = frame
        return frames

    def _legacy_chunk_file(self, episode_index: int) -> tuple[int, int]:
        return episode_index // self.chunks_size, episode_index % self.chunks_size

    def _episode_table(self) -> pd.DataFrame:
        if self._episode_table_cache is not None:
            return self._episode_table_cache
        paths = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not paths:
            self._episode_table_cache = pd.DataFrame()
            return self._episode_table_cache
        try:
            frames = [pd.read_parquet(path) for path in paths]
        except Exception as exc:  # noqa: BLE001 - parquet/pyarrow errors vary.
            logger.warning("Could not read LeRobot episode metadata: %s", exc)
            self._episode_table_cache = pd.DataFrame()
            return self._episode_table_cache
        table = pd.concat(frames, ignore_index=True)
        if "episode_index" in table.columns:
            table = table.sort_values("episode_index").reset_index(drop=True)
        self._episode_table_cache = table
        return table

    def _episode_row(self, episode_index: int) -> pd.Series | None:
        table = self._episode_table()
        if table.empty or "episode_index" not in table.columns:
            return None
        matches = table[table["episode_index"] == episode_index]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _load_episode_frames(self, episode_index: int) -> pd.DataFrame:
        chunk_index, file_index = self._legacy_chunk_file(episode_index)
        exact_path = self.root / self.data_path.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        row = self._episode_row(episode_index)
        preferred: list[Path] = []
        if row is not None:
            packed_chunk = _optional_int(row, "data/chunk_index")
            packed_file = _optional_int(row, "data/file_index")
            if packed_chunk is not None and packed_file is not None:
                meta_path = self.root / self.data_path.format(
                    chunk_index=packed_chunk,
                    file_index=packed_file,
                )
                if meta_path.is_file():
                    preferred.append(meta_path)
                    exact_path = meta_path
                    chunk_index = packed_chunk

        # Packed v3 files hold several episodes, so scan the chunk if the
        # metadata path is missing or does not contain this episode.
        chunk_paths = sorted(
            (self.root / "data" / f"chunk-{chunk_index:03d}").glob("*.parquet")
        )
        paths: list[Path] = []
        seen: set[Path] = set()
        for path in preferred + chunk_paths:
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
        if not paths:
            raise FileNotFoundError(
                f"No parquet data found for episode {episode_index}; expected "
                f"{exact_path}."
            )

        matches = []
        for path in paths:
            frame = pd.read_parquet(path)
            if "episode_index" in frame:
                frame = frame[frame["episode_index"] == episode_index]
            elif path != exact_path:
                # A legacy per-episode file without the column is only usable
                # when it is the exact file for this episode.
                continue
            if not frame.empty:
                matches.append(frame)
                if path in preferred:
                    break
        if not matches:
            raise ValueError(f"No rows found for episode {episode_index}.")
        return pd.concat(matches, ignore_index=True)

    @staticmethod
    def _extract_vectors(frames: pd.DataFrame, key: str) -> np.ndarray:
        if key in frames:
            try:
                return np.stack(
                    [
                        np.asarray(value, dtype=float).reshape(-1)
                        for value in frames[key]
                    ]
                )
            except ValueError as exc:
                raise ValueError(
                    f"Feature '{key}' contains inconsistent vectors."
                ) from exc

        prefixes = (f"{key}.", f"{key}_")
        columns = sorted(
            column
            for column in frames.columns
            if any(column.startswith(prefix) for prefix in prefixes)
        )
        if not columns:
            raise KeyError(
                f"Parquet data does not contain '{key}'. Available columns: "
                f"{list(frames.columns)}"
            )
        return frames[columns].to_numpy(dtype=float)


def _optional_int(row: pd.Series, key: str) -> int | None:
    if key not in row.index:
        return None
    value = row[key]
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(row: pd.Series, key: str) -> float | None:
    if key not in row.index:
        return None
    value = row[key]
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_first_video_frame(
    path: str | Path,
    *,
    timestamp_s: float = 0.0,
) -> np.ndarray | None:
    """Decode a video frame as uint8 BGR, seeking when episodes share one file."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if capture.isOpened():
            if timestamp_s > 0.0:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
            success, frame = capture.read()
            if success and frame is not None:
                return frame
    finally:
        capture.release()

    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(str(path)) as container:
            if timestamp_s > 0.0:
                container.seek(int(timestamp_s * av.time_base))
            for decoded in container.decode(video=0):
                frame_time = decoded.time
                if (
                    timestamp_s > 0.0
                    and frame_time is not None
                    and frame_time + 1e-3 < timestamp_s
                ):
                    continue
                return decoded.to_ndarray(format="bgr24")
    except Exception as exc:  # noqa: BLE001 - PyAV backend exceptions vary by codec.
        logger.debug("PyAV could not decode %s: %s", path, exc)
    return None


def camera_name_from_video_key(video_key: str, fallback_index: int) -> str:
    """Map observation.images.camera_0-style keys to live camera names."""
    match = re.search(r"camera[_-]?(\d+)", video_key)
    if match:
        return f"camera_{int(match.group(1))}"
    return f"camera_{fallback_index}"


def _uint8_rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert an airo-camera-toolkit RGB image into OpenCV BGR."""
    import cv2

    image = np.asarray(frame)
    if image.dtype != np.uint8:
        if image.size and np.issubdtype(image.dtype, np.floating):
            finite_max = float(np.nanmax(image))
            if finite_max <= 1.5:
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError(f"Unsupported live camera frame shape {image.shape}.")
    conversion = cv2.COLOR_RGBA2BGR if image.shape[2] == 4 else cv2.COLOR_RGB2BGR
    return cv2.cvtColor(image, conversion)


def _resize_to_match(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cover-crop ``frame`` to ``target`` HxW without stretching."""
    import cv2

    target_h, target_w = int(target.shape[0]), int(target.shape[1])
    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    if (src_h, src_w) == (target_h, target_w):
        return frame
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)
    cropped = resized[y0 : y0 + target_h, x0 : x0 + target_w]
    if cropped.shape[:2] == (target_h, target_w):
        return cropped
    canvas = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
    canvas[: cropped.shape[0], : cropped.shape[1]] = cropped
    return canvas


def _labeled_frame(frame: np.ndarray, label: str) -> np.ndarray:
    import cv2

    result = frame.copy()
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 260), 28), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return result


def _opencv_window_available() -> tuple[bool, str]:
    """Check common headless cases before calling cv2.imshow()."""
    import cv2

    gui_line = next(
        (
            line.strip()
            for line in cv2.getBuildInformation().splitlines()
            if line.strip().startswith("GUI:")
        ),
        "GUI: UNKNOWN",
    )
    if "NONE" in gui_line:
        return False, f"OpenCV has no GUI backend ({gui_line})."
    if os.name == "posix" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False, "DISPLAY/WAYLAND_DISPLAY is not set."
    return True, gui_line


def run_camera_alignment(
    dataset: RealManLeRobotDataset,
    episode_index: int,
) -> bool:
    """Block replay while dataset first frames are overlaid on live cameras.

    Returns True when an interactive alignment view was shown, otherwise False.
    """
    import cv2

    window_available, reason = _opencv_window_available()
    if not window_available:
        logger.warning("Skipping camera alignment: %s", reason)
        return False

    dataset_frames = dataset.load_first_camera_frames(episode_index)
    if not dataset_frames:
        logger.warning(
            "Skipping camera alignment: episode %d has no readable camera first frame.",
            episode_index,
        )
        return False

    try:
        import pyrealsense2 as rs
        from airo_camera_toolkit.cameras.realsense.realsense import Realsense
    except ImportError as exc:
        logger.warning(
            "Skipping camera alignment: RealSense support is unavailable: %s", exc
        )
        return False

    devices = rs.context().query_devices()
    serial_numbers: list[str] = []
    for index, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        serial_numbers.append(serial)
        logger.info("RealSense camera %d: %s, serial=%s", index, name, serial)
    if not serial_numbers:
        logger.warning("Skipping camera alignment: no RealSense camera is connected.")
        return False

    cameras: dict[str, Any] = {}
    windows: list[str] = []
    try:
        for fallback_index, video_key in enumerate(dataset_frames):
            camera_name = camera_name_from_video_key(video_key, fallback_index)
            match = re.search(r"camera_(\d+)$", camera_name)
            camera_index = int(match.group(1)) if match else fallback_index
            if camera_index >= len(serial_numbers):
                logger.warning(
                    "Dataset camera '%s' maps to %s, but only %d live camera(s) "
                    "were detected.",
                    video_key,
                    camera_name,
                    len(serial_numbers),
                )
                continue
            # Dataset videos are 640x480. airo's RESOLUTION_480 is 848x480, which
            # stretches into the overlay and makes the live scene look too tall.
            dataset_height, dataset_width = dataset_frames[video_key].shape[:2]
            live_resolution = (int(dataset_width), int(dataset_height))
            cameras[video_key] = Realsense(
                fps=30,
                resolution=live_resolution,
                enable_depth=False,
                enable_pointcloud=False,
                enable_hole_filling=False,
                serial_number=serial_numbers[camera_index],
            )
            windows.append(f"Camera alignment - {camera_name}")
            logger.info(
                "Live %s opened at %dx%d to match dataset frames.",
                camera_name,
                live_resolution[0],
                live_resolution[1],
            )

        if not cameras:
            logger.warning(
                "Skipping camera alignment: no dataset/live camera pair matched."
            )
            return False

        logger.info(
            "Align episode %d using dataset first frame | live camera | 50%% overlay.",
            episode_index,
        )
        logger.info("Press q, Enter, or Esc when camera alignment is complete.")
        while True:
            for (video_key, camera), window_name in zip(cameras.items(), windows):
                camera.grab_images()
                live_frame = _uint8_rgb_to_bgr(camera.retrieve_rgb_image())
                dataset_frame = dataset_frames[video_key]
                live_frame = _resize_to_match(live_frame, dataset_frame)
                overlay = cv2.addWeighted(dataset_frame, 0.5, live_frame, 0.5, 0.0)
                display = np.concatenate(
                    [
                        _labeled_frame(dataset_frame, "dataset first frame"),
                        _labeled_frame(live_frame, "live camera"),
                        _labeled_frame(overlay, "50% overlay"),
                    ],
                    axis=1,
                )
                cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF in {ord("q"), 10, 13, 27}:
                return True
    except (RuntimeError, ValueError, cv2.error) as exc:
        logger.warning(
            "Camera alignment stopped because capture/display failed: %s", exc
        )
        return False
    finally:
        for camera in cameras.values():
            close = getattr(camera, "close", None)
            try:
                if callable(close):
                    close()
                else:
                    pipeline = getattr(camera, "pipeline", None)
                    if pipeline is not None:
                        pipeline.stop()
            except RuntimeError as exc:
                logger.debug("Ignoring RealSense shutdown error: %s", exc)
        for window_name in windows:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass


def validate_trajectory_speed(
    trajectory: EpisodeTrajectory,
    maximum_joint_speed: float,
) -> None:
    if trajectory.control_mode != "joint":
        raise ValueError("validate_trajectory_speed requires a joint trajectory.")
    if maximum_joint_speed <= 0.0:
        raise ValueError("maximum_joint_speed must be positive.")
    measured = trajectory.maximum_joint_speed
    if measured > maximum_joint_speed:
        raise ValueError(
            f"Episode {trajectory.episode_index} reaches {measured:.3f} rad/s, "
            f"above the replay safety limit of {maximum_joint_speed:.3f} rad/s. "
            "Inspect the dataset before raising --max-joint-speed."
        )


def validate_tcp_trajectory_speed(
    trajectory: EpisodeTrajectory,
    maximum_linear_speed: float,
    maximum_angular_speed: float,
) -> None:
    if trajectory.control_mode != "tcp":
        raise ValueError("validate_tcp_trajectory_speed requires a TCP trajectory.")
    if maximum_linear_speed <= 0.0 or maximum_angular_speed <= 0.0:
        raise ValueError("TCP speed limits must be positive.")
    if trajectory.maximum_linear_speed > maximum_linear_speed:
        raise ValueError(
            f"Episode {trajectory.episode_index} reaches "
            f"{trajectory.maximum_linear_speed:.3f} m/s, above the replay safety "
            f"limit of {maximum_linear_speed:.3f} m/s. Inspect the dataset before "
            "raising --max-linear-speed."
        )
    if trajectory.maximum_angular_speed > maximum_angular_speed:
        raise ValueError(
            f"Episode {trajectory.episode_index} reaches "
            f"{trajectory.maximum_angular_speed:.3f} rad/s angular speed, above "
            f"the replay safety limit of {maximum_angular_speed:.3f} rad/s. "
            "Inspect the dataset before raising --max-angular-speed."
        )


def tcp_vector_to_pose(tcp_vector: np.ndarray) -> np.ndarray:
    """Convert [qx,qy,qz,qw,x,y,z] to a homogeneous TCP pose."""
    vector = np.asarray(tcp_vector, dtype=float)
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Expected seven finite TCP values, got {vector.shape}.")
    quaternion = vector[:4]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        raise ValueError("TCP quaternion norm is zero.")
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    pose[:3, 3] = vector[4:7]
    return pose


class RealManTrajectoryReplayer:
    """Send a validated joint or TCP trajectory through RealMan low-follow CAN-FD."""

    def __init__(
        self,
        robot: Any,
        *,
        control_mode: str = "joint",
        tcp_transform: np.ndarray | None = None,
        initial_speed: float = 0.8,
        initial_timeout: float = 30.0,
    ) -> None:
        if control_mode not in {"joint", "tcp"}:
            raise ValueError("control_mode must be 'joint' or 'tcp'.")
        if int(robot.dof) != REALMAN_DOF:
            raise ValueError(
                f"Connected RealMan reports {robot.dof} DoF; expected {REALMAN_DOF}."
            )
        if initial_speed <= 0.0:
            raise ValueError("initial_speed must be positive.")
        if initial_timeout <= 0.0:
            raise ValueError("initial_timeout must be positive.")
        self.robot = robot
        self.control_mode = control_mode
        self.tcp_transform = (
            np.eye(4)
            if tcp_transform is None
            else np.asarray(tcp_transform, dtype=float)
        )
        if self.tcp_transform.shape != (4, 4) or not np.all(
            np.isfinite(self.tcp_transform)
        ):
            raise ValueError("tcp_transform must be a finite 4x4 matrix.")
        self.inverse_tcp_transform = np.linalg.inv(self.tcp_transform)
        self.initial_speed = float(initial_speed)
        self.initial_timeout = float(initial_timeout)

    def move_to_start(self, target: np.ndarray) -> None:
        if self.control_mode == "joint":
            action = self.robot.move_to_joint_configuration(
                self._validate_joint_target(target),
                joint_speed=self.initial_speed,
            )
        else:
            action = self.robot.move_to_tcp_pose(
                self._robot_tcp_pose(target),
                joint_speed=self.initial_speed,
            )
        status = action.wait(timeout=self.initial_timeout, sleep_resolution=0.02)
        if getattr(status, "name", "") != "SUCCEEDED":
            self.stop_motion()
            raise TimeoutError(
                f"Timed out moving RealMan to the episode start pose after "
                f"{self.initial_timeout:.1f} seconds."
            )

    def replay(self, trajectory: EpisodeTrajectory) -> None:
        if trajectory.control_mode != self.control_mode:
            raise ValueError(
                f"Replayer mode is '{self.control_mode}', but episode mode is "
                f"'{trajectory.control_mode}'."
            )
        period = 1.0 / trajectory.fps
        if period <= 0.01:
            raise ValueError(
                "Dataset replay at 100 Hz or above requires a dedicated high-follow "
                "timing watchdog; this utility only uses low-follow CAN-FD."
            )

        next_tick = time.perf_counter()
        for frame_index, target in enumerate(trajectory.targets):
            if self.control_mode == "joint":
                self.robot.servo_to_joint_configuration(
                    self._validate_joint_target(target),
                    period,
                )
            else:
                self.robot.servo_to_tcp_pose(
                    self._robot_tcp_pose(target),
                    period,
                )
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            else:
                logger.warning(
                    "Replay deadline missed at episode %d frame %d by %.1f ms.",
                    trajectory.episode_index,
                    frame_index,
                    -delay * 1000.0,
                )
                next_tick = time.perf_counter()

    def stop_motion(self) -> None:
        """Request a controller-level trajectory slow stop when available."""
        arm = getattr(self.robot, "robot", None)
        slow_stop = getattr(arm, "rm_set_arm_slow_stop", None)
        if not callable(slow_stop):
            logger.warning(
                "RealMan SDK does not expose rm_set_arm_slow_stop; use the "
                "teach pendant stop if motion continues."
            )
            return
        result = int(slow_stop())
        if result != 0:
            logger.error("RealMan slow stop failed with error code %d.", result)

    @staticmethod
    def _validate_joint_target(joints: np.ndarray) -> np.ndarray:
        target = np.asarray(joints, dtype=float)
        if target.shape != (REALMAN_DOF,) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"Expected {REALMAN_DOF} finite joint values, got {target.shape}."
            )
        return target

    def _robot_tcp_pose(self, tcp_vector: np.ndarray) -> np.ndarray:
        tool_tcp_pose = tcp_vector_to_pose(tcp_vector)
        return tool_tcp_pose @ self.inverse_tcp_transform


def confirm(message: str, *, assume_yes: bool) -> None:
    if assume_yes:
        return
    while True:
        answer = input(f"{message} Press Enter to continue or type 'c' to quit: ")
        if answer.strip() == "":
            return
        if answer.strip().lower() == "c":
            raise KeyboardInterrupt("Replay canceled by user.")
        print("Please press Enter to continue or type 'c' to quit.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay joint or TCP data from a RealMan LeRobot dataset."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--robot-ip", default="192.168.1.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--from-episode", type=int, default=0)
    parser.add_argument(
        "--to-episode",
        type=int,
        default=None,
        help="Exclusive upper bound; defaults to all remaining episodes.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["joint", "tcp"],
        default="joint",
        help="Replay joint configurations or extra.tcp_pose Cartesian targets.",
    )
    parser.add_argument(
        "--source",
        choices=["action", "observation.state"],
        default="observation.state",
        help="Joint feature to command in --control-mode joint. "
        "observation.state (the measured trajectory) is the safe default; "
        "action may contain command-target glitches.",
    )
    parser.add_argument(
        "--initial-speed",
        type=float,
        default=2.0,
        help="Move-to-start joint speed (rad/s); the airo_robots wrapper scales "
        "it to a percentage of the arm max (3.14 rad/s = 100% on RM75).",
    )
    parser.add_argument("--initial-timeout", type=float, default=30.0)
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=2.5,
        help="Reject episodes with a larger recorded frame-to-frame speed (rad/s).",
    )
    parser.add_argument(
        "--max-linear-speed",
        type=float,
        default=0.75,
        help="TCP-mode frame-to-frame translation speed limit (m/s).",
    )
    parser.add_argument(
        "--max-angular-speed",
        type=float,
        default=2.0,
        help="TCP-mode frame-to-frame rotation speed limit (rad/s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize episodes without connecting to the robot.",
    )
    parser.add_argument(
        "--align-camera",
        action="store_true",
        help="Before each episode replay, overlay its dataset camera first frame "
        "with the matching live RealSense image for manual scene alignment.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompts (not recommended for first replay).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = RealManLeRobotDataset(
        args.dataset_dir,
        control_mode=args.control_mode,
        source=args.source,
    )
    episode_indices = dataset.episode_indices(
        episodes=args.episodes,
        from_episode=args.from_episode,
        to_episode=args.to_episode,
    )
    trajectories = [dataset.load_episode(index) for index in episode_indices]
    for trajectory in trajectories:
        if args.control_mode == "joint":
            validate_trajectory_speed(trajectory, args.max_joint_speed)
            logger.info(
                "Episode %d: %d joint frames, %.2f s, max joint speed %.3f rad/s.",
                trajectory.episode_index,
                len(trajectory.targets),
                trajectory.duration_s,
                trajectory.maximum_joint_speed,
            )
        else:
            validate_tcp_trajectory_speed(
                trajectory,
                args.max_linear_speed,
                args.max_angular_speed,
            )
            logger.info(
                "Episode %d: %d TCP frames, %.2f s, max linear speed %.3f "
                "m/s, max angular speed %.3f rad/s.",
                trajectory.episode_index,
                len(trajectory.targets),
                trajectory.duration_s,
                trajectory.maximum_linear_speed,
                trajectory.maximum_angular_speed,
            )

    if args.dry_run:
        logger.info("Dry run complete; no robot connection was opened.")
        return 0

    from airo_robots.manipulators.hardware.realman import RealmanControl

    from config import Config

    robot = RealmanControl(ip_address=args.robot_ip, port=args.port)
    replayer = None
    try:
        replayer = RealManTrajectoryReplayer(
            robot,
            control_mode=args.control_mode,
            tcp_transform=np.asarray(Config().TCP_TRANSFORM, dtype=float),
            initial_speed=args.initial_speed,
            initial_timeout=args.initial_timeout,
        )
        for trajectory in trajectories:
            if args.control_mode == "joint":
                start_description = (
                    f"joint pose {np.round(np.degrees(trajectory.targets[0]), 1).tolist()} "
                    "degrees"
                )
            else:
                start_description = (
                    "TCP pose [qx,qy,qz,qw,x,y,z] "
                    f"{np.round(trajectory.targets[0], 4).tolist()}"
                )
            confirm(
                f"Episode {trajectory.episode_index}: move to {start_description}.",
                assume_yes=args.yes,
            )
            replayer.move_to_start(trajectory.targets[0])
            if args.align_camera:
                run_camera_alignment(dataset, trajectory.episode_index)
            confirm(
                f"RealMan is at the start of episode {trajectory.episode_index}; "
                "begin trajectory replay.",
                assume_yes=args.yes,
            )
            logger.info("Replaying episode %d...", trajectory.episode_index)
            replayer.replay(trajectory)
            logger.info("Episode %d replay complete.", trajectory.episode_index)
    except KeyboardInterrupt:
        if replayer is not None:
            replayer.stop_motion()
        logger.info("Replay stopped; a RealMan trajectory slow stop was requested.")
        return 130
    except Exception:
        if replayer is not None:
            replayer.stop_motion()
        raise
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
