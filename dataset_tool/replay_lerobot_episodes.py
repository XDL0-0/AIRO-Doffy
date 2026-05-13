"""
Replay LeRobot Dataset on UR3e + Robotiq 2F-85
================================================
Compatible with LeRobot codebase_version v3.0 (parquet + mp4 format)

Dataset structure expected:
  datasets/
  └── <dataset_name>/
      ├── meta/
      │   ├── info.json        # fps, total_episodes, features, etc.
      │   ├── episodes/        # per-episode metadata (optional)
      │   └── stats.json
      ├── data/
      │   └── chunk-000/
      │       └── file-000.parquet
      └── videos/              # optional, for camera preview
          └── <video_key>/
              └── chunk-000/
                  └── file-000.mp4

Usage:
  python -m dataset_tool.replay_lerobot_episodes \
      --dataset_dir ./datasets/pick_and_place_lero\
      --from_episode 0 \
      --to_episode 66 \
      --robot_ip 10.42.0.162 \
      --align_camera\
      --no_robot \
      --show_video \
      --tactile

      [--data_type qpos]       # qpos | eef | tcp_quat
      [--no_robot]             # dry-run: print actions only, no robot needed
      [--show_video]           # replay dataset camera videos
      [--tactile]              # show tactile replay if observation.tactile exists
      [--visualization_backend auto|window|video|none]
"""

import argparse
import json
import os
import subprocess
import time
import glob
import re
import warnings
import sys
import logging


import numpy as np
import pandas as pd
import cv2


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Replay a LeRobot dataset on UR3e")
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="Path to the LeRobot dataset root folder")
parser.add_argument("--from_episode", type=int, default=0)
parser.add_argument("--to_episode",   type=int, default=None,
                    help="Exclusive upper bound. Defaults to total_episodes.")
parser.add_argument("--robot_ip",     type=str, default="10.42.0.162")
parser.add_argument("--robot_type",   type=str, default="ur3e",
                    choices=["ur3e", "ur5e"],
                    help="Robot model passed to make_robot()")
parser.add_argument("--data_type",    type=str, default="qpos",
                    choices=["qpos", "eef", "tcp_quat"],
                    help="Representation used in the dataset")

parser.add_argument("--no_robot",     action="store_true",
                    help="Dry-run mode: do not connect to the robot")
parser.add_argument("--torque_mode",  action="store_true",
                    help="URrtdeTorque: tmp_move/moveJ to initial pose, replay via ur.target_pos (RTDE torque loop)")
parser.add_argument("--fast_gripper", action="store_true",
                    help="Use FastRobotiq2F85 (persistent socket, non-blocking)")
parser.add_argument("--initial_gripper_timeout", type=float, default=2.0,
                    help="Max seconds to wait for the gripper when moving to the initial pose. Use 0 to skip waiting.")
parser.add_argument("--show_video",   action="store_true",
                    help="Show dataset camera videos (requires OpenCV window)")
parser.add_argument("--align_camera", action="store_true",
                    help="Before replaying the first loaded episode, blend its first dataset camera frame with live RealSense frames for environment alignment.")
parser.add_argument("--tactile",      action="store_true",
                    help="Show tactile visualization if observation.tactile exists")
parser.add_argument("--visualization_backend", type=str, default="auto",
                    choices=["auto", "window", "video", "none"],
                    help="Visualization output. auto uses an OpenCV window when available, otherwise saves mp4.")
parser.add_argument("--replay_video_dir", type=str, default=None,
                    help="Directory for saved replay visualization mp4 files when using video backend.")
parser.add_argument("--fps_override", type=float, default=None,
                    help="Override FPS from info.json")
args = parser.parse_args()

# ──────────────────────────────────────────────
# Load meta/info.json
# ──────────────────────────────────────────────
info_path = os.path.join(args.dataset_dir, "meta", "info.json")
with open(info_path, "r") as f:
    info = json.load(f)

fps            = args.fps_override if args.fps_override else info.get("fps", 10)
total_episodes = info.get("total_episodes", 1)
chunks_size    = info.get("chunks_size", 1000)
to_episode     = args.to_episode if args.to_episode is not None else total_episodes

logger.info("=" * 60)
logger.info(f"Dataset   : {args.dataset_dir}")
logger.info(f"Episodes  : {args.from_episode} -> {to_episode}  (total={total_episodes})")
logger.info(f"FPS       : {fps}")
logger.info(f"Data type : {args.data_type}")
if args.no_robot:
    logger.info("Robot     : DRY-RUN (no robot)")
else:
    logger.info(f"Robot     : {args.robot_ip}  [{args.robot_type}]")
    logger.info(f"Torque    : {'ON  (URrtdeTorque / target_pos)' if args.torque_mode else 'OFF (URrtde / servo)'}")
    logger.info(f"Gripper   : {'FastRobotiq2F85' if args.fast_gripper else 'Robotiq2F85 (standard)'}")
logger.info("=" * 60)

# ──────────────────────────────────────────────
# Resolve action / observation column names
# ──────────────────────────────────────────────
features     = info.get("features", {})
action_names = features.get("action",            {}).get("names", None)
state_names  = features.get("observation.state", {}).get("names", None)

def get_columns(df: pd.DataFrame, key: str, names=None):
    """Return a list of column names for a feature key."""
    if names:
        # Try direct names first (e.g. motor_0 … motor_6)
        cols = [c for c in names if c in df.columns]
        if cols:
            return cols
    # Fallback: look for columns like 'action_0', 'action_1', ...
    cols = [c for c in df.columns if c.startswith(key)]
    if cols:
        return sorted(cols)
    # Last resort: some datasets store as a single list column
    if key in df.columns:
        return [key]
    raise KeyError(f"Cannot find columns for key '{key}' in parquet. "
                   f"Available: {list(df.columns)}")

def natural_sort_key(text: str):
    """Sort flattened feature columns by embedded numbers."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text)]

def get_feature_columns(df: pd.DataFrame, key: str) -> list[str]:
    """Find a scalar/list feature stored either as one column or flattened columns."""
    if key in df.columns:
        return [key]
    prefix = f"{key}."
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        prefix = f"{key}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=natural_sort_key)

# ──────────────────────────────────────────────
# Helper: chunk + file index from episode index
# ──────────────────────────────────────────────
def episode_to_parquet_path(episode_idx: int) -> str:
    chunk_idx = episode_idx // chunks_size
    # file index within chunk
    file_idx  = episode_idx % chunks_size  # rough — real LeRobot uses episode_index column
    # glob the chunk folder and pick the right file
    chunk_dir = os.path.join(args.dataset_dir, "data", f"chunk-{chunk_idx:03d}")
    parquet_files = sorted(glob.glob(os.path.join(chunk_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {chunk_dir}")
    # Return all parquet files in this chunk (we'll filter by episode_index inside)
    return parquet_files

def load_episode_data(episode_idx: int) -> pd.DataFrame:
    """Load rows for a specific episode from the parquet dataset."""
    parquet_files = episode_to_parquet_path(episode_idx)
    frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "episode_index" in df.columns:
            df = df[df["episode_index"] == episode_idx]
            if len(df) > 0:
                frames.append(df)
    if not frames:
        raise ValueError(f"No data found for episode_index={episode_idx}")
    result = pd.concat(frames).sort_values("frame_index").reset_index(drop=True)
    return result

def extract_vector(row, col_names) -> np.ndarray:
    """Extract a float32 numpy vector from a DataFrame row."""
    values = []
    for c in col_names:
        v = row[c]
        # Some parquet stores list per cell (e.g. action stored as one list column)
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            values.extend(list(v))
        else:
            values.append(float(v))
    return np.array(values, dtype=np.float64)

def extract_array(row, col_names: list[str], expected_tail_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Extract a numpy array from one list-valued column or many flattened columns."""
    if not col_names:
        raise KeyError("No columns provided")
    if len(col_names) == 1:
        value = row[col_names[0]]
        arr = np.asarray(value, dtype=np.float32)
    else:
        arr = np.asarray([row[c] for c in col_names], dtype=np.float32)

    if expected_tail_shape is not None:
        tail_size = int(np.prod(expected_tail_shape))
        if arr.shape != expected_tail_shape and arr.size == tail_size:
            arr = arr.reshape(expected_tail_shape)
    return arr

# ──────────────────────────────────────────────
# Robot initialisation (skip in dry-run)
# ──────────────────────────────────────────────
ur             = None
gripper        = None
ik_solver      = None
tcp_transform  = None  # flange→TCP, used with IK when torque_mode + eef/tcp_quat

if not args.no_robot:
    from ur_teleop import make_robot, FastRobotiq2F85
    from airo_robots.grippers import Robotiq2F85
    from airo_spatial_algebra.se3 import SE3Container

    logger.info(f"Connecting to robot at {args.robot_ip} ...")
    ur, ik_solver = make_robot(
        ur_ip=args.robot_ip,
        robot_type=args.robot_type,
        torque_mode=args.torque_mode,
    )
    gripper = FastRobotiq2F85(args.robot_ip) if args.fast_gripper else Robotiq2F85(args.robot_ip)
    ur.gripper = gripper
    if args.torque_mode and args.data_type != "qpos":
        from config import Config
        tcp_transform = np.asarray(Config().TCP_TRANSFORM, dtype=float)
    logger.info("Robot connected.")

# ──────────────────────────────────────────────
# Camera / video helpers
# ──────────────────────────────────────────────
def get_video_keys() -> list[str]:
    """Return LeRobot video feature keys, preferring info.json and falling back to videos/."""
    keys = [
        key for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    ]
    if keys:
        return sorted(keys)

    videos_root = os.path.join(args.dataset_dir, "videos")
    if not os.path.isdir(videos_root):
        return []
    return sorted(
        key for key in os.listdir(videos_root)
        if os.path.isdir(os.path.join(videos_root, key))
    )

def find_video_for_episode(episode_idx: int, video_key: str = None) -> str | None:
    """Find the mp4 file for a given episode."""
    videos_root = os.path.join(args.dataset_dir, "videos")
    if not os.path.isdir(videos_root):
        return None
    if video_key is None:
        # pick first video key available
        keys = sorted(os.listdir(videos_root))
        if not keys:
            return None
        video_key = keys[0]
    chunk_idx = episode_idx // chunks_size
    file_idx  = episode_idx % chunks_size
    # Try formatted path from info.json template
    video_path_template = info.get("video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    video_path = os.path.join(
        args.dataset_dir,
        video_path_template.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx
        )
    )
    if os.path.isfile(video_path):
        return video_path
    # Fallback: glob
    pattern = os.path.join(videos_root, video_key,
                           f"chunk-{chunk_idx:03d}", "*.mp4")
    files = sorted(glob.glob(pattern))
    return files[file_idx] if file_idx < len(files) else None

class EpisodeVideoReader:
    """Small reader wrapper that prefers PyAV for LeRobot AV1 videos."""
    def __init__(self, path: str, prefer_pyav: bool = True, allow_opencv: bool = True):
        self.path = path
        self.cap = None
        self.container = None
        self.frames = None
        self.backend = None

        if prefer_pyav and self._open_pyav():
            return

        if not allow_opencv:
            raise RuntimeError("AV1 video requires PyAV; install package 'av' in this Python environment.")

        self.cap = cv2.VideoCapture(path)
        if self.cap.isOpened():
            self.backend = "opencv"
            return
        self.cap.release()
        self.cap = None

        if self._open_pyav():
            return

        raise RuntimeError(
            "Could not open video with PyAV or OpenCV. For AV1 LeRobot videos, "
            "install package 'av' in the environment used to run replay."
        )

    def _open_pyav(self) -> bool:
        try:
            import av
        except ImportError:
            return False

        try:
            self.container = av.open(self.path)
        except Exception:
            return False
        self.frames = iter(self.container.decode(video=0))
        self.backend = "pyav"
        return True

    def read(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return True, frame
            self.cap.release()
            self.cap = None
            if self._open_pyav():
                return self.read()
            return False, None

        try:
            frame = next(self.frames)
        except StopIteration:
            return False, None
        return True, frame.to_ndarray(format="bgr24")

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.container is not None:
            self.container.close()

def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (min(labeled.shape[1], 360), 28), (0, 0, 0), -1)
    cv2.putText(labeled, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return labeled

def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    width = max(1, int(frame.shape[1] * height / frame.shape[0]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def compose_frames(frames: list[tuple[str, np.ndarray]]) -> np.ndarray | None:
    valid = [(label, frame) for label, frame in frames if frame is not None]
    if not valid:
        return None
    target_h = min(frame.shape[0] for _, frame in valid)
    images = [add_label(resize_to_height(frame, target_h), label) for label, frame in valid]
    return np.concatenate(images, axis=1)

def ensure_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert camera/image arrays to uint8 without changing channel order."""
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating) and frame.size and np.nanmax(frame) <= 1.5:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)

def live_rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """RealSense get_rgb_image() returns RGB; OpenCV windows/blending use BGR."""
    frame = ensure_uint8(frame)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def dataset_bgr(frame: np.ndarray) -> np.ndarray:
    """EpisodeVideoReader returns BGR frames; normalize dtype/channels for blending."""
    frame = ensure_uint8(frame)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def resize_to_match(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_h, target_w = target.shape[:2]
    if frame.shape[:2] == (target_h, target_w):
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

def camera_name_from_video_key(video_key: str, fallback_idx: int) -> str:
    match = re.search(r"camera[_-]?(\d+)", video_key)
    if match:
        return f"camera_{int(match.group(1))}"
    return f"camera_{fallback_idx}"

def load_first_video_frames(episode_idx: int) -> list[tuple[str, np.ndarray]]:
    """Read the first frame for each LeRobot camera video in an episode."""
    first_frames = []
    for video_key in get_video_keys():
        vpath = find_video_for_episode(episode_idx, video_key)
        if not vpath:
            logger.warning(f"No video found for alignment key {video_key}.")
            continue
        reader = None
        try:
            video_info = features.get(video_key, {}).get("info", {})
            codec = str(video_info.get("video.codec", "")).lower()
            reader = EpisodeVideoReader(vpath, allow_opencv=(codec != "av1"))
            ret, frame = reader.read()
        except RuntimeError as exc:
            logger.warning(f"Could not open alignment video {video_key}: {exc}")
            continue
        finally:
            if reader is not None:
                reader.release()

        if ret:
            first_frames.append((video_key, dataset_bgr(frame)))
        else:
            logger.warning(f"Could not read first frame for alignment key {video_key}.")
    return first_frames

def create_alignment_cameras(camera_names: list[str]) -> dict[str, object]:
    """Create live RealSense cameras matching camera_0, camera_1, ... names."""
    import pyrealsense2 as rs
    from airo_camera_toolkit.cameras.realsense.realsense import Realsense

    context = rs.context()
    devices = context.query_devices()
    serial_numbers = []
    for i, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        logger.info(f"RealSense camera {i}: {name}  serial={serial}")
        serial_numbers.append(serial)

    if not serial_numbers:
        logger.warning("No RealSense cameras connected; skipping camera alignment.")
        return {}

    cameras = {}
    for camera_name in camera_names:
        match = re.search(r"camera_(\d+)$", camera_name)
        camera_idx = int(match.group(1)) if match else len(cameras)
        if camera_idx >= len(serial_numbers):
            logger.warning(
                f"{camera_name} requested but only {len(serial_numbers)} RealSense camera(s) found."
            )
            continue
        cameras[camera_name] = Realsense(
            fps=30,
            resolution=Realsense.RESOLUTION_480,
            enable_depth=True,
            enable_pointcloud=True,
            enable_hole_filling=False,
            serial_number=serial_numbers[camera_idx],
        )
    return cameras

def stop_alignment_cameras(cameras: dict[str, object]) -> None:
    for camera in cameras.values():
        if hasattr(camera, "pipeline"):
            camera.pipeline.stop()

def run_camera_alignment(episode_idx: int) -> None:
    """Overlay the start episode's first dataset frames with live cameras."""
    window_available, reason = opencv_window_status()
    if not window_available:
        logger.warning("OpenCV window backend is unavailable; skipping camera alignment.")
        logger.warning(f"Reason: {reason}")
        return

    first_frames = load_first_video_frames(episode_idx)
    if not first_frames:
        logger.warning("No dataset video frames available for camera alignment.")
        return

    frame_by_camera = {
        camera_name_from_video_key(video_key, idx): frame
        for idx, (video_key, frame) in enumerate(first_frames)
    }
    cameras = create_alignment_cameras(list(frame_by_camera.keys()))
    if not cameras:
        return

    logger.info("Camera alignment: dataset first frame | live camera | blended overlay")
    logger.info("Press 'q', Enter, or Esc when the environment is aligned.")

    window_names = [f"align-{camera_name}" for camera_name in frame_by_camera]
    try:
        while True:
            for camera_name, dataset_frame in frame_by_camera.items():
                camera = cameras.get(camera_name)
                if camera is None:
                    continue

                live_frame = live_rgb_to_bgr(camera.get_rgb_image())
                live_frame = resize_to_match(live_frame, dataset_frame)
                blended = cv2.addWeighted(dataset_frame, 0.5, live_frame, 0.5, 0)
                display = np.concatenate(
                    [
                        add_label(dataset_frame, "dataset first frame"),
                        add_label(live_frame, "live camera"),
                        add_label(blended, "blend"),
                    ],
                    axis=1,
                )
                cv2.imshow(f"align-{camera_name}", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27, 10, 13):
                break
    finally:
        for window_name in window_names:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        stop_alignment_cameras(cameras)

def opencv_window_status() -> tuple[bool, str]:
    """Return whether this OpenCV build/session can use HighGUI windows, plus a reason."""
    gui_line = next(
        (line.strip() for line in cv2.getBuildInformation().splitlines() if line.strip().startswith("GUI:")),
        "GUI: UNKNOWN",
    )
    if "NONE" in gui_line:
        return (
            False,
            f"{gui_line}. Install a GUI-enabled OpenCV package, e.g. opencv-python instead of opencv-python-headless.",
        )

    if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return (
            False,
            "DISPLAY/WAYLAND_DISPLAY is not set. Run from a local desktop session, "
            "forward X11, or export the active display, e.g. DISPLAY=:0.",
        )

    if os.environ.get("DISPLAY"):
        try:
            display_check = subprocess.run(
                ["xset", "q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return True, f"{gui_line}. DISPLAY is set; skipping external X server probe."
        if display_check.returncode != 0:
            detail = display_check.stderr.strip() or f"xset returned {display_check.returncode}"
            return False, f"{gui_line}, but the process cannot access DISPLAY={os.environ.get('DISPLAY')}: {detail}"

    return True, f"{gui_line}. OpenCV HighGUI should be available."

def opencv_windows_available() -> bool:
    """Return whether this OpenCV build/session can use HighGUI windows."""
    available, _ = opencv_window_status()
    return available

class VisualizationSink:
    """Show replay frames in a window, or save them when HighGUI is unavailable."""
    def __init__(self, episode_idx: int, fps_value: float):
        self.episode_idx = episode_idx
        self.fps = float(fps_value)
        self.mode = args.visualization_backend
        self.window_name = f"Episode {episode_idx} replay"
        self.writer = None
        self.output_path = None
        self.enabled = self.mode != "none"

        if not self.enabled:
            return

        if self.mode == "auto":
            self.mode = "window" if opencv_windows_available() else "video"
        elif self.mode == "window" and not opencv_windows_available():
            logger.warning("OpenCV window backend is unavailable; saving visualization mp4 instead.")
            self.mode = "video"

        if self.mode == "video":
            output_dir = args.replay_video_dir or os.path.join(args.dataset_dir, "replay_visualizations")
            os.makedirs(output_dir, exist_ok=True)
            self.output_path = os.path.join(output_dir, f"episode_{episode_idx:06d}_replay.mp4")
            logger.info(f"Visualization: saving to {self.output_path}")
        elif self.mode == "window":
            logger.info("Visualization: OpenCV window")

    def write(self, frame: np.ndarray):
        if not self.enabled or frame is None:
            return

        if self.mode == "window":
            try:
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
            except cv2.error:
                logger.warning("OpenCV window failed during replay; switching to saved mp4.")
                self.mode = "video"
                output_dir = args.replay_video_dir or os.path.join(args.dataset_dir, "replay_visualizations")
                os.makedirs(output_dir, exist_ok=True)
                self.output_path = os.path.join(output_dir, f"episode_{self.episode_idx:06d}_replay.mp4")
                self._write_video(frame)
            return

        if self.mode == "video":
            self._write_video(frame)

    def _write_video(self, frame: np.ndarray):
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                logger.warning(
                    f"Could not open VideoWriter for {self.output_path}; disabling visualization output."
                )
                self.enabled = False
                return

        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            logger.info(f"Saved visualization: {self.output_path}")
        if self.mode == "window":
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass

def render_tactile_frame(tactile_frame, height=480, width=320) -> np.ndarray:
    """
    Render a (41, 3) tactile frame as an arrow map.

    Layout matches the existing HDF5 visualizer:
    0-31 center matrix, 32-34 right side, 35-37 left side, 38-40 front.
    """
    tactile_frame = np.asarray(tactile_frame, dtype=np.float32)
    if tactile_frame.size == 123 and tactile_frame.shape != (41, 3):
        tactile_frame = tactile_frame.reshape(41, 3)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    arrow_scale = 0.01
    max_z_force = 1.0
    cx, cy = width // 2, height // 2
    spacing = max(20, min(width // 9, height // 12))

    def get_sensor_pos(idx):
        if idx < 32:
            r = idx // 8
            c = idx % 8
            start_x = cx - (4 * spacing) // 2 + spacing // 2
            start_y = cy - (8 * spacing) // 2 + spacing // 2
            return start_x + r * spacing, start_y + c * spacing
        if 32 <= idx <= 34:
            i = idx - 32
            return cx + (4 * spacing) // 2 + spacing, cy - (3 * spacing) // 2 + spacing // 2 + i * spacing
        if 35 <= idx <= 37:
            i = idx - 35
            return cx - (4 * spacing) // 2 - spacing, cy - (3 * spacing) // 2 + spacing // 2 + i * spacing
        if 38 <= idx <= 40:
            i = idx - 38
            return cx - (3 * spacing) // 2 + spacing // 2 + i * spacing, cy - (8 * spacing) // 2 - spacing
        return 0, 0

    for idx in range(min(41, tactile_frame.shape[0])):
        fx, fy, fz = tactile_frame[idx]
        px, py = get_sensor_pos(idx)
        px, py = int(np.clip(px, 0, width - 1)), int(np.clip(py, 0, height - 1))

        norm_z = int(np.clip(abs(float(fz)) / max_z_force, 0, 1) * 255)
        color = cv2.applyColorMap(np.array([[norm_z]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
        cv2.circle(canvas, (px, py), 5, (100, 100, 100), 2, cv2.LINE_AA)

        if abs(float(fx)) > 0.1 or abs(float(fy)) > 0.1:
            end_x = int(np.clip(px + fx * arrow_scale, 0, width - 1))
            end_y = int(np.clip(py + fy * arrow_scale, 0, height - 1))
            cv2.arrowedLine(canvas, (px, py), (end_x, end_y), (255, 255, 255), 2, tipLength=0.3)
        elif abs(float(fz)) > 0.5:
            cv2.circle(canvas, (px, py), max(1, int(abs(float(fz)))), color, 1, cv2.LINE_AA)

    cv2.putText(canvas, "Front", (cx - 22, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas

# ──────────────────────────────────────────────
# Move robot to initial pose
# ──────────────────────────────────────────────
def move_initial_gripper(width: float) -> None:
    """Move gripper at episode start without blocking replay/alignment forever."""
    action = ur.gripper.move(float(width))
    if args.initial_gripper_timeout <= 0:
        logger.info("Initial gripper command sent; not waiting (--initial_gripper_timeout <= 0).")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status = action.wait(timeout=args.initial_gripper_timeout, sleep_resolution=0.02)
    if getattr(status, "name", "") == "TIMEOUT":
        logger.warning(
            f"  ⚠  Initial gripper wait timed out after {args.initial_gripper_timeout:.1f}s; continuing."
        )

def move_to_initial_pose(pose: np.ndarray):
    if args.no_robot:
        logger.info(f"[dry-run] move to initial pose: {np.round(pose, 4)}")
        return

    # Torque backend (URrtdeTorque): blocking moveJ via tmp_move, then RTDE torque loop;
    # do not use servo_to_joint_configuration / servo_to_tcp_pose.
    if args.torque_mode:
        from airo_spatial_algebra.se3 import SE3Container
        if args.data_type == "qpos":
            ur.tmp_move(np.asarray(pose[:6], dtype=float))
            move_initial_gripper(pose[6])
        elif args.data_type == "eef":
            tcp = SE3Container.from_euler_angles_and_translation(pose[:3], pose[3:6])
            q_seed = np.asarray(ur.get_cached_joint_configuration(), dtype=float)
            sol = ik_solver.inverse_kinematics_closest_with_tcp(
                tcp.homogeneous_matrix, tcp_transform, *q_seed
            )
            if not sol:
                raise RuntimeError("IK failed for initial pose (eef)")
            ur.tmp_move(np.asarray(sol[0], dtype=float))
            move_initial_gripper(pose[6])
        elif args.data_type == "tcp_quat":
            tcp = SE3Container.from_quaternion_and_translation(pose[:4], pose[4:7])
            q_seed = np.asarray(ur.get_cached_joint_configuration(), dtype=float)
            sol = ik_solver.inverse_kinematics_closest_with_tcp(
                tcp.homogeneous_matrix, tcp_transform, *q_seed
            )
            if not sol:
                raise RuntimeError("IK failed for initial pose (tcp_quat)")
            ur.tmp_move(np.asarray(sol[0], dtype=float))
            move_initial_gripper(pose[7])
        return

    if args.data_type == "qpos":
        ur.servo_to_joint_configuration(pose[:6], 0.5)
        move_initial_gripper(pose[6])

    elif args.data_type == "eef":
        from airo_spatial_algebra.se3 import SE3Container
        tcp = SE3Container.from_euler_angles_and_translation(
            pose[:3], pose[3:6])
        ur.servo_to_tcp_pose(tcp.homogeneous_matrix, 0.5)
        move_initial_gripper(pose[6])

    elif args.data_type == "tcp_quat":
        from airo_spatial_algebra.se3 import SE3Container
        tcp = SE3Container.from_quaternion_and_translation(
            pose[:4], pose[4:7])
        ur.servo_to_tcp_pose(tcp.homogeneous_matrix, 0.5)
        move_initial_gripper(pose[7])

# ──────────────────────────────────────────────
# Execute one step
# ──────────────────────────────────────────────
def execute_step(pose: np.ndarray, dt: float):
    if args.no_robot:
        return
    if args.data_type == "qpos":
        ur.gripper.move(pose[6], 0.1)
        ur.servo_to_joint_configuration(pose[:6], dt)

    elif args.data_type == "eef":
        from airo_spatial_algebra.se3 import SE3Container
        ur.gripper.move(pose[6], 0.1)
        tcp = SE3Container.from_euler_angles_and_translation(
            pose[:3], pose[3:6])
        ur.servo_to_tcp_pose(tcp.homogeneous_matrix, dt)

    elif args.data_type == "tcp_quat":
        from airo_spatial_algebra.se3 import SE3Container
        ur.gripper.move(pose[7], 0.1)
        tcp = SE3Container.from_quaternion_and_translation(
            pose[:4], pose[4:7])
        ur.servo_to_tcp_pose(tcp.homogeneous_matrix, dt)


def execute_step_torque(pose: np.ndarray):
    """Update URrtdeTorque shared target at dataset rate; inner worker runs ~500 Hz PD torque."""
    from airo_spatial_algebra.se3 import SE3Container

    if args.data_type == "qpos":
        ur.target_pos = np.asarray(pose[:6], dtype=float)
        ur.gripper.move(pose[6], 0.1)

    elif args.data_type == "eef":
        ur.gripper.move(pose[6], 0.1)
        tcp = SE3Container.from_euler_angles_and_translation(pose[:3], pose[3:6])
        q_seed = np.asarray(ur.get_cached_joint_configuration(), dtype=float)
        sol = ik_solver.inverse_kinematics_closest_with_tcp(
            tcp.homogeneous_matrix, tcp_transform, *q_seed
        )
        if sol:
            ur.target_pos = np.asarray(sol[0], dtype=float)

    elif args.data_type == "tcp_quat":
        ur.gripper.move(pose[7], 0.1)
        tcp = SE3Container.from_quaternion_and_translation(pose[:4], pose[4:7])
        q_seed = np.asarray(ur.get_cached_joint_configuration(), dtype=float)
        sol = ik_solver.inverse_kinematics_closest_with_tcp(
            tcp.homogeneous_matrix, tcp_transform, *q_seed
        )
        if sol:
            ur.target_pos = np.asarray(sol[0], dtype=float)


# ──────────────────────────────────────────────
# Main replay loop
# ──────────────────────────────────────────────
dt = 1.0 / fps
state_cols = None
action_cols = None
tactile_cols = []
alignment_done = False

for ep_idx in range(args.from_episode, to_episode):
    logger.info("-" * 50)
    logger.info(f"Loading episode {ep_idx} ...")

    try:
        df = load_episode_data(ep_idx)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Skipping episode {ep_idx}: {e}")
        continue

    # Detect column names from the first successfully loaded episode.
    if state_cols is None or action_cols is None:
        # Try to figure out which columns hold the state
        try:
            state_cols = get_columns(df, "observation.state", state_names)
        except KeyError:
            # fall back to action columns as state proxy
            state_cols = get_columns(df, "action", action_names)
        action_cols = get_columns(df, "action", action_names)
        tactile_cols = get_feature_columns(df, "observation.tactile")
        logger.info(f"State columns  : {state_cols}")
        logger.info(f"Action columns : {action_cols}")
        if args.tactile:
            logger.info(f"Tactile columns: {tactile_cols if tactile_cols else 'not found'}")

    n_steps = len(df)
    logger.info(f"Frames  : {n_steps}  |  Duration: {n_steps/fps:.1f}s")

    # ── Move to start pose ──
    initial_state = extract_vector(df.iloc[0], state_cols)
    logger.info("Moving to initial pose ...")
    move_to_initial_pose(initial_state)

    if args.align_camera and not alignment_done:
        logger.info(f"Loading first camera frame from episode {ep_idx} for environment alignment ...")
        run_camera_alignment(ep_idx)
        alignment_done = True

    # ── Optional: open videos ──
    video_readers = []
    if args.show_video:
        for video_key in get_video_keys():
            vpath = find_video_for_episode(ep_idx, video_key)
            if not vpath:
                logger.warning(f"No video found for {video_key}.")
                continue
            try:
                video_info = features.get(video_key, {}).get("info", {})
                codec = str(video_info.get("video.codec", "")).lower()
                reader = EpisodeVideoReader(vpath, allow_opencv=(codec != "av1"))
            except RuntimeError as exc:
                logger.warning(f"Could not open {video_key}: {exc}")
                continue
            video_readers.append((video_key, reader))
            logger.info(f"Video: {video_key} -> {vpath} [{reader.backend}]")
        if not video_readers:
            logger.warning("No readable videos found for this episode.")

    if args.tactile and not tactile_cols:
        logger.warning("--tactile requested but observation.tactile was not found in parquet.")

    visualization_sink = None
    if args.show_video or args.tactile:
        visualization_sink = VisualizationSink(ep_idx, fps)

    input(f"\n  Press Enter to start replay  [episode {ep_idx}] …\n")

    # ── Replay steps ──
    for step_idx in range(n_steps):
        t0 = time.time()

        action = extract_vector(df.iloc[step_idx], action_cols)

        display_frames = []
        for video_key, reader in video_readers:
            ret, frame = reader.read()
            if ret:
                display_frames.append((video_key, frame))

        if args.tactile and tactile_cols:
            tactile = extract_array(df.iloc[step_idx], tactile_cols, (41, 3))
            display_frames.append(("observation.tactile", render_tactile_frame(tactile)))

        display = compose_frames(display_frames)
        if visualization_sink is not None:
            visualization_sink.write(display)

        if args.torque_mode and not args.no_robot:
            execute_step_torque(action)
        else:
            execute_step(action, dt)

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

        actual_hz = 1.0 / (time.time() - t0)
        sys.stdout.write(f"\r  step {step_idx+1:04d}/{n_steps}  |  {actual_hz:5.1f} Hz")
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    for _, reader in video_readers:
        reader.release()
    if visualization_sink is not None:
        visualization_sink.close()

    logger.info(f"Episode {ep_idx} replay finished.")

logger.info("=" * 60)
logger.info("All episodes replayed.")
