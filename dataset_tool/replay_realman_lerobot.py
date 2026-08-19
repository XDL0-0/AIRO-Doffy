"""Safely replay joint or TCP trajectories from a RealMan LeRobot dataset."""

from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


logger = logging.getLogger(__name__)
REALMAN_DOF = 7
DEFAULT_DATASET_DIR = "./datasets/WRM_grasp_lero"


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
            raise ValueError("This episode trajectory contains TCP targets, not joints.")
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
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        if self.fps <= 0.0:
            raise ValueError(f"Dataset fps must be positive, got {self.fps}.")
        if self.total_episodes <= 0:
            raise ValueError("Dataset contains no episodes.")
        robot_type = str(self.info.get("robot_type", "")).lower()
        if robot_type and robot_type != "realman":
            raise ValueError(
                f"Dataset robot_type is '{robot_type}', not 'realman'."
            )

        feature = self.info.get("features", {}).get(self.source)
        if not isinstance(feature, dict):
            raise ValueError(f"Dataset does not contain feature '{self.source}'.")
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

    def _load_episode_frames(self, episode_index: int) -> pd.DataFrame:
        chunk_index = episode_index // self.chunks_size
        exact_path = self.root / self.data_path.format(
            chunk_index=chunk_index,
            file_index=episode_index % self.chunks_size,
        )
        # Parquet files may hold several episodes (chunked by frames), so scan
        # the whole chunk directory rather than assuming one file per episode.
        paths = sorted(
            (self.root / "data" / f"chunk-{chunk_index:03d}").glob("*.parquet")
        )
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
        if not matches:
            raise ValueError(f"No rows found for episode {episode_index}.")
        return pd.concat(matches, ignore_index=True)

    @staticmethod
    def _extract_vectors(frames: pd.DataFrame, key: str) -> np.ndarray:
        if key in frames:
            try:
                return np.stack(
                    [np.asarray(value, dtype=float).reshape(-1) for value in frames[key]]
                )
            except ValueError as exc:
                raise ValueError(f"Feature '{key}' contains inconsistent vectors.") from exc

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
                "Episode %d: %d joint frames, %.2f s, max joint speed "
                "%.3f rad/s.",
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
