"""Collect RealMan freedrive demonstrations without cameras or VR hardware.

The LeRobot dataset contains measured joint configurations, TCP poses, and the
integrated six-axis force/torque sensor. The required LeRobot ``action`` field
mirrors the measured joints because freedrive has no commanded robot target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

import utils
from config import Config
from dataset import DatasetRecorder
from force_filter import WrenchFilter
from robot_backend import RobotBackend, make_robot_backend
from visualizer_config import VisualizerConfig


DEFAULT_DATASET_DIR = "./datasets/realman_teach"
FREEDRIVE_SENSITIVITY = 99


@dataclass(frozen=True)
class TeachSample:
    joints: np.ndarray
    tcp_pose: np.ndarray
    tcp_vector: np.ndarray
    wrench: np.ndarray
    timestamp_ns: int


def tcp_pose_to_vector(
    tcp_pose: np.ndarray,
    last_quaternion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a homogeneous TCP pose to LeRobot [qx,qy,qz,qw,x,y,z]."""
    pose = np.asarray(tcp_pose, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Expected a finite 4x4 TCP pose, got {pose.shape}.")
    quaternion = utils.quat_cal(pose[:3, :3], last_quaternion)
    vector = np.concatenate((quaternion, pose[:3, 3])).astype(np.float32)
    return vector, quaternion


class RealManTeachCollector:
    """Single-threaded freedrive sensor and LeRobot episode controller."""

    def __init__(
        self,
        cfg: Config,
        *,
        backend: RobotBackend | None = None,
        dataset: DatasetRecorder | None = None,
        visualizer_handle: Any | None = None,
    ) -> None:
        if cfg.ROBOT_TYPE != "realman":
            raise ValueError("realman_teachcollect.py requires ROBOT_TYPE='realman'.")
        if cfg.DATASET_TYPE != "l":
            raise ValueError("realman_teachcollect.py records LeRobot datasets only.")
        if cfg.DATA_TYPE != "both":
            raise ValueError("DATA_TYPE must be 'both' to store joints and TCP pose.")

        self.cfg = cfg
        self.backend = make_robot_backend(cfg) if backend is None else backend
        self.dataset = dataset
        self.visualizer_handle = visualizer_handle
        self.collecting = False
        self.freedrive_active = False
        self.last_quaternion: np.ndarray | None = None
        self.wrench_filter = WrenchFilter(
            moving_average_window=cfg.FORCE_MOVING_AVERAGE_WINDOW,
            low_pass_alpha=cfg.FORCE_LOW_PASS_ALPHA,
        )

        try:
            self._validate_backend()
            if self.dataset is None:
                self.dataset = DatasetRecorder(
                    camera_num=0,
                    robot_dof=self.backend.dof,
                    robot_type=self.backend.dataset_robot_type,
                    force_collect=True,
                    torque_collect=True,
                    gripper=False,
                    config=cfg,
                )
        except Exception:
            if backend is None:
                self.backend.cleanup()
            raise

    def _validate_backend(self) -> None:
        if self.backend.name != "realman":
            raise TypeError("The selected robot backend is not RealMan.")
        if not self.backend.supports_freedrive:
            raise RuntimeError("The RealMan backend does not expose freedrive mode.")
        if not self.backend.supports_force:
            raise RuntimeError(
                "The connected RealMan robot does not expose a force sensor."
            )

    def read_sample(self) -> TeachSample:
        timestamp_ns = time.monotonic_ns()
        joints = np.asarray(self.backend.get_joint_configuration(), dtype=float)
        tcp_pose = np.asarray(self.backend.get_tcp_pose(), dtype=float)
        raw_wrench = self.backend.get_tcp_force()
        if raw_wrench is None:
            raise RuntimeError("RealMan force sensor returned no wrench.")
        raw_wrench = np.asarray(raw_wrench, dtype=float).reshape(-1)

        if joints.shape != (self.backend.dof,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(
                f"RealMan returned invalid joints with shape {joints.shape}."
            )
        if raw_wrench.shape != (6,) or not np.all(np.isfinite(raw_wrench)):
            raise RuntimeError(
                f"RealMan returned an invalid wrench with shape {raw_wrench.shape}."
            )

        tcp_vector, self.last_quaternion = tcp_pose_to_vector(
            tcp_pose,
            self.last_quaternion,
        )
        wrench = self.wrench_filter.process(raw_wrench)
        return TeachSample(
            joints=joints,
            tcp_pose=tcp_pose,
            tcp_vector=tcp_vector,
            wrench=wrench,
            timestamp_ns=timestamp_ns,
        )

    def start_recording(self) -> bool:
        if self.collecting:
            return False
        self.collecting = True
        utils.logger.info(f"Recording episode {self.dataset.recorded_episodes}...")
        return True

    def stop_recording(self) -> bool:
        if not self.collecting:
            return False
        self.collecting = False
        if self.dataset.collect_step <= 0:
            utils.logger.warning("No frames collected; episode was not saved.")
            return False
        self.dataset.data_export(None)
        self.dataset._reset_data_dict()
        utils.logger.info(
            f"Saved episode {self.dataset.recorded_episodes - 1}."
        )
        return True

    def delete_last_episode(self) -> bool:
        if self.collecting and self.dataset.collect_step <= 0:
            self.collecting = False
            utils.logger.info("Canceled the empty active episode.")
            return True
        self.collecting = False
        removed = self.dataset.rollback_last_episode()
        if removed:
            utils.logger.info(
                "Discarded the current episode or deleted the latest saved episode."
            )
        return removed

    def handle_visualizer_commands(self) -> None:
        if self.visualizer_handle is None:
            return
        for command in self.visualizer_handle.drain_commands():
            name = command.get("command")
            if name == "toggle_recording":
                if self.collecting:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif name == "rollback_last_episode":
                self.delete_last_episode()

    def record_sample(self, sample: TeachSample) -> None:
        timestamps = {
            "collect_timestamp_ns": np.array(sample.timestamp_ns, dtype=np.int64),
            "robot_state_timestamp_ns": np.array(sample.timestamp_ns, dtype=np.int64),
            # Freedrive has no command stream; action mirrors measured joints.
            "robot_action_timestamp_ns": np.array(sample.timestamp_ns, dtype=np.int64),
            "tcp_pose": sample.tcp_vector,
        }
        self.dataset.data_collection(
            state=sample.joints,
            action=sample.joints,
            camera_images={},
            wrench_data=sample.wrench,
            extra_data=timestamps,
        )

    def publish_sample(self, sample: TeachSample, error: str = "") -> None:
        if self.visualizer_handle is None:
            return
        self.visualizer_handle.publish(
            {
                "timestamp": time.monotonic(),
                "wrench": sample.wrench.copy(),
                "joints": sample.joints.copy(),
                "tcp_translation": sample.tcp_pose[:3, 3].copy(),
                "camera_count": 0,
                "dataset": self.dataset.recording_status(collecting=self.collecting),
                "source_label": "RealMan freedrive",
                "connected": not error,
                "error": error,
            }
        )

    def enable_freedrive(self) -> None:
        self.backend.set_freedrive_sensitivity(FREEDRIVE_SENSITIVITY)
        utils.logger.info(
            "RealMan drag-teach sensitivity set to %d.",
            FREEDRIVE_SENSITIVITY,
        )
        self.backend.start_freedrive()
        self.freedrive_active = True

    def run(self) -> None:
        period = 1.0 / self.cfg.COLLECT_RATE
        next_tick = time.monotonic()
        try:
            self.enable_freedrive()
            utils.logger.info(
                "Freedrive enabled. Use Start record / Save episode / Undo episode "
                "in the visualizer; press Ctrl-C to stop."
            )
            while True:
                if (
                    self.visualizer_handle is not None
                    and not self.visualizer_handle.process.is_alive()
                ):
                    utils.logger.info("Visualizer closed; stopping the collector.")
                    break
                self.handle_visualizer_commands()
                sample = self.read_sample()
                if self.collecting:
                    self.record_sample(sample)
                self.publish_sample(sample)

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except KeyboardInterrupt:
            utils.logger.info("Stopping freedrive collector...")
        finally:
            self.close()

    def close(self) -> None:
        close_error: Exception | None = None
        if self.freedrive_active:
            try:
                self.backend.stop_freedrive()
            except Exception as exc:
                close_error = exc
                utils.logger.exception("Failed to stop RealMan freedrive")
            finally:
                self.freedrive_active = False
        try:
            if self.collecting and self.dataset.collect_step:
                utils.logger.info("Saving the active episode before exit.")
                self.stop_recording()
            self.dataset.close()
        except Exception as exc:
            if close_error is None:
                close_error = exc
            utils.logger.exception("Failed to finalize the teaching dataset")
        finally:
            if self.visualizer_handle is not None:
                self.visualizer_handle.close()
            try:
                self.backend.cleanup()
            except Exception as exc:
                if close_error is None:
                    close_error = exc
                utils.logger.exception("Failed to close the RealMan connection")
        if close_error is not None:
            raise close_error


def parse_args() -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Collect joint, TCP, and force data while RealMan is in freedrive."
    )
    parser.add_argument("--robot-ip", default=defaults.ROBOT_IP)
    parser.add_argument("--port", type=int, default=defaults.REALMAN_PORT)
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Dataset base path; the recorder appends '_lero'.",
    )
    parser.add_argument("--task", default="realman_freedrive_demonstration")
    parser.add_argument("--fps", type=int, default=defaults.COLLECT_RATE)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    cfg = Config(
        ROBOT_TYPE="realman",
        ROBOT_IP=args.robot_ip,
        REALMAN_PORT=args.port,
        GRIPPER=False,
        TORQUE_MODE=False,
        DATASET_TYPE="l",
        DATA_TYPE="both",
        DATASET_DIR=args.dataset_dir,
        TASK_NAME=args.task,
        COLLECT_RATE=args.fps,
        PUSH_TO_HUB=False,
        FORCE_COLLECT=True,
        TORQUE_COLLECT=True,
        DEPTH_INFO_ENABLE=False,
        TACTILE_ENABLE=False,
        TACTILE_TRANSFER=False,
    )
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    viz_cfg = VisualizerConfig()

    from visualizer import start_visualizer

    visualizer_handle = start_visualizer(
        hz=viz_cfg.HZ,
        window_s=viz_cfg.WINDOW_S,
        title="RealMan Freedrive Dataset Collector",
        force_panel_range=viz_cfg.FORCE_PANEL_RANGE,
        camera_num=0,
        show_rollback_button=True,
        show_record_button=True,
    )
    try:
        collector = RealManTeachCollector(cfg, visualizer_handle=visualizer_handle)
    except Exception:
        visualizer_handle.close()
        raise
    collector.run()


if __name__ == "__main__":
    main()
