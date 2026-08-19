"""Teach a RealMan path, replay it, and collect synchronized observations.

Drag-teach only stores an in-memory joint trajectory. Dataset frames are
captured later while the robot replays that trajectory, so ``observation.state``
contains measured joints and ``action`` contains the corresponding taught joint
target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Any

import numpy as np

import utils
from config import Config
from dataset import DatasetRecorder
from data_recording import DataRecordingService, RecordingControl, RecordingFrame
from force_filter import WrenchFilter
from robot_backend import RobotBackend, make_robot_backend
from visualizer_config import VisualizerConfig


FREEDRIVE_SENSITIVITY = 99
TEACH_MOTION_THRESHOLD_RAD = float(np.deg2rad(0.1))


class TeachState(str, Enum):
    IDLE = "idle"
    TEACHING = "teaching"
    READY = "ready"
    REPLAYING = "replaying"
    MOVING_INITIAL = "moving_initial"


@dataclass(frozen=True)
class TeachSample:
    joints: np.ndarray
    tcp_pose: np.ndarray
    tcp_vector: np.ndarray
    wrench: np.ndarray
    timestamp_ns: int
    force_valid: bool = True


def create_camera_manager(cfg: Config) -> Any:
    """Create local RealSense capture without UDP, WebRTC, or VR resources."""
    from realsense_camera import RealSenseCameraManager

    return RealSenseCameraManager(config=cfg)


def _camera_snapshot(
    camera_manager: Any | None,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, np.ndarray] | None]:
    if camera_manager is None:
        return {}, {}, None

    with camera_manager._lock:
        images = dict(camera_manager.camera_images)
        timestamps = dict(camera_manager.camera_image_timestamps_ns)
        depth_images = (
            dict(camera_manager.depth_images) if camera_manager.depth_mode else None
        )
    return images, timestamps, depth_images


def _visualizer_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    previews: dict[str, np.ndarray] = {}
    for name, image in sorted(images.items()):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            continue
        step_y = max(1, image.shape[0] // 240)
        step_x = max(1, image.shape[1] // 320)
        previews[name] = image[::step_y, ::step_x, :3].copy()
    return previews


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
    """Teach/replay workflow and LeRobot episode controller."""

    def __init__(
        self,
        cfg: Config,
        *,
        backend: RobotBackend | None = None,
        dataset: DatasetRecorder | None = None,
        camera_manager: Any | None = None,
        visualizer_handle: Any | None = None,
        beaver_reader: Any | None = None,
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
        self.camera_manager = camera_manager
        self.visualizer_handle = visualizer_handle
        self.beaver_reader = beaver_reader
        self.freedrive_active = False
        self.teach_state = TeachState.IDLE
        self.taught_trajectory: list[np.ndarray] = []
        self._teach_start_joints: np.ndarray | None = None
        self.workflow_message = "Press Teach to begin drag teaching."
        self._last_raw_wrench: np.ndarray | None = None
        self._force_data_available = False
        self._last_force_warning_time = 0.0
        self.last_quaternion: np.ndarray | None = None
        self._sample_lock = threading.Lock()
        self._latest_sample: TeachSample | None = None
        self._last_recorded_timestamp_ns = 0
        self._recording_stop_event: threading.Event | None = None
        self._recording_threads: list[threading.Thread] = []
        self.wrench_filter = WrenchFilter(
            moving_average_window=cfg.FORCE_MOVING_AVERAGE_WINDOW,
            low_pass_alpha=cfg.FORCE_LOW_PASS_ALPHA,
        )

        try:
            self._validate_backend()
            if self.dataset is None:
                self.dataset = DatasetRecorder(
                    camera_num=(
                        0 if self.camera_manager is None else self.camera_manager.camera_num
                    ),
                    robot_dof=self.backend.dof,
                    robot_type=self.backend.dataset_robot_type,
                    force_collect=True,
                    torque_collect=True,
                    gripper=False,
                    config=cfg,
                )
            self.recording_service = DataRecordingService(
                self.dataset,
                self._recording_frame,
                cfg.COLLECT_RATE,
                control=RecordingControl(),
                export_context=None,
                thread_name="realman-teach-dataset",
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

        if joints.shape != (self.backend.dof,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(
                f"RealMan returned invalid joints with shape {joints.shape}."
            )
        force_valid = raw_wrench is not None
        if force_valid:
            raw_wrench = np.asarray(raw_wrench, dtype=float).reshape(-1)
            force_valid = raw_wrench.shape == (6,) and np.all(np.isfinite(raw_wrench))
        if force_valid:
            self._last_raw_wrench = raw_wrench.copy()
            self._force_data_available = True
        else:
            self._force_data_available = False
            if self.teach_state is TeachState.REPLAYING:
                raise RuntimeError(
                    "RealMan force data became unavailable during replay collection."
                )
            now = time.monotonic()
            if now - self._last_force_warning_time >= 1.0:
                fallback = (
                    "last valid wrench"
                    if self._last_raw_wrench is not None
                    else "zeros"
                )
                utils.logger.warning(
                    "RealMan force data unavailable; using %s until the sensor recovers.",
                    fallback,
                )
                self._last_force_warning_time = now
            raw_wrench = (
                self._last_raw_wrench.copy()
                if self._last_raw_wrench is not None
                else np.zeros(6, dtype=float)
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
            force_valid=force_valid,
        )

    def delete_last_episode(self) -> bool:
        return self.recording_service.request_rollback()

    def handle_visualizer_commands(self) -> None:
        if self.visualizer_handle is None:
            return
        for command in self.visualizer_handle.drain_commands():
            name = command.get("command")
            if name == "toggle_teach":
                if self.teach_state is TeachState.TEACHING:
                    self.end_teach()
                else:
                    self.start_teach()
            elif name == "reteach":
                self.reteach()
            elif name == "replay_collect":
                try:
                    self.replay_collect()
                except Exception as exc:
                    self.workflow_message = f"Replay failed: {exc}"
                    utils.logger.exception("Replay collection failed")
            elif name == "initial_pose":
                self.move_to_initial_pose()
            elif name == "rollback_last_episode":
                self.delete_last_episode()
        self._process_pending_if_embedded()

    @property
    def collecting(self) -> bool:
        return self.recording_service.collecting

    def _frame_from_sample(
        self,
        sample: TeachSample,
        action: np.ndarray | None = None,
    ) -> RecordingFrame:
        images, camera_timestamps, depth_images = _camera_snapshot(
            self.camera_manager
        )
        beaver = (
            self.beaver_reader.snapshot()
            if self.beaver_reader is not None
            else None
        )
        return RecordingFrame(
            state=sample.joints,
            action=(
                sample.joints
                if action is None
                else np.asarray(action, dtype=float).copy()
            ),
            camera_images=images,
            wrench_data=sample.wrench,
            depth_images=depth_images,
            extra_data={
                "collect_timestamp_ns": np.array(sample.timestamp_ns, dtype=np.int64),
                "robot_state_timestamp_ns": np.array(sample.timestamp_ns, dtype=np.int64),
                "robot_action_timestamp_ns": np.array(sample.timestamp_ns, dtype=np.int64),
                "camera_timestamps_ns": camera_timestamps,
                "beaver_timestamp_ns": np.array(
                    0 if beaver is None else beaver.timestamp_ns,
                    dtype=np.int64,
                ),
                "tcp_pose": sample.tcp_vector,
            },
            beaver_data=beaver,
        )

    def _recording_frame(self) -> RecordingFrame | None:
        with self._sample_lock:
            sample = self._latest_sample
            if (
                sample is None
                or sample.timestamp_ns == self._last_recorded_timestamp_ns
            ):
                return None
            self._last_recorded_timestamp_ns = sample.timestamp_ns
        return self._frame_from_sample(sample)

    def record_sample(
        self,
        sample: TeachSample,
        action: np.ndarray | None = None,
    ) -> None:
        self.recording_service.record_frame(
            self._frame_from_sample(sample, action=action)
        )

    def publish_sample(self, sample: TeachSample, error: str = "") -> None:
        if self.visualizer_handle is None:
            return
        images, _camera_timestamps, _depth_images = _camera_snapshot(
            self.camera_manager
        )
        self.visualizer_handle.publish(
            {
                "timestamp": time.monotonic(),
                "wrench": sample.wrench.copy(),
                "joints": sample.joints.copy(),
                "tcp_translation": sample.tcp_pose[:3, 3].copy(),
                "images": _visualizer_images(images),
                "camera_count": (
                    0 if self.camera_manager is None else self.camera_manager.camera_num
                ),
                "dataset": self.recording_service.recording_status(),
                "teach": self.workflow_status(),
                **(
                    {"beaver": self.beaver_reader.visualizer_payload()}
                    if self.beaver_reader is not None
                    else {}
                ),
                "source_label": "RealMan teach/replay",
                "connected": not error and sample.force_valid,
                "error": (
                    error
                    or (
                        "Force data temporarily unavailable; waiting for a fresh sample."
                        if not sample.force_valid
                        else ""
                    )
                ),
            }
        )

    def enable_freedrive(self) -> None:
        if self.freedrive_active:
            return
        self.backend.set_freedrive_sensitivity(FREEDRIVE_SENSITIVITY)
        utils.logger.info(
            "RealMan drag-teach sensitivity set to %d.",
            FREEDRIVE_SENSITIVITY,
        )
        self.backend.start_freedrive()
        self.freedrive_active = True

    def disable_freedrive(self) -> None:
        if not self.freedrive_active:
            return
        self.backend.stop_freedrive()
        self.freedrive_active = False

    def workflow_status(self) -> dict[str, object]:
        state = self.teach_state.value
        return {
            "state": state,
            "trajectory_frames": len(self.taught_trajectory),
            "message": self.workflow_message,
            "teach_enabled": self.teach_state
            in {
                TeachState.IDLE,
                TeachState.TEACHING,
                TeachState.READY,
            },
            "reteach_enabled": self.teach_state in {TeachState.TEACHING, TeachState.READY},
            "replay_enabled": (
                self.teach_state is TeachState.READY
                and bool(self.taught_trajectory)
                and self._force_data_available
            ),
            "force_available": self._force_data_available,
            "initial_pose_enabled": self.teach_state in {TeachState.IDLE, TeachState.READY},
        }

    def start_teach(self) -> bool:
        if self.teach_state not in {TeachState.IDLE, TeachState.READY}:
            return False
        replacing_trajectory = bool(self.taught_trajectory)
        self.enable_freedrive()
        self.taught_trajectory.clear()
        with self._sample_lock:
            latest_sample = self._latest_sample
        self._teach_start_joints = (
            self.backend.get_joint_configuration()
            if latest_sample is None
            else latest_sample.joints
        )
        self._teach_start_joints = np.asarray(
            self._teach_start_joints, dtype=float
        ).copy()
        self.taught_trajectory.append(self._teach_start_joints.copy())
        self.teach_state = TeachState.TEACHING
        self.workflow_message = (
            "Existing trajectory cleared. Drag the robot, then press End Teach."
            if replacing_trajectory
            else "Drag the robot, then press End Teach."
        )
        utils.logger.info(self.workflow_message)
        return True

    def reteach(self) -> bool:
        if self.teach_state not in {TeachState.TEACHING, TeachState.READY}:
            return False
        self.disable_freedrive()
        self.taught_trajectory.clear()
        self._teach_start_joints = None
        self.teach_state = TeachState.IDLE
        self.workflow_message = (
            "Trajectory is cleared, please press Teach to create a new one."
        )
        utils.logger.info(self.workflow_message)
        return True

    def capture_teach_sample(self, sample: TeachSample) -> bool:
        if self.teach_state is not TeachState.TEACHING:
            return False
        if self._teach_start_joints is None:
            self._teach_start_joints = sample.joints.copy()
            self.taught_trajectory.append(self._teach_start_joints.copy())
        self.taught_trajectory.append(sample.joints.copy())
        return True

    @staticmethod
    def _trim_stationary_trajectory_edges(
        trajectory: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Trim idle samples at both ends while preserving the full path within."""
        if len(trajectory) < 2:
            return []
        joints = np.stack(trajectory)
        step_motion = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
        moving_steps = np.flatnonzero(step_motion >= TEACH_MOTION_THRESHOLD_RAD)
        if moving_steps.size == 0:
            return []
        start = int(moving_steps[0])
        stop = int(moving_steps[-1]) + 2
        return [sample.copy() for sample in trajectory[start:stop]]

    def end_teach(self) -> bool:
        if self.teach_state is not TeachState.TEACHING:
            return False
        self.disable_freedrive()
        raw_frame_count = len(self.taught_trajectory)
        discard_count = min(
            self.cfg.TEACH_INITIAL_DISCARD_FRAMES,
            raw_frame_count,
        )
        self.taught_trajectory = self._trim_stationary_trajectory_edges(
            self.taught_trajectory[discard_count:]
        )
        self._teach_start_joints = None
        if not self.taught_trajectory:
            self.teach_state = TeachState.IDLE
            self.workflow_message = (
                "Teaching ended before the initial discard window completed. "
                "Press Teach to try again."
                if raw_frame_count <= self.cfg.TEACH_INITIAL_DISCARD_FRAMES
                else "No motion was detected after the initial discard window. "
                "Press Teach to create a trajectory."
            )
            utils.logger.warning(self.workflow_message)
            return False
        self.teach_state = TeachState.READY
        self.workflow_message = "Trajectory ready. Press Replay Collect to record it."
        utils.logger.info(
            "Teach complete: discarded %d startup frames; retained %d waypoints. "
            "Replay Collect is enabled.",
            discard_count,
            len(self.taught_trajectory),
        )
        return True

    def _move_to_joints(self, joints: np.ndarray) -> None:
        target = np.asarray(joints, dtype=float)
        if target.shape != (self.backend.dof,) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"Expected {self.backend.dof} finite initial joint values, "
                f"got {target.shape}."
            )
        self.backend.reset(target)

    def move_to_initial_pose(self) -> bool:
        if self.teach_state not in {TeachState.IDLE, TeachState.READY}:
            return False
        previous_state = self.teach_state
        self.disable_freedrive()
        self.teach_state = TeachState.MOVING_INITIAL
        try:
            initial = self.backend.initial_joint_configuration(self.cfg.INITIAL_JOINT)
            self._move_to_joints(initial)
        finally:
            self.teach_state = previous_state
        utils.logger.info("Robot reached the configured initial pose.")
        self.workflow_message = "Robot reached the configured initial pose."
        return True

    def _process_pending_if_embedded(self) -> None:
        if not any(thread.is_alive() for thread in self._recording_threads):
            self.recording_service.process_pending_once()

    def _wait_for_pending_dataset_action(self, timeout: float = 30.0) -> None:
        if not any(thread.is_alive() for thread in self._recording_threads):
            self.recording_service.process_pending_once()
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _collecting, export_requested, rollback_requested = (
                self.recording_service.control.snapshot()
            )
            if not export_requested and not rollback_requested:
                return
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for the dataset operation to finish.")

    def _request_motion_stop(self) -> None:
        robot = getattr(self.backend, "robot", None)
        arm = getattr(robot, "robot", None)
        slow_stop = getattr(arm, "rm_set_arm_slow_stop", None)
        if callable(slow_stop):
            result = int(slow_stop())
            if result != 0:
                utils.logger.error(
                    "RealMan slow stop failed with error code %d.", result
                )

    def replay_collect(self) -> bool:
        if (
            self.teach_state is not TeachState.READY
            or not self.taught_trajectory
            or not self._force_data_available
        ):
            return False
        if self.cfg.COLLECT_RATE >= 100:
            raise ValueError(
                "Teach replay uses low-follow joint commands and requires "
                "COLLECT_RATE below 100 Hz."
            )

        trajectory = [target.copy() for target in self.taught_trajectory]
        period = 1.0 / self.cfg.COLLECT_RATE
        self.disable_freedrive()
        self.teach_state = TeachState.REPLAYING
        self.workflow_message = "Replaying trajectory and collecting data."
        recording_started = False
        recording_active = False
        episode_exported = False
        episode_count_before = int(self.dataset.recorded_episodes)
        try:
            self._move_to_joints(trajectory[0])
            self.recording_service.pause_event.set()
            recording_started = self.recording_service.start_recording()
            if not recording_started:
                raise RuntimeError("Dataset recorder is already collecting.")
            recording_active = True

            next_tick = time.perf_counter()
            for target in trajectory:
                self.backend.command_joint_configuration(target, period)
                sample = self.read_sample()
                self.record_sample(sample, action=target)
                self.publish_sample(sample)
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()

            if not self.recording_service.stop_recording():
                raise RuntimeError("Failed to queue replay episode export.")
            recording_active = False
            self.recording_service.pause_event.clear()
            self._wait_for_pending_dataset_action()
            if int(self.dataset.recorded_episodes) != episode_count_before + 1:
                raise RuntimeError("Replay finished, but the dataset was not exported.")
            episode_exported = True
            utils.logger.info(
                "Replay collection complete: %d frames exported.", len(trajectory)
            )
            if self.recording_service.collecting:
                raise RuntimeError(
                    "Dataset recorder remained active after replay export."
                )
            self.workflow_message = (
                "Replay collection complete; returning robot to initial pose."
            )
            self.teach_state = TeachState.READY
            if not self.move_to_initial_pose():
                raise RuntimeError("Failed to start the return to initial pose.")
            self.workflow_message = (
                "Replay collection complete; episode exported and robot returned "
                "to the initial pose."
            )
            return True
        except Exception as exc:
            self.workflow_message = (
                f"Episode exported, but return to initial pose failed: {exc}"
                if episode_exported
                else f"Replay failed: {exc}"
            )
            self._request_motion_stop()
            if recording_started and recording_active:
                self.recording_service.request_rollback()
                self.recording_service.pause_event.clear()
                self._wait_for_pending_dataset_action()
            raise
        finally:
            self.recording_service.pause_event.clear()
            self.teach_state = TeachState.READY

    def run(self) -> None:
        period = 1.0 / self.cfg.COLLECT_RATE
        next_tick = time.monotonic()
        try:
            if self.camera_manager is not None:
                self.camera_manager.start()
            self._recording_stop_event = threading.Event()
            if self.beaver_reader is not None:
                self.beaver_reader.start(self._recording_stop_event)
            self._recording_threads = self.recording_service.start(
                self._recording_stop_event
            )
            utils.logger.info(
                "Ready. Use Teach, Reteach, Replay Collect, and Initial Pose "
                "in the visualizer; press Ctrl-C to stop."
            )
            while True:
                if (
                    self.visualizer_handle is not None
                    and not self.visualizer_handle.process.is_alive()
                ):
                    utils.logger.info("Visualizer closed; stopping the collector.")
                    break
                sample = self.read_sample()
                with self._sample_lock:
                    self._latest_sample = sample
                self.capture_teach_sample(sample)
                self.publish_sample(sample)
                # Process commands after sampling so End Teach preserves the
                # final dragged pose from this control tick.
                self.handle_visualizer_commands()

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
        if self._recording_stop_event is not None:
            self._recording_stop_event.set()
        for thread in self._recording_threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5.0)
        if self.freedrive_active:
            try:
                self.disable_freedrive()
            except Exception as exc:
                close_error = exc
                utils.logger.exception("Failed to stop RealMan freedrive")
        try:
            self.recording_service.close()
        except Exception as exc:
            if close_error is None:
                close_error = exc
            utils.logger.exception("Failed to finalize the teaching dataset")
        finally:
            if self.beaver_reader is not None:
                try:
                    self.beaver_reader.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc
                    utils.logger.exception("Failed to close the Beaver reader")
            if self.camera_manager is not None:
                try:
                    self.camera_manager.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc
                    utils.logger.exception("Failed to close the RealSense cameras")
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
        description=(
            "Teach a RealMan joint path, replay it, and collect synchronized "
            "joint, TCP, force, and RealSense data."
        )
    )
    parser.add_argument("--robot-ip", default=defaults.ROBOT_IP)
    parser.add_argument("--port", type=int, default=defaults.REALMAN_PORT)
    parser.add_argument(
        "--dataset-dir",
        default=defaults.DATASET_DIR,
        help=(
            "Dataset base path; defaults to Config.DATASET_DIR and the recorder "
            "appends '_lero'."
        ),
    )
    parser.add_argument("--task", default=defaults.TASK_NAME)
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
    camera_manager = create_camera_manager(cfg)
    beaver_reader = None
    if cfg.beaver_enable:
        from beaver import BeaverReader

        beaver_reader = BeaverReader.from_config(cfg)

    from visualizer import start_visualizer

    try:
        visualizer_handle = start_visualizer(
            hz=viz_cfg.HZ,
            window_s=viz_cfg.WINDOW_S,
            title="RealMan Teach / Replay Dataset Collector",
            force_panel_range=viz_cfg.FORCE_PANEL_RANGE,
            camera_num=camera_manager.camera_num,
            show_rollback_button=True,
            show_record_button=False,
            show_teach_controls=True,
            show_tactile_panel=cfg.TACTILE_ENABLE,
            beaver_enabled=cfg.beaver_enable,
            beaver_layout=cfg.BEAVER_SENSOR_LAYOUT,
            beaver_max_mm=cfg.BEAVER_VISUALIZER_MAX_MM,
        )
        collector = RealManTeachCollector(
            cfg,
            camera_manager=camera_manager,
            visualizer_handle=visualizer_handle,
            beaver_reader=beaver_reader,
        )
    except Exception:
        if "visualizer_handle" in locals():
            visualizer_handle.close()
        camera_manager.close()
        raise
    collector.run()


if __name__ == "__main__":
    main()
