"""Reusable threaded data collection and episode export services.

Entry points provide a lightweight frame callback.  This module owns all
DatasetRecorder calls, export/rollback serialization, and dataset shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import threading
import time
from typing import Any, Callable, Protocol

import cv2
import numpy as np

import utils
from dataset import DatasetRecorder


@dataclass(frozen=True)
class RecordingFrame:
    state: np.ndarray
    action: np.ndarray
    camera_images: dict[str, np.ndarray]
    tactile_data: np.ndarray | None = None
    wrench_data: np.ndarray | None = None
    depth_images: dict[str, np.ndarray] | None = None
    extra_data: dict[str, object] | None = None
    beaver_data: object | None = None


class RecordingControlProtocol(Protocol):
    def snapshot(self) -> tuple[bool, bool, bool]: ...
    def start_recording(self) -> bool: ...
    def request_export(self) -> bool: ...
    def request_rollback(self) -> bool: ...
    def clear_export(self) -> None: ...
    def clear_rollback(self) -> None: ...
    def stop_recording(self) -> None: ...


class RecordingControl:
    """Thread-safe local recording state used by standalone collectors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collecting = False
        self._export_requested = False
        self._rollback_requested = False

    def snapshot(self) -> tuple[bool, bool, bool]:
        with self._lock:
            return (
                self._collecting,
                self._export_requested,
                self._rollback_requested,
            )

    def start_recording(self) -> bool:
        with self._lock:
            if self._collecting:
                return False
            self._collecting = True
            self._export_requested = False
            self._rollback_requested = False
            return True

    def request_export(self) -> bool:
        with self._lock:
            was_collecting = self._collecting
            self._collecting = False
            self._export_requested = True
            self._rollback_requested = False
            return was_collecting

    def request_rollback(self) -> bool:
        with self._lock:
            self._collecting = False
            self._export_requested = False
            self._rollback_requested = True
            return True

    def clear_export(self) -> None:
        with self._lock:
            self._export_requested = False

    def clear_rollback(self) -> None:
        with self._lock:
            self._rollback_requested = False

    def stop_recording(self) -> None:
        with self._lock:
            self._collecting = False


class ManagerRecordingControl:
    """Adapt the Start/Stop/Undo flags exposed by UDP camera managers."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def snapshot(self) -> tuple[bool, bool, bool]:
        with self.manager._lock:
            return (
                bool(self.manager.data_collecting_state),
                bool(self.manager.data_export_state),
                bool(self.manager.data_rollback_state),
            )

    def start_recording(self) -> bool:
        with self.manager._lock:
            if self.manager.data_collecting_state:
                return False
            self.manager.data_collecting_state = True
            self.manager.data_export_state = False
            self.manager.data_rollback_state = False
            return True

    def request_export(self) -> bool:
        with self.manager._lock:
            was_collecting = bool(self.manager.data_collecting_state)
            self.manager.data_collecting_state = False
            self.manager.data_export_state = True
            self.manager.data_rollback_state = False
            return was_collecting

    def request_rollback(self) -> bool:
        with self.manager._lock:
            self.manager.data_collecting_state = False
            self.manager.data_export_state = False
            self.manager.data_rollback_state = True
            return True

    def clear_export(self) -> None:
        with self.manager._lock:
            self.manager.data_export_state = False

    def clear_rollback(self) -> None:
        with self.manager._lock:
            self.manager.data_rollback_state = False

    def stop_recording(self) -> None:
        with self.manager._lock:
            self.manager.data_collecting_state = False


class DataRecordingService:
    """Collect frames and serialize dataset mutations on dedicated threads."""

    def __init__(
        self,
        dataset: DatasetRecorder,
        frame_provider: Callable[[], RecordingFrame | None],
        collect_rate: float,
        *,
        control: RecordingControlProtocol | None = None,
        export_context: object | None = None,
        background_errors: list[str] | None = None,
        thread_name: str = "dataset",
    ) -> None:
        if collect_rate <= 0:
            raise ValueError("collect_rate must be positive")
        self.dataset = dataset
        self.frame_provider = frame_provider
        self.collect_rate = float(collect_rate)
        self.control = RecordingControl() if control is None else control
        self.export_context = export_context
        self.background_errors = background_errors
        self.thread_name = thread_name
        self.pause_event = threading.Event()
        self._dataset_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._threads: list[threading.Thread] = []

        data_collection = getattr(self.dataset, "data_collection", None)
        if data_collection is None:
            self._dataset_accepts_named_frame = True
        else:
            parameters = inspect.signature(data_collection).parameters
            self._dataset_accepts_named_frame = (
                "state" in parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            )

    @property
    def collecting(self) -> bool:
        return self.control.snapshot()[0]

    def start_recording(self) -> bool:
        started = self.control.start_recording()
        if started:
            utils.logger.info(
                "Recording episode %d...", self.dataset.recorded_episodes
            )
        return started

    def stop_recording(self) -> bool:
        requested = self.control.request_export()
        if not requested:
            return False
        utils.logger.info("Episode export queued.")
        return True

    def request_rollback(self) -> bool:
        requested = self.control.request_rollback()
        if requested:
            utils.logger.info("Dataset rollback queued.")
        return requested

    def recording_status(self) -> dict[str, object]:
        with self._dataset_lock:
            status = dict(
                self.dataset.recording_status(collecting=self.collecting)
            )
            status["collect_rate_hz"] = self.collect_rate
            return status

    def handle_visualizer_commands(self, visualizer_handle: Any | None) -> None:
        if visualizer_handle is None:
            return
        for command in visualizer_handle.drain_commands():
            name = command.get("command")
            if name == "toggle_recording":
                if self.collecting:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif name == "rollback_last_episode":
                self.request_rollback()

    def _write_frame(self, frame: RecordingFrame) -> None:
        values = {
            "state": frame.state,
            "action": frame.action,
            "camera_images": frame.camera_images,
            "tactile_data": frame.tactile_data,
            "wrench_data": frame.wrench_data,
            "depth_images": frame.depth_images,
            "extra_data": frame.extra_data,
        }
        if frame.beaver_data is not None:
            values["beaver_data"] = frame.beaver_data
        if self._dataset_accepts_named_frame:
            self.dataset.data_collection(**values)
        else:
            self.dataset.data_collection(*values.values())

    def record_frame(self, frame: RecordingFrame) -> None:
        """Write one prepared frame; useful for deterministic unit tests."""
        with self._dataset_lock:
            self._write_frame(frame)

    def collect_once(self) -> bool:
        if self.pause_event.is_set() or not self.collecting:
            return False
        frame = self.frame_provider()
        if frame is None:
            return False
        with self._dataset_lock:
            if self.pause_event.is_set() or not self.collecting:
                return False
            self._write_frame(frame)
        return True

    def process_pending_once(self) -> bool:
        _collecting, export_requested, rollback_requested = self.control.snapshot()
        if not export_requested and not rollback_requested:
            return False

        self.pause_event.set()
        try:
            with self._dataset_lock:
                if rollback_requested:
                    removed = self.dataset.rollback_last_episode()
                    if removed:
                        utils.logger.info(
                            "Rollback complete. Next episode: %d",
                            self.dataset.recorded_episodes,
                        )
                    else:
                        utils.logger.warning(
                            "Rollback requested, but no episode was removed."
                        )
                elif self.dataset.collect_step:
                    self.dataset.data_export(self.export_context)
                    self.dataset._reset_data_dict()
                    utils.logger.info(
                        "Episode %d exported successfully",
                        self.dataset.recorded_episodes - 1,
                    )
                else:
                    utils.logger.warning("No data to export.")
            return True
        finally:
            if rollback_requested:
                self.control.clear_rollback()
            if export_requested:
                self.control.clear_export()
            self.pause_event.clear()

    def _report_failure(
        self,
        label: str,
        exc: Exception,
        stop_event: threading.Event,
    ) -> None:
        message = f"{self.thread_name} {label} failed: {exc}"
        utils.logger.exception(message)
        if self.background_errors is not None:
            self.background_errors.append(message)
        stop_event.set()

    def _collect_loop(self, stop_event: threading.Event) -> None:
        period = 1.0 / self.collect_rate
        next_tick = time.monotonic()
        try:
            while not stop_event.is_set():
                self.collect_once()
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    stop_event.wait(delay)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:
            self._report_failure("collection", exc, stop_event)

    def _export_loop(self, stop_event: threading.Event) -> None:
        try:
            while not stop_event.is_set():
                self.process_pending_once()
                stop_event.wait(0.05)
        except Exception as exc:
            self._report_failure("export", exc, stop_event)
        finally:
            self.close()

    def start(self, stop_event: threading.Event) -> list[threading.Thread]:
        threads = [
            threading.Thread(
                target=self._collect_loop,
                args=(stop_event,),
                name=f"{self.thread_name}-collector",
                daemon=True,
            ),
            threading.Thread(
                target=self._export_loop,
                args=(stop_event,),
                name=f"{self.thread_name}-exporter",
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        self._threads = threads
        return threads

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self.pause_event.set()
            try:
                _collecting, _export, rollback = self.control.snapshot()
                self.control.stop_recording()
                with self._dataset_lock:
                    if rollback:
                        self.dataset.rollback_last_episode()
                    elif self.dataset.collect_step:
                        utils.logger.info("Saving the active episode before exit.")
                        self.dataset.data_export(self.export_context)
                        self.dataset._reset_data_dict()
                    self.dataset.close()
                self._closed = True
            finally:
                self.pause_event.clear()


class RealManEpisodeRecorder(DataRecordingService):
    """RealMan frame adapter kept here so its entry point only wires services."""

    def __init__(
        self,
        cfg: Any,
        teleop: Any,
        camera_manager: Any,
        *,
        dataset: DatasetRecorder | None = None,
        background_errors: list[str] | None = None,
        beaver_reader: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.teleop = teleop
        self.camera_manager = camera_manager
        self.beaver_reader = beaver_reader
        self._last_state_quaternion: np.ndarray | None = None
        self._last_action_quaternion: np.ndarray | None = None
        if dataset is None:
            dataset = DatasetRecorder(
                camera_manager.camera_num,
                robot_dof=teleop.dof,
                robot_type=getattr(
                    teleop.backend,
                    "dataset_robot_type",
                    teleop.backend.name,
                ),
                force_collect=cfg.FORCE_COLLECT,
                torque_collect=cfg.TORQUE_COLLECT,
                gripper=False,
                config=cfg,
            )
        super().__init__(
            dataset,
            self._recording_frame,
            cfg.COLLECT_RATE,
            control=ManagerRecordingControl(camera_manager),
            export_context=camera_manager,
            background_errors=background_errors,
            thread_name="realman-dataset",
        )

    @staticmethod
    def _pose_vector(
        tcp_pose: np.ndarray,
        previous_quaternion: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(tcp_pose, dtype=float)
        quaternion = utils.quat_cal(pose[:3, :3], previous_quaternion)
        return np.concatenate([quaternion, pose[:3, 3]]).astype(np.float32), quaternion

    @staticmethod
    def _delta_tcp_action(
        measured_tcp: np.ndarray,
        target_tcp: np.ndarray,
    ) -> np.ndarray:
        delta_translation = target_tcp[:3, 3] - measured_tcp[:3, 3]
        delta_rotation = target_tcp[:3, :3] @ measured_tcp[:3, :3].T
        delta_rotvec, _ = cv2.Rodrigues(delta_rotation)
        return np.concatenate(
            [delta_translation, delta_rotvec.reshape(3)]
        ).astype(np.float32)

    def _state_and_action(
        self,
        robot: Any,
        action_joints: np.ndarray,
        action_tcp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_tcp, self._last_state_quaternion = self._pose_vector(
            robot.tcp_pose,
            self._last_state_quaternion,
        )
        if self.cfg.DATA_TYPE == "tcp":
            action, self._last_action_quaternion = self._pose_vector(
                action_tcp,
                self._last_action_quaternion,
            )
            return state_tcp, action, state_tcp
        if self.cfg.DATA_TYPE == "delta_tcp":
            return (
                robot.joints.astype(np.float32),
                self._delta_tcp_action(robot.tcp_pose, action_tcp),
                state_tcp,
            )
        return (
            robot.joints.astype(np.float32),
            np.asarray(action_joints, dtype=np.float32),
            state_tcp,
        )

    def _recording_frame(self) -> RecordingFrame | None:
        with self.camera_manager._lock:
            if not self.camera_manager.data_collecting_state:
                return None
            images = {
                name: np.asarray(image).copy()
                for name, image in self.camera_manager.camera_images.items()
            }
            image_timestamps = dict(
                getattr(self.camera_manager, "camera_image_timestamps_ns", {})
            )
            depth_images = (
                {
                    name: np.asarray(depth).copy()
                    for name, depth in self.camera_manager.depth_images.items()
                }
                if self.cfg.DEPTH_INFO_ENABLE
                else None
            )
            vr_input_timestamp_ns = int(
                getattr(self.camera_manager, "vr_input_timestamp_ns", 0)
            )

        if not (
            self.camera_manager.is_movement_exist()
            or self.teleop.reset_active()
        ):
            return None

        collect_timestamp_ns = time.monotonic_ns()
        robot, action_joints, action_tcp, action_timestamp_ns = (
            self.teleop.recording_snapshot()
        )
        state, action, tcp_vector = self._state_and_action(
            robot,
            action_joints,
            action_tcp,
        )
        beaver = self.beaver_reader.snapshot() if self.beaver_reader is not None else None
        return RecordingFrame(
            state=state,
            action=action,
            camera_images=images,
            wrench_data=(
                robot.wrench
                if self.cfg.FORCE_COLLECT or self.cfg.TORQUE_COLLECT
                else None
            ),
            depth_images=depth_images,
            extra_data={
                "collect_timestamp_ns": np.array(collect_timestamp_ns, dtype=np.int64),
                "robot_state_timestamp_ns": np.array(
                    robot.state_timestamp_ns, dtype=np.int64
                ),
                "robot_action_timestamp_ns": np.array(
                    action_timestamp_ns, dtype=np.int64
                ),
                "vr_input_timestamp_ns": np.array(
                    vr_input_timestamp_ns, dtype=np.int64
                ),
                "tactile_timestamp_ns": np.array(0, dtype=np.int64),
                "beaver_timestamp_ns": np.array(
                    0 if beaver is None else beaver.timestamp_ns,
                    dtype=np.int64,
                ),
                "camera_timestamps_ns": image_timestamps,
                "tcp_pose": tcp_vector,
            },
            beaver_data=beaver,
        )
