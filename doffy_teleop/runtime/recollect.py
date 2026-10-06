"""Replay a LeRobot dataset and recollect it with raw 16-bit Beaver data.

The source dataset is read-only.  For every selected episode the robot moves to
the recorded first joint pose and waits in a camera-calibration state.  The web
UI shows the source first frame beside the live RealSense stream; Enter starts
the deterministic replay and records the same synchronized fields as
``realman_teachcollect.py`` into a sibling ``*_recollect`` dataset.

If the current source trajectory is unsatisfactory, the operator can drag-teach
a replacement (same freedrive teach + replay-collect path as teachcollect).
The replacement is stored as the current output episode only; the next slot
still loads the original dataset's next source episode.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any
import webbrowser

import numpy as np

from doffy_teleop import utils
from doffy_teleop.config import Config
from doffy_teleop.recording.dataset import DatasetRecorder, normalize_lerobot_fps
from doffy_teleop.recording.replay import (
    EpisodeTrajectory,
    RealManLeRobotDataset,
    camera_name_from_video_key,
    interpolate_joint_speed_discontinuities,
    validate_trajectory_speed,
)
from doffy_teleop.runtime.teachcollect import (
    RealManTeachCollector,
    TeachState,
    create_camera_manager,
)
from doffy_teleop.robots.backend_api import RobotBackend


RECOLLECT_MANIFEST = "meta/recollect.json"
RECOLLECT_MANIFEST_VERSION = 1


class RecollectState(str, Enum):
    STARTING = "starting"
    MOVING_TO_START = "moving_to_start"
    CALIBRATING = "calibrating"
    TEACHING = "teaching"
    TEACH_READY = "teach_ready"
    REPLAYING = "replaying"
    EXPORTING = "exporting"
    ERROR = "error"
    COMPLETE = "complete"
    CLOSED = "closed"


class ReplayAborted(RuntimeError):
    pass


@dataclass(frozen=True)
class RecollectEpisode:
    source_episode_index: int
    length: int


def default_output_root(source_root: str | Path) -> Path:
    source = Path(source_root).expanduser().resolve()
    return source.with_name(f"{source.name}_recollect_2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(
    source: RealManLeRobotDataset,
    episode_indices: list[int],
) -> dict[str, Any]:
    return {
        "version": RECOLLECT_MANIFEST_VERSION,
        "source": {
            "root": str(source.root),
            "info_sha256": _sha256(source.root / "meta/info.json"),
            "total_episodes": source.total_episodes,
            "fps": source.fps,
        },
        "source_episode_indices": list(episode_indices),
        "beaver": {
            "wire_distance_bits": 16,
            "simulate_8bit": False,
        },
    }


def validate_output_location(
    source: RealManLeRobotDataset,
    output_root: str | Path,
    episode_indices: list[int],
) -> Path:
    """Validate a new/resumed target without ever changing the source."""
    output = Path(output_root).expanduser().resolve()
    if output == source.root:
        raise ValueError("Recollect output must differ from the source dataset.")
    if output.is_relative_to(source.root) or source.root.is_relative_to(output):
        raise ValueError("Source and recollect output must not contain one another.")
    if not output.exists():
        return output
    if not output.is_dir():
        raise FileExistsError(f"Recollect output is not a directory: {output}")
    manifest_path = output / RECOLLECT_MANIFEST
    info_path = output / "meta/info.json"
    if not manifest_path.is_file() or not info_path.is_file():
        raise FileExistsError(
            f"Refusing to overwrite non-recollect directory {output}. "
            f"Expected both meta/info.json and {RECOLLECT_MANIFEST}."
        )
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _manifest_payload(source, episode_indices)
    if stored != expected:
        raise ValueError(
            "Existing recollect output belongs to a different source or episode "
            "selection. Choose another --output-dataset."
        )
    return output


def write_recollect_manifest(
    output_root: str | Path,
    source: RealManLeRobotDataset,
    episode_indices: list[int],
) -> None:
    path = Path(output_root) / RECOLLECT_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_manifest_payload(source, episode_indices), indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_camera_resolution(source: RealManLeRobotDataset) -> tuple[int, int]:
    """Return a common source camera resolution as (width, height)."""
    shapes = []
    for key in source.video_keys:
        feature = source.info.get("features", {}).get(key, {})
        shape = tuple(feature.get("shape", ()))
        if len(shape) == 3 and shape[-1] in {1, 3, 4}:
            shapes.append((int(shape[1]), int(shape[0])))
    if not shapes:
        return tuple(Config().REALSENSE_RESOLUTION)
    if len(set(shapes)) != 1:
        raise ValueError(f"Source cameras have different resolutions: {shapes}")
    return shapes[0]


class RealManRecollector(RealManTeachCollector):
    """Thread-safe sequential source-episode replay/recollection state machine."""

    def __init__(
        self,
        cfg: Config,
        source_dataset: RealManLeRobotDataset,
        *,
        episode_indices: list[int] | None = None,
        maximum_joint_speed: float = 2.5,
        joint_jump_policy: str = "original",
        backend: RobotBackend | None = None,
        dataset: DatasetRecorder | None = None,
        camera_manager: Any | None = None,
        beaver_reader: Any | None = None,
    ) -> None:
        if cfg.BEAVER_SIMULATE_8BIT:
            raise ValueError(
                "Recollection requires BEAVER_SIMULATE_8BIT=False so raw 16-bit "
                "distances are preserved."
            )
        if cfg.COLLECT_RATE >= 100:
            raise ValueError(
                "Recollection uses RealMan low-follow commands and requires an "
                "episode FPS below 100 Hz."
            )
        if maximum_joint_speed <= 0:
            raise ValueError("maximum_joint_speed must be positive.")
        if joint_jump_policy not in {"error", "interpolate", "original"}:
            raise ValueError(
                "joint_jump_policy must be 'error', 'interpolate', or 'original'."
            )
        if source_dataset.control_mode != "joint":
            raise ValueError("Recollection currently requires joint control mode.")

        selected = source_dataset.episode_indices(episodes=episode_indices)
        if len(set(selected)) != len(selected):
            raise ValueError("Recollect episode selection must not contain duplicates.")
        if not source_dataset.video_keys:
            raise ValueError(
                "Recollection requires at least one source video camera for "
                "frame-0 calibration."
            )
        self.source_dataset = source_dataset
        self.episode_indices = selected
        self.episodes = [
            RecollectEpisode(index, source_dataset.episode_length(index))
            for index in selected
        ]
        self.maximum_joint_speed = float(maximum_joint_speed)
        self.joint_jump_policy = joint_jump_policy
        self._state_lock = threading.RLock()
        self._state = RecollectState.STARTING
        self._message = "Starting cameras, robot, and Beaver reader."
        self._position = 0
        self._trajectory: EpisodeTrajectory | None = None
        self._trajectory_safety: dict[str, Any] = {
            "policy": self.joint_jump_policy,
            "unsafe_original": False,
            "safety_limit": self.maximum_joint_speed,
            "repaired": False,
            "repair_count": 0,
            "original_max_joint_speed": None,
            "replay_max_joint_speed": None,
            "repairs": [],
        }
        self._reference_images: dict[str, np.ndarray] = {}
        self._reference_generation = 0
        self._replay_frame = 0
        self._collecting_replacement = False
        self._replaced_source_episodes: dict[int, int] = {}
        self._command_pending = False
        self._command_queue: queue.Queue[str] = queue.Queue(maxsize=4)
        self._abort_replay = threading.Event()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._started = False

        super().__init__(
            cfg,
            backend=backend,
            dataset=dataset,
            camera_manager=camera_manager,
            visualizer_handle=None,
            beaver_reader=beaver_reader,
        )
        completed = int(self.dataset.recorded_episodes)
        if completed > len(self.episodes):
            self.close()
            raise ValueError(
                f"Output already has {completed} episodes, but this recollect run "
                f"selects only {len(self.episodes)} source episodes."
            )
        self._position = completed

    @property
    def state(self) -> RecollectState:
        with self._state_lock:
            return self._state

    @property
    def current_source_episode(self) -> int | None:
        with self._state_lock:
            if self._position >= len(self.episodes):
                return None
            return self.episodes[self._position].source_episode_index

    def _set_state(self, state: RecollectState, message: str) -> None:
        with self._state_lock:
            self._state = state
            self._message = message

    def start(self) -> None:
        if self._started:
            return
        if self.camera_manager is not None:
            self.camera_manager.start()
        if self.beaver_reader is not None:
            self.beaver_reader.start(self._stop_event)
        self._started = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="realman-recollect-workflow",
            daemon=True,
        )
        self._worker.start()
        self._enqueue("prepare")

    def _enqueue(self, command: str) -> bool:
        with self._state_lock:
            if self._command_pending:
                return False
            self._command_pending = True
        try:
            self._command_queue.put_nowait(command)
            return True
        except queue.Full:
            with self._state_lock:
                self._command_pending = False
            return False

    def request_replay(self) -> bool:
        with self._state_lock:
            if self._state is not RecollectState.CALIBRATING:
                return False
        if self.readiness_issues():
            return False
        return self._enqueue("replay")

    def request_teach(self) -> bool:
        with self._state_lock:
            if self._state not in {
                RecollectState.CALIBRATING,
                RecollectState.TEACH_READY,
            }:
                return False
        return self._enqueue("start_teach")

    def request_end_teach(self) -> bool:
        with self._state_lock:
            if self._state is not RecollectState.TEACHING:
                return False
        return self._enqueue("end_teach")

    def request_reteach(self) -> bool:
        with self._state_lock:
            if self._state not in {
                RecollectState.TEACHING,
                RecollectState.TEACH_READY,
            }:
                return False
        return self._enqueue("start_teach")

    def request_cancel_teach(self) -> bool:
        with self._state_lock:
            if self._state not in {
                RecollectState.TEACHING,
                RecollectState.TEACH_READY,
            }:
                return False
        self._abort_replay.set()
        try:
            self.disable_freedrive()
        except Exception:
            utils.logger.exception("Failed to stop freedrive while cancelling teach")
        return self._enqueue("cancel_teach")

    def request_teach_collect(self) -> bool:
        with self._state_lock:
            if self._state is not RecollectState.TEACH_READY:
                return False
        if self.readiness_issues():
            return False
        if not self.taught_trajectory:
            return False
        return self._enqueue("teach_collect")

    def request_retry(self) -> bool:
        with self._state_lock:
            if self._state is not RecollectState.ERROR:
                return False
        return self._enqueue("prepare")

    def request_rollback(self) -> bool:
        with self._state_lock:
            allowed = self._state in {
                RecollectState.CALIBRATING,
                RecollectState.TEACHING,
                RecollectState.TEACH_READY,
                RecollectState.ERROR,
                RecollectState.COMPLETE,
            }
        if not allowed or int(self.dataset.recorded_episodes) <= 0:
            return False
        return self._enqueue("rollback")

    def request_stop(self) -> bool:
        with self._state_lock:
            state = self._state
        if state is RecollectState.TEACHING:
            self._abort_replay.set()
            try:
                self.disable_freedrive()
            except Exception:
                utils.logger.exception(
                    "Failed to stop freedrive during replacement-teach stop"
                )
            self.taught_trajectory.clear()
            self.teach_state = TeachState.IDLE
            return self._enqueue("prepare")
        if state not in {
            RecollectState.MOVING_TO_START,
            RecollectState.REPLAYING,
            RecollectState.EXPORTING,
        }:
            return False
        self._abort_replay.set()
        self._request_motion_stop()
        return True

    def _worker_loop(self) -> None:
        period = 1.0 / self.cfg.COLLECT_RATE
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            try:
                try:
                    command = self._command_queue.get_nowait()
                except queue.Empty:
                    command = None
                if command is not None:
                    try:
                        if command == "prepare":
                            self._prepare_current_episode()
                        elif command == "replay":
                            self._replay_current_episode()
                        elif command == "start_teach":
                            self._begin_replacement_teach()
                        elif command == "end_teach":
                            self._finish_replacement_teach()
                        elif command == "cancel_teach":
                            self._cancel_replacement_teach()
                        elif command == "teach_collect":
                            self._collect_taught_episode()
                        elif command == "rollback":
                            self._rollback_previous_episode()
                    except Exception as exc:
                        try:
                            self.disable_freedrive()
                        except Exception:
                            utils.logger.exception(
                                "Failed to stop freedrive after a workflow error"
                            )
                        self._set_state(RecollectState.ERROR, f"Workflow failed: {exc}")
                        utils.logger.exception("RealMan recollect workflow failed")
                    finally:
                        with self._state_lock:
                            self._command_pending = False
                elif self.state in {
                    RecollectState.CALIBRATING,
                    RecollectState.TEACHING,
                    RecollectState.TEACH_READY,
                    RecollectState.ERROR,
                }:
                    sample = self.read_sample()
                    with self._sample_lock:
                        self._latest_sample = sample
                    if self.state is RecollectState.TEACHING:
                        self.capture_teach_sample(sample)
            except Exception as exc:
                self._set_state(RecollectState.ERROR, f"Sensor update failed: {exc}")
                utils.logger.exception("RealMan recollect sensor update failed")

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                next_tick = time.monotonic()

    def _prepare_current_episode(self) -> None:
        self._abort_replay.clear()
        self._collecting_replacement = False
        try:
            self.disable_freedrive()
        except Exception:
            utils.logger.exception("Failed to stop freedrive before preparing an episode")
        self.taught_trajectory.clear()
        self._teach_start_joints = None
        self.teach_state = TeachState.IDLE
        with self._state_lock:
            position = self._position
            if position >= len(self.episodes):
                self._trajectory = None
                self._reference_images = {}
                self._reference_generation += 1
                self._trajectory_safety = {
                    "policy": self.joint_jump_policy,
                    "unsafe_original": False,
                    "safety_limit": self.maximum_joint_speed,
                    "repaired": False,
                    "repair_count": 0,
                    "original_max_joint_speed": None,
                    "replay_max_joint_speed": None,
                    "repairs": [],
                }
                complete = True
            else:
                self._replay_frame = 0
                self._trajectory_safety = {
                    "policy": self.joint_jump_policy,
                    "unsafe_original": False,
                    "safety_limit": self.maximum_joint_speed,
                    "repaired": False,
                    "repair_count": 0,
                    "original_max_joint_speed": None,
                    "replay_max_joint_speed": None,
                    "repairs": [],
                }
                complete = False
        if complete:
            self._set_state(
                RecollectState.COMPLETE,
                f"Recollection complete: {len(self.episodes)} episodes exported.",
            )
            return

        descriptor = self.episodes[position]
        self._set_state(
            RecollectState.MOVING_TO_START,
            f"Loading episode {descriptor.source_episode_index} and moving to its first pose.",
        )
        source_trajectory = self.source_dataset.load_episode(
            descriptor.source_episode_index
        )
        original_max_joint_speed = source_trajectory.maximum_joint_speed
        repairs = ()
        if self.joint_jump_policy == "interpolate":
            trajectory, repairs = interpolate_joint_speed_discontinuities(
                source_trajectory,
                self.maximum_joint_speed,
            )
        elif self.joint_jump_policy == "error":
            trajectory = source_trajectory
            validate_trajectory_speed(trajectory, self.maximum_joint_speed)
        else:
            trajectory = source_trajectory
        unsafe_original = (
            self.joint_jump_policy == "original"
            and original_max_joint_speed > self.maximum_joint_speed
        )
        trajectory_safety = {
            "policy": self.joint_jump_policy,
            "unsafe_original": unsafe_original,
            "safety_limit": self.maximum_joint_speed,
            "repaired": bool(repairs),
            "repair_count": len(repairs),
            "original_max_joint_speed": original_max_joint_speed,
            "replay_max_joint_speed": trajectory.maximum_joint_speed,
            "repairs": [asdict(repair) for repair in repairs],
        }
        if repairs:
            utils.logger.warning(
                "Episode %d contained %d unsafe joint discontinuity(s): "
                "local interpolation reduced the peak from %.3f to %.3f rad/s.",
                descriptor.source_episode_index,
                len(repairs),
                original_max_joint_speed,
                trajectory.maximum_joint_speed,
            )
        elif unsafe_original:
            utils.logger.warning(
                "Episode %d will replay the inspected source trajectory unchanged: "
                "peak %.3f rad/s exceeds the configured %.3f rad/s warning limit.",
                descriptor.source_episode_index,
                original_max_joint_speed,
                self.maximum_joint_speed,
            )
        if len(trajectory.targets) != descriptor.length:
            raise ValueError(
                f"Episode {descriptor.source_episode_index} metadata says "
                f"{descriptor.length} frames but {len(trajectory.targets)} were loaded."
            )
        reference_images = {}
        for fallback_index, (video_key, image) in enumerate(
            self.source_dataset.load_first_camera_frames(
                descriptor.source_episode_index
            ).items()
        ):
            reference_images[
                camera_name_from_video_key(video_key, fallback_index)
            ] = np.asarray(image).copy()
        if len(reference_images) != len(self.source_dataset.video_keys):
            raise RuntimeError(
                f"Decoded {len(reference_images)} of "
                f"{len(self.source_dataset.video_keys)} source camera first frames."
            )
        with self._state_lock:
            self._trajectory = trajectory
            self._trajectory_safety = trajectory_safety
            self._reference_images = reference_images
            self._reference_generation += 1
            self._replay_frame = 0
        self._move_to_joints(trajectory.targets[0])
        if self._abort_replay.is_set():
            raise ReplayAborted(
                "Move to the episode start was stopped; press Retry when ready."
            )
        sample = self.read_sample()
        with self._sample_lock:
            self._latest_sample = sample
        if repairs:
            safety_note = (
                f" Safety interpolation repaired {len(repairs)} source joint "
                f"discontinuity(s): {original_max_joint_speed:.3f} → "
                f"{trajectory.maximum_joint_speed:.3f} rad/s."
            )
        elif unsafe_original:
            safety_note = (
                f" Warning: explicit original-trajectory mode will preserve the "
                f"{original_max_joint_speed:.3f} rad/s source motion unchanged "
                f"(warning limit {self.maximum_joint_speed:.3f} rad/s)."
            )
        else:
            safety_note = ""
        self._set_state(
            RecollectState.CALIBRATING,
            f"Episode {descriptor.source_episode_index} is paused at frame 0."
            f"{safety_note} Align the live cameras to the source first frames, "
            "then press Enter.",
        )

    def _beaver_issue(self) -> str | None:
        if not self.cfg.beaver_enable:
            return None
        if self.beaver_reader is None:
            return "Beaver reader is not configured"
        snapshot = self.beaver_reader.snapshot()
        if not bool(getattr(snapshot, "connected", False)):
            return str(getattr(snapshot, "error", "") or "Beaver is not connected")
        wire_bits = int(getattr(snapshot, "wire_distance_bits", 0))
        if wire_bits != 16:
            return (
                "waiting for a 16-bit Beaver frame"
                if wire_bits == 0
                else f"Beaver firmware is sending legacy {wire_bits}-bit distances"
            )
        grid_width = int(getattr(snapshot, "grid_width", 0))
        if grid_width != self.cfg.BEAVER_GRID_WIDTH:
            return (
                f"Beaver grid is {grid_width}x{grid_width}, but the output schema "
                f"expects {self.cfg.BEAVER_GRID_WIDTH}x{self.cfg.BEAVER_GRID_WIDTH}"
            )
        present = np.asarray(getattr(snapshot, "present", ()), dtype=bool)
        expected_sensors = len(self.cfg.BEAVER_SENSOR_LAYOUT)
        if present.shape != (expected_sensors,) or not np.all(present):
            return (
                f"waiting for all {expected_sensors} Beaver sensors "
                f"({int(np.count_nonzero(present))} present)"
            )
        age = getattr(snapshot, "age_s", None)
        if callable(age) and age() > self.cfg.BEAVER_STALE_AFTER_S:
            return "Beaver data is stale"
        return None

    def readiness_issues(self) -> list[str]:
        issues = []
        with self._sample_lock:
            sample = self._latest_sample
        if sample is None:
            issues.append("waiting for a robot state sample")
        elif not sample.force_valid:
            issues.append("RealMan force data is unavailable")
        beaver_issue = self._beaver_issue()
        if beaver_issue:
            issues.append(beaver_issue)
        if self.camera_manager is not None:
            with self.camera_manager._lock:
                live_names = set(self.camera_manager.camera_images)
            expected = set(self._reference_images)
            missing = sorted(expected - live_names)
            if missing:
                issues.append(f"live camera frames missing: {', '.join(missing)}")
        return issues

    def _check_replay_abort(self) -> None:
        if self._abort_replay.is_set() or self._stop_event.is_set():
            raise ReplayAborted("Replay was stopped; the partial episode was discarded.")
        issue = self._beaver_issue()
        if issue:
            raise RuntimeError(f"Beaver became unavailable during replay: {issue}")

    def _begin_replacement_teach(self) -> None:
        with self._state_lock:
            trajectory = self._trajectory
            position = self._position
            state = self._state
        if state not in {
            RecollectState.CALIBRATING,
            RecollectState.TEACHING,
            RecollectState.TEACH_READY,
        }:
            raise RuntimeError("Replacement teach is only available for the current source episode.")
        if trajectory is None or position >= len(self.episodes):
            raise RuntimeError("No current source episode is available to replace.")
        descriptor = self.episodes[position]
        self._abort_replay.clear()
        self._collecting_replacement = False
        self.disable_freedrive()
        self.taught_trajectory.clear()
        self._teach_start_joints = None
        self.teach_state = TeachState.IDLE
        self._set_state(
            RecollectState.MOVING_TO_START,
            f"Moving to the start pose of source episode {descriptor.source_episode_index} for a replacement teach.",
        )
        self._move_to_joints(trajectory.targets[0])
        if self._abort_replay.is_set():
            raise ReplayAborted(
                "Move to the replacement-teach start was stopped; press Retry when ready."
            )
        sample = self.read_sample()
        with self._sample_lock:
            self._latest_sample = sample
        if not self.start_teach():
            raise RuntimeError("Could not enter drag-teach for a replacement trajectory.")
        self._set_state(
            RecollectState.TEACHING,
            f"Drag a replacement for source episode {descriptor.source_episode_index}. "
            "Press End teach when the new path is done. After collection, the next "
            "slot still loads the original dataset's next episode.",
        )

    def _finish_replacement_teach(self) -> None:
        with self._state_lock:
            if self._state is not RecollectState.TEACHING:
                raise RuntimeError("No replacement teach is in progress.")
            position = self._position
        if not self.end_teach():
            # Keep the replacement-teach session. Falling back to calibrating
            # would re-enable source replay from the dragged pose, and the
            # first commanded waypoint is the episode start.
            if self.start_teach():
                self._set_state(
                    RecollectState.TEACHING,
                    self.workflow_message
                    + " The arm is holding its current pose. Keep dragging, then End teach.",
                )
                return
            self._set_state(
                RecollectState.CALIBRATING,
                self.workflow_message
                + " The current slot still belongs to the original source episode.",
            )
            return
        descriptor = self.episodes[position]
        taught_start = np.asarray(self.taught_trajectory[0], dtype=float)
        self._set_state(
            RecollectState.MOVING_TO_START,
            f"Returning to the start of the taught replacement for source "
            f"episode {descriptor.source_episode_index} so the scene can be set up.",
        )
        self._move_to_joints(taught_start)
        if self._abort_replay.is_set():
            raise ReplayAborted(
                "Return to the taught start was stopped; press Retry when ready."
            )
        sample = self.read_sample()
        with self._sample_lock:
            self._latest_sample = sample
        self._set_state(
            RecollectState.TEACH_READY,
            f"Replacement for source episode {descriptor.source_episode_index} "
            f"has {len(self.taught_trajectory)} frames. The arm is at that taught "
            "start pose — set up the scene, then press Collect replacement. "
            "The following slot will still be the original next episode.",
        )

    def _cancel_replacement_teach(self) -> None:
        self.disable_freedrive()
        self.taught_trajectory.clear()
        self._teach_start_joints = None
        self.teach_state = TeachState.IDLE
        self._collecting_replacement = False
        self._prepare_current_episode()

    def _collect_taught_episode(self) -> None:
        with self._state_lock:
            state = self._state
            position = self._position
        if state is not RecollectState.TEACH_READY:
            raise RuntimeError("No taught replacement is ready to collect.")
        if position >= len(self.episodes):
            raise RuntimeError("No current source episode is available to replace.")
        if not self.taught_trajectory:
            raise RuntimeError("Taught replacement is empty.")
        issues = self.readiness_issues()
        if issues:
            raise RuntimeError("Cannot collect replacement: " + "; ".join(issues))
        targets = np.stack(
            [np.asarray(waypoint, dtype=float) for waypoint in self.taught_trajectory]
        )
        if targets.ndim != 2 or targets.shape[1] != self.backend.dof:
            raise RuntimeError(
                f"Taught replacement has invalid joint shape {targets.shape}."
            )
        descriptor = self.episodes[position]
        self._record_and_export_trajectory(
            targets,
            fps=float(self.source_dataset.fps),
            source_episode_index=descriptor.source_episode_index,
            replacement=True,
        )

    def _replay_current_episode(self) -> None:
        with self._state_lock:
            trajectory = self._trajectory
            state = self._state
        if state is not RecollectState.CALIBRATING or trajectory is None:
            raise RuntimeError("No calibrated episode is ready to replay.")
        issues = self.readiness_issues()
        if issues:
            raise RuntimeError("Cannot replay: " + "; ".join(issues))
        self._record_and_export_trajectory(
            np.asarray(trajectory.targets, dtype=float),
            fps=float(trajectory.fps),
            source_episode_index=int(trajectory.episode_index),
            replacement=False,
        )

    def _record_and_export_trajectory(
        self,
        targets: np.ndarray,
        *,
        fps: float,
        source_episode_index: int,
        replacement: bool,
    ) -> None:
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 2 or targets.shape[0] < 1:
            raise RuntimeError("Cannot record an empty joint trajectory.")
        period = 1.0 / float(fps)
        episode_before = int(self.dataset.recorded_episodes)
        recording_started = False
        recording_active = False
        episode_exported = False
        self._abort_replay.clear()
        self._collecting_replacement = replacement
        self.teach_state = TeachState.REPLAYING
        action_label = (
            f"Collecting taught replacement for source episode {source_episode_index}"
            if replacement
            else f"Replaying source episode {source_episode_index}"
        )
        self._set_state(
            RecollectState.REPLAYING,
            f"{action_label} and recording raw 16-bit Beaver data.",
        )
        try:
            if replacement:
                self.disable_freedrive()
                self._move_to_joints(targets[0])
                if self._abort_replay.is_set():
                    raise ReplayAborted(
                        "Move to the taught start was stopped; the partial episode was discarded."
                    )
            self.recording_service.pause_event.set()
            recording_started = self.recording_service.start_recording()
            if not recording_started:
                raise RuntimeError("Dataset recorder is already collecting.")
            recording_active = True
            next_tick = time.perf_counter()
            pending_next_joint_frame = None
            for frame_index, target in enumerate(targets):
                self._check_replay_abort()
                command_timestamp_ns = time.monotonic_ns()
                self.backend.command_joint_configuration(target, period)
                sample = self.read_sample()
                frame = self._frame_from_sample(
                    sample,
                    action=target,
                    action_timestamp_ns=command_timestamp_ns,
                )
                frame_wire_bits = int(
                    getattr(frame.beaver_data, "wire_distance_bits", 0)
                )
                if frame_wire_bits != 16:
                    raise RuntimeError(
                        "The Beaver frame selected for this robot sample was not "
                        "16-bit; the partial episode will be discarded."
                    )
                if self.cfg.TEACH_ACTION_MODE == "command":
                    self.recording_service.record_frame(frame)
                else:
                    if pending_next_joint_frame is not None:
                        self.recording_service.record_frame(
                            self._frame_with_action(
                                pending_next_joint_frame,
                                sample.joints,
                                sample.timestamp_ns,
                            )
                        )
                    pending_next_joint_frame = frame
                with self._sample_lock:
                    self._latest_sample = sample
                with self._state_lock:
                    self._replay_frame = frame_index + 1
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    self._stop_event.wait(delay)
                else:
                    next_tick = time.perf_counter()

            if pending_next_joint_frame is not None:
                final_timestamp_ns = int(
                    np.asarray(
                        pending_next_joint_frame.extra_data[
                            "robot_state_timestamp_ns"
                        ]
                    ).item()
                )
                self.recording_service.record_frame(
                    self._frame_with_action(
                        pending_next_joint_frame,
                        pending_next_joint_frame.state,
                        final_timestamp_ns,
                    )
                )

            expected_length = len(targets)
            if int(self.dataset.collect_step) != expected_length:
                raise RuntimeError(
                    f"Recorded {self.dataset.collect_step} frames; expected "
                    f"{expected_length}."
                )
            self._set_state(
                RecollectState.EXPORTING,
                f"Encoding and exporting episode {source_episode_index}.",
            )
            if not self.recording_service.stop_recording():
                raise RuntimeError("Failed to queue recollect episode export.")
            recording_active = False
            self.recording_service.pause_event.clear()
            self._wait_for_pending_dataset_action()
            if int(self.dataset.recorded_episodes) != episode_before + 1:
                raise RuntimeError("Replay finished, but the episode was not exported.")
            episode_exported = True
            with self._state_lock:
                if replacement:
                    self._replaced_source_episodes[int(source_episode_index)] = (
                        expected_length
                    )
                self._position = int(self.dataset.recorded_episodes)
            next_source = self.current_source_episode
            next_note = (
                " Recollection complete."
                if next_source is None
                else f" Next slot is original source episode {next_source}."
            )
            utils.logger.info(
                "%s source episode %d: %d frames exported.%s",
                "Taught replacement for" if replacement else "Recollected",
                source_episode_index,
                expected_length,
                next_note,
            )
            self.teach_state = TeachState.READY
            self.taught_trajectory.clear()
            self._teach_start_joints = None
            self._prepare_current_episode()
        except Exception:
            self._request_motion_stop()
            if (
                not episode_exported
                and recording_started
                and (recording_active or int(self.dataset.collect_step) > 0)
            ):
                self.recording_service.request_rollback()
                self.recording_service.pause_event.clear()
                self._wait_for_pending_dataset_action()
            raise
        finally:
            self.recording_service.pause_event.clear()
            self.teach_state = TeachState.READY
            self._collecting_replacement = False
            self._abort_replay.clear()

    def _rollback_previous_episode(self) -> None:
        if int(self.dataset.recorded_episodes) <= 0:
            raise RuntimeError("No recollected episode is available to undo.")
        self.recording_service.request_rollback()
        self._wait_for_pending_dataset_action()
        with self._state_lock:
            self._position = int(self.dataset.recorded_episodes)
            if self._position < len(self.episodes):
                self._replaced_source_episodes.pop(
                    self.episodes[self._position].source_episode_index,
                    None,
                )
        self._prepare_current_episode()

    def live_image(self, camera_name: str) -> np.ndarray:
        if self.camera_manager is None:
            raise KeyError("No camera manager is configured.")
        with self.camera_manager._lock:
            if camera_name not in self.camera_manager.camera_images:
                raise KeyError(f"Unknown live camera {camera_name!r}.")
            return np.asarray(
                self.camera_manager.camera_images[camera_name]
            ).copy()

    def reference_image(self, camera_name: str) -> np.ndarray:
        with self._state_lock:
            if camera_name not in self._reference_images:
                raise KeyError(f"No source first frame for camera {camera_name!r}.")
            return self._reference_images[camera_name].copy()

    def beaver_visualization(self) -> dict[str, Any]:
        """Return a lightweight, JSON-safe live Beaver heatmap payload."""
        layout = tuple(self.cfg.BEAVER_SENSOR_LAYOUT)
        payload: dict[str, Any] = {
            "enabled": bool(self.cfg.beaver_enable),
            "connected": False,
            "stale": True,
            "error": "Beaver reader is not configured",
            "grid_width": int(self.cfg.BEAVER_GRID_WIDTH),
            "wire_distance_bits": 0,
            "sequence": 0,
            "frame_count": 0,
            "lost_frames": 0,
            "age_ms": None,
            "max_display_mm": int(self.cfg.BEAVER_VISUALIZER_MAX_MM),
            "sensor_layout": [list(sensor) for sensor in layout],
            "distance_mm": [],
            "target_status": [],
            "present": [],
        }
        if self.beaver_reader is None:
            return payload

        snapshot = self.beaver_reader.snapshot()
        age_s = float(snapshot.age_s())
        snapshot_layout = tuple(getattr(snapshot, "sensor_layout", layout))
        payload.update(
            connected=bool(getattr(snapshot, "connected", False)),
            stale=age_s > self.cfg.BEAVER_STALE_AFTER_S,
            error=str(getattr(snapshot, "error", "")),
            grid_width=int(getattr(snapshot, "grid_width", 0)),
            wire_distance_bits=int(
                getattr(snapshot, "wire_distance_bits", 0)
            ),
            sequence=int(getattr(snapshot, "sequence", 0)),
            frame_count=int(getattr(snapshot, "frame_count", 0)),
            lost_frames=int(getattr(snapshot, "lost_frames", 0)),
            age_ms=None if not np.isfinite(age_s) else round(age_s * 1000.0, 1),
            sensor_layout=[list(sensor) for sensor in snapshot_layout],
            distance_mm=np.asarray(snapshot.distance_mm, dtype=np.uint16).tolist(),
            target_status=np.asarray(
                snapshot.target_status,
                dtype=np.uint8,
            ).tolist(),
            present=np.asarray(snapshot.present, dtype=np.uint8).tolist(),
        )
        return payload

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            state = self._state
            message = self._message
            position = self._position
            replay_frame = self._replay_frame
            command_pending = self._command_pending
            reference_names = sorted(self._reference_images)
            reference_generation = int(self._reference_generation)
            trajectory_safety = dict(self._trajectory_safety)
            trajectory_safety["repairs"] = [
                dict(repair) for repair in self._trajectory_safety["repairs"]
            ]
            collecting_replacement = bool(self._collecting_replacement)
            replaced_source_episodes = dict(self._replaced_source_episodes)
        completed = int(self.dataset.recorded_episodes)
        current = self.current_source_episode
        current_length = (
            None if position >= len(self.episodes) else self.episodes[position].length
        )
        taught_frames = len(self.taught_trajectory)
        if state in {RecollectState.TEACHING, RecollectState.TEACH_READY}:
            replay_frame = taught_frames
        if state is RecollectState.REPLAYING and collecting_replacement:
            current_length = max(current_length or 0, replay_frame)
        issues = (
            self.readiness_issues()
            if state in {RecollectState.CALIBRATING, RecollectState.TEACH_READY}
            else []
        )
        live_names = []
        if self.camera_manager is not None:
            with self.camera_manager._lock:
                live_names = sorted(self.camera_manager.camera_images)
        beaver = {
            "enabled": bool(self.cfg.beaver_enable),
            "precision": "waiting for 16-bit",
            "connected": False,
            "error": "",
        }
        if self.beaver_reader is not None:
            snapshot = self.beaver_reader.snapshot()
            wire_bits = int(getattr(snapshot, "wire_distance_bits", 0))
            beaver.update(
                connected=bool(getattr(snapshot, "connected", False)),
                error=str(getattr(snapshot, "error", "")),
                grid_width=int(getattr(snapshot, "grid_width", 0)),
                wire_distance_bits=wire_bits,
                precision=(
                    "16-bit raw"
                    if wire_bits == 16
                    else (
                        f"{wire_bits}-bit legacy"
                        if wire_bits
                        else "waiting for 16-bit"
                    )
                ),
                frame_count=int(getattr(snapshot, "frame_count", 0)),
                lost_frames=int(getattr(snapshot, "lost_frames", 0)),
            )
        episode_rows = []
        for offset, descriptor in enumerate(self.episodes):
            if offset < completed:
                episode_state = "done"
            elif offset == position and state is not RecollectState.COMPLETE:
                episode_state = "current"
            else:
                episode_state = "pending"
            replaced_length = replaced_source_episodes.get(
                descriptor.source_episode_index
            )
            episode_rows.append(
                {
                    "source_episode_index": descriptor.source_episode_index,
                    "length": descriptor.length,
                    "status": episode_state,
                    "replaced": replaced_length is not None,
                    "recorded_length": replaced_length,
                }
            )
        return {
            "source_dataset": str(self.source_dataset.root),
            "output_dataset": str(self.dataset.dataset_dir),
            "fps": self.source_dataset.fps,
            "total_episodes": len(self.episodes),
            "total_frames": sum(episode.length for episode in self.episodes),
            "completed_episodes": completed,
            "current_source_episode": current,
            "current_episode_length": current_length,
            "replay_frame": replay_frame,
            "taught_frames": taught_frames,
            "collecting_replacement": collecting_replacement
            or state
            in {RecollectState.TEACHING, RecollectState.TEACH_READY},
            "state": state.value,
            "message": message,
            "command_pending": command_pending,
            "readiness_issues": issues,
            "cameras": sorted(set(reference_names) | set(live_names)),
            "reference_cameras": reference_names,
            "reference_generation": reference_generation,
            "live_cameras": live_names,
            "beaver": beaver,
            "trajectory_safety": trajectory_safety,
            "episodes": episode_rows,
            "controls": {
                "replay_enabled": state is RecollectState.CALIBRATING
                and not command_pending,
                "replay_ready": state is RecollectState.CALIBRATING
                and not issues
                and not command_pending,
                "teach_enabled": state
                in {RecollectState.CALIBRATING, RecollectState.TEACH_READY}
                and not command_pending,
                "end_teach_enabled": state is RecollectState.TEACHING
                and not command_pending,
                "teach_collect_enabled": state is RecollectState.TEACH_READY
                and not command_pending,
                "teach_collect_ready": state is RecollectState.TEACH_READY
                and not issues
                and bool(self.taught_trajectory)
                and not command_pending,
                "cancel_teach_enabled": state
                in {RecollectState.TEACHING, RecollectState.TEACH_READY}
                and not command_pending,
                "retry_enabled": state is RecollectState.ERROR and not command_pending,
                "rollback_enabled": completed > 0
                and state
                in {
                    RecollectState.CALIBRATING,
                    RecollectState.TEACHING,
                    RecollectState.TEACH_READY,
                    RecollectState.ERROR,
                    RecollectState.COMPLETE,
                }
                and not command_pending,
                "stop_enabled": state
                in {
                    RecollectState.MOVING_TO_START,
                    RecollectState.TEACHING,
                    RecollectState.REPLAYING,
                    RecollectState.EXPORTING,
                },
            },
        }

    def close(self) -> None:
        if getattr(self, "_state", None) is RecollectState.CLOSED:
            return
        if hasattr(self, "_stop_event"):
            self._abort_replay.set()
            self._stop_event.set()
            try:
                self._request_motion_stop()
            except Exception:
                utils.logger.exception("Failed to request a RealMan stop during close")
        worker = getattr(self, "_worker", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=30.0)
        try:
            super().close()
        finally:
            if hasattr(self, "_state_lock"):
                self._set_state(RecollectState.CLOSED, "Recollector closed.")


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=(
            "Replay a RealMan LeRobot dataset and recollect synchronized robot, "
            "force, camera, and raw 16-bit Beaver observations. The UI can also "
            "drag-teach a replacement for the current source episode; the next "
            "slot still loads the original dataset's next episode."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--from-episode", type=int, default=0)
    parser.add_argument("--to-episode", type=int)
    parser.add_argument(
        "--source",
        choices=["observation.state", "action"],
        default="observation.state",
    )
    parser.add_argument("--robot-ip", default=defaults.ROBOT_IP)
    parser.add_argument("--port", type=int, default=defaults.REALMAN_PORT)
    parser.add_argument("--initial-speed", type=float, default=defaults.RESET_JOINT_SPEED)
    parser.add_argument("--max-joint-speed", type=float, default=2.5)
    parser.add_argument(
        "--joint-jump-policy",
        choices=["original", "error", "interpolate"],
        default="interpolate",
        help=(
            "How to handle isolated source frame-to-frame joint jumps above the "
            "safety limit. 'original' preserves inspected source targets and "
            "reports a warning (default); 'error' rejects them; 'interpolate' preserves "
            "episode length and endpoints while smoothing the smallest local "
            "window; 'original' explicitly replays inspected source targets "
            "unchanged and reports an operator warning."
        ),
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--beaver-port", default=defaults.BEAVER_PORT)
    parser.add_argument(
        "--beaver-grid-width",
        type=int,
        choices=[4, 8],
        default=defaults.BEAVER_GRID_WIDTH,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and validate selected episodes without opening hardware or output.",
    )
    return parser


def build_config(
    args: argparse.Namespace,
    source: RealManLeRobotDataset,
) -> Config:
    if args.initial_speed <= 0:
        raise ValueError("--initial-speed must be positive.")
    fps = normalize_lerobot_fps(source.fps)
    if fps >= 100:
        raise ValueError("Source FPS must be below 100 for low-follow recollection.")
    return Config(
        ROBOT_TYPE="realman",
        ROBOT_IP=args.robot_ip,
        REALMAN_PORT=args.port,
        RESET_JOINT_SPEED=args.initial_speed,
        GRIPPER=False,
        TORQUE_MODE=False,
        DATASET_TYPE="l",
        DATA_TYPE="both",
        TASK_NAME=args.task or source.root.name,
        COLLECT_RATE=fps,
        PUSH_TO_HUB=False,
        FORCE_COLLECT=True,
        TORQUE_COLLECT=True,
        DEPTH_INFO_ENABLE=False,
        TACTILE_ENABLE=False,
        TACTILE_TRANSFER=False,
        REALSENSE_RESOLUTION=source_camera_resolution(source),
        BEAVER_PORT=args.beaver_port,
        BEAVER_GRID_WIDTH=args.beaver_grid_width,
        BEAVER_SIMULATE_8BIT=False,
        beaver_enable=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = RealManLeRobotDataset(
        args.dataset_root,
        control_mode="joint",
        source=args.source,
    )
    selected = source.episode_indices(
        episodes=args.episodes,
        from_episode=args.from_episode,
        to_episode=args.to_episode,
    )
    if len(set(selected)) != len(selected):
        raise ValueError("--episodes must not contain duplicate indices.")
    if not source.video_keys:
        raise ValueError(
            "The source dataset has no video camera feature for frame-0 calibration."
        )
    for index in selected:
        length = source.episode_length(index)
        source_trajectory = source.load_episode(index)
        original_max_joint_speed = source_trajectory.maximum_joint_speed
        repairs = ()
        if args.joint_jump_policy == "interpolate":
            trajectory, repairs = interpolate_joint_speed_discontinuities(
                source_trajectory,
                args.max_joint_speed,
            )
        elif args.joint_jump_policy == "error":
            trajectory = source_trajectory
            validate_trajectory_speed(trajectory, args.max_joint_speed)
        else:
            trajectory = source_trajectory
        utils.logger.info(
            "Episode %d: %d frames, %.2f s, max joint speed %.3f rad/s.",
            index,
            length,
            trajectory.duration_s,
            trajectory.maximum_joint_speed,
        )
        if repairs:
            repair_windows = ", ".join(
                f"J{repair.joint_index} frames "
                f"{repair.window_start_frame}-{repair.window_end_frame}"
                for repair in repairs
            )
            utils.logger.warning(
                "Episode %d: %d unsafe source discontinuity(s) will be locally "
                "interpolated (%s); peak %.3f → %.3f rad/s.",
                index,
                len(repairs),
                repair_windows,
                original_max_joint_speed,
                trajectory.maximum_joint_speed,
            )
        elif trajectory.maximum_joint_speed > args.max_joint_speed:
            utils.logger.warning(
                "Episode %d: explicit original-trajectory mode preserves the "
                "%.3f rad/s source peak unchanged (warning limit %.3f rad/s).",
                index,
                trajectory.maximum_joint_speed,
                args.max_joint_speed,
            )
    if args.dry_run:
        utils.logger.info("Dry run complete; no hardware or output was opened.")
        return 0

    output_root = validate_output_location(
        source,
        args.output_dataset or default_output_root(source.root),
        selected,
    )
    cfg = build_config(args, source)
    camera_manager = create_camera_manager(cfg)
    if camera_manager.camera_num != len(source.video_keys):
        camera_manager.close()
        raise RuntimeError(
            f"Source has {len(source.video_keys)} cameras, but only "
            f"{camera_manager.camera_num} live RealSense cameras were detected."
        )

    from doffy_teleop.sensors.beaver import BeaverReader
    from doffy_teleop.recording.recollect_ui.app import RecollectController, create_app
    import uvicorn

    beaver_reader = BeaverReader.from_config(cfg)
    recorder = None
    recollector = None
    try:
        recorder = DatasetRecorder(
            camera_num=camera_manager.camera_num,
            robot_dof=7,
            robot_type="realman",
            force_collect=True,
            torque_collect=True,
            gripper=False,
            config=cfg,
            dataset_root=output_root,
            recreate_on_schema_mismatch=False,
        )
        write_recollect_manifest(output_root, source, selected)
        recollector = RealManRecollector(
            cfg,
            source,
            episode_indices=selected,
            maximum_joint_speed=args.max_joint_speed,
            joint_jump_policy=args.joint_jump_policy,
            dataset=recorder,
            camera_manager=camera_manager,
            beaver_reader=beaver_reader,
        )
        controller = RecollectController(recollector)
        application = create_app(controller)
        recollector.start()
        url = f"http://{args.host}:{args.ui_port}"
        print(f"Source: {source.root}")
        print(f"Output: {output_root}")
        print(f"Episodes: {len(selected)}; raw Beaver precision: 16-bit")
        print(f"Open {url}")
        if not args.no_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        uvicorn.run(application, host=args.host, port=args.ui_port, log_level="info")
    finally:
        if recollector is not None:
            recollector.close()
        else:
            beaver_reader.close()
            camera_manager.close()
            if recorder is not None:
                recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
