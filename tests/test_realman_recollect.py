from __future__ import annotations

import argparse
from collections import deque
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from doffy_teleop.sensors.beaver import BeaverSnapshot
from doffy_teleop.config import Config
from doffy_teleop.recording.recollect_ui.app import RecollectController
from doffy_teleop.recording.replay import EpisodeTrajectory
from doffy_teleop.runtime.recollect import (
    RealManRecollector,
    RecollectState,
    build_config,
    build_parser,
    default_output_root,
)


class FakeBackend:
    name = "realman"
    supports_freedrive = True
    supports_force = True
    dataset_robot_type = "realman"
    dof = 7

    def __init__(self) -> None:
        self.joints = np.zeros(7, dtype=float)
        self.wrench = np.arange(6, dtype=float)
        self.commanded = []
        self.reset_targets = []
        self.cleaned = False
        self.freedrive_started = False
        self.freedrive_stopped = False
        self.freedrive_sensitivity = None

    def get_joint_configuration(self):
        return self.joints.copy()

    def get_tcp_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return pose

    def get_tcp_force(self):
        return self.wrench.copy()

    def command_joint_configuration(self, joints, dt):
        self.joints = np.asarray(joints, dtype=float).copy()
        self.commanded.append((self.joints.copy(), dt))

    def reset(self, joints):
        self.joints = np.asarray(joints, dtype=float).copy()
        self.reset_targets.append(self.joints.copy())

    def start_freedrive(self):
        self.freedrive_started = True

    def stop_freedrive(self):
        self.freedrive_stopped = True

    def set_freedrive_sensitivity(self, grade):
        self.freedrive_sensitivity = grade

    def cleanup(self):
        self.cleaned = True


class FakeDataset:
    def __init__(self, root: Path) -> None:
        self.dataset_dir = root
        self.recorded_episodes = 0
        self.collect_step = 0
        self.frames = []
        self.exports = 0
        self.rollbacks = 0
        self.closed = False

    def data_collection(self, **frame):
        self.frames.append(frame)
        self.collect_step += 1

    def data_export(self, _context):
        self.exports += 1
        self.recorded_episodes += 1

    def _reset_data_dict(self):
        self.collect_step = 0

    def rollback_last_episode(self):
        self.rollbacks += 1
        if self.collect_step:
            self.collect_step = 0
            return True
        if self.recorded_episodes:
            self.recorded_episodes -= 1
            return True
        return False

    def recording_status(self, collecting=False):
        return {
            "recorded_episodes": self.recorded_episodes,
            "current_episode_frames": self.collect_step,
            "collecting": collecting,
        }

    def close(self):
        self.closed = True


class FakeCameraManager:
    def __init__(self) -> None:
        self.camera_num = 1
        self.depth_mode = False
        self._lock = threading.Lock()
        self.camera_images = {
            "camera_0": np.full((12, 16, 3), 80, dtype=np.uint8)
        }
        self.camera_image_timestamps_ns = {"camera_0": 100}
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True


class FakeBeaverReader:
    def __init__(self) -> None:
        distance = np.full((9, 4, 4), 1234, dtype=np.uint16)
        self.value = BeaverSnapshot(
            distance_mm=distance,
            target_status=np.full((9, 4, 4), 5, dtype=np.uint8),
            present=np.ones(9, dtype=np.uint8),
            valid_count=np.full(9, 16, dtype=np.uint8),
            average_mm=np.full(9, 1234, dtype=np.uint16),
            temperature_c=np.full(9, 22, dtype=np.int16),
            stream_count=np.ones(9, dtype=np.uint32),
            sensor_layout=tuple((0, index) for index in range(9)),
            grid_width=4,
            wire_distance_bits=16,
            sequence=7,
            timestamp_ns=time.monotonic_ns(),
            frame_count=4,
            connected=True,
            error="",
        )
        self.closed = False

    def snapshot(self):
        return self.value

    def snapshot_nearest(self, _timestamp_ns):
        return self.value

    def start(self, _stop_event):
        pass

    def close(self):
        self.closed = True


class FakeSource:
    control_mode = "joint"
    fps = 50.0
    video_keys = ["observation.images.camera_0"]

    def __init__(self, root: Path, episode_count: int = 1) -> None:
        self.root = root
        self.total_episodes = episode_count
        self.info = {
            "features": {
                "observation.images.camera_0": {
                    "shape": [12, 16, 3],
                }
            }
        }
        self._trajectories = {}
        for index in range(episode_count):
            offset = 0.01 * index
            targets = np.stack(
                [
                    np.full(7, offset),
                    np.full(7, offset + 0.005),
                    np.full(7, offset + 0.01),
                ]
            )
            if index > 0:
                targets = np.vstack([targets, np.full(7, offset + 0.015)])
            self._trajectories[index] = EpisodeTrajectory(
                index, targets, self.fps, "joint"
            )

    @property
    def trajectory(self):
        return self._trajectories[0]

    @trajectory.setter
    def trajectory(self, value):
        self._trajectories[0] = value

    def episode_indices(self, *, episodes=None, from_episode=0, to_episode=None):
        if episodes is not None:
            return list(episodes)
        return list(range(from_episode, self.total_episodes if to_episode is None else to_episode))

    def episode_length(self, episode_index):
        return len(self._trajectories[episode_index].targets)

    def load_episode(self, episode_index):
        return self._trajectories[episode_index]

    def load_first_camera_frames(self, episode_index):
        if episode_index not in self._trajectories:
            raise IndexError(episode_index)
        return {
            "observation.images.camera_0": np.full(
                (12, 16, 3), 40 + episode_index, dtype=np.uint8
            )
        }


def recollect_config() -> Config:
    return Config(
        ROBOT_TYPE="realman",
        GRIPPER=False,
        DATASET_TYPE="l",
        DATA_TYPE="both",
        FORCE_COLLECT=True,
        TORQUE_COLLECT=True,
        DEPTH_INFO_ENABLE=False,
        TACTILE_TRANSFER=False,
        COLLECT_RATE=50,
        FORCE_MOVING_AVERAGE_WINDOW=1,
        FORCE_LOW_PASS_ALPHA=0.0,
        BEAVER_SIMULATE_8BIT=False,
        beaver_enable=True,
        TEACH_ACTION_MODE="next_joint",
    )


class RealManRecollectorTests(unittest.TestCase):
    @staticmethod
    def _config_args():
        return argparse.Namespace(
            initial_speed=0.2,
            robot_ip="127.0.0.1",
            port=8080,
            task=None,
            beaver_port=None,
            beaver_grid_width=4,
        )

    def test_build_config_preserves_lerobot_fps_as_integer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = build_config(
                self._config_args(),
                FakeSource(Path(directory) / "source"),
            )

        self.assertEqual(cfg.COLLECT_RATE, 50)
        self.assertIs(type(cfg.COLLECT_RATE), int)

    def test_build_config_rejects_fractional_lerobot_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = FakeSource(Path(directory) / "source")
            source.fps = 29.97
            with self.assertRaisesRegex(ValueError, "integer FPS"):
                build_config(self._config_args(), source)

    def test_default_output_is_source_name_plus_recollect(self) -> None:
        self.assertEqual(
            default_output_root("/tmp/task_lero"),
            Path("/tmp/task_lero_recollect_2"),
        )

    def test_rejects_legacy_beaver_quantization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = recollect_config()
            cfg.BEAVER_SIMULATE_8BIT = True
            with self.assertRaisesRegex(ValueError, "raw 16-bit"):
                RealManRecollector(
                    cfg,
                    FakeSource(Path(directory) / "source"),
                    backend=FakeBackend(),
                    dataset=FakeDataset(Path(directory) / "output"),
                )

    def test_legacy_wire_frames_keep_replay_not_ready_but_button_clickable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            beaver = FakeBeaverReader()
            object.__setattr__(beaver.value, "wire_distance_bits", 8)
            collector = RealManRecollector(
                recollect_config(),
                FakeSource(Path(directory) / "source"),
                backend=FakeBackend(),
                dataset=FakeDataset(Path(directory) / "output"),
                camera_manager=FakeCameraManager(),
                beaver_reader=beaver,
            )
            collector._prepare_current_episode()

            status = collector.status()
            self.assertTrue(status["controls"]["replay_enabled"])
            self.assertFalse(status["controls"]["replay_ready"])
            self.assertIn("legacy 8-bit", " ".join(status["readiness_issues"]))
            self.assertEqual(status["beaver"]["precision"], "8-bit legacy")
            with self.assertRaises(HTTPException) as raised:
                RecollectController(collector).command("continue")
            self.assertIn("legacy 8-bit", raised.exception.detail)
            collector.close()

    def test_default_jump_policy_preserves_original_trajectory(self) -> None:
        args = build_parser().parse_args(["--dataset-root", "/tmp/source"])

        self.assertEqual(args.joint_jump_policy, "interpolate")

    def test_control_panel_precedes_compact_beaver_grid_without_hand_controls(self) -> None:
        static = (
            Path(__file__).resolve().parents[1]
            / "doffy_teleop"
            / "recording"
            / "recollect_ui"
            / "static"
        )
        index = (static / "index.html").read_text(encoding="utf-8")
        script = (static / "app.js").read_text(encoding="utf-8")
        style = (static / "style.css").read_text(encoding="utf-8")

        self.assertLess(
            index.index('class="control-card"'),
            index.index('class="beaver-card"'),
        )
        self.assertNotIn("Open hand", index)
        self.assertNotIn("grab-button", script)
        self.assertNotIn("scrollIntoView", script)
        self.assertIn("height: 24px", style)
        self.assertIn("position: sticky", style)
        self.assertNotIn("forceReference || !ui.reference_image.src", script)
        self.assertIn("dataset.loadedUrl", script)
        self.assertIn("reference_generation", script)

    def test_prepare_pauses_at_first_frame_and_reports_episode_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend()
            collector = RealManRecollector(
                recollect_config(),
                FakeSource(Path(directory) / "source"),
                backend=backend,
                dataset=FakeDataset(Path(directory) / "output"),
                camera_manager=FakeCameraManager(),
                beaver_reader=FakeBeaverReader(),
            )
            collector._prepare_current_episode()

            self.assertEqual(collector.state, RecollectState.CALIBRATING)
            np.testing.assert_allclose(backend.reset_targets[-1], np.zeros(7))
            status = collector.status()
            self.assertEqual(status["episodes"][0]["length"], 3)
            self.assertEqual(status["current_source_episode"], 0)
            self.assertTrue(status["controls"]["replay_enabled"])
            self.assertEqual(status["beaver"]["precision"], "16-bit raw")
            self.assertEqual(collector.reference_image("camera_0")[0, 0, 0], 40)
            self.assertGreaterEqual(status["reference_generation"], 1)
            self.assertEqual(status["reference_cameras"], ["camera_0"])
            collector.close()

    def test_prepare_interpolates_and_reports_unsafe_source_jump(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = FakeSource(Path(directory) / "source")
            targets = np.zeros((6, 7), dtype=float)
            targets[:, 0] = [0.0, 0.01, 0.02, 0.12, 0.13, 0.14]
            source.trajectory = EpisodeTrajectory(0, targets, source.fps, "joint")
            collector = RealManRecollector(
                recollect_config(),
                source,
                maximum_joint_speed=2.5,
                joint_jump_policy="interpolate",
                backend=FakeBackend(),
                dataset=FakeDataset(Path(directory) / "output"),
                camera_manager=FakeCameraManager(),
                beaver_reader=FakeBeaverReader(),
            )

            collector._prepare_current_episode()

            status = collector.status()
            self.assertEqual(collector.state, RecollectState.CALIBRATING)
            self.assertTrue(status["trajectory_safety"]["repaired"])
            self.assertGreater(status["trajectory_safety"]["repair_count"], 0)
            self.assertAlmostEqual(
                status["trajectory_safety"]["original_max_joint_speed"],
                5.0,
            )
            self.assertLessEqual(
                status["trajectory_safety"]["replay_max_joint_speed"],
                2.5,
            )
            self.assertIn("Safety interpolation", status["message"])
            collector.close()

    def test_original_policy_preserves_inspected_fast_grasp_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = FakeSource(Path(directory) / "source")
            targets = np.zeros((6, 7), dtype=float)
            targets[:, 0] = [0.0, 0.01, 0.02, 0.12, 0.13, 0.14]
            source.trajectory = EpisodeTrajectory(0, targets, source.fps, "joint")
            backend = FakeBackend()
            collector = RealManRecollector(
                recollect_config(),
                source,
                maximum_joint_speed=2.5,
                joint_jump_policy="original",
                backend=backend,
                dataset=FakeDataset(Path(directory) / "output"),
                camera_manager=FakeCameraManager(),
                beaver_reader=FakeBeaverReader(),
            )

            collector._prepare_current_episode()

            status = collector.status()
            self.assertTrue(status["trajectory_safety"]["unsafe_original"])
            self.assertFalse(status["trajectory_safety"]["repaired"])
            self.assertEqual(status["trajectory_safety"]["policy"], "original")
            np.testing.assert_array_equal(collector._trajectory.targets, targets)
            self.assertIn("preserve", status["message"])
            collector._replay_current_episode()
            np.testing.assert_array_equal(
                np.stack([command[0] for command in backend.commanded]),
                targets,
            )
            collector.close()

    def test_replay_records_exact_length_and_raw_beaver_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend()
            dataset = FakeDataset(Path(directory) / "output")
            beaver = FakeBeaverReader()
            collector = RealManRecollector(
                recollect_config(),
                FakeSource(Path(directory) / "source"),
                backend=backend,
                dataset=dataset,
                camera_manager=FakeCameraManager(),
                beaver_reader=beaver,
            )
            collector._prepare_current_episode()
            collector._replay_current_episode()

            self.assertEqual(collector.state, RecollectState.COMPLETE)
            self.assertEqual(dataset.exports, 1)
            self.assertEqual(dataset.recorded_episodes, 1)
            self.assertEqual(len(dataset.frames), 3)
            self.assertEqual(len(backend.commanded), 3)
            self.assertEqual(
                int(dataset.frames[0]["beaver_data"].distance_mm[0, 0, 0]),
                1234,
            )
            np.testing.assert_allclose(
                dataset.frames[0]["action"], np.full(7, 0.005)
            )
            np.testing.assert_allclose(
                dataset.frames[-1]["action"], np.full(7, 0.01)
            )
            controller = RecollectController(collector)
            self.assertEqual(controller.state()["completed_episodes"], 1)
            beaver_view = controller.beaver()
            self.assertEqual(beaver_view["wire_distance_bits"], 16)
            self.assertEqual(beaver_view["grid_width"], 4)
            self.assertEqual(len(beaver_view["distance_mm"]), 9)
            self.assertEqual(beaver_view["distance_mm"][0][0][0], 1234)
            self.assertEqual(beaver_view["target_status"][0][0][0], 5)
            self.assertEqual(beaver_view["sensor_layout"][0], [0, 0])
            self.assertTrue(
                controller.camera_jpeg("live", "camera_0").startswith(b"\xff\xd8")
            )
            collector.close()

    def test_enter_command_runs_the_threaded_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = FakeDataset(Path(directory) / "output")
            collector = RealManRecollector(
                recollect_config(),
                FakeSource(Path(directory) / "source"),
                backend=FakeBackend(),
                dataset=dataset,
                camera_manager=FakeCameraManager(),
                beaver_reader=FakeBeaverReader(),
            )
            collector.start()
            deadline = time.monotonic() + 2.0
            while collector.state is not RecollectState.CALIBRATING:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)

            result = RecollectController(collector).command("continue")
            self.assertTrue(result["accepted"])
            while collector.state is not RecollectState.COMPLETE:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)

            self.assertEqual(dataset.recorded_episodes, 1)
            self.assertEqual(len(dataset.frames), 3)
            collector.close()

    def test_teach_replacement_collects_new_length_then_loads_next_source_episode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = recollect_config()
            cfg.TEACH_INITIAL_DISCARD_FRAMES = 0
            backend = FakeBackend()
            dataset = FakeDataset(Path(directory) / "output")
            collector = RealManRecollector(
                cfg,
                FakeSource(Path(directory) / "source", episode_count=2),
                backend=backend,
                dataset=dataset,
                camera_manager=FakeCameraManager(),
                beaver_reader=FakeBeaverReader(),
            )
            collector._prepare_current_episode()
            self.assertEqual(collector.current_source_episode, 0)
            self.assertTrue(collector.status()["controls"]["teach_enabled"])

            collector._begin_replacement_teach()
            self.assertEqual(collector.state, RecollectState.TEACHING)
            self.assertTrue(getattr(backend, "freedrive_started", False))
            for step in range(1, 6):
                backend.joints = np.full(7, 0.04 * step)
                collector.capture_teach_sample(collector.read_sample())
            collector._finish_replacement_teach()
            self.assertEqual(collector.state, RecollectState.TEACH_READY)
            taught_frames = len(collector.taught_trajectory)
            self.assertGreater(taught_frames, 3)
            status = collector.status()
            self.assertTrue(status["controls"]["teach_collect_enabled"])
            self.assertFalse(status["controls"]["replay_enabled"])
            self.assertEqual(status["taught_frames"], taught_frames)

            collector._collect_taught_episode()

            self.assertEqual(dataset.exports, 1)
            self.assertEqual(dataset.recorded_episodes, 1)
            self.assertEqual(len(dataset.frames), taught_frames)
            self.assertEqual(collector.current_source_episode, 1)
            self.assertEqual(collector.state, RecollectState.CALIBRATING)
            status = collector.status()
            self.assertTrue(status["episodes"][0]["replaced"])
            self.assertEqual(status["episodes"][0]["recorded_length"], taught_frames)
            self.assertEqual(status["episodes"][1]["status"], "current")
            self.assertEqual(status["current_episode_length"], 4)
            np.testing.assert_allclose(
                backend.reset_targets[-1],
                np.full(7, 0.01),
            )
            collector.close()

    def test_cancel_teach_keeps_the_current_source_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = recollect_config()
            cfg.TEACH_INITIAL_DISCARD_FRAMES = 0
            backend = FakeBackend()
            collector = RealManRecollector(
                cfg,
                FakeSource(Path(directory) / "source", episode_count=2),
                backend=backend,
                dataset=FakeDataset(Path(directory) / "output"),
                camera_manager=FakeCameraManager(),
                beaver_reader=FakeBeaverReader(),
            )
            collector._prepare_current_episode()
            collector._begin_replacement_teach()
            backend.joints = np.full(7, 0.2)
            collector.capture_teach_sample(collector.read_sample())
            collector._cancel_replacement_teach()

            self.assertEqual(collector.state, RecollectState.CALIBRATING)
            self.assertEqual(collector.current_source_episode, 0)
            self.assertEqual(collector.taught_trajectory, [])
            self.assertTrue(collector.status()["controls"]["replay_enabled"])
            np.testing.assert_allclose(backend.reset_targets[-1], np.zeros(7))
            collector.close()

    def test_end_teach_returns_to_the_taught_trajectory_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = recollect_config()
            cfg.TEACH_INITIAL_DISCARD_FRAMES = 0
            backend = FakeBackend()
            collector = RealManRecollector(
                cfg,
                FakeSource(Path(directory) / "source"),
                backend=backend,
                dataset=FakeDataset(Path(directory) / "output"),
                camera_manager=FakeCameraManager(),
                beaver_reader=FakeBeaverReader(),
            )
            collector._prepare_current_episode()
            collector._begin_replacement_teach()
            taught_start = np.asarray(collector.taught_trajectory[0], dtype=float)
            backend.joints = np.full(7, 0.35)
            collector.capture_teach_sample(collector.read_sample())

            collector._finish_replacement_teach()

            self.assertEqual(collector.state, RecollectState.TEACH_READY)
            np.testing.assert_allclose(backend.reset_targets[-1], taught_start)
            np.testing.assert_allclose(backend.joints, taught_start)
            np.testing.assert_allclose(
                collector.taught_trajectory[0], taught_start
            )
            collector.close()

    def test_ui_exposes_replacement_teach_controls(self) -> None:
        static = (
            Path(__file__).resolve().parents[1]
            / "doffy_teleop"
            / "recording"
            / "recollect_ui"
            / "static"
        )
        index = (static / "index.html").read_text(encoding="utf-8")
        script = (static / "app.js").read_text(encoding="utf-8")
        self.assertIn("Teach replacement", index)
        self.assertIn("Collect replacement", index)
        self.assertIn('command("teach")', script)
        self.assertIn("teach_collect", script)
        self.assertIn("original dataset episode", index)

    def test_controller_rejects_teach_collect_before_end_teach(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collector = RealManRecollector(
                recollect_config(),
                FakeSource(Path(directory) / "source"),
                backend=FakeBackend(),
                dataset=FakeDataset(Path(directory) / "output"),
                camera_manager=FakeCameraManager(),
                beaver_reader=FakeBeaverReader(),
            )
            collector._prepare_current_episode()
            controller = RecollectController(collector)
            with self.assertRaises(HTTPException) as raised:
                controller.command("teach_collect")
            self.assertEqual(raised.exception.status_code, 409)
            self.assertTrue(controller.command("teach")["accepted"])
            collector.close()


if __name__ == "__main__":
    unittest.main()
