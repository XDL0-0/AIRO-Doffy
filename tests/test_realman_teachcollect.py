from __future__ import annotations

from collections import deque
import threading
import unittest
from unittest.mock import patch

import numpy as np

from config import Config
from realman_teachcollect import (
    RealManTeachCollector,
    TeachState,
    build_config,
    parse_args,
    tcp_pose_to_vector,
)


class FakeBackend:
    name = "realman"
    supports_freedrive = True
    supports_force = True
    dataset_robot_type = "realman"
    dof = 7

    def __init__(self) -> None:
        self.freedrive_started = False
        self.freedrive_stopped = False
        self.cleaned = False
        self.freedrive_sensitivity = None
        self.freedrive_events = []
        self.joints = np.arange(self.dof, dtype=float)
        self.commanded_joints = []
        self.reset_targets = []
        self.reset_observer = None
        self.wrench = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        self.drop_force_on_command = False

    def get_joint_configuration(self):
        return self.joints.copy()

    def get_tcp_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return pose

    def get_tcp_force(self):
        return None if self.wrench is None else self.wrench.copy()

    def start_freedrive(self):
        self.freedrive_started = True
        self.freedrive_events.append("start")

    def stop_freedrive(self):
        self.freedrive_stopped = True

    def set_freedrive_sensitivity(self, grade):
        self.freedrive_sensitivity = grade
        self.freedrive_events.append("sensitivity")

    def command_joint_configuration(self, joints, dt):
        self.joints = np.asarray(joints, dtype=float).copy()
        self.commanded_joints.append((self.joints.copy(), dt))
        if self.drop_force_on_command:
            self.wrench = None

    def initial_joint_configuration(self, configured):
        return np.asarray(configured, dtype=float).copy()

    def reset(self, joints):
        self.joints = np.asarray(joints, dtype=float).copy()
        self.reset_targets.append(self.joints.copy())
        if self.reset_observer is not None:
            self.reset_observer()

    def cleanup(self):
        self.cleaned = True


class LaggingFakeBackend(FakeBackend):
    """Return prescribed measurements without snapping state to each command."""

    def __init__(self, measured_joints) -> None:
        super().__init__()
        self._measured_joints = deque(
            np.asarray(joints, dtype=float).copy() for joints in measured_joints
        )

    def get_joint_configuration(self):
        if self._measured_joints:
            self.joints = self._measured_joints.popleft()
        return self.joints.copy()

    def command_joint_configuration(self, joints, dt):
        self.commanded_joints.append((np.asarray(joints, dtype=float).copy(), dt))


class FakeDataset:
    def __init__(self) -> None:
        self.recorded_episodes = 0
        self.collect_step = 0
        self.frames = []
        self.exports = 0
        self.rollbacks = 0
        self.closed = False

    def data_collection(self, **kwargs):
        self.frames.append(kwargs)
        self.collect_step += 1

    def data_export(self, manager):
        assert manager is None
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
            "dataset_type": "l",
            "recorded_episodes": self.recorded_episodes,
            "current_episode_frames": self.collect_step,
            "collecting": collecting,
        }

    def close(self):
        self.closed = True


class FakeVisualizer:
    def __init__(self) -> None:
        self.commands = deque()
        self.samples = []
        self.closed = False
        self.process = FakeProcess()

    def drain_commands(self):
        commands = list(self.commands)
        self.commands.clear()
        return commands

    def publish(self, sample):
        self.samples.append(sample)

    def close(self):
        self.closed = True


class FakeProcess:
    def is_alive(self):
        return False


class FakeCameraManager:
    def __init__(self, camera_num=2, depth_mode=True) -> None:
        self.camera_num = camera_num
        self.depth_mode = depth_mode
        self._lock = threading.Lock()
        self.camera_images = {
            f"camera_{i}": np.full((480, 640, 3), i, dtype=np.uint8)
            for i in range(camera_num)
        }
        self.camera_image_timestamps_ns = {
            f"camera_{i}": 100 + i for i in range(camera_num)
        }
        self.depth_images = {
            f"camera_{i}": np.full((480, 640), i + 1, dtype=np.float32)
            for i in range(camera_num)
        }
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


def teaching_config() -> Config:
    return Config(
        ROBOT_TYPE="realman",
        GRIPPER=False,
        DATASET_TYPE="l",
        DATA_TYPE="both",
        FORCE_COLLECT=True,
        TORQUE_COLLECT=True,
        DEPTH_INFO_ENABLE=False,
        TACTILE_TRANSFER=False,
        FORCE_MOVING_AVERAGE_WINDOW=1,
        FORCE_LOW_PASS_ALPHA=0.0,
        TEACH_INITIAL_DISCARD_FRAMES=0,
    )


class RealManTeachCollectorTests(unittest.TestCase):
    def test_teach_action_mode_validation_and_default(self) -> None:
        self.assertEqual(Config().TEACH_ACTION_MODE, "next_joint")
        self.assertEqual(
            Config(TEACH_ACTION_MODE=" COMMAND ").TEACH_ACTION_MODE,
            "command",
        )
        with self.assertRaisesRegex(ValueError, "TEACH_ACTION_MODE"):
            Config(TEACH_ACTION_MODE="target_or_state")

    def test_initial_teach_discard_frames_must_be_nonnegative_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "TEACH_INITIAL_DISCARD_FRAMES"):
            Config(TEACH_INITIAL_DISCARD_FRAMES=-1)
        with self.assertRaisesRegex(ValueError, "TEACH_INITIAL_DISCARD_FRAMES"):
            Config(TEACH_INITIAL_DISCARD_FRAMES=1.5)

    def test_dataset_dir_defaults_to_shared_config(self) -> None:
        with patch("sys.argv", ["realman_teachcollect.py"]):
            args = parse_args()

        self.assertEqual(args.dataset_dir, Config().DATASET_DIR)
        self.assertEqual(build_config(args).DATASET_DIR, Config().DATASET_DIR)

    def test_dataset_dir_cli_override_is_preserved(self) -> None:
        override = "./datasets/teach_override"
        with patch(
            "sys.argv",
            ["realman_teachcollect.py", "--dataset-dir", override],
        ):
            args = parse_args()

        self.assertEqual(build_config(args).DATASET_DIR, override)

    def test_tcp_pose_vector_uses_quaternion_then_translation(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        vector, quaternion = tcp_pose_to_vector(pose)
        np.testing.assert_allclose(vector, [0, 0, 0, 1, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(quaternion, [0, 0, 0, 1])

    def test_transient_force_dropout_uses_cache_and_disables_replay(self) -> None:
        backend = FakeBackend()
        visualizer = FakeVisualizer()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=FakeDataset(),
            visualizer_handle=visualizer,
        )
        fresh = collector.read_sample()
        self.assertTrue(fresh.force_valid)

        backend.wrench = None
        stale = collector.read_sample()
        self.assertFalse(stale.force_valid)
        np.testing.assert_allclose(stale.wrench, fresh.wrench)
        collector.publish_sample(stale)
        self.assertFalse(visualizer.samples[-1]["connected"])
        self.assertIn("Force data temporarily unavailable", visualizer.samples[-1]["error"])

        collector.teach_state = TeachState.READY
        collector.taught_trajectory = [backend.joints.copy()]
        self.assertFalse(collector.workflow_status()["replay_enabled"])
        self.assertFalse(collector.replay_collect())

        backend.wrench = np.ones(6)
        recovered = collector.read_sample()
        self.assertTrue(recovered.force_valid)
        self.assertTrue(collector.workflow_status()["replay_enabled"])
        collector.close()

    def test_force_dropout_during_replay_rolls_back_without_stopping_collector(self) -> None:
        backend = FakeBackend()
        dataset = FakeDataset()
        visualizer = FakeVisualizer()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=dataset,
            visualizer_handle=visualizer,
        )
        reset_states = []
        backend.reset_observer = lambda: reset_states.append(
            (collector.collecting, dataset.exports, len(dataset.frames))
        )
        collector.start_teach()
        backend.joints += 0.1
        collector.capture_teach_sample(collector.read_sample())
        collector.end_teach()
        backend.drop_force_on_command = True

        visualizer.commands.append({"command": "replay_collect"})
        collector.handle_visualizer_commands()

        self.assertEqual(collector.teach_state, TeachState.READY)
        self.assertEqual(dataset.exports, 0)
        self.assertEqual(dataset.rollbacks, 1)
        self.assertFalse(collector.collecting)
        self.assertEqual(len(backend.reset_targets), 1)
        self.assertEqual(reset_states, [(False, 0, 0)])
        self.assertIn("Replay failed", collector.workflow_message)
        collector.close()

    @patch("realman_teachcollect.time.sleep", return_value=None)
    def test_visualizer_commands_teach_then_replay_collect(self, _sleep) -> None:
        backend = FakeBackend()
        dataset = FakeDataset()
        visualizer = FakeVisualizer()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=dataset,
            visualizer_handle=visualizer,
        )
        reset_states = []
        backend.reset_observer = lambda: reset_states.append(
            (collector.collecting, dataset.exports, len(dataset.frames))
        )

        visualizer.commands.append({"command": "toggle_teach"})
        collector.handle_visualizer_commands()
        self.assertEqual(collector.teach_state, TeachState.TEACHING)
        self.assertTrue(backend.freedrive_started)

        first = collector.read_sample()
        collector.capture_teach_sample(first)
        backend.joints = np.arange(7, dtype=float) + 0.1
        second = collector.read_sample()
        collector.capture_teach_sample(second)
        self.assertEqual(dataset.collect_step, 0)

        visualizer.commands.append({"command": "toggle_teach"})
        collector.handle_visualizer_commands()
        self.assertEqual(collector.teach_state, TeachState.READY)
        self.assertTrue(backend.freedrive_stopped)
        self.assertTrue(collector.workflow_status()["replay_enabled"])

        visualizer.commands.append({"command": "replay_collect"})
        collector.handle_visualizer_commands()
        self.assertEqual(collector.teach_state, TeachState.READY)
        self.assertFalse(collector.collecting)
        self.assertEqual(dataset.exports, 1)
        self.assertEqual(dataset.recorded_episodes, 1)
        self.assertEqual(len(dataset.frames), 2)
        np.testing.assert_allclose(
            dataset.frames[0]["action"], np.arange(7) + 0.1
        )
        np.testing.assert_allclose(
            dataset.frames[1]["action"], np.arange(7) + 0.1
        )
        np.testing.assert_allclose(
            dataset.frames[1]["state"], np.arange(7) + 0.1
        )
        self.assertEqual(len(backend.commanded_joints), 2)
        self.assertEqual(len(backend.reset_targets), 2)
        np.testing.assert_allclose(
            backend.reset_targets[-1], collector.cfg.INITIAL_JOINT
        )
        self.assertEqual(reset_states[-1], (False, 1, 2))
        self.assertIn("robot returned", collector.workflow_message)

        sample = collector.read_sample()
        collector.publish_sample(sample)
        frame = dataset.frames[0]
        np.testing.assert_allclose(
            frame["wrench_data"], [1, 2, 3, 0.1, 0.2, 0.3]
        )
        np.testing.assert_allclose(
            frame["extra_data"]["tcp_pose"],
            [0, 0, 0, 1, 0.1, 0.2, 0.3],
        )
        self.assertEqual(visualizer.samples[-1]["teach"]["state"], "ready")

        visualizer.commands.append({"command": "rollback_last_episode"})
        collector.handle_visualizer_commands()
        self.assertEqual(dataset.rollbacks, 1)
        self.assertEqual(dataset.recorded_episodes, 0)

        collector.close()
        self.assertTrue(dataset.closed)
        self.assertTrue(visualizer.closed)
        self.assertTrue(backend.cleaned)

    @patch("realman_teachcollect.time.sleep", return_value=None)
    def test_next_joint_mode_uses_next_measured_state_and_its_timestamp(
        self, _sleep
    ) -> None:
        measured = [
            np.full(7, 10.0),
            np.full(7, 20.0),
            np.full(7, 30.0),
        ]
        backend = LaggingFakeBackend(measured)
        dataset = FakeDataset()
        cfg = teaching_config()
        cfg.TEACH_ACTION_MODE = "next_joint"
        collector = RealManTeachCollector(cfg, backend=backend, dataset=dataset)
        collector.teach_state = TeachState.READY
        collector.taught_trajectory = [
            np.full(7, 1.0),
            np.full(7, 2.0),
            np.full(7, 3.0),
        ]
        collector._force_data_available = True

        self.assertTrue(collector.replay_collect())

        np.testing.assert_allclose(
            np.stack([frame["state"] for frame in dataset.frames]), measured
        )
        np.testing.assert_allclose(
            np.stack([frame["action"] for frame in dataset.frames]),
            [measured[1], measured[2], measured[2]],
        )
        state_timestamps = [
            int(frame["extra_data"]["robot_state_timestamp_ns"])
            for frame in dataset.frames
        ]
        action_timestamps = [
            int(frame["extra_data"]["robot_action_timestamp_ns"])
            for frame in dataset.frames
        ]
        self.assertEqual(
            action_timestamps,
            [state_timestamps[1], state_timestamps[2], state_timestamps[2]],
        )
        collector.close()

    @patch("realman_teachcollect.time.sleep", return_value=None)
    def test_command_mode_uses_current_target_and_command_timestamp(self, _sleep) -> None:
        measured = [np.full(7, 10.0), np.full(7, 20.0)]
        targets = [np.full(7, 1.0), np.full(7, 2.0)]
        backend = LaggingFakeBackend(measured)
        dataset = FakeDataset()
        cfg = teaching_config()
        cfg.TEACH_ACTION_MODE = "command"
        collector = RealManTeachCollector(cfg, backend=backend, dataset=dataset)
        collector.teach_state = TeachState.READY
        collector.taught_trajectory = targets
        collector._force_data_available = True

        self.assertTrue(collector.replay_collect())

        np.testing.assert_allclose(
            np.stack([frame["action"] for frame in dataset.frames]), targets
        )
        for frame in dataset.frames:
            self.assertLessEqual(
                int(frame["extra_data"]["robot_action_timestamp_ns"]),
                int(frame["extra_data"]["robot_state_timestamp_ns"]),
            )
        collector.close()

    @patch("realman_teachcollect.time.sleep", return_value=None)
    def test_single_frame_next_joint_episode_repeats_its_own_state(self, _sleep) -> None:
        measured = np.full(7, 12.0)
        backend = LaggingFakeBackend([measured])
        dataset = FakeDataset()
        collector = RealManTeachCollector(
            teaching_config(), backend=backend, dataset=dataset
        )
        collector.teach_state = TeachState.READY
        collector.taught_trajectory = [np.full(7, 1.0)]
        collector._force_data_available = True

        self.assertTrue(collector.replay_collect())

        self.assertEqual(len(dataset.frames), 1)
        np.testing.assert_allclose(dataset.frames[0]["state"], measured)
        np.testing.assert_allclose(dataset.frames[0]["action"], measured)
        self.assertEqual(
            int(dataset.frames[0]["extra_data"]["robot_action_timestamp_ns"]),
            int(dataset.frames[0]["extra_data"]["robot_state_timestamp_ns"]),
        )
        collector.close()

    def test_reteach_only_clears_path_and_requires_teach_to_restart(self) -> None:
        backend = FakeBackend()
        collector = RealManTeachCollector(
            teaching_config(), backend=backend, dataset=FakeDataset()
        )
        collector.start_teach()
        backend.joints += 0.1
        collector.capture_teach_sample(collector.read_sample())
        collector.end_teach()

        self.assertTrue(collector.reteach())

        self.assertEqual(collector.teach_state, TeachState.IDLE)
        self.assertEqual(collector.taught_trajectory, [])
        self.assertFalse(collector.freedrive_active)
        self.assertEqual(backend.freedrive_events.count("start"), 1)
        self.assertEqual(
            collector.workflow_message,
            "Trajectory is cleared, please press Teach to create a new one.",
        )

        self.assertTrue(collector.start_teach())
        self.assertEqual(collector.teach_state, TeachState.TEACHING)
        self.assertEqual(backend.freedrive_events.count("start"), 2)
        collector.close()

    def test_teach_replaces_existing_ready_trajectory_immediately(self) -> None:
        backend = FakeBackend()
        dataset = FakeDataset()
        collector = RealManTeachCollector(
            teaching_config(), backend=backend, dataset=dataset
        )
        collector.start_teach()
        backend.joints += 0.1
        collector.capture_teach_sample(collector.read_sample())
        collector.end_teach()
        self.assertTrue(collector.taught_trajectory)
        self.assertTrue(collector.workflow_status()["teach_enabled"])

        self.assertTrue(collector.start_teach())

        self.assertEqual(collector.teach_state, TeachState.TEACHING)
        self.assertEqual(len(collector.taught_trajectory), 1)
        np.testing.assert_allclose(
            collector.taught_trajectory[0], backend.joints
        )
        self.assertTrue(collector.freedrive_active)
        self.assertEqual(backend.freedrive_events.count("start"), 2)
        self.assertEqual(dataset.recorded_episodes, 0)
        self.assertIn("Existing trajectory cleared", collector.workflow_message)
        collector.close()

    def test_teach_keeps_full_path_and_trims_only_stationary_edges(self) -> None:
        backend = FakeBackend()
        collector = RealManTeachCollector(
            teaching_config(), backend=backend, dataset=FakeDataset()
        )
        start = backend.joints.copy()
        collector.start_teach()

        for _ in range(2):
            self.assertTrue(collector.capture_teach_sample(collector.read_sample()))
        first_motion = start + np.deg2rad(0.2)
        backend.joints = first_motion
        self.assertTrue(collector.capture_teach_sample(collector.read_sample()))
        for _ in range(2):
            self.assertTrue(collector.capture_teach_sample(collector.read_sample()))
        internal_small_motion = first_motion + np.deg2rad(0.05)
        backend.joints = internal_small_motion
        self.assertTrue(collector.capture_teach_sample(collector.read_sample()))
        final = internal_small_motion + np.deg2rad(0.2)
        backend.joints = final
        self.assertTrue(collector.capture_teach_sample(collector.read_sample()))
        for _ in range(2):
            self.assertTrue(collector.capture_teach_sample(collector.read_sample()))
        self.assertEqual(len(collector.taught_trajectory), 10)

        self.assertTrue(collector.end_teach())

        self.assertEqual(len(collector.taught_trajectory), 6)
        np.testing.assert_allclose(collector.taught_trajectory[0], start)
        np.testing.assert_allclose(collector.taught_trajectory[-1], final)
        np.testing.assert_allclose(
            collector.taught_trajectory[4], internal_small_motion
        )
        np.testing.assert_allclose(
            collector.taught_trajectory[1], collector.taught_trajectory[2]
        )
        collector.close()

    def test_teach_rejects_fully_stationary_trajectory_after_postprocess(self) -> None:
        backend = FakeBackend()
        collector = RealManTeachCollector(
            teaching_config(), backend=backend, dataset=FakeDataset()
        )
        collector.start_teach()
        for _ in range(4):
            collector.capture_teach_sample(collector.read_sample())
        self.assertEqual(len(collector.taught_trajectory), 5)

        self.assertFalse(collector.end_teach())

        self.assertEqual(collector.teach_state, TeachState.IDLE)
        self.assertEqual(collector.taught_trajectory, [])
        self.assertIn("No motion was detected", collector.workflow_message)
        collector.close()

    def test_teach_discards_first_40_shaking_frames_before_edge_filter(self) -> None:
        cfg = teaching_config()
        cfg.TEACH_INITIAL_DISCARD_FRAMES = 40
        backend = FakeBackend()
        collector = RealManTeachCollector(cfg, backend=backend, dataset=FakeDataset())
        start = backend.joints.copy()
        collector.start_teach()

        for frame_index in range(1, 40):
            direction = -1.0 if frame_index % 2 else 1.0
            backend.joints = start + direction * np.deg2rad(0.5)
            collector.capture_teach_sample(collector.read_sample())
        self.assertEqual(len(collector.taught_trajectory), 40)

        path_start = start + np.deg2rad(1.0)
        backend.joints = path_start
        collector.capture_teach_sample(collector.read_sample())
        collector.capture_teach_sample(collector.read_sample())
        path_end = path_start + np.deg2rad(0.2)
        backend.joints = path_end
        collector.capture_teach_sample(collector.read_sample())
        collector.capture_teach_sample(collector.read_sample())

        self.assertTrue(collector.end_teach())

        self.assertEqual(len(collector.taught_trajectory), 2)
        np.testing.assert_allclose(collector.taught_trajectory[0], path_start)
        np.testing.assert_allclose(collector.taught_trajectory[-1], path_end)
        collector.close()

    def test_teach_rejects_path_shorter_than_initial_discard_window(self) -> None:
        cfg = teaching_config()
        cfg.TEACH_INITIAL_DISCARD_FRAMES = 40
        backend = FakeBackend()
        collector = RealManTeachCollector(cfg, backend=backend, dataset=FakeDataset())
        collector.start_teach()
        for frame_index in range(10):
            backend.joints += np.deg2rad(0.2 + frame_index * 0.01)
            collector.capture_teach_sample(collector.read_sample())

        self.assertFalse(collector.end_teach())

        self.assertEqual(collector.taught_trajectory, [])
        self.assertIn("discard window", collector.workflow_message)
        collector.close()

    def test_initial_pose_uses_configured_joints(self) -> None:
        cfg = teaching_config()
        cfg.INITIAL_JOINT = np.linspace(-0.3, 0.3, 7)
        backend = FakeBackend()
        collector = RealManTeachCollector(cfg, backend=backend, dataset=FakeDataset())

        self.assertTrue(collector.move_to_initial_pose())

        np.testing.assert_allclose(backend.reset_targets[-1], cfg.INITIAL_JOINT)
        self.assertEqual(collector.teach_state, TeachState.IDLE)
        collector.close()

    def test_camera_frames_are_recorded_and_published(self) -> None:
        backend = FakeBackend()
        dataset = FakeDataset()
        visualizer = FakeVisualizer()
        cameras = FakeCameraManager()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=dataset,
            camera_manager=cameras,
            visualizer_handle=visualizer,
        )

        sample = collector.read_sample()
        collector.record_sample(sample)
        collector.publish_sample(sample)

        frame = dataset.frames[-1]
        self.assertEqual(set(frame["camera_images"]), {"camera_0", "camera_1"})
        self.assertEqual(set(frame["depth_images"]), {"camera_0", "camera_1"})
        self.assertEqual(
            frame["extra_data"]["camera_timestamps_ns"],
            {"camera_0": 100, "camera_1": 101},
        )
        preview = visualizer.samples[-1]
        self.assertEqual(preview["camera_count"], 2)
        self.assertEqual(set(preview["images"]), {"camera_0", "camera_1"})
        self.assertEqual(preview["images"]["camera_0"].shape, (240, 320, 3))
        collector.close()
        self.assertTrue(cameras.closed)

    def test_detected_camera_count_initializes_dataset(self) -> None:
        dataset = FakeDataset()
        cameras = FakeCameraManager(camera_num=3, depth_mode=False)
        with patch(
            "realman_teachcollect.DatasetRecorder", return_value=dataset
        ) as recorder_type:
            collector = RealManTeachCollector(
                teaching_config(),
                backend=FakeBackend(),
                camera_manager=cameras,
            )

        self.assertEqual(recorder_type.call_args.kwargs["camera_num"], 3)
        collector.close()

    def test_run_starts_camera_but_keeps_freedrive_idle_until_teach(self) -> None:
        backend = FakeBackend()
        cameras = FakeCameraManager()
        visualizer = FakeVisualizer()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=FakeDataset(),
            camera_manager=cameras,
            visualizer_handle=visualizer,
        )

        collector.run()

        self.assertTrue(cameras.started)
        self.assertTrue(cameras.closed)
        self.assertFalse(backend.freedrive_started)
        self.assertFalse(backend.freedrive_stopped)
        self.assertTrue(visualizer.closed)

    def test_close_stops_freedrive_without_exporting_taught_path(self) -> None:
        backend = FakeBackend()
        dataset = FakeDataset()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=dataset,
        )
        collector.start_teach()
        collector.capture_teach_sample(collector.read_sample())

        collector.close()

        self.assertTrue(backend.freedrive_stopped)
        self.assertEqual(dataset.exports, 0)
        self.assertTrue(dataset.closed)
        self.assertTrue(backend.cleaned)

    def test_sets_sensitivity_before_starting_freedrive(self) -> None:
        backend = FakeBackend()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=FakeDataset(),
        )

        collector.enable_freedrive()

        self.assertEqual(backend.freedrive_sensitivity, 99)
        self.assertEqual(backend.freedrive_events, ["sensitivity", "start"])
        collector.close()

    def test_replay_is_disabled_until_teach_finishes(self) -> None:
        collector = RealManTeachCollector(
            teaching_config(),
            backend=FakeBackend(),
            dataset=FakeDataset(),
        )

        self.assertFalse(collector.replay_collect())
        self.assertFalse(collector.workflow_status()["replay_enabled"])
        collector.close()


if __name__ == "__main__":
    unittest.main()
