from __future__ import annotations

from collections import deque
import unittest

import numpy as np

from config import Config
from realman_teachcollect import RealManTeachCollector, tcp_pose_to_vector


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

    def get_joint_configuration(self):
        return np.arange(self.dof, dtype=float)

    def get_tcp_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return pose

    def get_tcp_force(self):
        return np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    def start_freedrive(self):
        self.freedrive_started = True
        self.freedrive_events.append("start")

    def stop_freedrive(self):
        self.freedrive_stopped = True

    def set_freedrive_sensitivity(self, grade):
        self.freedrive_sensitivity = grade
        self.freedrive_events.append("sensitivity")

    def cleanup(self):
        self.cleaned = True


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

    def drain_commands(self):
        commands = list(self.commands)
        self.commands.clear()
        return commands

    def publish(self, sample):
        self.samples.append(sample)

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
    )


class RealManTeachCollectorTests(unittest.TestCase):
    def test_tcp_pose_vector_uses_quaternion_then_translation(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        vector, quaternion = tcp_pose_to_vector(pose)
        np.testing.assert_allclose(vector, [0, 0, 0, 1, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(quaternion, [0, 0, 0, 1])

    def test_visualizer_commands_record_save_and_delete(self) -> None:
        backend = FakeBackend()
        dataset = FakeDataset()
        visualizer = FakeVisualizer()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=dataset,
            visualizer_handle=visualizer,
        )

        visualizer.commands.append({"command": "toggle_recording"})
        collector.handle_visualizer_commands()
        self.assertTrue(collector.collecting)

        sample = collector.read_sample()
        collector.record_sample(sample)
        collector.publish_sample(sample)
        self.assertEqual(dataset.collect_step, 1)
        frame = dataset.frames[0]
        np.testing.assert_allclose(frame["state"], np.arange(7))
        np.testing.assert_allclose(frame["action"], np.arange(7))
        np.testing.assert_allclose(frame["wrench_data"], [1, 2, 3, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            frame["extra_data"]["tcp_pose"],
            [0, 0, 0, 1, 0.1, 0.2, 0.3],
        )
        self.assertTrue(visualizer.samples[-1]["dataset"]["collecting"])

        visualizer.commands.append({"command": "toggle_recording"})
        collector.handle_visualizer_commands()
        self.assertFalse(collector.collecting)
        self.assertEqual(dataset.exports, 1)
        self.assertEqual(dataset.recorded_episodes, 1)

        visualizer.commands.append({"command": "rollback_last_episode"})
        collector.handle_visualizer_commands()
        self.assertEqual(dataset.rollbacks, 1)
        self.assertEqual(dataset.recorded_episodes, 0)

        collector.close()
        self.assertTrue(dataset.closed)
        self.assertTrue(visualizer.closed)
        self.assertTrue(backend.cleaned)

    def test_close_stops_freedrive_and_saves_active_episode(self) -> None:
        backend = FakeBackend()
        dataset = FakeDataset()
        collector = RealManTeachCollector(
            teaching_config(),
            backend=backend,
            dataset=dataset,
        )
        collector.enable_freedrive()
        collector.start_recording()
        collector.record_sample(collector.read_sample())

        collector.close()

        self.assertTrue(backend.freedrive_stopped)
        self.assertEqual(dataset.exports, 1)
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

    def test_delete_cancels_empty_recording_without_deleting_saved_data(self) -> None:
        dataset = FakeDataset()
        dataset.recorded_episodes = 2
        collector = RealManTeachCollector(
            teaching_config(),
            backend=FakeBackend(),
            dataset=dataset,
        )
        collector.start_recording()

        self.assertTrue(collector.delete_last_episode())

        self.assertFalse(collector.collecting)
        self.assertEqual(dataset.recorded_episodes, 2)
        self.assertEqual(dataset.rollbacks, 0)
        collector.close()


if __name__ == "__main__":
    unittest.main()
