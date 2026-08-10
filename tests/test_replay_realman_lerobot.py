from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dataset_tool.replay_realman_lerobot import (
    EpisodeTrajectory,
    RealManLeRobotDataset,
    RealManTrajectoryReplayer,
    confirm,
    tcp_vector_to_pose,
    validate_tcp_trajectory_speed,
    validate_trajectory_speed,
)


def write_dataset(root: Path, joints: np.ndarray, *, dof: int = 7) -> None:
    tcp = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3], dtype=np.float32),
        (len(joints), 1),
    )
    if len(tcp) > 1:
        tcp[:, 4] += np.arange(len(tcp), dtype=np.float32) * 0.01
    info = {
        "codebase_version": "v3.0",
        "robot_type": "realman",
        "total_episodes": 1,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [dof],
                "names": [f"joint_{idx}" for idx in range(dof)],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [dof],
                "names": [f"joint_{idx}" for idx in range(dof)],
            },
            "extra.tcp_pose": {
                "dtype": "float32",
                "shape": [7],
                "names": ["qx", "qy", "qz", "qw", "x", "y", "z"],
            },
        },
    }
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info))
    pd.DataFrame(
        {
            "action": list(joints),
            "observation.state": list(joints),
            "extra.tcp_pose": list(tcp),
            "frame_index": np.arange(len(joints)),
            "episode_index": np.zeros(len(joints), dtype=int),
        }
    ).to_parquet(data / "file-000.parquet")


class FakeStatus:
    name = "SUCCEEDED"


class FakeAction:
    def __init__(self) -> None:
        self.wait_args = None

    def wait(self, **kwargs):
        self.wait_args = kwargs
        return FakeStatus()


class FakeRealMan:
    dof = 7

    def __init__(self) -> None:
        self.initial_moves = []
        self.servo_targets = []
        self.initial_tcp_moves = []
        self.servo_tcp_targets = []
        self.action = FakeAction()
        self.robot = FakeArm()

    def move_to_joint_configuration(self, joints, joint_speed):
        self.initial_moves.append((np.asarray(joints), joint_speed))
        return self.action

    def servo_to_joint_configuration(self, joints, duration):
        self.servo_targets.append((np.asarray(joints), duration))

    def move_to_tcp_pose(self, pose, joint_speed):
        self.initial_tcp_moves.append((np.asarray(pose), joint_speed))
        return self.action

    def servo_to_tcp_pose(self, pose, duration):
        self.servo_tcp_targets.append((np.asarray(pose), duration))


class FakeArm:
    def __init__(self) -> None:
        self.slow_stops = 0

    def rm_set_arm_slow_stop(self):
        self.slow_stops += 1
        return 0


class ReplayRealManLeRobotTests(unittest.TestCase):
    @patch("builtins.input", return_value="")
    def test_confirmation_accepts_enter(self, mocked_input) -> None:
        confirm("Ready.", assume_yes=False)
        mocked_input.assert_called_once()

    @patch("builtins.input", return_value="c")
    def test_confirmation_accepts_c_to_quit(self, _mocked_input) -> None:
        with self.assertRaises(KeyboardInterrupt):
            confirm("Ready.", assume_yes=False)

    def test_loads_seven_joint_teaching_episode(self) -> None:
        joints = np.vstack([np.arange(7), np.arange(7) + 0.1]).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_dataset(root, joints)
            dataset = RealManLeRobotDataset(root)
            trajectory = dataset.load_episode(0)

        np.testing.assert_allclose(trajectory.joints, joints)
        self.assertEqual(trajectory.fps, 10)
        self.assertAlmostEqual(trajectory.maximum_joint_speed, 1.0, places=5)

    def test_rejects_non_realman_dof_schema(self) -> None:
        joints = np.zeros((2, 6), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_dataset(root, joints, dof=6)
            with self.assertRaisesRegex(ValueError, "seven-joint"):
                RealManLeRobotDataset(root)

    def test_rejects_trajectory_above_speed_limit(self) -> None:
        trajectory = EpisodeTrajectory(
            episode_index=4,
            targets=np.vstack([np.zeros(7), np.ones(7)]),
            fps=10,
        )
        with self.assertRaisesRegex(ValueError, "above the replay safety limit"):
            validate_trajectory_speed(trajectory, 2.5)

    def test_loads_and_validates_tcp_trajectory(self) -> None:
        joints = np.zeros((2, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_dataset(root, joints)
            dataset = RealManLeRobotDataset(root, control_mode="tcp")
            trajectory = dataset.load_episode(0)

        self.assertEqual(trajectory.control_mode, "tcp")
        self.assertAlmostEqual(trajectory.maximum_linear_speed, 0.1, places=5)
        self.assertAlmostEqual(trajectory.maximum_angular_speed, 0.0, places=5)
        validate_tcp_trajectory_speed(trajectory, 0.75, 2.0)

    def test_tcp_vector_to_pose_uses_quaternion_and_translation(self) -> None:
        pose = tcp_vector_to_pose(np.array([0, 0, 0, 1, 0.1, 0.2, 0.3]))
        np.testing.assert_allclose(pose[:3, :3], np.eye(3))
        np.testing.assert_allclose(pose[:3, 3], [0.1, 0.2, 0.3])

    @patch("dataset_tool.replay_realman_lerobot.time.sleep", return_value=None)
    @patch("dataset_tool.replay_realman_lerobot.time.perf_counter")
    def test_moves_to_start_then_replays_all_joint_targets(
        self,
        perf_counter,
        _sleep,
    ) -> None:
        perf_counter.side_effect = [0.0, 0.01, 0.11]
        joints = np.vstack([np.zeros(7), np.full(7, 0.1)])
        trajectory = EpisodeTrajectory(0, joints, 10)
        robot = FakeRealMan()
        replayer = RealManTrajectoryReplayer(robot, initial_speed=0.3)

        replayer.move_to_start(joints[0])
        replayer.replay(trajectory)

        np.testing.assert_allclose(robot.initial_moves[0][0], joints[0])
        self.assertEqual(robot.initial_moves[0][1], 0.3)
        self.assertEqual(len(robot.servo_targets), 2)
        np.testing.assert_allclose(robot.servo_targets[-1][0], joints[-1])
        self.assertEqual(robot.servo_targets[-1][1], 0.1)

        replayer.stop_motion()
        self.assertEqual(robot.robot.slow_stops, 1)

    @patch("dataset_tool.replay_realman_lerobot.time.sleep", return_value=None)
    @patch("dataset_tool.replay_realman_lerobot.time.perf_counter")
    def test_moves_and_replays_tcp_targets(self, perf_counter, _sleep) -> None:
        perf_counter.side_effect = [0.0, 0.01, 0.11]
        tcp_targets = np.array(
            [
                [0, 0, 0, 1, 0.1, 0.2, 0.3],
                [0, 0, 0, 1, 0.2, 0.2, 0.3],
            ],
            dtype=float,
        )
        trajectory = EpisodeTrajectory(0, tcp_targets, 10, "tcp")
        robot = FakeRealMan()
        tcp_transform = np.eye(4)
        tcp_transform[0, 3] = 0.05
        replayer = RealManTrajectoryReplayer(
            robot,
            control_mode="tcp",
            tcp_transform=tcp_transform,
        )

        replayer.move_to_start(tcp_targets[0])
        replayer.replay(trajectory)

        self.assertEqual(len(robot.initial_tcp_moves), 1)
        np.testing.assert_allclose(robot.initial_tcp_moves[0][0][:3, 3], [0.05, 0.2, 0.3])
        self.assertEqual(len(robot.servo_tcp_targets), 2)
        np.testing.assert_allclose(
            robot.servo_tcp_targets[-1][0][:3, 3],
            [0.15, 0.2, 0.3],
        )


if __name__ == "__main__":
    unittest.main()
