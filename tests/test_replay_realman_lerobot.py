from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dataset_tool.replay_realman_lerobot import (
    EpisodeTrajectory,
    RealManLeRobotDataset,
    RealManTrajectoryReplayer,
    build_parser,
    camera_name_from_video_key,
    confirm,
    read_first_video_frame,
    run_camera_alignment,
    _resize_to_match,
    tcp_vector_to_pose,
    validate_tcp_trajectory_speed,
    validate_trajectory_speed,
)

VIDEO_KEY = "observation.images.camera_0"


def write_dataset(
    root: Path,
    joints: np.ndarray,
    *,
    dof: int = 7,
    total_episodes: int = 1,
    episode_index: int = 0,
    data_file_index: int | None = None,
    video_file_index: int | None = None,
    video_from_timestamp: float = 0.0,
    write_episode_metadata: bool = False,
) -> None:
    tcp = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3], dtype=np.float32),
        (len(joints), 1),
    )
    if len(tcp) > 1:
        tcp[:, 4] += np.arange(len(tcp), dtype=np.float32) * 0.01
    info = {
        "codebase_version": "v3.0",
        "robot_type": "realman",
        "total_episodes": total_episodes,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
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
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
            },
        },
    }
    packed_data_file = 0 if data_file_index is None else data_file_index
    packed_video_file = packed_data_file if video_file_index is None else video_file_index
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info))
    pd.DataFrame(
        {
            "action": list(joints),
            "observation.state": list(joints),
            "extra.tcp_pose": list(tcp),
            "frame_index": np.arange(len(joints)),
            "episode_index": np.full(len(joints), episode_index, dtype=int),
        }
    ).to_parquet(data / f"file-{packed_data_file:03d}.parquet")
    if write_episode_metadata:
        episodes = root / "meta" / "episodes" / "chunk-000"
        episodes.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "episode_index": [episode_index],
                "length": [len(joints)],
                "data/chunk_index": [0],
                "data/file_index": [packed_data_file],
                f"videos/{VIDEO_KEY}/chunk_index": [0],
                f"videos/{VIDEO_KEY}/file_index": [packed_video_file],
                f"videos/{VIDEO_KEY}/from_timestamp": [video_from_timestamp],
            }
        ).to_parquet(episodes / f"file-{packed_data_file:03d}.parquet")


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
    def test_align_camera_flag_is_opt_in(self) -> None:
        parser = build_parser()
        self.assertFalse(parser.parse_args([]).align_camera)
        self.assertTrue(parser.parse_args(["--align-camera"]).align_camera)

    def test_resize_to_match_cover_crops_without_stretching(self) -> None:
        wide = np.zeros((48, 84, 3), dtype=np.uint8)
        wide[:, 10:20] = (0, 0, 255)
        target = np.zeros((48, 64, 3), dtype=np.uint8)
        matched = _resize_to_match(wide, target)
        self.assertEqual(matched.shape, (48, 64, 3))
        # 84→64 at the same height is a 10-pixel side crop, not a squeeze.
        self.assertEqual(int(np.sum(matched[0, :, 2] > 200)), 10)
        same = np.full((48, 64, 3), 7, dtype=np.uint8)
        self.assertIs(_resize_to_match(same, same), same)

    def test_maps_dataset_video_key_to_live_camera_name(self) -> None:
        self.assertEqual(
            camera_name_from_video_key("observation.images.camera_3", 0),
            "camera_3",
        )
        self.assertEqual(camera_name_from_video_key("observation.image", 2), "camera_2")

    @patch(
        "dataset_tool.replay_realman_lerobot._opencv_window_available",
        return_value=(False, "headless test"),
    )
    def test_alignment_skips_safely_without_display(self, _window_status) -> None:
        self.assertFalse(run_camera_alignment(object(), 0))

    @patch("dataset_tool.replay_realman_lerobot.read_first_video_frame")
    def test_loads_dataset_camera_first_frame(self, read_first_video_frame) -> None:
        joints = np.zeros((2, 7), dtype=np.float32)
        expected_frame = np.full((4, 6, 3), 127, dtype=np.uint8)
        read_first_video_frame.return_value = expected_frame
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_dataset(root, joints)
            video_path = (
                root
                / "videos"
                / "observation.images.camera_0"
                / "chunk-000"
                / "file-000.mp4"
            )
            video_path.parent.mkdir(parents=True)
            video_path.touch()
            dataset = RealManLeRobotDataset(root)

            frames = dataset.load_first_camera_frames(0)

        self.assertEqual(dataset.video_keys, [VIDEO_KEY])
        self.assertEqual(list(frames), [VIDEO_KEY])
        np.testing.assert_array_equal(frames[VIDEO_KEY], expected_frame)
        read_first_video_frame.assert_called_once_with(video_path, timestamp_s=0.0)

    @patch("dataset_tool.replay_realman_lerobot.read_first_video_frame")
    def test_resolves_packed_v3_video_file_and_timestamp(
        self,
        read_first_video_frame,
    ) -> None:
        joints = np.zeros((2, 7), dtype=np.float32)
        expected_frame = np.full((4, 6, 3), 90, dtype=np.uint8)
        read_first_video_frame.return_value = expected_frame
        from_timestamp = 614.041667
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_dataset(
                root,
                joints,
                total_episodes=76,
                episode_index=75,
                data_file_index=5,
                video_file_index=5,
                video_from_timestamp=from_timestamp,
                write_episode_metadata=True,
            )
            video_path = (
                root / "videos" / VIDEO_KEY / "chunk-000" / "file-005.mp4"
            )
            video_path.parent.mkdir(parents=True)
            video_path.touch()
            dataset = RealManLeRobotDataset(root)

            self.assertEqual(
                dataset.episode_video_path(75, VIDEO_KEY),
                video_path,
            )
            self.assertAlmostEqual(
                dataset.episode_video_from_timestamp(75, VIDEO_KEY),
                from_timestamp,
            )
            frames = dataset.load_first_camera_frames(75)

            self.assertEqual(list(frames), [VIDEO_KEY])
            np.testing.assert_array_equal(frames[VIDEO_KEY], expected_frame)
            self.assertEqual(read_first_video_frame.call_args.args, (video_path,))
            self.assertAlmostEqual(
                read_first_video_frame.call_args.kwargs["timestamp_s"],
                from_timestamp,
            )
            self.assertFalse(
                (root / "videos" / VIDEO_KEY / "chunk-000" / "file-075.mp4").exists()
            )

    def test_loads_packed_episode_from_shared_parquet(self) -> None:
        joints = np.vstack([np.arange(7), np.arange(7) + 0.2]).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_dataset(
                root,
                joints,
                total_episodes=76,
                episode_index=75,
                data_file_index=5,
                write_episode_metadata=True,
            )
            dataset = RealManLeRobotDataset(root)
            trajectory = dataset.load_episode(75)

        np.testing.assert_allclose(trajectory.joints, joints)
        self.assertEqual(trajectory.episode_index, 75)

    def test_seeks_into_packed_video_instead_of_file_start(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is required to encode a packed test video.")

        first = np.full((48, 64, 3), (0, 0, 255), dtype=np.uint8)
        second = np.full((48, 64, 3), (0, 255, 0), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "packed.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV could not open an mp4 writer.")
            for _ in range(10):
                writer.write(first)
            for _ in range(10):
                writer.write(second)
            writer.release()

            start_frame = read_first_video_frame(path, timestamp_s=0.0)
            episode_frame = read_first_video_frame(path, timestamp_s=1.0)

        self.assertIsNotNone(start_frame)
        self.assertIsNotNone(episode_frame)
        self.assertGreater(int(start_frame[..., 2].mean()), 200)
        self.assertGreater(int(episode_frame[..., 1].mean()), 200)

    def test_real_dataset_episode_75_uses_packed_camera_file(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "datasets"
            / "WRM_grasp_cylinder_different_sizes_lero"
        )
        if not (root / "meta" / "info.json").is_file():
            self.skipTest("WRM_grasp_cylinder_different_sizes_lero is not present.")
        dataset = RealManLeRobotDataset(root)
        self.assertEqual(dataset.episode_video_path(0, VIDEO_KEY).name, "file-000.mp4")
        self.assertAlmostEqual(dataset.episode_video_from_timestamp(0, VIDEO_KEY), 0.0)
        video_path = dataset.episode_video_path(75, VIDEO_KEY)
        self.assertEqual(video_path.name, "file-005.mp4")
        self.assertTrue(video_path.is_file(), video_path)
        self.assertAlmostEqual(
            dataset.episode_video_from_timestamp(75, VIDEO_KEY),
            614.041667,
            places=3,
        )
        frames = dataset.load_first_camera_frames(75)
        self.assertIn(VIDEO_KEY, frames)
        self.assertEqual(frames[VIDEO_KEY].shape, (480, 640, 3))

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
        np.testing.assert_allclose(
            robot.initial_tcp_moves[0][0][:3, 3], [0.05, 0.2, 0.3]
        )
        self.assertEqual(len(robot.servo_tcp_targets), 2)
        np.testing.assert_allclose(
            robot.servo_tcp_targets[-1][0][:3, 3],
            [0.15, 0.2, 0.3],
        )


if __name__ == "__main__":
    unittest.main()
