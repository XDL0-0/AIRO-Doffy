"""Serializer tests for detached recording episodes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from airo_doffy.recording import (
    Episode,
    FrozenArray,
    HDF5EpisodeWriter,
    HDF5Rollback,
    LeRobotEpisodeWriter,
    NamedArray,
    RecordingSample,
    RecordingSchemaMismatchError,
    build_recording_schema,
)


def _frozen(values: np.ndarray, dtype: str) -> FrozenArray:
    array = np.ascontiguousarray(values, dtype=dtype)
    return FrozenArray(
        data=array.tobytes(),
        shape=array.shape,
        dtype=dtype,
    )


def _episode(index: int = 2) -> Episode:
    schema = build_recording_schema(
        data_type="both",
        robot_dof=6,
        camera_count=1,
        resolution=(2, 1),
        tactile_shape=(4, 3),
        force_enabled=True,
        torque_enabled=True,
        depth_enabled=True,
    )
    sample = RecordingSample(
        state=tuple(float(value) for value in range(7)),
        action=tuple(float(value + 10) for value in range(7)),
        timestamps_ns=(1, 2, 3, 4, 5, 6),
        tcp_pose=(0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3),
        force=(1.0, 2.0, 3.0),
        torque=(4.0, 5.0, 6.0),
        tactile=_frozen(np.arange(12).reshape(4, 3), "float32"),
        images=(
            NamedArray(
                name="camera_0",
                value=_frozen(np.arange(6).reshape(1, 2, 3), "uint8"),
            ),
        ),
        depths=(
            NamedArray(
                name="camera_0",
                value=_frozen(np.array([[0.1, 70.0]]), "float32"),
            ),
        ),
    )
    return Episode(index=index, task="pick", schema=schema, samples=(sample,))


@unittest.skipIf(h5py is None, "h5py recording extra is not installed")
class HDF5EpisodeWriterTest(unittest.TestCase):
    def test_writer_preserves_legacy_paths_shapes_dtypes_and_description(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = HDF5EpisodeWriter(directory)
            path = writer.write(_episode())
            self.assertEqual(path.name, "episode_2.hdf5")
            with h5py.File(path, "r") as root:
                self.assertFalse(bool(root.attrs["sim"]))
                self.assertEqual(root["/observations/qpos"].shape, (1, 7))
                self.assertEqual(root["/observations/qpos"].dtype, np.dtype("float64"))
                self.assertEqual(root["/action"].dtype, np.dtype("float64"))
                self.assertEqual(root["/extra/timestamps_ns"].dtype, np.dtype("int64"))
                self.assertEqual(root["/extra/tcp_pose"].shape, (1, 7))
                self.assertEqual(root["/observations/force"].shape, (1, 3))
                self.assertEqual(root["/observations/torque"].shape, (1, 3))
                self.assertEqual(root["/observations/tactile"].shape, (1, 4, 3))
                self.assertEqual(
                    root["/observations/images/camera_0"].shape,
                    (1, 1, 2, 3),
                )
                self.assertEqual(
                    root["/observations/depth/camera_0"].shape,
                    (1, 1, 2),
                )
                names = [
                    value.decode()
                    for value in root["/extra/timestamps_ns"].attrs["names"]
                ]
                self.assertEqual(names[-1], "camera_0")
            description = Path(directory, "episode_descriptions.txt").read_text()
            self.assertEqual(description, "Episode 2: max_timesteps = 1\n")

    def test_numbering_uses_max_index_and_rollback_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "episode_0.hdf5").touch()
            Path(directory, "episode_4.hdf5").touch()
            Path(directory, "episode_bad.hdf5").touch()
            writer = HDF5EpisodeWriter(directory)
            self.assertEqual(writer.next_episode_index, 5)
            Path(directory, "episode_descriptions.txt").write_text(
                "Episode 0: max_timesteps = 1\n"
                "Episode 4: max_timesteps = 3\n",
                encoding="utf-8",
            )
            rollback = HDF5Rollback(directory)
            self.assertTrue(rollback.rollback(4))
            self.assertEqual(writer.next_episode_index, 1)
            self.assertEqual(
                Path(directory, "episode_descriptions.txt").read_text(
                    encoding="utf-8"
                ),
                "Episode 0: max_timesteps = 1\n",
            )
            self.assertFalse(rollback.rollback(4))

    def test_writer_refuses_to_overwrite_existing_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "episode_2.hdf5").touch()
            with self.assertRaises(FileExistsError):
                HDF5EpisodeWriter(directory).write(_episode())


class FakeLeRobotDataset:
    def __init__(self, episode: Episode) -> None:
        self.features = episode.schema.lerobot_features()
        self.root = "mock-dataset"
        self.frames: list[dict[str, object]] = []
        self.created_index: int | None = None
        self.saved = False
        self.finalized = False
        self.cleared = False

    def create_episode_buffer(self, *, episode_index: int):
        self.created_index = episode_index
        return []

    def add_frame(self, frame: dict[str, object]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved = True

    def clear_episode_buffer(self) -> None:
        self.cleared = True

    def finalize(self) -> None:
        self.finalized = True


class LeRobotEpisodeWriterTest(unittest.TestCase):
    def test_writer_preserves_frame_dtypes_and_depth_encoding(self) -> None:
        episode = _episode()
        dataset = FakeLeRobotDataset(episode)
        writer = LeRobotEpisodeWriter(
            lambda: dataset,
            expected_schema=episode.schema,
        )
        self.assertEqual(writer.write(episode), Path("mock-dataset"))
        self.assertEqual(dataset.created_index, 2)
        self.assertTrue(dataset.saved)
        self.assertTrue(dataset.finalized)
        frame = dataset.frames[0]
        self.assertEqual(frame["observation.state"].dtype, np.dtype("float32"))
        self.assertEqual(frame["action"].dtype, np.dtype("float32"))
        self.assertEqual(frame["extra.timestamps_ns"].dtype, np.dtype("int64"))
        depth = frame["observation.depth.camera_0"]
        self.assertEqual(depth.dtype, np.dtype("uint16"))
        self.assertEqual(depth.shape, (1, 2, 1))
        self.assertEqual(depth[:, :, 0].tolist(), [[100, 65535]])

    def test_schema_mismatch_does_not_write_or_delete(self) -> None:
        episode = _episode()
        dataset = FakeLeRobotDataset(episode)
        dataset.features.pop("extra.tcp_pose")
        writer = LeRobotEpisodeWriter(
            lambda: dataset,
            expected_schema=episode.schema,
        )
        with self.assertRaises(RecordingSchemaMismatchError):
            writer.write(episode)
        self.assertEqual(dataset.frames, [])
        self.assertFalse(dataset.finalized)


if __name__ == "__main__":
    unittest.main()
