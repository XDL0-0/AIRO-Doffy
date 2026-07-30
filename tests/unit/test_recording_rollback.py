"""Storage-only rollback tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from airo_doffy.core.errors import ModelValidationError
from airo_doffy.recording import HDF5Rollback, LeRobotRollback


class HDF5RollbackTest(unittest.TestCase):
    def test_deletes_exact_episode_and_description_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode_0.hdf5").touch()
            (root / "episode_1.hdf5").touch()
            description = root / "episode_descriptions.txt"
            description.write_text(
                "Episode 0: max_timesteps = 2\n"
                "Episode 1: max_timesteps = 3\n",
                encoding="utf-8",
            )
            rollback = HDF5Rollback(root)
            self.assertTrue(rollback.rollback(1))
            self.assertTrue((root / "episode_0.hdf5").exists())
            self.assertFalse((root / "episode_1.hdf5").exists())
            self.assertEqual(
                description.read_text(encoding="utf-8"),
                "Episode 0: max_timesteps = 2\n",
            )


class LeRobotRollbackTest(unittest.TestCase):
    def _write_info(self, root: Path) -> Path:
        path = root / "meta" / "info.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "total_episodes": 2,
                    "total_frames": 8,
                    "total_tasks": 1,
                    "chunks_size": 1000,
                    "data_path": "data/chunk-{chunk_index}/file-{file_index}.parquet",
                    "video_path": (
                        "videos/{video_key}/chunk-{chunk_index}/"
                        "file-{file_index}.mp4"
                    ),
                    "features": {
                        "observation.images.camera_0": {"dtype": "video"}
                    },
                    "splits": {"train": "0:2"},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_fallback_rollback_updates_info_and_removes_conventional_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info_path = self._write_info(root)
            video = (
                root
                / "videos"
                / "observation.images.camera_0"
                / "chunk-0"
                / "file-1.mp4"
            )
            video.parent.mkdir(parents=True)
            video.touch()
            stats = root / "meta" / "stats.json"
            stats.write_text("{}", encoding="utf-8")

            rollback = LeRobotRollback(root)
            self.assertTrue(rollback.rollback(1))
            self.assertFalse(video.exists())
            self.assertFalse(stats.exists())
            info = json.loads(info_path.read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 8)
            self.assertEqual(info["splits"], {"train": "0:1"})

    def test_only_latest_episode_can_be_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_info(root)
            self.assertFalse(LeRobotRollback(root).rollback(0))

    def test_metadata_paths_cannot_escape_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info_path = self._write_info(root)
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["video_path"] = "../outside-{file_index}.mp4"
            info_path.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaises(ModelValidationError):
                LeRobotRollback(root).rollback(1)


if __name__ == "__main__":
    unittest.main()
