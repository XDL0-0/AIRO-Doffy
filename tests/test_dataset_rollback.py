from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from doffy_teleop.config import Config
from doffy_teleop.recording.dataset import DatasetRecorder


class LeRobotRollbackStatsTests(unittest.TestCase):
    def make_recorder(self, root: Path, camera_num: int = 0) -> DatasetRecorder:
        config = Config(
            DATASET_TYPE="l",
            DATA_TYPE="qpos",
            PUSH_TO_HUB=False,
            REALSENSE_RESOLUTION=(64, 48),
            COLLECT_RATE=30,
            beaver_enable=True,
        )
        return DatasetRecorder(
            camera_num=camera_num,
            robot_dof=7,
            robot_type="realman",
            force_collect=False,
            torque_collect=False,
            gripper=False,
            config=config,
            dataset_root=root,
        )

    @staticmethod
    def record_episode(
        recorder: DatasetRecorder,
        value: float,
        image_value: int | None = None,
    ) -> None:
        images = (
            {}
            if image_value is None
            else {"camera_0": np.full((48, 64, 3), image_value, dtype=np.uint8)}
        )
        beaver_data = {
            "distance_mm": np.full((9, 4, 4), value + 1, dtype=np.float32),
            "target_status": np.full((9, 4, 4), value + 2, dtype=np.float32),
            "present": np.ones(9, dtype=np.float32),
        }
        recorder.data_collection(
            np.full(7, value, dtype=np.float32),
            np.full(7, value, dtype=np.float32),
            images,
            beaver_data=beaver_data,
        )
        recorder.data_export(None)
        recorder._reset_data_dict()

    @staticmethod
    def read_info(root: Path) -> dict:
        return json.loads((root / "meta" / "info.json").read_text())

    @staticmethod
    def read_stats(root: Path) -> dict:
        from lerobot.datasets.utils import load_stats

        stats = load_stats(root)
        if stats is None:
            raise AssertionError("expected persisted LeRobot dataset statistics")
        return stats

    def assert_feature_stats(
        self,
        stats: dict,
        feature: str,
        *,
        count: int,
        mean: float,
        minimum: float,
        maximum: float,
        std: float,
        shape: tuple[int, ...],
    ) -> None:
        feature_stats = stats[feature]
        self.assertEqual(feature_stats["count"].tolist(), [count])
        np.testing.assert_allclose(feature_stats["mean"], np.full(shape, mean))
        np.testing.assert_allclose(feature_stats["min"], np.full(shape, minimum))
        np.testing.assert_allclose(feature_stats["max"], np.full(shape, maximum))
        np.testing.assert_allclose(feature_stats["std"], np.full(shape, std))

    def test_undo_then_continue_rebuilds_all_retained_feature_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stats_resume"
            recorder = self.make_recorder(root)
            try:
                self.record_episode(recorder, 0)
                self.record_episode(recorder, 10)
                self.assertTrue(recorder.rollback_last_episode())
                self.assert_feature_stats(
                    self.read_stats(root),
                    "observation.state",
                    count=1,
                    mean=0,
                    minimum=0,
                    maximum=0,
                    std=0,
                    shape=(7,),
                )

                self.record_episode(recorder, 20)
                recorder.close()
                stats = self.read_stats(root)
                info = self.read_info(root)
                self.assertEqual(info["total_episodes"], 2)
                self.assertEqual(info["total_frames"], 2)
                for feature in ("action", "observation.state"):
                    self.assert_feature_stats(
                        stats,
                        feature,
                        count=2,
                        mean=10,
                        minimum=0,
                        maximum=20,
                        std=10,
                        shape=(7,),
                    )
                self.assert_feature_stats(
                    stats,
                    "observation.beaver.distance_mm",
                    count=2,
                    mean=11,
                    minimum=1,
                    maximum=21,
                    std=10,
                    shape=(9, 4, 4),
                )
            finally:
                if recorder.lerobot_dataset is not None:
                    recorder.close()

    def test_undo_all_then_continue_starts_fresh_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stats_empty"
            recorder = self.make_recorder(root)
            try:
                self.record_episode(recorder, 0)
                self.record_episode(recorder, 10)
                self.assertTrue(recorder.rollback_last_episode())
                self.assertTrue(recorder.rollback_last_episode())

                info = self.read_info(root)
                self.assertEqual(info["total_episodes"], 0)
                self.assertEqual(info["total_frames"], 0)
                self.assertFalse((root / "meta" / "stats.json").exists())
                self.assertEqual(list((root / "data").rglob("*.parquet")), [])

                self.record_episode(recorder, 30)
                recorder.close()
                stats = self.read_stats(root)
                self.assertEqual(self.read_info(root)["total_episodes"], 1)
                self.assert_feature_stats(
                    stats,
                    "observation.state",
                    count=1,
                    mean=30,
                    minimum=30,
                    maximum=30,
                    std=0,
                    shape=(7,),
                )
            finally:
                if recorder.lerobot_dataset is not None:
                    recorder.close()

    def test_shared_video_reference_survives_undo_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "video_stats"
            recorder = self.make_recorder(root, camera_num=1)
            try:
                self.record_episode(recorder, 0, image_value=30)
                self.record_episode(recorder, 10, image_value=120)
                self.assertTrue(recorder.rollback_last_episode())

                info = self.read_info(root)
                episode_meta_paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
                self.assertEqual(len(episode_meta_paths), 1)
                episode_meta = pd.read_parquet(episode_meta_paths[0])
                self.assertEqual(episode_meta["episode_index"].tolist(), [0])
                video_key = "observation.images.camera_0"
                video_chunk = int(episode_meta.iloc[0][f"videos/{video_key}/chunk_index"])
                video_file = int(episode_meta.iloc[0][f"videos/{video_key}/file_index"])
                retained_video = root / info["video_path"].format(
                    video_key=video_key,
                    chunk_index=video_chunk,
                    file_index=video_file,
                )
                self.assertTrue(retained_video.is_file())

                self.record_episode(recorder, 20, image_value=210)
                recorder.close()
                stats = self.read_stats(root)
                self.assertEqual(self.read_info(root)["total_episodes"], 2)
                self.assertEqual(
                    stats[video_key]["count"].tolist(),
                    [2],
                )
                self.assertTrue(retained_video.is_file())

                all_episode_meta = pd.concat(
                    [pd.read_parquet(path) for path in (root / "meta" / "episodes").rglob("*.parquet")],
                    ignore_index=True,
                )
                for _, row in all_episode_meta.iterrows():
                    episode_video = root / info["video_path"].format(
                        video_key=video_key,
                        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
                        file_index=int(row[f"videos/{video_key}/file_index"]),
                    )
                    self.assertTrue(episode_video.is_file())
            finally:
                if recorder.lerobot_dataset is not None:
                    recorder.close()

    def test_missing_latest_episode_metadata_never_guesses_a_shared_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "metadata_missing"
            recorder = self.make_recorder(root)
            try:
                self.record_episode(recorder, 0)
                self.record_episode(recorder, 10)
                recorder._finalize_lerobot_dataset()
                recorder.lerobot_dataset = None
                path = next((root / "meta" / "episodes").rglob("*.parquet"))
                metadata = pd.read_parquet(path)
                metadata[metadata["episode_index"] == 0].to_parquet(path, index=False)
                before = {
                    p.relative_to(root): p.read_bytes()
                    for p in root.rglob("*") if p.is_file()
                }
                self.assertFalse(recorder.rollback_last_episode())
                self.assertEqual(recorder.recorded_episodes, 2)
                self.assertEqual(before, {
                    p.relative_to(root): p.read_bytes()
                    for p in root.rglob("*") if p.is_file()
                })
            finally:
                if recorder.lerobot_dataset is not None:
                    recorder.close()

    def test_missing_retained_episode_stats_aborts_before_deleting_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stats_missing"
            recorder = self.make_recorder(root)
            try:
                self.record_episode(recorder, 0)
                self.record_episode(recorder, 10)
                recorder._finalize_lerobot_dataset()
                recorder.lerobot_dataset = None

                episode_meta_path = next((root / "meta" / "episodes").rglob("*.parquet"))
                metadata = pd.read_parquet(episode_meta_path)
                metadata = metadata.drop(columns=["stats/action/count"])
                metadata.to_parquet(episode_meta_path, index=False)
                info_before = self.read_info(root)
                data_before = {
                    path.relative_to(root): path.read_bytes()
                    for path in (root / "data").rglob("*.parquet")
                }

                self.assertFalse(recorder.rollback_last_episode())
                self.assertEqual(self.read_info(root), info_before)
                self.assertEqual(
                    {
                        path.relative_to(root): path.read_bytes()
                        for path in (root / "data").rglob("*.parquet")
                    },
                    data_before,
                )
                self.assertTrue((root / "meta" / "stats.json").is_file())
            finally:
                if recorder.lerobot_dataset is not None:
                    recorder.close()


if __name__ == "__main__":
    unittest.main()
