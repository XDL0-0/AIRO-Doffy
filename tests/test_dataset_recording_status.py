from __future__ import annotations

from pathlib import Path
import unittest

from data_recording import DataRecordingService
from dataset import DatasetRecorder


class DatasetRecordingStatusTests(unittest.TestCase):
    @staticmethod
    def empty_recorder() -> DatasetRecorder:
        recorder = DatasetRecorder.__new__(DatasetRecorder)
        recorder.dataset_type = "l"
        recorder.dataset_dir = Path("unused")
        recorder.collect_step = 0
        recorder.recorded_episodes = 0
        recorder._last_episode_length_cache = None
        recorder._last_episode_length_cache_for = None
        return recorder

    def test_service_status_includes_configured_collect_rate(self) -> None:
        service = DataRecordingService(
            self.empty_recorder(),
            lambda: None,
            collect_rate=24,
        )

        self.assertEqual(service.recording_status()["collect_rate_hz"], 24.0)

    def test_successful_export_retains_completed_episode_length(self) -> None:
        recorder = DatasetRecorder.__new__(DatasetRecorder)
        recorder.dataset_type = "l"
        recorder.dataset_dir = Path("unused")
        recorder.collect_step = 37
        recorder.recorded_episodes = 2
        recorder._last_episode_length_cache = None
        recorder._last_episode_length_cache_for = None
        recorder._export_lerobot = lambda: True

        recorder.data_export(None)

        self.assertEqual(recorder.recorded_episodes, 3)
        self.assertEqual(recorder.recording_status()["last_episode_length"], 37)

    def test_failed_export_does_not_publish_unsaved_length(self) -> None:
        recorder = DatasetRecorder.__new__(DatasetRecorder)
        recorder.dataset_type = "l"
        recorder.dataset_dir = Path("unused")
        recorder.collect_step = 37
        recorder.recorded_episodes = 0
        recorder._last_episode_length_cache = None
        recorder._last_episode_length_cache_for = None
        recorder._export_lerobot = lambda: False

        recorder.data_export(None)

        self.assertEqual(recorder.recorded_episodes, 0)
        self.assertIsNone(recorder.recording_status()["last_episode_length"])

    def test_missing_metadata_length_is_retried(self) -> None:
        recorder = DatasetRecorder.__new__(DatasetRecorder)
        recorder.recorded_episodes = 1
        recorder._last_episode_length_cache = None
        recorder._last_episode_length_cache_for = None
        lengths = iter((None, 19))
        recorder._last_episode_length = lambda _episode_index: next(lengths)

        self.assertIsNone(recorder._cached_last_episode_length())
        self.assertEqual(recorder._cached_last_episode_length(), 19)


if __name__ == "__main__":
    unittest.main()
