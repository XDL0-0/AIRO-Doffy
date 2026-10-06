from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hashlib
import tempfile
import unittest
from dataclasses import asdict

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dataset_tool.tightness_annotator.annotation_store import (
    PENDING_PATH,
    AnnotationRecord,
    AnnotationStore,
    atomic_write_json,
    transition_labels,
)
from dataset_tool.tightness_annotator.dataset_adapter import TightnessDatasetAdapter


def create_source_dataset(root: Path, lengths: tuple[int, ...] = (3, 4)) -> Path:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["episode", "frame"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["frame", "episode"],
        },
    }
    dataset = LeRobotDataset.create(
        "fixture",
        fps=10,
        features=features,
        root=root,
        use_videos=False,
    )
    for episode_index, length in enumerate(lengths):
        for frame_index in range(length):
            dataset.add_frame(
                {
                    "observation.state": np.asarray(
                        [episode_index, frame_index], dtype=np.float32
                    ),
                    "action": np.asarray(
                        [frame_index, episode_index], dtype=np.float32
                    ),
                    "task": "fixture task",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


def manifest(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def tightness_by_episode(root: Path) -> dict[int, list[int]]:
    result: dict[int, list[tuple[int, int]]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "tightness"])
        episodes = table["episode_index"].to_pylist()
        frames = table["frame_index"].to_pylist()
        values = table["tightness"].to_pylist()
        for episode, frame, value in zip(episodes, frames, values, strict=True):
            result.setdefault(int(episode), []).append((int(frame), int(value)))
    return {
        episode: [value for _, value in sorted(frame_values)]
        for episode, frame_values in result.items()
    }


class TransitionLabelsTest(unittest.TestCase):
    def test_middle_transition_is_inclusive(self) -> None:
        np.testing.assert_array_equal(
            transition_labels(5, 3), np.asarray([0, 0, 0, 1, 1])
        )

    def test_first_frame_transition(self) -> None:
        np.testing.assert_array_equal(transition_labels(5, 0), np.ones(5, dtype=np.int64))

    def test_last_frame_transition(self) -> None:
        np.testing.assert_array_equal(
            transition_labels(5, 4), np.asarray([0, 0, 0, 0, 1])
        )

    def test_no_tight_grasp(self) -> None:
        labels = transition_labels(5, None, has_tight_grasp=False)
        np.testing.assert_array_equal(labels, np.zeros(5, dtype=np.int64))
        self.assertEqual(labels.dtype, np.int64)

    def test_invalid_transitions_are_rejected(self) -> None:
        for length, frame in ((0, 0), (5, -1), (5, 5), (5, None)):
            with self.subTest(length=length, frame=frame), self.assertRaises(ValueError):
                transition_labels(length, frame)
        with self.assertRaises(ValueError):
            transition_labels(5, 2, has_tight_grasp=False)


class AnnotationStoreTest(unittest.TestCase):
    class Writer:
        def __init__(self) -> None:
            self.records: list[AnnotationRecord] = []

        def write_annotation(self, record: AnnotationRecord) -> None:
            self.records.append(record)

    def test_save_load_edit_and_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tightness-store-") as temporary:
            root = Path(temporary)
            store = AnnotationStore.create(
                root,
                source={"info_sha256": "fixture"},
                total_episodes=3,
            )
            writer = self.Writer()
            store.save(
                writer,
                episode_index=1,
                episode_length=5,
                has_tight_grasp=False,
                tight_frame_index=None,
            )
            self.assertEqual(store.first_unfinished, 0)
            self.assertTrue(store.get(1).annotated)
            self.assertFalse(store.get(1).has_tight_grasp)

            reloaded = AnnotationStore.load(root)
            reloaded.save(
                writer,
                episode_index=1,
                episode_length=5,
                has_tight_grasp=True,
                tight_frame_index=4,
            )
            self.assertEqual(reloaded.annotated_count, 1)
            self.assertTrue(reloaded.get(1).has_tight_grasp)
            self.assertEqual(reloaded.get(1).tight_frame_index, 4)
            self.assertFalse((root / PENDING_PATH).exists())

    def test_pending_save_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tightness-store-") as temporary:
            root = Path(temporary)
            store = AnnotationStore.create(
                root,
                source={"info_sha256": "fixture"},
                total_episodes=2,
            )
            record = AnnotationRecord.completed(
                0, has_tight_grasp=True, tight_frame_index=2
            )
            atomic_write_json(
                root / PENDING_PATH,
                {"schema_version": 1, "record": asdict(record)},
            )
            writer = self.Writer()
            recovered = store.recover_pending(writer)
            self.assertEqual(recovered, record)
            self.assertEqual(writer.records, [record])
            self.assertEqual(store.get(0), record)
            self.assertFalse((root / PENDING_PATH).exists())


class DatasetAdapterIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="tightness-adapter-")
        self.root = Path(self.temporary.name)
        self.source = create_source_dataset(self.root / "source")
        self.output = self.root / "source_tightness"
        self.source_manifest = manifest(self.source)
        self.adapter: TightnessDatasetAdapter | None = None

    def tearDown(self) -> None:
        if self.adapter is not None:
            self.adapter.close()
        self.temporary.cleanup()

    def open_adapter(self) -> TightnessDatasetAdapter:
        self.adapter = TightnessDatasetAdapter(
            source_root=self.source,
            output_root=self.output,
            repo_id="fixture",
            output_repo_id="fixture_tightness",
        )
        return self.adapter

    def test_creation_save_edit_resume_and_source_immutability(self) -> None:
        adapter = self.open_adapter()
        self.assertEqual(manifest(self.source), self.source_manifest)
        self.assertEqual(adapter.total_episodes, 2)
        self.assertEqual(adapter.episode_length(0), 3)
        self.assertEqual(adapter.episode_length(1), 4)
        self.assertEqual(tightness_by_episode(self.output), {0: [0, 0, 0], 1: [0, 0, 0, 0]})

        # Both episodes share one physical Parquet file. Updating one must not alter the other.
        self.assertEqual(len(list((self.output / "data").rglob("*.parquet"))), 1)
        adapter.save_annotation(0, has_tight_grasp=True, tight_frame_index=1)
        self.assertEqual(tightness_by_episode(self.output), {0: [0, 1, 1], 1: [0, 0, 0, 0]})
        adapter.save_annotation(1, has_tight_grasp=False, tight_frame_index=None)
        adapter.save_annotation(0, has_tight_grasp=True, tight_frame_index=0)
        self.assertEqual(tightness_by_episode(self.output), {0: [1, 1, 1], 1: [0, 0, 0, 0]})
        self.assertEqual(adapter.store.annotated_count, 2)
        self.assertEqual(manifest(self.source), self.source_manifest)

        adapter.close()
        self.adapter = None
        resumed = self.open_adapter()
        self.assertEqual(resumed.store.annotated_count, 2)
        self.assertEqual(resumed.store.get(0).tight_frame_index, 0)
        self.assertFalse(resumed.store.get(1).has_tight_grasp)

        output_dataset = LeRobotDataset(
            "fixture_tightness",
            root=self.output,
            download_videos=False,
            video_backend="pyav",
        )
        self.assertEqual(output_dataset.num_episodes, 2)
        self.assertEqual(output_dataset.num_frames, 7)
        self.assertEqual(output_dataset.features["tightness"]["dtype"], "int64")

        source_table = pq.read_table(next((self.source / "data").rglob("*.parquet")))
        output_table = pq.read_table(next((self.output / "data").rglob("*.parquet")))
        for name in source_table.column_names:
            self.assertTrue(
                source_table[name].equals(output_table[name]),
                f"Non-tightness column changed: {name}",
            )

    def test_pending_dataset_write_recovers_on_restart(self) -> None:
        adapter = self.open_adapter()
        record = AnnotationRecord.completed(1, has_tight_grasp=True, tight_frame_index=3)
        atomic_write_json(
            self.output / PENDING_PATH,
            {"schema_version": 1, "record": asdict(record)},
        )
        adapter.close()
        self.adapter = None

        resumed = self.open_adapter()
        self.assertEqual(resumed.store.get(1).tight_frame_index, 3)
        self.assertEqual(tightness_by_episode(self.output)[1], [0, 0, 0, 1])
        self.assertFalse((self.output / PENDING_PATH).exists())

    def test_nested_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain"):
            TightnessDatasetAdapter(
                source_root=self.source,
                output_root=self.source / "derived",
                repo_id="fixture",
            )

    def test_existing_output_without_progress_is_not_overwritten(self) -> None:
        self.output.mkdir()
        sentinel = self.output / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            self.open_adapter()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
