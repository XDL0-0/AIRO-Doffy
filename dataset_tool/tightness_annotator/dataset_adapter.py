"""LeRobot v3 derived-dataset creation and per-episode label persistence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .annotation_store import (
    AnnotationRecord,
    AnnotationStore,
    atomic_write_json,
    transition_labels,
)
from .lerobot_compat import LeRobotDatasetMetadata

TIGHTNESS_FEATURE = {"dtype": "int64", "shape": (1,), "names": None}


def default_output_root(source_root: Path) -> Path:
    source_root = Path(source_root)
    return source_root.with_name(f"{source_root.name}_tightness")


def derived_repo_id(repo_id: str) -> str:
    prefix, separator, name = repo_id.rpartition("/")
    derived_name = f"{name if separator else repo_id}_tightness"
    return f"{prefix}/{derived_name}" if separator else derived_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TightnessDatasetAdapter:
    """Own source/target LeRobot metadata and durable label updates."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        output_root: str | Path | None = None,
        repo_id: str | None = None,
        output_repo_id: str | None = None,
        video_backend: str = "pyav",
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.output_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else default_output_root(self.source_root)
        )
        if self.source_root == self.output_root:
            raise ValueError("Output dataset must differ from the source dataset")
        if self.output_root.is_relative_to(self.source_root) or self.source_root.is_relative_to(
            self.output_root
        ):
            raise ValueError("Source and output datasets must not contain one another")
        if not (self.source_root / "meta/info.json").is_file():
            raise FileNotFoundError(f"Not a LeRobot dataset: {self.source_root}")

        self.repo_id = repo_id or self.source_root.name
        self.output_repo_id = output_repo_id or derived_repo_id(self.repo_id)
        self.video_backend = video_backend
        self._write_lock = threading.RLock()

        self.source_dataset = LeRobotDataset(
            self.repo_id,
            root=self.source_root,
            download_videos=False,
            video_backend=video_backend,
        )
        self.source_meta = self.source_dataset.meta
        self._validate_v3(self.source_meta, "Source")
        if "tightness" in self.source_meta.features:
            raise ValueError("Source dataset already contains a tightness feature")

        if self.output_root.exists():
            if not self.output_root.is_dir():
                raise FileExistsError(f"Output path is not a directory: {self.output_root}")
            if not (self.output_root / "meta/info.json").is_file():
                raise FileNotFoundError(
                    f"Existing output is not a complete LeRobot dataset: {self.output_root}"
                )
        else:
            self._create_output_dataset()

        self.output_meta = LeRobotDatasetMetadata(
            self.output_repo_id, root=self.output_root
        )
        self._validate_output()
        self.store = AnnotationStore.load(self.output_root)
        self._validate_store_identity()
        self._acquire_process_lock()
        self.store.recover_pending(self)

    def _acquire_process_lock(self) -> None:
        """Prevent two annotator processes from rewriting one shared shard."""
        import fcntl

        lock_path = self.output_root / "meta/.tightness_annotator.lock"
        self._lock_stream = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_stream.close()
            raise RuntimeError(
                f"Another annotator is already writing {self.output_root}"
            ) from error

    def close(self) -> None:
        stream = getattr(self, "_lock_stream", None)
        if stream is None or stream.closed:
            return
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

    @staticmethod
    def _validate_v3(meta: LeRobotDatasetMetadata, label: str) -> None:
        version = str(meta.info.get("codebase_version", ""))
        if version != "v3.0":
            raise ValueError(f"{label} dataset must be LeRobot v3.0, got {version!r}")

    def _source_identity(self) -> dict[str, Any]:
        return {
            "root": str(self.source_root),
            "repo_id": self.repo_id,
            "info_sha256": _sha256(self.source_root / "meta/info.json"),
            "codebase_version": self.source_meta.info["codebase_version"],
            "total_episodes": self.source_meta.total_episodes,
            "total_frames": self.source_meta.total_frames,
            "fps": self.source_meta.fps,
        }

    def _create_output_dataset(self) -> None:
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_root.parent / (
            f".{self.output_root.name}.creating-{uuid.uuid4().hex}"
        )
        try:
            zeros = np.zeros(self.source_meta.total_frames, dtype=np.int64)
            add_features(
                self.source_dataset,
                {"tightness": (zeros, TIGHTNESS_FEATURE)},
                output_dir=staging,
                repo_id=self.output_repo_id,
            )
            self._preserve_source_info(staging)
            self._copy_unhandled_source_files(staging)
            AnnotationStore.create(
                staging,
                source=self._source_identity(),
                total_episodes=self.source_meta.total_episodes,
            )
            candidate = LeRobotDataset(
                self.output_repo_id,
                root=staging,
                download_videos=False,
                video_backend=self.video_backend,
            )
            if (
                candidate.num_episodes != self.source_meta.total_episodes
                or candidate.num_frames != self.source_meta.total_frames
                or "tightness" not in candidate.features
            ):
                raise ValueError("Official LeRobot validation of the staged output failed")
            os.replace(staging, self.output_root)
            _fsync_directory(self.output_root.parent)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _preserve_source_info(self, staging: Path) -> None:
        """Retain source-level metadata while keeping the generated feature addition."""
        with (self.source_root / "meta/info.json").open("r", encoding="utf-8") as stream:
            source_info = json.load(stream)
        with (staging / "meta/info.json").open("r", encoding="utf-8") as stream:
            generated_info = json.load(stream)
        source_info["features"] = {
            **source_info["features"],
            "tightness": generated_info["features"]["tightness"],
        }
        atomic_write_json(staging / "meta/info.json", source_info)

    def _copy_unhandled_source_files(self, staging: Path) -> None:
        """Keep optional dataset cards/licenses/custom metadata add_features omits."""
        for source_path in self.source_root.rglob("*"):
            if not source_path.is_file():
                continue
            relative = source_path.relative_to(self.source_root)
            destination = staging / relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

    def _validate_output(self) -> None:
        self._validate_v3(self.output_meta, "Output")
        feature = self.output_meta.features.get("tightness")
        if feature is None or feature["dtype"] != "int64" or tuple(feature["shape"]) != (1,):
            raise ValueError("Output dataset does not have scalar int64 tightness")
        comparisons = {
            "episode count": (self.source_meta.total_episodes, self.output_meta.total_episodes),
            "frame count": (self.source_meta.total_frames, self.output_meta.total_frames),
            "FPS": (self.source_meta.fps, self.output_meta.fps),
        }
        for name, (source_value, output_value) in comparisons.items():
            if source_value != output_value:
                raise ValueError(
                    f"Output {name} differs from source: {output_value} != {source_value}"
                )
        expected_features = set(self.source_meta.features) | {"tightness"}
        if set(self.output_meta.features) != expected_features:
            raise ValueError("Output dataset does not preserve the source feature set")
        if self.output_meta.camera_keys != self.source_meta.camera_keys:
            raise ValueError("Output camera keys differ from source")
        for episode_index in range(self.source_meta.total_episodes):
            source_episode = self.source_meta.episodes[episode_index]
            output_episode = self.output_meta.episodes[episode_index]
            if source_episode["length"] != output_episode["length"]:
                raise ValueError(f"Episode {episode_index} length differs in output")
            for key in ("episode_index", "dataset_from_index", "dataset_to_index"):
                if source_episode[key] != output_episode[key]:
                    raise ValueError(f"Episode {episode_index} metadata field {key} differs")
            for camera_key in self.source_meta.video_keys:
                for suffix in (
                    "chunk_index",
                    "file_index",
                    "from_timestamp",
                    "to_timestamp",
                ):
                    key = f"videos/{camera_key}/{suffix}"
                    if source_episode[key] != output_episode[key]:
                        raise ValueError(
                            f"Episode {episode_index} video metadata field {key} differs"
                        )

    def _validate_store_identity(self) -> None:
        expected = self._source_identity()
        stored = self.store.source
        for key in ("info_sha256", "total_episodes", "total_frames", "fps"):
            if stored.get(key) != expected[key]:
                raise ValueError(
                    f"Existing annotations belong to a different source dataset ({key})"
                )
        if self.store.total_episodes != self.source_meta.total_episodes:
            raise ValueError("Annotation progress episode count differs from source")

    @property
    def total_episodes(self) -> int:
        return self.source_meta.total_episodes

    @property
    def total_frames(self) -> int:
        return self.source_meta.total_frames

    @property
    def fps(self) -> int:
        return self.source_meta.fps

    @property
    def camera_keys(self) -> list[str]:
        return list(self.source_meta.camera_keys)

    def episode(self, episode_index: int) -> dict[str, Any]:
        if not 0 <= episode_index < self.total_episodes:
            raise IndexError(f"Unknown episode {episode_index}")
        return dict(self.source_meta.episodes[episode_index])

    def episode_length(self, episode_index: int) -> int:
        return int(self.episode(episode_index)["length"])

    def save_annotation(
        self,
        episode_index: int,
        *,
        has_tight_grasp: bool,
        tight_frame_index: int | None,
    ) -> AnnotationRecord:
        return self.store.save(
            self,
            episode_index=episode_index,
            episode_length=self.episode_length(episode_index),
            has_tight_grasp=has_tight_grasp,
            tight_frame_index=tight_frame_index,
        )

    def write_annotation(self, record: AnnotationRecord) -> None:
        episode_index = record.episode_index
        episode_length = self.episode_length(episode_index)
        labels = transition_labels(
            episode_length,
            record.tight_frame_index,
            has_tight_grasp=bool(record.has_tight_grasp),
        )
        relative_path = self.output_meta.get_data_file_path(episode_index)
        parquet_path = self.output_root / relative_path
        with self._write_lock:
            table = pq.read_table(parquet_path)
            episodes = table["episode_index"].to_numpy(zero_copy_only=False)
            frame_indices = table["frame_index"].to_numpy(zero_copy_only=False)
            mask = episodes == episode_index
            matched_frames = frame_indices[mask].astype(np.int64, copy=False)
            if len(matched_frames) != episode_length:
                raise ValueError(
                    f"Episode {episode_index} has {len(matched_frames)} rows, expected {episode_length}"
                )
            if not np.array_equal(np.sort(matched_frames), np.arange(episode_length)):
                raise ValueError(f"Episode {episode_index} frame indices are not 0..N-1")

            tightness = table["tightness"].to_numpy(zero_copy_only=False).astype(
                np.int64, copy=True
            )
            tightness[mask] = labels[matched_frames]
            column_index = table.schema.get_field_index("tightness")
            updated = table.set_column(
                column_index, "tightness", pa.array(tightness, type=pa.int64())
            )
            self._atomic_write_parquet(parquet_path, updated)

    @staticmethod
    def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            pq.write_table(
                table,
                temporary_path,
                compression="snappy",
                use_dictionary=True,
            )
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            _fsync_directory(path.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
