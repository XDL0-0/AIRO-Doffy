"""Crash-safe, explicit per-episode annotation progress."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SCHEMA_VERSION = 1
ANNOTATIONS_PATH = Path("meta/tightness_annotations.json")
PENDING_PATH = Path("meta/tightness_annotation_pending.json")


def transition_labels(
    episode_length: int,
    tight_frame_index: int | None,
    *,
    has_tight_grasp: bool = True,
) -> np.ndarray:
    """Build binary labels; the selected transition frame belongs to class 1."""
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if not has_tight_grasp:
        if tight_frame_index is not None:
            raise ValueError("no-tight annotations cannot have a transition frame")
        return np.zeros(episode_length, dtype=np.int64)
    if tight_frame_index is None:
        raise ValueError("a tight grasp annotation requires a transition frame")
    if not 0 <= tight_frame_index < episode_length:
        raise ValueError(
            f"tight_frame_index must be in [0, {episode_length - 1}], got {tight_frame_index}"
        )
    labels = np.zeros(episode_length, dtype=np.int64)
    labels[tight_frame_index:] = 1
    return labels


@dataclass(frozen=True)
class AnnotationRecord:
    episode_index: int
    annotated: bool
    has_tight_grasp: bool | None
    tight_frame_index: int | None
    updated_at: str | None

    @classmethod
    def unfinished(cls, episode_index: int) -> AnnotationRecord:
        return cls(episode_index, False, None, None, None)

    @classmethod
    def completed(
        cls,
        episode_index: int,
        *,
        has_tight_grasp: bool,
        tight_frame_index: int | None,
    ) -> AnnotationRecord:
        return cls(
            episode_index=episode_index,
            annotated=True,
            has_tight_grasp=has_tight_grasp,
            tight_frame_index=tight_frame_index,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )


class AnnotationWriter(Protocol):
    def write_annotation(self, record: AnnotationRecord) -> None: ...


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON using fsync + same-directory replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class AnnotationStore:
    """Own progress metadata and coordinate recoverable two-file saves."""

    def __init__(self, output_root: Path, document: dict[str, Any]) -> None:
        self.output_root = Path(output_root)
        self._document = document
        self._lock = threading.RLock()
        self._validate_document()

    @classmethod
    def create(
        cls,
        output_root: Path,
        *,
        source: dict[str, Any],
        total_episodes: int,
    ) -> AnnotationStore:
        now = datetime.now(timezone.utc).isoformat()
        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "feature": "tightness",
            "source": source,
            "created_at": now,
            "updated_at": now,
            "episodes": {
                str(index): asdict(AnnotationRecord.unfinished(index))
                for index in range(total_episodes)
            },
        }
        atomic_write_json(Path(output_root) / ANNOTATIONS_PATH, document)
        return cls(Path(output_root), document)

    @classmethod
    def load(cls, output_root: Path) -> AnnotationStore:
        path = Path(output_root) / ANNOTATIONS_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"Existing output dataset has no annotation progress metadata: {path}"
            )
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        return cls(Path(output_root), document)

    def _validate_document(self) -> None:
        if self._document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported tightness annotation metadata schema")
        if self._document.get("feature") != "tightness":
            raise ValueError("Annotation metadata does not describe tightness")
        episodes = self._document.get("episodes")
        if not isinstance(episodes, dict):
            raise TypeError("Annotation metadata episode map must be an object")
        expected = list(range(len(episodes)))
        actual = sorted(int(index) for index in episodes)
        if actual != expected:
            raise ValueError("Annotation episode indices must be contiguous from zero")
        for index in expected:
            record = self._record_from_dict(episodes[str(index)])
            if record.episode_index != index:
                raise ValueError(f"Annotation record {index} has the wrong episode_index")
            if record.annotated and record.has_tight_grasp is None:
                raise ValueError(f"Completed annotation {index} has no grasp state")
            if record.annotated and record.has_tight_grasp and record.tight_frame_index is None:
                raise ValueError(f"Tight annotation {index} has no transition frame")
            if record.annotated and not record.has_tight_grasp and record.tight_frame_index is not None:
                raise ValueError(f"No-tight annotation {index} has a transition frame")
            if not record.annotated and any(
                value is not None
                for value in (
                    record.has_tight_grasp,
                    record.tight_frame_index,
                    record.updated_at,
                )
            ):
                raise ValueError(f"Unfinished annotation {index} contains saved values")

    @staticmethod
    def _record_from_dict(value: dict[str, Any]) -> AnnotationRecord:
        return AnnotationRecord(
            episode_index=int(value["episode_index"]),
            annotated=bool(value["annotated"]),
            has_tight_grasp=value.get("has_tight_grasp"),
            tight_frame_index=value.get("tight_frame_index"),
            updated_at=value.get("updated_at"),
        )

    @property
    def total_episodes(self) -> int:
        return len(self._document["episodes"])

    @property
    def annotated_count(self) -> int:
        return sum(record.annotated for record in self.records())

    @property
    def first_unfinished(self) -> int:
        for record in self.records():
            if not record.annotated:
                return record.episode_index
        return max(0, self.total_episodes - 1)

    @property
    def source(self) -> dict[str, Any]:
        return dict(self._document["source"])

    def records(self) -> list[AnnotationRecord]:
        return [self.get(index) for index in range(self.total_episodes)]

    def get(self, episode_index: int) -> AnnotationRecord:
        try:
            value = self._document["episodes"][str(episode_index)]
        except KeyError as error:
            raise IndexError(f"Unknown episode {episode_index}") from error
        return self._record_from_dict(value)

    def save(
        self,
        writer: AnnotationWriter,
        *,
        episode_index: int,
        episode_length: int,
        has_tight_grasp: bool,
        tight_frame_index: int | None,
    ) -> AnnotationRecord:
        transition_labels(
            episode_length,
            tight_frame_index,
            has_tight_grasp=has_tight_grasp,
        )
        if not 0 <= episode_index < self.total_episodes:
            raise IndexError(f"Unknown episode {episode_index}")
        record = AnnotationRecord.completed(
            episode_index,
            has_tight_grasp=has_tight_grasp,
            tight_frame_index=tight_frame_index,
        )
        with self._lock:
            atomic_write_json(
                self.output_root / PENDING_PATH,
                {"schema_version": SCHEMA_VERSION, "record": asdict(record)},
            )
            writer.write_annotation(record)
            self._commit_record(record)
            self._clear_pending()
        return record

    def recover_pending(self, writer: AnnotationWriter) -> AnnotationRecord | None:
        """Finish an interrupted save. Rewriting the same labels is idempotent."""
        path = self.output_root / PENDING_PATH
        if not path.exists():
            return None
        with self._lock:
            with path.open("r", encoding="utf-8") as stream:
                pending = json.load(stream)
            if pending.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported pending annotation schema")
            record = self._record_from_dict(pending["record"])
            if not record.annotated or record.has_tight_grasp is None:
                raise ValueError("A pending annotation must be completed and explicit")
            writer.write_annotation(record)
            self._commit_record(record)
            self._clear_pending()
            return record

    def _commit_record(self, record: AnnotationRecord) -> None:
        self._document["episodes"][str(record.episode_index)] = asdict(record)
        self._document["updated_at"] = record.updated_at
        atomic_write_json(self.output_root / ANNOTATIONS_PATH, self._document)

    def _clear_pending(self) -> None:
        path = self.output_root / PENDING_PATH
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
