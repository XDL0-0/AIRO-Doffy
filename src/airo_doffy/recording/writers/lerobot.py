"""LeRobot serializer with an injected dataset provider."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from ...core.errors import OptionalDependencyError
from ..errors import RecordingSchemaMismatchError
from ..samples import Episode, FrozenArray, RecordingSample
from ..schema import RecordingSchema


def _numpy() -> ModuleType:
    try:
        import numpy
    except ImportError as exc:
        raise OptionalDependencyError(
            "LeRobot recording requires the 'recording' optional dependencies"
        ) from exc
    return numpy


def _array(value: FrozenArray, numpy: ModuleType):
    return numpy.frombuffer(value.data, dtype=value.dtype).reshape(value.shape)


class LeRobotEpisodeWriter:
    """Serialize a detached episode through a caller-provided LeRobot dataset."""

    def __init__(
        self,
        dataset_provider: Callable[[], Any],
        *,
        expected_schema: RecordingSchema,
    ) -> None:
        if not callable(dataset_provider):
            raise TypeError("dataset_provider must be callable")
        self._dataset_provider = dataset_provider
        self._schema = expected_schema
        self._closed = False

    def write(self, episode: Episode) -> Path | None:
        if self._closed:
            raise RuntimeError("LeRobot writer is closed")
        if episode.schema != self._schema:
            raise RecordingSchemaMismatchError(
                "episode schema does not match the LeRobot writer schema"
            )
        dataset = self._dataset_provider()
        self._validate_features(dataset)
        try:
            if hasattr(dataset, "create_episode_buffer"):
                dataset.episode_buffer = dataset.create_episode_buffer(
                    episode_index=episode.index
                )
            for sample in episode.samples:
                dataset.add_frame(self._frame(sample, episode.task))
            dataset.save_episode()
        except BaseException:
            if hasattr(dataset, "clear_episode_buffer"):
                dataset.clear_episode_buffer()
            raise
        finally:
            if hasattr(dataset, "finalize"):
                dataset.finalize()
        root = getattr(dataset, "root", None)
        return None if root is None else Path(root)

    def close(self) -> None:
        self._closed = True

    def _validate_features(self, dataset: Any) -> None:
        actual = {
            key
            for key in getattr(dataset, "features", {})
            if key.startswith(("action", "observation.", "extra."))
        }
        expected = set(self._schema.lerobot_features())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RecordingSchemaMismatchError(
                f"LeRobot feature mismatch; missing={missing}, extra={extra}"
            )

    def _frame(self, sample: RecordingSample, task: str) -> dict[str, object]:
        numpy = _numpy()
        frame: dict[str, object] = {
            "observation.state": numpy.asarray(sample.state, dtype=numpy.float32),
            "action": numpy.asarray(sample.action, dtype=numpy.float32),
            "extra.timestamps_ns": numpy.asarray(
                sample.timestamps_ns,
                dtype=numpy.int64,
            ),
            "task": task,
        }
        if sample.tcp_pose is not None:
            frame["extra.tcp_pose"] = numpy.asarray(
                sample.tcp_pose,
                dtype=numpy.float32,
            )
        if sample.force is not None:
            frame["observation.force"] = numpy.asarray(
                sample.force,
                dtype=numpy.float32,
            )
        if sample.torque is not None:
            frame["observation.torque"] = numpy.asarray(
                sample.torque,
                dtype=numpy.float32,
            )
        if sample.tactile is not None:
            frame["observation.tactile"] = _array(sample.tactile, numpy).copy()
        for item in sample.images:
            frame[f"observation.images.{item.name}"] = _array(
                item.value,
                numpy,
            ).copy()
        for item in sample.depths:
            depth_m = _array(item.value, numpy)
            depth_mm = (
                numpy.clip(depth_m, 0, 65.535) * 1000
            ).astype(numpy.uint16)
            frame[f"observation.depth.{item.name}"] = depth_mm[..., numpy.newaxis]
        return frame
