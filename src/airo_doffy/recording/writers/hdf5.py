"""ACT-compatible HDF5 serializer for immutable episodes."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from types import ModuleType

from ...core.errors import ModelValidationError, OptionalDependencyError
from ..samples import Episode, FrozenArray, RecordingSample

_EPISODE_PATTERN = re.compile(r"^episode_(\d+)\.hdf5$")


def _dependencies() -> tuple[ModuleType, ModuleType]:
    try:
        import h5py
        import numpy
    except ImportError as exc:
        raise OptionalDependencyError(
            "HDF5 recording requires the 'recording' optional dependencies"
        ) from exc
    return h5py, numpy


def _array(value: FrozenArray, numpy: ModuleType):
    return numpy.frombuffer(value.data, dtype=value.dtype).reshape(value.shape)


def _named(sample: RecordingSample, group: str, name: str) -> FrozenArray:
    arrays = sample.images if group == "images" else sample.depths
    for item in arrays:
        if item.name == name:
            return item.value
    raise ModelValidationError(f"sample has no {group} value named {name!r}")


class HDF5EpisodeWriter:
    """Atomically create legacy ``episode_<index>.hdf5`` files."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False

    @property
    def next_episode_index(self) -> int:
        return discover_hdf5_next_index(self.dataset_dir)

    def write(self, episode: Episode) -> Path:
        if self._closed:
            raise RuntimeError("HDF5 writer is closed")
        final_path = self.dataset_dir / f"episode_{episode.index}.hdf5"
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite existing episode: {final_path}")
        temporary_path = self.dataset_dir / (
            f".episode_{episode.index}.{uuid.uuid4().hex}.hdf5.tmp"
        )
        committed = False
        try:
            self._write_file(temporary_path, episode)
            os.replace(temporary_path, final_path)
            committed = True
            self._append_description(episode)
        except BaseException:
            if temporary_path.exists():
                temporary_path.unlink()
            if committed and final_path.exists():
                final_path.unlink()
            raise
        return final_path

    def close(self) -> None:
        self._closed = True

    def _write_file(self, path: Path, episode: Episode) -> None:
        h5py, numpy = _dependencies()
        schema = episode.schema
        samples = episode.samples
        with h5py.File(path, "w", rdcc_nbytes=2 * 1024**2) as root:
            root.attrs["sim"] = False
            root.require_group("observations")
            root.require_group("observations/images")
            root.require_group("extra")
            if schema.depth_enabled:
                root.require_group("observations/depth")

            for field in schema.hdf5_fields():
                values = self._field_values(samples, field.path, numpy)
                kwargs: dict[str, object] = {"data": values, "dtype": field.dtype}
                if field.path.startswith("/observations/images/") or field.path.startswith(
                    "/observations/depth/"
                ):
                    kwargs["chunks"] = (1, *field.shape)
                dataset = root.create_dataset(field.path, **kwargs)
                if field.names:
                    dataset.attrs["names"] = numpy.asarray(field.names, dtype="S")

    @staticmethod
    def _field_values(
        samples: tuple[RecordingSample, ...],
        path: str,
        numpy: ModuleType,
    ):
        if path == "/observations/qpos":
            return numpy.asarray([sample.state for sample in samples])
        if path == "/action":
            return numpy.asarray([sample.action for sample in samples])
        if path == "/extra/timestamps_ns":
            return numpy.asarray([sample.timestamps_ns for sample in samples])
        if path == "/extra/tcp_pose":
            return numpy.asarray([sample.tcp_pose for sample in samples])
        if path == "/observations/force":
            return numpy.asarray([sample.force for sample in samples])
        if path == "/observations/torque":
            return numpy.asarray([sample.torque for sample in samples])
        if path == "/observations/tactile":
            return numpy.stack(
                [_array(sample.tactile, numpy) for sample in samples]
            )
        image_prefix = "/observations/images/"
        if path.startswith(image_prefix):
            name = path.removeprefix(image_prefix)
            return numpy.stack(
                [_array(_named(sample, "images", name), numpy) for sample in samples]
            )
        depth_prefix = "/observations/depth/"
        if path.startswith(depth_prefix):
            name = path.removeprefix(depth_prefix)
            return numpy.stack(
                [_array(_named(sample, "depths", name), numpy) for sample in samples]
            )
        raise ModelValidationError(f"unsupported HDF5 field path: {path}")

    def _append_description(self, episode: Episode) -> None:
        description = self.dataset_dir / "episode_descriptions.txt"
        with description.open("a", encoding="utf-8") as stream:
            stream.write(
                f"Episode {episode.index}: max_timesteps = {len(episode.samples)}\n"
            )


class HDF5Rollback:
    """Rollback implementation for legacy HDF5 episode files."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)

    def rollback(self, episode_index: int) -> bool:
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ModelValidationError("episode_index must be a non-negative integer")
        path = self.dataset_dir / f"episode_{episode_index}.hdf5"
        if not path.exists():
            return False
        path.unlink()
        self._trim_description(episode_index)
        return True

    def _trim_description(self, episode_index: int) -> None:
        path = self.dataset_dir / "episode_descriptions.txt"
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        prefix = f"Episode {episode_index}:"
        kept = [line for line in lines if not line.startswith(prefix)]
        path.write_text(
            "\n".join(kept) + ("\n" if kept else ""),
            encoding="utf-8",
        )


def discover_hdf5_next_index(dataset_dir: str | Path) -> int:
    """Return one plus the largest valid legacy episode index."""
    root = Path(dataset_dir)
    if not root.exists():
        return 0
    existing = 0
    for path in root.iterdir():
        match = _EPISODE_PATTERN.match(path.name)
        if match is not None:
            existing = max(existing, int(match.group(1)) + 1)
    return existing
