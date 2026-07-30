"""Storage-only rollback for LeRobot datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...core.errors import ModelValidationError, OptionalDependencyError


class LeRobotRollback:
    """Remove the latest explicitly indexed LeRobot episode and metadata."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)

    def rollback(self, episode_index: int) -> bool:
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ModelValidationError("episode_index must be a non-negative integer")
        info_path = self.dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            return False
        info = json.loads(info_path.read_text(encoding="utf-8"))
        total_episodes = int(info.get("total_episodes", 0))
        if total_episodes <= 0 or episode_index != total_episodes - 1:
            return False

        episodes_by_file = self._load_episode_metadata()
        episode_row = None
        remaining_episodes = []
        for _path, frame in episodes_by_file:
            if "episode_index" not in frame:
                continue
            hit = frame[frame["episode_index"] == episode_index]
            if not hit.empty and episode_row is None:
                episode_row = hit.iloc[-1]
            remaining_episodes.append(
                frame[frame["episode_index"] != episode_index]
            )

        if episode_row is None:
            episode_length = self._rollback_files_by_index(info, episode_index)
        else:
            episode_length = int(episode_row.get("length", 0))
            self._rollback_data_file(info, episode_index, episode_row)
            self._rollback_video_files(
                info,
                episode_index,
                episode_row,
                remaining_episodes,
            )
        self._remove_episode_metadata(episodes_by_file, episode_index)
        self._update_info(info_path, info, episode_index, episode_length)
        self._remove_derived_metadata(episode_index)
        return True

    def _load_episode_metadata(self) -> list[tuple[Path, Any]]:
        try:
            import pandas
        except ImportError as exc:
            raise OptionalDependencyError(
                "LeRobot rollback requires pandas from the recording extra"
            ) from exc
        root = self.dataset_dir / "meta" / "episodes"
        if not root.exists():
            return []
        return [
            (path, pandas.read_parquet(path))
            for path in sorted(root.rglob("*.parquet"))
        ]

    def _remove_episode_metadata(
        self,
        episodes_by_file: list[tuple[Path, Any]],
        episode_index: int,
    ) -> None:
        stop = self.dataset_dir / "meta" / "episodes"
        for path, frame in episodes_by_file:
            if "episode_index" not in frame:
                continue
            kept = frame[frame["episode_index"] != episode_index]
            if kept.empty:
                path.unlink()
                self._remove_empty_parents(path.parent, stop)
            elif len(kept) != len(frame):
                self._delete_or_rewrite_parquet(
                    path,
                    episode_index,
                    empty_parent_stop=stop,
                )

    def _rollback_data_file(
        self,
        info: dict[str, Any],
        episode_index: int,
        episode_row: Any,
    ) -> None:
        path = self._format_path(
            info["data_path"],
            chunk_index=int(episode_row["data/chunk_index"]),
            file_index=int(episode_row["data/file_index"]),
        )
        self._delete_or_rewrite_parquet(path, episode_index)

    def _rollback_video_files(
        self,
        info: dict[str, Any],
        episode_index: int,
        episode_row: Any,
        remaining_episodes: list[Any],
    ) -> None:
        del episode_index
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") != "video":
                continue
            chunk_key = f"videos/{key}/chunk_index"
            file_key = f"videos/{key}/file_index"
            if chunk_key not in episode_row or file_key not in episode_row:
                continue
            chunk_index = int(episode_row[chunk_key])
            file_index = int(episode_row[file_key])
            still_referenced = any(
                chunk_key in frame
                and file_key in frame
                and not frame[
                    (frame[chunk_key] == chunk_index)
                    & (frame[file_key] == file_index)
                ].empty
                for frame in remaining_episodes
            )
            path = self._format_path(
                info["video_path"],
                video_key=key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not still_referenced and path.exists():
                path.unlink()
                self._remove_empty_parents(
                    path.parent,
                    self.dataset_dir / "videos" / key,
                )

    def _rollback_files_by_index(
        self,
        info: dict[str, Any],
        episode_index: int,
    ) -> int:
        chunks_size = int(info.get("chunks_size", 1000))
        chunk_index = episode_index // chunks_size
        file_index = episode_index % chunks_size
        data_path = self._format_path(
            info["data_path"],
            chunk_index=chunk_index,
            file_index=file_index,
        )
        episode_length = 0
        if data_path.exists():
            episode_length = self._delete_or_rewrite_parquet(
                data_path,
                episode_index,
            )
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") != "video":
                continue
            video_path = self._format_path(
                info["video_path"],
                video_key=key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if video_path.exists():
                video_path.unlink()
                self._remove_empty_parents(
                    video_path.parent,
                    self.dataset_dir / "videos" / key,
                )
        return episode_length

    def _delete_or_rewrite_parquet(
        self,
        path: Path,
        episode_index: int,
        *,
        empty_parent_stop: Path | None = None,
    ) -> int:
        try:
            import pyarrow
            import pyarrow.compute
            import pyarrow.parquet
        except ImportError as exc:
            raise OptionalDependencyError(
                "LeRobot rollback requires pyarrow from the recording extra"
            ) from exc
        if not path.exists():
            return 0
        parent_stop = empty_parent_stop or (self.dataset_dir / "data")
        table = pyarrow.parquet.read_table(path)
        if "episode_index" not in table.column_names:
            path.unlink()
            self._remove_empty_parents(path.parent, parent_stop)
            return 0
        episode_column = table["episode_index"]
        remove_mask = pyarrow.compute.equal(
            episode_column,
            pyarrow.scalar(episode_index, type=episode_column.type),
        )
        removed = int(
            pyarrow.compute.sum(
                pyarrow.compute.cast(remove_mask, pyarrow.int64())
            ).as_py()
            or 0
        )
        if removed == 0:
            return 0
        kept = table.filter(pyarrow.compute.invert(remove_mask))
        if kept.num_rows == 0:
            path.unlink()
            self._remove_empty_parents(path.parent, parent_stop)
        else:
            pyarrow.parquet.write_table(kept, path)
        return removed

    def _update_info(
        self,
        path: Path,
        info: dict[str, Any],
        episode_index: int,
        episode_length: int,
    ) -> None:
        info["total_episodes"] = episode_index
        info["total_frames"] = max(
            0,
            int(info.get("total_frames", 0)) - episode_length,
        )
        info["splits"] = {"train": f"0:{episode_index}"}
        if episode_index == 0:
            info["total_tasks"] = 0
        path.write_text(
            json.dumps(info, indent=4) + "\n",
            encoding="utf-8",
        )

    def _remove_derived_metadata(self, episode_index: int) -> None:
        if episode_index == 0:
            tasks_path = self.dataset_dir / "meta" / "tasks.parquet"
            if tasks_path.exists():
                tasks_path.unlink()
        stats_path = self.dataset_dir / "meta" / "stats.json"
        if stats_path.exists():
            stats_path.unlink()

    def _format_path(self, template: str, **values: object) -> Path:
        relative = Path(template.format(**values))
        if relative.is_absolute():
            raise ModelValidationError("LeRobot metadata path must be relative")
        root = self.dataset_dir.resolve()
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ModelValidationError(
                "LeRobot metadata path resolves outside the dataset directory"
            )
        return candidate

    @staticmethod
    def _remove_empty_parents(path: Path, stop: Path) -> None:
        stop = stop.resolve()
        path = path.resolve()
        while path != stop and stop in path.parents:
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent
