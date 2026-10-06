"""Lazy, frame-accurate LeRobot video/image decoding."""

from __future__ import annotations

import io
import threading
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.video_utils import decode_video_frames
from PIL import Image

from .dataset_adapter import TightnessDatasetAdapter
from .lerobot_compat import get_raw_item


class EpisodeVideoReader:
    """Decode only requested frames using v3 episode video offsets."""

    def __init__(self, adapter: TightnessDatasetAdapter, cache_size: int = 96) -> None:
        self.adapter = adapter
        self._decode_lock = threading.RLock()
        # lru_cache needs a compile-time maxsize, so wrap the implementation here.
        self._cached_jpeg = lru_cache(maxsize=cache_size)(self._decode_jpeg)
        self._cached_timestamps = lru_cache(maxsize=24)(self._load_episode_timestamps)

    def episode_timestamps(self, episode_index: int) -> tuple[float, ...]:
        return self._cached_timestamps(episode_index)

    def _load_episode_timestamps(self, episode_index: int) -> tuple[float, ...]:
        episode = self.adapter.episode(episode_index)
        length = int(episode["length"])
        path = self.adapter.source_root / self.adapter.source_meta.get_data_file_path(
            episode_index
        )
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "timestamp"]
        )
        episodes = table["episode_index"].to_numpy(zero_copy_only=False)
        frames = table["frame_index"].to_numpy(zero_copy_only=False)
        timestamps = table["timestamp"].to_numpy(zero_copy_only=False)
        selected = episodes == episode_index
        selected_frames = frames[selected].astype(np.int64, copy=False)
        selected_timestamps = timestamps[selected]
        if len(selected_frames) != length or not np.array_equal(
            np.sort(selected_frames), np.arange(length)
        ):
            raise ValueError(f"Episode {episode_index} frame timestamps are misaligned")
        ordered = np.empty(length, dtype=np.float64)
        ordered[selected_frames] = selected_timestamps
        return tuple(float(value) for value in ordered)

    def jpeg(self, episode_index: int, frame_index: int, camera_key: str) -> bytes:
        if camera_key not in self.adapter.camera_keys:
            raise KeyError(f"Unknown camera {camera_key!r}")
        length = self.adapter.episode_length(episode_index)
        if not 0 <= frame_index < length:
            raise IndexError(
                f"Frame {frame_index} is outside episode {episode_index} [0, {length - 1}]"
            )
        return self._cached_jpeg(episode_index, frame_index, camera_key)

    def _decode_jpeg(self, episode_index: int, frame_index: int, camera_key: str) -> bytes:
        dataset = self.adapter.source_dataset
        meta = self.adapter.source_meta
        episode = self.adapter.episode(episode_index)
        absolute_index = int(episode["dataset_from_index"]) + frame_index
        local_timestamp = self.episode_timestamps(episode_index)[frame_index]

        with self._decode_lock:
            if camera_key in meta.video_keys:
                video_path = self.adapter.source_root / meta.get_video_file_path(
                    episode_index, camera_key
                )
                offset = float(episode[f"videos/{camera_key}/from_timestamp"])
                image = decode_video_frames(
                    video_path,
                    [offset + local_timestamp],
                    dataset.tolerance_s,
                    self.adapter.video_backend,
                ).squeeze(0)
            else:
                image = get_raw_item(dataset, absolute_index)[camera_key]
        return self._encode_jpeg(image)

    @staticmethod
    def _encode_jpeg(image: object) -> bytes:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            if isinstance(image, torch.Tensor):
                array = image.detach().cpu().numpy()
            else:
                array = np.asarray(image)
            if array.ndim != 3:
                raise ValueError(f"Expected a 3-D camera frame, got shape {array.shape}")
            if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, 0.0, 1.0) * 255.0
            array = np.asarray(array, dtype=np.uint8)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            if array.shape[-1] == 4:
                array = array[..., :3]
            pil_image = Image.fromarray(array, mode="RGB")

        output = io.BytesIO()
        pil_image.save(output, format="JPEG", quality=90, optimize=False)
        return output.getvalue()
