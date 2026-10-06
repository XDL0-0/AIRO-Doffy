"""Camera storage, acquisition workers, and a transport-neutral video track."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import threading
import time
from typing import Any

import numpy as np

from .frames import prepare_rgb_frame

try:  # Optional until a WebRTC track is actually requested.
    from av import VideoFrame
    from aiortc.mediastreams import VideoStreamTrack
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    VideoFrame = Any  # type: ignore[assignment,misc]

    class VideoStreamTrack:  # type: ignore[no-redef]
        kind = "video"


class FrameStore:
    """Latest-frame store with the dictionaries expected by v1 callers."""

    def __init__(self, *, lock: threading.Lock | threading.RLock | None = None) -> None:
        self._lock = lock or threading.RLock()
        self.camera_images: dict[str, np.ndarray] = {}
        self.camera_image_timestamps_ns: dict[str, int] = {}
        self.depth_images: dict[str, np.ndarray] = {}
        self.depth_timestamps_ns: dict[str, int] = {}

    @property
    def lock(self) -> threading.Lock | threading.RLock:
        return self._lock

    def publish(
        self,
        name: str,
        image: np.ndarray,
        timestamp_ns: int,
        depth: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            self.camera_images[name] = image
            self.camera_image_timestamps_ns[name] = int(timestamp_ns)
            if depth is not None:
                self.depth_images[name] = depth
                self.depth_timestamps_ns[name] = int(timestamp_ns)

    def read(self, name: str) -> np.ndarray | None:
        with self._lock:
            return self.camera_images.get(name)


class CameraFrameProvider:
    """Read and process frames from either a ``FrameStore`` or legacy manager."""

    def __init__(
        self,
        camera_data: Mapping[str, np.ndarray],
        lock: threading.Lock | threading.RLock,
        zoom: list[float] | None = None,
        *,
        running: Callable[[], bool] | None = None,
    ) -> None:
        self.camera_data = camera_data
        self.lock = lock
        self.zoom = zoom if zoom is not None else []
        self._running = running

    @classmethod
    def from_owner(cls, owner: Any) -> "CameraFrameProvider":
        return cls(
            owner.camera_data,
            owner._lock,
            getattr(owner, "camera_zoom", None),
            running=lambda: bool(getattr(owner, "running", True)),
        )

    def read(self, cam_idx: int) -> np.ndarray | None:
        with self.lock:
            frame = self.camera_data.get(f"camera_{int(cam_idx)}")
        return frame

    def process(self, cam_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        frame = self.read(cam_idx)
        if frame is None:
            return None
        zoom = self.zoom[cam_idx] if 0 <= cam_idx < len(self.zoom) else 1.0
        return prepare_rgb_frame(frame, zoom=zoom)

    def is_running(self) -> bool:
        return self._running is None or bool(self._running())


class CameraCaptureService:
    """Own camera acquisition workers while leaving transport to another class."""

    def __init__(
        self,
        camera_list: Mapping[str, Any],
        store: FrameStore,
        *,
        fps: float = 30.0,
        depth_enabled: bool = False,
        max_retries: int = 10,
        retry_delay_s: float = 1.0,
    ) -> None:
        if fps <= 0.0 or max_retries < 1 or retry_delay_s < 0.0:
            raise ValueError("invalid camera capture timing or retry settings")
        self.camera_list = dict(camera_list)
        self.store = store
        self.fps = float(fps)
        self.depth_enabled = bool(depth_enabled)
        self.max_retries = int(max_retries)
        self.retry_delay_s = float(retry_delay_s)
        self.running = False
        self._closed = False
        self.threads: list[threading.Thread] = []

    def _read_thread(self, name: str, camera: Any) -> None:
        idx = int(name.rsplit("_", 1)[-1])
        failures = 0
        while self.running:
            try:
                started = time.monotonic_ns()
                camera.grab_images()
                image = camera.retrieve_rgb_image()
                if image.dtype != np.uint8:
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)
                depth = None
                if self.depth_enabled:
                    try:
                        depth = camera.retrieve_depth_map()
                    except (RuntimeError, AttributeError):
                        depth = None
                finished = time.monotonic_ns()
                self.store.publish(name, image, (started + finished) // 2, depth)
                failures = 0
                time.sleep(1.0 / self.fps)
            except (RuntimeError, OSError) as exc:
                failures += 1
                if failures >= self.max_retries:
                    break
                if self.retry_delay_s:
                    time.sleep(self.retry_delay_s)
                _ = exc

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("cannot restart a closed camera capture service")
        if self.running:
            return
        self.running = True
        self.threads = []
        for name, camera in sorted(self.camera_list.items()):
            thread = threading.Thread(target=self._read_thread, args=(name, camera), daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self) -> None:
        if self._closed:
            return
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        for camera in self.camera_list.values():
            try:
                pipeline = getattr(camera, "pipeline", None)
                if pipeline is not None and hasattr(pipeline, "stop"):
                    pipeline.stop()
                elif hasattr(camera, "close"):
                    camera.close()
            except Exception:
                pass
        self._closed = True


class CameraVideoTrack(VideoStreamTrack):
    """aiortc track that consumes a shared latest RGB frame."""

    kind = "video"

    def __init__(
        self,
        provider_or_owner: CameraFrameProvider | Any,
        cam_idx: int,
        *,
        fps: float = 30.0,
    ) -> None:
        super().__init__()
        self.provider = (
            provider_or_owner
            if isinstance(provider_or_owner, CameraFrameProvider)
            else CameraFrameProvider.from_owner(provider_or_owner)
        )
        self.cam_idx = int(cam_idx)
        self.fps = float(fps)

    async def recv(self) -> VideoFrame:
        import asyncio

        pts, time_base = await self.next_timestamp()
        while True:
            prepared = self.provider.process(self.cam_idx)
            if prepared is not None:
                frame_bgr, _rgb = prepared
                video_frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                video_frame.pts = pts
                video_frame.time_base = time_base
                return video_frame
            if not self.provider.is_running():
                raise RuntimeError("camera frame provider stopped before a frame arrived")
            await asyncio.sleep(0.005)
