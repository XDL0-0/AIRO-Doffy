"""Local-only multi-RealSense capture for recording and visualization."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
import pyrealsense2 as rs
from airo_camera_toolkit.cameras.realsense.realsense import Realsense

import utils
from config import Config


MAX_RETRIES = 10
WARMUP_TIMEOUT = 20.0


class RealSenseCameraManager:
    """Capture every connected RealSense without creating network resources."""

    def __init__(self, config: Config | None = None) -> None:
        cfg = Config() if config is None else config
        self.realsense_resolution = cfg.REALSENSE_RESOLUTION
        self.realsense_fps = cfg.REALSENSE_FPS
        self.depth_mode = cfg.DEPTH_INFO_ENABLE

        self._lock = threading.Lock()
        self.running = False
        self._closed = False
        self.threads: List[threading.Thread] = []
        self.camera_images: Dict[str, np.ndarray] = {}
        self.camera_image_timestamps_ns: Dict[str, int] = {}
        self.depth_images: Dict[str, np.ndarray] = {}
        self.depth_timestamps_ns: Dict[str, int] = {}
        self._sync_buffer_size = int(cfg.SENSOR_SYNC_BUFFER_SIZE)
        self.camera_frame_buffers: Dict[
            str, deque[tuple[int, np.ndarray, np.ndarray | None]]
        ] = {}

        self.camera_num, self.camera_series_num = self._detect_cameras()
        self.camera_list: Dict[str, Realsense] = {}
        try:
            for idx, serial in enumerate(self.camera_series_num):
                self._create_camera(idx, serial)
        except Exception:
            self.close()
            raise

    def _detect_cameras(self) -> Tuple[int, List[str]]:
        devices = rs.context().query_devices()
        serials: List[str] = []
        if len(devices) == 0:
            utils.logger.warning("No RealSense cameras connected.")
        else:
            for idx, device in enumerate(devices):
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                utils.logger.info(f"Camera {idx}: {name}, serial={serial}")
                serials.append(serial)
        return len(serials), serials

    def _create_camera(self, idx: int, serial: str) -> None:
        name = f"camera_{idx}"
        camera = Realsense(
            fps=self.realsense_fps,
            resolution=self.realsense_resolution,
            enable_depth=self.depth_mode,
            enable_pointcloud=False,
            enable_hole_filling=self.depth_mode,
            serial_number=serial,
        )
        self.camera_list[name] = camera
        self.camera_frame_buffers[name] = deque(maxlen=self._sync_buffer_size)
        width, height = self.realsense_resolution
        utils.logger.info(
            f"{name}: serial={serial}, fps={self.realsense_fps}, "
            f"res={width}x{height}"
        )

    def _camera_read_thread(self, camera: Realsense, idx: int) -> None:
        utils.logger.info(f"Local camera thread {idx} started")
        consecutive_errors = 0
        while self.running:
            try:
                capture_start_ns = time.monotonic_ns()
                camera.grab_images()
                image = camera.retrieve_rgb_image()
                if image.dtype != np.uint8:
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)

                depth = None
                if self.depth_mode:
                    try:
                        depth = camera.retrieve_depth_map()
                    except (RuntimeError, AttributeError):
                        pass

                capture_end_ns = time.monotonic_ns()
                # The wrapper does not expose the RealSense hardware timestamp.
                # Bracket the blocking acquisition instead of stamping after all
                # retrieval work, which otherwise biases every frame late.
                capture_timestamp_ns = (capture_start_ns + capture_end_ns) // 2
                with self._lock:
                    name = f"camera_{idx}"
                    self.camera_images[name] = image
                    self.camera_image_timestamps_ns[name] = capture_timestamp_ns
                    if depth is not None:
                        self.depth_images[name] = depth
                        self.depth_timestamps_ns[name] = capture_timestamp_ns
                    self.camera_frame_buffers[name].append(
                        (capture_timestamp_ns, image, depth)
                    )
                consecutive_errors = 0
                time.sleep(1 / self.realsense_fps)
            except RuntimeError as exc:
                consecutive_errors += 1
                if consecutive_errors >= MAX_RETRIES:
                    utils.logger.error(
                        f"Camera {idx}: {MAX_RETRIES} consecutive errors, stopping: {exc}"
                    )
                    break
                utils.logger.warning(
                    f"Camera {idx}: capture error ({consecutive_errors}/{MAX_RETRIES}), "
                    f"retrying in 1s: {exc}"
                )
                time.sleep(1.0)

    def snapshot_nearest(
        self,
        reference_timestamp_ns: int,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, int],
        dict[str, np.ndarray] | None,
    ]:
        """Copy the buffered frame nearest to a monotonic reference time."""
        reference_timestamp_ns = int(reference_timestamp_ns)
        images: dict[str, np.ndarray] = {}
        timestamps: dict[str, int] = {}
        depth_images: dict[str, np.ndarray] | None = {} if self.depth_mode else None
        with self._lock:
            for name in sorted(self.camera_list):
                history = self.camera_frame_buffers.get(name)
                if history:
                    timestamp_ns, image, depth = min(
                        history,
                        key=lambda frame: abs(frame[0] - reference_timestamp_ns),
                    )
                elif name in self.camera_images:
                    timestamp_ns = int(self.camera_image_timestamps_ns.get(name, 0))
                    image = self.camera_images[name]
                    depth = self.depth_images.get(name)
                else:
                    continue
                images[name] = np.asarray(image).copy()
                timestamps[name] = int(timestamp_ns)
                if depth_images is not None and depth is not None:
                    depth_images[name] = np.asarray(depth).copy()
        return images, timestamps, depth_images

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot restart a closed RealSense camera manager.")
        if self.running:
            return

        self.running = True
        self.threads = []
        for idx in range(self.camera_num):
            thread = threading.Thread(
                target=self._camera_read_thread,
                args=(self.camera_list[f"camera_{idx}"], idx),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

        utils.logger.info("Waiting for local cameras to warm up...")
        deadline = time.monotonic() + WARMUP_TIMEOUT
        while True:
            with self._lock:
                ready_count = len(self.camera_images)
            if ready_count >= self.camera_num:
                break
            if time.monotonic() > deadline:
                utils.logger.error(
                    f"Camera warmup timed out: {ready_count}/{self.camera_num} ready."
                )
                break
            time.sleep(0.1)
        utils.logger.info(f"Local cameras ready: {list(self.camera_images)}")

    def close(self) -> None:
        if self._closed:
            return
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

        for name, camera in self.camera_list.items():
            try:
                if hasattr(camera, "pipeline"):
                    camera.pipeline.stop()
                elif hasattr(camera, "close"):
                    camera.close()
            except Exception as exc:
                utils.logger.warning(f"Error stopping {name}: {exc}")
        self._closed = True
        utils.logger.info("Local RealSense camera resources released.")
