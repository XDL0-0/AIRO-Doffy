"""Deterministic on-demand camera source for hardware-free pipelines."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable
from enum import Enum

from ...config.models import CameraConfig
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError
from ...core.types import CameraFrame, ClockDomain, PixelFormat


class CameraMockMode(str, Enum):
    STATIC = "static"
    GENERATED = "generated"
    VIDEO = "video"


def _shape_and_size(
    config: CameraConfig,
    pixel_format: PixelFormat,
) -> tuple[tuple[int, ...], int]:
    width, height = config.resolution
    if pixel_format in {PixelFormat.RGB8, PixelFormat.BGR8}:
        return (height, width, 3), height * width * 3
    if pixel_format is PixelFormat.GRAY8:
        return (height, width), height * width
    return (height, width), height * width * 2


class MockCameraSource:
    """On-demand static, generated, or in-memory video camera source."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        mode: CameraMockMode | str = CameraMockMode.STATIC,
        pixel_format: PixelFormat | str = PixelFormat.RGB8,
        static_frame: bytes | None = None,
        video_frames: Iterable[bytes] = (),
        generator: Callable[[int, int, int, PixelFormat], bytes] | None = None,
        loop_video: bool = True,
        drop_every: int | None = None,
        artificial_delay_s: float = 0.0,
        clock: Clock | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        try:
            selected_mode = CameraMockMode(mode)
            selected_format = PixelFormat(pixel_format)
            delay = float(artificial_delay_s)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("invalid mock camera mode or pixel format") from exc
        if not math.isfinite(delay) or delay < 0:
            raise ModelValidationError(
                "artificial_delay_s must be finite and non-negative"
            )
        if drop_every is not None:
            if (
                isinstance(drop_every, bool)
                or not isinstance(drop_every, int)
                or drop_every < 1
            ):
                raise ModelValidationError("drop_every must be an integer >= 1")
        shape, size = _shape_and_size(config, selected_format)
        static = bytes(size) if static_frame is None else bytes(static_frame)
        frames = tuple(bytes(frame) for frame in video_frames)
        if selected_mode is CameraMockMode.VIDEO and not frames:
            raise ModelValidationError("video mode requires at least one frame")
        for frame in (static, *frames):
            self._validate_frame(
                config,
                selected_format,
                shape,
                frame,
            )
        self._config = config
        self._mode = selected_mode
        self._pixel_format = selected_format
        self._shape = shape
        self._size = size
        self._static_frame = static
        self._video_frames = frames
        self._generator = generator
        self._loop_video = bool(loop_video)
        self._drop_every = drop_every
        self._delay_s = delay
        self._clock = clock or MonotonicClock()
        self._sleep = sleep
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._disconnected = False
        self._sequence = 0
        self._video_index = 0

    @staticmethod
    def _validate_frame(
        config: CameraConfig,
        pixel_format: PixelFormat,
        shape: tuple[int, ...],
        data: bytes,
    ) -> None:
        CameraFrame(
            sequence=0,
            source_timestamp_ns=0,
            stream_id=config.stream_id,
            data=data,
            shape=shape,
            pixel_format=pixel_format,
        )

    @property
    def mode(self) -> CameraMockMode:
        return self._mode

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed mock camera")
            if self._started:
                raise LifecycleError("mock camera is already started")
            self._started = True

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("mock camera is closed")
        if not self._started:
            raise LifecycleError("mock camera has not been started")

    def set_disconnected(self, disconnected: bool) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("mock camera is closed")
            self._disconnected = bool(disconnected)

    def _next_data(self, sequence: int) -> bytes | None:
        if self._mode is CameraMockMode.STATIC:
            return self._static_frame
        if self._mode is CameraMockMode.GENERATED:
            if self._generator is not None:
                width, height = self._config.resolution
                return bytes(
                    self._generator(
                        sequence,
                        width,
                        height,
                        self._pixel_format,
                    )
                )
            return bytes([sequence % 256]) * self._size
        if self._video_index >= len(self._video_frames):
            if not self._loop_video:
                return None
            self._video_index = 0
        data = self._video_frames[self._video_index]
        self._video_index += 1
        return data

    def read_latest(self) -> CameraFrame | None:
        with self._lock:
            self._require_started()
            if self._disconnected:
                return None
            if self._delay_s:
                self._sleep(self._delay_s)
            sequence = self._sequence
            self._sequence += 1
            if self._drop_every is not None and (sequence + 1) % self._drop_every == 0:
                return None
            data = self._next_data(sequence)
            if data is None:
                return None
            self._validate_frame(
                self._config,
                self._pixel_format,
                self._shape,
                data,
            )
            return CameraFrame(
                sequence=sequence,
                source_timestamp_ns=self._clock.now_ns(),
                clock_domain=ClockDomain.MONOTONIC,
                stream_id=self._config.stream_id,
                data=data,
                shape=self._shape,
                pixel_format=self._pixel_format,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True


def create_mock_camera(config: CameraConfig) -> MockCameraSource:
    """Create an unstarted zero-filled static mock camera."""

    return MockCameraSource(config)
