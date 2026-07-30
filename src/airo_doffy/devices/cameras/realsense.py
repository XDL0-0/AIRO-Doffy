"""RealSense discovery and latest-frame acquisition without transport coupling."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from ...config.models import CameraConfig
from ...core.buffers import LatestValueBuffer
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError, OptionalDependencyError
from ...core.types import CameraFrame, ClockDomain, PixelFormat


@dataclass(frozen=True, slots=True)
class RealSenseDevice:
    """Stable discovery result used before selecting one physical camera."""

    name: str
    serial_number: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.name, "name"),
            (self.serial_number, "serial_number"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ModelValidationError(f"{field_name} must be a non-empty string")


def _realsense_sdk():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise OptionalDependencyError(
            "RealSense discovery requires the 'camera-realsense' optional dependency"
        ) from exc
    return rs


def discover_realsense_devices(rs_module=None) -> tuple[RealSenseDevice, ...]:
    """Enumerate RealSense name/serial pairs without opening a stream."""

    rs = rs_module or _realsense_sdk()
    devices = rs.context().query_devices()
    return tuple(
        RealSenseDevice(
            name=device.get_info(rs.camera_info.name),
            serial_number=device.get_info(rs.camera_info.serial_number),
        )
        for device in devices
    )


def _default_camera_factory(config: CameraConfig, serial_number: str):
    try:
        from airo_camera_toolkit.cameras.realsense.realsense import Realsense
    except ImportError as exc:
        raise OptionalDependencyError(
            "RealSense acquisition requires the 'camera-realsense' optional dependency"
        ) from exc
    return Realsense(
        fps=config.fps,
        resolution=config.resolution,
        enable_depth=config.depth_enabled,
        enable_pointcloud=False,
        enable_hole_filling=config.depth_enabled,
        serial_number=serial_number,
    )


def _frame_from_image(
    image: object,
    *,
    stream_id: str,
    sequence: int,
    timestamp_ns: int,
    pixel_format: PixelFormat,
) -> CameraFrame:
    try:
        shape = tuple(int(value) for value in image.shape)
        tobytes = image.tobytes
    except (AttributeError, TypeError, ValueError) as exc:
        raise LifecycleError("camera image must expose shape and tobytes()") from exc
    try:
        data = tobytes()
    except Exception as exc:
        raise LifecycleError("camera image could not be packed") from exc
    return CameraFrame(
        sequence=sequence,
        source_timestamp_ns=timestamp_ns,
        clock_domain=ClockDomain.MONOTONIC,
        stream_id=stream_id,
        data=data,
        shape=shape,
        pixel_format=pixel_format,
    )


class RealSenseCameraSource:
    """Own one RealSense handle and publish immutable color/depth snapshots."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        camera_factory: Callable[[CameraConfig, str], object] | None = None,
        discoverer: Callable[[], tuple[RealSenseDevice, ...]] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config
        self._camera_factory = camera_factory or _default_camera_factory
        self._discoverer = discoverer or discover_realsense_devices
        self._clock = clock or MonotonicClock()
        self._color = LatestValueBuffer[CameraFrame]()
        self._depth = LatestValueBuffer[CameraFrame]()
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._lock = threading.RLock()
        self._camera: object | None = None
        self._thread: threading.Thread | None = None
        self._selected_serial: str | None = None
        self._health_error: Exception | None = None
        self._started = False
        self._closed = False
        self._sequence = 0

    @property
    def selected_serial(self) -> str | None:
        with self._lock:
            return self._selected_serial

    @property
    def health_error(self) -> Exception | None:
        with self._lock:
            return self._health_error

    def _select_serial(self) -> str:
        configured = self._config.serial_number
        if configured is not None:
            return configured
        devices = self._discoverer()
        if not devices:
            raise LifecycleError("no RealSense camera was discovered")
        return devices[0].serial_number

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed RealSense source")
            if self._started:
                raise LifecycleError("RealSense source is already started")
        serial = self._select_serial()
        camera = self._camera_factory(self._config, serial)
        if not callable(getattr(camera, "get_rgb_image", None)):
            self._close_camera(camera)
            raise LifecycleError("RealSense adapter does not expose get_rgb_image()")
        thread = threading.Thread(
            target=self._worker,
            name=f"airo-doffy-{self._config.stream_id}",
            daemon=True,
        )
        with self._lock:
            self._selected_serial = serial
            self._camera = camera
            self._thread = thread
            self._started = True
            self._health_error = None
            self._stop_event.clear()
            self._first_frame_event.clear()
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._started = False
                self._thread = None
                self._camera = None
            self._close_camera(camera)
            raise

    def _read_depth(self, camera: object):
        if not self._config.depth_enabled:
            return None
        method = getattr(camera, "_retrieve_depth_map", None)
        if not callable(method):
            return None
        try:
            return method()
        except (RuntimeError, AttributeError):
            return None

    def _worker(self) -> None:
        consecutive_errors = 0
        interval_s = 1.0 / self._config.capture_rate_hz
        while not self._stop_event.is_set():
            camera = self._camera
            if camera is None:
                break
            try:
                image = camera.get_rgb_image()
                depth = self._read_depth(camera)
                timestamp_ns = self._clock.now_ns()
                sequence = self._sequence
                color_frame = _frame_from_image(
                    image,
                    stream_id=self._config.stream_id,
                    sequence=sequence,
                    timestamp_ns=timestamp_ns,
                    pixel_format=PixelFormat.RGB8,
                )
                depth_frame = (
                    None
                    if depth is None
                    else _frame_from_image(
                        depth,
                        stream_id=f"{self._config.stream_id}_depth",
                        sequence=sequence,
                        timestamp_ns=timestamp_ns,
                        pixel_format=PixelFormat.DEPTH_U16,
                    )
                )
                self._color.publish(color_frame)
                if depth_frame is not None:
                    self._depth.publish(depth_frame)
                self._sequence += 1
                consecutive_errors = 0
                with self._lock:
                    self._health_error = None
                self._first_frame_event.set()
                self._stop_event.wait(interval_s)
            except RuntimeError as exc:
                consecutive_errors += 1
                with self._lock:
                    self._health_error = exc
                if consecutive_errors >= self._config.max_consecutive_errors:
                    break
                self._stop_event.wait(self._config.retry_delay_s)
            except Exception as exc:
                with self._lock:
                    self._health_error = exc
                break
        self._first_frame_event.set()

    def wait_for_first_frame(self, timeout_s: float | None = None) -> bool:
        if timeout_s is not None and timeout_s < 0:
            raise ModelValidationError("timeout_s must be non-negative")
        self._first_frame_event.wait(timeout_s)
        return self._color.read() is not None

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("RealSense source is closed")
        if not self._started:
            raise LifecycleError("RealSense source has not been started")

    def read_latest(self) -> CameraFrame | None:
        with self._lock:
            self._require_started()
            return self._color.read()

    def read_latest_depth(self) -> CameraFrame | None:
        with self._lock:
            self._require_started()
            return self._depth.read()

    @staticmethod
    def _close_camera(camera: object) -> None:
        pipeline = getattr(camera, "pipeline", None)
        if pipeline is not None and callable(getattr(pipeline, "stop", None)):
            pipeline.stop()
            return
        close = getattr(camera, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._started:
                self._closed = True
                self._color.close()
                self._depth.close()
                return
            camera = self._camera
            thread = self._thread
            self._stop_event.set()
        cleanup_error: Exception | None = None
        if camera is not None:
            try:
                self._close_camera(camera)
            except Exception as exc:
                cleanup_error = exc
        if thread is not None:
            thread.join(timeout=max(2.0, self._config.retry_delay_s + 0.5))
            if thread.is_alive():
                raise LifecycleError(
                    "RealSense worker did not stop; source was not invalidated"
                )
        if cleanup_error is not None:
            raise LifecycleError("RealSense SDK cleanup failed") from cleanup_error
        with self._lock:
            self._started = False
            self._closed = True
            self._camera = None
            self._thread = None
            self._color.close()
            self._depth.close()


def create_realsense_camera(config: CameraConfig) -> RealSenseCameraSource:
    """Create an unstarted RealSense source without importing its SDK."""

    return RealSenseCameraSource(config)
