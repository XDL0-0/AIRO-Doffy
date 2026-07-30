"""Lazy PyAV H.264 encoder with explicit low-latency settings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from ...config.models import VideoStreamingConfig
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import (
    LifecycleError,
    ModelValidationError,
    OptionalDependencyError,
    VideoEncodingError,
)
from ...core.types import EncodedFrame, PixelFormat, ProcessedFrame, VideoCodec


class H264CodecBackend(Protocol):
    """Minimal codec adapter hidden behind the public encoder."""

    codec_name: str

    def encode(self, frame: ProcessedFrame) -> tuple[bytes, bool]:
        """Return one H.264 access unit and keyframe status."""

    def close(self) -> None:
        """Flush and release codec state."""


CodecBuilder = Callable[
    [str, "H264EncoderSettings", int, int, PixelFormat],
    H264CodecBackend,
]


@dataclass(frozen=True, slots=True)
class H264EncoderSettings:
    """Validated encoder knobs that affect latency and rate control."""

    backend: str
    bitrate_bps: int
    gop_frames: int
    target_fps: int

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "nvenc", "software"}:
            raise ModelValidationError("invalid H.264 backend")
        for value, name in (
            (self.bitrate_bps, "bitrate_bps"),
            (self.gop_frames, "gop_frames"),
            (self.target_fps, "target_fps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelValidationError(f"{name} must be an integer >= 1")

    @classmethod
    def from_config(cls, config: VideoStreamingConfig) -> H264EncoderSettings:
        return cls(
            backend=config.encoder_backend,
            bitrate_bps=config.bitrate_bps,
            gop_frames=config.gop_frames,
            target_fps=config.target_fps,
        )

    @property
    def codec_candidates(self) -> tuple[str, ...]:
        if self.backend == "nvenc":
            return ("h264_nvenc",)
        if self.backend == "software":
            return ("libx264",)
        return ("h264_nvenc", "libx264")

    def codec_options(self, codec_name: str) -> dict[str, str]:
        common = {
            "bf": "0",
            "g": str(self.gop_frames),
            "rc-lookahead": "0",
        }
        if codec_name == "h264_nvenc":
            return {
                **common,
                "preset": "p1",
                "tune": "ull",
                "zerolatency": "1",
            }
        if codec_name == "libx264":
            return {
                **common,
                "preset": "ultrafast",
                "tune": "zerolatency",
                "x264-params": (
                    "bframes=0:rc-lookahead=0:"
                    f"keyint={self.gop_frames}:"
                    f"min-keyint={self.gop_frames}:scenecut=0"
                ),
            }
        raise ModelValidationError(f"unsupported H.264 codec: {codec_name}")


def _frame_geometry(frame: ProcessedFrame) -> tuple[int, int]:
    if frame.pixel_format in {PixelFormat.RGB8, PixelFormat.BGR8}:
        height, width, _channels = frame.shape
    elif frame.pixel_format is PixelFormat.GRAY8:
        height, width = frame.shape[:2]
    else:
        raise ModelValidationError("H.264 encoder does not accept depth frames")
    if width % 2 or height % 2:
        raise ModelValidationError(
            "H.264 yuv420p input width and height must both be even"
        )
    return width, height


def _select_codec_backend(
    settings: H264EncoderSettings,
    width: int,
    height: int,
    pixel_format: PixelFormat,
    builder: CodecBuilder,
) -> H264CodecBackend:
    failures: list[str] = []
    for codec_name in settings.codec_candidates:
        try:
            return builder(codec_name, settings, width, height, pixel_format)
        except OptionalDependencyError:
            raise
        except Exception as exc:
            failures.append(f"{codec_name}: {exc}")
    details = "; ".join(failures)
    raise VideoEncodingError(
        f"no configured H.264 encoder could be initialized ({details})"
    )


class _PyAVCodecBackend:
    def __init__(
        self,
        av_module,
        numpy_module,
        codec_name: str,
        settings: H264EncoderSettings,
        width: int,
        height: int,
        pixel_format: PixelFormat,
    ) -> None:
        self.codec_name = codec_name
        self._av = av_module
        self._numpy = numpy_module
        self._pixel_format = pixel_format
        self._time_base = Fraction(1, settings.target_fps)
        codec = av_module.CodecContext.create(codec_name, "w")
        codec.width = width
        codec.height = height
        codec.pix_fmt = "yuv420p"
        codec.time_base = self._time_base
        codec.framerate = Fraction(settings.target_fps, 1)
        codec.bit_rate = settings.bitrate_bps
        codec.gop_size = settings.gop_frames
        codec.max_b_frames = 0
        codec.options = settings.codec_options(codec_name)
        codec.open()
        self._codec = codec

    def encode(self, frame: ProcessedFrame) -> tuple[bytes, bool]:
        formats = {
            PixelFormat.RGB8: "rgb24",
            PixelFormat.BGR8: "bgr24",
            PixelFormat.GRAY8: "gray",
        }
        array = self._numpy.frombuffer(frame.data, dtype=self._numpy.uint8)
        array = array.reshape(frame.shape)
        av_frame = self._av.VideoFrame.from_ndarray(
            array,
            format=formats[self._pixel_format],
        )
        av_frame.pts = frame.sequence
        av_frame.time_base = self._time_base
        packets = tuple(self._codec.encode(av_frame))
        if not packets:
            raise VideoEncodingError(
                "low-latency H.264 encoder produced no packet for an input frame"
            )
        return b"".join(bytes(packet) for packet in packets), any(
            bool(packet.is_keyframe) for packet in packets
        )

    def close(self) -> None:
        try:
            tuple(self._codec.encode(None))
        except Exception:
            pass
        close = getattr(self._codec, "close", None)
        if callable(close):
            close()


def _build_pyav_backend(
    codec_name: str,
    settings: H264EncoderSettings,
    width: int,
    height: int,
    pixel_format: PixelFormat,
) -> H264CodecBackend:
    try:
        import av
        import numpy
    except ImportError as exc:
        raise OptionalDependencyError(
            "H.264 encoding requires the 'video-webrtc' optional dependency: "
            "pip install 'airo-doffy[video-webrtc]'"
        ) from exc
    return _PyAVCodecBackend(
        av,
        numpy,
        codec_name,
        settings,
        width,
        height,
        pixel_format,
    )


class LowLatencyH264Encoder:
    """Encode processed frames without owning capture, queues, or transport."""

    def __init__(
        self,
        config: VideoStreamingConfig,
        *,
        clock: Clock | None = None,
        codec_builder: CodecBuilder = _build_pyav_backend,
    ) -> None:
        self._settings = H264EncoderSettings.from_config(config)
        self._clock = clock or MonotonicClock()
        self._codec_builder = codec_builder
        self._backend: H264CodecBackend | None = None
        self._geometry: tuple[int, int, PixelFormat] | None = None
        self._started = False
        self._closed = False

    @property
    def codec_name(self) -> str | None:
        return None if self._backend is None else self._backend.codec_name

    def start(self) -> None:
        if self._closed:
            raise LifecycleError("cannot start a closed H.264 encoder")
        if self._started:
            raise LifecycleError("H.264 encoder is already started")
        self._started = True

    def encode(self, frame: ProcessedFrame) -> EncodedFrame:
        if self._closed:
            raise LifecycleError("H.264 encoder is closed")
        if not self._started:
            raise LifecycleError("H.264 encoder has not been started")
        if not isinstance(frame, ProcessedFrame):
            raise ModelValidationError("frame must be a ProcessedFrame")
        width, height = _frame_geometry(frame)
        geometry = (width, height, frame.pixel_format)
        if self._geometry is not None and geometry != self._geometry:
            raise ModelValidationError(
                "H.264 stream geometry and pixel format cannot change after first frame"
            )
        if self._backend is None:
            self._backend = _select_codec_backend(
                self._settings,
                width,
                height,
                frame.pixel_format,
                self._codec_builder,
            )
            self._geometry = geometry
        payload, keyframe = self._backend.encode(frame)
        if not payload:
            raise VideoEncodingError("H.264 encoder returned an empty access unit")
        return EncodedFrame(
            sequence=frame.sequence,
            source_timestamp_ns=frame.source_timestamp_ns,
            receive_timestamp_ns=frame.receive_timestamp_ns,
            clock_domain=frame.clock_domain,
            stream_id=frame.stream_id,
            data=payload,
            codec=VideoCodec.H264,
            width=width,
            height=height,
            encoded_timestamp_ns=self._clock.now_ns(),
            keyframe=keyframe,
        )

    def close(self) -> None:
        if self._closed:
            return
        backend = self._backend
        self._backend = None
        self._started = False
        self._closed = True
        if backend is not None:
            backend.close()


def create_h264_encoder(
    config: VideoStreamingConfig,
) -> LowLatencyH264Encoder:
    """Create an unstarted lazy H.264 encoder."""

    return LowLatencyH264Encoder(config)
