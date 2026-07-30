"""Deprecated-compatible JPEG encoding and UDP chunk transport."""

from __future__ import annotations

import socket
import struct
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ...config.models import NetworkConfig, VideoStreamingConfig
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import (
    LifecycleError,
    ModelValidationError,
    OptionalDependencyError,
    VideoEncodingError,
)
from ...core.types import EncodedFrame, PixelFormat, ProcessedFrame, VideoCodec

LEGACY_JPEG_HEADER = struct.Struct("!IHHI")
LEGACY_JPEG_HEADER_SIZE = LEGACY_JPEG_HEADER.size
MAX_UDP_PAYLOAD = 65_507
MAX_CHUNK_PAYLOAD = MAX_UDP_PAYLOAD - LEGACY_JPEG_HEADER_SIZE


class DatagramSocket(Protocol):
    """Narrow outgoing datagram socket used by the legacy transport."""

    def sendto(self, data: bytes, target: tuple[str, int]) -> int:
        """Send one datagram."""

    def close(self) -> None:
        """Release the socket."""


JpegEncoder = Callable[[ProcessedFrame, int], bytes]
SocketFactory = Callable[[], DatagramSocket]


def packetize_legacy_jpeg(
    jpeg_data: bytes | bytearray | memoryview,
    *,
    frame_id: int,
    chunk_size: int = 60_000,
) -> tuple[bytes, ...]:
    """Return packets matching the existing Unity ``UdpSocketMultiHD`` format."""

    try:
        payload = bytes(jpeg_data)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("jpeg_data must support the bytes protocol") from exc
    if not payload:
        raise ModelValidationError("legacy JPEG payload must not be empty")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ModelValidationError("frame_id must be a non-negative integer")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or not 1 <= chunk_size <= MAX_CHUNK_PAYLOAD
    ):
        raise ModelValidationError(
            f"chunk_size must be an integer within [1, {MAX_CHUNK_PAYLOAD}]"
        )
    if len(payload) > 0xFFFFFFFF:
        raise ModelValidationError("legacy JPEG payload must fit uint32 length")
    chunk_count = (len(payload) + chunk_size - 1) // chunk_size
    if chunk_count > 0xFFFF:
        raise ModelValidationError("legacy JPEG chunk count must fit uint16")
    wire_frame_id = frame_id & 0xFFFFFFFF
    return tuple(
        LEGACY_JPEG_HEADER.pack(
            wire_frame_id,
            index,
            chunk_count,
            len(payload),
        )
        + payload[index * chunk_size : (index + 1) * chunk_size]
        for index in range(chunk_count)
    )


def _opencv_encode_jpeg(frame: ProcessedFrame, quality: int) -> bytes:
    try:
        import cv2
        import numpy
    except ImportError as exc:
        raise OptionalDependencyError(
            "legacy JPEG encoding requires OpenCV and NumPy: "
            "pip install 'airo-doffy[video-jpeg]'"
        ) from exc
    if frame.pixel_format is PixelFormat.DEPTH_U16:
        raise ModelValidationError("legacy JPEG encoder does not accept depth frames")
    array = numpy.frombuffer(frame.data, dtype=numpy.uint8).reshape(frame.shape)
    if frame.pixel_format is PixelFormat.RGB8:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        array,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise VideoEncodingError("OpenCV failed to encode a legacy JPEG frame")
    return bytes(encoded)


class LegacyJpegEncoder:
    """Encode one processed frame while preserving legacy quality behavior."""

    def __init__(
        self,
        config: VideoStreamingConfig,
        *,
        clock: Clock | None = None,
        jpeg_encoder: JpegEncoder = _opencv_encode_jpeg,
    ) -> None:
        self._quality = config.jpeg_quality
        self._clock = clock or MonotonicClock()
        self._jpeg_encoder = jpeg_encoder
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise LifecycleError("cannot start a closed legacy JPEG encoder")
        if self._started:
            raise LifecycleError("legacy JPEG encoder is already started")
        self._started = True

    def encode(self, frame: ProcessedFrame) -> EncodedFrame:
        if self._closed:
            raise LifecycleError("legacy JPEG encoder is closed")
        if not self._started:
            raise LifecycleError("legacy JPEG encoder has not been started")
        if not isinstance(frame, ProcessedFrame):
            raise ModelValidationError("frame must be a ProcessedFrame")
        if frame.pixel_format in {PixelFormat.RGB8, PixelFormat.BGR8}:
            height, width, _channels = frame.shape
        elif frame.pixel_format is PixelFormat.GRAY8:
            height, width = frame.shape[:2]
        else:
            raise ModelValidationError("legacy JPEG encoder does not accept depth frames")
        payload = self._jpeg_encoder(frame, self._quality)
        if not payload:
            raise VideoEncodingError("legacy JPEG encoder returned an empty payload")
        return EncodedFrame(
            sequence=frame.sequence,
            source_timestamp_ns=frame.source_timestamp_ns,
            receive_timestamp_ns=frame.receive_timestamp_ns,
            clock_domain=frame.clock_domain,
            stream_id=frame.stream_id,
            data=payload,
            codec=VideoCodec.JPEG,
            width=width,
            height=height,
            encoded_timestamp_ns=self._clock.now_ns(),
            keyframe=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._started = False
        self._closed = True


@dataclass(frozen=True, slots=True)
class LegacyJpegUdpMetrics:
    """Snapshot of frame, packet, byte, and error counts."""

    frames_sent: int
    packets_sent: int
    bytes_sent: int
    errors: int


def _udp_socket() -> DatagramSocket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class LegacyJpegUdpTransport:
    """Send already encoded JPEG frames using the frozen legacy wire format."""

    def __init__(
        self,
        target_host: str,
        target_port: int,
        *,
        chunk_size: int = 60_000,
        socket_factory: SocketFactory = _udp_socket,
    ) -> None:
        warnings.warn(
            "LegacyJpegUdpTransport is deprecated; use WebRTC H.264 for "
            "production after deployment validation",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(target_host, str) or not target_host.strip():
            raise ModelValidationError("target_host must be a non-empty string")
        if (
            isinstance(target_port, bool)
            or not isinstance(target_port, int)
            or not 1 <= target_port <= 65535
        ):
            raise ModelValidationError("target_port must be within [1, 65535]")
        packetize_legacy_jpeg(b"x", frame_id=0, chunk_size=chunk_size)
        self._target = (target_host, target_port)
        self._chunk_size = chunk_size
        self._socket_factory = socket_factory
        self._socket: DatagramSocket | None = None
        self._started = False
        self._closed = False
        self._frames_sent = 0
        self._packets_sent = 0
        self._bytes_sent = 0
        self._errors = 0

    @property
    def metrics(self) -> LegacyJpegUdpMetrics:
        return LegacyJpegUdpMetrics(
            frames_sent=self._frames_sent,
            packets_sent=self._packets_sent,
            bytes_sent=self._bytes_sent,
            errors=self._errors,
        )

    def start(self) -> None:
        if self._closed:
            raise LifecycleError("cannot start a closed legacy JPEG transport")
        if self._started:
            raise LifecycleError("legacy JPEG transport is already started")
        self._socket = self._socket_factory()
        self._started = True

    def send(self, frame: EncodedFrame) -> None:
        if self._closed:
            raise LifecycleError("legacy JPEG transport is closed")
        if not self._started or self._socket is None:
            raise LifecycleError("legacy JPEG transport has not been started")
        if not isinstance(frame, EncodedFrame):
            raise ModelValidationError("frame must be an EncodedFrame")
        if frame.codec is not VideoCodec.JPEG:
            raise ModelValidationError("legacy JPEG transport requires JPEG frames")
        packets = packetize_legacy_jpeg(
            frame.data,
            frame_id=frame.sequence,
            chunk_size=self._chunk_size,
        )
        try:
            for packet in packets:
                sent = self._socket.sendto(packet, self._target)
                if sent != len(packet):
                    raise OSError(
                        f"partial UDP datagram send: {sent}/{len(packet)} bytes"
                    )
        except OSError:
            self._errors += 1
            raise
        self._frames_sent += 1
        self._packets_sent += len(packets)
        self._bytes_sent += sum(len(packet) for packet in packets)

    def close(self) -> None:
        if self._closed:
            return
        udp_socket = self._socket
        self._socket = None
        self._started = False
        self._closed = True
        if udp_socket is not None:
            udp_socket.close()


def create_legacy_jpeg_encoder(
    config: VideoStreamingConfig,
) -> LegacyJpegEncoder:
    """Create an unstarted legacy-compatible JPEG encoder."""

    return LegacyJpegEncoder(config)


def create_legacy_jpeg_udp(
    config: VideoStreamingConfig,
    network: NetworkConfig,
) -> LegacyJpegUdpTransport:
    """Create camera-0's unstarted outgoing compatibility transport."""

    if network.vr_ip is None:
        raise ModelValidationError("legacy JPEG UDP transport requires network.vr_ip")
    return LegacyJpegUdpTransport(
        network.vr_ip,
        network.legacy_base_port,
        chunk_size=config.legacy_chunk_size,
    )
