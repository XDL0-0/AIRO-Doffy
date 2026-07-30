"""Golden tests for the deprecated-compatible JPEG UDP path."""

from __future__ import annotations

import struct
import unittest
import warnings

from airo_doffy.config import (
    EncoderFactory,
    NetworkConfig,
    VideoStreamingConfig,
    VideoTransportFactory,
)
from airo_doffy.core import (
    ClockDomain,
    EncodedFrame,
    LifecycleError,
    ModelValidationError,
    VideoCodec,
    VideoEncodingError,
)
from airo_doffy.streaming.video import (
    LEGACY_JPEG_HEADER,
    LegacyJpegEncoder,
    LegacyJpegUdpTransport,
    VideoEncoder,
    VideoTransport,
    packetize_legacy_jpeg,
)

from tests.unit.test_h264_encoder import frame


class _Clock:
    def now_ns(self) -> int:
        return 50


class _Socket:
    def __init__(self) -> None:
        self.sent = []
        self.closed = False

    def sendto(self, data: bytes, target: tuple[str, int]) -> int:
        self.sent.append((data, target))
        return len(data)

    def close(self) -> None:
        self.closed = True


def encoded(
    data: bytes = b"jpeg",
    *,
    sequence: int = 0,
    codec: VideoCodec = VideoCodec.JPEG,
) -> EncodedFrame:
    return EncodedFrame(
        sequence=sequence,
        source_timestamp_ns=1,
        clock_domain=ClockDomain.MONOTONIC,
        stream_id="camera_0",
        data=data,
        codec=codec,
        width=2,
        height=2,
        encoded_timestamp_ns=2,
        keyframe=True,
    )


class LegacyJpegUdpTest(unittest.TestCase):
    def test_single_chunk_matches_big_endian_golden(self) -> None:
        packets = packetize_legacy_jpeg(b"abc", frame_id=5)
        self.assertEqual(
            packets,
            (b"\x00\x00\x00\x05\x00\x00\x00\x01\x00\x00\x00\x03abc",),
        )
        self.assertEqual(LEGACY_JPEG_HEADER.format, "!IHHI")

    def test_multi_chunk_reconstructs_and_frame_id_wraps(self) -> None:
        payload = b"0123456789"
        packets = packetize_legacy_jpeg(
            payload,
            frame_id=2**32 + 7,
            chunk_size=4,
        )
        headers = [LEGACY_JPEG_HEADER.unpack(packet[:12]) for packet in packets]
        self.assertEqual(
            headers,
            [(7, 0, 3, 10), (7, 1, 3, 10), (7, 2, 3, 10)],
        )
        self.assertEqual(b"".join(packet[12:] for packet in packets), payload)

    def test_packet_validation(self) -> None:
        with self.assertRaises(ModelValidationError):
            packetize_legacy_jpeg(b"", frame_id=0)
        with self.assertRaises(ModelValidationError):
            packetize_legacy_jpeg(b"x", frame_id=-1)
        with self.assertRaises(ModelValidationError):
            packetize_legacy_jpeg(b"x", frame_id=0, chunk_size=65_496)
        with self.assertRaises(ModelValidationError):
            packetize_legacy_jpeg(b"x" * 65_536, frame_id=0, chunk_size=1)

    def test_encoder_preserves_metadata_quality_and_lifecycle(self) -> None:
        qualities = []

        def jpeg_encoder(_frame, quality):
            qualities.append(quality)
            return b"jpeg-bytes"

        encoder = LegacyJpegEncoder(
            VideoStreamingConfig(jpeg_quality=73),
            clock=_Clock(),
            jpeg_encoder=jpeg_encoder,
        )
        self.assertIsInstance(encoder, VideoEncoder)
        with self.assertRaises(LifecycleError):
            encoder.encode(frame())
        encoder.start()
        result = encoder.encode(frame())
        self.assertEqual(result.codec, VideoCodec.JPEG)
        self.assertEqual(result.data, b"jpeg-bytes")
        self.assertEqual(result.encoded_timestamp_ns, 50)
        self.assertEqual(qualities, [73])
        encoder.close()
        with self.assertRaises(LifecycleError):
            encoder.encode(frame())

        empty = LegacyJpegEncoder(
            VideoStreamingConfig(),
            jpeg_encoder=lambda *_args: b"",
        )
        empty.start()
        with self.assertRaises(VideoEncodingError):
            empty.encode(frame())
        empty.close()

    def test_transport_target_order_metrics_and_codec_rejection(self) -> None:
        udp_socket = _Socket()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transport = LegacyJpegUdpTransport(
                "192.0.2.10",
                8000,
                chunk_size=2,
                socket_factory=lambda: udp_socket,
            )
        self.assertEqual(caught[0].category, DeprecationWarning)
        self.assertIsInstance(transport, VideoTransport)
        with self.assertRaises(LifecycleError):
            transport.send(encoded())
        transport.start()
        transport.send(encoded(b"abcde", sequence=9))
        self.assertEqual(
            [target for _packet, target in udp_socket.sent],
            [("192.0.2.10", 8000)] * 3,
        )
        self.assertEqual(
            [struct.unpack("!IHHI", packet[:12]) for packet, _ in udp_socket.sent],
            [(9, 0, 3, 5), (9, 1, 3, 5), (9, 2, 3, 5)],
        )
        self.assertEqual(transport.metrics.frames_sent, 1)
        self.assertEqual(transport.metrics.packets_sent, 3)
        with self.assertRaises(ModelValidationError):
            transport.send(encoded(codec=VideoCodec.H264))
        transport.close()
        transport.close()
        self.assertTrue(udp_socket.closed)

    def test_factories_are_lazy_and_network_is_explicit(self) -> None:
        encoder = EncoderFactory(
            target=(
                "airo_doffy.streaming.video.legacy_jpeg_udp:"
                "create_legacy_jpeg_encoder"
            )
        ).create(VideoStreamingConfig())
        self.assertIsInstance(encoder, LegacyJpegEncoder)
        encoder.close()
        factory = VideoTransportFactory(
            target=(
                "airo_doffy.streaming.video.legacy_jpeg_udp:"
                "create_legacy_jpeg_udp"
            )
        )
        with self.assertRaises(ModelValidationError):
            factory.create(VideoStreamingConfig(), NetworkConfig())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            transport = factory.create(
                VideoStreamingConfig(),
                NetworkConfig(vr_ip="192.0.2.10"),
            )
        self.assertIsInstance(transport, LegacyJpegUdpTransport)
        transport.close()


if __name__ == "__main__":
    unittest.main()
