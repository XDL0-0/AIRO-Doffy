"""RFC 6184 packetization, jitter, and transport tests."""

from __future__ import annotations

import unittest

from airo_doffy.config import NetworkConfig, VideoStreamingConfig, VideoTransportFactory
from airo_doffy.core import ClockDomain, EncodedFrame, LifecycleError, ModelValidationError
from airo_doffy.core.types import VideoCodec
from airo_doffy.streaming.video import (
    RTP_HEADER,
    RtpH264JitterBuffer,
    RtpH264UdpTransport,
    VideoTransport,
    packetize_h264_rtp,
    parse_rtp_packet,
    split_h264_access_unit,
)


class _Clock:
    def __init__(self, value: int = 100) -> None:
        self.value = value

    def now_ns(self) -> int:
        return self.value


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
    sequence: int,
    *,
    source_timestamp_ns: int = 1_000_000_000,
    encoded_timestamp_ns: int = 100,
    stream_id: str = "camera_0",
) -> EncodedFrame:
    return EncodedFrame(
        sequence=sequence,
        source_timestamp_ns=source_timestamp_ns,
        clock_domain=ClockDomain.MONOTONIC,
        stream_id=stream_id,
        data=b"\x00\x00\x00\x01\x65payload",
        codec=VideoCodec.H264,
        width=2,
        height=2,
        encoded_timestamp_ns=encoded_timestamp_ns,
        keyframe=True,
    )


class RtpH264UdpTest(unittest.TestCase):
    def test_annex_b_avcc_and_raw_nal_splitting(self) -> None:
        annex_b = b"\x00\x00\x00\x01\x67abc\x00\x00\x01\x65def"
        self.assertEqual(
            split_h264_access_unit(annex_b),
            (b"\x67abc", b"\x65def"),
        )
        avcc = b"\x00\x00\x00\x04\x67abc\x00\x00\x00\x04\x65def"
        self.assertEqual(split_h264_access_unit(avcc), (b"\x67abc", b"\x65def"))
        self.assertEqual(split_h264_access_unit(b"\x65raw"), (b"\x65raw",))
        with self.assertRaises(ModelValidationError):
            split_h264_access_unit(b"")

    def test_single_nal_headers_sequence_wrap_and_marker(self) -> None:
        packets = packetize_h264_rtp(
            b"\x00\x00\x00\x01\x67a\x00\x00\x00\x01\x65b",
            sequence_start=0xFFFF,
            timestamp=123,
            ssrc=456,
            mtu=256,
        )
        parsed = tuple(parse_rtp_packet(packet) for packet in packets)
        self.assertEqual([item.sequence for item in parsed], [0xFFFF, 0])
        self.assertEqual([item.timestamp for item in parsed], [123, 123])
        self.assertEqual([item.ssrc for item in parsed], [456, 456])
        self.assertEqual([item.marker for item in parsed], [False, True])
        self.assertEqual([item.payload for item in parsed], [b"\x67a", b"\x65b"])

    def test_fu_a_fragmentation_and_out_of_order_jitter_reassembly(self) -> None:
        nal = b"\x65" + bytes(range(256)) * 3
        packets = packetize_h264_rtp(
            b"\x00\x00\x00\x01" + nal,
            sequence_start=7,
            timestamp=9,
            ssrc=11,
            mtu=256,
        )
        self.assertGreater(len(packets), 2)
        parsed = tuple(parse_rtp_packet(packet) for packet in packets)
        self.assertTrue(parsed[0].payload[1] & 0x80)
        self.assertTrue(parsed[-1].payload[1] & 0x40)
        self.assertTrue(parsed[-1].marker)
        jitter = RtpH264JitterBuffer(max_frames=2)
        order = (1, 0, *range(2, len(packets)))
        for index in order:
            self.assertTrue(jitter.push(packets[index]))
        self.assertEqual(jitter.pop_ready(), b"\x00\x00\x00\x01" + nal)
        self.assertFalse(jitter.push(packets[0]))
        self.assertEqual(jitter.metrics.dropped_late, 1)

    def test_jitter_capacity_duplicate_and_malformed(self) -> None:
        first = packetize_h264_rtp(
            b"\x65" + b"x" * 400,
            sequence_start=1,
            timestamp=1,
            ssrc=1,
            mtu=256,
        )
        second = packetize_h264_rtp(
            b"\x65" + b"y" * 400,
            sequence_start=10,
            timestamp=2,
            ssrc=1,
            mtu=256,
        )
        jitter = RtpH264JitterBuffer(max_frames=1)
        self.assertTrue(jitter.push(first[0]))
        self.assertFalse(jitter.push(first[0]))
        self.assertTrue(jitter.push(second[0]))
        self.assertEqual(jitter.metrics.duplicate_packets, 1)
        self.assertEqual(jitter.metrics.dropped_jitter, 1)
        self.assertFalse(jitter.push(b"short"))
        self.assertEqual(jitter.metrics.malformed_packets, 1)

    def test_transport_stale_late_timestamp_target_and_lifecycle(self) -> None:
        udp_socket = _Socket()
        clock = _Clock()
        transport = RtpH264UdpTransport(
            "192.0.2.20",
            5004,
            VideoStreamingConfig(max_frame_age_s=0.25),
            ssrc=3,
            initial_sequence=0xFFFF,
            initial_timestamp=10,
            clock=clock,
            socket_factory=lambda: udp_socket,
        )
        self.assertIsInstance(transport, VideoTransport)
        with self.assertRaises(LifecycleError):
            transport.send(encoded(0))
        transport.start()
        transport.send(encoded(0))
        transport.send(encoded(0))
        clock.value = 300_000_101
        transport.send(encoded(1, encoded_timestamp_ns=100))
        clock.value = 100
        transport.send(encoded(2, source_timestamp_ns=2_000_000_000))
        parsed = [parse_rtp_packet(packet) for packet, _ in udp_socket.sent]
        self.assertEqual([item.sequence for item in parsed], [0xFFFF, 0])
        self.assertEqual([item.timestamp for item in parsed], [10, 90_010])
        self.assertEqual(
            [target for _packet, target in udp_socket.sent],
            [("192.0.2.20", 5004)] * 2,
        )
        self.assertEqual(transport.metrics.frames_sent, 2)
        self.assertEqual(transport.metrics.dropped_stale, 1)
        self.assertEqual(transport.metrics.dropped_late, 1)
        with self.assertRaises(ModelValidationError):
            transport.send(encoded(3, stream_id="camera_1"))
        transport.close()
        transport.close()
        self.assertTrue(udp_socket.closed)

    def test_factory_uses_explicit_rtp_port(self) -> None:
        factory = VideoTransportFactory(
            target=(
                "airo_doffy.streaming.video.rtp_h264_udp:"
                "create_rtp_h264_udp"
            )
        )
        with self.assertRaises(ModelValidationError):
            factory.create(VideoStreamingConfig(), NetworkConfig())
        transport = factory.create(
            VideoStreamingConfig(),
            NetworkConfig(vr_ip="192.0.2.20", video_rtp_port=6000),
        )
        self.assertIsInstance(transport, RtpH264UdpTransport)
        transport.close()


if __name__ == "__main__":
    unittest.main()
