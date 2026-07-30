"""Hardware-free tests for WebRTC signaling and latest-only transport state."""

from __future__ import annotations

import threading
import unittest

from airo_doffy.config import NetworkConfig, VideoStreamingConfig, VideoTransportFactory
from airo_doffy.core import ClockDomain, EncodedFrame, LifecycleError, ModelValidationError
from airo_doffy.core.types import VideoCodec
from airo_doffy.streaming.video import (
    VideoTransport,
    WebRTCVideoTransport,
    parse_signaling_envelope,
    signaling_envelope,
)


class _Runtime:
    def __init__(self, _owner, host: str, port: int) -> None:
        self.ready = threading.Event()
        self.error = None
        self.host = host
        self.port = port
        self.stopped = threading.Event()

    def run(self) -> None:
        self.ready.set()
        self.stopped.wait(1.0)

    def stop(self) -> None:
        self.stopped.set()


class _FailedRuntime(_Runtime):
    def run(self) -> None:
        self.error = RuntimeError("injected signaling failure")
        self.ready.set()


class _TimeoutRuntime(_Runtime):
    def run(self) -> None:
        self.stopped.wait(1.0)


def encoded(
    sequence: int,
    *,
    stream_id: str = "camera_0",
    codec: VideoCodec = VideoCodec.H264,
) -> EncodedFrame:
    return EncodedFrame(
        sequence=sequence,
        source_timestamp_ns=sequence,
        clock_domain=ClockDomain.MONOTONIC,
        stream_id=stream_id,
        data=b"\x00\x00\x00\x01\x65frame",
        codec=codec,
        width=2,
        height=2,
        encoded_timestamp_ns=sequence,
        keyframe=True,
    )


class WebRTCTransportTest(unittest.TestCase):
    def test_signaling_envelope_compatibility_and_validation(self) -> None:
        raw = signaling_envelope("hello", "session-1")
        self.assertEqual(
            raw,
            {
                "type": "hello",
                "session_id": "session-1",
                "payload": {},
            },
        )
        parsed = parse_signaling_envelope(
            '{"type":"offer","session_id":"s","payload":{"sdp":"v=0"}}'
        )
        self.assertEqual(parsed.message_type, "offer")
        self.assertEqual(parsed.session_id, "s")
        self.assertEqual(parsed.payload, {"sdp": "v=0"})
        for invalid in (
            "[]",
            "{bad",
            {"session_id": "s"},
            {"type": "hello", "payload": []},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ModelValidationError):
                    parse_signaling_envelope(invalid)

    def test_latest_only_stale_rejection_lifecycle_and_metrics(self) -> None:
        runtimes = []

        def factory(*args):
            runtime = _Runtime(*args)
            runtimes.append(runtime)
            return runtime

        transport = WebRTCVideoTransport(
            "127.0.0.1",
            8765,
            runtime_factory=factory,
        )
        self.assertIsInstance(transport, VideoTransport)
        with self.assertRaises(LifecycleError):
            transport.send(encoded(0))
        transport.start()
        transport.send(encoded(0))
        transport.send(encoded(1))
        transport.send(encoded(1))
        self.assertEqual(transport.metrics.frames_submitted, 2)
        self.assertEqual(transport.metrics.dropped_latest, 1)
        self.assertEqual(transport.metrics.dropped_stale, 1)
        with self.assertRaises(ModelValidationError):
            transport.send(encoded(2, stream_id="camera_1"))
        with self.assertRaises(ModelValidationError):
            transport.send(encoded(2, codec=VideoCodec.JPEG))
        transport.close()
        transport.close()
        self.assertTrue(runtimes[0].stopped.is_set())
        self.assertEqual(transport.connection_state, "closed")

    def test_runtime_start_failure_is_explicit_and_closes_transport(self) -> None:
        transport = WebRTCVideoTransport(
            "127.0.0.1",
            8765,
            runtime_factory=_FailedRuntime,
        )
        with self.assertRaisesRegex(LifecycleError, "failed to start"):
            transport.start()
        with self.assertRaises(LifecycleError):
            transport.send(encoded(0))
        transport.close()

    def test_start_timeout_can_still_be_closed_without_thread_leak(self) -> None:
        transport = WebRTCVideoTransport(
            "127.0.0.1",
            8765,
            runtime_factory=_TimeoutRuntime,
            start_timeout_s=0.001,
        )
        with self.assertRaisesRegex(LifecycleError, "did not become ready"):
            transport.start()
        transport.close()
        with self.assertRaises(LifecycleError):
            transport.start()

    def test_factory_uses_network_bind_address_without_optional_imports(self) -> None:
        factory = VideoTransportFactory(
            target=(
                "airo_doffy.streaming.video.webrtc_transport:"
                "create_webrtc_video"
            )
        )
        transport = factory.create(
            VideoStreamingConfig(),
            NetworkConfig(pc_ip="127.0.0.1", signaling_port=9000),
        )
        self.assertIsInstance(transport, WebRTCVideoTransport)
        transport.close()


if __name__ == "__main__":
    unittest.main()
