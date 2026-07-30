"""Tests for comparable video timing, loss, bitrate, and resource metrics."""

from __future__ import annotations

import unittest

from airo_doffy.core import (
    ClockDomain,
    EncodedFrame,
    ModelValidationError,
    VideoCodec,
)
from airo_doffy.streaming.video import (
    BenchmarkInput,
    DeliveryReceipt,
    VideoBenchmarkPath,
    VideoBenchmarkRunner,
    compare_video_paths,
    summarize_latencies,
)

from tests.unit.test_h264_encoder import frame


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000

    def now_ns(self) -> int:
        self.value += 100
        return self.value


class _Encoder:
    def __init__(self) -> None:
        self.closed = False

    def start(self) -> None:
        pass

    def encode(self, source) -> EncodedFrame:
        return EncodedFrame(
            sequence=source.sequence,
            source_timestamp_ns=source.source_timestamp_ns,
            receive_timestamp_ns=source.receive_timestamp_ns,
            clock_domain=ClockDomain.MONOTONIC,
            stream_id=source.stream_id,
            data=b"1234",
            codec=VideoCodec.H264,
            width=source.shape[1],
            height=source.shape[0],
            encoded_timestamp_ns=1_500,
        )

    def close(self) -> None:
        self.closed = True


class _Metrics:
    bytes_sent = 20
    dropped_stale = 1


class _Transport:
    def __init__(self) -> None:
        self.metrics = _Metrics()
        self.closed = False

    def start(self) -> None:
        pass

    def send(self, _encoded) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _Probe:
    def wait_for(self, sequence: int, _timeout_s: float):
        if sequence == 1:
            return None
        return DeliveryReceipt(
            sequence=sequence,
            received_timestamp_ns=1_700,
            displayed_timestamp_ns=2_000,
            wire_bytes=12,
        )


class VideoBenchmarkTest(unittest.TestCase):
    def test_latency_summary_nearest_rank(self) -> None:
        summary = summarize_latencies((5, 1, 3, 2, 4))
        self.assertEqual(summary.minimum_ns, 1)
        self.assertEqual(summary.p50_ns, 3)
        self.assertEqual(summary.p95_ns, 5)
        self.assertEqual(summary.maximum_ns, 5)
        self.assertEqual(summarize_latencies(()).count, 0)

    def test_runner_reports_comparable_metrics_and_optional_receipts(self) -> None:
        encoder = _Encoder()
        transport = _Transport()
        clock = _Clock()
        cpu_values = iter((100, 300))
        runner = VideoBenchmarkRunner(
            clock=clock,
            cpu_clock=lambda: next(cpu_values),
            gpu_sampler=lambda: 25.0,
        )
        inputs = tuple(
            BenchmarkInput(
                frame=frame(index),
                queued_timestamp_ns=1_000,
            )
            for index in range(2)
        )
        result = runner.run(
            VideoBenchmarkPath("webrtc_h264", encoder, transport),
            inputs,
            delivery_probe=_Probe(),
        )
        self.assertEqual(result.frames_attempted, 2)
        self.assertEqual(result.frames_encoded, 2)
        self.assertEqual(result.frames_submitted, 2)
        self.assertEqual(result.encoded_bytes, 8)
        self.assertEqual(result.wire_bytes, 12)
        self.assertEqual(result.component_drops, 1)
        self.assertEqual(result.delivery_loss_rate, 0.5)
        self.assertEqual(result.gpu_utilization_percent, 25.0)
        self.assertEqual(result.receive_latency.count, 1)
        self.assertEqual(result.end_to_end_latency.count, 1)
        self.assertTrue(encoder.closed)
        self.assertTrue(transport.closed)
        self.assertEqual(result.to_mapping()["path_name"], "webrtc_h264")

    def test_compare_requires_unique_paths_and_reuses_immutable_inputs(self) -> None:
        inputs = (BenchmarkInput(frame=frame(), queued_timestamp_ns=0),)
        paths = (
            VideoBenchmarkPath("legacy", _Encoder(), _Transport()),
            VideoBenchmarkPath("rtp", _Encoder(), _Transport()),
        )
        results = compare_video_paths(paths, inputs)
        self.assertEqual([result.path_name for result in results], ["legacy", "rtp"])
        with self.assertRaisesRegex(ModelValidationError, "unique"):
            compare_video_paths(
                (
                    VideoBenchmarkPath("same", _Encoder(), _Transport()),
                    VideoBenchmarkPath("same", _Encoder(), _Transport()),
                ),
                inputs,
            )


if __name__ == "__main__":
    unittest.main()
