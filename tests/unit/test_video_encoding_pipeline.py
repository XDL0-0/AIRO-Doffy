"""Tests for bounded drop-oldest asynchronous video encoding."""

from __future__ import annotations

import threading
import time
import unittest

from airo_doffy.config import VideoStreamingConfig
from airo_doffy.core import EncodedFrame, LifecycleError, VideoCodec
from airo_doffy.streaming.video import (
    LatestVideoEncodingPipeline,
    VideoEncodingPipeline,
)

from tests.unit.test_h264_encoder import frame


class _Encoder:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def start(self) -> None:
        pass

    def encode(self, source) -> EncodedFrame:
        if source.sequence == 0:
            self.entered.set()
            self.release.wait(1.0)
        return EncodedFrame(
            sequence=source.sequence,
            source_timestamp_ns=source.source_timestamp_ns,
            receive_timestamp_ns=source.receive_timestamp_ns,
            clock_domain=source.clock_domain,
            stream_id=source.stream_id,
            data=f"frame-{source.sequence}".encode(),
            codec=VideoCodec.H264,
            width=source.shape[1],
            height=source.shape[0],
            encoded_timestamp_ns=source.processing_timestamp_ns,
        )

    def close(self) -> None:
        self.closed = True


class VideoEncodingPipelineTest(unittest.TestCase):
    def test_drop_oldest_stale_rejection_metrics_and_close(self) -> None:
        encoder = _Encoder()
        pipeline = LatestVideoEncodingPipeline(
            encoder,
            VideoStreamingConfig(
                input_queue_capacity=1,
                output_queue_capacity=1,
            ),
        )
        self.assertIsInstance(pipeline, VideoEncodingPipeline)
        with self.assertRaises(LifecycleError):
            pipeline.submit(frame())
        pipeline.start()
        self.assertTrue(pipeline.submit(frame(0)))
        self.assertTrue(encoder.entered.wait(0.5))
        self.assertTrue(pipeline.submit(frame(1)))
        self.assertTrue(pipeline.submit(frame(2)))
        self.assertFalse(pipeline.submit(frame(2)))
        encoder.release.set()
        deadline = time.monotonic() + 0.5
        while pipeline.metrics.encoded < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        latest = pipeline.read_latest()
        self.assertEqual(latest.sequence, 2)
        metrics = pipeline.metrics
        self.assertEqual(metrics.submitted, 3)
        self.assertEqual(metrics.encoded, 2)
        self.assertEqual(metrics.dropped_input, 1)
        self.assertEqual(metrics.dropped_output, 1)
        self.assertEqual(metrics.rejected_stale, 1)
        self.assertEqual(metrics.bytes_encoded, 14)
        pipeline.close()
        pipeline.close()
        self.assertTrue(encoder.closed)
        with self.assertRaises(LifecycleError):
            pipeline.read_latest()

    def test_encoder_error_is_observable_and_thread_stops(self) -> None:
        class BrokenEncoder(_Encoder):
            def encode(self, _source) -> EncodedFrame:
                raise RuntimeError("injected encode failure")

        encoder = BrokenEncoder()
        pipeline = LatestVideoEncodingPipeline(
            encoder,
            VideoStreamingConfig(),
        )
        pipeline.start()
        pipeline.submit(frame())
        deadline = time.monotonic() + 0.5
        while pipeline.health_error is None and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertIsInstance(pipeline.health_error, RuntimeError)
        self.assertEqual(pipeline.metrics.errors, 1)
        pipeline.close()
        self.assertTrue(encoder.closed)

    def test_close_before_start_releases_owned_encoder(self) -> None:
        encoder = _Encoder()
        pipeline = LatestVideoEncodingPipeline(
            encoder,
            VideoStreamingConfig(),
        )
        pipeline.close()
        pipeline.close()
        self.assertTrue(encoder.closed)


if __name__ == "__main__":
    unittest.main()
