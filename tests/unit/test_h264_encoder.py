"""Tests for lazy H.264 selection, low-latency options, and metadata."""

from __future__ import annotations

import unittest

from airo_doffy.config import EncoderFactory, VideoStreamingConfig
from airo_doffy.core import (
    ClockDomain,
    LifecycleError,
    ModelValidationError,
    PixelFormat,
    ProcessedFrame,
    VideoCodec,
    VideoEncodingError,
)
from airo_doffy.streaming.video import (
    H264EncoderSettings,
    LowLatencyH264Encoder,
    VideoEncoder,
)
from airo_doffy.streaming.video.h264_encoder import _select_codec_backend


class _Clock:
    def now_ns(self) -> int:
        return 99


class _Backend:
    def __init__(self, codec_name: str = "fake264") -> None:
        self.codec_name = codec_name
        self.frames = []
        self.closed = False

    def encode(self, frame: ProcessedFrame) -> tuple[bytes, bool]:
        self.frames.append(frame)
        return b"\x00\x00\x00\x01\x65payload", frame.sequence == 0

    def close(self) -> None:
        self.closed = True


def frame(
    sequence: int = 0,
    *,
    width: int = 2,
    height: int = 2,
    pixel_format: PixelFormat = PixelFormat.BGR8,
) -> ProcessedFrame:
    if pixel_format in {PixelFormat.RGB8, PixelFormat.BGR8}:
        shape = (height, width, 3)
        size = height * width * 3
    elif pixel_format is PixelFormat.GRAY8:
        shape = (height, width)
        size = height * width
    else:
        shape = (height, width)
        size = height * width * 2
    return ProcessedFrame(
        sequence=sequence,
        source_timestamp_ns=10,
        receive_timestamp_ns=11,
        clock_domain=ClockDomain.MONOTONIC,
        stream_id="camera_0",
        data=bytes(size),
        shape=shape,
        pixel_format=pixel_format,
        processing_timestamp_ns=12,
    )


class H264EncoderTest(unittest.TestCase):
    def test_low_latency_options_and_codec_candidates(self) -> None:
        auto = H264EncoderSettings.from_config(VideoStreamingConfig())
        self.assertEqual(auto.codec_candidates, ("h264_nvenc", "libx264"))
        self.assertEqual(auto.codec_options("h264_nvenc")["bf"], "0")
        self.assertEqual(auto.codec_options("h264_nvenc")["rc-lookahead"], "0")
        software = H264EncoderSettings.from_config(
            VideoStreamingConfig(encoder_backend="software")
        )
        self.assertEqual(software.codec_candidates, ("libx264",))
        self.assertIn("bframes=0", software.codec_options("libx264")["x264-params"])
        self.assertIn(
            "rc-lookahead=0",
            software.codec_options("libx264")["x264-params"],
        )
        with self.assertRaises(ModelValidationError):
            H264EncoderSettings("invalid", 1, 1, 1)

    def test_auto_falls_back_but_explicit_nvenc_does_not(self) -> None:
        calls = []

        def builder(codec_name, *_args):
            calls.append(codec_name)
            if codec_name == "h264_nvenc":
                raise RuntimeError("unavailable")
            return _Backend(codec_name)

        settings = H264EncoderSettings.from_config(VideoStreamingConfig())
        selected = _select_codec_backend(
            settings,
            2,
            2,
            PixelFormat.BGR8,
            builder,
        )
        self.assertEqual(selected.codec_name, "libx264")
        self.assertEqual(calls, ["h264_nvenc", "libx264"])
        with self.assertRaises(VideoEncodingError):
            _select_codec_backend(
                H264EncoderSettings.from_config(
                    VideoStreamingConfig(encoder_backend="nvenc")
                ),
                2,
                2,
                PixelFormat.BGR8,
                builder,
            )

    def test_encode_metadata_fixed_geometry_and_lifecycle(self) -> None:
        backend = _Backend()
        encoder = LowLatencyH264Encoder(
            VideoStreamingConfig(),
            clock=_Clock(),
            codec_builder=lambda *_args: backend,
        )
        self.assertIsInstance(encoder, VideoEncoder)
        with self.assertRaises(LifecycleError):
            encoder.encode(frame())
        encoder.start()
        encoded = encoder.encode(frame())
        self.assertEqual(encoded.codec, VideoCodec.H264)
        self.assertEqual(encoded.data, b"\x00\x00\x00\x01\x65payload")
        self.assertEqual(encoded.encoded_timestamp_ns, 99)
        self.assertEqual((encoded.width, encoded.height), (2, 2))
        self.assertTrue(encoded.keyframe)
        with self.assertRaises(ModelValidationError):
            encoder.encode(frame(1, width=4))
        encoder.close()
        encoder.close()
        self.assertTrue(backend.closed)
        with self.assertRaises(LifecycleError):
            encoder.start()

    def test_rejects_depth_odd_dimensions_and_empty_payload(self) -> None:
        with self.assertRaises(ModelValidationError):
            encoder = LowLatencyH264Encoder(
                VideoStreamingConfig(),
                codec_builder=lambda *_args: _Backend(),
            )
            encoder.start()
            encoder.encode(frame(pixel_format=PixelFormat.DEPTH_U16))
        with self.assertRaises(ModelValidationError):
            encoder = LowLatencyH264Encoder(
                VideoStreamingConfig(),
                codec_builder=lambda *_args: _Backend(),
            )
            encoder.start()
            encoder.encode(frame(width=3))

        class EmptyBackend(_Backend):
            def encode(self, frame: ProcessedFrame) -> tuple[bytes, bool]:
                return b"", False

        encoder = LowLatencyH264Encoder(
            VideoStreamingConfig(),
            codec_builder=lambda *_args: EmptyBackend(),
        )
        encoder.start()
        with self.assertRaises(VideoEncodingError):
            encoder.encode(frame())
        encoder.close()

    def test_factory_constructs_without_loading_pyav(self) -> None:
        factory = EncoderFactory(
            target="airo_doffy.streaming.video.h264_encoder:create_h264_encoder"
        )
        encoder = factory.create(VideoStreamingConfig())
        self.assertIsInstance(encoder, LowLatencyH264Encoder)
        encoder.close()


if __name__ == "__main__":
    unittest.main()
