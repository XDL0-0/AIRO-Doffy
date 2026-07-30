"""Video processing, encoding, and transport adapters."""

from .base import (
    FrameProcessor,
    VideoEncoder,
    VideoEncodingPipeline,
    VideoTransport,
)
from .encoding_pipeline import (
    LatestVideoEncodingPipeline,
    VideoEncodingMetrics,
)
from .frame_processor import FrameTransform, PackedFrameProcessor
from .h264_encoder import (
    H264EncoderSettings,
    LowLatencyH264Encoder,
    create_h264_encoder,
)
from .legacy_jpeg_udp import (
    LEGACY_JPEG_HEADER,
    LEGACY_JPEG_HEADER_SIZE,
    LegacyJpegEncoder,
    LegacyJpegUdpMetrics,
    LegacyJpegUdpTransport,
    create_legacy_jpeg_encoder,
    create_legacy_jpeg_udp,
    packetize_legacy_jpeg,
)

__all__ = [
    "FrameProcessor",
    "FrameTransform",
    "H264EncoderSettings",
    "LatestVideoEncodingPipeline",
    "LEGACY_JPEG_HEADER",
    "LEGACY_JPEG_HEADER_SIZE",
    "LegacyJpegEncoder",
    "LegacyJpegUdpMetrics",
    "LegacyJpegUdpTransport",
    "LowLatencyH264Encoder",
    "PackedFrameProcessor",
    "VideoEncoder",
    "VideoEncodingPipeline",
    "VideoEncodingMetrics",
    "VideoTransport",
    "create_h264_encoder",
    "create_legacy_jpeg_encoder",
    "create_legacy_jpeg_udp",
    "packetize_legacy_jpeg",
]
