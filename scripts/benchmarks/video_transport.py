"""Run generated-frame benchmarks for the three Phase 7 video paths."""

from __future__ import annotations

import argparse
import json
import time

from airo_doffy.config import NetworkConfig, VideoStreamingConfig
from airo_doffy.core import ClockDomain, PixelFormat, ProcessedFrame
from airo_doffy.streaming.video import (
    BenchmarkInput,
    LegacyJpegEncoder,
    LegacyJpegUdpTransport,
    LowLatencyH264Encoder,
    RtpH264UdpTransport,
    VideoBenchmarkPath,
    VideoBenchmarkRunner,
    WebRTCVideoTransport,
)


def _inputs(count: int, width: int, height: int) -> tuple[BenchmarkInput, ...]:
    now_ns = time.monotonic_ns()
    payload = bytes(width * height * 3)
    return tuple(
        BenchmarkInput(
            frame=ProcessedFrame(
                sequence=index,
                source_timestamp_ns=now_ns + index * 33_333_333,
                receive_timestamp_ns=now_ns + index * 33_333_333,
                clock_domain=ClockDomain.MONOTONIC,
                stream_id="camera_0",
                data=payload,
                shape=(height, width, 3),
                pixel_format=PixelFormat.BGR8,
                processing_timestamp_ns=now_ns + index * 33_333_333,
            ),
            queued_timestamp_ns=time.monotonic_ns(),
        )
        for index in range(count)
    )


def _path(
    name: str,
    config: VideoStreamingConfig,
    network: NetworkConfig,
) -> VideoBenchmarkPath:
    if name == "legacy_jpeg_udp":
        return VideoBenchmarkPath(
            name=name,
            encoder=LegacyJpegEncoder(config),
            transport=LegacyJpegUdpTransport(
                network.vr_ip,
                network.legacy_base_port,
                chunk_size=config.legacy_chunk_size,
            ),
        )
    encoder = LowLatencyH264Encoder(config)
    if name == "rtp_h264":
        transport = RtpH264UdpTransport(
            network.vr_ip,
            network.video_rtp_port,
            config,
        )
    else:
        transport = WebRTCVideoTransport(
            network.pc_ip or "0.0.0.0",
            network.signaling_port,
        )
    return VideoBenchmarkPath(name=name, encoder=encoder, transport=transport)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        choices=("legacy_jpeg_udp", "webrtc_h264", "rtp_h264"),
        action="append",
        required=True,
    )
    parser.add_argument("--vr-ip", default="127.0.0.1")
    parser.add_argument("--pc-ip", default="127.0.0.1")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if args.frames < 1 or args.width < 2 or args.height < 2:
        parser.error("frames, width, and height must be positive")
    if args.width % 2 or args.height % 2:
        parser.error("H.264 width and height must be even")

    config = VideoStreamingConfig()
    network = NetworkConfig(pc_ip=args.pc_ip, vr_ip=args.vr_ip)
    inputs = _inputs(args.frames, args.width, args.height)
    runner = VideoBenchmarkRunner()
    results = [
        runner.run(_path(name, config, network), inputs).to_mapping()
        for name in args.path
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
