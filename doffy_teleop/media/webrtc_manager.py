"""Packaged adapter for WebRTC media transport.

``WebRTCSession`` and ``WebRTCSignalingServer`` now live in
``doffy_teleop.media.webrtc``.  The adapter below retains the classic UDP pose,
record-control, tactile, and ``control`` DataChannel behavior.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2  # noqa: F401 - historical public module attribute
import numpy as np

import doffy_teleop.media.udp_comms as U
import doffy_teleop.utils as utils
from doffy_teleop.config import Config
from doffy_teleop.protocol.parse_vr import detect_packet_type, parse_data, parse_hand_data
from doffy_teleop.media.camera import CameraVideoTrack
from doffy_teleop.media.webrtc import (
    CONNECTION_TIMEOUT,
    MAX_CAMERAS,
    STREAM_FPS,
    TACTILE_FPS,
    VR_RECEIVE_HZ,
    WebRTCUDPManagerCore,
    WebRTCSession,
    WebRTCSignalingServer,
    _dummy_controller,
)
from doffy_teleop.protocol.vr import LegacyVRPacketDecoder

try:  # Keep historical module attributes while remaining import-safe.
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - minimal install only.
    rs = None  # type: ignore[assignment]

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
    from aiortc.contrib.media import MediaRelay
    from aiortc.mediastreams import VideoStreamTrack
except ImportError:  # pragma: no cover - optional dependency path.
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    RTCIceCandidate = None  # type: ignore[assignment]
    MediaRelay = None  # type: ignore[assignment]
    VideoStreamTrack = None  # type: ignore[assignment]

try:
    from av import VideoFrame
except ImportError:  # pragma: no cover - optional dependency path.
    VideoFrame = None  # type: ignore[assignment]

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - optional dependency path.
    web = None  # type: ignore[assignment]

try:
    from airo_camera_toolkit.cameras.realsense.realsense import Realsense
except ImportError:  # pragma: no cover - only minimal installs without SDK.
    Realsense = None  # type: ignore[assignment,misc]


class RealsenseCameraTrack(CameraVideoTrack):
    """Legacy name for the split transport-neutral camera track."""

    def __init__(
        self,
        manager_or_provider: object,
        cam_idx: int,
        *,
        fps: float = STREAM_FPS,
    ) -> None:
        # Keep the private names used by the v1 diagnostics while the actual
        # frame lookup is delegated to the transport-neutral provider.
        self._manager = manager_or_provider
        self._cam_idx = int(cam_idx)
        self._frame_interval = 1.0 / float(fps)
        super().__init__(manager_or_provider, cam_idx, fps=fps)


class WebRTCUDPManager(WebRTCUDPManagerCore):
    """Thin compatibility adapter over the split WebRTC/UDP implementation."""

    def __init__(
        self,
        config: Config | None = None,
        camera_manager: object | None = None,
    ) -> None:
        cfg = Config() if config is None else config

        def detect_cameras() -> list[str]:
            if rs is None:
                return []
            serials: list[str] = []
            for device in rs.context().query_devices():
                serials.append(device.get_info(rs.camera_info.serial_number))
            return serials

        def make_camera(**kwargs: object) -> object:
            if Realsense is None:
                raise RuntimeError("Realsense camera adapter is unavailable")
            return Realsense(**kwargs)

        decoder = LegacyVRPacketDecoder(
            detect=detect_packet_type,
            parse_controller=parse_data,
            parse_hand=parse_hand_data,
        )
        super().__init__(
            cfg,
            camera_manager=camera_manager,
            camera_detector=detect_cameras,
            camera_factory=make_camera,
            socket_module=U,
            decoder=decoder,
            pc_factory=RTCPeerConnection,
            description_factory=RTCSessionDescription,
            web_module=web,
            track_factory=RealsenseCameraTrack,
            logger=utils.logger,
        )

    def _detect_cameras(self) -> tuple[int, list[str]]:
        """Retain the v1 diagnostic hook for enumerating RealSense devices."""

        if rs is None:
            return 0, []
        serials: list[str] = []
        for device in rs.context().query_devices():
            serials.append(device.get_info(rs.camera_info.serial_number))
        return len(serials), serials


__all__ = [
    "CONNECTION_TIMEOUT",
    "Config",
    "MAX_CAMERAS",
    "MediaRelay",
    "RTCIceCandidate",
    "RTCPeerConnection",
    "RTCSessionDescription",
    "RealsenseCameraTrack",
    "STREAM_FPS",
    "TACTILE_FPS",
    "VideoFrame",
    "VideoStreamTrack",
    "VR_RECEIVE_HZ",
    "WebRTCSession",
    "WebRTCSignalingServer",
    "WebRTCUDPManager",
    "detect_packet_type",
    "parse_data",
    "parse_hand_data",
    "rs",
]
