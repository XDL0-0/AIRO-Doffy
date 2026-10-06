"""Packaged adapter for the legacy UDP media manager.

The implementation now lives under :mod:`doffy_teleop.media.udp`; this module
keeps the constructor and dependency replacement surface used by the runtimes.
"""

from __future__ import annotations

import struct  # re-exported for integrations that imported it from this module
import threading
import time
from typing import Dict, List, Tuple

import cv2  # noqa: F401 - historical public module attribute
import numpy as np  # noqa: F401 - historical public module attribute

import doffy_teleop.media.udp_comms as U
import doffy_teleop.utils as utils
from doffy_teleop.config import Config
from doffy_teleop.protocol.parse_vr import detect_packet_type, parse_data, parse_hand_data
from doffy_teleop.media.udp import (
    CONNECTION_TIMEOUT,
    MAX_STREAM_CAMERAS,
    STREAM_FPS,
    TACTILE_FPS,
    VR_RECEIVE_HZ,
    UDPManagerCore,
    _dummy_controller,
)
from doffy_teleop.protocol.jpeg import HD_HEADER_FMT, HD_HEADER_SIZE
from doffy_teleop.protocol.vr import LegacyVRPacketDecoder

try:
    from doffy_teleop.media.realsense_camera import RealSenseCameraManager
except ImportError:  # pragma: no cover - only minimal installs without SDK.
    RealSenseCameraManager = None  # type: ignore[assignment,misc]


class UDPManager(UDPManagerCore):
    """Thin compatibility adapter over the split media implementation."""

    def __init__(
        self,
        config: Config | None = None,
        camera_manager: object | None = None,
    ) -> None:
        cfg = Config() if config is None else config

        def make_camera(value: Config) -> object:
            if RealSenseCameraManager is None:
                raise RuntimeError("RealSenseCameraManager is unavailable")
            return RealSenseCameraManager(config=value)

        # Bind the legacy root functions at construction time.  This preserves
        # tests and downstream users that patch ``udp.parse_data`` or
        # ``udp.U.UdpComms`` while sharing all receive/control logic.
        decoder = LegacyVRPacketDecoder(
            detect=detect_packet_type,
            parse_controller=parse_data,
            parse_hand=parse_hand_data,
        )
        super().__init__(
            cfg,
            camera_manager=camera_manager,
            camera_factory=make_camera,
            socket_module=U,
            decoder=decoder,
            logger=utils.logger,
        )


__all__ = [
    "CONNECTION_TIMEOUT",
    "Config",
    "HD_HEADER_FMT",
    "HD_HEADER_SIZE",
    "MAX_STREAM_CAMERAS",
    "STREAM_FPS",
    "TACTILE_FPS",
    "UDPManager",
    "VR_RECEIVE_HZ",
    "detect_packet_type",
    "parse_data",
    "parse_hand_data",
]
