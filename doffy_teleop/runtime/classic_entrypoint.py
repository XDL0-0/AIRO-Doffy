"""Packaged entry point for classic teleoperation and data collection."""

from __future__ import annotations

import threading
import time

import numpy as np
import doffy_teleop.utils as utils

from doffy_teleop.runtime import classic as _classic
from doffy_teleop.runtime.classic_config import (
    MIN_DT,
    TACTILE_BRIDGE_HZ,
    TELEOP_HZ,
)
from doffy_teleop.runtime.classic_helpers import (
    CameraManager,
    RecordingFrame,
    TactileDataHolder,
    controller_reset_requested as _controller_reset_requested,
    create_recording_frame,
    create_tactile_reader,
    request_tactile_recalibration,
    run_tactile_reader,
    tactile_bridge_loop,
    visualizer_publish_loop,
    _visualizer_image,
    _visualizer_images,
    _visualizer_tcp_translation,
)
from doffy_teleop.config import Config
from doffy_teleop.visualization.config import VisualizerConfig

cfg = _classic.cfg
viz_cfg = _classic.viz_cfg
DatasetRecorder = _classic.DatasetRecorder
DataRecordingService = _classic.DataRecordingService
ManagerRecordingControl = _classic.ManagerRecordingControl
RobotTeleop = _classic.RobotTeleop
UDPManager = _classic.UDPManager
WebRTCUDPManager = _classic.WebRTCUDPManager


def controller_reset_requested(data) -> bool:
    """Preserve the historical helper's live ``main.cfg`` patch point."""
    return _controller_reset_requested(data, cfg=cfg)


def main() -> None:
    """Run the classic teleoperation data collection lifecycle."""
    _classic.cfg = cfg
    _classic.viz_cfg = viz_cfg
    _classic.RobotTeleop = RobotTeleop
    _classic.UDPManager = UDPManager
    _classic.WebRTCUDPManager = WebRTCUDPManager
    _classic.main(config=cfg, visualizer_config=viz_cfg)


__all__ = [
    "CameraManager",
    "Config",
    "DataRecordingService",
    "DatasetRecorder",
    "MIN_DT",
    "ManagerRecordingControl",
    "RecordingFrame",
    "RobotTeleop",
    "TACTILE_BRIDGE_HZ",
    "TELEOP_HZ",
    "TactileDataHolder",
    "UDPManager",
    "VisualizerConfig",
    "WebRTCUDPManager",
    "cfg",
    "controller_reset_requested",
    "create_recording_frame",
    "create_tactile_reader",
    "main",
    "request_tactile_recalibration",
    "run_tactile_reader",
    "tactile_bridge_loop",
    "visualizer_publish_loop",
    "viz_cfg",
]


if __name__ == "__main__":
    main()
