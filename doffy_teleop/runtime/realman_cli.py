"""Packaged entry point for the modular RealMan teleoperation runtime.

The implementation lives under :mod:`doffy_teleop`; these imports intentionally
provide the public runtime API and patchable integration dependencies.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import cv2
import numpy as np

import doffy_teleop.utils as utils
from airo_spatial_algebra.se3 import SE3Container
from doffy_teleop.config import Config
from doffy_teleop.sensors.force_filter import WrenchFilter
from doffy_teleop.robots.backend_api import RealManBackend, make_robot_backend
from doffy_teleop.control.canfd_loop import CanfdCommandLoop
from doffy_teleop.control.contracts import (
    CanfdLoopSnapshot,
    QuestTcpStateSender,
    RealManStateSnapshot,
    pack_quest_tcp_state_packet,
)
from doffy_teleop.control.realman_qp import RealManRemoteIkSolver
from doffy_teleop.runtime.publisher import (
    _create_camera_manager,
    _visualizer_image,
    _visualizer_images,
    visualizer_publish_loop,
)
from doffy_teleop.runtime.realman import (
    RealManTeleop as _RealManTeleop,
    _UNCLOSED_REALMAN_TELEOPS,
)
from doffy_teleop.recording.service import RealManEpisodeRecorder
from doffy_teleop.visualization.config import VisualizerConfig


class RealManTeleop(_RealManTeleop):
    """Backward-compatible facade with patchable legacy backend factory.

    The class implementation is inherited from ``doffy_teleop.runtime.realman``;
    passing the factory keeps integrations that patch
    ``realman_teleop.make_robot_backend`` working. The implementation validates
    configuration before constructing any backend.
    """

    def __init__(
        self,
        initial_controller_data: list[dict],
        *,
        cfg: Config | None = None,
        backend: Any | None = None,
    ) -> None:
        super().__init__(
            initial_controller_data,
            cfg=cfg,
            backend=backend,
            backend_factory=make_robot_backend,
        )


def main() -> None:
    """Run the stable RealMan composition entry point."""
    from doffy_teleop.runtime.realman_entrypoint import main as run_main

    run_main(
        config_factory=Config,
        visualizer_config_factory=VisualizerConfig,
        camera_manager_factory=_create_camera_manager,
        teleop_factory=RealManTeleop,
        recorder_factory=RealManEpisodeRecorder,
        tcp_state_sender_factory=QuestTcpStateSender,
        visualizer_publish_loop_fn=visualizer_publish_loop,
    )


__all__ = [
    "CanfdCommandLoop",
    "CanfdLoopSnapshot",
    "Config",
    "QuestTcpStateSender",
    "RealManBackend",
    "RealManEpisodeRecorder",
    "RealManRemoteIkSolver",
    "RealManStateSnapshot",
    "RealManTeleop",
    "_UNCLOSED_REALMAN_TELEOPS",
    "_create_camera_manager",
    "_visualizer_image",
    "_visualizer_images",
    "main",
    "pack_quest_tcp_state_packet",
    "SE3Container",
    "VisualizerConfig",
    "WrenchFilter",
    "visualizer_publish_loop",
]


if __name__ == "__main__":
    main()
