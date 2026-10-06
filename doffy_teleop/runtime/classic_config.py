"""Configuration constants shared by the classic data collection runtime."""

from __future__ import annotations

from doffy_teleop.config import Config
from doffy_teleop.visualization.config import VisualizerConfig


cfg = Config()
viz_cfg = VisualizerConfig()

TELEOP_HZ = cfg.UR_CTRL_RATE
MIN_DT = 1.0 / TELEOP_HZ
TACTILE_BRIDGE_HZ = 100.0
