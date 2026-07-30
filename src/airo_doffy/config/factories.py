"""Focused, dependency-injected factories with lazy adapter imports."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from ..core.errors import ModelValidationError, OptionalDependencyError
from ..devices.cameras.base import CameraSource
from ..devices.tactile.base import TactileSensor
from ..devices.vr.base import VRInputSource
from ..recording.base import EpisodeRecorder
from ..robots.base import RobotBackend
from ..streaming.video.base import VideoEncoder, VideoTransport
from ..visualization.base import SnapshotConsumer
from .models import (
    CameraConfig,
    NetworkConfig,
    RecordingConfig,
    RobotConfig,
    TactileConfig,
    VideoStreamingConfig,
    VisualizationConfig,
    VRConfig,
)


def _load_constructor(target: str) -> Callable[..., object]:
    module_name, separator, symbol = target.partition(":")
    if not separator or not module_name or not symbol:
        raise ModelValidationError(f"factory target must use module:symbol syntax: {target!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"cannot load configured adapter {target!r}; install its optional dependencies"
        ) from exc
    try:
        constructor = getattr(module, symbol)
    except AttributeError as exc:
        raise ModelValidationError(f"factory target does not exist: {target!r}") from exc
    if not callable(constructor):
        raise ModelValidationError(f"factory target is not callable: {target!r}")
    return constructor


def _construct(target: str, expected_type: type, *args: object) -> Any:
    component = _load_constructor(target)(*args)
    if not isinstance(component, expected_type):
        raise ModelValidationError(
            f"factory target {target!r} returned {type(component).__name__}, "
            f"which does not satisfy {expected_type.__name__}"
        )
    return component


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotFactory:
    """Create a selected robot backend only when requested."""

    target: str

    def create(self, config: RobotConfig) -> RobotBackend:
        return cast(RobotBackend, _construct(self.target, RobotBackend, config))


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraFactory:
    """Create a selected camera source only when requested."""

    target: str

    def create(self, config: CameraConfig) -> CameraSource:
        return cast(CameraSource, _construct(self.target, CameraSource, config))


@dataclass(frozen=True, slots=True, kw_only=True)
class EncoderFactory:
    """Create a video encoder without importing its codec at configuration time."""

    target: str

    def create(self, config: VideoStreamingConfig) -> VideoEncoder:
        return cast(VideoEncoder, _construct(self.target, VideoEncoder, config))


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoTransportFactory:
    """Create a video transport from only video and network sections."""

    target: str

    def create(
        self,
        config: VideoStreamingConfig,
        network: NetworkConfig,
    ) -> VideoTransport:
        return cast(
            VideoTransport,
            _construct(self.target, VideoTransport, config, network),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VRSourceFactory:
    """Create a controller or hand-tracking source from its narrow sections."""

    target: str

    def create(self, config: VRConfig, network: NetworkConfig) -> VRInputSource:
        return cast(
            VRInputSource,
            _construct(self.target, VRInputSource, config, network),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TactileFactory:
    """Create the supported tactile sensor without loading BLE at import time."""

    target: str

    def create(self, config: TactileConfig) -> TactileSensor:
        return cast(TactileSensor, _construct(self.target, TactileSensor, config))


@dataclass(frozen=True, slots=True, kw_only=True)
class RecorderFactory:
    """Create a selected episode recorder from recording values only."""

    target: str

    def create(self, config: RecordingConfig) -> EpisodeRecorder:
        return cast(EpisodeRecorder, _construct(self.target, EpisodeRecorder, config))


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualizerFactory:
    """Create a lifecycle-owned visualizer from visualization values only."""

    target: str

    def create(self, config: VisualizationConfig) -> SnapshotConsumer:
        return cast(
            SnapshotConsumer,
            _construct(self.target, SnapshotConsumer, config),
        )
