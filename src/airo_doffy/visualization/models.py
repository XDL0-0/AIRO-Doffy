"""Immutable snapshots consumed by visualization adapters."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ModelValidationError
from ..core.types import (
    ClockDomain,
    ProcessedFrame,
    RobotState,
    SequencedSample,
    TactileSample,
    WrenchSample,
)


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "" if allow_empty else " non-empty"
        raise ModelValidationError(f"{name} must be a{qualifier} string")
    return value


def _non_negative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelValidationError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordingView:
    """Read-only recording status copied into a visualization snapshot."""

    dataset_type: str
    dataset_dir: str
    recorded_episodes: int
    current_episode_frames: int
    last_episode_length: int | None = None
    collecting: bool = False
    pending_exports: int = 0
    last_error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_type",
            _text(self.dataset_type, "dataset_type"),
        )
        object.__setattr__(
            self,
            "dataset_dir",
            _text(self.dataset_dir, "dataset_dir"),
        )
        object.__setattr__(
            self,
            "recorded_episodes",
            _non_negative(self.recorded_episodes, "recorded_episodes"),
        )
        object.__setattr__(
            self,
            "current_episode_frames",
            _non_negative(
                self.current_episode_frames,
                "current_episode_frames",
            ),
        )
        if self.last_episode_length is not None:
            object.__setattr__(
                self,
                "last_episode_length",
                _non_negative(self.last_episode_length, "last_episode_length"),
            )
        object.__setattr__(
            self,
            "pending_exports",
            _non_negative(self.pending_exports, "pending_exports"),
        )
        object.__setattr__(self, "collecting", bool(self.collecting))
        object.__setattr__(
            self,
            "last_error",
            _text(self.last_error, "last_error", allow_empty=True),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualizationSnapshot(SequencedSample):
    """Complete optional display state assembled outside the visualizer."""

    robot: RobotState | None = None
    frames: tuple[ProcessedFrame, ...] = ()
    tactile: TactileSample | None = None
    wrench: WrenchSample | None = None
    recording: RecordingView | None = None
    source_label: str = "teleop data"
    status_extra: str = ""
    connected: bool = True
    error: str = ""
    clock_domain: ClockDomain = ClockDomain.MONOTONIC

    def __post_init__(self) -> None:
        SequencedSample.__post_init__(self)
        frames = tuple(self.frames)
        if self.robot is not None and not isinstance(self.robot, RobotState):
            raise ModelValidationError("robot must be a RobotState or None")
        if any(not isinstance(frame, ProcessedFrame) for frame in frames):
            raise ModelValidationError("frames must contain ProcessedFrame values")
        stream_ids = tuple(frame.stream_id for frame in frames)
        if len(set(stream_ids)) != len(stream_ids):
            raise ModelValidationError("visualization frame stream_ids must be unique")
        if self.tactile is not None and not isinstance(self.tactile, TactileSample):
            raise ModelValidationError("tactile must be a TactileSample or None")
        if self.wrench is not None and not isinstance(self.wrench, WrenchSample):
            raise ModelValidationError("wrench must be a WrenchSample or None")
        if self.recording is not None and not isinstance(
            self.recording,
            RecordingView,
        ):
            raise ModelValidationError("recording must be a RecordingView or None")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(
            self,
            "source_label",
            _text(self.source_label, "source_label"),
        )
        object.__setattr__(
            self,
            "status_extra",
            _text(self.status_extra, "status_extra", allow_empty=True),
        )
        object.__setattr__(self, "connected", bool(self.connected))
        object.__setattr__(
            self,
            "error",
            _text(self.error, "error", allow_empty=True),
        )
