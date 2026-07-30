"""Thread-safe episode recording lifecycle independent of serialization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock

from ..core.errors import LifecycleError, ModelValidationError


class EpisodeState(str, Enum):
    """Externally visible recording lifecycle."""

    IDLE = "idle"
    RECORDING = "recording"
    EXPORT_PENDING = "export_pending"
    EXPORT_FAILED = "export_failed"
    ROLLBACK_PENDING = "rollback_pending"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class EpisodeStatus:
    """Immutable state snapshot for status consumers."""

    state: EpisodeState
    next_episode_index: int
    active_samples: int
    pending_episode_index: int | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class RollbackRequest:
    """Result of a rollback request before storage work is performed."""

    discard_active: bool
    episode_index: int | None


class EpisodeStateMachine:
    """Own episode transitions and stable index allocation."""

    def __init__(self, *, next_episode_index: int = 0) -> None:
        if (
            isinstance(next_episode_index, bool)
            or not isinstance(next_episode_index, int)
            or next_episode_index < 0
        ):
            raise ModelValidationError(
                "next_episode_index must be a non-negative integer"
            )
        self._state = EpisodeState.IDLE
        self._next_episode_index = next_episode_index
        self._active_samples = 0
        self._pending_episode_index: int | None = None
        self._last_error: str | None = None
        self._lock = RLock()

    def snapshot(self) -> EpisodeStatus:
        with self._lock:
            return EpisodeStatus(
                state=self._state,
                next_episode_index=self._next_episode_index,
                active_samples=self._active_samples,
                pending_episode_index=self._pending_episode_index,
                last_error=self._last_error,
            )

    def start_episode(self) -> int:
        with self._lock:
            self._require(EpisodeState.IDLE)
            self._state = EpisodeState.RECORDING
            self._active_samples = 0
            self._last_error = None
            return self._next_episode_index

    def note_sample(self) -> int:
        with self._lock:
            self._require(EpisodeState.RECORDING)
            self._active_samples += 1
            return self._active_samples

    def request_finish(self) -> int:
        with self._lock:
            self._require(EpisodeState.RECORDING)
            if self._active_samples == 0:
                raise LifecycleError("cannot finish an empty episode")
            index = self._next_episode_index
            self._state = EpisodeState.EXPORT_PENDING
            self._pending_episode_index = index
            return index

    def export_succeeded(self, episode_index: int) -> None:
        with self._lock:
            self._require_pending(EpisodeState.EXPORT_PENDING, episode_index)
            self._next_episode_index = episode_index + 1
            self._return_to_idle()

    def export_failed(self, episode_index: int, error: str) -> None:
        with self._lock:
            self._require_pending(EpisodeState.EXPORT_PENDING, episode_index)
            if not isinstance(error, str) or not error.strip():
                raise ModelValidationError("export error must be a non-empty string")
            self._state = EpisodeState.EXPORT_FAILED
            self._last_error = error

    def retry_export(self) -> int:
        with self._lock:
            self._require(EpisodeState.EXPORT_FAILED)
            assert self._pending_episode_index is not None
            self._state = EpisodeState.EXPORT_PENDING
            self._last_error = None
            return self._pending_episode_index

    def discard_failed_export(self) -> int:
        with self._lock:
            self._require(EpisodeState.EXPORT_FAILED)
            assert self._pending_episode_index is not None
            index = self._pending_episode_index
            self._return_to_idle()
            return index

    def request_rollback(self) -> RollbackRequest:
        with self._lock:
            if self._state is EpisodeState.RECORDING:
                request = RollbackRequest(
                    discard_active=True,
                    episode_index=None,
                )
                self._return_to_idle()
                return request
            self._require(EpisodeState.IDLE)
            if self._next_episode_index == 0:
                return RollbackRequest(
                    discard_active=False,
                    episode_index=None,
                )
            episode_index = self._next_episode_index - 1
            self._state = EpisodeState.ROLLBACK_PENDING
            self._pending_episode_index = episode_index
            return RollbackRequest(
                discard_active=False,
                episode_index=episode_index,
            )

    def rollback_succeeded(self, episode_index: int) -> None:
        with self._lock:
            self._require_pending(EpisodeState.ROLLBACK_PENDING, episode_index)
            self._next_episode_index = episode_index
            self._return_to_idle()

    def rollback_failed(self, episode_index: int, error: str | None = None) -> None:
        with self._lock:
            self._require_pending(EpisodeState.ROLLBACK_PENDING, episode_index)
            self._return_to_idle()
            self._last_error = error

    def close(self) -> None:
        with self._lock:
            if self._state in {
                EpisodeState.EXPORT_PENDING,
                EpisodeState.ROLLBACK_PENDING,
            }:
                raise LifecycleError("cannot close while storage work is pending")
            self._state = EpisodeState.CLOSED
            self._active_samples = 0
            self._pending_episode_index = None

    def _require(self, expected: EpisodeState) -> None:
        if self._state is not expected:
            raise LifecycleError(
                f"episode state must be {expected.value}, got {self._state.value}"
            )

    def _require_pending(
        self,
        expected: EpisodeState,
        episode_index: int,
    ) -> None:
        self._require(expected)
        if episode_index != self._pending_episode_index:
            raise LifecycleError(
                f"pending episode is {self._pending_episode_index}, got {episode_index}"
            )

    def _return_to_idle(self) -> None:
        self._state = EpisodeState.IDLE
        self._active_samples = 0
        self._pending_episode_index = None
        self._last_error = None
