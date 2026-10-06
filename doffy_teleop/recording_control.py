"""Ordered recording requests, independent of camera and robot locks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any, Protocol


@dataclass(frozen=True)
class RecordingRequest:
    sequence: int
    command: str


class RecordingControlProtocol(Protocol):
    def snapshot(self) -> tuple[bool, bool, bool]: ...
    def start_recording(self) -> bool: ...
    def request_export(self) -> bool: ...
    def request_rollback(self) -> bool: ...
    def take_pending(self) -> RecordingRequest | None: ...
    def finish_pending(self, request: RecordingRequest) -> None: ...
    def stop_recording(self) -> None: ...
    def begin_shutdown(self) -> None: ...


class RecordingControl:
    """Queue dataset mutations; resume collection only after they finish.

    Start describes the latest requested collection state. Stop and Undo are
    individual operations, so a request arriving during disk I/O cannot be
    erased when an older request completes. No camera lock is acquired here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collecting = False
        self._pending: deque[RecordingRequest] = deque()
        self._active: RecordingRequest | None = None
        self._sequence = 0
        self._accepting = True

    def snapshot(self) -> tuple[bool, bool, bool]:
        with self._lock:
            operations = tuple(self._pending) + ((self._active,) if self._active else ())
            return (
                self._collecting and not operations,
                any(item.command == "Stop" for item in operations),
                any(item.command == "Undo" for item in operations),
            )

    def start_recording(self) -> bool:
        with self._lock:
            if not self._accepting or self._collecting:
                return False
            self._collecting = True
            return True

    def _enqueue(self, command: str) -> None:
        self._sequence += 1
        self._pending.append(RecordingRequest(self._sequence, command))

    def request_export(self) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            was_collecting = self._collecting
            self._collecting = False
            self._enqueue("Stop")
            return was_collecting

    def request_rollback(self) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            self._collecting = False
            self._enqueue("Undo")
            return True

    def take_pending(self) -> RecordingRequest | None:
        with self._lock:
            if self._active is not None or not self._pending:
                return None
            self._active = self._pending.popleft()
            return self._active

    def finish_pending(self, request: RecordingRequest) -> None:
        with self._lock:
            if self._active != request:
                raise ValueError("Recording request does not match the active operation")
            self._active = None

    def _clear_legacy_flag(self, command: str) -> None:
        # Compatibility for callers assigning the old flags. A running
        # operation belongs to its consumer and cannot be acknowledged here.
        with self._lock:
            for item in self._pending:
                if item.command == command:
                    self._pending.remove(item)
                    break

    def clear_export(self) -> None:
        self._clear_legacy_flag("Stop")

    def clear_rollback(self) -> None:
        self._clear_legacy_flag("Undo")

    def stop_recording(self) -> None:
        with self._lock:
            self._collecting = False

    def begin_shutdown(self) -> None:
        with self._lock:
            self._accepting = False
            self._collecting = False


class _LegacyFlagsControl:
    """Compatibility for external camera managers exposing plain flags only.

    Production managers use RecordingControl. For old flag-only integrations,
    claim a flag *before* I/O so completion cannot clear a newer request.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self._active: RecordingRequest | None = None
        self._sequence = 0

    def snapshot(self) -> tuple[bool, bool, bool]:
        with self.manager._lock:
            return (
                bool(self.manager.data_collecting_state) and self._active is None,
                bool(self.manager.data_export_state) or bool(self._active and self._active.command == "Stop"),
                bool(self.manager.data_rollback_state) or bool(self._active and self._active.command == "Undo"),
            )

    def start_recording(self) -> bool:
        with self.manager._lock:
            if self.manager.data_collecting_state:
                return False
            self.manager.data_collecting_state = True
            return True

    def request_export(self) -> bool:
        with self.manager._lock:
            was_collecting = bool(self.manager.data_collecting_state)
            self.manager.data_collecting_state = False
            self.manager.data_export_state = True
            return was_collecting

    def request_rollback(self) -> bool:
        with self.manager._lock:
            self.manager.data_collecting_state = False
            self.manager.data_rollback_state = True
            return True

    def take_pending(self) -> RecordingRequest | None:
        with self.manager._lock:
            if self._active is not None:
                return None
            if self.manager.data_rollback_state:
                command = "Undo"
                self.manager.data_rollback_state = False
            elif self.manager.data_export_state:
                command = "Stop"
                self.manager.data_export_state = False
            else:
                return None
            self._sequence += 1
            self._active = RecordingRequest(self._sequence, command)
            return self._active

    def finish_pending(self, request: RecordingRequest) -> None:
        with self.manager._lock:
            if self._active != request:
                raise ValueError("Recording request does not match the active operation")
            self._active = None

    def clear_export(self) -> None:
        with self.manager._lock:
            self.manager.data_export_state = False

    def clear_rollback(self) -> None:
        with self.manager._lock:
            self.manager.data_rollback_state = False

    def stop_recording(self) -> None:
        with self.manager._lock:
            self.manager.data_collecting_state = False

    def begin_shutdown(self) -> None:
        self.stop_recording()


class ManagerRecordingControl:
    """Use the manager's recording state without nesting its camera lock."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        state = getattr(manager, "control_state", None)
        self._control = getattr(state, "recording", None) or _LegacyFlagsControl(manager)

    def snapshot(self) -> tuple[bool, bool, bool]:
        return self._control.snapshot()

    def start_recording(self) -> bool:
        return self._control.start_recording()

    def request_export(self) -> bool:
        return self._control.request_export()

    def request_rollback(self) -> bool:
        return self._control.request_rollback()

    def take_pending(self) -> RecordingRequest | None:
        return self._control.take_pending()

    def finish_pending(self, request: RecordingRequest) -> None:
        self._control.finish_pending(request)

    def clear_export(self) -> None:
        self._control.clear_export()

    def clear_rollback(self) -> None:
        self._control.clear_rollback()

    def stop_recording(self) -> None:
        self._control.stop_recording()

    def begin_shutdown(self) -> None:
        self._control.begin_shutdown()
