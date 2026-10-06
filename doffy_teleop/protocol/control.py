"""Shared legacy record, zoom, and VR state semantics."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .vr import LegacyVRPacketDecoder
from doffy_teleop.recording_control import RecordingControl


@dataclass(frozen=True)
class RecordControl:
    """Normalized state transition emitted by a classic control packet."""

    command: str
    collecting: bool
    export: bool
    rollback: bool


class LegacyControlState:
    """Thread-safe state shared by UDP and WebRTC compatibility managers.

    Attribute names deliberately match the old managers and the recording
    callers in the Classic runtime and recording service.  Mutating those attributes is
    still supported through manager properties; packet handling itself is
    centralized here so the two transports cannot drift.
    """

    def __init__(
        self,
        camera_num: int,
        initial_port: int,
        *,
        lock: threading.RLock | threading.Lock | None = None,
    ) -> None:
        self.lock = lock or threading.RLock()
        self.initial_port = int(initial_port)
        self.camera_num = max(0, int(camera_num))
        self.data: list[dict] | None = None
        self.hand_data: dict[str, dict] = {}
        self.recording = RecordingControl()
        self.tactile_byte: bytes | None = None
        self.tactile_data = None
        self.tactile_timestamp_ns = 0
        self.vr_input_timestamp_ns = 0
        self.camera_zoom = [1.0] * self.camera_num

    @property
    def data_collecting_state(self) -> bool:
        return self.recording.snapshot()[0]

    @data_collecting_state.setter
    def data_collecting_state(self, value: bool) -> None:
        if value:
            self.recording.start_recording()
        else:
            self.recording.stop_recording()

    @property
    def data_export_state(self) -> bool:
        return self.recording.snapshot()[1]

    @data_export_state.setter
    def data_export_state(self, value: bool) -> None:
        if value:
            self.recording.request_export()
        else:
            self.recording.clear_export()

    @property
    def data_rollback_state(self) -> bool:
        return self.recording.snapshot()[2]

    @data_rollback_state.setter
    def data_rollback_state(self, value: bool) -> None:
        if value:
            self.recording.request_rollback()
        else:
            self.recording.clear_rollback()

    def consume_vr(
        self,
        raw: bytes | bytearray | memoryview | str,
        decoder: LegacyVRPacketDecoder,
        *,
        receive_timestamp_ns: int | None = None,
    ) -> bool:
        """Decode and publish one packet using the historical replacement rule.

        A controller packet clears hand state; a hand packet clears controller
        state and updates only the corresponding side.  This is the behavior
        expected by the existing teleoperation loop.
        """

        decoded = decoder.decode(raw)
        if decoded is None:
            return False
        packet_type, parsed = decoded
        now = time.monotonic_ns() if receive_timestamp_ns is None else int(receive_timestamp_ns)
        with self.lock:
            if packet_type == "controller":
                self.data = parsed
                self.hand_data.clear()
            else:
                self.hand_data[str(parsed["side"])] = parsed
                self.data = None
            self.vr_input_timestamp_ns = now
        return True

    def apply_record_control(self, value: bytes | str | None) -> RecordControl | None:
        """Apply ``Start``, ``Stop``, ``Undo``/aliases and return the transition."""

        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        command = str(value).strip()
        if command == "Start":
            self.recording.start_recording()
            return RecordControl(command, True, False, False)
        if command == "Stop":
            self.recording.request_export()
            return RecordControl(command, False, True, False)
        if command in {"Undo", "Rollback", "DeleteLast"}:
            self.recording.request_rollback()
            return RecordControl(command, False, False, True)
        return None

    def update_zoom(
        self,
        value: bytes | str | None,
        *,
        key_mode: str = "camera",
        logger: Callable[[str], None] | None = None,
    ) -> list[int]:
        """Apply ``camera,zoom`` entries from either old control convention.

        UDP v1 used a port-like key and mapped it with
        ``(key % initial_port) // 2``.  The WebRTC DataChannel used a direct
        camera index.  Both remain accepted explicitly.
        """

        if value is None:
            return []
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip("; \t\r\n")
        if not text:
            return []
        changed: list[int] = []
        for item in text.split(";"):
            if not item:
                continue
            try:
                raw_key, raw_zoom = (piece.strip() for piece in item.split(",", 1))
                key = int(raw_key)
                if not raw_zoom:
                    continue
                # Legacy messages use e.g. ``0,x1.5``.  Also accept a plain
                # number for the standalone protocol utility.
                numeric = raw_zoom[1:] if raw_zoom[0].isalpha() else raw_zoom
                zoom = float(numeric)
                if not math.isfinite(zoom) or zoom <= 0.0:
                    continue
                if key_mode == "port":
                    if self.initial_port <= 0:
                        continue
                    camera_idx = (key % self.initial_port) // 2
                elif key_mode == "camera":
                    camera_idx = key
                else:
                    raise ValueError(f"Unsupported zoom key mode: {key_mode}")
                if not 0 <= camera_idx < self.camera_num:
                    if logger:
                        logger(f"Invalid camera index {camera_idx}, total={self.camera_num}")
                    continue
                with self.lock:
                    if self.camera_zoom[camera_idx] != zoom:
                        self.camera_zoom[camera_idx] = zoom
                        changed.append(camera_idx)
            except (TypeError, ValueError, IndexError):
                # The old receive thread ignored malformed control after
                # logging its outer exception.  A bad command must never stop
                # VR or camera receive threads in the refactored path.
                continue
        return changed

    def movement_exists(self) -> bool:
        with self.lock:
            data = self.data
            if data is None or len(data) < 2:
                return False
            try:
                return bool(data[1]["GripTrigger"]) or abs(float(data[1]["Joystick"][1])) > 0.7
            except (KeyError, IndexError, TypeError, ValueError):
                return False

    def snapshot_vr(self) -> tuple[list[dict] | None, dict[str, dict], int]:
        with self.lock:
            return self.data, dict(self.hand_data), int(self.vr_input_timestamp_ns)
