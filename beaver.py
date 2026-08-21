"""VL53L7CX Beaver USB driver with a non-blocking latest-frame API."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import struct
import threading
import time
from typing import Iterable

import numpy as np

import utils


MAGIC = b"\x5a\x5a"
FRAME_HEADER = struct.Struct(">HHBB")
SENSOR_PREFIX = struct.Struct(">BBBHbI")
FRAME_HEADER_SIZE = FRAME_HEADER.size
SUPPORTED_GRID_WIDTHS = (4, 8)
LEGACY_GRID_WIDTH = 8
DEFAULT_GRID_WIDTH = 4
SENSOR_SIZE = SENSOR_PREFIX.size + 64 + 64  # Legacy 8x8 compatibility constant.
MAX_WIRE_SENSORS = 18
VALID_STATUSES = (5, 9)


def grid_width_from_flags(flags: int) -> int:
    """Decode grid width; firmware predating this field sent zero for 8x8."""
    grid_width = int(flags) or LEGACY_GRID_WIDTH
    if grid_width not in SUPPORTED_GRID_WIDTHS:
        raise ValueError(f"unsupported Beaver grid width {grid_width}")
    return grid_width


def sensor_size(grid_width: int) -> int:
    return SENSOR_PREFIX.size + 2 * grid_width * grid_width


def _default_layout() -> tuple[tuple[int, int], ...]:
    return (
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    )


@dataclass(frozen=True)
class BeaverSnapshot:
    distance_mm: np.ndarray
    target_status: np.ndarray
    present: np.ndarray
    valid_count: np.ndarray
    average_mm: np.ndarray
    temperature_c: np.ndarray
    stream_count: np.ndarray
    sensor_layout: tuple[tuple[int, int], ...]
    grid_width: int = DEFAULT_GRID_WIDTH
    sequence: int = 0
    timestamp_ns: int = 0
    frame_count: int = 0
    lost_frames: int = 0
    connected: bool = False
    error: str = "waiting for Beaver USB data"

    def visualizer_payload(self, stale_after_s: float) -> dict[str, object]:
        age_s = (
            float("inf")
            if not self.timestamp_ns
            else max(0.0, (time.monotonic_ns() - self.timestamp_ns) / 1e9)
        )
        return {
            "distance_mm": self.distance_mm.copy(),
            "target_status": self.target_status.copy(),
            "present": self.present.copy(),
            "sensor_layout": self.sensor_layout,
            "grid_width": self.grid_width,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "frame_count": self.frame_count,
            "lost_frames": self.lost_frames,
            "connected": self.connected,
            "stale": age_s > stale_after_s,
            "error": self.error,
        }


def empty_snapshot(
    sensor_layout: Iterable[tuple[int, int]] | None = None,
    grid_width: int = DEFAULT_GRID_WIDTH,
) -> BeaverSnapshot:
    if grid_width not in SUPPORTED_GRID_WIDTHS:
        raise ValueError(f"Beaver grid_width must be one of {SUPPORTED_GRID_WIDTHS}")
    layout = tuple(_default_layout() if sensor_layout is None else sensor_layout)
    count = len(layout)
    return BeaverSnapshot(
        distance_mm=np.zeros((count, grid_width, grid_width), dtype=np.uint16),
        target_status=np.full(
            (count, grid_width, grid_width), 255, dtype=np.uint8
        ),
        present=np.zeros((count,), dtype=np.uint8),
        valid_count=np.zeros((count,), dtype=np.uint8),
        average_mm=np.zeros((count,), dtype=np.uint16),
        temperature_c=np.zeros((count,), dtype=np.int16),
        stream_count=np.zeros((count,), dtype=np.uint32),
        sensor_layout=layout,
        grid_width=grid_width,
    )


def parse_sensor(
    raw: bytes,
    grid_width: int = LEGACY_GRID_WIDTH,
) -> dict[str, object]:
    zones = grid_width * grid_width
    expected_size = sensor_size(grid_width)
    if len(raw) != expected_size:
        raise ValueError(
            f"Beaver sensor record is {len(raw)} bytes; expected {expected_size}"
        )
    bus, index, valid, average, temperature, stream_count = (
        SENSOR_PREFIX.unpack_from(raw)
    )
    if bus >= 2 or index >= MAX_WIRE_SENSORS or valid > zones:
        raise ValueError("invalid Beaver sensor metadata")
    status_offset = SENSOR_PREFIX.size + zones
    return {
        "bus": bus,
        "index": index,
        "valid_count": valid,
        "average_mm": average,
        "temperature_c": temperature,
        "stream_count": stream_count,
        "grid_width": grid_width,
        "distance_mm": np.frombuffer(
            raw[SENSOR_PREFIX.size:status_offset], dtype=np.uint8
        )
        .astype(np.uint16)
        .reshape(grid_width, grid_width)
        * 10,
        "target_status": np.frombuffer(
            raw[status_offset:status_offset + zones], dtype=np.uint8
        )
        .copy()
        .reshape(grid_width, grid_width),
    }


def parse_frame(raw: bytes) -> dict[str, object]:
    if len(raw) < FRAME_HEADER_SIZE:
        raise ValueError("short Beaver frame header")
    magic, sequence, sensor_count, flags = FRAME_HEADER.unpack_from(raw)
    if magic != 0x5A5A:
        raise ValueError("bad Beaver frame magic")
    if not 1 <= sensor_count <= MAX_WIRE_SENSORS:
        raise ValueError("invalid Beaver sensor count")
    grid_width = grid_width_from_flags(flags)
    record_size = sensor_size(grid_width)
    expected = FRAME_HEADER_SIZE + sensor_count * record_size
    if len(raw) != expected:
        raise ValueError(f"Beaver frame is {len(raw)} bytes; expected {expected}")
    sensors = []
    for sensor_index in range(sensor_count):
        start = FRAME_HEADER_SIZE + sensor_index * record_size
        sensors.append(
            parse_sensor(raw[start : start + record_size], grid_width)
        )
    return {
        "sequence": sequence,
        "flags": flags,
        "grid_width": grid_width,
        "sensors": sensors,
    }


class FrameDecoder:
    """Incremental decoder that tolerates boot logs and partial USB reads."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.text_buffer = bytearray()
        self.messages: list[str] = []

    def _consume_text(self, data: bytes | bytearray) -> None:
        self.text_buffer.extend(data)
        while b"\n" in self.text_buffer:
            line, _, remainder = self.text_buffer.partition(b"\n")
            self.text_buffer[:] = remainder
            message = line.strip(b"\r\x00").decode("utf-8", errors="replace")
            if message:
                self.messages.append(message)
        if len(self.text_buffer) > 1024:
            self.text_buffer.clear()

    def take_messages(self) -> list[str]:
        messages, self.messages = self.messages, []
        return messages

    def feed(self, data: bytes) -> list[dict[str, object]]:
        self.buffer.extend(data)
        frames = []
        while True:
            magic_at = self.buffer.find(MAGIC)
            if magic_at < 0:
                keep = 1 if self.buffer.endswith(MAGIC[:1]) else 0
                if len(self.buffer) > keep:
                    self._consume_text(
                        self.buffer[:-keep] if keep else self.buffer
                    )
                self.buffer[:] = MAGIC[:1] if keep else b""
                break
            if magic_at:
                self._consume_text(self.buffer[:magic_at])
                del self.buffer[:magic_at]
            if len(self.buffer) < FRAME_HEADER_SIZE:
                break
            sensor_count = self.buffer[4]
            if not 1 <= sensor_count <= MAX_WIRE_SENSORS:
                del self.buffer[0]
                continue
            try:
                grid_width = grid_width_from_flags(self.buffer[5])
            except ValueError:
                del self.buffer[0]
                continue
            frame_size = (
                FRAME_HEADER_SIZE + sensor_count * sensor_size(grid_width)
            )
            if len(self.buffer) < frame_size:
                break
            candidate = bytes(self.buffer[:frame_size])
            try:
                frame = parse_frame(candidate)
            except ValueError:
                del self.buffer[0]
                continue
            del self.buffer[:frame_size]
            frames.append(frame)
        maximum = FRAME_HEADER_SIZE + SENSOR_SIZE * MAX_WIRE_SENSORS
        if len(self.buffer) > maximum:
            self.buffer.clear()
        return frames


def find_port() -> str | None:
    import serial.tools.list_ports

    for port in serial.tools.list_ports.comports():
        if port.vid == 0x303A and port.pid == 0x1001:
            return port.device
    return None


def open_port(device: str, baudrate: int = 115200):
    import serial

    port = serial.Serial(
        port=None,
        baudrate=baudrate,
        timeout=0.1,
        write_timeout=1,
    )
    # Configure control lines before opening; asserted defaults reset ESP32-S3.
    port.dtr = False
    port.rts = False
    port.port = device
    port.open()
    return port


class BeaverReader:
    """Own the serial port on one thread and expose atomic copied snapshots."""

    def __init__(
        self,
        *,
        device: str | None = None,
        baudrate: int = 115200,
        sensor_layout: Iterable[tuple[int, int]] | None = None,
        grid_width: int = DEFAULT_GRID_WIDTH,
        stale_after_s: float = 1.0,
        reconnect_delay_s: float = 2.0,
        sync_buffer_size: int = 8,
    ) -> None:
        layout = tuple(_default_layout() if sensor_layout is None else sensor_layout)
        if len(layout) != 9 or len(set(layout)) != len(layout):
            raise ValueError("Beaver sensor_layout must contain 9 unique (bus, index) IDs")
        if grid_width not in SUPPORTED_GRID_WIDTHS:
            raise ValueError(
                f"Beaver grid_width must be one of {SUPPORTED_GRID_WIDTHS}"
            )
        if stale_after_s <= 0 or reconnect_delay_s <= 0:
            raise ValueError("Beaver timing values must be positive")
        if (
            isinstance(sync_buffer_size, bool)
            or not isinstance(sync_buffer_size, (int, np.integer))
            or sync_buffer_size < 1
        ):
            raise ValueError("sync_buffer_size must be a positive integer")
        self.device = device or None
        self.baudrate = int(baudrate)
        self.sensor_layout = layout
        self.grid_width = int(grid_width)
        self.stale_after_s = float(stale_after_s)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self._slot_by_sensor = {sensor: slot for slot, sensor in enumerate(layout)}
        self._lock = threading.Lock()
        self._snapshot = empty_snapshot(layout, self.grid_width)
        self._history: deque[BeaverSnapshot] = deque(maxlen=int(sync_buffer_size))
        self._serial = None
        self._thread: threading.Thread | None = None
        self._local_stop = threading.Event()

    @classmethod
    def from_config(cls, cfg) -> "BeaverReader":
        return cls(
            device=getattr(cfg, "BEAVER_PORT", None),
            baudrate=cfg.BEAVER_BAUDRATE,
            sensor_layout=cfg.BEAVER_SENSOR_LAYOUT,
            grid_width=cfg.BEAVER_GRID_WIDTH,
            stale_after_s=cfg.BEAVER_STALE_AFTER_S,
            reconnect_delay_s=cfg.BEAVER_RECONNECT_DELAY_S,
            sync_buffer_size=cfg.SENSOR_SYNC_BUFFER_SIZE,
        )

    @staticmethod
    def _copy_snapshot(value: BeaverSnapshot) -> BeaverSnapshot:
        return replace(
            value,
            distance_mm=value.distance_mm.copy(),
            target_status=value.target_status.copy(),
            present=value.present.copy(),
            valid_count=value.valid_count.copy(),
            average_mm=value.average_mm.copy(),
            temperature_c=value.temperature_c.copy(),
            stream_count=value.stream_count.copy(),
        )

    def snapshot(self) -> BeaverSnapshot:
        with self._lock:
            return self._copy_snapshot(self._snapshot)

    def snapshot_nearest(self, reference_timestamp_ns: int) -> BeaverSnapshot:
        """Return the buffered frame nearest to a monotonic reference time."""
        reference_timestamp_ns = int(reference_timestamp_ns)
        with self._lock:
            value = (
                min(
                    self._history,
                    key=lambda frame: abs(
                        frame.timestamp_ns - reference_timestamp_ns
                    ),
                )
                if self._history
                else self._snapshot
            )
            return self._copy_snapshot(value)

    def visualizer_payload(self) -> dict[str, object]:
        return self.snapshot().visualizer_payload(self.stale_after_s)

    def _publish_error(self, message: str) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                connected=False,
                error=message,
            )

    def _publish_frame(
        self,
        frame: dict[str, object],
        frame_count: int,
        lost_frames: int,
    ) -> None:
        frame_grid_width = int(frame["grid_width"])
        if frame_grid_width != self.grid_width:
            raise ValueError(
                "Beaver firmware grid is "
                f"{frame_grid_width}x{frame_grid_width}, but config expects "
                f"{self.grid_width}x{self.grid_width}"
            )
        snapshot = empty_snapshot(self.sensor_layout, self.grid_width)
        distance = snapshot.distance_mm
        status = snapshot.target_status
        present = snapshot.present
        valid = snapshot.valid_count
        average = snapshot.average_mm
        temperature = snapshot.temperature_c
        stream_count = snapshot.stream_count
        for sensor in frame["sensors"]:
            key = (int(sensor["bus"]), int(sensor["index"]))
            slot = self._slot_by_sensor.get(key)
            if slot is None:
                continue
            distance[slot] = sensor["distance_mm"]
            status[slot] = sensor["target_status"]
            present[slot] = 1
            valid[slot] = sensor["valid_count"]
            average[slot] = sensor["average_mm"]
            temperature[slot] = sensor["temperature_c"]
            stream_count[slot] = sensor["stream_count"]
        value = BeaverSnapshot(
            distance_mm=distance,
            target_status=status,
            present=present,
            valid_count=valid,
            average_mm=average,
            temperature_c=temperature,
            stream_count=stream_count,
            sensor_layout=self.sensor_layout,
            grid_width=self.grid_width,
            sequence=int(frame["sequence"]),
            timestamp_ns=time.monotonic_ns(),
            frame_count=frame_count,
            lost_frames=lost_frames,
            connected=True,
            error="",
        )
        with self._lock:
            self._snapshot = value
            self._history.append(value)

    def _stopped(self, external_stop: threading.Event | None) -> bool:
        return self._local_stop.is_set() or (
            external_stop is not None and external_stop.is_set()
        )

    def _wait(
        self,
        seconds: float,
        external_stop: threading.Event | None,
    ) -> bool:
        deadline = time.monotonic() + seconds
        while not self._stopped(external_stop):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._local_stop.wait(min(remaining, 0.1))
        return True

    def run(self, stop_event: threading.Event | None = None) -> None:
        frame_count = 0
        lost_frames = 0
        previous_sequence: int | None = None
        while not self._stopped(stop_event):
            device = self.device or find_port()
            if device is None:
                self._publish_error("Beaver ESP32-S3 USB port not found")
                self._wait(self.reconnect_delay_s, stop_event)
                continue
            try:
                serial_port = open_port(device, self.baudrate)
                self._serial = serial_port
                utils.logger.info("Beaver reader opened %s", device)
                decoder = FrameDecoder()
                while not self._stopped(stop_event):
                    waiting = serial_port.in_waiting
                    # Some USB serial backends briefly return None while the
                    # ESP32-S3 reconnects. Handle that like an empty buffer.
                    data = serial_port.read(max(1, int(waiting or 0)))
                    frames = decoder.feed(data)
                    for message in decoder.take_messages():
                        utils.logger.info("Beaver ESP: %s", message)
                    for frame in frames:
                        sequence = int(frame["sequence"])
                        if previous_sequence is not None:
                            expected = (previous_sequence + 1) & 0xFFFF
                            lost_frames += (sequence - expected) & 0xFFFF
                        previous_sequence = sequence
                        frame_count += 1
                        self._publish_frame(frame, frame_count, lost_frames)
            except (OSError, PermissionError) as exc:
                self._publish_error(f"Beaver serial error: {exc}")
                utils.logger.warning("Beaver serial error: %s", exc)
            except Exception as exc:
                # pyserial is imported lazily, so keep this thread optional.
                self._publish_error(f"Beaver reader error: {exc}")
                utils.logger.warning("Beaver reader error: %s", exc)
            finally:
                serial_port = self._serial
                self._serial = None
                if serial_port is not None:
                    try:
                        serial_port.close()
                    except Exception:
                        pass
            self._wait(self.reconnect_delay_s, stop_event)

    def start(
        self,
        stop_event: threading.Event | None = None,
    ) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._local_stop.clear()
        self._thread = threading.Thread(
            target=self.run,
            args=(stop_event,),
            name="beaver-usb-reader",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def close(self) -> None:
        self._local_stop.set()
        serial_port = self._serial
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
