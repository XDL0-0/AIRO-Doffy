"""VL53L7CX BEAVER USB reader with a non-blocking latest-frame API.

Adapted from ~/Downloads/beaver.py (streaming protocol, magic 0x5A5A frames).
Changes:
  * No `utils` dependency (logging only).
  * The (bus, index) layout is arbitrary-length, not hard-coded to 9.
  * grid_width 8x8 is the default (firmware builds); 4x4 still accepted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import logging
import struct
import threading
import time
from typing import Iterable

import numpy as np

logger = logging.getLogger("beaver")

MAGIC = b"\x5a\x5a"
FRAME_HEADER = struct.Struct(">HHBB")
SENSOR_PREFIX = struct.Struct(">BBBHbI")
FRAME_HEADER_SIZE = FRAME_HEADER.size
SUPPORTED_GRID_WIDTHS = (4, 8)
LEGACY_GRID_WIDTH = 8
BUSES_PER_DEVICE = 2
DEFAULT_GRID_WIDTH = 8
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
    return SENSOR_PREFIX.size + 3 * grid_width * grid_width


def parse_sensor(raw: bytes, grid_width: int = LEGACY_GRID_WIDTH) -> dict[str, object]:
    zones = grid_width * grid_width
    expected_size = sensor_size(grid_width)
    if len(raw) != expected_size:
        raise ValueError(
            f"Beaver sensor record is {len(raw)} bytes; expected {expected_size}"
        )
    bus, index, valid, average, temperature, stream_count = (
        SENSOR_PREFIX.unpack_from(raw)
    )
    if bus >= BUSES_PER_DEVICE or index >= MAX_WIRE_SENSORS or valid > zones:
        raise ValueError("invalid Beaver sensor metadata")
    status_offset = SENSOR_PREFIX.size + zones*2
    return {
        "bus": bus,
        "index": index,
        "valid_count": valid,
        "average_mm": average,
        "temperature_c": temperature,
        "stream_count": stream_count,
        "grid_width": grid_width,
        "distance_mm": np.frombuffer(
            raw[SENSOR_PREFIX.size:status_offset], dtype=np.uint16
        )
        .reshape(grid_width, grid_width),
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
        sensors.append(parse_sensor(raw[start:start + record_size], grid_width))
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
                    self._consume_text(self.buffer[:-keep] if keep else self.buffer)
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


def find_ports() -> Iterable[str] | None:
    import serial.tools.list_ports
    found = []
    for port in serial.tools.list_ports.comports():
        if port.vid == 0x303A and port.pid == 0x1001:
            found.append(port.device)
    
    return found if len(found) > 0 else None


def open_port(device: str, baudrate: int = 115200):
    import serial

    port = serial.Serial(port=None, baudrate=baudrate, timeout=0.1, write_timeout=1)
    # Configure control lines before opening; asserted defaults reset ESP32-S3.
    port.dtr = False
    port.rts = False
    port.port = device
    port.open()
    return port


@dataclass(frozen=True)
class BeaverSnapshot:
    distance_mm: np.ndarray       # (n, g, g) uint16
    target_status: np.ndarray     # (n, g, g) uint8
    present: np.ndarray           # (n,) uint8
    valid_count: np.ndarray       # (n,) uint8
    average_mm: np.ndarray        # (n,) uint16
    temperature_c: np.ndarray     # (n,) int16
    stream_count: np.ndarray      # (n,) uint32
    sensor_layout: tuple[tuple[int, int], ...]
    grid_width: int = DEFAULT_GRID_WIDTH
    sequence: int = 0
    timestamp_ns: int = 0
    frame_count: int = 0
    lost_frames: int = 0
    connected: bool = False
    error: str = "waiting for Beaver USB data"

    def age_s(self) -> float:
        if not self.timestamp_ns:
            return float("inf")
        return max(0.0, (time.monotonic_ns() - self.timestamp_ns) / 1e9)


def empty_snapshot(
    sensor_layout: Iterable[tuple[int, int]],
    grid_width: int = DEFAULT_GRID_WIDTH,
) -> BeaverSnapshot:
    if grid_width not in SUPPORTED_GRID_WIDTHS:
        raise ValueError(f"Beaver grid_width must be one of {SUPPORTED_GRID_WIDTHS}")
    layout = tuple(sensor_layout)
    if not layout or len(set(layout)) != len(layout):
        raise ValueError("Beaver sensor_layout must contain unique (bus, index) IDs")
    count = len(layout)
    return BeaverSnapshot(
        distance_mm=np.zeros((count, grid_width, grid_width), dtype=np.uint16),
        target_status=np.full((count, grid_width, grid_width), 255, dtype=np.uint8),
        present=np.zeros((count,), dtype=np.uint8),
        valid_count=np.zeros((count,), dtype=np.uint8),
        average_mm=np.zeros((count,), dtype=np.uint16),
        temperature_c=np.zeros((count,), dtype=np.int16),
        stream_count=np.zeros((count,), dtype=np.uint32),
        sensor_layout=layout,
        grid_width=grid_width,
    )


class BeaverReader:
    """Owns the serial port on one thread and exposes an atomic copied snapshot."""

    def __init__(
        self,
        *,
        devices: Iterable[str] | None = None,
        baudrate: int = 115200,
        sensor_layout: Iterable[tuple[int, int]],
        grid_width: int = DEFAULT_GRID_WIDTH,
        reconnect_delay_s: float = 2.0,
        sync_buffer_size: int = 8,
    ) -> None:
        layout = tuple(sensor_layout)
        if not layout or len(set(layout)) != len(layout):
            raise ValueError("Beaver sensor_layout must contain unique (bus, index) IDs")
        if grid_width not in SUPPORTED_GRID_WIDTHS:
            raise ValueError(f"Beaver grid_width must be one of {SUPPORTED_GRID_WIDTHS}")
        if reconnect_delay_s <= 0:
            raise ValueError("Beaver timing values must be positive")
        self.devices = devices or None
        self.baudrate = int(baudrate)
        self.sensor_layout = layout
        self.grid_width = int(grid_width)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self._slot_by_sensor = {sensor: slot for slot, sensor in enumerate(layout)}
        self._lock = threading.Lock()
        self._snapshot = empty_snapshot(layout, self.grid_width)
        self._history: deque[BeaverSnapshot] = deque(maxlen=int(sync_buffer_size))
        self._serials = None
        self._frames = None
        self._thread: threading.Thread | None = None
        self._local_stop = threading.Event()

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

    def _publish_error(self, message: str) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, connected=False, error=message)

    def _publish_frame(
        self, device_id: int, frame: dict[str, object]
    ) -> None:
        self._frames[device_id] = frame
        if any(item is None for item in self._frames):
            return
        if device_id != 0:
            return
        frame_grid_width = int(frame["grid_width"])
        with self._lock:
            if frame_grid_width != self.grid_width:
                # First frame revealed the firmware's actual grid (4x4 vs 8x8);
                # re-size the snapshot so downstream consumers see it.
                logger.info(
                    "Beaver firmware grid is %dx%d; re-sizing snapshot "
                    "(config expected %d)",
                    frame_grid_width, frame_grid_width, self.grid_width,
                )
                self.grid_width = frame_grid_width
                self._snapshot = empty_snapshot(self.sensor_layout, frame_grid_width)
            snapshot = empty_snapshot(self.sensor_layout, frame_grid_width)

            # order frames (devices with only one bus last)
            has_only_one = [all(s["bus"] == 0 for s in f["sensors"]) for f in self._frames]
            ordered_frames = [x for _, x in sorted(zip(has_only_one, self._frames))]

            for i in range(len(ordered_frames)):
                device_frame = ordered_frames[i]
                for sensor in device_frame["sensors"]:
                    key = (int(sensor["bus"])+i*BUSES_PER_DEVICE, int(sensor["index"]))
                    slot = self._slot_by_sensor.get(key)
                    if slot is None:
                        continue
                    snapshot.distance_mm[slot] = sensor["distance_mm"]
                    snapshot.target_status[slot] = sensor["target_status"]
                    snapshot.present[slot] = 1
                    snapshot.valid_count[slot] = sensor["valid_count"]
                    snapshot.average_mm[slot] = sensor["average_mm"]
                    snapshot.temperature_c[slot] = sensor["temperature_c"]
                    snapshot.stream_count[slot] = sensor["stream_count"]
            self._frame_count += 1
            value = BeaverSnapshot(
                distance_mm=snapshot.distance_mm,
                target_status=snapshot.target_status,
                present=snapshot.present,
                valid_count=snapshot.valid_count,
                average_mm=snapshot.average_mm,
                temperature_c=snapshot.temperature_c,
                stream_count=snapshot.stream_count,
                sensor_layout=self.sensor_layout,
                grid_width=frame_grid_width,
                sequence=int(frame["sequence"]),
                timestamp_ns=time.monotonic_ns(),
                frame_count=self._frame_count,
                lost_frames=self._lost_frames,
                connected=True,
                error="",
            )
            self._snapshot = value
            self._history.append(value)

    def _stopped(self, external_stop: threading.Event | None) -> bool:
        return self._local_stop.is_set() or (
            external_stop is not None and external_stop.is_set()
        )

    def _wait(
        self, seconds: float, external_stop: threading.Event | None
    ) -> bool:
        deadline = time.monotonic() + seconds
        while not self._stopped(external_stop):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._local_stop.wait(min(remaining, 0.1))
        return True

    def run(self, stop_event: threading.Event | None = None) -> None:
        self._frame_count = 0
        self._lost_frames = 0
        last_grid_time = 0
        previous_sequence: int | None = None
        while not self._stopped(stop_event):
            devices = self.devices or find_ports()
            if devices is None:
                self._publish_error("Beaver ESP32-S3 USB port not found")
                self._wait(self.reconnect_delay_s, stop_event)
                continue

            try:
                serial_ports = []
                decoders = []
                self._frames = []
                for device in devices:
                    serial_port = open_port(device, self.baudrate)
                    serial_ports.append(serial_port)
                    decoders.append(FrameDecoder())
                    self._frames.append(None)
                self._serials = serial_ports
                logger.info("Beaver reader opened " + '[%s]' % ', '.join(map(str, devices)))
                while not self._stopped(stop_event) and self._serials is not None:
                    send_grid = False
                    time_passed = time.time()-last_grid_time
                    if time_passed > 1:
                        last_grid_time = time.time()
                        send_grid = True

                    for i in range(len(serial_ports)):
                        serial_port = serial_ports[i]
                        if send_grid:
                            serial_port.write(bytes([self.grid_width & 0xFF]))
                            serial_port.flush()
                        decoder = decoders[i]
                        waiting = serial_port.in_waiting
                        # Some USB serial backends briefly return None while the
                        # ESP32-S3 reconnects. Handle that like an empty buffer.
                        data = serial_port.read(max(1, int(waiting or 0)))
                        frames = decoder.feed(data)
                        for message in decoder.take_messages():
                            logger.debug("Beaver ESP: %s", message)
                        for frame in frames:
                            sequence = int(frame["sequence"])
                            if previous_sequence is not None:
                                expected = (previous_sequence + 1) & 0xFFFF
                                self._lost_frames += (sequence - expected) & 0xFFFF
                            previous_sequence = sequence
                            self._publish_frame(i, frame)
            except (OSError, PermissionError) as exc:
                self._publish_error(f"Beaver serial error: {exc}")
                logger.warning("Beaver serial error: %s", exc, exc_info=True)
            except Exception as exc:
                # pyserial is imported lazily, so keep this thread optional.
                self._publish_error(f"Beaver reader error: {exc}")
                logger.warning("Beaver reader error: %s", exc, exc_info=True)
            finally:
                serial_ports = self._serials
                if serial_ports is not None:
                    for serial_port in serial_ports:
                        if serial_port is not None:
                            try:
                                serial_port.close()
                            except Exception:
                                pass
                    self._serials = None
                
            self._wait(self.reconnect_delay_s, stop_event)

    def start(self, stop_event: threading.Event | None = None) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._local_stop.clear()
        self._thread = threading.Thread(
            target=self.run, args=(stop_event,), name="beaver-usb-reader", daemon=True
        )
        self._thread.start()
        return self._thread

    def close(self) -> None:
        self._local_stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            # The reader thread closes its own port in its finally block;
            # joining first avoids closing the port out from under it.
            self._thread.join(timeout=2.0)

        if self._serials is not None:
            for serial_port in self._serials:
                if serial_port is not None:
                    try:
                        serial_port.close()
                    except Exception:
                        pass
        self._serials = None
