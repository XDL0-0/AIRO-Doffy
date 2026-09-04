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
BUSES_PER_DEVICE = 2
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


def sensor_size(grid_width: int, is_16bit: bool = True) -> int:
    bytes_per_zone = 3 if is_16bit else 2
    return SENSOR_PREFIX.size + bytes_per_zone * grid_width * grid_width


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

    def visualizer_payload(self, stale_after_s: float) -> dict[str, object]:
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
            "stale": self.age_s() > stale_after_s,
            "error": self.error,
        }


def empty_snapshot(
    sensor_layout: Iterable[tuple[int, int]] | None = None,
    grid_width: int = DEFAULT_GRID_WIDTH,
) -> BeaverSnapshot:
    if grid_width not in SUPPORTED_GRID_WIDTHS:
        raise ValueError(f"Beaver grid_width must be one of {SUPPORTED_GRID_WIDTHS}")
    layout = tuple(_default_layout() if sensor_layout is None else sensor_layout)
    if not layout or len(set(layout)) != len(layout):
        raise ValueError("Beaver sensor_layout must contain unique (bus, index) IDs")
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
    expected_size_16 = SENSOR_PREFIX.size + 3 * zones
    expected_size_8 = SENSOR_PREFIX.size + 2 * zones
    if len(raw) == expected_size_16:
        is_16bit = True
    elif len(raw) == expected_size_8:
        is_16bit = False
    else:
        raise ValueError(
            f"Beaver sensor record is {len(raw)} bytes; expected {expected_size_16} (16-bit) or {expected_size_8} (8-bit)"
        )
    bus, index, valid, average, temperature, stream_count = (
        SENSOR_PREFIX.unpack_from(raw)
    )
    if bus >= BUSES_PER_DEVICE * 2 or index >= MAX_WIRE_SENSORS or valid > zones:
        raise ValueError("invalid Beaver sensor metadata")

    if is_16bit:
        status_offset = SENSOR_PREFIX.size + zones * 2
        distance_mm = np.frombuffer(
            raw[SENSOR_PREFIX.size:status_offset], dtype=np.uint16
        ).reshape(grid_width, grid_width)
    else:
        status_offset = SENSOR_PREFIX.size + zones
        distance_mm = (
            np.frombuffer(raw[SENSOR_PREFIX.size:status_offset], dtype=np.uint8)
            .astype(np.uint16)
            .reshape(grid_width, grid_width)
            * 10
        )

    target_status = (
        np.frombuffer(raw[status_offset:status_offset + zones], dtype=np.uint8)
        .copy()
        .reshape(grid_width, grid_width)
    )

    return {
        "bus": bus,
        "index": index,
        "valid_count": valid,
        "average_mm": average,
        "temperature_c": temperature,
        "stream_count": stream_count,
        "grid_width": grid_width,
        "distance_mm": distance_mm,
        "target_status": target_status,
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
    zones = grid_width * grid_width
    rec_size_16 = SENSOR_PREFIX.size + 3 * zones
    rec_size_8 = SENSOR_PREFIX.size + 2 * zones
    exp_16 = FRAME_HEADER_SIZE + sensor_count * rec_size_16
    exp_8 = FRAME_HEADER_SIZE + sensor_count * rec_size_8

    if len(raw) == exp_16:
        record_size = rec_size_16
    elif len(raw) == exp_8:
        record_size = rec_size_8
    else:
        raise ValueError(
            f"Beaver frame is {len(raw)} bytes; expected {exp_16} (16-bit) or {exp_8} (8-bit)"
        )
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

            zones = grid_width * grid_width
            size_16 = FRAME_HEADER_SIZE + sensor_count * (
                SENSOR_PREFIX.size + 3 * zones
            )
            size_8 = FRAME_HEADER_SIZE + sensor_count * (
                SENSOR_PREFIX.size + 2 * zones
            )

            matched_size = None
            frame = None

            # Check 16-bit frame first (higher precision format)
            if len(self.buffer) >= size_16:
                try:
                    frame = parse_frame(bytes(self.buffer[:size_16]))
                    matched_size = size_16
                except ValueError:
                    frame = None

            # Fallback to check 8-bit frame (legacy format)
            if matched_size is None and len(self.buffer) >= size_8:
                try:
                    cand_frame = parse_frame(bytes(self.buffer[:size_8]))
                    if len(self.buffer) >= size_8 + 2:
                        if self.buffer[size_8 : size_8 + 2] == MAGIC:
                            matched_size = size_8
                            frame = cand_frame
                    elif len(self.buffer) < size_16:
                        matched_size = size_8
                        frame = cand_frame
                except ValueError:
                    frame = None

            if matched_size is None:
                if len(self.buffer) < size_16:
                    break
                del self.buffer[0]
                continue

            del self.buffer[:matched_size]
            frames.append(frame)

        maximum = (
            FRAME_HEADER_SIZE + (SENSOR_PREFIX.size + 3 * 64) * MAX_WIRE_SENSORS
        )
        if len(self.buffer) > maximum:
            self.buffer.clear()
        return frames


def find_ports() -> list[str] | None:
    import serial.tools.list_ports

    found = []
    for port in serial.tools.list_ports.comports():
        if port.vid == 0x303A and port.pid == 0x1001:
            found.append(port.device)
    return found if len(found) > 0 else None


def find_port() -> str | None:
    ports = find_ports()
    return ports[0] if ports else None


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
    """Own serial port(s) on one thread and expose atomic copied snapshots."""

    def __init__(
        self,
        *,
        device: str | None = None,
        devices: Iterable[str] | None = None,
        baudrate: int = 115200,
        sensor_layout: Iterable[tuple[int, int]] | None = None,
        grid_width: int = DEFAULT_GRID_WIDTH,
        stale_after_s: float = 1.0,
        reconnect_delay_s: float = 2.0,
        sync_buffer_size: int = 8,
        simulate_8bit: bool = False,
    ) -> None:
        layout = tuple(_default_layout() if sensor_layout is None else sensor_layout)
        if not layout or len(set(layout)) != len(layout):
            raise ValueError(
                "Beaver sensor_layout must contain unique (bus, index) IDs"
            )
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

        if devices is not None:
            self.devices = list(devices)
            self.device = self.devices[0] if self.devices else None
        elif device is not None:
            self.devices = [device]
            self.device = device
        else:
            self.devices = None
            self.device = None

        self.baudrate = int(baudrate)
        self.sensor_layout = layout
        self.grid_width = int(grid_width)
        self.stale_after_s = float(stale_after_s)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self.simulate_8bit = bool(simulate_8bit)
        self._slot_by_sensor = {sensor: slot for slot, sensor in enumerate(layout)}
        self._lock = threading.Lock()
        self._snapshot = empty_snapshot(layout, self.grid_width)
        self._history: deque[BeaverSnapshot] = deque(
            maxlen=int(sync_buffer_size)
        )
        self._serials: list | None = None
        self._frames: list | None = None
        self._frame_count = 0
        self._lost_frames = 0
        self._thread: threading.Thread | None = None
        self._local_stop = threading.Event()

    @property
    def _serial(self):
        """Compatibility property for single serial access."""
        return self._serials[0] if self._serials else None

    @_serial.setter
    def _serial(self, val):
        if val is None:
            self._serials = None
        elif isinstance(val, list):
            self._serials = val
        else:
            self._serials = [val]

    @classmethod
    def from_config(cls, cfg) -> "BeaverReader":
        ports = getattr(cfg, "BEAVER_PORTS", None)
        port = getattr(cfg, "BEAVER_PORT", None)
        simulate_8bit = getattr(cfg, "BEAVER_SIMULATE_8BIT", False)
        return cls(
            device=port,
            devices=ports,
            baudrate=cfg.BEAVER_BAUDRATE,
            sensor_layout=cfg.BEAVER_SENSOR_LAYOUT,
            grid_width=cfg.BEAVER_GRID_WIDTH,
            stale_after_s=cfg.BEAVER_STALE_AFTER_S,
            reconnect_delay_s=cfg.BEAVER_RECONNECT_DELAY_S,
            sync_buffer_size=cfg.SENSOR_SYNC_BUFFER_SIZE,
            simulate_8bit=simulate_8bit,
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
        frame_or_device_id: dict[str, object] | int,
        frame: dict[str, object] | int | None = None,
        frame_count: int | None = None,
        lost_frames: int | None = None,
    ) -> None:
        if isinstance(frame_or_device_id, int):
            device_id = frame_or_device_id
            frame_obj: dict[str, object] = frame  # type: ignore
            if self._frames is None or len(self._frames) <= device_id:
                frames_len = (
                    len(self.devices) if self.devices else device_id + 1
                )
                self._frames = [None] * max(frames_len, device_id + 1)
            self._frames[device_id] = frame_obj
            if any(item is None for item in self._frames):
                return
            if device_id != 0:
                return
            frames_to_publish = self._frames
            target_frame_for_meta = self._frames[0]
        else:
            frame_obj = frame_or_device_id
            if isinstance(frame, (int, np.integer)):
                self._frame_count = int(frame)
            elif frame_count is not None:
                self._frame_count = int(frame_count)
            if lost_frames is not None:
                self._lost_frames = int(lost_frames)
            frames_to_publish = [frame_obj]
            target_frame_for_meta = frame_obj

        frame_grid_width = int(target_frame_for_meta["grid_width"])
        with self._lock:
            if frame_grid_width != self.grid_width:
                utils.logger.info(
                    "Beaver firmware grid is %dx%d; re-sizing snapshot "
                    "(config expected %d)",
                    frame_grid_width,
                    frame_grid_width,
                    self.grid_width,
                )
                self.grid_width = frame_grid_width
                self._snapshot = empty_snapshot(
                    self.sensor_layout, frame_grid_width
                )

            snapshot = empty_snapshot(self.sensor_layout, frame_grid_width)
            distance = snapshot.distance_mm
            status = snapshot.target_status
            present = snapshot.present
            valid = snapshot.valid_count
            average = snapshot.average_mm
            temperature = snapshot.temperature_c
            stream_count = snapshot.stream_count

            # Order frames (devices with only one bus last)
            ordered_frames = sorted(
                frames_to_publish,
                key=lambda f: 1 if all(s["bus"] == 0 for s in f["sensors"]) else 0,
            )

            for i, device_frame in enumerate(ordered_frames):
                for sensor in device_frame["sensors"]:
                    key = (
                        int(sensor["bus"]) + i * BUSES_PER_DEVICE,
                        int(sensor["index"]),
                    )
                    slot = self._slot_by_sensor.get(key)
                    if slot is None:
                        continue
                    dist = sensor["distance_mm"]
                    if self.simulate_8bit:
                        dist = (
                            np.clip(dist // 10, 0, 255)
                            .astype(np.uint16)
                            * 10
                        )
                    distance[slot] = dist
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
                grid_width=frame_grid_width,
                sequence=int(target_frame_for_meta["sequence"]),
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
        self._frame_count = 0
        self._lost_frames = 0
        last_grid_time = 0.0
        previous_sequence: int | None = None

        while not self._stopped(stop_event):
            devices = self.devices or find_ports()
            if not devices:
                self._publish_error("Beaver ESP32-S3 USB port not found")
                self._wait(self.reconnect_delay_s, stop_event)
                continue

            try:
                serial_ports = []
                decoders = []
                self._frames = [None] * len(devices)
                for dev in devices:
                    serial_ports.append(open_port(dev, self.baudrate))
                    decoders.append(FrameDecoder())
                self._serials = serial_ports
                utils.logger.info(
                    "Beaver reader opened [%s]", ", ".join(map(str, devices))
                )

                while not self._stopped(stop_event) and self._serials is not None:
                    send_grid = False
                    time_passed = time.time() - last_grid_time
                    if time_passed > 1.0:
                        last_grid_time = time.time()
                        send_grid = True

                    for i, serial_port in enumerate(serial_ports):
                        if send_grid and hasattr(serial_port, "write"):
                            try:
                                serial_port.write(bytes([self.grid_width & 0xFF]))
                                if hasattr(serial_port, "flush"):
                                    serial_port.flush()
                            except Exception:
                                pass
                        decoder = decoders[i]
                        waiting = serial_port.in_waiting
                        data = serial_port.read(max(1, int(waiting or 0)))
                        frames = decoder.feed(data)
                        for message in decoder.take_messages():
                            utils.logger.info("Beaver ESP: %s", message)
                        for frame in frames:
                            sequence = int(frame["sequence"])
                            if previous_sequence is not None:
                                expected = (previous_sequence + 1) & 0xFFFF
                                self._lost_frames += (
                                    (sequence - expected) & 0xFFFF
                                )
                            previous_sequence = sequence
                            self._frame_count += 1
                            if len(devices) > 1:
                                self._publish_frame(i, frame)
                            else:
                                self._publish_frame(
                                    frame,
                                    self._frame_count,
                                    self._lost_frames,
                                )
            except (OSError, PermissionError) as exc:
                self._publish_error(f"Beaver serial error: {exc}")
                utils.logger.warning("Beaver serial error: %s", exc)
            except Exception as exc:
                if not self._stopped(stop_event):
                    self._publish_error(f"Beaver reader error: {exc}")
                    utils.logger.warning("Beaver reader error: %s", exc, exc_info=True)
            finally:
                serial_ports = self._serials
                self._serials = None
                if serial_ports is not None:
                    for sp in serial_ports:
                        try:
                            sp.close()
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
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

        serial_ports = self._serials
        self._serials = None
        if serial_ports is not None:
            for sp in serial_ports:
                try:
                    sp.close()
                except Exception:
                    pass
