"""Legacy HD JPEG chunk protocol and a loss-tolerant frame assembler."""

from __future__ import annotations

import cv2
import numpy as np

from collections import deque
from dataclasses import dataclass, field
import socket
import struct
import time
from typing import Callable


HD_HEADER_FMT = "!IHHI"
HD_HEADER_SIZE = struct.calcsize(HD_HEADER_FMT)


@dataclass(frozen=True)
class JpegDatagram:
    """Decoded HD header plus its payload."""

    frame_id: int
    chunk_index: int
    chunk_count: int
    total_bytes: int
    payload: bytes

    @classmethod
    def unpack(cls, packet: bytes | bytearray | memoryview) -> "JpegDatagram":
        raw = bytes(packet)
        if len(raw) < HD_HEADER_SIZE:
            raise ValueError("JPEG datagram is shorter than its 12-byte header")
        frame_id, chunk_index, chunk_count, total_bytes = struct.unpack(
            HD_HEADER_FMT, raw[:HD_HEADER_SIZE]
        )
        if chunk_count <= 0:
            raise ValueError("JPEG chunk count must be positive")
        if chunk_index >= chunk_count:
            raise ValueError("JPEG chunk index is outside the frame")
        if total_bytes <= 0:
            raise ValueError("JPEG frame size must be positive")
        payload = raw[HD_HEADER_SIZE:]
        if not payload:
            raise ValueError("JPEG chunk payload is empty")
        if len(payload) > total_bytes:
            raise ValueError("JPEG chunk payload exceeds declared frame size")
        return cls(frame_id, chunk_index, chunk_count, total_bytes, payload)

    def pack(self) -> bytes:
        return struct.pack(
            HD_HEADER_FMT,
            int(self.frame_id) & 0xFFFFFFFF,
            int(self.chunk_index),
            int(self.chunk_count),
            int(self.total_bytes) & 0xFFFFFFFF,
        ) + bytes(self.payload)


@dataclass
class _Assembly:
    chunk_count: int
    total_bytes: int
    chunks: dict[int, bytes] = field(default_factory=dict)
    last_update: float = 0.0


class JpegChunkAssembler:
    """Reassemble out-of-order UDP chunks with duplicate/bad/expiry metrics.

    A completed frame id is remembered until ``max_completed`` is reached so a
    retransmitted complete frame cannot be emitted twice.  ``reset`` clears
    this state, which is needed when the Unity/PC session restarts and the
    sender counter begins again at zero.
    """

    def __init__(
        self,
        *,
        frame_timeout_s: float = 0.5,
        max_frame_bytes: int = 64 * 1024 * 1024,
        max_inflight: int = 32,
        max_completed: int = 128,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if frame_timeout_s <= 0.0:
            raise ValueError("frame_timeout_s must be positive")
        if max_frame_bytes <= 0 or max_inflight <= 0 or max_completed <= 0:
            raise ValueError("assembler capacities must be positive")
        self.frame_timeout_s = float(frame_timeout_s)
        self.max_frame_bytes = int(max_frame_bytes)
        self.max_inflight = int(max_inflight)
        self._max_completed = int(max_completed)
        self._clock = clock
        self._frames: dict[int, _Assembly] = {}
        self._completed: deque[int] = deque(maxlen=self._max_completed)
        self._completed_set: set[int] = set()
        self.stats: dict[str, int] = {
            "packets": 0,
            "frames": 0,
            "duplicates": 0,
            "malformed": 0,
            "expired": 0,
            "evicted": 0,
            "incomplete": 0,
            "restarts": 0,
        }

    def _remember_completed(self, frame_id: int) -> None:
        if len(self._completed) == self._completed.maxlen:
            old = self._completed.popleft()
            self._completed_set.discard(old)
        self._completed.append(frame_id)
        self._completed_set.add(frame_id)

    def expire(self, now: float | None = None) -> int:
        now = self._clock() if now is None else float(now)
        expired = [
            frame_id
            for frame_id, assembly in self._frames.items()
            if now - assembly.last_update >= self.frame_timeout_s
        ]
        for frame_id in expired:
            self._frames.pop(frame_id, None)
        self.stats["expired"] += len(expired)
        self.stats["incomplete"] += len(expired)
        return len(expired)

    def reset(self) -> None:
        self._frames.clear()
        self._completed.clear()
        self._completed_set.clear()
        self.stats["restarts"] += 1

    def feed(self, packet: bytes | bytearray | memoryview, now: float | None = None) -> bytes | None:
        """Feed one datagram and return a JPEG byte string only on completion."""

        now = self._clock() if now is None else float(now)
        self.expire(now)
        self.stats["packets"] += 1
        try:
            datagram = JpegDatagram.unpack(packet)
            if datagram.total_bytes > self.max_frame_bytes:
                raise ValueError("declared JPEG frame exceeds assembler capacity")
        except (TypeError, ValueError, struct.error):
            self.stats["malformed"] += 1
            return None

        frame_id = datagram.frame_id
        if frame_id in self._completed_set:
            self.stats["duplicates"] += 1
            return None

        assembly = self._frames.get(frame_id)
        if assembly is None:
            if len(self._frames) >= self.max_inflight:
                oldest = min(self._frames, key=lambda key: self._frames[key].last_update)
                self._frames.pop(oldest, None)
                self.stats["evicted"] += 1
                self.stats["incomplete"] += 1
            assembly = _Assembly(
                chunk_count=datagram.chunk_count,
                total_bytes=datagram.total_bytes,
                last_update=now,
            )
            self._frames[frame_id] = assembly
        elif (
            assembly.chunk_count != datagram.chunk_count
            or assembly.total_bytes != datagram.total_bytes
        ):
            self._frames.pop(frame_id, None)
            self.stats["malformed"] += 1
            return None

        previous = assembly.chunks.get(datagram.chunk_index)
        if previous is not None:
            if previous == datagram.payload:
                self.stats["duplicates"] += 1
            else:
                self.stats["malformed"] += 1
                self._frames.pop(frame_id, None)
            return None
        assembly.chunks[datagram.chunk_index] = datagram.payload
        assembly.last_update = now

        if len(assembly.chunks) != assembly.chunk_count:
            return None
        image = b"".join(assembly.chunks[index] for index in range(assembly.chunk_count))
        self._frames.pop(frame_id, None)
        if len(image) != assembly.total_bytes:
            self.stats["malformed"] += 1
            return None
        self._remember_completed(frame_id)
        self.stats["frames"] += 1
        return image


class JpegChunkSender:
    """Encode BGR frames and emit the frozen ``!IHHI`` chunk format."""

    def __init__(self, *, chunk_size: int = 60_000, quality: int = 50) -> None:
        if not 1 <= int(chunk_size) <= 65_535:
            raise ValueError("chunk_size must fit a UDP chunk and uint16 count")
        if not 0 <= int(quality) <= 100:
            raise ValueError("JPEG quality must be between 0 and 100")
        self.chunk_size = int(chunk_size)
        self.quality = int(quality)
        self._counters: dict[int, int] = {}

    def encode(self, frame_bgr: np.ndarray, *, quality: int | None = None) -> bytes:
        q = self.quality if quality is None else int(quality)
        if not 0 <= q <= 100:
            raise ValueError("JPEG quality must be between 0 and 100")
        ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            raise ValueError("OpenCV could not encode the frame as JPEG")
        return buffer.tobytes()

    def packets(
        self,
        frame_bgr: np.ndarray,
        *,
        cam_idx: int = 0,
        quality: int | None = None,
    ) -> tuple[int, list[bytes]]:
        image_bytes = self.encode(frame_bgr, quality=quality)
        if len(image_bytes) > 0xFFFFFFFF:
            raise ValueError("JPEG frame is too large for the legacy header")
        count = (len(image_bytes) + self.chunk_size - 1) // self.chunk_size
        if count > 0xFFFF:
            raise ValueError("JPEG chunk count does not fit uint16")
        counter = self._counters.get(int(cam_idx), 0)
        frame_id = counter & 0xFFFFFFFF
        self._counters[int(cam_idx)] = counter + 1
        packets = [
            JpegDatagram(
                frame_id=frame_id,
                chunk_index=index,
                chunk_count=count,
                total_bytes=len(image_bytes),
                payload=image_bytes[start : min(start + self.chunk_size, len(image_bytes))],
            ).pack()
            for index, start in enumerate(range(0, len(image_bytes), self.chunk_size))
        ]
        return frame_id, packets

    def send(
        self,
        frame_bgr: np.ndarray,
        sock: socket.socket,
        target: tuple[str, int],
        *,
        cam_idx: int = 0,
        quality: int | None = None,
    ) -> int:
        _frame_id, packets = self.packets(frame_bgr, cam_idx=cam_idx, quality=quality)
        for packet in packets:
            sock.sendto(packet, target)
        return len(packets)


class JpegDatagramReceiver:
    """Small real-socket helper used by loopback diagnostics."""

    def __init__(self, sock: socket.socket, assembler: JpegChunkAssembler | None = None) -> None:
        self.socket = sock
        self.assembler = assembler or JpegChunkAssembler()

    def recv(self, timeout_s: float = 0.1) -> bytes | None:
        self.socket.settimeout(timeout_s)
        try:
            packet, _address = self.socket.recvfrom(65_535)
        except TimeoutError:
            return None
        return self.assembler.feed(packet)
