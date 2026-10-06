"""Dedicated BODY telemetry UDP reception, independent of media devices."""

from __future__ import annotations

from collections import deque
import socket
import time

from doffy_teleop.protocol.body import BodyFrame, parse_body_packet


class BodyReceiver:
    """Single owner of diagnostic UDP8015; never binds teleop's control ports."""

    def __init__(self, bind_ip: str = "0.0.0.0", port: int = 8015):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((bind_ip, port))
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise
        self.address = self.socket.getsockname()
        self.latest: BodyFrame | None = None
        self.source: tuple[str, int] | None = None
        self.accepted = 0
        self.rejected = 0
        self._arrivals: deque[int] = deque(maxlen=240)

    def poll(self, *, limit: int = 128) -> BodyFrame | None:
        # Bound the amount of work per render so a burst cannot starve the GUI.
        for _ in range(limit):
            try:
                raw, source = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            received_ns = time.monotonic_ns()
            frame = parse_body_packet(raw, received_ns=received_ns)
            if frame is None:
                self.rejected += 1
                continue
            if self.latest is not None and source == self.source and frame.timestamp_ns < self.latest.timestamp_ns:
                continue
            self.latest, self.source = frame, source
            self.accepted += 1
            self._arrivals.append(received_ns)
        return self.latest

    def receive_hz(self, now_ns: int) -> float:
        recent = [stamp for stamp in self._arrivals if now_ns - stamp <= 1_000_000_000]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) * 1e9 / max(recent[-1] - recent[0], 1)

    def close(self) -> None:
        self.socket.close()


__all__ = ["BodyReceiver"]
