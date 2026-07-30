"""Robotiq 2F-85 URCap socket adapter with explicit connection lifecycle."""

from __future__ import annotations

import math
import socket
import threading
from collections.abc import Callable
from typing import Protocol

from ...config.models import RobotConfig
from ...core.errors import LifecycleError, ModelValidationError


class _Socket(Protocol):
    def sendall(self, data: bytes) -> None: ...

    def recv(self, size: int) -> bytes: ...

    def close(self) -> None: ...


SocketFactory = Callable[[str, int, float], _Socket]


def _open_socket(host: str, port: int, timeout_s: float) -> _Socket:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.settimeout(timeout_s)
    connection.connect((host, port))
    return connection


class Robotiq2F85Gripper:
    """Persistent TCP control preserving the legacy 0..230 position mapping."""

    def __init__(
        self,
        host: str,
        *,
        port: int = 63352,
        max_width_m: float = 0.085,
        timeout_s: float = 0.02,
        socket_factory: SocketFactory = _open_socket,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ModelValidationError("Robotiq host must be a non-empty string")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ModelValidationError("Robotiq port must be within 1..65535")
        width = float(max_width_m)
        timeout = float(timeout_s)
        if not math.isfinite(width) or width <= 0:
            raise ModelValidationError("max_width_m must be positive and finite")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ModelValidationError("timeout_s must be positive and finite")
        self._host = host
        self._port = port
        self._max_width_m = width
        self._timeout_s = timeout
        self._socket_factory = socket_factory
        self._socket: _Socket | None = None
        self._started = False
        self._closed = False
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "robotiq_2f85"

    @property
    def max_width_m(self) -> float:
        return self._max_width_m

    def _new_socket(self) -> _Socket:
        return self._socket_factory(self._host, self._port, self._timeout_s)

    def _close_socket(self) -> None:
        connection, self._socket = self._socket, None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _exchange(self, command: str) -> str:
        if not self._started or self._socket is None:
            raise LifecycleError("Robotiq gripper has not been started")
        payload = (command.strip() + "\n").encode("ascii")
        for attempt in range(2):
            try:
                self._socket.sendall(payload)
                return self._socket.recv(1024).decode("ascii").strip()
            except (OSError, UnicodeError):
                self._close_socket()
                if attempt:
                    raise
                self._socket = self._new_socket()
        raise RuntimeError("Robotiq exchange retry ended unexpectedly")

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed Robotiq gripper")
            if self._started:
                raise LifecycleError("Robotiq gripper is already started")
            self._socket = self._new_socket()
            self._started = True
            try:
                self._exchange("SET SPE 255")
                self._exchange("GET SPE")
            except Exception:
                self._started = False
                self._close_socket()
                raise

    def _width_to_register(self, width_m: float) -> int:
        width = min(max(width_m, 0.0), self._max_width_m)
        return round((self._max_width_m - width) * 230 / self._max_width_m)

    def _register_to_width(self, register: int) -> float:
        clipped = min(max(register, 0), 230)
        return self._max_width_m * (230 - clipped) / 230

    def move(self, width_m: float) -> None:
        width = float(width_m)
        if not math.isfinite(width):
            raise ModelValidationError("gripper width must be finite")
        with self._lock:
            register = self._width_to_register(width)
            self._exchange(f"SET POS {register}")

    def open(self) -> None:
        self.move(self._max_width_m)

    def read_width(self) -> float:
        with self._lock:
            response = self._exchange("GET POS")
            parts = response.split()
            if len(parts) < 2:
                raise RuntimeError(f"invalid Robotiq position response: {response!r}")
            try:
                register = int(parts[-1])
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid Robotiq position response: {response!r}"
                ) from exc
            return self._register_to_width(register)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True
            self._close_socket()


def create_robotiq_2f85(config: RobotConfig) -> Robotiq2F85Gripper:
    """Create an unstarted gripper from the robot connection section."""

    if config.ip is None:
        raise LifecycleError("robot.ip must be configured before creating a Robotiq gripper")
    return Robotiq2F85Gripper(
        config.ip,
        max_width_m=config.gripper_max_width_m,
    )
