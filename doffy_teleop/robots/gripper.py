"""Robot tool adapters."""

from __future__ import annotations

import socket

import numpy as np

import doffy_teleop.utils as utils
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85, rescale_range

class NullGripper:
    """Small gripper stand-in for manipulators without a configured gripper."""

    def __init__(self, max_width: float) -> None:
        self._width = float(max_width)
        self._max_width = float(max_width)

    def open(self) -> None:
        self._width = self._max_width

    def move(self, target_width_in_meters: float, *_, **__):
        self._set_target_width(target_width_in_meters)
        return None

    def _set_target_width(self, target_width_in_meters: float) -> None:
        self._width = float(np.clip(target_width_in_meters, 0.0, self._max_width))

    def get_current_width(self) -> float:
        return self._width

    def close(self) -> None:
        pass


class FastRobotiq2F85(Robotiq2F85):
    """Robotiq 2F-85 with a persistent TCP socket and non-blocking setpoint writes."""

    def __init__(self, host_ip: str, port: int = 63352, fingers_max_stroke=None):
        super().__init__(host_ip, port, fingers_max_stroke)
        self._persistent_sock = self._make_sock(host_ip, port)
        self._fast_communicate("SET SPE 255")
        verify = self._fast_communicate("GET SPE")
        utils.logger.info(f"FastRobotiq2F85 ready (SPE={verify})")

    def _make_sock(self, host: str, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(0.02)
        sock.connect((host, port))
        return sock

    def _fast_communicate(self, command: str) -> str:
        try:
            self._persistent_sock.sendall((command.strip() + "\n").encode())
            data = self._persistent_sock.recv(2**10)
            return data.decode()[:-1]
        except (socket.timeout, BrokenPipeError, ConnectionResetError, OSError) as exc:
            utils.logger.warning(f"Gripper socket error: {exc}, reconnecting...")
            try:
                self._persistent_sock.close()
            except Exception:
                pass
            self._persistent_sock = self._make_sock(self.host_ip, self.port)
            self._persistent_sock.sendall((command.strip() + "\n").encode())
            data = self._persistent_sock.recv(2**10)
            return data.decode()[:-1]

    def _set_target_width(self, target_width_in_meters: float) -> None:
        target_width_in_meters = np.clip(
            target_width_in_meters,
            self._gripper_specs.min_width,
            self._gripper_specs.max_width,
        )
        register_val = round(
            rescale_range(
                target_width_in_meters,
                self._gripper_specs.min_width,
                self._gripper_specs.max_width,
                230,
                0,
            )
        )
        self._fast_communicate(f"SET  POS {register_val}")

    def get_current_width(self) -> float:
        register_value = int(self._fast_communicate("GET POS").split(" ")[1])
        return rescale_range(
            register_value,
            0,
            230,
            self._gripper_specs.max_width,
            self._gripper_specs.min_width,
        )

    def close(self) -> None:
        try:
            self._persistent_sock.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
