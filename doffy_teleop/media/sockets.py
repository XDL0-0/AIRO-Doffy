"""Shared legacy socket allocation with partial-startup cleanup."""

from __future__ import annotations

from typing import Any
import logging


class UDPSocketAllocator:
    """Allocate the fixed v1 socket numbering without owning a manager loop."""

    def __init__(self, socket_module: Any, logger: Any = None) -> None:
        self.socket_module = socket_module
        self.logger = logger or logging.getLogger(__name__)

    def allocate(
        self,
        *,
        pc_ip: str,
        vr_ip: str,
        base_port: int,
        camera_num: int,
        tactile_enabled: bool,
        tactile_port: int,
    ) -> tuple[dict[str, Any], int]:
        sockets: dict[str, Any] = {}
        port = int(base_port)

        def one(idx: int, enable_rx: bool) -> None:
            nonlocal port
            name = "socket_tactile" if idx == -1 else f"socket_{idx}"
            sockets[name] = self.socket_module.UdpComms(
                udp_ip=pc_ip,
                send_ip=vr_ip,
                port_tx=port,
                port_rx=port + 1,
                enable_rx=enable_rx,
                suppress_warnings=True,
            )
            self.logger.info(
                f"{name}: TX={port}, RX={port + 1}, enableRX={enable_rx}"
            )
            port += 2

        try:
            stream_count = min(max(0, int(camera_num)), 5)
            for idx in range(stream_count):
                one(idx, idx < 3)
            for idx in range(stream_count, max(3, stream_count)):
                one(idx, True)
            if tactile_enabled:
                port = int(tactile_port)
                one(-1, False)
            return sockets, port
        except BaseException:
            for sock in sockets.values():
                try:
                    sock.close()
                except Exception as exc:
                    self.logger.warning(f"Socket cleanup after allocation failure: {exc}")
            raise
