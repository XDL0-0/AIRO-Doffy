"""Two-way UDP communication (Python ↔ Unity/VR).

Based on work by Youssef Elashry, refactored with thread-safe reads.
Licensed under Apache License 2.0.
"""

import socket
import threading


class UdpComms:
    def __init__(
        self,
        udp_ip: str,
        send_ip: str,
        port_tx: int,
        port_rx: int,
        enable_rx: bool = False,
        suppress_warnings: bool = True,
    ):
        self.udp_ip = udp_ip
        self.send_ip = send_ip
        self.udp_send_port = port_tx
        self.udp_rcv_port = port_rx
        self.enable_rx = enable_rx
        self.suppress_warnings = suppress_warnings

        self._rx_lock = threading.Lock()
        self._is_data_received = False
        self._data_rx = None

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((udp_ip, port_rx))

        if enable_rx:
            self._rx_thread = threading.Thread(
                target=self._read_udp_loop, daemon=True
            )
            self._rx_thread.start()

    def __del__(self):
        self.close()

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass

    def send(self, data: bytes | str) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._sock.sendto(data, (self.send_ip, self.udp_send_port))

    def read(self) -> str | None:
        """Return data received since the last call, or None."""
        with self._rx_lock:
            if self._is_data_received:
                self._is_data_received = False
                data = self._data_rx
                self._data_rx = None
                return data
        return None

    # ── internal ──────────────────────────────────────────────────────────

    def _receive_blocking(self) -> str | None:
        if not self.enable_rx:
            raise ValueError(
                "Attempting to receive data without enabling RX in constructor"
            )
        try:
            data, _ = self._sock.recvfrom(1024)
            return data.decode("utf-8")
        except OSError as e:
            is_windows_reset = hasattr(e, "winerror") and e.winerror == 10054
            is_linux_error = hasattr(e, "errno") and e.errno in [9, 104]
            if is_windows_reset or is_linux_error:
                if not self.suppress_warnings:
                    print("Not connected to the other application yet.")
                return None
            raise

    def _read_udp_loop(self) -> None:
        while True:
            data = self._receive_blocking()
            if data is not None:
                with self._rx_lock:
                    self._data_rx = data
                    self._is_data_received = True

    # ── backward-compatible aliases ───────────────────────────────────────
    SendData = send
    ReadReceivedData = read
    CloseSocket = close
