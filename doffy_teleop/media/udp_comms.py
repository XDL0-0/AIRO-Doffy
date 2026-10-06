"""Two-way UDP communication (Python ↔ Unity/VR).

Based on work by Youssef Elashry, refactored with thread-safe reads.
Licensed under Apache License 2.0.
"""

import socket
import threading
from collections import deque
import logging


logger = logging.getLogger(__name__)


class UdpComms:
    _RX_POLL_INTERVAL = 0.1
    _RX_JOIN_TIMEOUT = 1.0

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
        self._rx_queue: deque[str] = deque(maxlen=128)
        self._rx_stop = threading.Event()
        self._rx_thread: threading.Thread | None = None

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # A timeout makes shutdown reliable even on platforms where closing
            # a UDP socket from another thread does not wake recvfrom().
            self._sock.settimeout(self._RX_POLL_INTERVAL)
            self._sock.bind((udp_ip, port_rx))
        except BaseException:
            self._sock.close()
            raise

        if enable_rx:
            self._rx_thread = threading.Thread(
                target=self._read_udp_loop, daemon=True
            )
            try:
                self._rx_thread.start()
            except BaseException:
                self.close()
                raise

    def __del__(self):
        # __del__ can run after a partial constructor failure.
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        stop_event = getattr(self, "_rx_stop", None)
        if stop_event is not None:
            stop_event.set()

        sock = getattr(self, "_sock", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        thread = getattr(self, "_rx_thread", None)
        if (
            thread is not None
            and thread.ident is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=self._RX_JOIN_TIMEOUT)

    def _is_expected_receive_error(self, error: OSError) -> bool:
        is_windows_reset = getattr(error, "winerror", None) == 10054
        is_linux_error = getattr(error, "errno", None) in (9, 104)
        return is_windows_reset or is_linux_error

    def send(self, data: bytes | str) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._sock.sendto(data, (self.send_ip, self.udp_send_port))

    def read(self) -> str | None:
        """Return the oldest unread packet, or None if queue is empty."""
        with self._rx_lock:
            if self._rx_queue:
                return self._rx_queue.popleft()
        return None

    def read_all(self) -> list[str]:
        """Return all buffered packets (oldest first) and clear the queue."""
        with self._rx_lock:
            items = list(self._rx_queue)
            self._rx_queue.clear()
        return items

    # ── internal ──────────────────────────────────────────────────────────

    def _receive_blocking(self) -> str | None:
        if not self.enable_rx:
            raise ValueError(
                "Attempting to receive data without enabling RX in constructor"
            )
        try:
            data, _ = self._sock.recvfrom(4096)
            return data.decode("utf-8")
        except socket.timeout:
            return None
        except UnicodeDecodeError:
            if not self.suppress_warnings:
                logger.warning("Discarding UDP datagram containing invalid UTF-8")
            return None
        except OSError as e:
            if self._rx_stop.is_set() or self._is_expected_receive_error(e):
                if not self.suppress_warnings:
                    logger.warning("Not connected to the other application yet.")
                return None
            raise

    def _read_udp_loop(self) -> None:
        while not self._rx_stop.is_set():
            try:
                data = self._receive_blocking()
            except OSError:
                if self._rx_stop.is_set():
                    return
                raise
            if data is not None:
                with self._rx_lock:
                    self._rx_queue.append(data)

    # ── backward-compatible aliases ───────────────────────────────────────
    SendData = send
    ReadReceivedData = read
    CloseSocket = close
