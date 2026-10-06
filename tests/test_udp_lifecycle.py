from __future__ import annotations

import socket
import time
import unittest
from unittest.mock import patch

from doffy_teleop.media.udp_comms import UdpComms


class UdpCommsLifecycleTests(unittest.TestCase):
    def make_receiver(self) -> tuple[UdpComms, int]:
        receiver = UdpComms(
            udp_ip="127.0.0.1",
            send_ip="127.0.0.1",
            port_tx=0,
            port_rx=0,
            enable_rx=True,
        )
        self.addCleanup(receiver.close)
        return receiver, receiver._sock.getsockname()[1]

    def test_close_stops_idle_receiver_and_is_idempotent(self) -> None:
        receiver, _ = self.make_receiver()
        thread = receiver._rx_thread
        self.assertIsNotNone(thread)

        # Give the worker time to enter recvfrom() with no traffic arriving.
        time.sleep(0.02)
        receiver.close()
        receiver.close()

        self.assertFalse(thread.is_alive())

    def test_close_after_datagram_burst_stops_receiver(self) -> None:
        receiver, port = self.make_receiver()
        thread = receiver._rx_thread
        self.assertIsNotNone(thread)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for index in range(64):
                sender.sendto(f"packet-{index}".encode(), ("127.0.0.1", port))

        receiver.close()
        self.assertFalse(thread.is_alive())

    def test_invalid_utf8_datagram_does_not_stop_receiver(self) -> None:
        receiver, port = self.make_receiver()
        thread = receiver._rx_thread
        self.assertIsNotNone(thread)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"\xff\xfe", ("127.0.0.1", port))
            sender.sendto(b"valid packet", ("127.0.0.1", port))

        deadline = time.monotonic() + 1.0
        packet = None
        while packet is None and time.monotonic() < deadline:
            packet = receiver.read()
            if packet is None:
                time.sleep(0.005)

        self.assertEqual(packet, "valid packet")
        self.assertTrue(thread.is_alive())

    def test_bind_failure_closes_socket_created_by_constructor(self) -> None:
        class BindFailSocket:
            closed = False

            def setsockopt(self, *_args) -> None:
                pass

            def settimeout(self, _timeout) -> None:
                pass

            def bind(self, _address) -> None:
                raise OSError("bind failed")

            def close(self) -> None:
                self.closed = True

        failed_socket = BindFailSocket()
        with (
            patch("doffy_teleop.media.udp_comms.socket.socket", return_value=failed_socket),
            self.assertRaisesRegex(OSError, "bind failed"),
        ):
            UdpComms("127.0.0.1", "127.0.0.1", 0, 12345, enable_rx=True)

        self.assertTrue(failed_socket.closed)


if __name__ == "__main__":
    unittest.main()
