"""Transport-free receiver tests for malformed, stale, wrap, and hand aggregation."""

from __future__ import annotations

import base64
import struct
import time
import unittest

from airo_doffy.core import HandSide, LifecycleError
from airo_doffy.devices.vr import (
    BINARY_V2_MAGIC,
    RawVRTransport,
    VRInputSource,
    VRReceiver,
)


def controller_packet(sequence: int) -> str:
    hand = (1, 2, 3, 0, 0, 0, 1, 0, 0, 0.5, 0.5, 0, 0, 0)
    return ",".join(str(value) for value in ("C", sequence, sequence * 1000, *hand, *hand))


def binary_hand(side: str, sequence: int) -> str:
    header = bytes((0x48, ord(side), 26, 0)) + struct.pack("<I", sequence)
    joints = b"".join(struct.pack("<fff", index, 0, 0) for index in range(26))
    return "HB," + base64.b64encode(header + joints).decode("ascii")


class _Transport:
    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def receive(self, _timeout_s: float) -> str | None:
        if self.messages:
            return self.messages.pop(0)
        raise RuntimeError("injected receive failure")

    def close(self) -> None:
        self.closed = True


class VrReceiverTest(unittest.TestCase):
    def test_push_lifecycle_malformed_stale_and_newer(self) -> None:
        receiver = VRReceiver()
        self.assertIsInstance(receiver, VRInputSource)
        with self.assertRaises(LifecycleError):
            receiver.read_latest()
        receiver.start()
        self.assertTrue(receiver.accept_message(controller_packet(2), receive_timestamp_ns=20))
        self.assertFalse(receiver.accept_message("malformed", receive_timestamp_ns=21))
        self.assertFalse(
            receiver.accept_message(BINARY_V2_MAGIC + b"short", receive_timestamp_ns=21)
        )
        self.assertFalse(receiver.accept_message(controller_packet(2), receive_timestamp_ns=22))
        self.assertFalse(receiver.accept_message(controller_packet(1), receive_timestamp_ns=23))
        self.assertTrue(receiver.accept_message(controller_packet(3), receive_timestamp_ns=24))
        latest = receiver.read_latest()
        self.assertEqual(latest.sequence, 1)
        self.assertEqual(latest.controllers[0].sequence, 3)
        self.assertEqual(receiver.stats.accepted, 2)
        self.assertEqual(receiver.stats.malformed, 2)
        self.assertEqual(receiver.stats.stale, 2)
        receiver.close()
        receiver.close()

    def test_uint32_wrap_is_accepted(self) -> None:
        receiver = VRReceiver()
        receiver.start()
        self.assertTrue(
            receiver.accept_message(controller_packet(2**32 - 1), receive_timestamp_ns=1)
        )
        self.assertTrue(receiver.accept_message(controller_packet(0), receive_timestamp_ns=2))
        self.assertEqual(receiver.read_latest().controllers[0].sequence, 0)
        receiver.close()

    def test_left_and_right_hand_packets_aggregate_at_equal_frame_id(self) -> None:
        receiver = VRReceiver()
        receiver.start()
        self.assertTrue(receiver.accept_message(binary_hand("L", 7), receive_timestamp_ns=10))
        self.assertTrue(receiver.accept_message(binary_hand("R", 7), receive_timestamp_ns=11))
        latest = receiver.read_latest()
        self.assertEqual(len(latest.hands), 2)
        self.assertEqual(
            tuple(hand.side for hand in latest.hands),
            (HandSide.LEFT, HandSide.RIGHT),
        )
        self.assertFalse(
            receiver.accept_message(binary_hand("L", 6), receive_timestamp_ns=12)
        )
        self.assertEqual(receiver.stats.stale, 1)
        receiver.close()

    def test_polled_transport_lifecycle_and_error_reporting(self) -> None:
        transport = _Transport([controller_packet(5)])
        self.assertIsInstance(transport, RawVRTransport)
        receiver = VRReceiver(transport, poll_timeout_s=0.001)
        receiver.start()
        deadline = time.monotonic() + 0.5
        while receiver.health_error is None and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(transport.started)
        self.assertEqual(receiver.read_latest().controllers[0].sequence, 5)
        self.assertIsInstance(receiver.health_error, RuntimeError)
        self.assertEqual(receiver.stats.transport_errors, 1)
        receiver.close()
        self.assertTrue(transport.closed)


if __name__ == "__main__":
    unittest.main()
