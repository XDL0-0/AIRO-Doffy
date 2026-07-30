"""Tests for latest-only state channel policies and adapters."""

from __future__ import annotations

import threading
import time
import unittest

from airo_doffy.core import LifecycleError, ModelValidationError
from airo_doffy.streaming.state import (
    LatestStateReceiver,
    LatestStateSender,
    RealtimeStateChannel,
    UdpDiagnosticStateChannel,
    create_aiortc_realtime_state_channel,
    decode_state_packet,
    encode_state_packet,
)

from tests.unit.test_state_protocol import robot_state, vr_state


class _Channel:
    ordered = False
    max_retransmits = 0

    def __init__(self, *, block_first: bool = False) -> None:
        self.block_first = block_first
        self.entered = threading.Event()
        self.release = threading.Event()
        self.started = False
        self.closed = False
        self.sent: list[bytes] = []

    def start(self) -> None:
        self.started = True

    def send(self, data: bytes) -> None:
        if self.block_first and not self.sent:
            self.entered.set()
            self.release.wait(1.0)
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


class StateChannelTest(unittest.TestCase):
    def test_latest_only_overwrite_stale_rejection_and_metrics(self) -> None:
        channel = _Channel(block_first=True)
        sender = LatestStateSender(channel)
        self.assertIsInstance(channel, RealtimeStateChannel)
        with self.assertRaises(LifecycleError):
            sender.submit(robot_state(0))
        sender.start()
        self.assertTrue(sender.submit(robot_state(0)))
        self.assertTrue(channel.entered.wait(0.5))
        self.assertTrue(sender.submit(robot_state(1)))
        self.assertTrue(sender.submit(robot_state(2)))
        self.assertFalse(sender.submit(robot_state(2)))
        channel.release.set()
        deadline = time.monotonic() + 0.5
        while len(channel.sent) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        sender.close()
        sequences = [
            decode_state_packet(packet, receive_timestamp_ns=1).sequence
            for packet in channel.sent
        ]
        self.assertEqual(sequences, [0, 2])
        self.assertEqual(sender.metrics.submitted, 3)
        self.assertEqual(sender.metrics.sent, 2)
        self.assertEqual(sender.metrics.dropped_overwritten, 1)
        self.assertEqual(sender.metrics.rejected_stale, 1)
        self.assertTrue(channel.closed)

    def test_sender_requires_unordered_zero_retransmit_channel(self) -> None:
        channel = _Channel()
        channel.ordered = True
        with self.assertRaises(ModelValidationError):
            LatestStateSender(channel)
        channel.ordered = False
        channel.max_retransmits = 1
        with self.assertRaises(ModelValidationError):
            LatestStateSender(channel)

    def test_receiver_tracks_types_independently_and_accepts_uint32_wrap(self) -> None:
        receiver = LatestStateReceiver()
        for sequence in (0xFFFFFFFE, 0xFFFFFFFF, 0):
            self.assertTrue(receiver.accept(encode_state_packet(robot_state(sequence))))
        self.assertFalse(
            receiver.accept(encode_state_packet(robot_state(0xFFFFFFFF)))
        )
        self.assertTrue(receiver.accept(encode_state_packet(vr_state(5))))
        self.assertFalse(receiver.accept(b"bad"))
        self.assertEqual(receiver.latest_robot.sequence, 0)
        self.assertEqual(receiver.latest_vr.sequence, 5)
        self.assertEqual(receiver.metrics.accepted, 4)
        self.assertEqual(receiver.metrics.rejected_stale, 1)
        self.assertEqual(receiver.metrics.rejected_malformed, 1)

    def test_aiortc_factory_applies_exact_state_policy(self) -> None:
        class RawChannel:
            ordered = False
            maxRetransmits = 0

            def __init__(self) -> None:
                self.sent = []
                self.closed = False

            def send(self, data: bytes) -> None:
                self.sent.append(data)

            def close(self) -> None:
                self.closed = True

        class Peer:
            def __init__(self) -> None:
                self.call = None
                self.channel = RawChannel()

            def createDataChannel(self, label, **kwargs):
                self.call = (label, kwargs)
                return self.channel

        peer = Peer()
        channel = create_aiortc_realtime_state_channel(peer)
        self.assertEqual(
            peer.call,
            ("realtime_state", {"ordered": False, "maxRetransmits": 0}),
        )
        channel.start()
        channel.send(b"state")
        channel.close()
        self.assertEqual(peer.channel.sent, [b"state"])
        self.assertTrue(peer.channel.closed)

    def test_udp_diagnostic_target_lifecycle_and_metrics(self) -> None:
        class Socket:
            def __init__(self) -> None:
                self.calls = []
                self.closed = False

            def sendto(self, data, target):
                self.calls.append((data, target))
                return len(data)

            def close(self) -> None:
                self.closed = True

        udp_socket = Socket()
        channel = UdpDiagnosticStateChannel(
            "127.0.0.1",
            5005,
            socket_factory=lambda: udp_socket,
        )
        with self.assertRaises(LifecycleError):
            channel.send(b"packet")
        channel.start()
        channel.send(b"packet")
        channel.close()
        channel.close()
        self.assertEqual(udp_socket.calls, [(b"packet", ("127.0.0.1", 5005))])
        self.assertEqual(channel.metrics.packets_sent, 1)
        self.assertEqual(channel.metrics.bytes_sent, 6)
        self.assertTrue(udp_socket.closed)


if __name__ == "__main__":
    unittest.main()
