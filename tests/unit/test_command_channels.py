"""Tests for reliable command delivery, ACK matching, and deduplication."""

from __future__ import annotations

import threading
import unittest

from airo_doffy.config import CommandTransportConfig
from airo_doffy.core import (
    CommandTimeoutError,
    LifecycleError,
    ModelValidationError,
    RuntimeEvent,
    RuntimeEventSeverity,
    RuntimeEventType,
)
from airo_doffy.streaming.commands import (
    CommandAcknowledgement,
    CommandAckStatus,
    ReliableCommandChannel,
    ReliableCommandReceiver,
    ReliableCommandSender,
    create_aiortc_reliable_command_channel,
    decode_acknowledgement,
    encode_acknowledgement,
    encode_command,
)

from tests.unit.test_command_protocol import command


class _Channel:
    ordered = True
    max_retransmits = None
    max_packet_lifetime = None

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.sent: list[bytes] = []
        self.on_send = None

    def start(self) -> None:
        self.started = True

    def send(self, data: bytes) -> None:
        self.sent.append(data)
        if self.on_send is not None:
            self.on_send(data)

    def close(self) -> None:
        self.closed = True


class _Dispatcher:
    def __init__(
        self,
        *,
        kind: RuntimeEventType = RuntimeEventType.COMMAND_ACCEPTED,
        severity: RuntimeEventSeverity = RuntimeEventSeverity.INFO,
    ) -> None:
        self.kind = kind
        self.severity = severity
        self.commands = []

    def dispatch(self, incoming):
        self.commands.append(incoming)
        return RuntimeEvent(
            kind=self.kind,
            sequence=len(self.commands),
            timestamp_ns=100 + len(self.commands),
            severity=self.severity,
            component="test",
            message="routed",
            command_id=incoming.command_id,
        )


class ReliableCommandSenderTest(unittest.TestCase):
    def test_ack_is_matched_to_waiting_command(self) -> None:
        channel = _Channel()
        sender = ReliableCommandSender(channel, CommandTransportConfig())
        self.assertIsInstance(channel, ReliableCommandChannel)
        sender.start()
        outgoing = command()
        expected = CommandAcknowledgement(
            command_id=outgoing.command_id,
            command_sequence=outgoing.sequence,
            timestamp_ns=456,
            status=CommandAckStatus.ACCEPTED,
        )
        channel.on_send = lambda _data: sender.accept_acknowledgement(
            encode_acknowledgement(expected)
        )
        self.assertEqual(sender.send(outgoing), expected)
        self.assertEqual(sender.metrics.commands_sent, 1)
        self.assertEqual(sender.metrics.acknowledgements_received, 1)
        sender.close()
        self.assertTrue(channel.closed)

    def test_timeout_and_late_ack_are_observable(self) -> None:
        channel = _Channel()
        sender = ReliableCommandSender(
            channel,
            CommandTransportConfig(ack_timeout_s=0.005),
        )
        sender.start()
        outgoing = command()
        with self.assertRaises(CommandTimeoutError):
            sender.send(outgoing)
        late = CommandAcknowledgement(
            command_id=outgoing.command_id,
            command_sequence=outgoing.sequence,
            timestamp_ns=456,
            status=CommandAckStatus.ACCEPTED,
        )
        self.assertFalse(
            sender.accept_acknowledgement(encode_acknowledgement(late))
        )
        self.assertFalse(sender.accept_acknowledgement(b"bad"))
        self.assertEqual(sender.metrics.timeouts, 1)
        self.assertEqual(sender.metrics.unexpected_acknowledgements, 1)
        self.assertEqual(sender.metrics.malformed_acknowledgements, 1)
        sender.close()

    def test_close_unblocks_waiting_sender(self) -> None:
        channel = _Channel()
        sent = threading.Event()
        channel.on_send = lambda _data: sent.set()
        sender = ReliableCommandSender(
            channel,
            CommandTransportConfig(ack_timeout_s=1),
        )
        sender.start()
        errors = []

        def wait_for_ack() -> None:
            try:
                sender.send(command())
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=wait_for_ack)
        thread.start()
        self.assertTrue(sent.wait(0.5))
        sender.close()
        thread.join(0.5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], LifecycleError)


class ReliableCommandReceiverTest(unittest.TestCase):
    def test_duplicate_replays_ack_without_dispatching_twice(self) -> None:
        channel = _Channel()
        dispatcher = _Dispatcher()
        receiver = ReliableCommandReceiver(
            channel,
            dispatcher,
            CommandTransportConfig(),
        )
        receiver.start()
        packet = encode_command(command())
        self.assertTrue(receiver.accept_command(packet))
        self.assertTrue(receiver.accept_command(packet))
        self.assertEqual(len(dispatcher.commands), 1)
        first, second = map(decode_acknowledgement, channel.sent)
        self.assertEqual(first.status, CommandAckStatus.ACCEPTED)
        self.assertFalse(first.duplicate)
        self.assertEqual(second.status, CommandAckStatus.ACCEPTED)
        self.assertTrue(second.duplicate)
        self.assertEqual(receiver.metrics.duplicate_replays, 1)
        receiver.close()

    def test_conflicting_duplicate_is_rejected(self) -> None:
        channel = _Channel()
        dispatcher = _Dispatcher()
        receiver = ReliableCommandReceiver(
            channel,
            dispatcher,
            CommandTransportConfig(),
        )
        receiver.start()
        receiver.accept_command(encode_command(command(sequence=1)))
        receiver.accept_command(encode_command(command(sequence=2)))
        acknowledgement = decode_acknowledgement(channel.sent[-1])
        self.assertEqual(acknowledgement.status, CommandAckStatus.REJECTED)
        self.assertTrue(acknowledgement.duplicate)
        self.assertEqual(len(dispatcher.commands), 1)
        self.assertEqual(receiver.metrics.conflicting_duplicates, 1)
        receiver.close()

    def test_dedupe_memory_is_bounded_and_evicts_oldest(self) -> None:
        channel = _Channel()
        dispatcher = _Dispatcher()
        receiver = ReliableCommandReceiver(
            channel,
            dispatcher,
            CommandTransportConfig(dedupe_capacity=2),
        )
        receiver.start()
        for index in range(3):
            receiver.accept_command(
                encode_command(command(command_id=f"command-{index}", sequence=index))
            )
        receiver.accept_command(
            encode_command(command(command_id="command-0", sequence=0))
        )
        self.assertEqual(len(dispatcher.commands), 4)
        self.assertEqual(receiver.metrics.dedupe_evictions, 2)
        receiver.close()

    def test_dispatch_error_severity_becomes_error_ack(self) -> None:
        channel = _Channel()
        dispatcher = _Dispatcher(
            kind=RuntimeEventType.COMMAND_REJECTED,
            severity=RuntimeEventSeverity.ERROR,
        )
        receiver = ReliableCommandReceiver(
            channel,
            dispatcher,
            CommandTransportConfig(),
        )
        receiver.start()
        receiver.accept_command(encode_command(command()))
        acknowledgement = decode_acknowledgement(channel.sent[-1])
        self.assertEqual(acknowledgement.status, CommandAckStatus.ERROR)
        receiver.close()

    def test_malformed_command_is_counted_without_ack(self) -> None:
        channel = _Channel()
        receiver = ReliableCommandReceiver(
            channel,
            _Dispatcher(),
            CommandTransportConfig(),
        )
        receiver.start()
        self.assertFalse(receiver.accept_command(b"bad"))
        self.assertEqual(receiver.metrics.malformed_commands, 1)
        self.assertEqual(channel.sent, [])
        receiver.close()


class ReliableCommandChannelPolicyTest(unittest.TestCase):
    def test_invalid_channel_policy_is_rejected(self) -> None:
        channel = _Channel()
        channel.ordered = False
        with self.assertRaises(ModelValidationError):
            ReliableCommandSender(channel, CommandTransportConfig())
        channel.ordered = True
        channel.max_retransmits = 0
        with self.assertRaises(ModelValidationError):
            ReliableCommandSender(channel, CommandTransportConfig())

    def test_aiortc_factory_uses_ordered_fully_reliable_defaults(self) -> None:
        class RawChannel:
            ordered = True
            maxRetransmits = None
            maxPacketLifeTime = None

            def __init__(self) -> None:
                self.sent = []
                self.closed = False

            def send(self, data) -> None:
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
        channel = create_aiortc_reliable_command_channel(peer)
        self.assertEqual(peer.call, ("commands", {"ordered": True}))
        channel.start()
        channel.send(b"command")
        channel.close()
        self.assertEqual(peer.channel.sent, [b"command"])
        self.assertTrue(peer.channel.closed)


if __name__ == "__main__":
    unittest.main()
