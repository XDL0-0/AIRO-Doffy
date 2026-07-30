"""Reliable ordered command channels with ACK, timeout, and bounded deduplication."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from ...config.models import CommandTransportConfig
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import CommandTimeoutError, LifecycleError, ModelValidationError
from ...core.events import (
    RuntimeCommand,
    RuntimeEvent,
    RuntimeEventSeverity,
    RuntimeEventType,
)
from ...core.interfaces import Lifecycle
from .protocol import (
    CommandAcknowledgement,
    CommandAckStatus,
    decode_acknowledgement,
    decode_command,
    encode_acknowledgement,
    encode_command,
)


@runtime_checkable
class ReliableCommandChannel(Lifecycle, Protocol):
    """Raw ordered DataChannel with retransmission limits disabled."""

    @property
    def ordered(self) -> bool:
        """Whether the channel preserves message order."""

    @property
    def max_retransmits(self) -> int | None:
        """Configured retry limit; reliable channels require ``None``."""

    @property
    def max_packet_lifetime(self) -> int | None:
        """Configured lifetime limit; reliable channels require ``None``."""

    def send(self, data: bytes) -> None:
        """Send one complete reliable message."""


@runtime_checkable
class CommandDispatcher(Protocol):
    """Narrow routing boundary consumed by the command receiver."""

    def dispatch(self, command: RuntimeCommand) -> RuntimeEvent:
        """Route one command and return its observable outcome."""


def _validate_channel(channel: ReliableCommandChannel) -> None:
    if (
        not channel.ordered
        or channel.max_retransmits is not None
        or channel.max_packet_lifetime is not None
    ):
        raise ModelValidationError(
            "command channel must be ordered and have no retransmit or lifetime limit"
        )


@dataclass(slots=True)
class _PendingAck:
    command_sequence: int
    event: threading.Event
    acknowledgement: CommandAcknowledgement | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class CommandSenderMetrics:
    """Snapshot of command sender delivery and acknowledgement counters."""

    commands_sent: int
    acknowledgements_received: int
    timeouts: int
    unexpected_acknowledgements: int
    malformed_acknowledgements: int
    send_errors: int
    bytes_sent: int
    bytes_received: int


class ReliableCommandSender:
    """Send commands and synchronously wait for their matching acknowledgements."""

    def __init__(
        self,
        channel: ReliableCommandChannel,
        config: CommandTransportConfig,
    ) -> None:
        _validate_channel(channel)
        self._channel = channel
        self._ack_timeout_s = config.ack_timeout_s
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingAck] = {}
        self._started = False
        self._closed = False
        self._commands_sent = 0
        self._acknowledgements_received = 0
        self._timeouts = 0
        self._unexpected_acknowledgements = 0
        self._malformed_acknowledgements = 0
        self._send_errors = 0
        self._bytes_sent = 0
        self._bytes_received = 0

    @property
    def metrics(self) -> CommandSenderMetrics:
        with self._lock:
            return CommandSenderMetrics(
                commands_sent=self._commands_sent,
                acknowledgements_received=self._acknowledgements_received,
                timeouts=self._timeouts,
                unexpected_acknowledgements=self._unexpected_acknowledgements,
                malformed_acknowledgements=self._malformed_acknowledgements,
                send_errors=self._send_errors,
                bytes_sent=self._bytes_sent,
                bytes_received=self._bytes_received,
            )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed command sender")
            if self._started:
                raise LifecycleError("command sender is already started")
        self._channel.start()
        with self._lock:
            self._started = True

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("command sender is closed")
        if not self._started:
            raise LifecycleError("command sender has not been started")

    def send(
        self,
        command: RuntimeCommand,
        *,
        timeout_s: float | None = None,
    ) -> CommandAcknowledgement:
        """Send a command and return its ACK, or raise on deadline expiry."""

        packet = encode_command(command)
        timeout = self._ack_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise ModelValidationError("command timeout_s must be positive")
        pending = _PendingAck(
            command_sequence=command.sequence,
            event=threading.Event(),
        )
        with self._lock:
            self._require_started()
            if command.command_id in self._pending:
                raise ModelValidationError(
                    f"command_id {command.command_id!r} is already awaiting an ACK"
                )
            self._pending[command.command_id] = pending
        try:
            self._channel.send(packet)
        except Exception:
            with self._lock:
                self._pending.pop(command.command_id, None)
                self._send_errors += 1
            raise
        with self._lock:
            self._commands_sent += 1
            self._bytes_sent += len(packet)
        if not pending.event.wait(timeout):
            with self._lock:
                current = self._pending.pop(command.command_id, None)
                if current is not None and current.acknowledgement is None:
                    self._timeouts += 1
                    raise CommandTimeoutError(
                        f"command {command.command_id!r} timed out after {timeout:g}s"
                    )
        with self._lock:
            self._pending.pop(command.command_id, None)
            error = pending.error
            acknowledgement = pending.acknowledgement
        if error is not None:
            raise error
        if acknowledgement is None:
            raise LifecycleError("command ACK wait ended without a result")
        return acknowledgement

    def accept_acknowledgement(
        self,
        data: bytes | bytearray | memoryview,
    ) -> bool:
        """Match an incoming ACK to a waiting command."""

        try:
            size = len(data)
            acknowledgement = decode_acknowledgement(data)
        except (ModelValidationError, TypeError):
            with self._lock:
                self._malformed_acknowledgements += 1
            return False
        with self._lock:
            pending = self._pending.get(acknowledgement.command_id)
            if pending is None:
                self._unexpected_acknowledgements += 1
                return False
            if acknowledgement.command_sequence != pending.command_sequence:
                self._malformed_acknowledgements += 1
                return False
            if pending.acknowledgement is not None:
                self._unexpected_acknowledgements += 1
                return False
            pending.acknowledgement = acknowledgement
            self._acknowledgements_received += 1
            self._bytes_received += size
            pending.event.set()
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True
            for pending in self._pending.values():
                pending.error = LifecycleError(
                    "command sender closed while awaiting acknowledgement"
                )
                pending.event.set()
        self._channel.close()


@dataclass(frozen=True, slots=True)
class CommandReceiverMetrics:
    """Snapshot of receiver routing, dedupe, ACK, and error counters."""

    commands_received: int
    commands_dispatched: int
    duplicate_replays: int
    conflicting_duplicates: int
    malformed_commands: int
    dedupe_evictions: int
    acknowledgements_sent: int
    acknowledgement_errors: int
    bytes_received: int
    bytes_sent: int


@dataclass(frozen=True, slots=True)
class _DedupeRecord:
    fingerprint: bytes
    acknowledgement: CommandAcknowledgement


class ReliableCommandReceiver:
    """Route new commands once and replay cached ACKs for safe retries."""

    def __init__(
        self,
        channel: ReliableCommandChannel,
        dispatcher: CommandDispatcher,
        config: CommandTransportConfig,
        *,
        clock: Clock | None = None,
    ) -> None:
        _validate_channel(channel)
        if not isinstance(dispatcher, CommandDispatcher):
            raise ModelValidationError("dispatcher must satisfy CommandDispatcher")
        self._channel = channel
        self._dispatcher = dispatcher
        self._dedupe_capacity = config.dedupe_capacity
        self._clock = clock or MonotonicClock()
        self._lock = threading.Lock()
        self._dedupe: OrderedDict[str, _DedupeRecord] = OrderedDict()
        self._started = False
        self._closed = False
        self._commands_received = 0
        self._commands_dispatched = 0
        self._duplicate_replays = 0
        self._conflicting_duplicates = 0
        self._malformed_commands = 0
        self._dedupe_evictions = 0
        self._acknowledgements_sent = 0
        self._acknowledgement_errors = 0
        self._bytes_received = 0
        self._bytes_sent = 0

    @property
    def metrics(self) -> CommandReceiverMetrics:
        with self._lock:
            return CommandReceiverMetrics(
                commands_received=self._commands_received,
                commands_dispatched=self._commands_dispatched,
                duplicate_replays=self._duplicate_replays,
                conflicting_duplicates=self._conflicting_duplicates,
                malformed_commands=self._malformed_commands,
                dedupe_evictions=self._dedupe_evictions,
                acknowledgements_sent=self._acknowledgements_sent,
                acknowledgement_errors=self._acknowledgement_errors,
                bytes_received=self._bytes_received,
                bytes_sent=self._bytes_sent,
            )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed command receiver")
            if self._started:
                raise LifecycleError("command receiver is already started")
        self._channel.start()
        with self._lock:
            self._started = True

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("command receiver is closed")
        if not self._started:
            raise LifecycleError("command receiver has not been started")

    @staticmethod
    def _ack_status(event: RuntimeEvent) -> CommandAckStatus:
        if event.kind is RuntimeEventType.COMMAND_ACCEPTED:
            return CommandAckStatus.ACCEPTED
        if event.kind is RuntimeEventType.COMMAND_REJECTED:
            if event.severity in {
                RuntimeEventSeverity.ERROR,
                RuntimeEventSeverity.CRITICAL,
            }:
                return CommandAckStatus.ERROR
            return CommandAckStatus.REJECTED
        return CommandAckStatus.ERROR

    def _dispatch(self, command: RuntimeCommand) -> CommandAcknowledgement:
        try:
            event = self._dispatcher.dispatch(command)
            if not isinstance(event, RuntimeEvent):
                raise TypeError("dispatcher did not return RuntimeEvent")
            if event.command_id not in {None, command.command_id}:
                raise ValueError("dispatcher event command_id does not match command")
            status = self._ack_status(event)
            message = event.message
        except Exception as exc:
            status = CommandAckStatus.ERROR
            message = f"command dispatcher failed: {type(exc).__name__}: {exc}"
        return CommandAcknowledgement(
            command_id=command.command_id,
            command_sequence=command.sequence,
            timestamp_ns=self._clock.now_ns(),
            status=status,
            message=message,
        )

    def accept_command(self, data: bytes | bytearray | memoryview) -> bool:
        """Route or deduplicate one valid command and send an explicit ACK."""

        try:
            size = len(data)
            command = decode_command(data)
        except (ModelValidationError, TypeError):
            with self._lock:
                self._malformed_commands += 1
            return False
        fingerprint = encode_command(command)
        with self._lock:
            self._require_started()
            self._commands_received += 1
            self._bytes_received += size
            record = self._dedupe.get(command.command_id)
            if record is not None:
                self._dedupe.move_to_end(command.command_id)
                if record.fingerprint == fingerprint:
                    acknowledgement = replace(
                        record.acknowledgement,
                        timestamp_ns=self._clock.now_ns(),
                        duplicate=True,
                    )
                    self._duplicate_replays += 1
                else:
                    acknowledgement = CommandAcknowledgement(
                        command_id=command.command_id,
                        command_sequence=command.sequence,
                        timestamp_ns=self._clock.now_ns(),
                        status=CommandAckStatus.REJECTED,
                        message="command_id was reused with a different payload",
                        duplicate=True,
                    )
                    self._conflicting_duplicates += 1
            else:
                acknowledgement = self._dispatch(command)
                self._commands_dispatched += 1
                self._dedupe[command.command_id] = _DedupeRecord(
                    fingerprint=fingerprint,
                    acknowledgement=acknowledgement,
                )
                if len(self._dedupe) > self._dedupe_capacity:
                    self._dedupe.popitem(last=False)
                    self._dedupe_evictions += 1
        packet = encode_acknowledgement(acknowledgement)
        try:
            self._channel.send(packet)
        except Exception:
            with self._lock:
                self._acknowledgement_errors += 1
            raise
        with self._lock:
            self._acknowledgements_sent += 1
            self._bytes_sent += len(packet)
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True
        self._channel.close()


class _RtcDataChannel(Protocol):
    ordered: bool
    maxRetransmits: int | None
    maxPacketLifeTime: int | None

    def send(self, data: bytes) -> None: ...

    def close(self) -> None: ...


class WebRtcCommandDataChannel:
    """Lifecycle adapter around an aiortc-compatible reliable DataChannel."""

    def __init__(self, data_channel: _RtcDataChannel) -> None:
        self._data_channel = data_channel
        self._started = False
        self._closed = False

    @property
    def ordered(self) -> bool:
        return bool(self._data_channel.ordered)

    @property
    def max_retransmits(self) -> int | None:
        return self._data_channel.maxRetransmits

    @property
    def max_packet_lifetime(self) -> int | None:
        return self._data_channel.maxPacketLifeTime

    def start(self) -> None:
        if self._closed:
            raise LifecycleError("cannot start a closed WebRTC command channel")
        if self._started:
            raise LifecycleError("WebRTC command channel is already started")
        self._started = True

    def send(self, data: bytes) -> None:
        if self._closed:
            raise LifecycleError("WebRTC command channel is closed")
        if not self._started:
            raise LifecycleError("WebRTC command channel has not been started")
        self._data_channel.send(data)

    def close(self) -> None:
        if self._closed:
            return
        self._started = False
        self._closed = True
        self._data_channel.close()


def create_aiortc_reliable_command_channel(
    peer_connection: Any,
    *,
    label: str = "commands",
) -> WebRtcCommandDataChannel:
    """Create an ordered fully reliable channel without importing aiortc."""

    if not isinstance(label, str) or not label.strip():
        raise ModelValidationError("command channel label must be a non-empty string")
    channel = peer_connection.createDataChannel(label, ordered=True)
    return WebRtcCommandDataChannel(channel)
