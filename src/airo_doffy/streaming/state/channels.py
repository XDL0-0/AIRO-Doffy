"""Latest-only realtime state channels and transport adapters."""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ...core.buffers import is_newer_sequence
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError
from ...core.interfaces import Lifecycle
from ...core.types import RobotState, VRInputState
from .protocol import (
    STATE_SEQUENCE_MODULUS,
    StateMessageType,
    StateSample,
    decode_state_packet,
    encode_state_packet,
    state_message_type,
)


@runtime_checkable
class RealtimeStateChannel(Lifecycle, Protocol):
    """Raw state channel configured for unordered best-effort delivery."""

    @property
    def ordered(self) -> bool:
        """Whether the transport waits for ordered delivery."""

    @property
    def max_retransmits(self) -> int | None:
        """Maximum retransmissions; realtime state requires zero."""

    def send(self, data: bytes) -> None:
        """Send one complete state packet."""


@dataclass(frozen=True, slots=True)
class StateSenderMetrics:
    """Snapshot of realtime state sender counters."""

    submitted: int
    sent: int
    dropped_overwritten: int
    rejected_stale: int
    bytes_sent: int
    send_errors: int


class LatestStateSender:
    """Send at most the newest pending packet for each state message type."""

    def __init__(self, channel: RealtimeStateChannel) -> None:
        if channel.ordered or channel.max_retransmits != 0:
            raise ModelValidationError(
                "realtime state channel must be unordered with max_retransmits=0"
            )
        self._channel = channel
        self._condition = threading.Condition()
        self._pending: dict[StateMessageType, bytes] = {}
        self._last_submitted: dict[StateMessageType, int] = {}
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._stop = False
        self._health_error: Exception | None = None
        self._submitted = 0
        self._sent = 0
        self._dropped_overwritten = 0
        self._rejected_stale = 0
        self._bytes_sent = 0
        self._send_errors = 0

    @property
    def health_error(self) -> Exception | None:
        with self._condition:
            return self._health_error

    @property
    def metrics(self) -> StateSenderMetrics:
        with self._condition:
            return StateSenderMetrics(
                submitted=self._submitted,
                sent=self._sent,
                dropped_overwritten=self._dropped_overwritten,
                rejected_stale=self._rejected_stale,
                bytes_sent=self._bytes_sent,
                send_errors=self._send_errors,
            )

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise LifecycleError("cannot start a closed state sender")
            if self._started:
                raise LifecycleError("state sender is already started")
        self._channel.start()
        with self._condition:
            self._started = True
            self._thread = threading.Thread(
                target=self._worker,
                name="airo-doffy-state-sender",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception:
                self._started = False
                self._thread = None
                self._channel.close()
                raise

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("state sender is closed")
        if not self._started:
            raise LifecycleError("state sender has not been started")

    def submit(self, sample: StateSample) -> bool:
        message_type = state_message_type(sample)
        packet = encode_state_packet(sample)
        with self._condition:
            self._require_started()
            previous = self._last_submitted.get(message_type)
            if previous is not None and not is_newer_sequence(
                sample.sequence,
                previous,
                STATE_SEQUENCE_MODULUS,
            ):
                self._rejected_stale += 1
                return False
            self._last_submitted[message_type] = sample.sequence
            self._submitted += 1
            if message_type in self._pending:
                self._dropped_overwritten += 1
            self._pending[message_type] = packet
            self._condition.notify()
            return True

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                _message_type, packet = self._pending.popitem()
            try:
                self._channel.send(packet)
            except Exception as exc:
                with self._condition:
                    self._health_error = exc
                    self._send_errors += 1
                    self._stop = True
                    self._condition.notify_all()
                return
            with self._condition:
                self._sent += 1
                self._bytes_sent += len(packet)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if not self._started:
                self._closed = True
                self._channel.close()
                return
            self._stop = True
            self._pending.clear()
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise LifecycleError("state sender worker did not stop")
        channel_error: Exception | None = None
        try:
            self._channel.close()
        except Exception as exc:
            channel_error = exc
        with self._condition:
            self._started = False
            self._closed = True
            self._thread = None
        if channel_error is not None:
            raise LifecycleError("state channel cleanup failed") from channel_error


@dataclass(frozen=True, slots=True)
class StateReceiverMetrics:
    """Snapshot of accepted, stale, and malformed state packets."""

    accepted: int
    rejected_stale: int
    rejected_malformed: int
    bytes_received: int


class LatestStateReceiver:
    """Decode and retain one newest sample per state type."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or MonotonicClock()
        self._latest: dict[StateMessageType, StateSample] = {}
        self._lock = threading.Lock()
        self._accepted = 0
        self._rejected_stale = 0
        self._rejected_malformed = 0
        self._bytes_received = 0

    @property
    def metrics(self) -> StateReceiverMetrics:
        with self._lock:
            return StateReceiverMetrics(
                accepted=self._accepted,
                rejected_stale=self._rejected_stale,
                rejected_malformed=self._rejected_malformed,
                bytes_received=self._bytes_received,
            )

    @property
    def latest_vr(self) -> VRInputState | None:
        with self._lock:
            sample = self._latest.get(StateMessageType.VR_INPUT)
            return sample if isinstance(sample, VRInputState) else None

    @property
    def latest_robot(self) -> RobotState | None:
        with self._lock:
            sample = self._latest.get(StateMessageType.ROBOT_STATE)
            return sample if isinstance(sample, RobotState) else None

    def accept(self, data: bytes | bytearray | memoryview) -> bool:
        """Accept one packet when it is valid and newer for its message type."""

        try:
            size = len(data)
            sample = decode_state_packet(
                data,
                receive_timestamp_ns=self._clock.now_ns(),
            )
        except (ModelValidationError, TypeError):
            with self._lock:
                self._rejected_malformed += 1
            return False
        message_type = state_message_type(sample)
        with self._lock:
            previous = self._latest.get(message_type)
            if previous is not None and not is_newer_sequence(
                sample.sequence,
                previous.sequence,
                STATE_SEQUENCE_MODULUS,
            ):
                self._rejected_stale += 1
                return False
            self._latest[message_type] = sample
            self._accepted += 1
            self._bytes_received += size
            return True


class _RtcDataChannel(Protocol):
    ordered: bool
    maxRetransmits: int | None

    def send(self, data: bytes) -> None: ...

    def close(self) -> None: ...


class WebRtcStateDataChannel:
    """Lifecycle adapter around an aiortc-compatible RTCDataChannel."""

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

    def start(self) -> None:
        if self._closed:
            raise LifecycleError("cannot start a closed WebRTC state channel")
        if self._started:
            raise LifecycleError("WebRTC state channel is already started")
        self._started = True

    def send(self, data: bytes) -> None:
        if self._closed:
            raise LifecycleError("WebRTC state channel is closed")
        if not self._started:
            raise LifecycleError("WebRTC state channel has not been started")
        self._data_channel.send(data)

    def close(self) -> None:
        if self._closed:
            return
        self._started = False
        self._closed = True
        self._data_channel.close()


def create_aiortc_realtime_state_channel(
    peer_connection: Any,
    *,
    label: str = "realtime_state",
) -> WebRtcStateDataChannel:
    """Create the required aiortc state channel without importing aiortc."""

    if not isinstance(label, str) or not label.strip():
        raise ModelValidationError("state channel label must be a non-empty string")
    channel = peer_connection.createDataChannel(
        label,
        ordered=False,
        maxRetransmits=0,
    )
    return WebRtcStateDataChannel(channel)


class DatagramSocket(Protocol):
    def sendto(self, data: bytes, target: tuple[str, int]) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class UdpStateChannelMetrics:
    """Snapshot of diagnostic UDP send counters."""

    packets_sent: int
    bytes_sent: int
    errors: int


def _udp_socket() -> DatagramSocket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class UdpDiagnosticStateChannel:
    """Optional one-datagram adapter for diagnostics, not production reliability."""

    ordered = False
    max_retransmits = 0

    def __init__(
        self,
        target_host: str,
        target_port: int,
        *,
        socket_factory=_udp_socket,
    ) -> None:
        if not isinstance(target_host, str) or not target_host.strip():
            raise ModelValidationError("target_host must be a non-empty string")
        if (
            isinstance(target_port, bool)
            or not isinstance(target_port, int)
            or not 1 <= target_port <= 65535
        ):
            raise ModelValidationError("target_port must be within [1, 65535]")
        self._target = (target_host, target_port)
        self._socket_factory = socket_factory
        self._socket: DatagramSocket | None = None
        self._started = False
        self._closed = False
        self._packets_sent = 0
        self._bytes_sent = 0
        self._errors = 0

    @property
    def metrics(self) -> UdpStateChannelMetrics:
        return UdpStateChannelMetrics(
            packets_sent=self._packets_sent,
            bytes_sent=self._bytes_sent,
            errors=self._errors,
        )

    def start(self) -> None:
        if self._closed:
            raise LifecycleError("cannot start a closed UDP state channel")
        if self._started:
            raise LifecycleError("UDP state channel is already started")
        self._socket = self._socket_factory()
        self._started = True

    def send(self, data: bytes) -> None:
        if self._closed:
            raise LifecycleError("UDP state channel is closed")
        if not self._started or self._socket is None:
            raise LifecycleError("UDP state channel has not been started")
        try:
            payload = bytes(data)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("state packet must support the bytes protocol") from exc
        try:
            sent = self._socket.sendto(payload, self._target)
            if sent != len(payload):
                raise OSError(f"partial UDP datagram send: {sent}/{len(payload)} bytes")
        except OSError:
            self._errors += 1
            raise
        self._packets_sent += 1
        self._bytes_sent += sent

    def close(self) -> None:
        if self._closed:
            return
        udp_socket = self._socket
        self._socket = None
        self._started = False
        self._closed = True
        if udp_socket is not None:
            udp_socket.close()
