"""Transport-neutral VR receiver with decoding, aggregation, and stale rejection."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...core.buffers import LatestValueBuffer, is_newer_sequence
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError
from ...core.interfaces import Lifecycle
from ...core.types import HandSide, HandState, VRInputMode, VRInputState
from .protocol import decode_vr_message

RawVRMessage = str | bytes | bytearray | memoryview


@runtime_checkable
class RawVRTransport(Lifecycle, Protocol):
    """Transport that returns complete messages without interpreting them."""

    def receive(self, timeout_s: float) -> RawVRMessage | None:
        """Return one complete raw message or ``None`` on timeout."""


@dataclass(frozen=True, slots=True)
class VRReceiverStats:
    """Observable receiver counters without exposing mutable internals."""

    accepted: int
    malformed: int
    stale: int
    transport_errors: int


class VRReceiver:
    """Decode pushed or polled messages into one latest typed VR state."""

    def __init__(
        self,
        transport: RawVRTransport | None = None,
        *,
        clock: Clock | None = None,
        poll_timeout_s: float = 0.05,
    ) -> None:
        timeout = float(poll_timeout_s)
        if timeout <= 0:
            raise ModelValidationError("poll_timeout_s must be positive")
        self._transport = transport
        self._clock = clock or MonotonicClock()
        self._poll_timeout_s = timeout
        self._latest = LatestValueBuffer[VRInputState]()
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._health_error: Exception | None = None
        self._last_wire_sequence: dict[str, int] = {}
        self._hands: dict[HandSide, HandState] = {}
        self._output_sequence = 0
        self._accepted = 0
        self._malformed = 0
        self._stale = 0
        self._transport_errors = 0

    @property
    def health_error(self) -> Exception | None:
        with self._lock:
            return self._health_error

    @property
    def stats(self) -> VRReceiverStats:
        with self._lock:
            return VRReceiverStats(
                accepted=self._accepted,
                malformed=self._malformed,
                stale=self._stale,
                transport_errors=self._transport_errors,
            )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed VR receiver")
            if self._started:
                raise LifecycleError("VR receiver is already started")
        if self._transport is not None:
            self._transport.start()
        with self._lock:
            self._started = True
            self._stop_event.clear()
            if self._transport is not None:
                self._thread = threading.Thread(
                    target=self._worker,
                    name="airo-doffy-vr-receiver",
                    daemon=True,
                )
                try:
                    self._thread.start()
                except Exception:
                    self._started = False
                    self._thread = None
                    self._transport.close()
                    raise

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("VR receiver is closed")
        if not self._started:
            raise LifecycleError("VR receiver has not been started")

    @staticmethod
    def _is_newer(candidate: int, current: int) -> bool:
        if candidate < 2**32 and current < 2**32:
            return is_newer_sequence(candidate, current, 2**32)
        return is_newer_sequence(candidate, current)

    def _stream_sequences(self, state: VRInputState) -> tuple[tuple[str, int], ...]:
        if state.mode is VRInputMode.CONTROLLERS:
            return (("controllers", state.sequence),)
        return tuple(
            (f"hand:{hand.side.value}", hand.sequence)
            for hand in state.hands
        )

    def _is_stale(self, state: VRInputState) -> bool:
        streams = self._stream_sequences(state)
        return any(
            key in self._last_wire_sequence
            and not self._is_newer(sequence, self._last_wire_sequence[key])
            for key, sequence in streams
        )

    def _aggregate(self, state: VRInputState) -> VRInputState:
        if state.mode is VRInputMode.CONTROLLERS:
            self._hands.clear()
            controllers = state.controllers
            hands = ()
        else:
            for hand in state.hands:
                self._hands[hand.side] = hand
            controllers = ()
            hands = tuple(
                self._hands[side]
                for side in (HandSide.LEFT, HandSide.RIGHT)
                if side in self._hands
            )
        result = VRInputState(
            sequence=self._output_sequence,
            source_timestamp_ns=state.source_timestamp_ns,
            receive_timestamp_ns=state.receive_timestamp_ns,
            clock_domain=state.clock_domain,
            mode=state.mode,
            controllers=controllers,
            hands=hands,
        )
        self._output_sequence += 1
        return result

    def accept_message(
        self,
        message: RawVRMessage,
        *,
        receive_timestamp_ns: int | None = None,
    ) -> bool:
        """Decode and publish one message; return whether it was accepted."""

        with self._lock:
            self._require_started()
            received_ns = (
                self._clock.now_ns()
                if receive_timestamp_ns is None
                else receive_timestamp_ns
            )
            try:
                state = decode_vr_message(
                    message,
                    receive_timestamp_ns=received_ns,
                )
            except (TypeError, ValueError):
                state = None
            if state is None:
                self._malformed += 1
                return False
            if self._is_stale(state):
                self._stale += 1
                return False
            for key, sequence in self._stream_sequences(state):
                self._last_wire_sequence[key] = sequence
            output = self._aggregate(state)
            self._latest.publish(output)
            self._accepted += 1
            return True

    def _worker(self) -> None:
        assert self._transport is not None
        while not self._stop_event.is_set():
            try:
                message = self._transport.receive(self._poll_timeout_s)
            except Exception as exc:
                with self._lock:
                    self._health_error = exc
                    self._transport_errors += 1
                break
            if message is not None:
                self.accept_message(message)

    def read_latest(self) -> VRInputState | None:
        with self._lock:
            self._require_started()
            return self._latest.read()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._started:
                self._closed = True
                self._latest.close()
                return
            transport = self._transport
            thread = self._thread
            self._stop_event.set()
        cleanup_error: Exception | None = None
        if transport is not None:
            try:
                transport.close()
            except Exception as exc:
                cleanup_error = exc
        if thread is not None:
            thread.join(timeout=max(1.0, self._poll_timeout_s + 0.5))
            if thread.is_alive():
                raise LifecycleError(
                    "VR receiver worker did not stop; receiver was not invalidated"
                )
        with self._lock:
            self._started = False
            self._closed = True
            self._thread = None
            self._latest.close()
        if cleanup_error is not None:
            raise LifecycleError("VR transport cleanup failed") from cleanup_error
