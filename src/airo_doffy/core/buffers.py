"""Thread-safe constant-memory latest-value storage for real-time samples."""

from __future__ import annotations

import math
import threading
import time
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .errors import BufferClosedError, ModelValidationError


@runtime_checkable
class Sequenced(Protocol):
    """Value exposing the monotonic or modular sequence used for ordering."""

    @property
    def sequence(self) -> int:
        """Sequence number assigned by the producer."""


_T = TypeVar("_T", bound=Sequenced)


def is_newer_sequence(candidate: int, current: int, modulus: int | None = None) -> bool:
    """Return whether *candidate* is newer, optionally using modular wrap order.

    Modular comparison accepts forward movement of less than half the sequence
    space. Equal and exactly half-range values are rejected as stale/ambiguous.
    """

    for value, name in ((candidate, "candidate"), (current, "current")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelValidationError(f"{name} sequence must be a non-negative integer")
    if modulus is None:
        return candidate > current
    if isinstance(modulus, bool) or not isinstance(modulus, int) or modulus < 3:
        raise ModelValidationError("sequence modulus must be an integer >= 3")
    if candidate >= modulus or current >= modulus:
        raise ModelValidationError("modular sequence values must be smaller than the modulus")
    delta = (candidate - current) % modulus
    return 0 < delta <= (modulus - 1) // 2


class LatestValueBuffer(Generic[_T]):
    """Keep one immutable latest value and reject duplicate or stale updates."""

    __slots__ = (
        "_accepted_count",
        "_closed",
        "_condition",
        "_rejected_count",
        "_sequence",
        "_sequence_modulus",
        "_value",
    )

    def __init__(self, *, sequence_modulus: int | None = None) -> None:
        if sequence_modulus is not None:
            is_newer_sequence(1, 0, sequence_modulus)
        self._sequence_modulus = sequence_modulus
        self._condition = threading.Condition()
        self._value: _T | None = None
        self._sequence: int | None = None
        self._closed = False
        self._accepted_count = 0
        self._rejected_count = 0

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def latest_sequence(self) -> int | None:
        with self._condition:
            return self._sequence

    @property
    def accepted_count(self) -> int:
        with self._condition:
            return self._accepted_count

    @property
    def rejected_count(self) -> int:
        with self._condition:
            return self._rejected_count

    def publish(self, value: _T) -> bool:
        """Publish *value* if its sequence is newer; return whether accepted."""

        try:
            sequence = value.sequence
        except AttributeError as exc:
            raise ModelValidationError("latest-value entries must expose sequence") from exc
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ModelValidationError("value.sequence must be a non-negative integer")
        if self._sequence_modulus is not None and sequence >= self._sequence_modulus:
            raise ModelValidationError("value.sequence must be smaller than the modulus")

        with self._condition:
            if self._closed:
                raise BufferClosedError("cannot publish to a closed latest-value buffer")
            if self._sequence is not None and not is_newer_sequence(
                sequence,
                self._sequence,
                self._sequence_modulus,
            ):
                self._rejected_count += 1
                return False
            self._value = value
            self._sequence = sequence
            self._accepted_count += 1
            self._condition.notify_all()
            return True

    def read(self) -> _T | None:
        """Return the current immutable value without blocking."""

        with self._condition:
            return self._value

    def wait_for_new(
        self,
        *,
        after_sequence: int | None = None,
        timeout: float | None = None,
    ) -> _T | None:
        """Wait for a value newer than *after_sequence*.

        With ``after_sequence=None``, an already available value is returned
        immediately. A timeout or a closed buffer with no qualifying value
        returns ``None``.
        """

        if after_sequence is not None:
            if (
                isinstance(after_sequence, bool)
                or not isinstance(after_sequence, int)
                or after_sequence < 0
            ):
                raise ModelValidationError("after_sequence must be a non-negative integer")
            if (
                self._sequence_modulus is not None
                and after_sequence >= self._sequence_modulus
            ):
                raise ModelValidationError("after_sequence must be smaller than the modulus")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise ModelValidationError("timeout must be a non-negative finite number")
            timeout = float(timeout)
            if timeout < 0 or not math.isfinite(timeout):
                raise ModelValidationError("timeout must be a non-negative finite number")

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._value is not None and (
                    after_sequence is None
                    or is_newer_sequence(
                        self._value.sequence,
                        after_sequence,
                        self._sequence_modulus,
                    )
                ):
                    return self._value
                if self._closed:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def close(self) -> None:
        """Close idempotently and wake all waiting consumers."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()

    def __enter__(self) -> LatestValueBuffer[_T]:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
