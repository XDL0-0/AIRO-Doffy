"""Deterministic and synthetic 4-taxel sources for hardware-free tests."""

from __future__ import annotations

import math
import random
import threading
from enum import Enum

from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError
from ...core.types import ClockDomain, TactileSample

_ZERO_VALUES = tuple((0.0, 0.0, 0.0) for _ in range(4))


class TactileMockMode(str, Enum):
    FIXED = "fixed"
    RANDOM = "random"
    PERIODIC = "periodic"
    DISCONNECTED = "disconnected"
    DELAYED = "delayed"


class MockTactileSensor:
    """On-demand tactile source supporting the Phase 4 mock modes."""

    def __init__(
        self,
        *,
        mode: TactileMockMode | str = TactileMockMode.FIXED,
        fixed_values=_ZERO_VALUES,
        amplitude: float = 1.0,
        frequency_hz: float = 1.0,
        connection_delay_s: float = 0.0,
        random_seed: int | None = 0,
        clock: Clock | None = None,
    ) -> None:
        try:
            selected_mode = TactileMockMode(mode)
        except ValueError as exc:
            raise ModelValidationError(f"unsupported tactile mock mode: {mode!r}") from exc
        amplitude_value = float(amplitude)
        frequency = float(frequency_hz)
        delay = float(connection_delay_s)
        if not math.isfinite(amplitude_value) or amplitude_value < 0:
            raise ModelValidationError("tactile amplitude must be finite and non-negative")
        if not math.isfinite(frequency) or frequency <= 0:
            raise ModelValidationError("tactile frequency_hz must be positive and finite")
        if not math.isfinite(delay) or delay < 0:
            raise ModelValidationError(
                "tactile connection_delay_s must be finite and non-negative"
            )
        checked = TactileSample(
            sequence=0,
            source_timestamp_ns=0,
            values=fixed_values,
        ).values
        self._mode = selected_mode
        self._fixed_values = checked
        self._amplitude = amplitude_value
        self._frequency_hz = frequency
        self._connection_delay_ns = int(delay * 1e9)
        self._random = random.Random(random_seed)
        self._clock = clock or MonotonicClock()
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._forced_disconnect = False
        self._start_ns: int | None = None
        self._sequence = 0
        self._recalibration_count = 0

    @property
    def mode(self) -> TactileMockMode:
        return self._mode

    @property
    def recalibration_count(self) -> int:
        with self._lock:
            return self._recalibration_count

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed mock tactile sensor")
            if self._started:
                raise LifecycleError("mock tactile sensor is already started")
            self._started = True
            self._start_ns = self._clock.now_ns()

    def _require_started(self) -> int:
        if self._closed:
            raise LifecycleError("mock tactile sensor is closed")
        if not self._started or self._start_ns is None:
            raise LifecycleError("mock tactile sensor has not been started")
        return self._start_ns

    def set_disconnected(self, disconnected: bool) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("mock tactile sensor is closed")
            self._forced_disconnect = bool(disconnected)

    def _values(self, now_ns: int, start_ns: int):
        if self._mode is TactileMockMode.FIXED:
            return self._fixed_values
        if self._mode is TactileMockMode.RANDOM:
            return tuple(
                tuple(
                    self._random.uniform(-self._amplitude, self._amplitude)
                    for _axis in range(3)
                )
                for _taxel in range(4)
            )
        if self._mode is TactileMockMode.PERIODIC:
            elapsed_s = (now_ns - start_ns) / 1e9
            value = self._amplitude * math.sin(
                2 * math.pi * self._frequency_hz * elapsed_s
            )
            return tuple((value, value, value) for _taxel in range(4))
        if self._mode is TactileMockMode.DELAYED:
            if now_ns - start_ns < self._connection_delay_ns:
                return None
            return self._fixed_values
        return None

    def read_latest(self) -> TactileSample | None:
        with self._lock:
            start_ns = self._require_started()
            if self._forced_disconnect or self._mode is TactileMockMode.DISCONNECTED:
                return None
            now_ns = self._clock.now_ns()
            values = self._values(now_ns, start_ns)
            if values is None:
                return None
            sample = TactileSample(
                sequence=self._sequence,
                source_timestamp_ns=now_ns,
                clock_domain=ClockDomain.MONOTONIC,
                values=values,
            )
            self._sequence += 1
            return sample

    def recalibrate(self) -> None:
        with self._lock:
            self._require_started()
            self._recalibration_count += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True
