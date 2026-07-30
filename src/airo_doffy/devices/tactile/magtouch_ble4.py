"""Supported 4-taxel MagTouch sensor with private latest state."""

from __future__ import annotations

import threading

from ...config.models import TactileConfig
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError
from ...core.types import ClockDomain, TactileSample
from .filters import Ble4SignalFilter, TaxelValues
from .source import Ble4RawSource, SensorCommBle4Source


class MagtouchBle4Sensor:
    """Calibrate and filter a raw BLE source without mutating external holders."""

    def __init__(
        self,
        config: TactileConfig,
        *,
        source: Ble4RawSource | None = None,
        signal_filter: Ble4SignalFilter | None = None,
        clock: Clock | None = None,
    ) -> None:
        if config.backend != "ble4" or config.shape != (4, 3):
            raise ModelValidationError("MagtouchBle4Sensor requires ble4 shape (4, 3)")
        self._config = config
        self._source = source or SensorCommBle4Source(config)
        if not isinstance(self._source, Ble4RawSource):
            raise ModelValidationError("source must satisfy Ble4RawSource")
        self._filter = signal_filter or Ble4SignalFilter(config)
        self._clock = clock or MonotonicClock()
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._calibrating = True
        self._calibration_samples: list[TaxelValues] = []
        self._latest: TactileSample | None = None
        self._sequence = 0
        self._disconnect_count = 0

    @property
    def calibrated(self) -> bool:
        with self._lock:
            return self._filter.calibrated and not self._calibrating

    @property
    def disconnect_count(self) -> int:
        with self._lock:
            return self._disconnect_count

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed BLE4 sensor")
            if self._started:
                raise LifecycleError("BLE4 sensor is already started")
            self._started = True
        try:
            self._source.start(self._on_raw_sample, self._on_disconnect)
        except Exception:
            with self._lock:
                self._started = False
            raise

    def _on_disconnect(self) -> None:
        with self._lock:
            if not self._started or self._closed:
                return
            self._disconnect_count += 1
            self._latest = None
            self._filter.reset()

    def _on_raw_sample(self, sample: TaxelValues) -> None:
        with self._lock:
            if not self._started or self._closed:
                return
            if self._calibrating:
                self._calibration_samples.append(sample)
                if len(self._calibration_samples) >= self._config.ble_window_size:
                    self._filter.calibrate(self._calibration_samples)
                    self._calibration_samples.clear()
                    self._calibrating = False
                    self._latest = None
                return
            filtered = self._filter.process(sample)
            now_ns = self._clock.now_ns()
            self._latest = TactileSample(
                sequence=self._sequence,
                source_timestamp_ns=now_ns,
                clock_domain=ClockDomain.MONOTONIC,
                values=filtered,
            )
            self._sequence += 1

    def read_latest(self) -> TactileSample | None:
        with self._lock:
            if self._closed:
                raise LifecycleError("BLE4 sensor is closed")
            if not self._started:
                raise LifecycleError("BLE4 sensor has not been started")
            return self._latest

    def recalibrate(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("BLE4 sensor is closed")
            if not self._started:
                raise LifecycleError("BLE4 sensor has not been started")
            self._calibrating = True
            self._calibration_samples.clear()
            self._latest = None
            self._filter.reset()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._started:
                self._closed = True
                return
        self._source.close()
        with self._lock:
            self._started = False
            self._closed = True
            self._latest = None


def create_magtouch_ble4(config: TactileConfig) -> MagtouchBle4Sensor:
    """Create an unstarted supported BLE4 sensor."""

    return MagtouchBle4Sensor(config)
