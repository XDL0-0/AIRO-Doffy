"""Pure 4-taxel baseline, reliability, EMA, and Kalman filtering."""

from __future__ import annotations

import math
import statistics
import threading
from collections.abc import Iterable

from ...config.models import TactileConfig
from ...core.errors import ModelValidationError

TaxelValues = tuple[tuple[float, ...], ...]


def _sample(values: Iterable[Iterable[object]], name: str = "sample") -> TaxelValues:
    try:
        result = tuple(tuple(float(value) for value in row) for row in values)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must have shape (4, 3)") from exc
    if len(result) != 4 or any(len(row) != 3 for row in result):
        raise ModelValidationError(f"{name} must have shape (4, 3)")
    return result


def _zeros(value: float = 0.0) -> list[list[float]]:
    return [[value for _axis in range(3)] for _taxel in range(4)]


def _tuple(values: list[list[float]]) -> TaxelValues:
    return tuple(tuple(row) for row in values)


class Ble4SignalFilter:
    """Stateful numerical pipeline preserving the legacy BLE4 operation order."""

    def __init__(self, config: TactileConfig) -> None:
        self._config = config
        self._baseline = _zeros()
        self._deadband = _zeros(config.noise_floor)
        self._ema: list[list[float]] | None = None
        self._kalman_x: list[list[float]] | None = None
        self._kalman_p: list[list[float]] | None = None
        self._last_good: list[list[float]] | None = None
        self._calibrated = False
        self._lock = threading.RLock()

    @property
    def calibrated(self) -> bool:
        with self._lock:
            return self._calibrated

    @property
    def baseline(self) -> TaxelValues:
        with self._lock:
            return _tuple(self._baseline)

    @property
    def deadband(self) -> TaxelValues:
        with self._lock:
            return _tuple(self._deadband)

    def reset(self) -> None:
        """Reset dynamic filters without discarding calibration."""

        with self._lock:
            self._ema = None
            self._kalman_x = None
            self._kalman_p = None
            self._last_good = None

    def calibrate(self, samples: Iterable[Iterable[Iterable[object]]]) -> None:
        """Set per-axis median baseline and robust MAD deadband."""

        checked = tuple(_sample(sample, "calibration sample") for sample in samples)
        if not checked:
            raise ModelValidationError("calibration requires at least one sample")
        baseline = _zeros()
        deadband = _zeros()
        for taxel in range(4):
            for axis in range(3):
                values = [sample[taxel][axis] for sample in checked]
                median = statistics.median(values)
                mad = statistics.median(abs(value - median) for value in values) * 1.4826
                baseline[taxel][axis] = median
                deadband[taxel][axis] = max(
                    self._config.noise_floor,
                    self._config.deadband_sigma * mad,
                )
        with self._lock:
            self._baseline = baseline
            self._deadband = deadband
            self._calibrated = True
            self.reset()

    def _track_baseline(self, raw: TaxelValues) -> list[list[float]]:
        centered = [
            [raw[taxel][axis] - self._baseline[taxel][axis] for axis in range(3)]
            for taxel in range(4)
        ]
        if self._config.baseline_drift_alpha <= 0:
            return centered
        norms = [
            math.sqrt(sum(value * value for value in centered[taxel]))
            for taxel in range(4)
        ]
        if max(norms) < self._config.baseline_drift_threshold:
            alpha = self._config.baseline_drift_alpha
            for taxel in range(4):
                for axis in range(3):
                    self._baseline[taxel][axis] += alpha * centered[taxel][axis]
                    centered[taxel][axis] = (
                        raw[taxel][axis] - self._baseline[taxel][axis]
                    )
        return centered

    def process(self, sample: Iterable[Iterable[object]]) -> TaxelValues:
        raw = _sample(sample)
        with self._lock:
            values = self._track_baseline(raw)
            for taxel in range(4):
                for axis in range(3):
                    value = values[taxel][axis]
                    if abs(value) < self._deadband[taxel][axis]:
                        value = 0.0
                    value = min(max(value, -self._config.max_abs), self._config.max_abs)
                    if self._last_good is not None:
                        delta = value - self._last_good[taxel][axis]
                        delta = min(
                            max(delta, -self._config.max_delta),
                            self._config.max_delta,
                        )
                        value = self._last_good[taxel][axis] + delta
                    values[taxel][axis] = value

            if self._ema is None:
                self._ema = [row.copy() for row in values]
            else:
                alpha = self._config.filter_alpha
                for taxel in range(4):
                    for axis in range(3):
                        previous = self._ema[taxel][axis]
                        self._ema[taxel][axis] = previous + alpha * (
                            values[taxel][axis] - previous
                        )

            if not self._config.use_kalman:
                filtered = [row.copy() for row in self._ema]
            elif self._kalman_x is None:
                self._kalman_x = [row.copy() for row in self._ema]
                self._kalman_p = _zeros(1.0)
                filtered = [row.copy() for row in self._kalman_x]
            else:
                assert self._kalman_p is not None
                for taxel in range(4):
                    for axis in range(3):
                        predicted = (
                            self._kalman_p[taxel][axis] + self._config.kalman_q
                        )
                        gain = predicted / (predicted + self._config.kalman_r)
                        self._kalman_x[taxel][axis] += gain * (
                            self._ema[taxel][axis] - self._kalman_x[taxel][axis]
                        )
                        self._kalman_p[taxel][axis] = (1.0 - gain) * predicted
                filtered = [row.copy() for row in self._kalman_x]

            for taxel in range(4):
                for axis in range(3):
                    if not math.isfinite(filtered[taxel][axis]):
                        filtered[taxel][axis] = 0.0
            self._last_good = [row.copy() for row in filtered]
            return _tuple(filtered)
