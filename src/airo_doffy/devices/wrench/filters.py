"""Pure wrench processing independent of robot SDKs and NumPy."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable

from ...core.errors import ModelValidationError


class WrenchFilter:
    """Deadband, moving average, then low-pass filter for one 6D wrench."""

    def __init__(
        self,
        moving_average_window: int = 1,
        low_pass_alpha: float = 0.0,
        force_deadband: float = 0.0,
        torque_deadband: float = 0.0,
    ) -> None:
        try:
            window = max(1, int(moving_average_window))
            alpha = min(max(float(low_pass_alpha), 0.0), 1.0)
            force_threshold = max(0.0, float(force_deadband))
            torque_threshold = max(0.0, float(torque_deadband))
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("invalid wrench filter parameter") from exc
        self.moving_average_window = window
        self.low_pass_alpha = alpha
        self.force_deadband = force_threshold
        self.torque_deadband = torque_threshold
        self.filtered = (0.0,) * 6
        self.initialized = False
        self._moving_window: deque[tuple[float, ...]] = deque(maxlen=window)

    def reset(self) -> None:
        self.filtered = (0.0,) * 6
        self.initialized = False
        self._moving_window.clear()

    @staticmethod
    def _deadband(value: float, threshold: float) -> float:
        if threshold <= 0:
            return value
        if abs(value) < threshold:
            return 0.0
        if math.isnan(value):
            return value
        return value - math.copysign(threshold, value)

    def process(self, wrench: Iterable[object]) -> tuple[float, ...]:
        try:
            values = tuple(float(value) for value in wrench)[:6]
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("wrench must be an iterable of numbers") from exc
        if len(values) < 6:
            return (0.0,) * 6
        deadbanded = tuple(
            self._deadband(
                value,
                self.force_deadband if index < 3 else self.torque_deadband,
            )
            for index, value in enumerate(values)
        )
        if self.moving_average_window > 1:
            self._moving_window.append(deadbanded)
            count = len(self._moving_window)
            averaged = tuple(
                sum(sample[index] for sample in self._moving_window) / count
                for index in range(6)
            )
        else:
            averaged = deadbanded

        if self.low_pass_alpha <= 0:
            return averaged
        if not self.initialized:
            self.filtered = averaged
            self.initialized = True
        else:
            alpha = self.low_pass_alpha
            self.filtered = tuple(
                alpha * value + (1 - alpha) * previous
                for value, previous in zip(averaged, self.filtered, strict=True)
            )
        return self.filtered
