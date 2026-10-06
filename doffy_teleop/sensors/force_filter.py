from collections import deque

import numpy as np


class WrenchFilter:
    def __init__(
        self,
        moving_average_window: int = 1,
        low_pass_alpha: float = 0.0,
        force_deadband: float = 0.0,
        torque_deadband: float = 0.0,
    ) -> None:
        self.moving_average_window = max(1, int(moving_average_window))
        self.low_pass_alpha = float(np.clip(low_pass_alpha, 0.0, 1.0))
        self.force_deadband = max(0.0, float(force_deadband))
        self.torque_deadband = max(0.0, float(torque_deadband))
        self.filtered = np.zeros(6, dtype=float)
        self.initialized = False
        self._moving_window: deque[np.ndarray] = deque(maxlen=self.moving_average_window)

    def reset(self) -> None:
        self.filtered = np.zeros(6, dtype=float)
        self.initialized = False
        self._moving_window.clear()

    def process(self, wrench: np.ndarray) -> np.ndarray:
        values = np.asarray(wrench, dtype=float).reshape(-1)[:6]
        if values.size < 6:
            return np.zeros(6, dtype=float)
        values = values.copy()
        values[:3] = self._apply_deadband(values[:3], self.force_deadband)
        values[3:] = self._apply_deadband(values[3:], self.torque_deadband)
        values = self._apply_moving_average(values)

        if self.low_pass_alpha <= 0.0:
            return values.copy()
        if not self.initialized:
            self.filtered = values.copy()
            self.initialized = True
        else:
            alpha = self.low_pass_alpha
            self.filtered = alpha * values + (1.0 - alpha) * self.filtered
        return self.filtered.copy()

    def _apply_moving_average(self, values: np.ndarray) -> np.ndarray:
        if self.moving_average_window <= 1:
            return values
        self._moving_window.append(values.copy())
        return np.mean(np.vstack(self._moving_window), axis=0)

    @staticmethod
    def _apply_deadband(values: np.ndarray, threshold: float) -> np.ndarray:
        if threshold <= 0.0:
            return values
        result = values.copy()
        mask = np.abs(result) < threshold
        result[mask] = 0.0
        result[~mask] -= np.sign(result[~mask]) * threshold
        return result
