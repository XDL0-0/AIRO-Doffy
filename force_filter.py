"""NumPy compatibility adapter for the v2 pure wrench filter."""

from __future__ import annotations

import numpy as np

from airo_doffy.devices.wrench.filters import WrenchFilter as _WrenchFilter


class WrenchFilter(_WrenchFilter):
    """Keep the root module's NumPy return contract during migration."""

    def process(self, wrench: np.ndarray) -> np.ndarray:
        return np.asarray(super().process(np.asarray(wrench).reshape(-1)), dtype=float)

    @staticmethod
    def _apply_deadband(values: np.ndarray, threshold: float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if threshold <= 0.0:
            return values
        result = values.copy()
        mask = np.abs(result) < threshold
        result[mask] = 0.0
        result[~mask] -= np.sign(result[~mask]) * threshold
        return result
