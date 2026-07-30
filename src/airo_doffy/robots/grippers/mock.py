"""Dependency-free null gripper for disabled hardware and tests."""

from __future__ import annotations

import math
import threading

from ...core.errors import LifecycleError, ModelValidationError


class NullGripper:
    """In-memory gripper that preserves width semantics without I/O."""

    def __init__(self, *, max_width_m: float = 0.085) -> None:
        width = float(max_width_m)
        if not math.isfinite(width) or width <= 0:
            raise ModelValidationError("max_width_m must be positive and finite")
        self._max_width_m = width
        self._width_m = width
        self._started = False
        self._closed = False
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "none"

    @property
    def max_width_m(self) -> float:
        return self._max_width_m

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("null gripper is closed")
        if not self._started:
            raise LifecycleError("null gripper has not been started")

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed null gripper")
            if self._started:
                raise LifecycleError("null gripper is already started")
            self._started = True

    def read_width(self) -> float:
        with self._lock:
            self._require_started()
            return self._width_m

    def move(self, width_m: float) -> None:
        width = float(width_m)
        if not math.isfinite(width):
            raise ModelValidationError("gripper width must be finite")
        with self._lock:
            self._require_started()
            self._width_m = min(max(width, 0.0), self._max_width_m)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True
