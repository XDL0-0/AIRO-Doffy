"""Ordered observable composition of independent action filters."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

from ...core.errors import ModelValidationError
from ...core.types import RobotAction, RobotState
from .base import ActionFilter


@dataclass(frozen=True, slots=True)
class SafetyFilterChainMetrics:
    """Snapshot of chain throughput and per-filter rejection counts."""

    processed: int
    accepted: int
    rejected: int
    last_rejected_by: str | None
    rejections_by_filter: tuple[tuple[str, int], ...]


class SafetyFilterChain:
    """Apply filters in declared order and stop on the first rejection."""

    def __init__(self, filters: Iterable[ActionFilter]) -> None:
        checked = tuple(filters)
        if any(not isinstance(item, ActionFilter) for item in checked):
            raise ModelValidationError("all safety chain entries must satisfy ActionFilter")
        self._filters = checked
        self._names = tuple(type(item).__name__ for item in checked)
        self._lock = threading.Lock()
        self._processed = 0
        self._accepted = 0
        self._rejected = 0
        self._last_rejected_by: str | None = None
        self._rejections = {name: 0 for name in self._names}

    @property
    def filters(self) -> tuple[ActionFilter, ...]:
        return self._filters

    @property
    def metrics(self) -> SafetyFilterChainMetrics:
        with self._lock:
            return SafetyFilterChainMetrics(
                processed=self._processed,
                accepted=self._accepted,
                rejected=self._rejected,
                last_rejected_by=self._last_rejected_by,
                rejections_by_filter=tuple(self._rejections.items()),
            )

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        if not isinstance(action, RobotAction) or not isinstance(robot_state, RobotState):
            raise ModelValidationError("safety chain requires RobotAction and RobotState")
        with self._lock:
            self._processed += 1
        current: RobotAction | None = action
        for name, action_filter in zip(self._names, self._filters):
            assert current is not None
            current = action_filter.apply(current, robot_state, now_ns)
            if current is None:
                with self._lock:
                    self._rejected += 1
                    self._last_rejected_by = name
                    self._rejections[name] += 1
                return None
        with self._lock:
            self._accepted += 1
            self._last_rejected_by = None
        return current
