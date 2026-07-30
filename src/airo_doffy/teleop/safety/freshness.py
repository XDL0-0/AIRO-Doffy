"""Action freshness and deterministic command-rate filters."""

from __future__ import annotations

import math
import threading

from ...core.errors import ModelValidationError
from ...core.types import ClockDomain, RobotAction, RobotCommandType, RobotState


class ActionFreshnessFilter:
    """Reject actions whose comparable local timestamp is stale or too far ahead."""

    def __init__(
        self,
        max_age_s: float,
        *,
        future_tolerance_s: float = 0.0,
    ) -> None:
        age = float(max_age_s)
        tolerance = float(future_tolerance_s)
        if not math.isfinite(age) or age <= 0:
            raise ModelValidationError("max_age_s must be positive and finite")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ModelValidationError(
                "future_tolerance_s must be non-negative and finite"
            )
        self._max_age_ns = round(age * 1_000_000_000)
        self._future_tolerance_ns = round(tolerance * 1_000_000_000)

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        del robot_state
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ModelValidationError("now_ns must be a non-negative integer")
        if action.receive_timestamp_ns is not None:
            timestamp_ns = action.receive_timestamp_ns
        elif action.clock_domain is ClockDomain.MONOTONIC:
            timestamp_ns = action.source_timestamp_ns
        else:
            return None
        age_ns = now_ns - timestamp_ns
        if age_ns < -self._future_tolerance_ns or age_ns > self._max_age_ns:
            return None
        return action


class ActionRateLimitFilter:
    """Accept no more than one active action per configured interval."""

    def __init__(self, max_rate_hz: float) -> None:
        rate = float(max_rate_hz)
        if not math.isfinite(rate) or rate <= 0:
            raise ModelValidationError("max_rate_hz must be positive and finite")
        self._minimum_interval_ns = max(1, round(1_000_000_000 / rate))
        self._last_accepted_ns: int | None = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._last_accepted_ns = None

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        del robot_state
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ModelValidationError("now_ns must be a non-negative integer")
        if action.command_type in {RobotCommandType.HOLD, RobotCommandType.STOP}:
            return action
        with self._lock:
            if self._last_accepted_ns is not None:
                elapsed = now_ns - self._last_accepted_ns
                if elapsed < self._minimum_interval_ns:
                    return None
            self._last_accepted_ns = now_ns
            return action
