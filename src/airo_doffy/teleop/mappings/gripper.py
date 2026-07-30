"""Pure controller and hand signals to gripper-width targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

from ...core.errors import ModelValidationError


class GripperDirection(IntEnum):
    """Discrete direction shared by controller and hand mappings."""

    CLOSE = -1
    HOLD = 0
    OPEN = 1


def _positive(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ModelValidationError(f"{name} must be positive and finite")
    return result


def _width(value: object, maximum: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0:
        raise ModelValidationError(f"{name} must be finite and non-negative")
    return min(result, maximum)


@dataclass(frozen=True, slots=True)
class IncrementalGripperMapper:
    """Integrate a discrete direction at a configured physical speed."""

    speed_m_s: float
    max_width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "speed_m_s", _positive(self.speed_m_s, "speed_m_s"))
        object.__setattr__(
            self,
            "max_width_m",
            _positive(self.max_width_m, "max_width_m"),
        )

    def target(
        self,
        direction: GripperDirection | int,
        *,
        current_width_m: float,
        dt_s: float,
    ) -> float:
        """Return a clamped width target without touching gripper hardware."""

        try:
            checked_direction = GripperDirection(direction)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("gripper direction must be close, hold, or open") from exc
        current = _width(current_width_m, self.max_width_m, "current_width_m")
        dt = _positive(dt_s, "dt_s")
        return min(
            self.max_width_m,
            max(0.0, current + checked_direction.value * self.speed_m_s * dt),
        )


@dataclass(frozen=True, slots=True)
class ControllerGripperMapping:
    """Map legacy negative joystick-Y convention to a width target."""

    integrator: IncrementalGripperMapper
    deadzone: float = 0.7

    def __post_init__(self) -> None:
        deadzone = float(self.deadzone)
        if not math.isfinite(deadzone) or not 0 <= deadzone <= 1:
            raise ModelValidationError("controller gripper deadzone must be within [0, 1]")
        object.__setattr__(self, "deadzone", deadzone)

    def direction(self, joystick_y: float) -> GripperDirection:
        value = -float(joystick_y)
        if not math.isfinite(value):
            raise ModelValidationError("joystick_y must be finite")
        if value > self.deadzone:
            return GripperDirection.OPEN
        if value < -self.deadzone:
            return GripperDirection.CLOSE
        return GripperDirection.HOLD

    def target(self, joystick_y: float, *, current_width_m: float, dt_s: float) -> float:
        return self.integrator.target(
            self.direction(joystick_y),
            current_width_m=current_width_m,
            dt_s=dt_s,
        )


@dataclass(frozen=True, slots=True)
class HandGripperMapping:
    """Map thumb-index distance with a hold band to a width target."""

    integrator: IncrementalGripperMapper
    open_distance_m: float
    close_distance_m: float

    def __post_init__(self) -> None:
        open_distance = _positive(self.open_distance_m, "open_distance_m")
        close_distance = _positive(self.close_distance_m, "close_distance_m")
        if close_distance >= open_distance:
            raise ModelValidationError(
                "hand close distance must be smaller than open distance"
            )
        object.__setattr__(self, "open_distance_m", open_distance)
        object.__setattr__(self, "close_distance_m", close_distance)

    def direction(self, finger_distance_m: float) -> GripperDirection:
        distance = _width(
            finger_distance_m,
            float("inf"),
            "finger_distance_m",
        )
        if distance > self.open_distance_m:
            return GripperDirection.OPEN
        if distance < self.close_distance_m:
            return GripperDirection.CLOSE
        return GripperDirection.HOLD

    def target(
        self,
        finger_distance_m: float,
        *,
        current_width_m: float,
        dt_s: float,
    ) -> float:
        return self.integrator.target(
            self.direction(finger_distance_m),
            current_width_m=current_width_m,
            dt_s=dt_s,
        )
