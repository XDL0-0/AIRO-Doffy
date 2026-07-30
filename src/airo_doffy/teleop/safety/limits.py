"""Workspace, joint, velocity, and acceleration action filters."""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from typing import Iterable

from ...core.errors import ModelValidationError
from ...core.types import RobotAction, RobotCommandType, RobotState
from ..transforms import (
    RotationComposition,
    flatten_transform,
    map_relative_pose,
    pose_delta,
    validate_transform,
    vector3,
)


def _positive(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ModelValidationError(f"{name} must be positive and finite")
    return result


def _positive_vector(values: Iterable[object], name: str) -> tuple[float, ...]:
    try:
        result = tuple(_positive(value, f"{name}[{index}]") for index, value in enumerate(values))
    except TypeError as exc:
        raise ModelValidationError(f"{name} must contain positive values") from exc
    if len(result) not in {6, 7}:
        raise ModelValidationError(f"{name} must contain 6 or 7 values")
    return result


def _bounds(
    lower: Iterable[object],
    upper: Iterable[object],
    name: str,
    *,
    sizes: set[int],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        lower_values = tuple(float(value) for value in lower)
        upper_values = tuple(float(value) for value in upper)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} bounds must contain finite numbers") from exc
    if (
        len(lower_values) not in sizes
        or len(lower_values) != len(upper_values)
        or any(not math.isfinite(value) for value in (*lower_values, *upper_values))
        or any(low >= high for low, high in zip(lower_values, upper_values))
    ):
        allowed = " or ".join(str(size) for size in sorted(sizes))
        raise ModelValidationError(
            f"{name} bounds must contain {allowed} finite lower<upper pairs"
        )
    return lower_values, upper_values


def _duration(action: RobotAction, fallback: float) -> float:
    return action.duration_s if action.duration_s is not None else fallback


def _matrix(values: tuple[float, ...]):
    return tuple(
        tuple(values[row * 4 + column] for column in range(4)) for row in range(4)
    )


def _clip_norm(values: tuple[float, ...], maximum: float) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= maximum or norm <= 1e-12:
        return values
    factor = maximum / norm
    return tuple(value * factor for value in values)


class WorkspaceBoundsFilter:
    """Reject TCP targets or finite-duration twists outside an axis-aligned box."""

    def __init__(self, minimum_m: Iterable[object], maximum_m: Iterable[object]) -> None:
        lower, upper = _bounds(minimum_m, maximum_m, "workspace", sizes={3})
        self._minimum = vector3(lower)
        self._maximum = vector3(upper)

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        del now_ns
        position: tuple[float, ...] | None = None
        if action.command_type is RobotCommandType.TCP_POSE:
            position = (action.values[3], action.values[7], action.values[11])
        elif (
            action.command_type is RobotCommandType.TCP_TWIST
            and action.duration_s is not None
        ):
            position = tuple(
                robot_state.tcp_pose[index][3]
                + action.values[index] * action.duration_s
                for index in range(3)
            )
        if position is None:
            return action
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(position, self._minimum, self._maximum)
        ):
            return None
        return action


class JointLimitsFilter:
    """Reject joint targets outside configured per-joint physical limits."""

    def __init__(self, lower_rad: Iterable[object], upper_rad: Iterable[object]) -> None:
        self._lower, self._upper = _bounds(
            lower_rad,
            upper_rad,
            "joint",
            sizes={6, 7},
        )

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        del now_ns
        candidate: tuple[float, ...] | None = None
        if action.command_type in {
            RobotCommandType.JOINT_POSITION,
            RobotCommandType.JOINT_VELOCITY,
        } and len(action.values) != len(robot_state.joints_rad):
            return None
        if action.command_type is RobotCommandType.JOINT_POSITION:
            candidate = action.values
        elif (
            action.command_type is RobotCommandType.JOINT_VELOCITY
            and action.duration_s is not None
        ):
            candidate = tuple(
                current + velocity * action.duration_s
                for current, velocity in zip(
                    robot_state.joints_rad,
                    action.values,
                    strict=True,
                )
            )
        if candidate is None:
            return action
        if len(candidate) != len(self._lower):
            return None
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                candidate,
                self._lower,
                self._upper,
                strict=True,
            )
        ):
            return None
        return action


class JointVelocityLimitFilter:
    """Slew-limit joint positions and clip explicit joint velocities."""

    def __init__(
        self,
        max_velocity_rad_s: Iterable[object],
        *,
        default_dt_s: float = 0.01,
    ) -> None:
        self._maximum = _positive_vector(max_velocity_rad_s, "max_velocity_rad_s")
        self._default_dt_s = _positive(default_dt_s, "default_dt_s")

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        del now_ns
        if len(robot_state.joints_rad) != len(self._maximum):
            return None
        if action.command_type is RobotCommandType.JOINT_VELOCITY:
            return replace(
                action,
                values=tuple(
                    min(max(value, -maximum), maximum)
                    for value, maximum in zip(
                        action.values,
                        self._maximum,
                        strict=True,
                    )
                ),
            )
        if action.command_type is not RobotCommandType.JOINT_POSITION:
            return action
        dt = _duration(action, self._default_dt_s)
        values = []
        for target, current, maximum in zip(
            action.values,
            robot_state.joints_rad,
            self._maximum,
            strict=True,
        ):
            delta = math.atan2(math.sin(target - current), math.cos(target - current))
            step = min(max(delta, -maximum * dt), maximum * dt)
            values.append(current + step)
        return replace(action, values=tuple(values))


class CartesianVelocityLimitFilter:
    """Limit TCP pose deltas or twist norms over one command duration."""

    def __init__(
        self,
        *,
        max_linear_m_s: float,
        max_angular_rad_s: float,
        default_dt_s: float = 0.01,
    ) -> None:
        self._linear = _positive(max_linear_m_s, "max_linear_m_s")
        self._angular = _positive(max_angular_rad_s, "max_angular_rad_s")
        self._default_dt_s = _positive(default_dt_s, "default_dt_s")

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        del now_ns
        if action.command_type is RobotCommandType.TCP_TWIST:
            linear = _clip_norm(action.values[:3], self._linear)
            angular = _clip_norm(action.values[3:], self._angular)
            return replace(action, values=(*linear, *angular))
        if action.command_type is not RobotCommandType.TCP_POSE:
            return action
        dt = _duration(action, self._default_dt_s)
        try:
            current = validate_transform(robot_state.tcp_pose)
            target = validate_transform(_matrix(action.values))
        except ModelValidationError:
            return None
        delta = pose_delta(current, target)
        translation_norm = math.sqrt(sum(value * value for value in delta.translation_m))
        rotation_angle = math.acos(
            max(
                -1.0,
                min(1.0, (sum(delta.rotation[i][i] for i in range(3)) - 1) / 2),
            )
        )
        translation_scale = (
            1.0
            if translation_norm <= self._linear * dt
            else self._linear * dt / translation_norm
        )
        rotation_scale = (
            1.0
            if rotation_angle <= self._angular * dt or rotation_angle <= 1e-12
            else self._angular * dt / rotation_angle
        )
        limited = map_relative_pose(
            current,
            target,
            current,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            rotation_composition=RotationComposition.RIGHT,
        )
        return replace(action, values=flatten_transform(limited))


class JointAccelerationLimitFilter:
    """Statefully limit changes in commanded joint velocity."""

    def __init__(
        self,
        max_acceleration_rad_s2: Iterable[object],
        *,
        default_dt_s: float = 0.01,
    ) -> None:
        self._maximum = _positive_vector(
            max_acceleration_rad_s2,
            "max_acceleration_rad_s2",
        )
        self._default_dt_s = _positive(default_dt_s, "default_dt_s")
        self._previous_velocity = (0.0,) * len(self._maximum)
        self._last_now_ns: int | None = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._previous_velocity = (0.0,) * len(self._maximum)
            self._last_now_ns = None

    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None:
        if action.command_type not in {
            RobotCommandType.JOINT_POSITION,
            RobotCommandType.JOINT_VELOCITY,
        }:
            return action
        if len(robot_state.joints_rad) != len(self._maximum):
            return None
        if len(action.values) != len(self._maximum):
            return None
        with self._lock:
            if self._last_now_ns is not None and now_ns < self._last_now_ns:
                return None
            elapsed = (
                None
                if self._last_now_ns is None
                else (now_ns - self._last_now_ns) / 1_000_000_000
            )
            dt = _duration(
                action,
                elapsed if elapsed is not None and elapsed > 0 else self._default_dt_s,
            )
            if action.command_type is RobotCommandType.JOINT_VELOCITY:
                desired_velocity = action.values
            else:
                desired_velocity = tuple(
                    math.atan2(math.sin(target - current), math.cos(target - current)) / dt
                    for target, current in zip(
                        action.values,
                        robot_state.joints_rad,
                        strict=True,
                    )
                )
            limited_velocity = tuple(
                previous
                + min(
                    max(desired - previous, -maximum * dt),
                    maximum * dt,
                )
                for desired, previous, maximum in zip(
                    desired_velocity,
                    self._previous_velocity,
                    self._maximum,
                    strict=True,
                )
            )
            self._previous_velocity = limited_velocity
            self._last_now_ns = now_ns
        if action.command_type is RobotCommandType.JOINT_VELOCITY:
            return replace(action, values=limited_velocity)
        return replace(
            action,
            values=tuple(
                current + velocity * dt
                for current, velocity in zip(
                    robot_state.joints_rad,
                    limited_velocity,
                    strict=True,
                )
            ),
        )
