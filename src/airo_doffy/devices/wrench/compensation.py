"""Pure gravity and sensor-bias compensation for six-axis wrenches."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ...core.errors import ModelValidationError

WrenchValues = tuple[float, float, float, float, float, float]
RotationMatrix = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _vector(values: Iterable[object], size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must contain numbers") from exc
    if len(result) != size:
        raise ModelValidationError(f"{name} must contain {size} values")
    if not all(math.isfinite(value) for value in result):
        raise ModelValidationError(f"{name} must contain only finite values")
    return result


def _rotation(values: Iterable[Iterable[object]]) -> RotationMatrix:
    try:
        rows = tuple(_vector(row, 3, "rotation row") for row in values)
    except TypeError as exc:
        raise ModelValidationError("rotation must be a 3 x 3 matrix") from exc
    if len(rows) != 3:
        raise ModelValidationError("rotation must be a 3 x 3 matrix")
    return rows


class GravityCompensator:
    """Remove base-frame payload gravity and a calibrated sensor bias."""

    GRAVITY_M_S2 = 9.81

    def __init__(
        self,
        mass_kg: float,
        center_of_mass_m: Iterable[object],
        *,
        filter_alpha: float = 0.15,
    ) -> None:
        try:
            mass = float(mass_kg)
            alpha = float(filter_alpha)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("invalid gravity compensation parameter") from exc
        if mass <= 0:
            raise ModelValidationError("mass_kg must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ModelValidationError("filter_alpha must be between 0 and 1")
        self.mass_kg = mass
        self.center_of_mass_m = _vector(
            center_of_mass_m,
            3,
            "center_of_mass_m",
        )
        self.filter_alpha = alpha
        self.force_bias: WrenchValues = (0.0,) * 6
        self.calibrated = False
        self._calibration_samples: list[WrenchValues] = []
        self._filtered: WrenchValues | None = None

    def gravity_wrench(
        self,
        rotation_tool_to_base: Iterable[Iterable[object]],
    ) -> WrenchValues:
        rotation = _rotation(rotation_tool_to_base)
        com = self.center_of_mass_m
        com_base = tuple(
            sum(rotation[row][column] * com[column] for column in range(3))
            for row in range(3)
        )
        force = (0.0, 0.0, -self.mass_kg * self.GRAVITY_M_S2)
        torque = (
            com_base[1] * force[2] - com_base[2] * force[1],
            com_base[2] * force[0] - com_base[0] * force[2],
            com_base[0] * force[1] - com_base[1] * force[0],
        )
        return (*force, *torque)

    def add_calibration_sample(
        self,
        raw_wrench: Iterable[object],
        rotation_tool_to_base: Iterable[Iterable[object]],
    ) -> None:
        raw = _vector(raw_wrench, 6, "raw_wrench")
        gravity = self.gravity_wrench(rotation_tool_to_base)
        self._calibration_samples.append(
            tuple(raw[index] - gravity[index] for index in range(6))
        )

    def finish_calibration(self) -> WrenchValues:
        if not self._calibration_samples:
            return self.force_bias
        count = len(self._calibration_samples)
        self.force_bias = tuple(
            sum(sample[index] for sample in self._calibration_samples) / count
            for index in range(6)
        )
        self._calibration_samples.clear()
        self.calibrated = True
        self._filtered = None
        return self.force_bias

    def reset_baseline(self, bias: Iterable[object] | None = None) -> None:
        """Clear calibration, or atomically replace it with an explicit bias."""

        self.force_bias = (
            (0.0,) * 6
            if bias is None
            else _vector(bias, 6, "bias")
        )
        self.calibrated = bias is not None
        self._calibration_samples.clear()
        self._filtered = None

    def compensate(
        self,
        raw_wrench: Iterable[object],
        rotation_tool_to_base: Iterable[Iterable[object]],
    ) -> WrenchValues:
        raw = _vector(raw_wrench, 6, "raw_wrench")
        gravity = self.gravity_wrench(rotation_tool_to_base)
        contact = tuple(
            raw[index] - gravity[index] - self.force_bias[index]
            for index in range(6)
        )
        if self._filtered is None:
            self._filtered = contact
        else:
            alpha = self.filter_alpha
            self._filtered = tuple(
                alpha * value + (1.0 - alpha) * previous
                for value, previous in zip(contact, self._filtered, strict=True)
            )
        return self._filtered
