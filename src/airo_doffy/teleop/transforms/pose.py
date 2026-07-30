"""Dependency-free rigid-pose transforms for deterministic teleoperation mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ...core.errors import ModelValidationError

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Transform4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


class RotationComposition(str, Enum):
    """Where a controller-relative rotation is composed on the robot reference."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class PoseDelta:
    """Controller-relative translation and local rotation."""

    translation_m: Vector3
    rotation: Matrix3


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ModelValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ModelValidationError(f"{name} must be finite")
    return result


def vector3(values: Iterable[object], name: str = "vector") -> Vector3:
    """Validate and normalize a three-vector."""

    try:
        result = tuple(
            _finite(value, f"{name}[{index}]") for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise ModelValidationError(f"{name} must contain three numbers") from exc
    if len(result) != 3:
        raise ModelValidationError(f"{name} must contain three numbers")
    return result  # type: ignore[return-value]


def _matrix3(values: Iterable[Iterable[object]], name: str) -> Matrix3:
    try:
        result = tuple(
            vector3(row, f"{name}[{index}]") for index, row in enumerate(values)
        )
    except TypeError as exc:
        raise ModelValidationError(f"{name} must have shape (3, 3)") from exc
    if len(result) != 3:
        raise ModelValidationError(f"{name} must have shape (3, 3)")
    return result  # type: ignore[return-value]


def matmul3(left: Matrix3, right: Matrix3) -> Matrix3:
    """Multiply two 3x3 matrices."""

    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def transpose3(matrix: Matrix3) -> Matrix3:
    """Transpose one 3x3 matrix."""

    return tuple(
        tuple(matrix[column][row] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def determinant3(matrix: Matrix3) -> float:
    """Return the determinant of a 3x3 matrix."""

    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def validate_rotation(values: Iterable[Iterable[object]], name: str = "rotation") -> Matrix3:
    """Validate a proper orthonormal 3x3 rotation."""

    matrix = _matrix3(values, name)
    product = matmul3(transpose3(matrix), matrix)
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            if not math.isclose(product[row][column], expected, abs_tol=1e-7):
                raise ModelValidationError(f"{name} must be orthonormal")
    if not math.isclose(determinant3(matrix), 1.0, abs_tol=1e-7):
        raise ModelValidationError(f"{name} must have determinant +1")
    return matrix


def validate_axis_transform(
    values: Iterable[Iterable[object]],
    name: str = "axis_transform",
) -> Matrix3:
    """Validate an orthonormal axis transform with determinant ±1."""

    matrix = _matrix3(values, name)
    product = matmul3(transpose3(matrix), matrix)
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            if not math.isclose(product[row][column], expected, abs_tol=1e-7):
                raise ModelValidationError(f"{name} must be orthonormal")
    if not math.isclose(abs(determinant3(matrix)), 1.0, abs_tol=1e-7):
        raise ModelValidationError(f"{name} determinant must have magnitude 1")
    return matrix


def _matrix_vector(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def quaternion_xyzw_to_rotation(
    quaternion: Iterable[object],
) -> Matrix3:
    """Convert and normalize an XYZW quaternion."""

    try:
        values = tuple(
            _finite(value, f"quaternion[{index}]")
            for index, value in enumerate(quaternion)
        )
    except TypeError as exc:
        raise ModelValidationError("quaternion must contain four numbers") from exc
    if len(values) != 4:
        raise ModelValidationError("quaternion must contain four numbers")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ModelValidationError("quaternion must not be zero")
    x, y, z, w = (value / norm for value in values)
    return (
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ),
        (
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ),
        (
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
    )


def transform(
    rotation: Iterable[Iterable[object]],
    translation: Iterable[object],
) -> Transform4:
    """Build a validated homogeneous transform."""

    checked_rotation = validate_rotation(rotation)
    checked_translation = vector3(translation, "translation")
    return (
        (*checked_rotation[0], checked_translation[0]),
        (*checked_rotation[1], checked_translation[1]),
        (*checked_rotation[2], checked_translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def validate_transform(
    values: Iterable[Iterable[object]],
    name: str = "transform",
) -> Transform4:
    """Validate a finite rigid 4x4 transform."""

    try:
        rows = tuple(tuple(row) for row in values)
    except TypeError as exc:
        raise ModelValidationError(f"{name} must have shape (4, 4)") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ModelValidationError(f"{name} must have shape (4, 4)")
    checked = tuple(
        tuple(_finite(value, f"{name}[{row}][{column}]") for column, value in enumerate(line))
        for row, line in enumerate(rows)
    )
    if any(
        not math.isclose(checked[3][index], expected, abs_tol=1e-7)
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        raise ModelValidationError(f"{name} must have homogeneous bottom row")
    rotation = validate_rotation(
        tuple(tuple(checked[row][column] for column in range(3)) for row in range(3)),
        f"{name}.rotation",
    )
    return transform(rotation, (checked[0][3], checked[1][3], checked[2][3]))


def flatten_transform(values: Iterable[Iterable[object]]) -> tuple[float, ...]:
    """Return one validated transform in row-major order."""

    checked = validate_transform(values)
    return tuple(value for row in checked for value in row)


def vr_pose_to_transform(
    position_m: Iterable[object],
    orientation_xyzw: Iterable[object],
    axis_transform: Iterable[Iterable[object]],
) -> Transform4:
    """Map a Quest pose through an orthonormal robot-axis transform."""

    axes = validate_axis_transform(axis_transform)
    position = _matrix_vector(axes, vector3(position_m, "position_m"))
    quaternion = tuple(
        _finite(value, f"orientation_xyzw[{index}]")
        for index, value in enumerate(orientation_xyzw)
    )
    if len(quaternion) != 4:
        raise ModelValidationError("orientation_xyzw must contain four numbers")
    sign = 1.0 if determinant3(axes) > 0 else -1.0
    vector = _matrix_vector(
        axes,
        (quaternion[0], quaternion[1], quaternion[2]),
    )
    mapped_quaternion = (
        sign * vector[0],
        sign * vector[1],
        sign * vector[2],
        quaternion[3],
    )
    return transform(quaternion_xyzw_to_rotation(mapped_quaternion), position)


def pose_delta(
    reference: Iterable[Iterable[object]],
    current: Iterable[Iterable[object]],
) -> PoseDelta:
    """Compute world translation and reference-local rotation delta."""

    reference_pose = validate_transform(reference, "reference")
    current_pose = validate_transform(current, "current")
    reference_rotation = tuple(tuple(row[:3]) for row in reference_pose[:3])
    current_rotation = tuple(tuple(row[:3]) for row in current_pose[:3])
    translation = tuple(
        current_pose[index][3] - reference_pose[index][3] for index in range(3)
    )
    rotation = matmul3(transpose3(reference_rotation), current_rotation)
    return PoseDelta(
        translation_m=translation,  # type: ignore[arg-type]
        rotation=validate_rotation(rotation),
    )


def _axis_angle(rotation: Matrix3) -> tuple[Vector3, float]:
    cosine = max(-1.0, min(1.0, (sum(rotation[i][i] for i in range(3)) - 1) / 2))
    angle = math.acos(cosine)
    if angle < 1e-10:
        return (1.0, 0.0, 0.0), 0.0
    if math.pi - angle < 1e-6:
        components = [
            math.sqrt(max(0.0, (rotation[index][index] + 1) / 2))
            for index in range(3)
        ]
        largest = max(range(3), key=components.__getitem__)
        if components[largest] < 1e-8:
            return (1.0, 0.0, 0.0), angle
        if largest == 0:
            components[1] = (rotation[0][1] + rotation[1][0]) / (4 * components[0])
            components[2] = (rotation[0][2] + rotation[2][0]) / (4 * components[0])
        elif largest == 1:
            components[0] = (rotation[0][1] + rotation[1][0]) / (4 * components[1])
            components[2] = (rotation[1][2] + rotation[2][1]) / (4 * components[1])
        else:
            components[0] = (rotation[0][2] + rotation[2][0]) / (4 * components[2])
            components[1] = (rotation[1][2] + rotation[2][1]) / (4 * components[2])
        return vector3(components, "rotation axis"), angle
    scale = 1 / (2 * math.sin(angle))
    return (
        (
            (rotation[2][1] - rotation[1][2]) * scale,
            (rotation[0][2] - rotation[2][0]) * scale,
            (rotation[1][0] - rotation[0][1]) * scale,
        ),
        angle,
    )


def _axis_angle_rotation(axis: Vector3, angle: float) -> Matrix3:
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1e-12 or abs(angle) <= 1e-12:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    x, y, z = (value / norm for value in axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1 - cosine
    return (
        (
            cosine + x * x * one_minus,
            x * y * one_minus - z * sine,
            x * z * one_minus + y * sine,
        ),
        (
            y * x * one_minus + z * sine,
            cosine + y * y * one_minus,
            y * z * one_minus - x * sine,
        ),
        (
            z * x * one_minus - y * sine,
            z * y * one_minus + x * sine,
            cosine + z * z * one_minus,
        ),
    )


def scale_pose_delta(
    delta: PoseDelta,
    *,
    translation_scale: float,
    rotation_scale: float,
) -> PoseDelta:
    """Scale translation linearly and rotation in axis-angle space."""

    translation_factor = _finite(translation_scale, "translation_scale")
    rotation_factor = _finite(rotation_scale, "rotation_scale")
    if translation_factor < 0 or rotation_factor < 0:
        raise ModelValidationError("pose delta scales must be non-negative")
    axis, angle = _axis_angle(validate_rotation(delta.rotation))
    return PoseDelta(
        translation_m=tuple(
            value * translation_factor for value in delta.translation_m
        ),  # type: ignore[arg-type]
        rotation=_axis_angle_rotation(axis, angle * rotation_factor),
    )


def apply_pose_delta(
    robot_reference: Iterable[Iterable[object]],
    delta: PoseDelta,
    *,
    freeze_rotation: bool,
    rotation_composition: RotationComposition | str,
) -> Transform4:
    """Apply a scaled source delta to a robot reference."""

    reference = validate_transform(robot_reference, "robot_reference")
    reference_rotation = tuple(tuple(row[:3]) for row in reference[:3])
    delta_rotation = validate_rotation(delta.rotation, "delta.rotation")
    try:
        composition = RotationComposition(rotation_composition)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("rotation composition must be 'left' or 'right'") from exc
    if freeze_rotation:
        target_rotation = reference_rotation
    elif composition is RotationComposition.LEFT:
        target_rotation = matmul3(delta_rotation, reference_rotation)
    else:
        target_rotation = matmul3(reference_rotation, delta_rotation)
    target_translation = tuple(
        reference[index][3] + delta.translation_m[index] for index in range(3)
    )
    return transform(target_rotation, target_translation)


def map_relative_pose(
    source_reference: Iterable[Iterable[object]],
    source_current: Iterable[Iterable[object]],
    robot_reference: Iterable[Iterable[object]],
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    freeze_rotation: bool = False,
    rotation_composition: RotationComposition | str = RotationComposition.LEFT,
) -> Transform4:
    """Map one source-relative pose to a robot reference in a single pure call."""

    delta = scale_pose_delta(
        pose_delta(source_reference, source_current),
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
    )
    return apply_pose_delta(
        robot_reference,
        delta,
        freeze_rotation=freeze_rotation,
        rotation_composition=rotation_composition,
    )
