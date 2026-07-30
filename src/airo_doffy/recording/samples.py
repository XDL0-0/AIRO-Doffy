"""Immutable, hardware-independent episode samples and buffers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import LifecycleError, ModelValidationError
from .schema import RecordingSchema

_DTYPE_SIZES = {
    "float32": 4,
    "float64": 8,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
}


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _float_tuple(values: object, size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must contain numeric values") from exc
    if len(result) != size:
        raise ModelValidationError(f"{name} must have shape ({size},)")
    return result


def _int_tuple(values: object, size: int, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must contain integer values") from exc
    if len(result) != size or any(value < 0 for value in result):
        raise ModelValidationError(
            f"{name} must have shape ({size},) and contain non-negative values"
        )
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenArray:
    """Detached contiguous array represented without NumPy."""

    data: bytes
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        try:
            data = bytes(self.data)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("array data must support the bytes protocol") from exc
        shape = tuple(self.shape)
        if not shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        ):
            raise ModelValidationError("array shape dimensions must be positive")
        if self.dtype not in _DTYPE_SIZES:
            supported = ", ".join(sorted(_DTYPE_SIZES))
            raise ModelValidationError(f"unsupported array dtype; supported: {supported}")
        expected = _product(shape) * _DTYPE_SIZES[self.dtype]
        if len(data) != expected:
            raise ModelValidationError(
                f"array has {len(data)} bytes; {self.dtype} {shape} requires {expected}"
            )
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedArray:
    """Array associated with a stable camera or sensor name."""

    name: str
    value: FrozenArray

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ModelValidationError("array name must be a non-empty string")
        if not isinstance(self.value, FrozenArray):
            raise ModelValidationError("value must be a FrozenArray")


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordingSample:
    """One detached sample ready for buffering or serialization."""

    state: tuple[float, ...]
    action: tuple[float, ...]
    timestamps_ns: tuple[int, ...]
    tcp_pose: tuple[float, ...] | None = None
    force: tuple[float, ...] | None = None
    torque: tuple[float, ...] | None = None
    tactile: FrozenArray | None = None
    images: tuple[NamedArray, ...] = ()
    depths: tuple[NamedArray, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", tuple(self.state))
        object.__setattr__(self, "action", tuple(self.action))
        object.__setattr__(self, "timestamps_ns", tuple(self.timestamps_ns))
        object.__setattr__(
            self,
            "tcp_pose",
            None if self.tcp_pose is None else tuple(self.tcp_pose),
        )
        object.__setattr__(
            self,
            "force",
            None if self.force is None else tuple(self.force),
        )
        object.__setattr__(
            self,
            "torque",
            None if self.torque is None else tuple(self.torque),
        )
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "depths", tuple(self.depths))


@dataclass(frozen=True, slots=True, kw_only=True)
class Episode:
    """An immutable, indexed episode detached from the control loop."""

    index: int
    task: str
    schema: RecordingSchema
    samples: tuple[RecordingSample, ...]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ModelValidationError("episode index must be a non-negative integer")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ModelValidationError("episode task must be a non-empty string")
        samples = tuple(self.samples)
        if not samples:
            raise ModelValidationError("episode must contain at least one sample")
        object.__setattr__(self, "samples", samples)


class SampleBuffer:
    """Mutable episode buffer with strict schema validation and a seal boundary."""

    def __init__(
        self,
        schema: RecordingSchema,
        *,
        capacity: int | None = None,
    ) -> None:
        if capacity is not None and (
            isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0
        ):
            raise ModelValidationError("capacity must be a positive integer or None")
        self._schema = schema
        self._capacity = capacity
        self._samples: list[RecordingSample] = []
        self._sealed = False

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def append(self, sample: RecordingSample) -> None:
        if self._sealed:
            raise LifecycleError("cannot append to a sealed episode buffer")
        if self._capacity is not None and len(self._samples) >= self._capacity:
            raise BufferError("episode sample buffer is full")
        self._validate(sample)
        self._samples.append(sample)

    def clear(self) -> None:
        if self._sealed:
            raise LifecycleError("cannot clear a sealed episode buffer")
        self._samples.clear()

    def seal(self, *, index: int, task: str) -> Episode:
        if self._sealed:
            raise LifecycleError("episode buffer is already sealed")
        if not self._samples:
            raise LifecycleError("cannot seal an empty episode")
        self._sealed = True
        return Episode(
            index=index,
            task=task,
            schema=self._schema,
            samples=tuple(self._samples),
        )

    def _validate(self, sample: RecordingSample) -> None:
        if not isinstance(sample, RecordingSample):
            raise ModelValidationError("sample must be a RecordingSample")
        schema = self._schema
        _float_tuple(sample.state, schema.state_dim, "state")
        _float_tuple(sample.action, schema.action_dim, "action")
        _int_tuple(
            sample.timestamps_ns,
            len(schema.timestamp_names),
            "timestamps_ns",
        )
        self._validate_optional_vector(
            sample.tcp_pose,
            required=schema.stores_tcp_pose,
            size=7,
            name="tcp_pose",
        )
        self._validate_optional_vector(
            sample.force,
            required=schema.force_enabled,
            size=3,
            name="force",
        )
        self._validate_optional_vector(
            sample.torque,
            required=schema.torque_enabled,
            size=3,
            name="torque",
        )
        if (sample.tactile is None) != (schema.tactile_shape is None):
            raise ModelValidationError("tactile presence must match the recording schema")
        if sample.tactile is not None and (
            sample.tactile.shape != schema.tactile_shape
            or sample.tactile.dtype != "float32"
        ):
            raise ModelValidationError(
                "tactile must be float32 with the configured tactile shape"
            )
        self._validate_named_arrays(
            sample.images,
            expected_names=schema.camera_names,
            expected_shape=schema.image_shape,
            expected_dtype="uint8",
            name="images",
        )
        expected_depth_names = schema.camera_names if schema.depth_enabled else ()
        self._validate_named_arrays(
            sample.depths,
            expected_names=expected_depth_names,
            expected_shape=schema.depth_shape,
            expected_dtype="float32",
            name="depths",
        )

    @staticmethod
    def _validate_optional_vector(
        values: tuple[float, ...] | None,
        *,
        required: bool,
        size: int,
        name: str,
    ) -> None:
        if (values is not None) != required:
            raise ModelValidationError(f"{name} presence must match the recording schema")
        if values is not None:
            _float_tuple(values, size, name)

    @staticmethod
    def _validate_named_arrays(
        arrays: tuple[NamedArray, ...],
        *,
        expected_names: tuple[str, ...],
        expected_shape: tuple[int, ...],
        expected_dtype: str,
        name: str,
    ) -> None:
        actual_names = tuple(item.name for item in arrays)
        if actual_names != expected_names:
            raise ModelValidationError(
                f"{name} names must be {expected_names}, got {actual_names}"
            )
        for item in arrays:
            if (
                item.value.shape != expected_shape
                or item.value.dtype != expected_dtype
            ):
                raise ModelValidationError(
                    f"{name}.{item.name} must be {expected_dtype} {expected_shape}"
                )
