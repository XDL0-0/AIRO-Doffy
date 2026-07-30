"""Hardware-independent recording schema definitions."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ModelValidationError

_DATA_TYPE_ALIASES = {
    "qpos": "qpos",
    "joint": "qpos",
    "joint_configuration": "qpos",
    "both": "both",
    "tcp": "tcp",
    "tcp_quat": "tcp",
    "eef": "tcp",
    "delta_tcp": "delta_tcp",
}


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One serialized field, excluding the leading episode dimension."""

    path: str
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordingSchema:
    """Complete dataset layout derived without importing serializer libraries."""

    data_type: str
    robot_dof: int
    camera_names: tuple[str, ...]
    resolution: tuple[int, int]
    tactile_shape: tuple[int, ...] | None = None
    force_enabled: bool = False
    torque_enabled: bool = False
    depth_enabled: bool = False

    def __post_init__(self) -> None:
        data_type = normalize_data_type(self.data_type)
        if isinstance(self.robot_dof, bool) or self.robot_dof <= 0:
            raise ModelValidationError("robot_dof must be a positive integer")
        camera_names = tuple(self.camera_names)
        if any(not name or not isinstance(name, str) for name in camera_names):
            raise ModelValidationError("camera_names must contain non-empty strings")
        if len(set(camera_names)) != len(camera_names):
            raise ModelValidationError("camera_names must be unique")
        resolution = tuple(self.resolution)
        if len(resolution) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolution
        ):
            raise ModelValidationError("resolution must be positive (width, height)")
        tactile_shape = (
            None if self.tactile_shape is None else tuple(self.tactile_shape)
        )
        if tactile_shape is not None and (
            not tactile_shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in tactile_shape
            )
        ):
            raise ModelValidationError("tactile_shape dimensions must be positive")
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "camera_names", camera_names)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "tactile_shape", tactile_shape)
        object.__setattr__(self, "force_enabled", bool(self.force_enabled))
        object.__setattr__(self, "torque_enabled", bool(self.torque_enabled))
        object.__setattr__(self, "depth_enabled", bool(self.depth_enabled))

    @property
    def state_dim(self) -> int:
        return 8 if self.data_type == "tcp" else self.robot_dof + 1

    @property
    def action_dim(self) -> int:
        if self.data_type == "tcp":
            return 8
        if self.data_type == "delta_tcp":
            return 7
        return self.robot_dof + 1

    @property
    def state_names(self) -> tuple[str, ...]:
        if self.data_type == "tcp":
            return ("qx", "qy", "qz", "qw", "x", "y", "z", "gripper")
        return tuple(f"joint_{index}" for index in range(self.robot_dof)) + (
            "gripper",
        )

    @property
    def action_names(self) -> tuple[str, ...]:
        if self.data_type == "tcp":
            return ("qx", "qy", "qz", "qw", "x", "y", "z", "gripper")
        if self.data_type == "delta_tcp":
            return (
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_rotvec_x",
                "delta_rotvec_y",
                "delta_rotvec_z",
                "gripper",
            )
        return tuple(f"joint_{index}" for index in range(self.robot_dof)) + (
            "gripper",
        )

    @property
    def stores_tcp_pose(self) -> bool:
        return self.data_type in {"both", "delta_tcp"}

    @property
    def timestamp_names(self) -> tuple[str, ...]:
        return (
            "collect",
            "robot_state",
            "robot_action",
            "vr_input",
            "tactile",
            *self.camera_names,
        )

    @property
    def image_shape(self) -> tuple[int, int, int]:
        width, height = self.resolution
        return (height, width, 3)

    @property
    def depth_shape(self) -> tuple[int, int]:
        width, height = self.resolution
        return (height, width)

    def hdf5_fields(self) -> tuple[FieldSpec, ...]:
        """Return the legacy ACT/HDF5 paths and their current dtypes."""
        fields = [
            FieldSpec("/observations/qpos", "float64", (self.state_dim,)),
            FieldSpec("/action", "float64", (self.action_dim,)),
            FieldSpec(
                "/extra/timestamps_ns",
                "int64",
                (len(self.timestamp_names),),
                self.timestamp_names,
            ),
        ]
        if self.stores_tcp_pose:
            fields.append(FieldSpec("/extra/tcp_pose", "float64", (7,)))
        if self.force_enabled:
            fields.append(FieldSpec("/observations/force", "float64", (3,)))
        if self.torque_enabled:
            fields.append(FieldSpec("/observations/torque", "float64", (3,)))
        if self.tactile_shape is not None:
            fields.append(
                FieldSpec(
                    "/observations/tactile",
                    "float64",
                    self.tactile_shape,
                )
            )
        for name in self.camera_names:
            fields.append(
                FieldSpec(
                    f"/observations/images/{name}",
                    "uint8",
                    self.image_shape,
                )
            )
        if self.depth_enabled:
            for name in self.camera_names:
                fields.append(
                    FieldSpec(
                        f"/observations/depth/{name}",
                        "float32",
                        self.depth_shape,
                    )
                )
        return tuple(fields)

    def lerobot_features(self) -> dict[str, dict[str, object]]:
        """Return the legacy LeRobot feature declaration."""
        features: dict[str, dict[str, object]] = {
            "action": {
                "dtype": "float32",
                "shape": (self.action_dim,),
                "names": list(self.action_names),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": list(self.state_names),
            },
            "extra.timestamps_ns": {
                "dtype": "int64",
                "shape": (len(self.timestamp_names),),
                "names": list(self.timestamp_names),
            },
        }
        if self.force_enabled:
            features["observation.force"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": ["Fx", "Fy", "Fz"],
            }
        if self.torque_enabled:
            features["observation.torque"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": ["Tx", "Ty", "Tz"],
            }
        if self.stores_tcp_pose:
            features["extra.tcp_pose"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": ["qx", "qy", "qz", "qw", "x", "y", "z"],
            }
        if self.tactile_shape is not None:
            features["observation.tactile"] = {
                "dtype": "float32",
                "shape": self.tactile_shape,
                "names": ["sensor_idx", "axis"],
            }
        for name in self.camera_names:
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channel"],
            }
        if self.depth_enabled:
            height, width = self.depth_shape
            for name in self.camera_names:
                features[f"observation.depth.{name}"] = {
                    "dtype": "image",
                    "shape": (height, width, 1),
                    "names": ["height", "width", "channel"],
                }
        return features


def normalize_data_type(data_type: str) -> str:
    """Normalize all legacy recording representation aliases."""
    try:
        return _DATA_TYPE_ALIASES[data_type.lower()]
    except (AttributeError, KeyError) as exc:
        supported = ", ".join(sorted(_DATA_TYPE_ALIASES))
        raise ModelValidationError(
            f"unsupported recording data_type {data_type!r}; supported: {supported}"
        ) from exc


def build_recording_schema(
    *,
    data_type: str,
    robot_dof: int,
    camera_count: int,
    resolution: tuple[int, int],
    tactile_shape: tuple[int, ...] | None = None,
    force_enabled: bool = False,
    torque_enabled: bool = False,
    depth_enabled: bool = False,
) -> RecordingSchema:
    """Build a schema with the legacy ``camera_<index>`` naming convention."""
    if isinstance(camera_count, bool) or not isinstance(camera_count, int):
        raise ModelValidationError("camera_count must be a non-negative integer")
    if camera_count < 0:
        raise ModelValidationError("camera_count must be non-negative")
    return RecordingSchema(
        data_type=data_type,
        robot_dof=robot_dof,
        camera_names=tuple(f"camera_{index}" for index in range(camera_count)),
        resolution=resolution,
        tactile_shape=tactile_shape,
        force_enabled=force_enabled,
        torque_enabled=torque_enabled,
        depth_enabled=depth_enabled,
    )
