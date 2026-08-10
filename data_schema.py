"""Robot state/action schema helpers for recording and policy I/O."""

from __future__ import annotations

from dataclasses import dataclass


JOINT_DATA_TYPES = {"qpos", "joint", "joint_configuration"}
TCP_DATA_TYPES = {"tcp", "tcp_quat", "eef"}
DELTA_TCP_DATA_TYPES = {"delta_tcp"}


@dataclass(frozen=True)
class DataSchema:
    data_type: str
    dof: int
    state_dim: int
    action_dim: int
    state_names: list[str]
    action_names: list[str]
    tcp_pose_dim: int = 7
    gripper_dim: int = 1


def normalize_data_type(data_type: str) -> str:
    aliases = {
        "qpos": "qpos",
        "joint": "qpos",
        "joint_configuration": "qpos",
        "both": "both",
        "tcp": "tcp",
        "tcp_quat": "tcp",
        "eef": "tcp",
        "delta_tcp": "delta_tcp",
    }
    try:
        return aliases[data_type]
    except KeyError as exc:
        supported = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported DATA_TYPE '{data_type}'. Supported: {supported}") from exc


def state_representation(data_type: str) -> str:
    normalized = normalize_data_type(data_type)
    if normalized == "tcp":
        return "tcp"
    return "joint"


def action_representation(data_type: str) -> str:
    normalized = normalize_data_type(data_type)
    if normalized == "tcp":
        return "tcp"
    if normalized == "delta_tcp":
        return "delta_tcp"
    return "joint"


def should_store_extra_tcp_pose(data_type: str) -> bool:
    normalized = normalize_data_type(data_type)
    return normalized in {"both", "delta_tcp"}


def build_data_schema(
    data_type: str,
    dof: int,
    gripper: bool = True,
) -> DataSchema:
    normalized = normalize_data_type(data_type)
    gripper_names = ["gripper"] if gripper else []
    gripper_dim = int(gripper)
    joint_dim = int(dof) + gripper_dim
    tcp_dim = 7 + gripper_dim
    delta_tcp_dim = 6 + gripper_dim

    if normalized == "tcp":
        state_dim = tcp_dim
        action_dim = tcp_dim
        state_names = ["qx", "qy", "qz", "qw", "x", "y", "z"] + gripper_names
        action_names = ["qx", "qy", "qz", "qw", "x", "y", "z"] + gripper_names
    elif normalized == "delta_tcp":
        state_dim = joint_dim
        action_dim = delta_tcp_dim
        state_names = [f"joint_{i}" for i in range(dof)] + gripper_names
        action_names = [
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_rotvec_x",
            "delta_rotvec_y",
            "delta_rotvec_z",
        ] + gripper_names
    else:
        state_dim = joint_dim
        action_dim = joint_dim
        state_names = [f"joint_{i}" for i in range(dof)] + gripper_names
        action_names = [f"joint_{i}" for i in range(dof)] + gripper_names

    return DataSchema(
        data_type=normalized,
        dof=int(dof),
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=state_names,
        action_names=action_names,
        gripper_dim=gripper_dim,
    )
