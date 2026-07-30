"""Immutable, dependency-light configuration sections for v2 components."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.errors import ModelValidationError


def _text(value: str | None, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{name} must be a non-empty string")
    return value


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ModelValidationError(f"{name} must be positive and finite")
    return result


def _alpha(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ModelValidationError(f"{name} must be within [0, 1]")
    return result


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _port(value: int, name: str) -> int:
    result = _integer(value, name, 1)
    if result > 65535:
        raise ModelValidationError(f"{name} must be <= 65535")
    return result


def _vector(values, size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must contain {size} numbers") from exc
    if len(result) != size or any(not math.isfinite(value) for value in result):
        raise ModelValidationError(f"{name} must contain {size} finite numbers")
    return result


def _matrix3(values, name: str) -> tuple[tuple[float, ...], ...]:
    try:
        result = tuple(_vector(row, 3, f"{name}[{index}]") for index, row in enumerate(values))
    except TypeError as exc:
        raise ModelValidationError(f"{name} must have shape (3, 3)") from exc
    if len(result) != 3:
        raise ModelValidationError(f"{name} must have shape (3, 3)")
    for i in range(3):
        for j in range(3):
            dot = sum(result[k][i] * result[k][j] for k in range(3))
            if not math.isclose(dot, 1.0 if i == j else 0.0, abs_tol=1e-8):
                raise ModelValidationError(f"{name} must be orthogonal")
    return result


def _data_type(value: str) -> str:
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
        return aliases[value.lower()]
    except (AttributeError, KeyError) as exc:
        raise ModelValidationError(f"unsupported recording data_type: {value!r}") from exc


def _tcp_transform(pose: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    x, y, z, rx, ry, rz = pose
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle == 0:
        rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    else:
        kx, ky, kz = rx / angle, ry / angle, rz / angle
        c, s, v = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
        rotation = (
            (c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s),
            (ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s),
            (kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v),
        )
    return (
        (*rotation[0], x),
        (*rotation[1], y),
        (*rotation[2], z),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class NetworkConfig:
    """Host addressing and legacy port compatibility."""

    pc_ip: str | None = None
    vr_ip: str | None = None
    legacy_base_port: int = 8000
    pose_port: int = 8001
    control_port: int = 8005
    signaling_port: int = 8765
    video_rtp_port: int = 5004
    state_diagnostic_port: int = 5005

    def __post_init__(self) -> None:
        object.__setattr__(self, "pc_ip", _text(self.pc_ip, "pc_ip", optional=True))
        object.__setattr__(self, "vr_ip", _text(self.vr_ip, "vr_ip", optional=True))
        for name in (
            "legacy_base_port",
            "pose_port",
            "control_port",
            "signaling_port",
            "video_rtp_port",
            "state_diagnostic_port",
        ):
            object.__setattr__(self, name, _port(getattr(self, name), name))


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotConfig:
    """Robot identity, connection, capabilities, and RealMan timing."""

    robot_type: str = "ur3e"
    ip: str | None = None
    realman_port: int = 8080
    realman_read_retries: int = 3
    realman_retry_delay_s: float = 0.05
    realman_control_rate_hz: int = 200
    realman_min_canfd_rate_hz: float = 100.0
    realman_rate_check_window_s: float = 1.0
    realman_rate_failure_windows: int = 3
    realman_heartbeat_timeout_s: float = 0.05
    realman_sensor_rate_hz: float = 30.0
    realman_vr_timeout_s: float = 0.25
    realman_max_joint_speed_rad_s: float = 2.0
    realman_max_linear_speed_m_s: float = 0.25
    realman_max_angular_speed_rad_s: float = 1.0
    realman_canfd_trajectory_mode: int = 0
    realman_canfd_radio: int = 0
    realman_realtime_state_push: bool = True
    realman_state_push_cycle_ms: int = 5
    realman_state_push_port: int = 8098
    realman_state_push_timeout_s: float = 2.0
    realman_force_coordinate: int = 0
    torque_mode: bool = False
    gripper_enabled: bool = False
    gripper_speed_m_s: float = 0.1
    gripper_max_width_m: float = 0.085
    initial_joints_rad: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        robot_type = self.robot_type.lower()
        if robot_type not in {"ur3e", "ur5e", "realman"}:
            raise ModelValidationError(f"unsupported robot_type: {self.robot_type!r}")
        if self.torque_mode and robot_type == "realman":
            raise ModelValidationError("torque_mode is currently supported only for UR robots")
        initial = self.initial_joints_rad
        if initial is None:
            initial = (
                (
                    2.65586749,
                    -0.06628761,
                    -0.14056882,
                    -1.26216978,
                    0.11116002,
                    -1.11919238,
                    -0.45881216,
                )
                if robot_type == "realman"
                else (1.57, -1.57, 1.57, -1.57, -1.57, 0.0)
            )
        initial = _vector(initial, 7 if robot_type == "realman" else 6, "initial_joints_rad")
        if self.realman_min_canfd_rate_hz < 100:
            raise ModelValidationError("realman_min_canfd_rate_hz must be at least 100")
        if self.realman_control_rate_hz <= self.realman_min_canfd_rate_hz:
            raise ModelValidationError(
                "realman_control_rate_hz must be greater than realman_min_canfd_rate_hz"
            )
        if self.realman_heartbeat_timeout_s <= 0.01:
            raise ModelValidationError("realman_heartbeat_timeout_s must exceed 10 ms")
        mode = _integer(self.realman_canfd_trajectory_mode, "realman_canfd_trajectory_mode")
        radio = _integer(self.realman_canfd_radio, "realman_canfd_radio")
        if mode not in {0, 1, 2}:
            raise ModelValidationError("realman_canfd_trajectory_mode must be 0, 1, or 2")
        if (mode == 1 and radio > 100) or (mode == 2 and radio > 999):
            raise ModelValidationError("realman_canfd_radio exceeds the selected mode limit")
        cycle = _integer(self.realman_state_push_cycle_ms, "realman_state_push_cycle_ms", 5)
        if cycle % 5:
            raise ModelValidationError("realman_state_push_cycle_ms must be a multiple of 5")
        object.__setattr__(self, "robot_type", robot_type)
        object.__setattr__(self, "ip", _text(self.ip, "ip", optional=True))
        object.__setattr__(self, "realman_port", _port(self.realman_port, "realman_port"))
        object.__setattr__(
            self,
            "realman_read_retries",
            _integer(self.realman_read_retries, "realman_read_retries", 1),
        )
        if self.realman_retry_delay_s < 0:
            raise ModelValidationError("realman_retry_delay_s cannot be negative")
        for name in (
            "realman_rate_check_window_s",
            "realman_sensor_rate_hz",
            "realman_vr_timeout_s",
            "realman_max_joint_speed_rad_s",
            "realman_max_linear_speed_m_s",
            "realman_max_angular_speed_rad_s",
            "realman_state_push_timeout_s",
            "gripper_speed_m_s",
            "gripper_max_width_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "realman_rate_failure_windows",
            _integer(self.realman_rate_failure_windows, "realman_rate_failure_windows", 1),
        )
        object.__setattr__(
            self,
            "realman_state_push_port",
            _port(self.realman_state_push_port, "realman_state_push_port"),
        )
        if self.realman_force_coordinate not in {0, 1, 2}:
            raise ModelValidationError("realman_force_coordinate must be 0, 1, or 2")
        object.__setattr__(self, "initial_joints_rad", initial)


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraConfig:
    """Camera acquisition and depth settings."""

    resolution: tuple[int, int] = (640, 480)
    fps: int = 60
    capture_rate_hz: float = 30.0
    depth_enabled: bool = False
    stream_id: str = "camera_0"
    serial_number: str | None = None
    max_consecutive_errors: int = 10
    retry_delay_s: float = 1.0

    def __post_init__(self) -> None:
        try:
            width, height = self.resolution
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("resolution must contain width and height") from exc
        object.__setattr__(
            self,
            "resolution",
            (_integer(width, "resolution.width", 1), _integer(height, "resolution.height", 1)),
        )
        object.__setattr__(self, "fps", _integer(self.fps, "fps", 1))
        object.__setattr__(
            self,
            "capture_rate_hz",
            _positive(self.capture_rate_hz, "capture_rate_hz"),
        )
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        object.__setattr__(
            self,
            "serial_number",
            _text(self.serial_number, "serial_number", optional=True),
        )
        object.__setattr__(
            self,
            "max_consecutive_errors",
            _integer(self.max_consecutive_errors, "max_consecutive_errors", 1),
        )
        object.__setattr__(
            self,
            "retry_delay_s",
            _positive(self.retry_delay_s, "retry_delay_s"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VRConfig:
    """VR source mode without transport or mapping behavior."""

    tracking_mode: str = "controller"

    def __post_init__(self) -> None:
        mode = self.tracking_mode.lower()
        if mode not in {"controller", "hand"}:
            raise ModelValidationError("tracking_mode must be 'controller' or 'hand'")
        object.__setattr__(self, "tracking_mode", mode)


@dataclass(frozen=True, slots=True, kw_only=True)
class TeleopConfig:
    """Mapping, control-rate, filter, and trajectory settings."""

    command_mode: str = "joint"
    rotation_composition: str = "left"
    freeze_rotation: bool = True
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    fine_translation_scale: float = 0.3
    fine_rotation_scale: float = 0.4
    controller_grip_threshold: float = 0.0
    controller_gripper_deadzone: float = 0.7
    ur_control_rate_hz: int = 60
    controller_reset_trigger_threshold: float = 0.8
    hand_palm_jump_threshold_m: float = 0.15
    hand_gripper_open_distance_m: float = 0.06
    hand_gripper_close_distance_m: float = 0.03
    hand_mode_toggle_distance_m: float = 0.02
    hand_reset_distance_m: float = 0.02
    vr_to_robot_axes: tuple[tuple[float, ...], ...] = (
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
    tcp_pose: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    move_threshold: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9, 1.4, 1.4)
    ruckig_enabled: bool = True
    ruckig_max_velocity: tuple[float, ...] = (2.5, 2.5, 2.5, 3.0, 4.0, 4.0)
    ruckig_max_acceleration: tuple[float, ...] = (15.0, 15.0, 15.0, 18.0, 25.0, 25.0)
    ruckig_max_jerk: tuple[float, ...] = (150.0, 150.0, 150.0, 180.0, 250.0, 250.0)
    cartesian_position_filter_hz: float = 8.0
    cartesian_rotation_filter_hz: float = 6.0
    hand_joint_filter_hz: float = 10.0

    def __post_init__(self) -> None:
        mode = self.command_mode.lower()
        if mode not in {"joint", "tcp"}:
            raise ModelValidationError("command_mode must be 'joint' or 'tcp'")
        object.__setattr__(self, "command_mode", mode)
        composition = self.rotation_composition.lower()
        if composition not in {"left", "right"}:
            raise ModelValidationError("rotation_composition must be 'left' or 'right'")
        object.__setattr__(self, "rotation_composition", composition)
        for name in (
            "translation_scale",
            "rotation_scale",
            "fine_translation_scale",
            "fine_rotation_scale",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "controller_grip_threshold",
            _alpha(self.controller_grip_threshold, "controller_grip_threshold"),
        )
        object.__setattr__(
            self,
            "controller_gripper_deadzone",
            _alpha(self.controller_gripper_deadzone, "controller_gripper_deadzone"),
        )
        object.__setattr__(
            self,
            "ur_control_rate_hz",
            _integer(self.ur_control_rate_hz, "ur_control_rate_hz", 1),
        )
        object.__setattr__(
            self,
            "controller_reset_trigger_threshold",
            _alpha(self.controller_reset_trigger_threshold, "controller_reset_trigger_threshold"),
        )
        for name in (
            "hand_palm_jump_threshold_m",
            "hand_gripper_open_distance_m",
            "hand_gripper_close_distance_m",
            "hand_mode_toggle_distance_m",
            "hand_reset_distance_m",
            "cartesian_position_filter_hz",
            "cartesian_rotation_filter_hz",
            "hand_joint_filter_hz",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.hand_gripper_close_distance_m >= self.hand_gripper_open_distance_m:
            raise ModelValidationError("hand close distance must be smaller than open distance")
        object.__setattr__(
            self,
            "vr_to_robot_axes",
            _matrix3(self.vr_to_robot_axes, "vr_to_robot_axes"),
        )
        object.__setattr__(self, "tcp_pose", _vector(self.tcp_pose, 6, "tcp_pose"))
        object.__setattr__(
            self,
            "move_threshold",
            _vector(self.move_threshold, 6, "move_threshold"),
        )
        for name in ("ruckig_max_velocity", "ruckig_max_acceleration", "ruckig_max_jerk"):
            values = _vector(getattr(self, name), 6, name)
            if any(value <= 0 for value in values):
                raise ModelValidationError(f"{name} values must be positive")
            object.__setattr__(self, name, values)

    @property
    def tcp_transform(self) -> tuple[tuple[float, ...], ...]:
        return _tcp_transform(self.tcp_pose)


@dataclass(frozen=True, slots=True, kw_only=True)
class TactileConfig:
    """Supported 4-taxel BLE sensor settings."""

    enabled: bool = True
    transfer_enabled: bool = False
    port: int = 8012
    backend: str = "ble4"
    shape: tuple[int, int] = (4, 3)
    ble_device_mac: str = "ARDUINO7"
    ble_hci: str = "hci0"
    ble_window_size: int = 100
    filter_alpha: float = 0.75
    use_kalman: bool = False
    kalman_q: float = 0.02
    kalman_r: float = 0.02
    deadband_sigma: float = 3.0
    noise_floor: float = 2.0
    max_delta: float = 10000.0
    max_abs: float = 20000.0
    baseline_drift_alpha: float = 0.0
    baseline_drift_threshold: float = 80.0

    def __post_init__(self) -> None:
        if self.backend.lower() != "ble4" or tuple(self.shape) != (4, 3):
            raise ModelValidationError("supported tactile backend/shape is ble4 with (4, 3)")
        object.__setattr__(self, "backend", "ble4")
        object.__setattr__(self, "shape", (4, 3))
        object.__setattr__(self, "port", _port(self.port, "port"))
        object.__setattr__(self, "ble_device_mac", _text(self.ble_device_mac, "ble_device_mac"))
        object.__setattr__(self, "ble_hci", _text(self.ble_hci, "ble_hci"))
        object.__setattr__(
            self,
            "ble_window_size",
            _integer(self.ble_window_size, "ble_window_size", 1),
        )
        object.__setattr__(self, "filter_alpha", _alpha(self.filter_alpha, "filter_alpha"))
        object.__setattr__(
            self,
            "baseline_drift_alpha",
            _alpha(self.baseline_drift_alpha, "baseline_drift_alpha"),
        )
        for name in (
            "kalman_q",
            "kalman_r",
            "deadband_sigma",
            "noise_floor",
            "max_delta",
            "max_abs",
            "baseline_drift_threshold",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordingConfig:
    """Dataset identity, representation, and export behavior."""

    task_name: str = "pick_and_place"
    dataset_dir: str = "./datasets/pnp_long"
    dataset_type: str = "l"
    push_to_hub: bool = False
    save_eef: bool = False
    data_type: str = "both"

    def __post_init__(self) -> None:
        dataset_type = self.dataset_type.lower()
        if dataset_type not in {"a", "l"}:
            raise ModelValidationError("dataset_type must be 'a' or 'l'")
        object.__setattr__(self, "task_name", _text(self.task_name, "task_name"))
        object.__setattr__(self, "dataset_dir", _text(self.dataset_dir, "dataset_dir"))
        object.__setattr__(self, "dataset_type", dataset_type)
        object.__setattr__(self, "data_type", _data_type(self.data_type))


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualizationConfig:
    """Visualizer enablement and bounded display history."""

    enabled: bool = True
    hz: float = 30.0
    window_s: float = 8.0
    force_panel_range: float = 30.0

    def __post_init__(self) -> None:
        for name in ("hz", "window_s", "force_panel_range"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoStreamingConfig:
    """Selected video path, encoder policy, and bounded transport settings."""

    transport: str = "webrtc"
    jpeg_quality: int = 100
    legacy_chunk_size: int = 60000
    encoder_backend: str = "auto"
    bitrate_bps: int = 4_000_000
    gop_frames: int = 30
    target_fps: int = 30
    input_queue_capacity: int = 1
    output_queue_capacity: int = 1
    rtp_mtu: int = 1200
    rtp_payload_type: int = 96
    max_frame_age_s: float = 0.25

    def __post_init__(self) -> None:
        transport = self.transport.lower()
        if transport not in {"webrtc", "udp", "rtp_udp"}:
            raise ModelValidationError(
                "video transport must be 'webrtc', 'udp', or 'rtp_udp'"
            )
        backend = self.encoder_backend.lower()
        if backend not in {"auto", "nvenc", "software"}:
            raise ModelValidationError(
                "encoder_backend must be 'auto', 'nvenc', or 'software'"
            )
        quality = _integer(self.jpeg_quality, "jpeg_quality", 1)
        if quality > 100:
            raise ModelValidationError("jpeg_quality must be <= 100")
        payload_type = _integer(self.rtp_payload_type, "rtp_payload_type")
        if payload_type > 127:
            raise ModelValidationError("rtp_payload_type must be <= 127")
        mtu = _integer(self.rtp_mtu, "rtp_mtu", 256)
        if mtu > 65507:
            raise ModelValidationError("rtp_mtu must be <= 65507")
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "encoder_backend", backend)
        object.__setattr__(self, "jpeg_quality", quality)
        object.__setattr__(
            self,
            "legacy_chunk_size",
            _integer(self.legacy_chunk_size, "legacy_chunk_size", 1),
        )
        for name in (
            "bitrate_bps",
            "gop_frames",
            "target_fps",
            "input_queue_capacity",
            "output_queue_capacity",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, 1),
            )
        object.__setattr__(self, "rtp_mtu", mtu)
        object.__setattr__(self, "rtp_payload_type", payload_type)
        object.__setattr__(
            self,
            "max_frame_age_s",
            _positive(self.max_frame_age_s, "max_frame_age_s"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StateTransportConfig:
    """Latest-only high-frequency state channel policy."""

    transport: str = "webrtc"
    channel_label: str = "realtime_state"
    ordered: bool = False
    max_retransmits: int = 0

    def __post_init__(self) -> None:
        transport = self.transport.lower()
        if transport not in {"webrtc", "udp"}:
            raise ModelValidationError("state transport must be 'webrtc' or 'udp'")
        if self.ordered or self.max_retransmits != 0:
            raise ModelValidationError(
                "state transport must be unordered with max_retransmits=0"
            )
        object.__setattr__(self, "transport", transport)
        object.__setattr__(
            self,
            "channel_label",
            _text(self.channel_label, "channel_label"),
        )
        object.__setattr__(
            self,
            "max_retransmits",
            _integer(self.max_retransmits, "max_retransmits"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandTransportConfig:
    """Reliable ordered runtime-command channel policy."""

    transport: str = "webrtc"
    channel_label: str = "commands"
    ordered: bool = True
    reliable: bool = True
    ack_timeout_s: float = 1.0
    dedupe_capacity: int = 1024

    def __post_init__(self) -> None:
        if self.transport.lower() != "webrtc" or not self.ordered or not self.reliable:
            raise ModelValidationError("command transport must be reliable ordered WebRTC")
        object.__setattr__(self, "transport", "webrtc")
        object.__setattr__(
            self,
            "channel_label",
            _text(self.channel_label, "channel_label"),
        )
        object.__setattr__(
            self,
            "ack_timeout_s",
            _positive(self.ack_timeout_s, "ack_timeout_s"),
        )
        object.__setattr__(
            self,
            "dedupe_capacity",
            _integer(self.dedupe_capacity, "dedupe_capacity", 1),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class WrenchConfig:
    """Wrench collection, compensation, and filtering."""

    force_collect: bool = False
    torque_collect: bool = False
    gravity_compensation: bool = False
    tool_mass_kg: float = 0.925
    tool_com_m: tuple[float, ...] = (0.0, 0.0, 0.058)
    gravity_filter_alpha: float = 0.15
    moving_average_window: int = 8
    low_pass_alpha: float = 0.15
    calibration_samples: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_mass_kg", _positive(self.tool_mass_kg, "tool_mass_kg"))
        object.__setattr__(self, "tool_com_m", _vector(self.tool_com_m, 3, "tool_com_m"))
        object.__setattr__(
            self,
            "gravity_filter_alpha",
            _alpha(self.gravity_filter_alpha, "gravity_filter_alpha"),
        )
        object.__setattr__(self, "low_pass_alpha", _alpha(self.low_pass_alpha, "low_pass_alpha"))
        object.__setattr__(
            self,
            "moving_average_window",
            _integer(self.moving_average_window, "moving_average_window", 1),
        )
        object.__setattr__(
            self,
            "calibration_samples",
            _integer(self.calibration_samples, "calibration_samples", 1),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeConfig:
    """Non-device loop rates and bounded inference settings."""

    kelo_control_rate_hz: int = 10
    collect_rate_hz: int = 10
    inference_fps: int = 10
    inference_max_steps: int = 1000
    inference_episodes: int = 1

    def __post_init__(self) -> None:
        for name in (
            "kelo_control_rate_hz",
            "collect_rate_hz",
            "inference_fps",
            "inference_max_steps",
            "inference_episodes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))


@dataclass(frozen=True, slots=True, kw_only=True)
class AiroDoffyConfig:
    """Complete immutable configuration assembled at the composition boundary."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vr: VRConfig = field(default_factory=VRConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    video: VideoStreamingConfig = field(default_factory=VideoStreamingConfig)
    state_transport: StateTransportConfig = field(default_factory=StateTransportConfig)
    command_transport: CommandTransportConfig = field(default_factory=CommandTransportConfig)
    wrench: WrenchConfig = field(default_factory=WrenchConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
