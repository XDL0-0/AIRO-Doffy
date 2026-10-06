"""Tactile, recording, and visualizer helpers for classic teleop."""

from __future__ import annotations

import threading
import time

import numpy as np

import doffy_teleop.utils as utils
from doffy_teleop.recording.service import (
    DataRecordingService,
    RecordingFrame,
)
from doffy_teleop.robots.teleop import RobotTeleop
from doffy_teleop.media.udp_manager import UDPManager
from doffy_teleop.media.webrtc_manager import WebRTCUDPManager

from . import classic_config

CameraManager = UDPManager | WebRTCUDPManager

class TactileDataHolder:
    """Small tactile-only target for the BLE callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tactile_data: np.ndarray | None = None
        self.tactile_byte: bytes | None = None
        self.tactile_timestamp_ns: int = 0
        self.data = (None, {"Joystick_Press": False})
        self.tactile_recalibrate_requested = False
        self.tactile_reader = None


def controller_reset_requested(data, *, cfg=None) -> bool:
    if data is None:
        return False
    try:
        right = data[1]
        config = classic_config.cfg if cfg is None else cfg
        return (
            bool(right["Joystick_Press"])
            and right["IndexTrigger"] >= config.CONTROLLER_RESET_TRIGGER_THRESHOLD
        )
    except (IndexError, KeyError, TypeError):
        return False


def request_tactile_recalibration(tactile_holder: TactileDataHolder | None) -> None:
    if tactile_holder is None:
        return
    with tactile_holder._lock:
        tactile_holder.tactile_recalibrate_requested = True


def create_tactile_reader():
    if classic_config.cfg.TACTILE_READER == "ble4":
        from sensor_comm_dds.communication.config.ble_config import DeviceMAC, SensorUuid
        from sensor_comm_dds.communication.readers.magtouch_ble_reader import (
            MagTouchBleReaderConfig,
        )
        from doffy_teleop.sensors.tactile_4point import FourPointTactileBleReader

        return FourPointTactileBleReader(
            config=MagTouchBleReaderConfig(
                ENABLE_WS=False,
                NUM_SENSORS=1,
                NUM_TAXELS=4,
                MODEL_NAMES=np.array([None]),
                WINDOW_SIZE=classic_config.cfg.TACTILE_BLE_WINDOW_SIZE,
                uuid=SensorUuid.DATA_CHAR_MAGTOUCH,
                device_mac=DeviceMAC[classic_config.cfg.TACTILE_BLE_DEVICE_MAC],
                hci=classic_config.cfg.TACTILE_BLE_HCI,
            ),
            filter_alpha=classic_config.cfg.TACTILE_FILTER_ALPHA,
            use_kalman=classic_config.cfg.TACTILE_USE_KALMAN,
            kalman_q=classic_config.cfg.TACTILE_KALMAN_Q,
            kalman_r=classic_config.cfg.TACTILE_KALMAN_R,
            max_delta=classic_config.cfg.TACTILE_MAX_DELTA,
            baseline_drift_alpha=classic_config.cfg.TACTILE_BASELINE_DRIFT_ALPHA,
            baseline_drift_threshold=classic_config.cfg.TACTILE_BASELINE_DRIFT_THRESHOLD,
            reset_trigger_threshold=classic_config.cfg.CONTROLLER_RESET_TRIGGER_THRESHOLD,
        )

    from doffy_teleop.sensors.tactile import MagtouchIliasSerialReader, MagtouchIliasSerialReaderConfig

    return MagtouchIliasSerialReader(
        config=MagtouchIliasSerialReaderConfig(
            ENABLE_WS=False,
            COM=classic_config.cfg.TACTILE_SERIAL_COM,
            START_BYTE=0xAA,
            END_BYTE=0xCC,
        )
    )


def run_tactile_reader(tactile_holder: TactileDataHolder) -> None:
    tactile_manager = create_tactile_reader()
    with tactile_holder._lock:
        tactile_holder.tactile_reader = tactile_manager
    if classic_config.cfg.TACTILE_READER == "ble4":
        tactile_manager.run(
            tactile_holder,
            start_visualizer=False,
            visualizer_topic="MagTouchRaw0",
        )
    else:
        tactile_manager.run(tactile_holder)


# ── Background loops ─────────────────────────────────────────────────────

def create_recording_frame(
    teleop: RobotTeleop,
    cu_manager: CameraManager,
    beaver_reader=None,
) -> RecordingFrame | None:
    """Prepare one atomic multimodal frame for DataRecordingService."""
    controller_motion = cu_manager.is_movement_exist()
    with cu_manager._lock:
        has_hand_motion = bool(cu_manager.hand_data) and teleop.tracking_mode == "hand"
        if not (
            controller_motion
            or has_hand_motion
            or teleop.reset_sign
        ):
            return None
        images = {
            name: np.asarray(image).copy()
            for name, image in cu_manager.camera_images.items()
        }
        image_timestamps = dict(
            getattr(cu_manager, "camera_image_timestamps_ns", {})
        )
        depth_images = (
            {
                name: np.asarray(depth).copy()
                for name, depth in cu_manager.depth_images.items()
            }
            if cu_manager.depth_mode
            else None
        )
        tactile = (
            None
            if cu_manager.tactile_data is None
            else cu_manager.tactile_data.copy()
        )
        tactile_timestamp = int(getattr(cu_manager, "tactile_timestamp_ns", 0))
        vr_input_timestamp = int(getattr(cu_manager, "vr_input_timestamp_ns", 0))

    collect_timestamp_ns = time.monotonic_ns()
    state, action, wrench, extra = teleop.get_state_snapshot()
    extra = {} if extra is None else dict(extra)
    beaver = beaver_reader.snapshot() if beaver_reader is not None else None
    extra.update(
        {
            "collect_timestamp_ns": np.array(collect_timestamp_ns, dtype=np.int64),
            "camera_timestamps_ns": image_timestamps,
            "tactile_timestamp_ns": np.array(tactile_timestamp, dtype=np.int64),
            "vr_input_timestamp_ns": np.array(vr_input_timestamp, dtype=np.int64),
            "beaver_timestamp_ns": np.array(
                0 if beaver is None else beaver.timestamp_ns,
                dtype=np.int64,
            ),
        }
    )
    return RecordingFrame(
        state=state,
        action=action,
        camera_images=images,
        tactile_data=tactile,
        wrench_data=wrench if teleop.wrench_mode else None,
        depth_images=depth_images,
        extra_data=extra,
        beaver_data=beaver,
    )


def tactile_bridge_loop(
    tactile_holder: TactileDataHolder,
    cu_manager: CameraManager,
    stop_event: threading.Event,
) -> None:
    """Copy tactile samples from the BLE holder into the camera manager."""
    dt = 1.0 / classic_config.TACTILE_BRIDGE_HZ
    last_timestamp_ns = -1
    while not stop_event.is_set():
        with tactile_holder._lock:
            timestamp_ns = tactile_holder.tactile_timestamp_ns
            if timestamp_ns != last_timestamp_ns:
                data = (
                    None
                    if tactile_holder.tactile_data is None
                    else tactile_holder.tactile_data.copy()
                )
                tactile_byte = tactile_holder.tactile_byte
            else:
                data = None
                tactile_byte = None

        if timestamp_ns != last_timestamp_ns:
            with cu_manager._lock:
                cu_manager.tactile_data = data
                cu_manager.tactile_byte = tactile_byte
                cu_manager.tactile_timestamp_ns = timestamp_ns
            last_timestamp_ns = timestamp_ns

        if stop_event.wait(timeout=dt):
            break


def _visualizer_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    step_y = max(1, image.shape[0] // 240)
    step_x = max(1, image.shape[1] // 320)
    return image[::step_y, ::step_x, :3].copy()


def _visualizer_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    visualizer_images: dict[str, np.ndarray] = {}
    for name, image in sorted(images.items()):
        preview = _visualizer_image(image)
        if preview is not None:
            visualizer_images[name] = preview
    return visualizer_images


def _visualizer_tcp_translation(extra: dict[str, object]) -> np.ndarray | None:
    tcp_pose = extra.get("tcp_pose")
    if tcp_pose is None:
        return None
    tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(-1)
    if tcp_pose.size < 3:
        return None
    return tcp_pose[:3].copy()


def visualizer_publish_loop(
    visualizer_handle,
    teleop: RobotTeleop,
    cu_manager: CameraManager,
    recording_service: DataRecordingService,
    tactile_holder: TactileDataHolder | None,
    stop_event: threading.Event,
    beaver_reader=None,
) -> None:
    dt = 1.0 / classic_config.viz_cfg.HZ
    while not stop_event.is_set():
        error = ""
        try:
            if teleop.wrench_mode:
                teleop.refresh_wrench_snapshot()
            state, _action, wrench, extra = teleop.get_state_snapshot()
        except Exception as exc:
            state = np.zeros(teleop.schema.state_dim, dtype=float)
            wrench = np.zeros(6, dtype=float)
            extra = {}
            error = str(exc)

        with cu_manager._lock:
            images = dict(cu_manager.camera_data)
            images.update(cu_manager.camera_images)

        if tactile_holder is not None:
            with tactile_holder._lock:
                tactile = (
                    None
                    if tactile_holder.tactile_data is None
                    else tactile_holder.tactile_data.copy()
                )
                tactile_timestamp_ns = tactile_holder.tactile_timestamp_ns
        else:
            with cu_manager._lock:
                tactile = (
                    None
                    if cu_manager.tactile_data is None
                    else cu_manager.tactile_data.copy()
                )
                tactile_timestamp_ns = getattr(cu_manager, "tactile_timestamp_ns", 0)

        visualizer_handle.publish(
            {
                "timestamp": time.monotonic(),
                "wrench": np.asarray(wrench, dtype=float).copy(),
                "joints": np.asarray(state[: teleop.dof], dtype=float).copy()
                if state.size >= teleop.dof
                else None,
                "tcp_translation": _visualizer_tcp_translation(extra),
                "images": _visualizer_images(images),
                "camera_count": cu_manager.camera_num,
                "tactile": tactile,
                "tactile_timestamp_ns": tactile_timestamp_ns,
                "dataset": recording_service.recording_status(),
                **(
                    {"beaver": beaver_reader.visualizer_payload()}
                    if beaver_reader is not None
                    else {}
                ),
                "connected": not error,
                "error": error,
            }
        )

        if stop_event.wait(timeout=dt):
            break
