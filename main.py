"""Data-collection entry point: VR teleop + camera streaming + recording."""

from __future__ import annotations

import threading
import time

import numpy as np

import utils
from config import Config
from visualizer_config import VisualizerConfig
from robot_teleop import RobotTeleop
from dataset import DatasetRecorder
from camera_udp import CameraUDPManager
from WebRTC_udp import WebRTCUDPManager

CameraManager = CameraUDPManager | WebRTCUDPManager

cfg = Config()
viz_cfg = VisualizerConfig()

TELEOP_HZ = cfg.UR_CTRL_RATE
MIN_DT = 1.0 / TELEOP_HZ
TACTILE_BRIDGE_HZ = 100.0


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


def controller_reset_requested(data) -> bool:
    if data is None:
        return False
    try:
        right = data[1]
        return (
            bool(right["Joystick_Press"])
            and right["IndexTrigger"] >= cfg.CONTROLLER_RESET_TRIGGER_THRESHOLD
        )
    except (IndexError, KeyError, TypeError):
        return False


def request_tactile_recalibration(tactile_holder: TactileDataHolder | None) -> None:
    if tactile_holder is None:
        return
    with tactile_holder._lock:
        tactile_holder.tactile_recalibrate_requested = True


def create_tactile_reader():
    if cfg.TACTILE_READER == "ble4":
        from sensor_comm_dds.communication.config.ble_config import DeviceMAC, SensorUuid
        from sensor_comm_dds.communication.readers.magtouch_ble_reader import (
            MagTouchBleReaderConfig,
        )
        from tactile_4point import FourPointTactileBleReader

        return FourPointTactileBleReader(
            config=MagTouchBleReaderConfig(
                ENABLE_WS=False,
                NUM_SENSORS=1,
                NUM_TAXELS=4,
                MODEL_NAMES=np.array([None]),
                WINDOW_SIZE=cfg.TACTILE_BLE_WINDOW_SIZE,
                uuid=SensorUuid.DATA_CHAR_MAGTOUCH,
                device_mac=DeviceMAC[cfg.TACTILE_BLE_DEVICE_MAC],
                hci=cfg.TACTILE_BLE_HCI,
            ),
            filter_alpha=cfg.TACTILE_FILTER_ALPHA,
            use_kalman=cfg.TACTILE_USE_KALMAN,
            kalman_q=cfg.TACTILE_KALMAN_Q,
            kalman_r=cfg.TACTILE_KALMAN_R,
            max_delta=cfg.TACTILE_MAX_DELTA,
            baseline_drift_alpha=cfg.TACTILE_BASELINE_DRIFT_ALPHA,
            baseline_drift_threshold=cfg.TACTILE_BASELINE_DRIFT_THRESHOLD,
            reset_trigger_threshold=cfg.CONTROLLER_RESET_TRIGGER_THRESHOLD,
        )

    from tactile import MagtouchIliasSerialReader, MagtouchIliasSerialReaderConfig

    return MagtouchIliasSerialReader(
        config=MagtouchIliasSerialReaderConfig(
            ENABLE_WS=False,
            COM=cfg.TACTILE_SERIAL_COM,
            START_BYTE=0xAA,
            END_BYTE=0xCC,
        )
    )


def run_tactile_reader(tactile_holder: TactileDataHolder) -> None:
    tactile_manager = create_tactile_reader()
    with tactile_holder._lock:
        tactile_holder.tactile_reader = tactile_manager
    if cfg.TACTILE_READER == "ble4":
        tactile_manager.run(
            tactile_holder,
            start_visualizer=False,
            visualizer_topic="MagTouchRaw0",
        )
    else:
        tactile_manager.run(tactile_holder)


# ── Background loops ─────────────────────────────────────────────────────

def collect_loop(
    teleop: RobotTeleop,
    cu_manager: CameraManager,
    dataset: DatasetRecorder,
    collect_rate: int,
    stop_event: threading.Event,
    pause_event: threading.Event | None = None,
) -> None:
    """Sample state/action at *collect_rate* Hz and buffer into *dataset*."""
    dt = 1.0 / collect_rate
    next_tick = time.monotonic()
    while not stop_event.is_set():
        t0 = time.monotonic()
        collect_ts_ns = time.monotonic_ns()

        # Skip collection while export is in progress
        if pause_event is not None and pause_event.is_set():
            next_tick += dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
            continue

        collecting = cu_manager.data_collecting_state
        has_hand_motion = bool(cu_manager.hand_data) and teleop.tracking_mode == "hand"
        has_motion = cu_manager.is_movement_exist() or has_hand_motion or teleop.reset_sign

        if collecting and has_motion:
            # ── Atomic snapshot: one lock covers ALL cu_manager modalities ──
            with cu_manager._lock:
                images = dict(cu_manager.camera_images)
                image_timestamps = dict(
                    getattr(cu_manager, "camera_image_timestamps_ns", {})
                )
                depth_imgs = dict(cu_manager.depth_images) if cu_manager.depth_mode else None
                # .copy() to break shared reference; None is fine (dataset handles it)
                tactile = cu_manager.tactile_data.copy() if cu_manager.tactile_data is not None else None
                tactile_timestamp = getattr(cu_manager, "tactile_timestamp_ns", 0)
                vr_input_timestamp = getattr(cu_manager, "vr_input_timestamp_ns", 0)

            # Robot state uses its own _state_lock inside get_state_snapshot()
            state, action, wrench, extra = teleop.get_state_snapshot()
            if extra is None:
                extra = {}
            extra["collect_timestamp_ns"] = np.array(collect_ts_ns, dtype=np.int64)
            extra["camera_timestamps_ns"] = image_timestamps
            extra["tactile_timestamp_ns"] = np.array(
                tactile_timestamp, dtype=np.int64
            )
            extra["vr_input_timestamp_ns"] = np.array(
                vr_input_timestamp, dtype=np.int64
            )

            wrench_val = wrench if teleop.wrench_mode else None
            dataset.data_collection(
                state, action, images, tactile, wrench_val, depth_imgs, extra
            )

        next_tick += dt
        remaining = next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        elif time.monotonic() - t0 > dt:
            next_tick = time.monotonic()


def tactile_bridge_loop(
    tactile_holder: TactileDataHolder,
    cu_manager: CameraManager,
    stop_event: threading.Event,
) -> None:
    """Copy tactile samples from the BLE holder into the camera manager."""
    dt = 1.0 / TACTILE_BRIDGE_HZ
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
    dataset: DatasetRecorder,
    tactile_holder: TactileDataHolder | None,
    stop_event: threading.Event,
) -> None:
    dt = 1.0 / viz_cfg.HZ
    while not stop_event.is_set():
        error = ""
        try:
            if teleop.wrench_mode:
                teleop.refresh_wrench_snapshot()
            state, _action, wrench, extra = teleop.get_state_snapshot()
        except Exception as exc:
            state = np.zeros(teleop.dof + 1, dtype=float)
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
                "dataset": dataset.recording_status(
                    collecting=bool(cu_manager.data_collecting_state)
                ),
                "connected": not error,
                "error": error,
            }
        )

        if stop_event.wait(timeout=dt):
            break


def export_loop(
    dataset: DatasetRecorder,
    cu_manager: CameraManager,
    stop_event: threading.Event,
    pause_event: threading.Event | None = None,
) -> None:
    """Wait for export signals and write episodes to disk."""
    try:
        while not stop_event.is_set():
            if getattr(cu_manager, "data_rollback_state", False):
                if pause_event is not None:
                    pause_event.set()
                try:
                    if dataset.rollback_last_episode():
                        utils.logger.info(
                            f"Rollback complete. Next episode: {dataset.recorded_episodes}"
                        )
                    else:
                        utils.logger.warning("Rollback requested, but no episode was removed.")
                except Exception:
                    utils.logger.exception("Rollback failed")
                finally:
                    if pause_event is not None:
                        pause_event.clear()
                    cu_manager.data_rollback_state = False

            elif cu_manager.data_export_state:
                if not dataset.collect_step:
                    utils.logger.error("No data to export")
                    cu_manager.data_export_state = False
                else:
                    # Pause collection to avoid frames leaking across episodes
                    if pause_event is not None:
                        pause_event.set()
                    dataset.data_export(cu_manager)
                    dataset._reset_data_dict()
                    if pause_event is not None:
                        pause_event.clear()
                    utils.logger.info(
                        f"Episode {dataset.recorded_episodes - 1} exported successfully"
                    )
                    cu_manager.data_export_state = False

            if stop_event.wait(timeout=0.5):
                break

    except Exception:
        utils.logger.exception("Export loop error")
    finally:
        utils.logger.info("Cleaning up export...")
        try:
            dataset.close()
        except Exception as e:
            utils.logger.error(f"Error finalizing dataset: {e}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    stop_event = threading.Event()
    utils.logger.info(f"TASK: {cfg.TASK_NAME}")
    utils.logger.info(f"VIDEO_TRANSPORT: {cfg.VIDEO_TRANSPORT}")

    if cfg.VIDEO_TRANSPORT.lower() == "webrtc":
        cu_manager = WebRTCUDPManager()
    else:
        cu_manager = CameraUDPManager()
    teleop = RobotTeleop(cu_manager.test_connection())
    if viz_cfg.ENABLED and not teleop.wrench_mode:
        utils.logger.warning(
            "VISUALIZER is enabled, but the selected robot backend does not expose TCP force."
        )

    tactile_enabled = cfg.TACTILE_ENABLE and (cfg.TACTILE_TRANSFER or viz_cfg.ENABLED)
    tactile_holder = TactileDataHolder() if tactile_enabled else None
    t_tactile_reader = None
    t_tactile_bridge = None
    if tactile_enabled:
        t_tactile_reader = threading.Thread(
            target=run_tactile_reader,
            args=(tactile_holder,),
            daemon=True,
        )
        t_tactile_reader.start()
        utils.logger.info(
            "Tactile reader enabled for %s.",
            "VR transfer/dataset" if cfg.TACTILE_TRANSFER else "visualizer",
        )

    if cfg.TACTILE_TRANSFER and tactile_holder is not None:
        t_tactile_bridge = threading.Thread(
            target=tactile_bridge_loop,
            args=(tactile_holder, cu_manager, stop_event),
            daemon=True,
        )
        t_tactile_bridge.start()

    time.sleep(5)
    dataset = DatasetRecorder(
        cu_manager.camera_num,
        robot_dof=teleop.dof,
        robot_type=teleop.backend.dataset_robot_type,
        force_collect=cfg.FORCE_COLLECT and teleop.backend.supports_force,
        torque_collect=cfg.TORQUE_COLLECT and teleop.backend.supports_force,
    )
    cu_manager.start_comms_threads()

    # Synchronization: pause collection during episode export
    pause_event = threading.Event()

    visualizer_handle = None
    t_visualizer = None
    if viz_cfg.ENABLED:
        from visualizer import start_visualizer

        visualizer_handle = start_visualizer(
            hz=viz_cfg.HZ,
            window_s=viz_cfg.WINDOW_S,
            title="Teleop Visualizer",
            force_panel_range=viz_cfg.FORCE_PANEL_RANGE,
            camera_num=cu_manager.camera_num,
        )
        t_visualizer = threading.Thread(
            target=visualizer_publish_loop,
            args=(visualizer_handle, teleop, cu_manager, dataset, tactile_holder, stop_event),
            daemon=True,
        )
        t_visualizer.start()
        utils.logger.info(
            f"Teleop visualizer enabled (force MA={cfg.FORCE_MOVING_AVERAGE_WINDOW}, "
            f"force LPF={cfg.FORCE_LOW_PASS_ALPHA:.2f}, "
            f"panel range=+/-{viz_cfg.FORCE_PANEL_RANGE:g} N)."
        )

    t_collect = threading.Thread(
        target=collect_loop,
        args=(teleop, cu_manager, dataset, cfg.COLLECT_RATE, stop_event, pause_event),
        daemon=True,
    )
    t_export = threading.Thread(
        target=export_loop,
        args=(dataset, cu_manager, stop_event, pause_event),
        daemon=True,
    )
    t_collect.start()
    t_export.start()

    try:
        prev_time = time.monotonic()
        next_tick = prev_time
        reset_request_held = False
        while True:
            now = time.monotonic()
            dt = min(now - prev_time, 0.05)
            prev_time = now

            with cu_manager._lock:
                data = cu_manager.data
                fine = cu_manager.fine_mode
                hand = dict(cu_manager.hand_data) if cu_manager.hand_data else None
            if data is not None or hand is not None:
                reset_before = teleop.reset_sign
                controller_reset = controller_reset_requested(data)
                teleop.step(data, fine, dt, hand_data=hand)
                reset_requested = controller_reset or (
                    teleop.reset_sign and not reset_before
                )
                if tactile_enabled and reset_requested and not reset_request_held:
                    request_tactile_recalibration(tactile_holder)
                reset_request_held = reset_requested

            if visualizer_handle is not None:
                for command in visualizer_handle.drain_commands():
                    if command.get("command") == "rollback_last_episode":
                        with cu_manager._lock:
                            cu_manager.data_collecting_state = False
                            cu_manager.data_export_state = False
                            cu_manager.data_rollback_state = True
                        utils.logger.info("Rollback requested from visualizer.")

            next_tick += MIN_DT
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        utils.logger.info("Stopping...")

    finally:
        utils.logger.info("Cleaning up...")

        stop_event.set()
        if tactile_holder is not None:
            with tactile_holder._lock:
                tactile_reader = tactile_holder.tactile_reader
            if tactile_reader is not None and hasattr(tactile_reader, "stop"):
                try:
                    tactile_reader.stop()
                except Exception as e:
                    utils.logger.warning(f"Error stopping tactile reader: {e}")
        try:
            cu_manager.close()
        except Exception as e:
            utils.logger.error(f"Error closing cu_manager: {e}")

        if t_tactile_reader is not None:
            t_tactile_reader.join(timeout=3.0)
        if t_tactile_bridge is not None:
            t_tactile_bridge.join(timeout=1.0)
        if t_visualizer is not None:
            t_visualizer.join(timeout=1.0)
        if visualizer_handle is not None:
            visualizer_handle.close()
        t_collect.join(timeout=3.0)
        t_export.join(timeout=5.0)

        try:
            teleop.close()
        except Exception as e:
            utils.logger.error(f"Error closing teleop: {e}")

        utils.logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
