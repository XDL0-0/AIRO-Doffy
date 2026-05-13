"""Data-collection entry point: VR teleop + camera streaming + recording."""

from __future__ import annotations

import threading
import time

import utils
from config import Config
from ur_teleop import URTeleop
from dataset_new import DatasetRecorder
from camera_udp import CameraUDPManager
from WebRTC_udp import WebRTCUDPManager
from tactile import MagtouchIliasSerialReader, MagtouchIliasSerialReaderConfig

CameraManager = CameraUDPManager | WebRTCUDPManager

cfg = Config()

TELEOP_HZ = 100
MIN_DT = 1.0 / TELEOP_HZ


# ── Background loops ─────────────────────────────────────────────────────

def collect_loop(
    teleop: URTeleop,
    cu_manager: CameraManager,
    dataset: DatasetRecorder,
    collect_rate: int,
    stop_event: threading.Event,
    pause_event: threading.Event | None = None,
) -> None:
    """Sample state/action at *collect_rate* Hz and buffer into *dataset*."""
    dt = 1.0 / collect_rate
    while not stop_event.is_set():
        t0 = time.time()

        # Skip collection while export is in progress
        if pause_event is not None and pause_event.is_set():
            time.sleep(dt)
            continue

        collecting = cu_manager.data_collecting_state
        has_motion = cu_manager.is_movement_exist() or teleop.reset_sign

        if collecting and has_motion:
            # ── Atomic snapshot: one lock covers ALL cu_manager modalities ──
            with cu_manager._lock:
                images = dict(cu_manager.camera_images)
                depth_imgs = dict(cu_manager.depth_images) if cu_manager.depth_mode else None
                # .copy() to break shared reference; None is fine (dataset handles it)
                tactile = cu_manager.tactile_data.copy() if cu_manager.tactile_data is not None else None

            # Robot state uses its own _state_lock inside get_state_snapshot()
            state, action, force, extra = teleop.get_state_snapshot()

            force_val = force if (cfg.FORCE_COLLECT and cfg.TORQUE_MODE) else None
            dataset.data_collection(
                state, action, images, tactile, force_val, depth_imgs, extra
            )

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


def export_loop(
    dataset: DatasetRecorder,
    cu_manager: CameraManager,
    stop_event: threading.Event,
    pause_event: threading.Event | None = None,
) -> None:
    """Wait for export signals and write episodes to disk."""
    try:
        while not stop_event.is_set():
            if cu_manager.data_export_state:
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

    except Exception as e:
        utils.logger.error(f"Export loop error: {e}")
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
    teleop = URTeleop(cu_manager.test_connection())

    if cfg.TACTILE_TRANSFER:
        tactile_manager = MagtouchIliasSerialReader(
            config=MagtouchIliasSerialReaderConfig(
                ENABLE_WS=False,
                COM="/dev/serial/by-id/usb-Arduino_IO_Coupling_C6E76762B4D1E02A-if00",
                START_BYTE=0xAA,
                END_BYTE=0xCC,
            )
        )
        threading.Thread(
            target=tactile_manager.run, args=(cu_manager,), daemon=True
        ).start()

    time.sleep(5)
    dataset = DatasetRecorder(cu_manager.camera_num)
    cu_manager.start_comms_threads()

    # Synchronization: pause collection during episode export
    pause_event = threading.Event()

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
        prev_time = time.time()
        while True:
            now = time.time()
            dt = min(now - prev_time, 0.05)
            prev_time = now

            with cu_manager._lock:
                data = cu_manager.data
                fine = cu_manager.fine_mode
                hand = dict(cu_manager.hand_data) if cu_manager.hand_data else None
            if data is not None or hand is not None:
                teleop.step(data, fine, dt, hand_data=hand)

            loop_time = time.time() - prev_time
            if loop_time < MIN_DT:
                time.sleep(MIN_DT - loop_time)

    except KeyboardInterrupt:
        utils.logger.info("Stopping...")

    finally:
        utils.logger.info("Cleaning up...")

        try:
            cu_manager.close()
        except Exception as e:
            utils.logger.error(f"Error closing cu_manager: {e}")

        stop_event.set()
        t_export.join(timeout=5.0)

        try:
            if cfg.TORQUE_MODE:
                teleop.ur.disable_torque_control()
        except AttributeError:
            pass
        except Exception as e:
            utils.logger.error(f"Error disabling robot: {e}")

        utils.logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
