"""RealMan teleoperation composition root."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import doffy_teleop.utils as utils
from doffy_teleop.config import Config
from doffy_teleop.visualization.config import VisualizerConfig
from doffy_teleop.recording.service import RealManEpisodeRecorder
from ..control.contracts import QuestTcpStateSender
from .publisher import _create_camera_manager, visualizer_publish_loop
from .realman import RealManTeleop

def main(
    *,
    config_factory: Callable[[], Config] = Config,
    visualizer_config_factory: Callable[[], VisualizerConfig] = VisualizerConfig,
    camera_manager_factory: Callable[[Config], Any] = _create_camera_manager,
    teleop_factory: Callable[..., RealManTeleop] = RealManTeleop,
    recorder_factory: Callable[..., RealManEpisodeRecorder] = RealManEpisodeRecorder,
    tcp_state_sender_factory: Callable[..., QuestTcpStateSender] = QuestTcpStateSender,
    visualizer_publish_loop_fn: Callable[..., None] = visualizer_publish_loop,
) -> None:
    """Run the RealMan lifecycle with explicit integration dependencies.

    The defaults keep the package entry point self-contained. Compatibility
    facades pass their historical module globals here so existing test and
    integration replacement points remain effective.
    """
    cfg = config_factory()
    viz_cfg = visualizer_config_factory()
    stop_event = threading.Event()
    camera_manager = None
    teleop: RealManTeleop | None = None
    recorder: RealManEpisodeRecorder | None = None
    tcp_state_sender: QuestTcpStateSender | None = None
    wrm_receiver = None
    beaver_reader = None
    visualizer_handle = None
    worker_threads: list[threading.Thread] = []
    background_errors: list[str] = []
    interrupted = False
    fatal_error = ""
    shutdown_failed = False

    utils.logger.info(
        f"Starting RealMan-only teleoperation: {cfg.TELEOP_COMMAND_MODE} control, "
        f"{cfg.REALMAN_CTRL_RATE} Hz CAN-FD, {cfg.VIDEO_TRANSPORT} video"
    )

    try:
        camera_manager = camera_manager_factory(cfg)
        if cfg.WRM_enable:
            from doffy_teleop.control.wrm_akm import WrmUdpReceiver

            # Port 8005 was the legacy socket_2 resolution/fine-control
            # channel. Transfer ownership instead of binding two UDP sockets
            # and non-deterministically splitting Unity datagrams.
            legacy_control_socket = camera_manager.socket_list.pop(
                "socket_2",
                None,
            )
            if legacy_control_socket is not None:
                legacy_control_socket.close()
            wrm_receiver = WrmUdpReceiver(cfg.PC_IP, cfg.CONTROL_PORT)
            wrm_thread = threading.Thread(
                target=wrm_receiver.run,
                args=(stop_event,),
                name="wrm-unity-udp-receiver",
                daemon=True,
            )
            wrm_thread.start()
            worker_threads.append(wrm_thread)
            utils.logger.info(
                "WRM Unity tracking enabled on UDP %s:%d.",
                cfg.PC_IP,
                cfg.CONTROL_PORT,
            )

            # Accept either a complete controller+elbow packet on 8005 or the
            # legacy controller pose on 8001 plus elbow-only packets on 8005.
            initial_data = None
            connection_deadline = time.monotonic() + 60.0
            while initial_data is None and time.monotonic() < connection_deadline:
                camera_manager.send_and_receive_data()
                with camera_manager._lock:
                    legacy_controller = camera_manager.data
                wrm_sample, _ = wrm_receiver.snapshot()
                if wrm_sample is not None and wrm_sample.has_controller_pose:
                    initial_data = wrm_sample.as_controller_data(legacy_controller)
                elif legacy_controller is not None:
                    initial_data = legacy_controller
                if initial_data is None:
                    stop_event.wait(0.01)
            if initial_data is None:
                raise TimeoutError(
                    "No VR controller pose received on UDP 8001 or WRM UDP 8005."
                )
        else:
            initial_data = camera_manager.test_connection()
        teleop = teleop_factory(initial_data, cfg=cfg)
        if wrm_receiver is not None:
            initial_wrm_sample, _ = wrm_receiver.snapshot()
            teleop.update_wrm_tracking(initial_wrm_sample)
        if cfg.beaver_enable:
            from doffy_teleop.sensors.beaver import BeaverReader

            beaver_reader = BeaverReader.from_config(cfg)
            beaver_thread = beaver_reader.start(stop_event)
            worker_threads.append(beaver_thread)
            utils.logger.info("Beaver USB acquisition enabled for 9 sensors.")
        # Dataset discovery/import can be expensive. Finish it before the
        # high-follow sender starts so it cannot steal time from CAN-FD's
        # strict 10 ms packet deadline.
        recorder = recorder_factory(
            cfg,
            teleop,
            camera_manager,
            background_errors=background_errors,
            beaver_reader=beaver_reader,
        )
        camera_manager.start_comms_threads()
        worker_threads.extend(teleop.start(stop_event))
        if cfg.FORCE_ENABLE:
            tcp_state_sender = tcp_state_sender_factory(
                teleop,
                quest_ip=cfg.VR_IP,
                port=cfg.FORCE_PORT,
                send_rate_hz=cfg.FORCE_SEND_RATE,
            )
            tcp_state_thread = threading.Thread(
                target=tcp_state_sender.run,
                args=(stop_event, background_errors),
                name="quest-tcp-state-sender",
                daemon=True,
            )
            tcp_state_thread.start()
            worker_threads.append(tcp_state_thread)
        worker_threads.extend(recorder.start(stop_event))

        if viz_cfg.ENABLED:
            from doffy_teleop.visualization.dashboard import start_visualizer

            visualizer_handle = start_visualizer(
                hz=viz_cfg.HZ,
                window_s=viz_cfg.WINDOW_S,
                title="RealMan Teleop Visualizer",
                force_panel_range=viz_cfg.FORCE_PANEL_RANGE,
                camera_num=camera_manager.camera_num,
                show_rollback_button=True,
                show_record_button=False,
                beaver_enabled=cfg.beaver_enable,
                beaver_layout=cfg.BEAVER_SENSOR_LAYOUT,
                beaver_max_mm=cfg.BEAVER_VISUALIZER_MAX_MM,
            )
            visualizer_thread = threading.Thread(
                target=visualizer_publish_loop_fn,
                args=(
                    visualizer_handle,
                    teleop,
                    camera_manager,
                    viz_cfg,
                    stop_event,
                    background_errors,
                    recorder,
                    beaver_reader,
                ),
                name="realman-visualizer-publisher",
                daemon=True,
            )
            visualizer_thread.start()
            worker_threads.append(visualizer_thread)

        # Starting worker threads and, especially, spawning the visualizer can
        # briefly preempt the CAN-FD sender. Keep the fail-fast watchdog in its
        # startup probation state until every background service is running,
        # then require a fresh clean window under the final system load.
        timing = teleop.canfd.wait_until_healthy(stop_event)
        verified_hz = timing.achieved_hz
        if verified_hz is None:
            raise RuntimeError(
                "CAN-FD timing verification returned no measured rate."
            )
        utils.logger.info(
            f"RealMan CAN-FD timing verified at {verified_hz:.1f} Hz; "
            "VR motion input enabled."
        )

        previous_input_timestamp_ns = 0
        previous_control_time = time.monotonic()
        while not stop_event.is_set():
            recorder.handle_visualizer_commands(visualizer_handle)

            heartbeat_error = teleop.canfd.heartbeat_error()
            if heartbeat_error:
                fatal_error = heartbeat_error
                teleop.canfd.report_external_failure(
                    heartbeat_error,
                    stop_event,
                )
                break

            robot_snapshot = teleop.state_snapshot()
            state_age_s = (
                time.monotonic_ns() - robot_snapshot.state_timestamp_ns
            ) / 1e9
            if state_age_s > teleop.sensor_stale_after_s:
                teleop.mark_input_stale(
                    f"robot state is stale ({state_age_s * 1000.0:.0f} ms)"
                )

            with camera_manager._lock:
                controller_data = camera_manager.data
                hand_data = (
                    dict(camera_manager.hand_data)
                    if camera_manager.hand_data
                    else None
                )
                input_timestamp_ns = camera_manager.vr_input_timestamp_ns

            if wrm_receiver is not None:
                wrm_sample, _ = wrm_receiver.snapshot()
                # Called even without a new packet so timeout/low confidence
                # freezes the elbow objective while TCP follows existing logic.
                teleop.update_wrm_tracking(wrm_sample)
                if (
                    wrm_sample is not None
                    and wrm_sample.has_controller_pose
                    and wrm_sample.received_ns >= input_timestamp_ns
                ):
                    controller_data = wrm_sample.as_controller_data(controller_data)
                    input_timestamp_ns = wrm_sample.received_ns

            new_input = input_timestamp_ns > previous_input_timestamp_ns
            if cfg.TRACKING_MODE == "hand" and hand_data is not None and new_input:
                now = time.monotonic()
                dt = min(max(now - previous_control_time, 1e-4), 0.05)
                previous_control_time = now
                teleop.process_hand(hand_data, dt)
                previous_input_timestamp_ns = input_timestamp_ns
            elif cfg.TRACKING_MODE == "controller" and controller_data is not None and new_input:
                now = time.monotonic()
                dt = min(max(now - previous_control_time, 1e-4), 0.05)
                previous_control_time = now
                teleop.process_controller(controller_data, dt)
                previous_input_timestamp_ns = input_timestamp_ns
            elif new_input:
                expected = "hand" if cfg.TRACKING_MODE == "hand" else "controller"
                teleop.mark_input_stale(
                    f"Received non-{expected} VR input in {expected} tracking mode"
                )
                previous_control_time = time.monotonic()
                previous_input_timestamp_ns = input_timestamp_ns

            if input_timestamp_ns:
                input_age_s = (time.monotonic_ns() - input_timestamp_ns) / 1e9
                if input_age_s > cfg.REALMAN_VR_TIMEOUT:
                    teleop.mark_input_stale(
                        f"VR input timed out ({input_age_s * 1000.0:.0f} ms)"
                    )

            canfd_error = teleop.canfd.snapshot().error
            if canfd_error:
                fatal_error = canfd_error
                stop_event.set()
                break

            stop_event.wait(0.005)

        fatal_error = (
            teleop.canfd.snapshot().error
            or (background_errors[0] if background_errors else "")
            or fatal_error
        )

    except KeyboardInterrupt:
        interrupted = True
        utils.logger.info("Stopping RealMan teleoperation...")
    finally:
        stop_event.set()

        if camera_manager is not None:
            try:
                camera_manager.close()
            except Exception as exc:
                utils.logger.warning(f"Error closing camera manager: {exc}")

        if wrm_receiver is not None:
            try:
                wrm_receiver.close()
            except Exception as exc:
                utils.logger.warning(f"Error closing WRM UDP receiver: {exc}")

        if beaver_reader is not None:
            try:
                beaver_reader.close()
            except Exception as exc:
                utils.logger.warning(f"Error closing Beaver reader: {exc}")

        all_threads = list(worker_threads)
        if teleop is not None:
            for thread in teleop.sdk_worker_threads():
                if thread not in all_threads:
                    all_threads.append(thread)
        for thread in all_threads:
            thread.join(timeout=2.0)

        if visualizer_handle is not None:
            try:
                visualizer_handle.close()
            except Exception as exc:
                utils.logger.warning(
                    f"Error closing RealMan visualizer: {exc}"
                )

        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                fatal_error = fatal_error or f"Error closing dataset: {exc}"
                utils.logger.error(f"Error closing RealMan dataset: {exc}")

        if teleop is not None:
            live_workers = teleop.live_sdk_workers()
            if live_workers:
                shutdown_error = (
                    "RealMan worker(s) did not stop; the SDK handle was left "
                    f"open to avoid invalidating an in-flight call: "
                    f"{', '.join(live_workers)}."
                )
                fatal_error = fatal_error or shutdown_error
                shutdown_failed = True
                utils.logger.critical(shutdown_error)
                try:
                    teleop.quarantine_without_sdk_cleanup()
                except Exception as exc:
                    utils.logger.critical(
                        "Could not drain RealMan state callbacks while "
                        f"quarantining the live SDK handle: {exc}"
                    )
            else:
                try:
                    teleop.close()
                except Exception as exc:
                    shutdown_error = (
                        f"Error closing RealMan connection: {exc}"
                    )
                    fatal_error = fatal_error or shutdown_error
                    shutdown_failed = True
                    utils.logger.error(shutdown_error)
                    try:
                        teleop.quarantine_without_sdk_cleanup()
                    except Exception as callback_exc:
                        utils.logger.critical(
                            "Could not drain RealMan state callbacks after "
                            f"SDK cleanup failed: {callback_exc}"
                        )

        utils.logger.info("RealMan teleoperation shutdown complete.")

    if fatal_error and (not interrupted or shutdown_failed):
        raise RuntimeError(f"RealMan teleoperation stopped: {fatal_error}")


if __name__ == "__main__":
    main()
