"""Classic data collection entry point.

The historical ``main.py`` module remains a compatibility facade; this module
contains the actual lifecycle orchestration. Tactile, recording, and
visualizer helpers live in :mod:`classic_helpers`.
"""

from __future__ import annotations

import inspect
import threading
import time

import doffy_teleop.utils as utils
from doffy_teleop.recording.dataset import DatasetRecorder
from doffy_teleop.recording.service import DataRecordingService, ManagerRecordingControl
from doffy_teleop.robots.teleop import RobotTeleop
from doffy_teleop.media.udp_manager import UDPManager
from doffy_teleop.media.webrtc_manager import WebRTCUDPManager

from . import classic_config
from .classic_config import MIN_DT, TACTILE_BRIDGE_HZ, TELEOP_HZ
from .classic_helpers import (
    CameraManager,
    TactileDataHolder,
    controller_reset_requested,
    create_recording_frame,
    create_tactile_reader,
    request_tactile_recalibration,
    run_tactile_reader,
    tactile_bridge_loop,
    visualizer_publish_loop,
)

cfg = classic_config.cfg
viz_cfg = classic_config.viz_cfg


def _sync_config_aliases(config=None, visualizer_config=None) -> None:
    """Keep helper config references aligned with compatibility monkeypatches."""
    classic_config.cfg = cfg if config is None else config
    classic_config.viz_cfg = (
        viz_cfg if visualizer_config is None else visualizer_config
    )


def _create_camera_manager(config):
    """Build the configured legacy media manager."""
    if config.VIDEO_TRANSPORT.lower() == "webrtc":
        try:
            parameters = inspect.signature(WebRTCUDPManager).parameters.values()
        except (TypeError, ValueError):
            return WebRTCUDPManager(config=config)
        if any(
            parameter.name == "config"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            return WebRTCUDPManager(config=config)
        return WebRTCUDPManager()
    return UDPManager(config=config)


def _close_resource(label: str, callback) -> None:
    try:
        callback()
    except Exception as exc:
        utils.logger.error("Error %s: %s", label, exc)


def _create_teleop(initial_data, config, visualizer_config):
    """Pass config to current teleop factories while keeping old patches usable."""
    try:
        parameters = inspect.signature(RobotTeleop).parameters.values()
    except (TypeError, ValueError):
        return RobotTeleop(
            initial_data,
            cfg=config,
            visualizer_config=visualizer_config,
        )
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    names = {parameter.name for parameter in parameters}
    kwargs = {}
    if accepts_kwargs or "cfg" in names:
        kwargs["cfg"] = config
    if accepts_kwargs or "visualizer_config" in names:
        kwargs["visualizer_config"] = visualizer_config
    return RobotTeleop(initial_data, **kwargs)


def main(*, config=None, visualizer_config=None) -> None:
    previous_config = classic_config.cfg
    previous_visualizer_config = classic_config.viz_cfg
    _sync_config_aliases(config, visualizer_config)
    active_cfg = classic_config.cfg
    active_viz_cfg = classic_config.viz_cfg
    control_period = (
        1.0 / float(active_cfg.UR_CTRL_RATE)
        if hasattr(active_cfg, "UR_CTRL_RATE")
        else MIN_DT
    )
    stop_event = threading.Event()
    utils.logger.info(f"TASK: {active_cfg.TASK_NAME}")
    utils.logger.info(f"VIDEO_TRANSPORT: {active_cfg.VIDEO_TRANSPORT}")

    cu_manager = None
    teleop = None
    beaver_reader = None
    tactile_holder = None
    t_tactile_reader = None
    t_tactile_bridge = None
    dataset = None
    recording_service = None
    recording_threads = []
    visualizer_handle = None
    t_visualizer = None
    try:
        cu_manager = _create_camera_manager(active_cfg)
        teleop = _create_teleop(
            cu_manager.test_connection(),
            active_cfg,
            active_viz_cfg,
        )
        if active_cfg.beaver_enable:
            from doffy_teleop.sensors.beaver import BeaverReader

            beaver_reader = BeaverReader.from_config(active_cfg)
            beaver_reader.start(stop_event)
            utils.logger.info("Beaver USB acquisition enabled for 9 sensors.")
        if active_viz_cfg.ENABLED and not teleop.wrench_mode:
            utils.logger.warning(
                "VISUALIZER is enabled, but the selected robot backend does not expose TCP force."
            )

        tactile_enabled = active_cfg.TACTILE_ENABLE and (
            active_cfg.TACTILE_TRANSFER or active_viz_cfg.ENABLED
        )
        tactile_holder = TactileDataHolder() if tactile_enabled else None
        if tactile_enabled:
            t_tactile_reader = threading.Thread(
                target=run_tactile_reader,
                args=(tactile_holder,),
                daemon=True,
            )
            t_tactile_reader.start()
            utils.logger.info(
                "Tactile reader enabled for %s.",
                "VR transfer/dataset"
                if active_cfg.TACTILE_TRANSFER
                else "visualizer",
            )

        if active_cfg.TACTILE_TRANSFER and tactile_holder is not None:
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
            force_collect=active_cfg.FORCE_COLLECT
            and teleop.backend.supports_force,
            torque_collect=active_cfg.TORQUE_COLLECT
            and teleop.backend.supports_force,
            gripper=teleop.gripper_enabled,
            config=active_cfg,
        )
        cu_manager.start_comms_threads()
        recording_service = DataRecordingService(
            dataset,
            lambda: create_recording_frame(teleop, cu_manager, beaver_reader),
            active_cfg.COLLECT_RATE,
            control=ManagerRecordingControl(cu_manager),
            export_context=cu_manager,
            thread_name="teleop-dataset",
        )
        recording_threads = recording_service.start(stop_event)

        if active_viz_cfg.ENABLED:
            from doffy_teleop.visualization.dashboard import start_visualizer

            visualizer_handle = start_visualizer(
                hz=active_viz_cfg.HZ,
                window_s=active_viz_cfg.WINDOW_S,
                title="Teleop Visualizer",
                force_panel_range=active_viz_cfg.FORCE_PANEL_RANGE,
                camera_num=cu_manager.camera_num,
                beaver_enabled=active_cfg.beaver_enable,
                beaver_layout=active_cfg.BEAVER_SENSOR_LAYOUT,
                beaver_max_mm=active_cfg.BEAVER_VISUALIZER_MAX_MM,
            )
            t_visualizer = threading.Thread(
                target=visualizer_publish_loop,
                args=(
                    visualizer_handle,
                    teleop,
                    cu_manager,
                    recording_service,
                    tactile_holder,
                    stop_event,
                    beaver_reader,
                ),
                daemon=True,
            )
            t_visualizer.start()
            utils.logger.info(
                f"Teleop visualizer enabled (force MA={active_cfg.FORCE_MOVING_AVERAGE_WINDOW}, "
                f"force LPF={active_cfg.FORCE_LOW_PASS_ALPHA:.2f}, "
                f"panel range=+/-{active_viz_cfg.FORCE_PANEL_RANGE:g} N)."
            )

        prev_time = time.monotonic()
        next_tick = prev_time
        reset_request_held = False
        while not stop_event.is_set():
            now = time.monotonic()
            dt = min(now - prev_time, 0.05)
            prev_time = now

            with cu_manager._lock:
                data = cu_manager.data
                hand = dict(cu_manager.hand_data) if cu_manager.hand_data else None
            if data is not None or hand is not None:
                reset_before = teleop.reset_sign
                controller_reset = controller_reset_requested(data)
                teleop.step(data, dt, hand_data=hand)
                reset_requested = controller_reset or (
                    teleop.reset_sign and not reset_before
                )
                if tactile_enabled and reset_requested and not reset_request_held:
                    request_tactile_recalibration(tactile_holder)
                reset_request_held = reset_requested

            if visualizer_handle is not None:
                recording_service.handle_visualizer_commands(visualizer_handle)

            next_tick += control_period
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
                _close_resource("stopping tactile reader", tactile_reader.stop)
        if cu_manager is not None:
            _close_resource("closing cu_manager", cu_manager.close)
        if beaver_reader is not None:
            _close_resource("closing beaver_reader", beaver_reader.close)

        if t_tactile_reader is not None:
            _close_resource(
                "joining tactile reader thread",
                lambda: t_tactile_reader.join(timeout=3.0),
            )
        if t_tactile_bridge is not None:
            _close_resource(
                "joining tactile bridge thread",
                lambda: t_tactile_bridge.join(timeout=1.0),
            )
        if t_visualizer is not None:
            _close_resource(
                "joining visualizer thread",
                lambda: t_visualizer.join(timeout=1.0),
            )
        if visualizer_handle is not None:
            _close_resource("closing visualizer", visualizer_handle.close)
        for thread in recording_threads:
            _close_resource(
                "joining recording thread",
                lambda thread=thread: thread.join(timeout=5.0),
            )
        if recording_service is not None:
            _close_resource("closing recording service", recording_service.close)
        elif dataset is not None:
            close_dataset = getattr(dataset, "close", None)
            if callable(close_dataset):
                _close_resource("closing dataset", close_dataset)

        if teleop is not None:
            close_teleop = getattr(teleop, "close", None)
            if callable(close_teleop):
                _close_resource("closing teleop", close_teleop)

        utils.logger.info("Shutdown complete.")
        classic_config.cfg = previous_config
        classic_config.viz_cfg = previous_visualizer_config


if __name__ == "__main__":
    main()
