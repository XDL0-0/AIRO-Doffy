"""RealMan entrypoint camera selection and dashboard publication."""

from __future__ import annotations

import inspect
import threading
import time

import numpy as np

import doffy_teleop.utils as utils
from doffy_teleop.config import Config
from doffy_teleop.recording.service import RealManEpisodeRecorder
from doffy_teleop.visualization.config import VisualizerConfig


def _make_webrtc_camera_manager(factory, cfg: Config):
    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return factory(config=cfg)
    accepts_config = any(
        parameter.name == "config"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    return factory(config=cfg) if accepts_config else factory()


def _create_camera_manager(cfg: Config):
    if cfg.VIDEO_TRANSPORT.lower() == "webrtc":
        from doffy_teleop.media.webrtc_manager import WebRTCUDPManager

        return _make_webrtc_camera_manager(WebRTCUDPManager, cfg)

    if cfg.VIDEO_TRANSPORT.lower() == "udp":
        from doffy_teleop.media.udp_manager import UDPManager

        return UDPManager(config=cfg)

    raise ValueError(f"Unsupported VIDEO_TRANSPORT: {cfg.VIDEO_TRANSPORT}")


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
    previews: dict[str, np.ndarray] = {}
    for name, image in sorted(images.items()):
        preview = _visualizer_image(image)
        if preview is not None:
            previews[name] = preview
    return previews


def visualizer_publish_loop(
    visualizer_handle,
    teleop: RealManTeleop,
    camera_manager,
    viz_cfg: VisualizerConfig,
    stop_event: threading.Event,
    background_errors: list[str] | None = None,
    recorder: RealManEpisodeRecorder | None = None,
    beaver_reader=None,
) -> None:
    period_s = 1.0 / viz_cfg.HZ
    force_frame = ("sensor", "work", "tool")[
        teleop.cfg.REALMAN_FORCE_COORDINATE
    ]
    try:
        while not stop_event.is_set():
            now_ns = time.monotonic_ns()
            robot = teleop.state_snapshot()
            canfd = teleop.canfd.snapshot()
            with camera_manager._lock:
                images = dict(camera_manager.camera_data)
                images.update(camera_manager.camera_images)
                capture_timestamps = dict(
                    getattr(
                        camera_manager,
                        "camera_data_timestamps_ns",
                        {},
                    )
                )
                display_timestamps = dict(
                    getattr(
                        camera_manager,
                        "camera_image_timestamps_ns",
                        {},
                    )
                )

            image_timestamps = capture_timestamps
            image_timestamps.update(display_timestamps)
            camera_timestamp_support = hasattr(
                camera_manager,
                "camera_data_timestamps_ns",
            )
            camera_stale_after_s = max(
                0.25,
                5.0
                / max(
                    1.0,
                    float(getattr(camera_manager, "realsense_fps", 30.0)),
                ),
            )
            camera_ages_s: list[float] = []
            camera_errors: list[str] = []
            if camera_manager.camera_num <= 0:
                camera_errors.append("no camera connected")
            for camera_index in range(camera_manager.camera_num):
                key = f"camera_{camera_index}"
                if key not in images:
                    camera_errors.append(f"{key} frame is unavailable")
                    continue
                timestamp_ns = int(image_timestamps.get(key, 0))
                if timestamp_ns:
                    age_s = max(0.0, (now_ns - timestamp_ns) / 1e9)
                    camera_ages_s.append(age_s)
                    if age_s > camera_stale_after_s:
                        camera_errors.append(
                            f"{key} frame is stale ({age_s * 1000.0:.0f} ms)"
                        )
                elif camera_timestamp_support:
                    camera_errors.append(f"{key} timestamp is unavailable")

            state_age_s = max(
                0.0,
                (now_ns - robot.state_timestamp_ns) / 1e9,
            )
            force_age_s = max(
                0.0,
                (now_ns - robot.force_timestamp_ns) / 1e9,
            )
            achieved = (
                "measuring"
                if canfd.achieved_hz is None
                else f"{canfd.achieved_hz:.1f} Hz"
            )
            status = (
                f"{teleop.control_mode} CAN-FD {achieved} "
                f"(target {canfd.target_hz:g} Hz)\n"
                f"max gap {canfd.max_gap_ms:.2f} ms, "
                f"control step {canfd.max_sdk_call_ms:.2f} ms, "
                f">10 ms violations "
                f"{canfd.high_follow_gap_violations + canfd.sdk_call_overruns}\n"
                f"state age {state_age_s * 1000.0:.0f} ms, "
                f"force age {force_age_s * 1000.0:.0f} ms"
            )
            if camera_ages_s:
                status += (
                    f", camera age "
                    f"{max(camera_ages_s) * 1000.0:.0f} ms"
                )
            errors = [
                message
                for message in (
                    canfd.error,
                    robot.state_error,
                    robot.force_error,
                    teleop.canfd.heartbeat_error(),
                )
                if message
            ]
            errors.extend(camera_errors)
            if not canfd.running:
                errors.append("CAN-FD sender is not running")
            if robot.input_stale:
                errors.append("VR controller input is stale")
            if state_age_s > teleop.sensor_stale_after_s:
                errors.append(
                    f"robot state is stale ({state_age_s * 1000.0:.0f} ms)"
                )
            if force_age_s > teleop.sensor_stale_after_s:
                errors.append(
                    f"robot force is stale ({force_age_s * 1000.0:.0f} ms)"
                )
            visualizer_handle.publish(
                {
                    "timestamp": min(
                        robot.state_timestamp_ns,
                        robot.force_timestamp_ns,
                    )
                    / 1e9,
                    "wrench": robot.wrench,
                    "joints": robot.joints,
                    "tcp_translation": robot.tcp_pose[:3, 3],
                    "images": _visualizer_images(images),
                    "camera_count": camera_manager.camera_num,
                    "source_label": (
                        f"RealMan robot force ({force_frame} frame)"
                    ),
                    "status_extra": status,
                    **(
                        {"wrm": teleop.wrm_visualizer_state()}
                        if teleop.wrm_enabled
                        else {}
                    ),
                    **(
                        {"dataset": recorder.recording_status()}
                        if recorder is not None
                        else {}
                    ),
                    **(
                        {"beaver": beaver_reader.visualizer_payload()}
                        if beaver_reader is not None
                        else {}
                    ),
                    "connected": not errors,
                    "error": " | ".join(errors),
                }
            )
            stop_event.wait(period_s)
    except Exception as exc:
        message = f"RealMan visualizer publisher failed: {exc}"
        utils.logger.exception(message)
        if background_errors is not None:
            background_errors.append(message)
        stop_event.set()
