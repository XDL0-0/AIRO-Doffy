"""Shared RealMan teleop snapshots and Quest state transport."""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import threading
import time
from typing import Any, Callable

import numpy as np

import doffy_teleop.utils as utils

def pack_quest_tcp_state_packet(tcp_pose: np.ndarray, wrench: np.ndarray, tcp_display_axes: np.ndarray) -> bytes:
    """Encode one RealMan TCP state for Quest ``TCPPoseReceiver``.

    Position and rotation are converted from RealMan base frame (X forward,
    Y left, Z up) into the Unity world frame (X right, Y up, Z forward)
    so that they line up correctly after the user's calibration.

    Force ``[Fx, Fy, Fz]`` is in newtons; torque is not currently sent.
    """

    pose = np.asarray(tcp_pose, dtype=float)
    values = np.asarray(wrench, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("Quest TCP state packets require a finite 4x4 TCP pose.")
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Quest TCP state packets require six finite wrench values.")

    to_unity = np.asarray(tcp_display_axes, dtype=float)

    position_unity = to_unity @ pose[:3, 3]
    rotation_unity = to_unity @ pose[:3, :3] @ to_unity.T
    force_unity = to_unity @ values[:3]

    quaternion_xyzw = utils.quat_cal(rotation_unity)
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    message = {
        "rightTCP": {
            "position": position_unity.tolist(),
            "rotation": quaternion_wxyz.tolist(),
            "force": force_unity.tolist(),
        }
    }
    return json.dumps(message).encode("utf-8")


@dataclass(frozen=True)
class CanfdLoopSnapshot:
    """Thread-safe diagnostic snapshot for the high-rate command loop."""

    target_hz: float
    achieved_hz: float | None
    total_commands: int
    deadline_misses: int
    high_follow_gap_violations: int
    sdk_call_overruns: int
    max_gap_ms: float
    max_sdk_call_ms: float
    last_command_start_ns: int
    last_command_success_ns: int
    completed_timing_windows: int
    consecutive_timing_failure_windows: int
    timing_verified: bool
    running: bool
    error: str


@dataclass(frozen=True)
class RealManStateSnapshot:
    """Latest measured robot state, copied out of the sensor cache."""

    joints: np.ndarray
    tcp_pose: np.ndarray
    wrench: np.ndarray
    state_timestamp_ns: int
    force_timestamp_ns: int
    state_error: str
    force_error: str
    input_stale: bool


class QuestTcpStateSender:
    """Publish cached RealMan TCP pose and force to Quest at a fixed rate."""

    def __init__(
        self,
        teleop: "RealManTeleop",
        *,
        quest_ip: str,
        port: int,
        send_rate_hz: float,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ) -> None:
        if not str(quest_ip).strip():
            raise ValueError("quest_ip cannot be empty.")
        if not 1 <= int(port) <= 65535:
            raise ValueError("Quest TCP state port must be between 1 and 65535.")
        if float(send_rate_hz) <= 0.0:
            raise ValueError("Quest TCP state send rate must be positive.")

        self.teleop = teleop
        self.destination = (str(quest_ip), int(port))
        self.period_s = 1.0 / float(send_rate_hz)
        self._socket_factory = socket_factory

    def run(
        self,
        stop_event: threading.Event,
        background_errors: list[str] | None = None,
    ) -> None:
        """Send the latest fresh TCP state until the stop event is set."""

        udp_socket = None
        next_send = time.monotonic()
        skipped_reason = ""
        try:
            udp_socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            utils.logger.info(
                "Quest TCP state transfer enabled: %s:%d at %.1f Hz.",
                self.destination[0],
                self.destination[1],
                1.0 / self.period_s,
            )
            while not stop_event.is_set():
                snapshot = self.teleop.state_snapshot()
                force_age_s = max(
                    0.0,
                    (time.monotonic_ns() - snapshot.force_timestamp_ns) / 1e9,
                )
                state_age_s = max(
                    0.0,
                    (time.monotonic_ns() - snapshot.state_timestamp_ns) / 1e9,
                )
                reason = snapshot.state_error or snapshot.force_error
                if not reason and state_age_s > self.teleop.sensor_stale_after_s:
                    reason = f"TCP pose is stale ({state_age_s * 1000.0:.0f} ms)"
                if not reason and force_age_s > self.teleop.sensor_stale_after_s:
                    reason = f"force is stale ({force_age_s * 1000.0:.0f} ms)"

                if reason:
                    if reason != skipped_reason:
                        utils.logger.warning(
                            "Quest TCP state transfer paused: %s.",
                            reason,
                        )
                    skipped_reason = reason
                else:
                    if skipped_reason:
                        utils.logger.info("Quest TCP state transfer resumed.")
                    skipped_reason = ""
                    packet = pack_quest_tcp_state_packet(
                        snapshot.tcp_pose,
                        snapshot.wrench,
                        self.teleop.cfg.TCP_DISPLAY_AXES,
                    )
                    udp_socket.sendto(packet, self.destination)

                next_send += self.period_s
                now = time.monotonic()
                if next_send < now:
                    next_send = now
                stop_event.wait(next_send - now)
        except Exception as exc:
            message = f"Quest TCP state sender failed: {exc}"
            utils.logger.exception(message)
            if background_errors is not None:
                background_errors.append(message)
            stop_event.set()
        finally:
            if udp_socket is not None:
                udp_socket.close()
