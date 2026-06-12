"""Realtime UR TCP force/torque visualization dashboard.

Run with a real robot:
    python test_tool/ForceVisualize.py --ip 10.42.0.162 --robot-type ur3e

Preview the UI without hardware:
    python test_tool/ForceVisualize.py --mock
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import time
from collections import deque
from dataclasses import dataclass

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")
COLORS = ("#24d9ff", "#47f091", "#ff9b66", "#ff4db8", "#b063ff", "#ffd94d")


@dataclass
class WrenchSample:
    timestamp: float
    wrench: np.ndarray
    joints: np.ndarray | None = None
    tcp_translation: np.ndarray | None = None
    connected: bool = True
    error: str = ""


class ReceiveOnlyRobot:
    """Small adapter around RTDEReceiveInterface for read-only visualization."""

    def __init__(self, ip: str) -> None:
        from rtde_receive import RTDEReceiveInterface

        self.rtde_receive = RTDEReceiveInterface(ip)

    def close(self) -> None:
        disconnect = getattr(self.rtde_receive, "disconnect", None)
        if disconnect is not None:
            disconnect()


class URForceSource:
    """Read TCP wrench from UR RTDE, or synthesize data for UI testing."""

    def __init__(
        self,
        ip: str,
        robot_type: str,
        torque_mode: bool = False,
        mock: bool = False,
        receive_only: bool = False,
        zero_ft_sensor: bool = False,
        payload_mass: float | None = None,
        payload_cog: tuple[float, float, float] | None = None,
    ) -> None:
        self.ip = ip
        self.robot_type = robot_type.lower()
        self.torque_mode = torque_mode
        self.mock = mock
        self.receive_only = receive_only
        self.zero_ft_sensor = zero_ft_sensor
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.connection_mode = "mock" if mock else "rtde"
        self.robot = None
        self._freedrive_control = None
        self._freedrive_active = False
        self._start = time.monotonic()

        if not mock:
            self.robot = self._connect_robot()
            self._apply_ur_force_setup()

    def _connect_robot(self):
        if self.receive_only:
            logger.info("Connecting to %s with RTDE receive-only", self.ip)
            self.connection_mode = "receive-only"
            return ReceiveOnlyRobot(self.ip)

        if self.torque_mode:
            from airo_robots.manipulators.hardware.ur_rtde_torque import (
                URrtdeTorque as URrtde,
            )
            from config import Config

            cfg = Config()
            kwargs = {"initial_joint_configuration": cfg.INITIAL_JOINT}
        else:
            from airo_robots.manipulators.hardware.ur_rtde import URrtde

            kwargs = {}

        if self.robot_type == "ur3e":
            robot_config = URrtde.UR3E_CONFIG
        elif self.robot_type == "ur5e":
            robot_config = URrtde.UR5E_CONFIG
        else:
            raise ValueError(f"Unsupported robot type: {self.robot_type}")

        logger.info(
            "Connecting to %s at %s (%s mode)",
            self.robot_type,
            self.ip,
            "torque" if self.torque_mode else "rtde",
        )
        try:
            robot = URrtde(self.ip, robot_config, **kwargs)
        except RuntimeError as exc:
            logger.warning("Full URrtde connection failed: %s", exc)
            logger.warning("Falling back to RTDE receive-only mode for visualization.")
            self.connection_mode = "receive-only"
            return ReceiveOnlyRobot(self.ip)

        self.connection_mode = "torque" if self.torque_mode else "rtde"
        return robot

    def _apply_ur_force_setup(self) -> None:
        if self.robot is None or (self.payload_mass is None and not self.zero_ft_sensor):
            return

        control = self._control_interface(persistent=False)
        if control is None:
            return

        try:
            if self.payload_mass is not None:
                cog = self.payload_cog if self.payload_cog is not None else (0.0, 0.0, 0.0)
                logger.info("Setting UR payload: mass=%.3f kg, cog=%s m", self.payload_mass, cog)
                control.setPayload(float(self.payload_mass), list(cog))
            if self.zero_ft_sensor:
                logger.info("Calling UR zeroFtSensor(); keep the TCP unloaded and still.")
                control.zeroFtSensor()
        except RuntimeError as exc:
            logger.warning("UR force setup failed: %s", exc)
        finally:
            if control is not getattr(self.robot, "rtde_control", None):
                disconnect = getattr(control, "disconnect", None)
                if disconnect is not None:
                    disconnect()

    def _control_interface(self, persistent: bool):
        control = getattr(self.robot, "rtde_control", None)
        if control is not None:
            return control
        if persistent and self._freedrive_control is not None:
            return self._freedrive_control

        try:
            from rtde_control import RTDEControlInterface

            control = RTDEControlInterface(self.ip)
        except RuntimeError as exc:
            logger.warning("Could not open RTDE control interface: %s", exc)
            return None

        if persistent:
            self._freedrive_control = control
        return control

    def start_freedrive(self) -> bool:
        if self.mock or self._freedrive_active:
            return self._freedrive_active

        control = self._control_interface(persistent=True)
        if control is None:
            return False

        try:
            servo_stop = getattr(control, "servoStop", None)
            if servo_stop is not None:
                servo_stop()
            start = getattr(control, "freedriveMode", None) or getattr(control, "teachMode")
            start()
            self._freedrive_active = True
            logger.info("Freedrive enabled.")
            return True
        except RuntimeError as exc:
            logger.warning("Could not enable freedrive: %s", exc)
            return False

    def stop_freedrive(self) -> None:
        if not self._freedrive_active:
            return

        control = getattr(self.robot, "rtde_control", None) or self._freedrive_control
        if control is not None:
            try:
                stop = getattr(control, "endFreedriveMode", None) or getattr(control, "endTeachMode")
                stop()
                logger.info("Freedrive disabled.")
            except RuntimeError as exc:
                logger.warning("Could not disable freedrive cleanly: %s", exc)
        self._freedrive_active = False

    def read(self) -> WrenchSample:
        now = time.monotonic()
        if self.mock:
            return self._mock_sample(now)

        try:
            wrench = self._read_tcp_force()
            joints = self._read_joints(optional=True)
            tcp_pose = self._read_tcp_pose(optional=True)
            tcp_translation = self._translation_from_pose(tcp_pose)
            return WrenchSample(now, wrench, joints, tcp_translation)
        except Exception as exc:
            return WrenchSample(
                now,
                np.full(6, np.nan, dtype=float),
                connected=False,
                error=str(exc),
            )

    def _read_tcp_force(self) -> np.ndarray:
        names = ["get_cached_tcp_force", "get_tcp_force"]
        value = self._call_robot_method(names, required=False)
        if value is None:
            value = self._call_rtde_receive_method(
                ["getActualTCPForce", "get_actual_tcp_force"],
                required=True,
            )
        wrench = np.asarray(value, dtype=float).reshape(-1)
        if wrench.size < 6:
            raise RuntimeError(f"TCP force response has {wrench.size} values, expected 6")
        return wrench[:6]

    def _read_joints(self, optional: bool = False) -> np.ndarray | None:
        names = ["get_cached_joint_configuration", "get_joint_configuration"]
        value = self._call_robot_method(names, required=False)
        if value is None:
            value = self._call_rtde_receive_method(
                ["getActualQ", "get_actual_q"],
                required=not optional,
            )
        if value is None:
            return None
        joints = np.asarray(value, dtype=float).reshape(-1)
        return joints[:6] if joints.size >= 6 else None

    def _read_tcp_pose(self, optional: bool = False) -> np.ndarray | None:
        names = ["get_cached_tcp_pose", "get_tcp_pose"]
        value = self._call_robot_method(names, required=False)
        if value is None:
            value = self._call_rtde_receive_method(
                ["getActualTCPPose", "get_actual_tcp_pose"],
                required=not optional,
            )
        if value is None:
            return None
        return np.asarray(value, dtype=float)

    def _call_robot_method(self, names: list[str], required: bool) -> object | None:
        return self._call_first_available(self.robot, names, required)

    def _call_rtde_receive_method(self, names: list[str], required: bool) -> object | None:
        receiver = getattr(self.robot, "rtde_receive", None)
        return self._call_first_available(receiver, names, required)

    @staticmethod
    def _call_first_available(
        obj: object | None,
        names: list[str],
        required: bool,
    ) -> object | None:
        if obj is None:
            if required:
                raise RuntimeError("RTDE receive interface is unavailable")
            return None

        for name in names:
            method = getattr(obj, name, None)
            if method is not None:
                return method()

        if required:
            available = ", ".join(names)
            raise AttributeError(f"No compatible method found: {available}")
        return None

    @staticmethod
    def _translation_from_pose(tcp_pose: np.ndarray | None) -> np.ndarray | None:
        if tcp_pose is None:
            return None
        if tcp_pose.shape == (4, 4):
            return tcp_pose[:3, 3]
        if tcp_pose.size >= 3:
            return tcp_pose.reshape(-1)[:3]
        return None

    def _mock_sample(self, now: float) -> WrenchSample:
        t = now - self._start
        wrench = np.array(
            [
                3.0 * math.sin(1.6 * t),
                2.2 * math.cos(1.1 * t + 0.4),
                6.0 + 1.5 * math.sin(0.7 * t),
                0.18 * math.sin(2.0 * t + 1.1),
                0.12 * math.cos(1.4 * t),
                0.08 * math.sin(1.0 * t - 0.6),
            ],
            dtype=float,
        )
        joints = np.array(
            [
                1.57 + 0.02 * math.sin(t),
                -2.07,
                1.25,
                -1.20 + 0.03 * math.cos(0.5 * t),
                -1.62,
                0.02 * math.sin(0.8 * t),
            ],
            dtype=float,
        )
        tcp_translation = np.array(
            [0.32 + 0.01 * math.sin(t), -0.08, 0.24 + 0.01 * math.cos(t)]
        )
        return WrenchSample(now, wrench, joints, tcp_translation)

    def close(self) -> None:
        if self.robot is None:
            return
        self.stop_freedrive()
        if self.torque_mode and hasattr(self.robot, "disable_torque_control"):
            try:
                self.robot.disable_torque_control()
            except Exception as exc:
                logger.warning("Could not disable torque control cleanly: %s", exc)
        if self._freedrive_control is not None:
            disconnect = getattr(self._freedrive_control, "disconnect", None)
            if disconnect is not None:
                disconnect()
            self._freedrive_control = None
        close = getattr(self.robot, "close", None)
        if close is not None:
            close()


class CameraSource:
    def __init__(self, camera_index: int | None) -> None:
        self.capture = None
        if camera_index is not None:
            self.capture = cv2.VideoCapture(camera_index)
            if not self.capture.isOpened():
                logger.warning("Could not open camera index %s", camera_index)
                self.capture.release()
                self.capture = None

    def read_rgb(self) -> np.ndarray | None:
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


class WrenchProcessor:
    def __init__(
        self,
        bias_samples: int,
        low_pass_alpha: float,
        force_deadband: float,
        torque_deadband: float,
    ) -> None:
        self.bias_samples = max(0, int(bias_samples))
        self.low_pass_alpha = float(np.clip(low_pass_alpha, 0.0, 1.0))
        self.force_deadband = max(0.0, float(force_deadband))
        self.torque_deadband = max(0.0, float(torque_deadband))
        self.bias = np.zeros(6, dtype=float)
        self.filtered = np.zeros(6, dtype=float)
        self.initialized = False
        self.just_calibrated = False
        self._bias_buffer: list[np.ndarray] = []

    @property
    def calibrating(self) -> bool:
        return len(self._bias_buffer) < self.bias_samples

    @property
    def calibration_count(self) -> int:
        return min(len(self._bias_buffer), self.bias_samples)

    def process(self, wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench, dtype=float)
        self.just_calibrated = False
        if self.bias_samples > 0 and len(self._bias_buffer) < self.bias_samples:
            self._bias_buffer.append(wrench.copy())
            if len(self._bias_buffer) == self.bias_samples:
                self.bias = np.mean(np.vstack(self._bias_buffer), axis=0)
                self.just_calibrated = True
                logger.info("Software force bias calibrated: %s", np.array2string(self.bias, precision=4))
            return np.zeros(6, dtype=float)

        compensated = wrench - self.bias
        compensated[:3] = self._apply_deadband(compensated[:3], self.force_deadband)
        compensated[3:] = self._apply_deadband(compensated[3:], self.torque_deadband)

        if self.low_pass_alpha <= 0.0:
            return compensated
        if not self.initialized:
            self.filtered = compensated.copy()
            self.initialized = True
        else:
            alpha = self.low_pass_alpha
            self.filtered = alpha * compensated + (1.0 - alpha) * self.filtered
        return self.filtered.copy()

    def reset_bias(self) -> None:
        self.bias = np.zeros(6, dtype=float)
        self.filtered = np.zeros(6, dtype=float)
        self.initialized = False
        self.just_calibrated = False
        self._bias_buffer.clear()

    @staticmethod
    def _apply_deadband(values: np.ndarray, threshold: float) -> np.ndarray:
        if threshold <= 0.0:
            return values
        result = values.copy()
        mask = np.abs(result) < threshold
        result[mask] = 0.0
        result[~mask] -= np.sign(result[~mask]) * threshold
        return result


class ForceDashboard:
    def __init__(
        self,
        source: URForceSource,
        camera: CameraSource,
        processor: WrenchProcessor,
        window_s: float,
        hz: float,
        freedrive_after_zero: bool,
    ) -> None:
        self.source = source
        self.camera = camera
        self.processor = processor
        self.window_s = window_s
        self.freedrive_after_zero = freedrive_after_zero
        self._freedrive_requested = False
        self.interval_ms = max(10, int(1000 / hz))
        self.times: deque[float] = deque(maxlen=max(10, int(window_s * hz * 2)))
        self.values: deque[np.ndarray] = deque(maxlen=max(10, int(window_s * hz * 2)))
        self.last_sample: WrenchSample | None = None

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(15.5, 8.6), facecolor="#080b10")
        self.fig.canvas.manager.set_window_title("UR Force/Torque Visualization")
        outer = GridSpec(
            1,
            2,
            figure=self.fig,
            width_ratios=[2.15, 1.0],
            wspace=0.08,
            left=0.035,
            right=0.985,
            top=0.9,
            bottom=0.07,
        )
        plot_grid = outer[0, 0].subgridspec(3, 2, hspace=0.28, wspace=0.18)
        side_grid = outer[0, 1].subgridspec(4, 1, height_ratios=[0.65, 1.0, 1.0, 0.75], hspace=0.16)

        self.axes = []
        self.lines = []
        self.value_texts = []
        for idx, name in enumerate(CHANNELS):
            ax = self.fig.add_subplot(plot_grid[idx // 2, idx % 2])
            self._style_plot_axis(ax, name, idx)
            (line,) = ax.plot([], [], color=COLORS[idx], linewidth=1.8)
            value_text = ax.text(
                0.02,
                0.88,
                "--",
                transform=ax.transAxes,
                color=COLORS[idx],
                fontsize=12,
                fontweight="bold",
            )
            self.axes.append(ax)
            self.lines.append(line)
            self.value_texts.append(value_text)

        self.status_ax = self.fig.add_subplot(side_grid[0])
        self.vector_ax = self.fig.add_subplot(side_grid[1])
        self.camera_ax = self.fig.add_subplot(side_grid[2])
        self.pose_ax = self.fig.add_subplot(side_grid[3])
        self._style_panel_axis(self.status_ax, "Status")
        self._style_panel_axis(self.vector_ax, "TCP Force Vector")
        self._style_panel_axis(self.camera_ax, "Camera")
        self._style_panel_axis(self.pose_ax, "Robot State")

        self.status_text = self.status_ax.text(0.04, 0.72, "", transform=self.status_ax.transAxes, fontsize=11)
        self.force_mag_text = self.status_ax.text(
            0.04,
            0.34,
            "",
            transform=self.status_ax.transAxes,
            fontsize=18,
            fontweight="bold",
            color="#24d9ff",
        )
        self.torque_mag_text = self.status_ax.text(
            0.62,
            0.34,
            "",
            transform=self.status_ax.transAxes,
            fontsize=18,
            fontweight="bold",
            color="#ffd94d",
        )

        self.vector_ax.set_xlim(-1.0, 1.0)
        self.vector_ax.set_ylim(-1.0, 1.0)
        self.vector_ax.set_aspect("equal", adjustable="box")
        self.vector_ax.axhline(0, color="#263446", linewidth=1)
        self.vector_ax.axvline(0, color="#263446", linewidth=1)
        self.vector_arrow = self.vector_ax.arrow(0, 0, 0, 0, color="#24d9ff", width=0.015)
        self.vector_label = self.vector_ax.text(0.04, 0.9, "", transform=self.vector_ax.transAxes, fontsize=10)

        self.camera_ax.set_xticks([])
        self.camera_ax.set_yticks([])
        self.camera_image = None
        self.camera_placeholder = self.camera_ax.text(
            0.5,
            0.5,
            "No camera",
            transform=self.camera_ax.transAxes,
            ha="center",
            va="center",
            color="#7e8ca0",
            fontsize=12,
        )

        self.pose_text = self.pose_ax.text(
            0.04,
            0.8,
            "",
            transform=self.pose_ax.transAxes,
            va="top",
            family="monospace",
            fontsize=10,
        )

        mode = source.connection_mode.upper()
        self.fig.suptitle(
            f"6D Force/Torque Visualization    {source.robot_type.upper()} {source.ip}    {mode}",
            x=0.035,
            ha="left",
            color="#e8f1ff",
            fontsize=16,
            fontweight="bold",
        )

    @staticmethod
    def _style_plot_axis(ax, name: str, idx: int) -> None:
        ax.set_facecolor("#101722")
        for spine in ax.spines.values():
            spine.set_color("#425066")
        ax.grid(True, color="#263446", alpha=0.7, linewidth=0.8)
        ax.tick_params(colors="#9fb0c5", labelsize=8)
        ax.set_title(f"{name} ({UNITS[idx]})", loc="left", color="#e8f1ff", fontsize=10, pad=7)
        ax.set_xlabel("seconds", color="#9fb0c5", fontsize=8)
        ax.set_ylabel(UNITS[idx], color="#9fb0c5", fontsize=8)

    @staticmethod
    def _style_panel_axis(ax, title: str) -> None:
        ax.set_facecolor("#101722")
        for spine in ax.spines.values():
            spine.set_color("#425066")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left", color="#e8f1ff", fontsize=10, pad=7)

    def start(self) -> None:
        previous_sigint = signal.getsignal(signal.SIGINT)

        def close_on_sigint(_signum, _frame):
            logger.info("Ctrl-C received, closing force visualization UI.")
            plt.close(self.fig)

        signal.signal(signal.SIGINT, close_on_sigint)
        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        self._animation = ani
        try:
            plt.show()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, closing force visualization UI.")
            plt.close(self.fig)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            self.source.close()
            self.camera.close()

    def _update(self, _frame):
        sample = self.source.read()
        if np.all(np.isfinite(sample.wrench)):
            sample.wrench = self.processor.process(sample.wrench)
            if self.processor.just_calibrated:
                self._start_freedrive_after_zero()
        self.last_sample = sample
        if np.all(np.isfinite(sample.wrench)):
            if not self.times:
                self.t0 = sample.timestamp
            self.times.append(sample.timestamp - self.t0)
            self.values.append(sample.wrench.copy())

        self._update_plots(sample)
        self._update_status(sample)
        self._update_vector(sample)
        self._update_camera()
        self._update_pose(sample)
        return []

    def _start_freedrive_after_zero(self) -> None:
        if not self.freedrive_after_zero or self._freedrive_requested:
            return
        self._freedrive_requested = True
        if not self.source.start_freedrive():
            logger.warning("Freedrive was requested after zeroing, but could not be enabled.")

    def _update_plots(self, sample: WrenchSample) -> None:
        if not self.times:
            return
        xs = np.asarray(self.times)
        ys = np.vstack(self.values)
        xmin = max(0.0, xs[-1] - self.window_s)
        xmax = max(self.window_s, xs[-1])

        for idx, ax in enumerate(self.axes):
            self.lines[idx].set_data(xs, ys[:, idx])
            ax.set_xlim(xmin, xmax)
            recent = ys[xs >= xmin, idx]
            center = 0.0
            spread = 1.0
            if recent.size:
                lo = float(np.nanmin(recent))
                hi = float(np.nanmax(recent))
                center = 0.5 * (lo + hi)
                spread = max(1e-3, hi - lo)
            margin = max(0.4 if idx < 3 else 0.04, spread * 0.65)
            ax.set_ylim(center - margin, center + margin)
            value = sample.wrench[idx]
            self.value_texts[idx].set_text(f"{value:+.3f} {UNITS[idx]}")

    def _update_status(self, sample: WrenchSample) -> None:
        age = 0.0 if sample is None else time.monotonic() - sample.timestamp
        state = "CONNECTED" if sample.connected else "ERROR"
        color = "#47f091" if sample.connected else "#ff6b6b"
        self.status_text.set_color(color)
        mode = self.source.connection_mode
        extra = ""
        if self.processor.bias_samples > 0 and self.processor.calibrating:
            extra = (
                f"  |  zeroing {self.processor.calibration_count}/"
                f"{self.processor.bias_samples}"
            )
        elif self.processor.bias_samples > 0:
            extra = "  |  software zero"
        if self.source._freedrive_active:
            extra += "  |  freedrive"
        if self.processor.low_pass_alpha > 0.0:
            extra += f"  |  LPF {self.processor.low_pass_alpha:.2f}"
        self.status_text.set_text(f"{state}  |  {mode}  |  latency {age * 1000:.0f} ms{extra}")

        wrench = sample.wrench if np.all(np.isfinite(sample.wrench)) else np.zeros(6)
        force_mag = float(np.linalg.norm(wrench[:3]))
        torque_mag = float(np.linalg.norm(wrench[3:]))
        self.force_mag_text.set_text(f"|F| {force_mag:5.2f} N")
        self.torque_mag_text.set_text(f"|T| {torque_mag:5.3f} Nm")

        if sample.error:
            self.status_text.set_text(f"{state}  |  {sample.error[:58]}")

    def _update_vector(self, sample: WrenchSample) -> None:
        self.vector_arrow.remove()
        wrench = sample.wrench if np.all(np.isfinite(sample.wrench)) else np.zeros(6)
        fxy = wrench[:2]
        fz = float(wrench[2])
        norm = max(1.0, float(np.linalg.norm(fxy)))
        dx, dy = np.clip(fxy / norm, -0.9, 0.9)
        self.vector_arrow = self.vector_ax.arrow(
            0,
            0,
            dx * 0.82,
            dy * 0.82,
            color="#24d9ff",
            width=0.018,
            length_includes_head=True,
            head_width=0.09,
        )
        self.vector_label.set_text(f"Fx/Fy direction   Fz {fz:+.2f} N")

    def _update_camera(self) -> None:
        frame = self.camera.read_rgb()
        if frame is None:
            return
        self.camera_placeholder.set_visible(False)
        if self.camera_image is None:
            self.camera_image = self.camera_ax.imshow(frame)
        else:
            self.camera_image.set_data(frame)

    def _update_pose(self, sample: WrenchSample) -> None:
        lines = []
        if sample.tcp_translation is not None:
            x, y, z = sample.tcp_translation
            lines.append(f"tcp xyz  {x:+.3f} {y:+.3f} {z:+.3f} m")
        else:
            lines.append("tcp xyz  unavailable")

        if sample.joints is not None and sample.joints.size >= 6:
            deg = np.degrees(sample.joints[:6])
            lines.append("joint deg")
            lines.append(" ".join(f"{v:+5.1f}" for v in deg[:3]))
            lines.append(" ".join(f"{v:+5.1f}" for v in deg[3:6]))
        else:
            lines.append("joint deg unavailable")
        self.pose_text.set_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime UR force/torque visualization")
    parser.add_argument("--ip", default=None, help="UR robot IP. Defaults to config.Config.UR_IP.")
    parser.add_argument("--robot-type", default=None, choices=["ur3e", "ur5e"], help="UR model.")
    parser.add_argument("--torque-mode", action="store_true", help="Read cached force from URrtdeTorque.")
    parser.add_argument("--receive-only", action="store_true", help="Only open RTDE receive; useful when control is unavailable.")
    parser.add_argument("--ur-zero-ft", action="store_true", help="Call UR zeroFtSensor() at startup; requires RTDE control.")
    parser.add_argument("--freedrive", action="store_true", help="Enter freedrive immediately after startup.")
    parser.add_argument("--no-freedrive-after-zero", action="store_true", help="Do not enter freedrive after zeroing finishes.")
    parser.add_argument("--payload-mass", type=float, default=None, help="Set UR payload mass in kg before reading force.")
    parser.add_argument(
        "--payload-cog",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Set UR payload center of gravity in meters, e.g. --payload-cog 0 0 0.058.",
    )
    parser.add_argument("--bias-samples", type=int, default=0, help="Average this many startup samples and subtract them.")
    parser.add_argument("--low-pass-alpha", type=float, default=0.0, help="Exponential low-pass alpha, e.g. 0.15. 0 disables.")
    parser.add_argument("--force-deadband", type=float, default=0.0, help="Subtract force deadband in N after zeroing.")
    parser.add_argument("--torque-deadband", type=float, default=0.0, help="Subtract torque deadband in Nm after zeroing.")
    parser.add_argument("--mock", action="store_true", help="Run the UI with synthetic force data.")
    parser.add_argument("--camera-index", type=int, default=None, help="Optional OpenCV camera index.")
    parser.add_argument("--window", type=float, default=8.0, help="Plot history window in seconds.")
    parser.add_argument("--hz", type=float, default=30.0, help="UI refresh rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ip is None or args.robot_type is None:
        from config import Config

        cfg = Config()
        args.ip = args.ip or cfg.UR_IP
        args.robot_type = args.robot_type or cfg.ROBOT_TYPE

    source = URForceSource(
        ip=args.ip,
        robot_type=args.robot_type,
        torque_mode=args.torque_mode,
        mock=args.mock,
        receive_only=args.receive_only,
        zero_ft_sensor=args.ur_zero_ft,
        payload_mass=args.payload_mass,
        payload_cog=tuple(args.payload_cog) if args.payload_cog is not None else None,
    )
    freedrive_after_zero = (
        not args.no_freedrive_after_zero
        and not args.mock
        and (args.ur_zero_ft or args.bias_samples > 0)
    )
    if freedrive_after_zero and args.ur_zero_ft and args.bias_samples <= 0:
        source.start_freedrive()
    if args.freedrive and not args.mock:
        source.start_freedrive()

    camera = CameraSource(args.camera_index)
    processor = WrenchProcessor(
        bias_samples=args.bias_samples,
        low_pass_alpha=args.low_pass_alpha,
        force_deadband=args.force_deadband,
        torque_deadband=args.torque_deadband,
    )
    dashboard = ForceDashboard(
        source,
        camera,
        processor,
        window_s=args.window,
        hz=args.hz,
        freedrive_after_zero=freedrive_after_zero,
    )
    dashboard.start()


if __name__ == "__main__":
    main()
