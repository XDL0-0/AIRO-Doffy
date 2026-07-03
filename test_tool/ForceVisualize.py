"""Realtime UR TCP force/torque visualization dashboard.

Run with a real robot:
    python test_tool/ForceVisualize.py --ip 10.42.0.162 --robot-type ur3e

Move the TCP xyz with airo-mono servo_to_tcp_pose while keeping rotation fixed:
    python test_tool/ForceVisualize.py --ip 10.42.0.162 --robot-type ur3e \
        --payload-cog 0 0 0.058 --tcp-xyz-experiment

Preview the UI without hardware:
    python test_tool/ForceVisualize.py --mock
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from force_filter import WrenchFilter


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WrenchSample:
    timestamp: float
    wrench: np.ndarray
    joints: np.ndarray | None = None
    tcp_translation: np.ndarray | None = None
    connected: bool = True
    error: str = ""


@dataclass
class TactileSample:
    data: np.ndarray | None
    timestamp_ns: int = 0
    connected: bool = False
    mode: str = "off"
    error: str = ""


class TactileDataHolder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tactile_data: np.ndarray | None = None
        self.tactile_byte: bytes | None = None
        self.tactile_timestamp_ns: int = 0
        self.data = (None, {"Joystick_Press": False})
        self.tactile_recalibrate_requested = False


class TactileSource:
    def __init__(
        self,
        enabled: bool,
        mock: bool,
        shape: tuple[int, int],
    ) -> None:
        self.enabled = enabled
        self.mock = mock
        self.shape = shape
        self.holder = TactileDataHolder()
        self._start = time.monotonic()
        self._error = ""
        self._thread: threading.Thread | None = None
        self._reader = None
        self._close_requested = threading.Event()

        if enabled and not mock:
            self._thread = threading.Thread(target=self._run_reader, daemon=True)
            self._thread.start()

    @property
    def mode(self) -> str:
        if self.mock:
            return "mock"
        if self.enabled:
            return "sensor"
        return "off"

    def _run_reader(self) -> None:
        try:
            reader = self._create_reader()
            self._reader = reader
            if self._close_requested.is_set():
                stop = getattr(reader, "stop", None)
                if callable(stop):
                    stop()
                return
            if self._reader_name() == "ble4":
                reader.run(
                    self.holder,
                    start_visualizer=False,
                    visualizer_topic="MagTouchRaw0",
                )
            else:
                reader.run(self.holder)
        except Exception as exc:
            self._error = str(exc)
            logger.exception("Tactile reader stopped: %s", exc)

    @staticmethod
    def _reader_name() -> str:
        from config import Config

        return Config().TACTILE_READER

    @staticmethod
    def _create_reader():
        from config import Config

        cfg = Config()
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

    def read(self) -> TactileSample:
        if self.mock:
            return TactileSample(
                data=self._mock_tactile(),
                timestamp_ns=time.monotonic_ns(),
                connected=True,
                mode="mock",
            )
        if not self.enabled:
            return TactileSample(data=None, mode="off")

        with self.holder._lock:
            data = None if self.holder.tactile_data is None else self.holder.tactile_data.copy()
            timestamp_ns = self.holder.tactile_timestamp_ns

        return TactileSample(
            data=data,
            timestamp_ns=timestamp_ns,
            connected=data is not None,
            mode="sensor",
            error=self._error,
        )

    def _mock_tactile(self) -> np.ndarray:
        t = time.monotonic() - self._start
        n, axes = self.shape
        data = np.zeros((n, axes), dtype=np.float32)
        idx = np.arange(n, dtype=np.float32)
        phase = t * 2.0 + idx * 0.7
        data[:, 0] = 0.7 * np.sin(phase)
        data[:, 1] = 0.7 * np.cos(phase * 0.8)
        data[:, 2] = 2.5 * np.maximum(0.0, np.sin(t * 1.4 + idx * 0.35))
        return data

    def close(self) -> None:
        self._close_requested.set()
        stop = getattr(self._reader, "stop", None)
        if callable(stop):
            stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Tactile reader thread did not stop within 5 seconds.")


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
        self._tcp_experiment_thread: threading.Thread | None = None
        self._tcp_experiment_stop = threading.Event()
        self._tcp_experiment_original_pose: np.ndarray | None = None
        self.tcp_experiment_label = ""
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

    def start_tcp_pose_experiment(
        self,
        xyz_deltas: list[tuple[float, float, float]],
        dwell_s: float,
        cycles: int,
        servo_dt: float,
    ) -> bool:
        if self.mock:
            logger.warning("TCP pose experiment is disabled in mock mode.")
            return False
        if self.receive_only:
            logger.warning("TCP pose experiment needs airo-mono robot control; --receive-only cannot move TCP.")
            return False
        if self._tcp_experiment_thread is not None and self._tcp_experiment_thread.is_alive():
            logger.warning("TCP pose experiment is already running.")
            return False
        if not xyz_deltas:
            logger.warning("TCP pose experiment has no xyz deltas.")
            return False
        if self._freedrive_active:
            logger.info("Stopping freedrive before TCP pose experiment.")
            self.stop_freedrive()

        try:
            original_pose = self._pose_matrix_from_value(self._read_tcp_pose(optional=False))
        except Exception as exc:
            logger.warning("Could not read current TCP pose; experiment not started: %s", exc)
            return False

        self._tcp_experiment_original_pose = original_pose.copy()
        self._tcp_experiment_stop.clear()
        self._tcp_experiment_thread = threading.Thread(
            target=self._run_tcp_pose_experiment,
            args=(
                original_pose,
                xyz_deltas,
                max(0.2, float(dwell_s)),
                max(0, int(cycles)),
                max(0.02, float(servo_dt)),
            ),
            daemon=True,
        )
        self._tcp_experiment_thread.start()
        logger.info(
            "TCP pose experiment started from xyz %s. Only xyz will move; rotation stays fixed.",
            np.array2string(original_pose[:3, 3], precision=4),
        )
        return True

    def _run_tcp_pose_experiment(
        self,
        original_pose: np.ndarray,
        xyz_deltas: list[tuple[float, float, float]],
        dwell_s: float,
        cycles: int,
        servo_dt: float,
    ) -> None:
        loop_idx = 0
        try:
            while not self._tcp_experiment_stop.is_set() and (cycles == 0 or loop_idx < cycles):
                for point_idx, delta in enumerate(xyz_deltas, start=1):
                    if self._tcp_experiment_stop.is_set():
                        break
                    target_pose = original_pose.copy()
                    target_pose[:3, 3] = original_pose[:3, 3] + np.asarray(delta, dtype=float)
                    self.tcp_experiment_label = (
                        f"tcp exp {point_idx}/{len(xyz_deltas)} "
                        f"dxyz {delta[0]:+.3f} {delta[1]:+.3f} {delta[2]:+.3f}"
                    )
                    logger.info(
                        "servo_to_tcp_pose experiment xyz target: %s",
                        np.array2string(target_pose[:3, 3], precision=4),
                    )
                    self._servo_to_tcp_pose(target_pose, servo_dt)
                    self._tcp_experiment_stop.wait(dwell_s)
                loop_idx += 1
        except Exception as exc:
            logger.warning("TCP pose experiment stopped after error: %s", exc)
        finally:
            self.tcp_experiment_label = "tcp exp restoring"
            try:
                self._servo_to_tcp_pose(original_pose, servo_dt)
                logger.info("Restored original TCP pose xyz: %s", np.array2string(original_pose[:3, 3], precision=4))
            except Exception as exc:
                logger.warning("Could not restore original TCP pose: %s", exc)
            self.tcp_experiment_label = ""

    def _servo_to_tcp_pose(self, tcp_pose: np.ndarray, dt: float) -> None:
        servo = getattr(self.robot, "servo_to_tcp_pose", None)
        if servo is None:
            raise AttributeError("Robot object has no servo_to_tcp_pose method")
        servo(np.asarray(tcp_pose, dtype=float), dt)

    @staticmethod
    def _pose_matrix_from_value(tcp_pose: np.ndarray | None) -> np.ndarray:
        if tcp_pose is None:
            raise RuntimeError("TCP pose is unavailable")
        pose = np.asarray(tcp_pose, dtype=float)
        if pose.shape == (4, 4):
            return pose.copy()
        raise RuntimeError(f"Expected a 4x4 TCP pose for servo_to_tcp_pose, got shape {pose.shape}")

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
        self._tcp_experiment_stop.set()
        if self._tcp_experiment_thread is not None:
            self._tcp_experiment_thread.join(timeout=2.0)
            self._tcp_experiment_thread = None
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
        moving_average_window: int,
        low_pass_alpha: float,
        force_deadband: float,
        torque_deadband: float,
    ) -> None:
        self.bias_samples = max(0, int(bias_samples))
        self.bias = np.zeros(6, dtype=float)
        self.filter = WrenchFilter(
            moving_average_window=moving_average_window,
            low_pass_alpha=low_pass_alpha,
            force_deadband=force_deadband,
            torque_deadband=torque_deadband,
        )
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
        return self.filter.process(compensated)

    def reset_bias(self) -> None:
        self.bias = np.zeros(6, dtype=float)
        self.filter.reset()
        self.just_calibrated = False
        self._bias_buffer.clear()


def _visualizer_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    step_y = max(1, image.shape[0] // 240)
    step_x = max(1, image.shape[1] // 320)
    return image[::step_y, ::step_x, :3].copy()


def _force_status_extra(
    source: URForceSource,
    tactile: TactileSource,
    processor: WrenchProcessor,
) -> str:
    parts = []
    if processor.bias_samples > 0 and processor.calibrating:
        parts.append(f"zeroing {processor.calibration_count}/{processor.bias_samples}")
    elif processor.bias_samples > 0:
        parts.append("software zero")
    if source._freedrive_active:
        parts.append("freedrive")
    if source.tcp_experiment_label:
        parts.append(source.tcp_experiment_label)
    if processor.filter.moving_average_window > 1:
        parts.append(f"MA {processor.filter.moving_average_window}")
    if processor.filter.low_pass_alpha > 0.0:
        parts.append(f"LPF {processor.filter.low_pass_alpha:.2f}")
    if tactile.mode != "off":
        parts.append(f"tactile {tactile.mode}")
    return "  |  ".join(parts)


def run_shared_force_visualizer(
    source: URForceSource,
    camera: CameraSource,
    tactile: TactileSource,
    processor: WrenchProcessor,
    window_s: float,
    hz: float,
    freedrive_after_zero: bool,
    force_panel_range: float,
) -> None:
    from visualizer import start_visualizer

    visualizer_handle = start_visualizer(
        moving_average_window=1,
        low_pass_alpha=0.0,
        hz=hz,
        window_s=window_s,
        title="UR Force/Torque Visualization",
        force_panel_range=force_panel_range,
        camera_num=1 if camera.capture is not None else 0,
        show_rollback_button=False,
    )
    dt = 1.0 / max(1e-6, hz)
    freedrive_requested = False

    try:
        while visualizer_handle.process.is_alive():
            sample = source.read()
            wrench = sample.wrench
            connected = sample.connected
            error = sample.error
            if np.all(np.isfinite(wrench)):
                wrench = processor.process(wrench)
                if (
                    processor.just_calibrated
                    and freedrive_after_zero
                    and not freedrive_requested
                ):
                    freedrive_requested = True
                    if not source.start_freedrive():
                        logger.warning(
                            "Freedrive was requested after zeroing, but could not be enabled."
                        )
            else:
                wrench = np.zeros(6, dtype=float)
                connected = False

            tactile_sample = tactile.read()
            frame = _visualizer_image(camera.read_rgb())
            images = {"camera_0": frame} if frame is not None else {}
            source_label = f"{source.connection_mode} force"
            visualizer_handle.publish(
                {
                    "timestamp": sample.timestamp,
                    "wrench": np.asarray(wrench, dtype=float).copy(),
                    "joints": None
                    if sample.joints is None
                    else np.asarray(sample.joints, dtype=float).copy(),
                    "tcp_translation": None
                    if sample.tcp_translation is None
                    else np.asarray(sample.tcp_translation, dtype=float).copy(),
                    "images": images,
                    "camera_count": 1 if camera.capture is not None else 0,
                    "tactile": tactile_sample.data,
                    "tactile_timestamp_ns": tactile_sample.timestamp_ns,
                    "source_label": source_label,
                    "status_extra": _force_status_extra(source, tactile, processor),
                    "connected": connected,
                    "error": error,
                }
            )
            time.sleep(dt)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, closing force visualization UI.")
    finally:
        visualizer_handle.close()
        source.close()
        camera.close()
        tactile.close()


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
    parser.add_argument(
        "--tcp-xyz-experiment",
        action="store_true",
        help="Move TCP pose xyz with servo_to_tcp_pose during visualization; TCP rotation is kept from the starting pose.",
    )
    parser.add_argument(
        "--tcp-xyz-deltas",
        type=float,
        nargs="*",
        default=None,
        metavar="D",
        help=(
            "Relative TCP pose xyz deltas in meters, grouped as DX DY DZ. "
            "Default: 0, +/-1 cm on X/Y/Z, then 0."
        ),
    )
    parser.add_argument(
        "--tcp-experiment-dwell",
        type=float,
        default=3.0,
        help="Seconds to hold each TCP xyz experiment point after servo_to_tcp_pose is sent.",
    )
    parser.add_argument(
        "--tcp-servo-dt",
        type=float,
        default=0.5,
        help="Duration passed to airo-mono servo_to_tcp_pose for each TCP xyz target.",
    )
    parser.add_argument(
        "--tcp-experiment-cycles",
        type=int,
        default=1,
        help="Number of TCP xyz experiment cycles. Use 0 to repeat until the UI closes.",
    )
    parser.add_argument("--bias-samples", type=int, default=0, help="Average this many startup samples and subtract them.")
    parser.add_argument(
        "--low-pass-alpha",
        type=float,
        default=None,
        help="Exponential low-pass alpha. Defaults to Config.FORCE_LOW_PASS_ALPHA.",
    )
    parser.add_argument("--force-deadband", type=float, default=0.0, help="Subtract force deadband in N after zeroing.")
    parser.add_argument("--torque-deadband", type=float, default=0.0, help="Subtract torque deadband in Nm after zeroing.")
    parser.add_argument("--mock", action="store_true", help="Run the UI with synthetic force data.")
    parser.add_argument("--tactile", action="store_true", help="Start the configured tactile reader and show it in the UI.")
    parser.add_argument("--mock-tactile", action="store_true", help="Show synthetic tactile data without starting sensor hardware.")
    parser.add_argument("--no-tactile", action="store_true", help="Hide tactile panel data, including the default --mock preview.")
    parser.add_argument("--camera-index", type=int, default=None, help="Optional OpenCV camera index.")
    parser.add_argument("--window", type=float, default=None, help="Plot history window in seconds.")
    parser.add_argument("--hz", type=float, default=None, help="UI refresh rate.")
    parser.add_argument(
        "--force-panel-range",
        type=float,
        default=None,
        help="Fx/Fy vector panel +/- range in N. Defaults to VisualizerConfig.FORCE_PANEL_RANGE.",
    )
    args = parser.parse_args()
    if args.tcp_xyz_deltas is not None and len(args.tcp_xyz_deltas) % 3 != 0:
        parser.error("--tcp-xyz-deltas must contain groups of three numbers: DX DY DZ")
    if args.force_panel_range is not None and args.force_panel_range <= 0:
        parser.error("--force-panel-range must be positive")
    if args.window is not None and args.window <= 0:
        parser.error("--window must be positive")
    if args.hz is not None and args.hz <= 0:
        parser.error("--hz must be positive")
    if args.low_pass_alpha is not None and not 0.0 <= args.low_pass_alpha <= 1.0:
        parser.error("--low-pass-alpha must be between 0 and 1")
    return args


def tcp_xyz_experiment_deltas(values: list[float] | None) -> list[tuple[float, float, float]]:
    if values is None:
        return [
            (0.0, 0.0, 0.0),
            (0.01, 0.0, 0.0),
            (-0.01, 0.0, 0.0),
            (0.0, 0.01, 0.0),
            (0.0, -0.01, 0.0),
            (0.0, 0.0, 0.01),
            (0.0, 0.0, -0.01),
            (0.0, 0.0, 0.0),
        ]
    return [
        (float(values[idx]), float(values[idx + 1]), float(values[idx + 2]))
        for idx in range(0, len(values), 3)
    ]


def main() -> None:
    args = parse_args()
    from config import Config
    from visualizer_config import VisualizerConfig

    cfg = Config()
    viz_cfg = VisualizerConfig()
    if args.ip is None or args.robot_type is None:
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
        and not args.tcp_xyz_experiment
        and not args.mock
        and (args.ur_zero_ft or args.bias_samples > 0)
    )
    if freedrive_after_zero and args.ur_zero_ft and args.bias_samples <= 0:
        source.start_freedrive()
    if args.freedrive and not args.mock:
        source.start_freedrive()
    if args.tcp_xyz_experiment:
        source.start_tcp_pose_experiment(
            tcp_xyz_experiment_deltas(args.tcp_xyz_deltas),
            dwell_s=args.tcp_experiment_dwell,
            cycles=args.tcp_experiment_cycles,
            servo_dt=args.tcp_servo_dt,
        )

    force_panel_range = (
        viz_cfg.FORCE_PANEL_RANGE
        if args.force_panel_range is None
        else args.force_panel_range
    )
    window_s = viz_cfg.WINDOW_S if args.window is None else args.window
    hz = viz_cfg.HZ if args.hz is None else args.hz
    low_pass_alpha = (
        cfg.FORCE_LOW_PASS_ALPHA if args.low_pass_alpha is None else args.low_pass_alpha
    )
    camera = CameraSource(args.camera_index)
    tactile_mock = args.mock_tactile or (args.mock and not args.tactile and not args.no_tactile)
    tactile_enabled = (args.tactile or tactile_mock) and not args.no_tactile
    tactile = TactileSource(
        enabled=tactile_enabled,
        mock=tactile_mock,
        shape=tuple(cfg.TACTILE_SHAPE),
    )
    processor = WrenchProcessor(
        bias_samples=args.bias_samples,
        moving_average_window=cfg.FORCE_MOVING_AVERAGE_WINDOW,
        low_pass_alpha=low_pass_alpha,
        force_deadband=args.force_deadband,
        torque_deadband=args.torque_deadband,
    )
    run_shared_force_visualizer(
        source,
        camera,
        tactile,
        processor,
        window_s=window_s,
        hz=hz,
        freedrive_after_zero=freedrive_after_zero,
        force_panel_range=force_panel_range,
    )


if __name__ == "__main__":
    main()
