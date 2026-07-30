"""Deprecated 4-taxel BLE implementation retained behind a lazy facade."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import multiprocessing as mp
import os
import queue
import sys
import time

import numpy as np
from loguru import logger

from sensor_comm_dds.communication.config.ble_config import (
    DeviceMAC,
    SensorUuid,
)
from sensor_comm_dds.communication.data_classes.magtouch_taxel import MagTouchTaxel
from sensor_comm_dds.communication.readers.magtouch_ble_reader import (
    MagTouchBleReader,
    MagTouchBleReaderConfig,
)

RECONNECT_DELAY_S = 1.0
PANEL_QUEUE_SIZE = 2
CALIBRATION_STALE_TIMEOUT_S = 3.0


class _PanelVisualizerHandle:
    def __init__(self, process: mp.Process, data_queue: mp.Queue):
        self.process = process
        self.data_queue = data_queue

    def publish(self, tactile_data: np.ndarray) -> None:
        while True:
            try:
                self.data_queue.put_nowait(tactile_data.astype(np.float32, copy=True))
                return
            except queue.Full:
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    return

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        try:
            self.data_queue.cancel_join_thread()
            self.data_queue.close()
        except Exception:
            pass


def _format_panel_data(tactile_data: np.ndarray | None) -> np.ndarray:
    data = np.zeros((2, 2, 3), dtype=np.float64)
    if tactile_data is None:
        return data

    tactile = np.asarray(tactile_data, dtype=np.float64)
    if tactile.shape == (2, 2, 3):
        return tactile
    if tactile.ndim != 2 or tactile.shape[0] < 4 or tactile.shape[1] < 3:
        return data

    for i, taxel in enumerate(tactile[:4, :3]):
        mapped = np.array([-taxel[0], taxel[1], taxel[2]], dtype=np.float64)
        if i == 0:
            data[1, 0] = mapped
        elif i == 1:
            data[1, 1] = mapped
        elif i == 2:
            data[0, 1] = mapped
        elif i == 3:
            data[0, 0] = mapped
    return data


def _panel_color(values: np.ndarray) -> tuple[float, float, float]:
    norm = min(1.0, float(np.linalg.norm(values)) / (4000.0 * np.sqrt(3.0)))
    cold = np.array([0x43, 0x4A, 0x52], dtype=np.float64) / 255.0
    hot = np.array([0x7D, 0xB5, 0xA8], dtype=np.float64) / 255.0
    return tuple((1.0 - norm) * cold + norm * hot)


def _panel_visualizer_process(data_queue: mp.Queue, topic_name: str) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-airo-doffy"

    import matplotlib

    if os.environ.get("DISPLAY"):
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(5.0, 5.2), facecolor="#080b10")
    fig.canvas.manager.set_window_title("4-Point Tactile")
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.08, 0.92)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#101722")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(topic_name, color="#e8f1ff")

    base_x = np.array([[0.34, 0.66], [0.34, 0.66]], dtype=np.float64)
    base_y = np.array([[0.66, 0.66], [0.34, 0.34]], dtype=np.float64)
    radius_min = 0.045
    radius_max = 0.13
    offset_max = 0.13
    max_xy = 4000.0
    min_norm = 100.0
    max_norm = max_xy * np.sqrt(3.0)
    latest = np.zeros((2, 2, 3), dtype=np.float64)

    lines = []
    circles = []
    for row in range(2):
        line_row = []
        circle_row = []
        for col in range(2):
            line = ax.plot([], [], color="#d6ae72", linewidth=2.4)[0]
            circle = Circle(
                (base_x[row, col], base_y[row, col]),
                radius_min,
                facecolor="#434a52",
                edgecolor="#434a52",
                linewidth=0.8,
            )
            ax.add_patch(circle)
            line_row.append(line)
            circle_row.append(circle)
        lines.append(line_row)
        circles.append(circle_row)

    status = ax.text(
        0.5,
        0.04,
        "waiting for tactile data",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#7e8ca0",
        fontsize=9,
    )

    def update(_frame):
        nonlocal latest
        got_sample = False
        while True:
            try:
                latest = _format_panel_data(data_queue.get_nowait())
                got_sample = True
            except queue.Empty:
                break

        for row in range(2):
            for col in range(2):
                values = latest[row, col]
                offset_x = values[0] / max_xy * offset_max
                offset_y = values[1] / max_xy * offset_max
                x = base_x[row, col] + offset_x
                y = base_y[row, col] - offset_y
                radius = (
                    np.sqrt(max(abs(values[2]) - min_norm, 0.0) / max_norm)
                    * radius_max
                    + radius_min
                )
                radius = float(np.clip(radius, radius_min, radius_max))
                color = _panel_color(values)

                lines[row][col].set_data([base_x[row, col], x], [base_y[row, col], y])
                circles[row][col].center = (x, y)
                circles[row][col].radius = radius
                circles[row][col].set_facecolor(color)
                circles[row][col].set_edgecolor(color)

        if got_sample:
            status.set_text("live")
            status.set_color("#47f091")
        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=33,
        blit=False,
        cache_frame_data=False,
    )
    fig._tactile_animation = ani
    plt.show()


class FourPointTactileBleReader(MagTouchBleReader):
    """BLE reader for one 2x2 MagTouch sensor.

    Data written to ``cu`` is a fixed ``(4, 3)`` float32 array. The same
    filtered values are published on ``MagTouchRaw0`` for the raw visualizer.
    """

    def __init__(
        self,
        config: MagTouchBleReaderConfig,
        filter_alpha: float = 0.75,
        use_kalman: bool = False,
        kalman_q: float = 2e-2,
        kalman_r: float = 2e-2,
        deadband_sigma: float = 3.0,
        noise_floor: float = 2.0,
        max_delta: float = 10000.0,
        max_abs: float = 20000.0,
        baseline_drift_alpha: float = 0.0,
        baseline_drift_threshold: float = 80.0,
        reset_trigger_threshold: float = 0.8,
    ):
        if config.NUM_SENSORS != 1 or config.NUM_TAXELS != 4:
            raise ValueError("FourPointTactileBleReader expects one 4-taxel sensor")

        self.filter_alpha = float(np.clip(filter_alpha, 0.0, 1.0))
        self.use_kalman = bool(use_kalman)
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.deadband_sigma = deadband_sigma
        self.noise_floor = noise_floor
        self.max_delta = max_delta
        self.max_abs = max_abs
        self.baseline_drift_alpha = float(np.clip(baseline_drift_alpha, 0.0, 1.0))
        self.baseline_drift_threshold = baseline_drift_threshold
        self.reset_trigger_threshold = reset_trigger_threshold

        self._cu = None
        self._raw_baseline = np.zeros((1, 4, 3), dtype=np.float64)
        self._deadband = np.full((1, 4, 3), noise_floor, dtype=np.float64)
        self._ema_state: np.ndarray | None = None
        self._kalman_x: np.ndarray | None = None
        self._kalman_p: np.ndarray | None = None
        self._last_good: np.ndarray | None = None
        self._raw_unwrapped_prev: np.ndarray | None = None
        self._reset_held = False
        self._panel_visualizer: _PanelVisualizerHandle | None = None
        self._subscribed = False
        self._stop_requested = False

        self._ensure_event_loop()
        registered_atexit_callbacks = []
        original_atexit_register = atexit.register

        def _capture_atexit_register(func, *args, **kwargs):
            registered_atexit_callbacks.append(func)
            return original_atexit_register(func, *args, **kwargs)

        atexit.register = _capture_atexit_register
        try:
            super().__init__(config=config)
        finally:
            atexit.register = original_atexit_register
            for callback in registered_atexit_callbacks:
                with contextlib.suppress(Exception):
                    atexit.unregister(callback)
        if self.connection.is_connected:
            self.disconnected_event.clear()
        self._configure_logging()

    @staticmethod
    def _ensure_event_loop() -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    @staticmethod
    def _configure_logging() -> None:
        logger.remove()
        logger.add(
            sys.stderr,
            filter=lambda record: "refresh" not in record["extra"],
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )
        logger.add(
            lambda msg: sys.stderr.write(msg.rstrip() + "\r") or sys.stderr.flush(),
            format="{message}",
            filter=lambda record: "refresh" in record["extra"],
        )

    def _reset_filter_state(self) -> None:
        self._ema_state = None
        self._kalman_x = None
        self._kalman_p = None
        self._last_good = None

    def _clear_tactile_output(self) -> None:
        if self._cu is None:
            return
        with self._cu._lock:
            self._cu.tactile_data = None
            self._cu.tactile_byte = None
            self._cu.tactile_timestamp_ns = 0

    async def connect(self):
        while not self._stop_requested:
            if self.connection.is_connected:
                logger.debug(f"Tried to connect to {self.device_mac} but already connected.")
                self.disconnected_event.clear()
                return

            try:
                logger.debug(f"Connecting to {self.device_mac}")
                await self.connection.connect()
            except Exception as exc:
                logger.warning(
                    f"BLE connect to {self.device_mac} failed: {exc}; retrying in "
                    f"{RECONNECT_DELAY_S:.1f}s"
                )
                await asyncio.sleep(RECONNECT_DELAY_S)
                continue

            if self.connection.is_connected:
                logger.debug(f"Connected to {self.device_mac}")
                self.disconnected_event.clear()
                return

            logger.warning(
                f"BLE connect to {self.device_mac} returned without a connection; "
                f"retrying in {RECONNECT_DELAY_S:.1f}s"
            )
            await asyncio.sleep(RECONNECT_DELAY_S)
        logger.info(f"BLE connect cancelled for {self.device_mac}")

    async def _subscribe_with_retry(self, callback=None) -> None:
        while not self._stop_requested:
            try:
                if not self.connection.is_connected:
                    await self.connect()
                if self._stop_requested:
                    return
                await self.subscribe(callback=callback)
                self._subscribed = True
                self.disconnected_event.clear()
                return
            except Exception as exc:
                self._subscribed = False
                logger.warning(
                    f"BLE subscribe to {self.device_mac} failed: {exc}; reconnecting in "
                    f"{RECONNECT_DELAY_S:.1f}s"
                )
                await self._safe_disconnect()
                await asyncio.sleep(RECONNECT_DELAY_S)
                await self.connect()

    async def _safe_unsubscribe(self) -> None:
        if not self._subscribed:
            return
        with contextlib.suppress(Exception):
            await self.unsubscribe()
        self._subscribed = False

    async def _safe_disconnect(self) -> None:
        await self._safe_unsubscribe()
        if self.connection.is_connected:
            with contextlib.suppress(Exception):
                await self.disconnect()

    def data_convert(self, data_bytes):
        wrapped = np.zeros(
            (self.config.NUM_SENSORS, self.config.NUM_TAXELS, 3), dtype=np.float64
        )
        for sensor_idx in range(self.config.NUM_SENSORS):
            offset = sensor_idx * 6 * self.config.NUM_TAXELS
            for taxel_idx in range(self.config.NUM_TAXELS):
                base = offset + taxel_idx * 6
                for axis in range(3):
                    word = (data_bytes[base + axis * 2] << 8) + data_bytes[base + axis * 2 + 1]
                    wrapped[sensor_idx, taxel_idx, axis] = ~word

        sample_raw = self._unwrap_raw_sample(wrapped)
        sample = self._scale_raw_sample(sample_raw)
        return sample, sample_raw

    def _unwrap_raw_sample(self, wrapped: np.ndarray) -> np.ndarray:
        if self._raw_unwrapped_prev is None:
            self._raw_unwrapped_prev = wrapped.copy()
            return wrapped.copy()

        candidates = np.stack(
            (wrapped - 65536.0, wrapped, wrapped + 65536.0), axis=0
        )
        best_idx = np.argmin(
            np.abs(candidates - self._raw_unwrapped_prev[np.newaxis, ...]), axis=0
        )
        sample_raw = np.take_along_axis(
            candidates, best_idx[np.newaxis, ...], axis=0
        )[0]
        self._raw_unwrapped_prev = sample_raw.copy()
        return sample_raw

    def _scale_raw_sample(self, sample_raw: np.ndarray) -> np.ndarray:
        gain = self.config.GAIN
        resolution = self.config.RESOLUTION
        lsb_xy = self.config.mlx90393_lsb_lookup[0][gain][resolution][0]
        lsb_z = self.config.mlx90393_lsb_lookup[0][gain][resolution][1]
        scale = np.array([lsb_xy, lsb_xy, lsb_z], dtype=np.float64)
        offset = scale * 65536.0
        return sample_raw * scale + offset

    async def _collect_calibration_samples(self) -> None:
        self.calibration_ctr = 0
        self.calibration_samples.fill(0.0)
        self.raw_calibration_samples.fill(0.0)
        self.reading_first_sample = True
        self._raw_unwrapped_prev = None

        await self._subscribe_with_retry(callback=self.calibration_callback)
        last_ctr = self.calibration_ctr
        last_progress = time.monotonic()
        while self.calibration_ctr < self.config.WINDOW_SIZE and not self._stop_requested:
            if self.disconnected_event.is_set():
                logger.warning(
                    "BLE tactile disconnected during calibration from "
                    f"{self.device_mac}; reconnecting..."
                )
                self._subscribed = False
                await self._safe_disconnect()
                await self.connect()
                await self._subscribe_with_retry(callback=self.calibration_callback)
                self.disconnected_event.clear()
                last_ctr = self.calibration_ctr
                last_progress = time.monotonic()

            if self.calibration_ctr > last_ctr:
                last_ctr = self.calibration_ctr
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > CALIBRATION_STALE_TIMEOUT_S:
                logger.warning(
                    f"No BLE tactile calibration samples from {self.device_mac} for "
                    f"{CALIBRATION_STALE_TIMEOUT_S:.1f}s; reconnecting..."
                )
                self._subscribed = False
                await self._safe_disconnect()
                await self.connect()
                await self._subscribe_with_retry(callback=self.calibration_callback)
                self.disconnected_event.clear()
                last_progress = time.monotonic()

            await asyncio.sleep(0.05)
        await self._safe_unsubscribe()
        if self._stop_requested:
            return

        self.means = np.median(self.calibration_samples, axis=0)
        self.raw_means = np.median(self.raw_calibration_samples, axis=0)
        self._raw_baseline = self.raw_means.astype(np.float64)

        deviation = np.abs(self.raw_calibration_samples - self._raw_baseline)
        mad = np.median(deviation, axis=0) * 1.4826
        self._deadband = np.maximum(self.noise_floor, self.deadband_sigma * mad)

        logger.info("4-point tactile calibration done.")
        logger.info(f"Raw baseline:\n{self._raw_baseline[0]}")
        logger.info(f"Deadband:\n{self._deadband[0]}")
        self._reset_filter_state()

    async def calibrate(self):
        logger.warning("Starting 4-point tactile calibration... DO NOT TOUCH SENSOR")
        await self.connect()
        await self._collect_calibration_samples()

    async def recalibrate(self) -> None:
        logger.warning("Reset requested: recalibrating 4-point tactile sensor...")
        try:
            await self._safe_unsubscribe()
        except Exception as exc:
            logger.debug(f"Could not unsubscribe before recalibration: {exc}")

        await self._collect_calibration_samples()
        if self._stop_requested:
            return
        await self._subscribe_with_retry()

    async def calibration_callback(self, handle: int, data: bytearray):
        if self.calibration_ctr >= self.config.WINDOW_SIZE:
            return
        sample, sample_raw = self.data_convert(list(data))
        self.calibration_samples[self.calibration_ctr] = sample
        self.raw_calibration_samples[self.calibration_ctr] = sample_raw
        self.calibration_ctr += 1
        self.reading_first_sample = False

    def _consume_recalibration_request(self) -> bool:
        if self._cu is None:
            return False

        data = None
        explicit_request = False
        with self._cu._lock:
            data = getattr(self._cu, "data", None)
            explicit_request = bool(
                getattr(self._cu, "tactile_recalibrate_requested", False)
            )
            if explicit_request:
                setattr(self._cu, "tactile_recalibrate_requested", False)

        reset_pressed = False
        if data is not None:
            try:
                right = data[1]
                reset_pressed = (
                    bool(right["Joystick_Press"])
                    and right["IndexTrigger"] >= self.reset_trigger_threshold
                )
            except (IndexError, KeyError, TypeError):
                reset_pressed = False

        edge_request = reset_pressed and not self._reset_held
        self._reset_held = reset_pressed
        return explicit_request or edge_request

    async def async_run(self):
        await self._subscribe_with_retry()
        if self._stop_requested:
            return
        logger.info(f"BLE tactile streaming from {self.device_mac}")

        while not self._stop_requested:
            if self.disconnected_event.is_set():
                logger.warning(f"BLE tactile disconnected from {self.device_mac}; reconnecting...")
                self._clear_tactile_output()
                self._subscribed = False
                await self._safe_disconnect()
                await self.connect()
                await self._subscribe_with_retry()
                self.disconnected_event.clear()
                self._reset_filter_state()
                self._raw_unwrapped_prev = None
                logger.info(f"BLE tactile reconnected to {self.device_mac}")

            if self._consume_recalibration_request():
                await self.recalibrate()

            await asyncio.sleep(0.05)

        await self._safe_disconnect()
        self._clear_tactile_output()
        logger.info(f"BLE tactile reader stopped for {self.device_mac}")

    def stop(self) -> None:
        self._stop_requested = True
        with contextlib.suppress(Exception):
            self.disconnected_event.set()

    def _track_unloaded_baseline(self, sample_raw: np.ndarray) -> np.ndarray:
        z = sample_raw.astype(np.float64) - self._raw_baseline
        if self.baseline_drift_alpha <= 0.0:
            return z

        taxel_norms = np.linalg.norm(z[0], axis=1)
        idle = float(np.max(taxel_norms)) < self.baseline_drift_threshold
        if idle:
            self._raw_baseline += self.baseline_drift_alpha * z
            z = sample_raw.astype(np.float64) - self._raw_baseline
        return z

    def _filter_sample(self, sample_raw: np.ndarray) -> np.ndarray:
        z = self._track_unloaded_baseline(sample_raw)
        z = np.where(np.abs(z) < self._deadband, 0.0, z)
        z = np.clip(z, -self.max_abs, self.max_abs)

        if self._last_good is not None:
            delta = np.clip(z - self._last_good, -self.max_delta, self.max_delta)
            z = self._last_good + delta

        if self._ema_state is None:
            self._ema_state = z.copy()
        else:
            self._ema_state += self.filter_alpha * (z - self._ema_state)

        if not self.use_kalman:
            filtered = np.nan_to_num(self._ema_state, copy=True)
            self._last_good = filtered.copy()
            return filtered

        if self._kalman_x is None:
            self._kalman_x = self._ema_state.copy()
            self._kalman_p = np.ones_like(self._kalman_x)
        else:
            p_pred = self._kalman_p + self.kalman_q
            gain = p_pred / (p_pred + self.kalman_r)
            self._kalman_x += gain * (self._ema_state - self._kalman_x)
            self._kalman_p = (1.0 - gain) * p_pred

        filtered = np.nan_to_num(self._kalman_x, copy=True)
        self._last_good = filtered.copy()
        return filtered

    def _publish_raw(self, sensor_idx: int, values: np.ndarray) -> None:
        for taxel_idx in range(self.config.NUM_TAXELS):
            x, y, z = values[taxel_idx]
            self.taxels_raw[sensor_idx, taxel_idx] = MagTouchTaxel(
                x=float(x),
                y=float(y),
                z=float(z),
            )
        self.magtouch_data_raw[sensor_idx].taxels = self.taxels_raw[sensor_idx]
        self.raw_data_publishers[sensor_idx].publish_sensor_data(
            self.magtouch_data_raw[sensor_idx]
        )

    async def data_callback(self, handle: int, data: bytearray):
        _, sample_raw = self.data_convert(list(data))
        filtered = self._filter_sample(sample_raw)
        snapshot = filtered[0].astype(np.float32, copy=True)

        self._publish_raw(0, snapshot)
        if self._panel_visualizer is not None:
            self._panel_visualizer.publish(snapshot)

        if self._cu is not None:
            with self._cu._lock:
                self._cu.tactile_data = snapshot
                self._cu.tactile_byte = snapshot.tobytes()
                self._cu.tactile_timestamp_ns = time.monotonic_ns()

        self.reading_first_sample = False

    def run(
        self,
        cu=None,
        start_visualizer: bool = False,
        visualizer_topic: str = "MagTouchRaw0",
    ) -> None:
        self._cu = cu
        logger.info("Starting 4-point MagTouch BLE reader loop...")
        if start_visualizer:
            self._panel_visualizer = self._start_visualizer(visualizer_topic)

        try:
            loop = self._ensure_event_loop()
            loop.run_until_complete(self.async_run())
        finally:
            with contextlib.suppress(Exception):
                loop = self._ensure_event_loop()
                if not loop.is_closed():
                    loop.run_until_complete(self._safe_disconnect())
            if self._panel_visualizer is not None:
                self._panel_visualizer.close()
                self._panel_visualizer = None

    @staticmethod
    def _start_visualizer(topic: str) -> _PanelVisualizerHandle:
        data_queue = mp.Queue(maxsize=PANEL_QUEUE_SIZE)
        process = mp.Process(
            target=_panel_visualizer_process,
            args=(data_queue, topic),
            daemon=True,
        )
        process.start()
        logger.info(f"Starting 4-point tactile panel visualizer for {topic}")
        return _PanelVisualizerHandle(process, data_queue)

    def run_test(
        self,
        start_visualizer: bool = True,
        visualizer_topic: str = "MagTouchRaw0",
    ) -> None:
        self.run(
            cu=None,
            start_visualizer=start_visualizer,
            visualizer_topic=visualizer_topic,
        )


def _device_mac_from_arg(value: str) -> DeviceMAC:
    try:
        return DeviceMAC[value]
    except KeyError as exc:
        choices = ", ".join(DeviceMAC.__members__)
        raise argparse.ArgumentTypeError(
            f"Unknown device MAC enum '{value}'. Choices: {choices}"
        ) from exc


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(
        description="Read one 4-taxel MagTouch BLE sensor into a filtered (4, 3) array."
    )
    parser.add_argument("--device-mac", type=_device_mac_from_arg, default=DeviceMAC.ARDUINO7)
    parser.add_argument("--hci", default="hci0")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--use-kalman", action="store_true")
    parser.add_argument("--kalman-q", type=float, default=2e-2)
    parser.add_argument("--kalman-r", type=float, default=2e-2)
    parser.add_argument("--max-delta", type=float, default=10000.0)
    parser.add_argument("--baseline-drift-alpha", type=float, default=0.0)
    parser.add_argument("--baseline-drift-threshold", type=float, default=80.0)
    parser.add_argument("--no-visualizer", action="store_true")
    parser.add_argument("--visualizer-topic", default="MagTouchRaw0")
    args = parser.parse_args()

    reader = FourPointTactileBleReader(
        config=MagTouchBleReaderConfig(
            ENABLE_WS=False,
            NUM_SENSORS=1,
            NUM_TAXELS=4,
            MODEL_NAMES=np.array([None]),
            WINDOW_SIZE=args.window_size,
            uuid=SensorUuid.DATA_CHAR_MAGTOUCH,
            device_mac=args.device_mac,
            hci=args.hci,
        ),
        filter_alpha=args.alpha,
        use_kalman=args.use_kalman,
        kalman_q=args.kalman_q,
        kalman_r=args.kalman_r,
        max_delta=args.max_delta,
        baseline_drift_alpha=args.baseline_drift_alpha,
        baseline_drift_threshold=args.baseline_drift_threshold,
    )
    reader.run_test(
        start_visualizer=not args.no_visualizer,
        visualizer_topic=args.visualizer_topic,
    )
