"""Raw BLE4 sample-source contract and lazy sensor-comm implementation."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ...config.models import TactileConfig
from ...core.errors import LifecycleError, OptionalDependencyError
from .filters import TaxelValues

RawSampleCallback = Callable[[TaxelValues], None]
DisconnectCallback = Callable[[], None]


@runtime_checkable
class Ble4RawSource(Protocol):
    """Source of uncalibrated `(4, 3)` magnetic samples."""

    def start(
        self,
        on_sample: RawSampleCallback,
        on_disconnect: DisconnectCallback,
    ) -> None:
        """Start acquisition and register private sensor callbacks."""

    def close(self) -> None:
        """Stop acquisition and release BLE resources idempotently."""


def decode_ble4_packet(
    data: bytes,
    previous: TaxelValues | None = None,
) -> TaxelValues:
    """Decode one legacy BLE4 packet and unwrap it against the prior sample."""

    if len(data) < 24:
        raise ValueError(f"BLE4 packet must contain at least 24 bytes, got {len(data)}")
    rows = []
    for taxel in range(4):
        base = taxel * 6
        axes = []
        for axis in range(3):
            word = (data[base + axis * 2] << 8) + data[base + axis * 2 + 1]
            axes.append(float(~word))
        rows.append(tuple(axes))
    return unwrap_ble4_sample(tuple(rows), previous)


def unwrap_ble4_sample(
    wrapped: TaxelValues,
    previous: TaxelValues | None,
) -> TaxelValues:
    """Choose the nearest value across the legacy 16-bit wrap boundary."""

    if previous is None:
        return wrapped
    return tuple(
        tuple(
            min(
                (
                    wrapped[taxel][axis] - 65536.0,
                    wrapped[taxel][axis],
                    wrapped[taxel][axis] + 65536.0,
                ),
                key=lambda candidate: abs(candidate - previous[taxel][axis]),
            )
            for axis in range(3)
        )
        for taxel in range(4)
    )


class SensorCommBle4Source:
    """Own the sensor-comm event loop and expose only raw samples."""

    def __init__(
        self,
        config: TactileConfig,
        *,
        reconnect_delay_s: float = 1.0,
        startup_timeout_s: float = 5.0,
    ) -> None:
        self._config = config
        self._reconnect_delay_s = float(reconnect_delay_s)
        self._startup_timeout_s = float(startup_timeout_s)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader = None
        self._error: Exception | None = None
        self._closed = False
        self._previous: TaxelValues | None = None
        self._on_sample: RawSampleCallback | None = None
        self._on_disconnect: DisconnectCallback | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _dependencies():
        try:
            import numpy as np
            from sensor_comm_dds.communication.config.ble_config import (
                DeviceMAC,
                SensorUuid,
            )
            from sensor_comm_dds.communication.readers.magtouch_ble_reader import (
                MagTouchBleReader,
                MagTouchBleReaderConfig,
            )
        except ImportError as exc:
            raise OptionalDependencyError(
                "BLE4 tactile requires the 'tactile-ble4' optional dependency"
            ) from exc
        return np, DeviceMAC, SensorUuid, MagTouchBleReader, MagTouchBleReaderConfig

    def _build_reader(self):
        np, DeviceMAC, SensorUuid, reader_base, config_type = self._dependencies()
        owner = self

        class CallbackReader(reader_base):
            async def data_callback(self, _handle: int, data: bytearray):
                sample = decode_ble4_packet(bytes(data), owner._previous)
                owner._previous = sample
                callback = owner._on_sample
                if callback is not None:
                    callback(sample)
                self.reading_first_sample = False

        reader_config = config_type(
            ENABLE_WS=False,
            NUM_SENSORS=1,
            NUM_TAXELS=4,
            MODEL_NAMES=np.array([None]),
            WINDOW_SIZE=self._config.ble_window_size,
            uuid=SensorUuid.DATA_CHAR_MAGTOUCH,
            device_mac=DeviceMAC[self._config.ble_device_mac],
            hci=self._config.ble_hci,
        )
        return CallbackReader(config=reader_config)

    async def _disconnect(self) -> None:
        reader = self._reader
        if reader is None:
            return
        with contextlib.suppress(Exception):
            await reader.unsubscribe()
        if getattr(reader.connection, "is_connected", False):
            with contextlib.suppress(Exception):
                await reader.disconnect()

    async def _run_async(self) -> None:
        reader = self._reader
        assert reader is not None
        first_connection = True
        while not self._stop_event.is_set():
            try:
                if not reader.connection.is_connected:
                    await reader.connect()
                await reader.subscribe(callback=reader.data_callback)
                reader.disconnected_event.clear()
                if first_connection:
                    first_connection = False
                    self._ready_event.set()
                while not self._stop_event.is_set():
                    if reader.disconnected_event.is_set():
                        break
                    await asyncio.sleep(0.05)
            except Exception:
                if self._stop_event.is_set():
                    break
            if self._stop_event.is_set():
                break
            if not first_connection:
                callback = self._on_disconnect
                if callback is not None:
                    callback()
            self._previous = None
            await self._disconnect()
            await asyncio.sleep(self._reconnect_delay_s)
        await self._disconnect()

    def _worker(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._reader = self._build_reader()
            loop.run_until_complete(self._run_async())
        except Exception as exc:
            self._error = exc
            self._ready_event.set()
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(self._disconnect())
            loop.close()
            self._loop = None

    def start(
        self,
        on_sample: RawSampleCallback,
        on_disconnect: DisconnectCallback,
    ) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed BLE4 source")
            if self._thread is not None:
                raise LifecycleError("BLE4 source is already started")
            self._on_sample = on_sample
            self._on_disconnect = on_disconnect
            self._thread = threading.Thread(
                target=self._worker,
                name="airo-doffy-ble4",
                daemon=True,
            )
            self._thread.start()
        if not self._ready_event.wait(self._startup_timeout_s):
            self.close()
            raise LifecycleError("BLE4 source startup timed out")
        if self._error is not None:
            error = self._error
            self.close()
            raise error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            thread = self._thread
            self._stop_event.set()
            loop = self._loop
            if loop is not None:
                loop.call_soon_threadsafe(lambda: None)
        if thread is not None:
            thread.join(timeout=max(1.0, self._reconnect_delay_s + 0.5))
            if thread.is_alive():
                raise LifecycleError("BLE4 worker did not stop; source was not invalidated")
        with self._lock:
            self._closed = True
            self._thread = None
            self._reader = None
            self._on_sample = None
            self._on_disconnect = None
