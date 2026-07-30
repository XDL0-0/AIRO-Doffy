"""Hardware-free tests for the private-state MagTouch BLE4 sensor."""

from __future__ import annotations

import threading
import unittest

from airo_doffy.config import TactileConfig, TactileFactory
from airo_doffy.core import LifecycleError
from airo_doffy.devices.tactile import (
    Ble4RawSource,
    MagtouchBle4Sensor,
    SensorCommBle4Source,
    TactileSensor,
    decode_ble4_packet,
    unwrap_ble4_sample,
)


def uniform(value: float):
    return tuple((value, value, value) for _ in range(4))


class _Source:
    def __init__(self) -> None:
        self.on_sample = None
        self.on_disconnect = None
        self.started = False
        self.close_count = 0

    def start(self, on_sample, on_disconnect) -> None:
        self.on_sample = on_sample
        self.on_disconnect = on_disconnect
        self.started = True

    def push(self, values) -> None:
        self.on_sample(values)

    def disconnect(self) -> None:
        self.on_disconnect()

    def close(self) -> None:
        self.close_count += 1


class _Connection:
    is_connected = False


class _FlakyReader:
    def __init__(self) -> None:
        self.connection = _Connection()
        self.disconnected_event = threading.Event()
        self.connect_count = 0

    async def connect(self) -> None:
        self.connect_count += 1
        if self.connect_count == 1:
            raise OSError("temporary BLE failure")
        self.connection.is_connected = True

    async def subscribe(self, callback) -> None:
        self.callback = callback

    async def unsubscribe(self) -> None:
        pass

    async def disconnect(self) -> None:
        self.connection.is_connected = False

    async def data_callback(self, _handle, _data) -> None:
        pass


class _FlakySensorCommSource(SensorCommBle4Source):
    def __init__(self, config: TactileConfig) -> None:
        super().__init__(
            config,
            reconnect_delay_s=0.001,
            startup_timeout_s=0.5,
        )
        self.fake_reader = _FlakyReader()

    def _build_reader(self):
        return self.fake_reader


class MagtouchBle4SensorTest(unittest.TestCase):
    def test_calibration_private_latest_recalibration_and_disconnect(self) -> None:
        source = _Source()
        config = TactileConfig(
            ble_window_size=3,
            filter_alpha=1.0,
            noise_floor=0.1,
        )
        sensor = MagtouchBle4Sensor(config, source=source)
        self.assertIsInstance(source, Ble4RawSource)
        self.assertIsInstance(sensor, TactileSensor)
        sensor.start()
        for value in (10, 10, 10):
            source.push(uniform(value))
        self.assertTrue(sensor.calibrated)
        self.assertIsNone(sensor.read_latest())
        source.push(uniform(12))
        first = sensor.read_latest()
        self.assertEqual(first.sequence, 0)
        self.assertEqual(first.values, uniform(2))

        source.disconnect()
        self.assertEqual(sensor.disconnect_count, 1)
        self.assertIsNone(sensor.read_latest())
        source.push(uniform(13))
        self.assertEqual(sensor.read_latest().sequence, 1)

        sensor.recalibrate()
        self.assertIsNone(sensor.read_latest())
        for value in (20, 20, 20):
            source.push(uniform(value))
        source.push(uniform(21))
        self.assertEqual(sensor.read_latest().values, uniform(1))
        sensor.close()
        sensor.close()
        self.assertEqual(source.close_count, 1)
        with self.assertRaises(LifecycleError):
            sensor.read_latest()

    def test_packet_unwrap_selects_nearest_16_bit_candidate(self) -> None:
        wrapped = tuple((-65530.0, -10.0, -20.0) for _ in range(4))
        previous = tuple((5.0, -8.0, -18.0) for _ in range(4))
        unwrapped = unwrap_ble4_sample(wrapped, previous)
        self.assertEqual(unwrapped[0], (6.0, -10.0, -20.0))

    def test_packet_decode_preserves_legacy_complement_and_shape(self) -> None:
        packet = bytes(range(24))
        decoded = decode_ble4_packet(packet)
        self.assertEqual(len(decoded), 4)
        self.assertEqual(decoded[0], (-2.0, -516.0, -1030.0))
        self.assertEqual(decoded[3], (-4628.0, -5142.0, -5656.0))
        with self.assertRaises(ValueError):
            decode_ble4_packet(packet[:23])

    def test_typed_factory_target_constructs_without_hardware(self) -> None:
        factory = TactileFactory(
            target=(
                "airo_doffy.devices.tactile.magtouch_ble4:"
                "create_magtouch_ble4"
            )
        )
        sensor = factory.create(TactileConfig())
        self.assertIsInstance(sensor, MagtouchBle4Sensor)
        sensor.close()

    def test_sensor_comm_source_retries_initial_connection(self) -> None:
        source = _FlakySensorCommSource(TactileConfig())
        disconnects = []
        source.start(lambda _sample: None, lambda: disconnects.append(True))
        self.assertEqual(source.fake_reader.connect_count, 2)
        self.assertEqual(disconnects, [])
        source.close()
        source.close()


if __name__ == "__main__":
    unittest.main()
