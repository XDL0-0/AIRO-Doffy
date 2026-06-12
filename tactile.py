"""MagTouch tactile sensor serial reader + Kalman filter."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import serial
from loguru import logger

from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from sensor_comm_dds.communication.config.serial_config import SerialConfig
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
from sensor_comm_dds.communication.data_classes.magtouchcls import MagTouchCls
from sensor_comm_dds.communication.data_classes.magtouch_taxel import MagTouchTaxel


@dataclass
class MagtouchIliasSerialReaderConfig(SerialConfig, WebsocketConfig):
    NUM_X: int = 4
    NUM_Y: int = 8
    NUM_FRONT: int = NUM_X * NUM_Y
    NUM_SIDE: int = 3
    NUM_TOP: int = 3
    BYTES_PER_PACKET: int = 11
    CALIBRATION_SAMPLES: int = 320


# Sensor groups for uniform handling
_GROUPS = [
    ("front", 32, 0),    # (name_prefix, num_sensors, index_offset)
    ("side_r", 3, 32),
    ("side_l", 3, 35),
    ("top", 3, 38),
]


class MagtouchIliasSerialReader:
    """Serial reader for the MagTouch ILIAS tactile sensor.

    The taxels are read in the following order (taxels facing out of screen):

            TIP END

          38  40  41

    35    28 29 30 31    32
          24 25 26 27
    36    20 21 22 23    33
          16 17 18 19
    37    12 13 14 15    34
           8  9 10 11
           4  5  6  7
           0  1  2  3

           GRIPPER END
    """

    def __init__(
        self,
        config: MagtouchIliasSerialReaderConfig,
        use_kalman: bool = True,
        kalman_q: float = 1e-3,
        kalman_r: float = 1e-2,
    ):
        self.config = config
        self.ser = serial.Serial(config.COM, config.BAUD, timeout=None)

        self._num_sensors = {
            "front": config.NUM_FRONT,
            "side_r": config.NUM_SIDE,
            "side_l": config.NUM_SIDE,
            "top": config.NUM_TOP,
        }

        self._xyz: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._baselines: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, n in self._num_sensors.items():
            self._xyz[name] = (np.zeros(n), np.zeros(n), np.zeros(n))
            self._baselines[name] = (np.zeros(n), np.zeros(n), np.zeros(n))

        self.use_kalman = use_kalman
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.kalman_x: np.ndarray | None = None
        self.kalman_P: np.ndarray | None = None

        self.publisher = DataPublisher(
            topic_name="MagTouch", topic_data_type=MagTouchCls
        )
        self._measure_baseline()

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
        self.monitor_logger = logger.bind(refresh=True)

    # ── Packet reading ────────────────────────────────────────────────────

    def _read_packet(self) -> tuple[int, int] | None:
        b = self.ser.read(1)
        if not b or b[0] != self.config.START_BYTE:
            return None

        packet = self.ser.read(self.config.BYTES_PER_PACKET - 1)
        if len(packet) != (self.config.BYTES_PER_PACKET - 1):
            logger.warning("Incomplete packet")
            return None
        if packet[-1] != self.config.END_BYTE:
            logger.warning(f"Invalid end byte: {packet}")
            return None

        index = packet[0]
        x = int.from_bytes(packet[1:3], "little", signed=True)
        y = int.from_bytes(packet[3:5], "little", signed=True)
        z = int.from_bytes(packet[5:7], "little", signed=True)
        mic = packet[8]

        if index < self.config.NUM_FRONT:
            grp = "front"
            i = index
            if index == 9:
                z = -z
        elif 32 <= index < 35:
            grp, i = "side_r", index - 32
        elif 35 <= index < 38:
            grp, i = "side_l", index - 35
        elif 38 <= index < 41:
            grp, i = "top", index - 38
        else:
            return index, mic

        ax, ay, az = self._xyz[grp]
        ax[i], ay[i], az[i] = x, y, z

        return index, mic

    # ── Baseline calibration ──────────────────────────────────────────────

    def _measure_baseline(self) -> None:
        logger.info("Measuring sensor baselines...")

        accum: dict[str, tuple[list, list, list]] = {}
        for name, n in self._num_sensors.items():
            accum[name] = (
                [[] for _ in range(n)],
                [[] for _ in range(n)],
                [[] for _ in range(n)],
            )

        ctr = 0
        while ctr < self.config.CALIBRATION_SAMPLES:
            pkt = self._read_packet()
            if not pkt:
                continue
            ctr += 1
            index, _ = pkt

            for name, num_base, idx_offset in _GROUPS:
                if idx_offset <= index < idx_offset + num_base:
                    i = index - idx_offset
                    ax, ay, az = self._xyz[name]
                    bx, by, bz = accum[name]
                    bx[i].append(ax[i])
                    by[i].append(ay[i])
                    bz[i].append(az[i])
                    break

        for name in self._num_sensors:
            bx, by, bz = accum[name]
            self._baselines[name] = (
                np.array([np.median(v) if v else 0 for v in bx]),
                np.array([np.median(v) if v else 0 for v in by]),
                np.array([np.median(v) if v else 0 for v in bz]),
            )

        logger.info("Baseline measurement complete.")

    # ── Data retrieval ────────────────────────────────────────────────────

    @staticmethod
    def _diff_group(
        xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
        baseline: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        dx = (xyz[0] - baseline[0]).astype(np.int32)
        dy = (xyz[1] - baseline[1]).astype(np.int32)
        dz = (xyz[2] - baseline[2]).astype(np.int32)
        return np.stack((dx, dy, dz), axis=1)

    def get_data(self) -> np.ndarray:
        """Baseline-subtracted sensor data, shape (N_total, 3)."""
        parts = [
            self._diff_group(self._xyz[name], self._baselines[name])
            for name in ("front", "side_r", "side_l", "top")
        ]
        return np.concatenate(parts, axis=0)

    def _publish_data(self) -> None:
        taxels = []
        for name in ("front", "side_r", "side_l", "top"):
            diff = self._diff_group(self._xyz[name], self._baselines[name])
            for row in diff:
                taxels.append(MagTouchTaxel(x=int(row[0]), y=int(row[1]), z=int(row[2])))

        data = MagTouchCls(np.array(taxels).flatten())
        self.publisher.publish_sensor_data(data)

    # ── Kalman filter ─────────────────────────────────────────────────────

    def _kalman_update(self, z: np.ndarray) -> np.ndarray:
        if self.kalman_x is None:
            self.kalman_x = z.astype(np.float64).copy()
            self.kalman_P = np.ones_like(self.kalman_x)
            return self.kalman_x

        P_pred = self.kalman_P + self.kalman_q
        K = P_pred / (P_pred + self.kalman_r)
        self.kalman_x = self.kalman_x + K * (z - self.kalman_x)
        self.kalman_P = (1.0 - K) * P_pred
        return self.kalman_x

    # ── Main loops ────────────────────────────────────────────────────────

    def run(self, cu) -> None:
        """Main acquisition loop. Writes filtered data to *cu* object."""
        logger.info("Starting MagTouch reader loop...")
        t_start = time.time()
        ctr = 0

        while True:
            pkt = self._read_packet()
            if not pkt:
                continue

            new_data = self.get_data()
            filtered = self._kalman_update(new_data) if self.use_kalman else new_data
            # .copy() prevents race: Kalman mutates self.kalman_x in-place
            snapshot = filtered.copy()
            with cu._lock:
                cu.tactile_data = snapshot
                cu.tactile_byte = snapshot.astype(np.int32).tobytes()
                cu.tactile_timestamp_ns = time.monotonic_ns()

            ctr += 1
            if ctr == 300:
                hz = ctr / (time.time() - t_start)
                self.monitor_logger.info(f"MagTouch running at {hz:5.2f} Hz")
                ctr = 0
                t_start = time.time()

            joystick_press = False
            with cu._lock:
                data = cu.data
            if data is not None:
                try:
                    joystick_press = bool(data[1]["Joystick_Press"])
                except (IndexError, KeyError, TypeError):
                    pass

            if joystick_press:
                time.sleep(1)
                self._measure_baseline()

    def run_test(self) -> None:
        """Standalone test loop (no cu object)."""
        logger.info("Starting MagTouch test loop...")
        t_start = time.time()
        ctr = 0

        while True:
            pkt = self._read_packet()
            if not pkt:
                continue

            new_data = self.get_data()
            filtered = self._kalman_update(new_data) if self.use_kalman else new_data
            logger.info(f"Filtered tactile preview:\n{filtered[0:3, :]}")

            ctr += 1
            if ctr == 300:
                logger.info(f"Loop frequency: {ctr / (time.time() - t_start):.1f} Hz")
                ctr = 0
                t_start = time.time()


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(
        description="Read data from MagTouch ILIAS sensor over serial."
    )
    parser.add_argument(
        "--port",
        default="/dev/serial/by-id/usb-Arduino_IO_Coupling_C6E76762B4D1E02A-if00",
    )
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    reader = MagtouchIliasSerialReader(
        config=MagtouchIliasSerialReaderConfig(
            ENABLE_WS=False,
            COM=args.port,
            START_BYTE=0xAA,
            END_BYTE=0xCC,
        )
    )
    reader.run_test()
