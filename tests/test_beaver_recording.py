from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from beaver import (
    FRAME_HEADER,
    FrameDecoder,
    SENSOR_PREFIX,
    BeaverReader,
    empty_snapshot,
    parse_frame,
)
from data_recording import DataRecordingService, RecordingControl, RecordingFrame
from config import Config
from dataset import DatasetRecorder


LAYOUT = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 1), (1, 2), (1, 3),
)


def encoded_frame(sequence: int = 7) -> bytes:
    records = []
    for slot, (bus, index) in enumerate(LAYOUT):
        distance_units = bytes([slot + 1] * 16)
        statuses = bytes([5] * 16)
        records.append(
            SENSOR_PREFIX.pack(bus, index, 16, (slot + 1) * 10, 22, slot)
            + distance_units
            + statuses
        )
    return FRAME_HEADER.pack(0x5A5A, sequence, len(records), 4) + b"".join(records)


class BeaverProtocolTests(unittest.TestCase):
    def test_legacy_zero_flags_decode_as_8x8(self) -> None:
        record = (
            SENSOR_PREFIX.pack(0, 0, 64, 100, 22, 1)
            + bytes([10] * 64)
            + bytes([5] * 64)
        )
        raw = FRAME_HEADER.pack(0x5A5A, 3, 1, 0) + record
        frame = parse_frame(raw)
        self.assertEqual(frame["grid_width"], 8)
        self.assertEqual(frame["sensors"][0]["distance_mm"].shape, (8, 8))

    def test_decoder_tolerates_boot_text_and_partial_reads(self) -> None:
        decoder = FrameDecoder()
        raw = b"ESP booting\r\n" + encoded_frame()
        frames = []
        for end in range(0, len(raw), 31):
            frames.extend(decoder.feed(raw[end : end + 31]))
        frames.extend(decoder.feed(raw[(len(raw) // 31) * 31 :]))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["sequence"], 7)
        self.assertEqual(len(frames[0]["sensors"]), 9)
        self.assertIn("ESP booting", decoder.take_messages())

    def test_reader_maps_physical_ids_to_nine_stable_slots(self) -> None:
        decoder = FrameDecoder()
        frame = decoder.feed(encoded_frame(11))[0]
        reader = BeaverReader(sensor_layout=LAYOUT)
        reader._publish_frame(frame, frame_count=3, lost_frames=1)
        snapshot = reader.snapshot()
        self.assertEqual(snapshot.distance_mm.shape, (9, 4, 4))
        self.assertTrue(np.all(snapshot.present == 1))
        self.assertEqual(int(snapshot.distance_mm[5, 0, 0]), 60)
        self.assertEqual(snapshot.sequence, 11)
        self.assertEqual(snapshot.lost_frames, 1)

    def test_reader_selects_buffered_snapshot_nearest_to_reference(self) -> None:
        decoder = FrameDecoder()
        first = decoder.feed(encoded_frame(11))[0]
        second = decoder.feed(encoded_frame(12))[0]
        reader = BeaverReader(sensor_layout=LAYOUT, sync_buffer_size=2)
        with patch("beaver.time.monotonic_ns", side_effect=[100, 300]):
            reader._publish_frame(first, frame_count=1, lost_frames=0)
            reader._publish_frame(second, frame_count=2, lost_frames=0)

        self.assertEqual(reader.snapshot_nearest(140).sequence, 11)
        self.assertEqual(reader.snapshot_nearest(260).sequence, 12)

    def test_reader_treats_none_in_waiting_as_empty_buffer(self) -> None:
        stop_event = threading.Event()

        class FakeSerial:
            in_waiting = None

            def __init__(self) -> None:
                self.read_sizes = []
                self.closed = False

            def read(self, size):
                self.read_sizes.append(size)
                stop_event.set()
                return b""

            def close(self):
                self.closed = True

        serial_port = FakeSerial()
        reader = BeaverReader(device="/dev/fake", sensor_layout=LAYOUT)
        with patch("beaver.open_port", return_value=serial_port):
            reader.run(stop_event)

        self.assertEqual(serial_port.read_sizes, [1])
        self.assertTrue(serial_port.closed)


class FakeDataset:
    def __init__(self) -> None:
        self.recorded_episodes = 0
        self.collect_step = 0
        self.frames = []
        self.exports = 0
        self.rollbacks = 0
        self.closed = False

    def data_collection(self, **frame) -> None:
        self.frames.append(frame)
        self.collect_step += 1

    def data_export(self, context) -> None:
        self.exports += 1
        self.recorded_episodes += 1

    def _reset_data_dict(self) -> None:
        self.collect_step = 0

    def rollback_last_episode(self) -> bool:
        self.rollbacks += 1
        return True

    def recording_status(self, collecting=False):
        return {"collecting": collecting, "current_episode_frames": self.collect_step}

    def close(self) -> None:
        self.closed = True


class DataRecordingServiceTests(unittest.TestCase):
    def test_collection_and_export_are_owned_by_service(self) -> None:
        dataset = FakeDataset()
        control = RecordingControl()
        beaver = empty_snapshot(LAYOUT)
        frame = RecordingFrame(
            state=np.zeros(7),
            action=np.ones(7),
            camera_images={},
            beaver_data=beaver,
        )
        service = DataRecordingService(
            dataset,
            lambda: frame,
            10,
            control=control,
        )
        self.assertTrue(service.start_recording())
        self.assertTrue(service.collect_once())
        self.assertIs(dataset.frames[0]["beaver_data"], beaver)
        self.assertTrue(service.stop_recording())
        self.assertTrue(service.process_pending_once())
        self.assertEqual(dataset.exports, 1)
        self.assertEqual(dataset.collect_step, 0)
        service.close()
        self.assertTrue(dataset.closed)

    def test_dataset_beaver_formatter_uses_fixed_shape_and_missing_defaults(self) -> None:
        recorder = object.__new__(DatasetRecorder)
        recorder.beaver_shape = (9, 4, 4)
        distance, status, present = recorder._format_beaver(None)
        self.assertEqual(distance.shape, (9, 4, 4))
        self.assertTrue(np.all(status == 255))
        self.assertTrue(np.all(present == 0))

    def test_hdf5_export_contains_nine_sensor_beaver_schema(self) -> None:
        class CameraContext:
            camera_images = {}

        with tempfile.TemporaryDirectory() as directory:
            cfg = Config(
                DATASET_TYPE="a",
                DATA_TYPE="qpos",
                DATASET_DIR=str(Path(directory) / "beaver"),
                TCP_TOOL="None",
                beaver_enable=True,
            )
            recorder = DatasetRecorder(
                camera_num=0,
                robot_dof=7,
                robot_type="realman",
                gripper=False,
                config=cfg,
            )
            snapshot = empty_snapshot(LAYOUT)
            recorder.data_collection(
                state=np.zeros(7),
                action=np.ones(7),
                camera_images={},
                extra_data={"beaver_timestamp_ns": np.int64(123)},
                beaver_data=snapshot,
            )
            recorder.data_export(CameraContext())
            recorder._reset_data_dict()
            recorder.close()

            path = Path(str(cfg.DATASET_DIR) + "_hdf5") / "episode_0.hdf5"
            with h5py.File(path, "r") as root:
                self.assertEqual(
                    root["/observations/beaver/distance_mm"].shape,
                    (1, 9, 4, 4),
                )
                self.assertEqual(
                    root["/observations/beaver"].attrs["grid_width"], 4
                )
                names = [
                    value.decode()
                    for value in root["/extra/timestamps_ns"].attrs["names"]
                ]
                self.assertIn("beaver", names)


if __name__ == "__main__":
    unittest.main()
