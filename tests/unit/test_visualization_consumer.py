"""Typed visualization model and consumer tests."""

from __future__ import annotations

import time
import unittest

from airo_doffy.core import (
    ClockDomain,
    PixelFormat,
    ProcessedFrame,
    RobotState,
    TactileSample,
    WrenchSample,
)
from airo_doffy.core.errors import ModelValidationError
from airo_doffy.visualization import (
    MemorySnapshotRenderer,
    RecordingView,
    TypedSnapshotConsumer,
    VisualizationSnapshot,
)


def _snapshot(sequence: int, *, with_sensors: bool = True) -> VisualizationSnapshot:
    if not with_sensors:
        return VisualizationSnapshot(
            sequence=sequence,
            source_timestamp_ns=sequence + 1,
        )
    robot = RobotState(
        sequence=sequence,
        source_timestamp_ns=sequence + 1,
        joints_rad=(0.0,) * 6,
        tcp_pose=(
            (1.0, 0.0, 0.0, 0.1),
            (0.0, 1.0, 0.0, 0.2),
            (0.0, 0.0, 1.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    frame = ProcessedFrame(
        sequence=sequence,
        source_timestamp_ns=sequence + 1,
        processing_timestamp_ns=sequence + 2,
        stream_id="camera_0",
        data=bytes(range(6)),
        shape=(1, 2, 3),
        pixel_format=PixelFormat.RGB8,
    )
    tactile = TactileSample(
        sequence=sequence,
        source_timestamp_ns=sequence + 1,
        values=((0.0, 0.0, 0.0),) * 4,
    )
    wrench = WrenchSample(
        sequence=sequence,
        source_timestamp_ns=sequence + 1,
        values=(0.0,) * 6,
    )
    recording = RecordingView(
        dataset_type="l",
        dataset_dir="./datasets/mock",
        recorded_episodes=2,
        current_episode_frames=3,
        last_episode_length=5,
        collecting=True,
        pending_exports=1,
    )
    return VisualizationSnapshot(
        sequence=sequence,
        source_timestamp_ns=sequence + 1,
        robot=robot,
        frames=(frame,),
        tactile=tactile,
        wrench=wrench,
        recording=recording,
        clock_domain=ClockDomain.MONOTONIC,
    )


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FailingRenderer:
    def __init__(self) -> None:
        self.closed = False

    def start(self) -> None:
        pass

    def render(self, snapshot: VisualizationSnapshot) -> bool:
        del snapshot
        raise RuntimeError("window failed")

    def close(self) -> None:
        self.closed = True


class VisualizationSnapshotTest(unittest.TestCase):
    def test_all_sensor_fields_are_optional_for_mock_mode(self) -> None:
        snapshot = _snapshot(0, with_sensors=False)
        self.assertIsNone(snapshot.robot)
        self.assertEqual(snapshot.frames, ())
        self.assertIsNone(snapshot.tactile)
        self.assertIsNone(snapshot.wrench)
        self.assertIsNone(snapshot.recording)

    def test_complete_snapshot_is_frozen_and_typed(self) -> None:
        snapshot = _snapshot(2)
        self.assertEqual(snapshot.robot.joints_rad, (0.0,) * 6)
        self.assertEqual(snapshot.frames[0].stream_id, "camera_0")
        self.assertEqual(snapshot.recording.pending_exports, 1)

    def test_duplicate_camera_streams_and_invalid_recording_counts_fail(self) -> None:
        snapshot = _snapshot(1)
        with self.assertRaises(ModelValidationError):
            VisualizationSnapshot(
                sequence=1,
                source_timestamp_ns=1,
                frames=(snapshot.frames[0], snapshot.frames[0]),
            )
        with self.assertRaises(ModelValidationError):
            RecordingView(
                dataset_type="l",
                dataset_dir="mock",
                recorded_episodes=-1,
                current_episode_frames=0,
            )


class TypedSnapshotConsumerTest(unittest.TestCase):
    def test_consumer_renders_typed_snapshots(self) -> None:
        renderer = MemorySnapshotRenderer()
        consumer = TypedSnapshotConsumer(renderer)
        consumer.start()
        self.assertTrue(consumer.publish(_snapshot(0)))
        self.assertTrue(_wait_until(lambda: len(renderer.snapshots) == 1))
        consumer.close()
        self.assertTrue(renderer.closed)

    def test_closed_renderer_does_not_raise_in_publisher(self) -> None:
        renderer = MemorySnapshotRenderer(close_after=1)
        consumer = TypedSnapshotConsumer(renderer)
        consumer.start()
        self.assertTrue(consumer.publish(_snapshot(0, with_sensors=False)))
        self.assertTrue(_wait_until(lambda: consumer.metrics().renderer_closed))
        self.assertFalse(consumer.publish(_snapshot(1, with_sensors=False)))
        consumer.close()

    def test_latest_only_rejects_stale_and_duplicate_sequences(self) -> None:
        renderer = MemorySnapshotRenderer()
        consumer = TypedSnapshotConsumer(renderer)
        consumer.start()
        self.assertTrue(consumer.publish(_snapshot(2, with_sensors=False)))
        self.assertFalse(consumer.publish(_snapshot(2, with_sensors=False)))
        self.assertFalse(consumer.publish(_snapshot(1, with_sensors=False)))
        self.assertTrue(_wait_until(lambda: len(renderer.snapshots) == 1))
        metrics = consumer.metrics()
        self.assertEqual(metrics.published, 1)
        self.assertEqual(metrics.rejected, 2)
        consumer.close()

    def test_disabled_sensors_and_repeated_close_are_supported(self) -> None:
        renderer = MemorySnapshotRenderer()
        consumer = TypedSnapshotConsumer(renderer)
        consumer.start()
        self.assertTrue(consumer.publish(_snapshot(0, with_sensors=False)))
        self.assertTrue(_wait_until(lambda: len(renderer.snapshots) == 1))
        consumer.close()
        consumer.close()
        self.assertFalse(consumer.publish(_snapshot(1, with_sensors=False)))

    def test_renderer_failure_is_contained_from_runtime_publisher(self) -> None:
        renderer = FailingRenderer()
        consumer = TypedSnapshotConsumer(renderer)
        consumer.start()
        self.assertTrue(consumer.publish(_snapshot(0, with_sensors=False)))
        self.assertTrue(_wait_until(lambda: consumer.metrics().last_error is not None))
        self.assertIn("window failed", consumer.metrics().last_error)
        self.assertFalse(consumer.publish(_snapshot(1, with_sensors=False)))
        consumer.close()
        self.assertTrue(renderer.closed)


if __name__ == "__main__":
    unittest.main()
