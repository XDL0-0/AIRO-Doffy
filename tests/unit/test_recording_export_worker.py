"""Bounded non-blocking recording export worker tests."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from threading import Event

from airo_doffy.core.errors import LifecycleError
from airo_doffy.recording import (
    Episode,
    ExportQueueFullError,
    ExportTaskKind,
    ExportWorker,
    RecordingSample,
    build_recording_schema,
)


def _episode(index: int) -> Episode:
    schema = build_recording_schema(
        data_type="qpos",
        robot_dof=6,
        camera_count=0,
        resolution=(2, 1),
    )
    return Episode(
        index=index,
        task="pick",
        schema=schema,
        samples=(
            RecordingSample(
                state=(0.0,) * 7,
                action=(0.0,) * 7,
                timestamps_ns=(1,) * 5,
            ),
        ),
    )


class BlockingWriter:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.indices: list[int] = []
        self.closed = False

    def write(self, episode: Episode) -> Path:
        self.started.set()
        if not self.release.wait(2.0):
            raise TimeoutError("test writer was not released")
        self.indices.append(episode.index)
        return Path(f"episode_{episode.index}.data")

    def close(self) -> None:
        self.closed = True


class FailingWriter:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def write(self, episode: Episode) -> Path:
        self.calls += 1
        if self.calls == 1:
            raise OSError("disk full")
        return Path(f"episode_{episode.index}.data")

    def close(self) -> None:
        self.closed = True


class FakeRollback:
    def __init__(self) -> None:
        self.indices: list[int] = []

    def rollback(self, episode_index: int) -> bool:
        self.indices.append(episode_index)
        return True


class ExportWorkerTest(unittest.TestCase):
    def test_submit_is_non_blocking_and_queue_full_is_explicit(self) -> None:
        writer = BlockingWriter()
        worker = ExportWorker(writer, capacity=1)
        worker.start()
        started_at = time.monotonic()
        first = worker.submit_export(_episode(0))
        elapsed = time.monotonic() - started_at
        self.assertLess(elapsed, 0.1)
        self.assertTrue(writer.started.wait(1.0))

        second = worker.submit_export(_episode(1))
        with self.assertRaises(ExportQueueFullError):
            worker.submit_export(_episode(2))
        self.assertFalse(first.done)
        self.assertFalse(second.done)

        writer.release.set()
        self.assertTrue(first.result(1.0).succeeded)
        self.assertTrue(second.result(1.0).succeeded)
        metrics = worker.metrics()
        self.assertEqual(metrics.submitted, 2)
        self.assertEqual(metrics.completed, 2)
        self.assertEqual(metrics.rejected, 1)
        worker.close()
        self.assertTrue(writer.closed)

    def test_failure_is_reported_and_worker_continues(self) -> None:
        writer = FailingWriter()
        worker = ExportWorker(writer, capacity=2)
        worker.start()
        failed = worker.submit_export(_episode(0))
        succeeded = worker.submit_export(_episode(0))
        failed_result = failed.result(1.0)
        succeeded_result = succeeded.result(1.0)
        self.assertFalse(failed_result.succeeded)
        self.assertIsInstance(failed_result.error, OSError)
        self.assertTrue(succeeded_result.succeeded)
        worker.close()

    def test_rollback_uses_the_same_serial_storage_queue(self) -> None:
        writer = FailingWriter()
        rollback = FakeRollback()
        worker = ExportWorker(writer, rollback=rollback, capacity=1)
        worker.start()
        ticket = worker.submit_rollback(3)
        result = ticket.result(1.0)
        self.assertEqual(result.kind, ExportTaskKind.ROLLBACK)
        self.assertTrue(result.succeeded)
        self.assertTrue(result.storage_changed)
        self.assertEqual(rollback.indices, [3])
        worker.close()

    def test_submit_requires_started_worker(self) -> None:
        worker = ExportWorker(FailingWriter())
        with self.assertRaises(LifecycleError):
            worker.submit_export(_episode(0))


if __name__ == "__main__":
    unittest.main()
