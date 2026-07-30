"""Runtime lifecycle manager and worker ownership tests."""

from __future__ import annotations

import time
import unittest
from threading import Event

from airo_doffy.core.errors import LifecycleError
from airo_doffy.runtime import (
    LifecycleManager,
    LifecycleManagerState,
    ManagedWorker,
)


class FakeLifecycle:
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        fail_start: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.name = name
        self.calls = calls
        self.fail_start = fail_start
        self.fail_close = fail_close

    def start(self) -> None:
        self.calls.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(f"start {self.name}")

    def close(self) -> None:
        self.calls.append(f"close:{self.name}")
        if self.fail_close:
            raise RuntimeError(f"close {self.name}")


class LifecycleManagerTest(unittest.TestCase):
    def test_start_order_and_reverse_close_are_deterministic(self) -> None:
        calls = []
        components = [
            (name, FakeLifecycle(name, calls))
            for name in ("robot", "executor", "vr", "visualization")
        ]
        manager = LifecycleManager(components)
        manager.start()
        self.assertEqual(
            calls,
            [
                "start:robot",
                "start:executor",
                "start:vr",
                "start:visualization",
            ],
        )
        manager.close()
        self.assertEqual(
            calls[-4:],
            [
                "close:visualization",
                "close:vr",
                "close:executor",
                "close:robot",
            ],
        )
        manager.close()
        self.assertEqual(manager.snapshot().state, LifecycleManagerState.CLOSED)

    def test_partial_start_failure_closes_only_successful_components(self) -> None:
        calls = []
        manager = LifecycleManager(
            (
                ("first", FakeLifecycle("first", calls)),
                ("broken", FakeLifecycle("broken", calls, fail_start=True)),
                ("never", FakeLifecycle("never", calls)),
            )
        )
        with self.assertRaisesRegex(LifecycleError, "broken"):
            manager.start()
        self.assertEqual(
            calls,
            ["start:first", "start:broken", "close:first"],
        )
        self.assertEqual(manager.snapshot().state, LifecycleManagerState.FAILED)
        manager.close()

    def test_close_attempts_all_resources_when_one_cleanup_fails(self) -> None:
        calls = []
        manager = LifecycleManager(
            (
                ("first", FakeLifecycle("first", calls)),
                ("broken", FakeLifecycle("broken", calls, fail_close=True)),
                ("last", FakeLifecycle("last", calls)),
            )
        )
        manager.start()
        with self.assertRaisesRegex(LifecycleError, "broken"):
            manager.close()
        self.assertEqual(
            calls[-3:],
            ["close:last", "close:broken", "close:first"],
        )
        manager.close()


class ManagedWorkerTest(unittest.TestCase):
    def test_worker_stops_and_joins_without_leaving_a_thread(self) -> None:
        entered = Event()

        def target(stop: Event) -> None:
            entered.set()
            stop.wait()

        worker = ManagedWorker(target, name="test-worker")
        worker.start()
        self.assertTrue(entered.wait(1.0))
        self.assertTrue(worker.snapshot().running)
        worker.close()
        self.assertFalse(worker.snapshot().running)
        worker.close()

    def test_worker_failure_is_reported_by_health_check(self) -> None:
        def target(_stop: Event) -> None:
            raise RuntimeError("boom")

        worker = ManagedWorker(target, name="failing-worker")
        worker.start()
        deadline = time.monotonic() + 1.0
        while worker.snapshot().error is None and time.monotonic() < deadline:
            time.sleep(0.005)
        with self.assertRaisesRegex(LifecycleError, "boom"):
            worker.check_health()
        worker.close()


if __name__ == "__main__":
    unittest.main()
