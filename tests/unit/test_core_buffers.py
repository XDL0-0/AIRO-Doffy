"""Concurrency and wrap-order tests for latest-value buffers."""

from __future__ import annotations

import threading
import unittest
from dataclasses import dataclass

from airo_doffy.core import (
    BufferClosedError,
    LatestValueBuffer,
    ModelValidationError,
    is_newer_sequence,
)


@dataclass(frozen=True)
class Item:
    sequence: int
    payload: str = ""


class SequenceOrderTest(unittest.TestCase):
    def test_unbounded_sequence_order(self) -> None:
        self.assertTrue(is_newer_sequence(11, 10))
        self.assertFalse(is_newer_sequence(10, 10))
        self.assertFalse(is_newer_sequence(9, 10))

    def test_modular_wrap_order(self) -> None:
        self.assertTrue(is_newer_sequence(0, 7, 8))
        self.assertTrue(is_newer_sequence(1, 7, 8))
        self.assertFalse(is_newer_sequence(7, 1, 8))
        self.assertFalse(is_newer_sequence(5, 1, 8))

    def test_sequence_validation(self) -> None:
        with self.assertRaises(ModelValidationError):
            is_newer_sequence(-1, 0)
        with self.assertRaises(ModelValidationError):
            is_newer_sequence(3, 0, 3)


class LatestValueBufferTest(unittest.TestCase):
    def test_keeps_only_latest_and_rejects_stale(self) -> None:
        buffer = LatestValueBuffer[Item]()
        self.assertTrue(buffer.publish(Item(1, "one")))
        self.assertFalse(buffer.publish(Item(1, "duplicate")))
        self.assertFalse(buffer.publish(Item(0, "stale")))
        self.assertTrue(buffer.publish(Item(2, "two")))
        self.assertEqual(buffer.read(), Item(2, "two"))
        self.assertEqual(buffer.accepted_count, 2)
        self.assertEqual(buffer.rejected_count, 2)

    def test_many_updates_use_latest_only_semantics(self) -> None:
        buffer = LatestValueBuffer[Item]()
        for sequence in range(10_000):
            self.assertTrue(buffer.publish(Item(sequence)))
        self.assertEqual(buffer.read(), Item(9_999))
        self.assertEqual(buffer.accepted_count, 10_000)

    def test_modular_buffer_accepts_wrap(self) -> None:
        buffer = LatestValueBuffer[Item](sequence_modulus=8)
        for sequence in (6, 7, 0, 1):
            self.assertTrue(buffer.publish(Item(sequence)))
        self.assertFalse(buffer.publish(Item(7)))
        self.assertEqual(buffer.latest_sequence, 1)

    def test_wait_returns_existing_or_new_value(self) -> None:
        buffer = LatestValueBuffer[Item]()
        buffer.publish(Item(1))
        self.assertEqual(buffer.wait_for_new(timeout=0), Item(1))

        ready = threading.Event()
        result: list[Item | None] = []

        def consumer() -> None:
            ready.set()
            result.append(buffer.wait_for_new(after_sequence=1, timeout=1.0))

        thread = threading.Thread(target=consumer)
        thread.start()
        self.assertTrue(ready.wait(1.0))
        buffer.publish(Item(2))
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [Item(2)])

    def test_timeout_returns_none(self) -> None:
        buffer = LatestValueBuffer[Item]()
        self.assertIsNone(buffer.wait_for_new(after_sequence=0, timeout=0))

    def test_close_is_idempotent_and_wakes_waiter(self) -> None:
        buffer = LatestValueBuffer[Item]()
        ready = threading.Event()
        result: list[Item | None] = []

        def consumer() -> None:
            ready.set()
            result.append(buffer.wait_for_new(after_sequence=0))

        thread = threading.Thread(target=consumer)
        thread.start()
        self.assertTrue(ready.wait(1.0))
        buffer.close()
        buffer.close()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])
        self.assertTrue(buffer.closed)
        with self.assertRaises(BufferClosedError):
            buffer.publish(Item(1))

    def test_context_manager_closes_buffer(self) -> None:
        with LatestValueBuffer[Item]() as buffer:
            buffer.publish(Item(1))
        self.assertTrue(buffer.closed)

    def test_wait_argument_validation(self) -> None:
        buffer = LatestValueBuffer[Item](sequence_modulus=8)
        with self.assertRaises(ModelValidationError):
            buffer.wait_for_new(after_sequence=8)
        with self.assertRaises(ModelValidationError):
            buffer.wait_for_new(timeout=float("inf"))


if __name__ == "__main__":
    unittest.main()
