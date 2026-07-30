"""Tests for dependency-injected runtime command routing."""

from __future__ import annotations

import unittest

from airo_doffy.core import (
    CommandRejectedError,
    ModelValidationError,
    RuntimeCommandType,
    RuntimeEventSeverity,
    RuntimeEventType,
)
from airo_doffy.streaming.commands import (
    CommandDispatcher,
    CommandRouter,
)

from tests.unit.test_command_protocol import command


class _Clock:
    def __init__(self) -> None:
        self.value = 100

    def now_ns(self) -> int:
        self.value += 1
        return self.value


class CommandRouterTest(unittest.TestCase):
    def test_selects_only_matching_handler_and_reports_acceptance(self) -> None:
        calls = []

        def start(incoming):
            calls.append(("start", incoming.command_id))
            return "recording started"

        def stop(incoming):
            calls.append(("stop", incoming.command_id))
            return None

        router = CommandRouter(
            {
                RuntimeCommandType.START_RECORDING: start,
                RuntimeCommandType.STOP_RECORDING: stop,
            },
            clock=_Clock(),
        )
        self.assertIsInstance(router, CommandDispatcher)
        event = router.dispatch(command())
        self.assertEqual(calls, [("start", "command-1")])
        self.assertEqual(event.kind, RuntimeEventType.COMMAND_ACCEPTED)
        self.assertEqual(event.message, "recording started")
        self.assertEqual(event.command_id, "command-1")
        self.assertEqual(event.sequence, 0)
        self.assertEqual(router.metrics.accepted, 1)

    def test_unhandled_command_is_explicitly_rejected(self) -> None:
        router = CommandRouter({}, clock=_Clock())
        event = router.dispatch(command())
        self.assertEqual(event.kind, RuntimeEventType.COMMAND_REJECTED)
        self.assertEqual(event.severity, RuntimeEventSeverity.WARNING)
        self.assertIn("no handler", event.message)
        self.assertEqual(router.metrics.unhandled, 1)
        self.assertEqual(router.metrics.rejected, 1)

    def test_handler_can_reject_without_becoming_internal_error(self) -> None:
        def reject(_incoming):
            raise CommandRejectedError("recording is already active")

        router = CommandRouter(
            {RuntimeCommandType.START_RECORDING: reject},
            clock=_Clock(),
        )
        event = router.dispatch(command())
        self.assertEqual(event.kind, RuntimeEventType.COMMAND_REJECTED)
        self.assertEqual(event.severity, RuntimeEventSeverity.WARNING)
        self.assertEqual(event.message, "recording is already active")
        self.assertEqual(router.metrics.rejected, 1)
        self.assertEqual(router.metrics.errors, 0)

    def test_unexpected_handler_exception_is_reported_as_error(self) -> None:
        def fail(_incoming):
            raise RuntimeError("disk unavailable")

        router = CommandRouter(
            {RuntimeCommandType.START_RECORDING: fail},
            clock=_Clock(),
        )
        event = router.dispatch(command())
        self.assertEqual(event.kind, RuntimeEventType.COMMAND_REJECTED)
        self.assertEqual(event.severity, RuntimeEventSeverity.ERROR)
        self.assertIn("RuntimeError", event.message)
        self.assertEqual(event.details[-1], ("error_type", "RuntimeError"))
        self.assertEqual(router.metrics.errors, 1)

    def test_invalid_handler_result_and_configuration_are_observable(self) -> None:
        router = CommandRouter(
            {RuntimeCommandType.START_RECORDING: lambda _incoming: False},
            clock=_Clock(),
        )
        event = router.dispatch(command())
        self.assertEqual(event.severity, RuntimeEventSeverity.ERROR)
        self.assertIn("TypeError", event.message)
        with self.assertRaises(ModelValidationError):
            CommandRouter({"unknown": lambda _incoming: None})  # type: ignore[dict-item]
        with self.assertRaises(ModelValidationError):
            CommandRouter(
                {RuntimeCommandType.START_RECORDING: None}  # type: ignore[dict-item]
            )

    def test_handler_mapping_is_copied_at_composition_boundary(self) -> None:
        calls = []
        handlers = {
            RuntimeCommandType.START_RECORDING: lambda _incoming: calls.append("old")
        }
        router = CommandRouter(handlers, clock=_Clock())
        handlers[RuntimeCommandType.START_RECORDING] = lambda _incoming: calls.append(
            "new"
        )
        router.dispatch(command())
        self.assertEqual(calls, ["old"])


if __name__ == "__main__":
    unittest.main()
