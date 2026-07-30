"""Thin application entry point tests."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from airo_doffy.apps.common import (
    parse_overrides,
    resolve_session_factory,
    run_application,
    run_session,
)
from airo_doffy.config import AiroDoffyConfig
from airo_doffy.core.errors import ModelValidationError


class FakeSession:
    def __init__(self, calls: list[str], *, interrupt: bool = False) -> None:
        self.calls = calls
        self.interrupt = interrupt

    def start(self) -> None:
        self.calls.append("start")

    def run(self) -> None:
        self.calls.append("run")
        if self.interrupt:
            raise KeyboardInterrupt

    def request_stop(self) -> None:
        self.calls.append("request_stop")

    def close(self) -> None:
        self.calls.append("close")


class AppEntrypointTest(unittest.TestCase):
    def test_run_session_closes_normally_and_on_keyboard_interrupt(self) -> None:
        normal = []
        self.assertEqual(run_session(FakeSession(normal)), 0)
        self.assertEqual(normal, ["start", "run", "close"])

        interrupted = []
        self.assertEqual(
            run_session(FakeSession(interrupted, interrupt=True)),
            0,
        )
        self.assertEqual(
            interrupted,
            ["start", "run", "request_stop", "close"],
        )

    def test_overrides_require_dotted_unique_assignments(self) -> None:
        self.assertEqual(
            parse_overrides(
                ["robot.robot_type=realman", "runtime.collect_rate_hz=50"]
            ),
            {
                "robot.robot_type": "realman",
                "runtime.collect_rate_hz": "50",
            },
        )
        with self.assertRaises(ModelValidationError):
            parse_overrides(["bad"])
        with self.assertRaises(ModelValidationError):
            parse_overrides(["robot.robot_type=ur", "robot.robot_type=realman"])

    def test_run_application_loads_config_and_calls_explicit_factory(self) -> None:
        module_name = "_airo_doffy_test_app_factory"
        module = types.ModuleType(module_name)
        calls = []
        received = []

        def build(config: AiroDoffyConfig) -> FakeSession:
            received.append(config)
            return FakeSession(calls)

        module.build = build
        sys.modules[module_name] = module
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "default.yaml")
                path.write_text("{}", encoding="utf-8")
                result = run_application(
                    mode="teleop",
                    argv=[
                        "--config",
                        str(path),
                        "--session-factory",
                        f"{module_name}:build",
                        "--set",
                        "runtime.collect_rate_hz=25",
                    ],
                    environment={},
                )
        finally:
            sys.modules.pop(module_name, None)
        self.assertEqual(result, 0)
        self.assertEqual(calls, ["start", "run", "close"])
        self.assertEqual(received[0].runtime.collect_rate_hz, 25)

    def test_factory_target_validation_is_explicit(self) -> None:
        with self.assertRaises(ModelValidationError):
            resolve_session_factory("invalid")


if __name__ == "__main__":
    unittest.main()
