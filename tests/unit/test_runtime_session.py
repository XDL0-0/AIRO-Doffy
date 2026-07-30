"""Mock-only tests for composable teleoperation session orchestration."""

from __future__ import annotations

import unittest
from threading import Event

from airo_doffy.core import (
    ClockDomain,
    RobotAction,
    RobotCommandType,
    RobotState,
)
from airo_doffy.config import TeleopConfig, VRConfig
from airo_doffy.devices.vr import MockVRInputSource
from airo_doffy.runtime import TeleopCycle, TeleopSession
from airo_doffy.teleop.safety import TeleopWatchdog, WatchdogState
from tests.unit.test_mock_vr import controller_state


class FakeClock:
    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.value = now_ns

    def now_ns(self) -> int:
        result = self.value
        self.value += 10_000_000
        return result


class FakeStateSource:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.sequence = 0

    def read_state(self) -> RobotState:
        state = RobotState(
            sequence=self.sequence,
            source_timestamp_ns=self.clock.value,
            joints_rad=(0.0,) * 6,
            tcp_pose=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        self.sequence += 1
        return state


class FakeMapping:
    def map_input(self, vr_input, robot_state, dt_s: float) -> RobotAction:
        del robot_state, dt_s
        return RobotAction(
            sequence=vr_input.sequence,
            source_timestamp_ns=vr_input.source_timestamp_ns,
            clock_domain=ClockDomain.MONOTONIC,
            command_type=RobotCommandType.JOINT_POSITION,
            values=(0.1,) * 6,
        )


class FakeFilter:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject

    def apply(self, action, robot_state, now_ns: int):
        del robot_state, now_ns
        return None if self.reject else action


class FakeExecutor:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.actions: list[RobotAction] = []
        self.started = False

    def start(self) -> None:
        self.calls.append("start:executor")
        self.started = True

    def run(self, external_stop: Event | None = None) -> None:
        self.calls.append("run:executor")
        assert external_stop is not None
        external_stop.wait()

    def submit(self, action: RobotAction) -> bool:
        self.actions.append(action)
        return True

    def close(self) -> None:
        self.calls.append("close:executor")
        self.started = False


class FakeExtension:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.cycles: list[TeleopCycle] = []

    def start(self) -> None:
        self.calls.append("start:extension")

    def on_cycle(self, cycle: object) -> None:
        assert isinstance(cycle, TeleopCycle)
        self.cycles.append(cycle)

    def close(self) -> None:
        self.calls.append("close:extension")


class TeleopSessionTest(unittest.TestCase):
    def create_session(
        self,
        *,
        reject: bool = False,
    ) -> tuple[TeleopSession, FakeExecutor, FakeExtension, list[str]]:
        clock = FakeClock()
        calls: list[str] = []
        vr = MockVRInputSource(
            VRConfig(),
            script=(
                controller_state(sequence=0),
            ),
        )
        executor = FakeExecutor(calls)
        extension = FakeExtension(calls)
        session = TeleopSession(
            vr_source=vr,
            state_source=FakeStateSource(clock),
            mapping=FakeMapping(),
            action_filter=FakeFilter(reject=reject),
            executor=executor,
            target_hz=100,
            extensions=(extension,),
            clock=clock,
        )
        return session, executor, extension, calls

    def test_mock_session_maps_filters_submits_and_notifies_extensions(self) -> None:
        session, executor, extension, calls = self.create_session()
        session.start()
        cycle = session.step_once()
        self.assertTrue(cycle.submitted)
        self.assertEqual(
            cycle.safe_action.command_type,
            RobotCommandType.JOINT_POSITION,
        )
        self.assertEqual(executor.actions, [cycle.safe_action])
        self.assertEqual(extension.cycles, [cycle])
        session.close()
        self.assertEqual(calls[0], "start:executor")
        self.assertIn("start:extension", calls)
        self.assertEqual(calls[-2:], ["close:extension", "close:executor"])

    def test_safety_rejection_submits_hold_instead_of_repeating_old_action(self) -> None:
        session, executor, _extension, _calls = self.create_session(reject=True)
        session.start()
        cycle = session.step_once()
        self.assertEqual(cycle.safe_action.command_type, RobotCommandType.HOLD)
        self.assertEqual(executor.actions[-1].command_type, RobotCommandType.HOLD)
        self.assertEqual(session.metrics().rejected_actions, 1)
        session.close()

    def test_run_max_cycles_reuses_same_loop_and_optional_components_can_be_absent(
        self,
    ) -> None:
        clock = FakeClock()
        calls: list[str] = []
        vr = MockVRInputSource(
            VRConfig(),
            script=(
                controller_state(sequence=0),
                controller_state(sequence=1),
            ),
        )
        executor = FakeExecutor(calls)
        session = TeleopSession(
            vr_source=vr,
            state_source=FakeStateSource(clock),
            mapping=FakeMapping(),
            action_filter=FakeFilter(),
            executor=executor,
            target_hz=1000,
            clock=clock,
        )
        session.start()
        session.run(max_cycles=2)
        self.assertEqual(session.metrics().cycles, 2)
        session.close()

    def test_missing_vr_submits_hold_and_next_mapping_sequence_stays_monotonic(
        self,
    ) -> None:
        clock = FakeClock()
        calls: list[str] = []
        vr = MockVRInputSource(VRConfig(), drop_every=2, clock=clock)
        executor = FakeExecutor(calls)
        session = TeleopSession(
            vr_source=vr,
            state_source=FakeStateSource(clock),
            mapping=FakeMapping(),
            action_filter=FakeFilter(),
            executor=executor,
            target_hz=100,
            clock=clock,
        )
        session.start()
        first = session.step_once()
        missing = session.step_once()
        recovered = session.step_once()
        self.assertEqual(first.safe_action.command_type, RobotCommandType.JOINT_POSITION)
        self.assertEqual(missing.safe_action.command_type, RobotCommandType.HOLD)
        self.assertGreater(missing.safe_action.sequence, first.safe_action.sequence)
        self.assertGreater(recovered.safe_action.sequence, missing.safe_action.sequence)
        session.close()

    def test_watchdog_rebuilds_reference_then_holds_on_stale_vr(self) -> None:
        clock = FakeClock()
        calls: list[str] = []
        vr = MockVRInputSource(VRConfig(), clock=clock)
        executor = FakeExecutor(calls)
        watchdog = TeleopWatchdog(TeleopConfig())
        session = TeleopSession(
            vr_source=vr,
            state_source=FakeStateSource(clock),
            mapping=FakeMapping(),
            action_filter=FakeFilter(),
            executor=executor,
            watchdog=watchdog,
            target_hz=100,
            clock=clock,
        )
        session.start()
        active = session.step_once()
        self.assertEqual(active.watchdog.state, WatchdogState.ACTIVE)
        self.assertEqual(active.safe_action.command_type, RobotCommandType.JOINT_POSITION)
        vr.set_stale(True)
        held = session.step_once()
        self.assertEqual(held.watchdog.state, WatchdogState.HOLDING)
        self.assertEqual(held.safe_action.command_type, RobotCommandType.HOLD)
        session.close()

    def test_close_is_idempotent(self) -> None:
        session, _executor, _extension, _calls = self.create_session()
        session.start()
        session.close()
        session.close()


if __name__ == "__main__":
    unittest.main()
