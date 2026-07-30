"""Tests for stale input hold and explicit teleoperation recovery."""

from __future__ import annotations

import unittest

from airo_doffy.config import TeleopConfig
from airo_doffy.core import (
    ClockDomain,
    ControllerState,
    HandSide,
    LifecycleError,
    RobotAction,
    RobotCommandType,
    RobotState,
    VRInputMode,
    VRInputState,
)
from airo_doffy.teleop import TeleopWatchdog, WatchdogState

IDENTITY_POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def vr(sequence: int, timestamp_ns: int, *, receive: bool = True) -> VRInputState:
    controllers = tuple(
        ControllerState(
            sequence=sequence,
            source_timestamp_ns=timestamp_ns,
            receive_timestamp_ns=timestamp_ns if receive else None,
            clock_domain=ClockDomain.DEVICE,
            side=side,
            position_m=(0, 0, 0),
            orientation_xyzw=(0, 0, 0, 1),
            joystick_xy=(0, 0),
            index_trigger=0,
            grip_trigger=0,
            buttons=frozenset(),
        )
        for side in (HandSide.LEFT, HandSide.RIGHT)
    )
    return VRInputState(
        sequence=sequence,
        source_timestamp_ns=timestamp_ns,
        receive_timestamp_ns=timestamp_ns if receive else None,
        clock_domain=ClockDomain.DEVICE,
        mode=VRInputMode.CONTROLLERS,
        controllers=controllers,
    )


def robot(sequence: int, timestamp_ns: int) -> RobotState:
    return RobotState(
        sequence=sequence,
        source_timestamp_ns=timestamp_ns,
        receive_timestamp_ns=timestamp_ns,
        clock_domain=ClockDomain.MONOTONIC,
        joints_rad=(0.0,) * 6,
        tcp_pose=IDENTITY_POSE,
    )


def action(
    sequence: int,
    command_type: RobotCommandType = RobotCommandType.JOINT_POSITION,
) -> RobotAction:
    sizes = {
        RobotCommandType.JOINT_POSITION: 6,
        RobotCommandType.JOINT_VELOCITY: 6,
        RobotCommandType.TCP_TWIST: 6,
    }
    return RobotAction(
        sequence=sequence,
        source_timestamp_ns=sequence,
        receive_timestamp_ns=sequence,
        clock_domain=ClockDomain.MONOTONIC,
        command_type=command_type,
        values=(1.0,) * sizes.get(command_type, 0),
        duration_s=0.01 if command_type in sizes else None,
        gripper_width_m=0.04,
    )


class TeleopWatchdogTest(unittest.TestCase):
    def test_initial_recovery_requires_fresh_samples_and_reference_ack(self) -> None:
        watchdog = TeleopWatchdog(
            TeleopConfig(watchdog_recovery_samples=2),
        )
        first = watchdog.evaluate(vr(1, 900), robot(1, 900), now_ns=1_000)
        repeated = watchdog.evaluate(vr(1, 900), robot(1, 900), now_ns=1_001)
        ready = watchdog.evaluate(vr(2, 901), robot(2, 901), now_ns=1_002)
        self.assertEqual(first.state, WatchdogState.RECOVERING)
        self.assertEqual(first.recovery_samples, 1)
        self.assertEqual(repeated.recovery_samples, 1)
        self.assertTrue(ready.reference_ready)
        with self.assertRaises(LifecycleError):
            watchdog.acknowledge_reference(vr_sequence=1, robot_sequence=1)
        watchdog.acknowledge_reference(vr_sequence=2, robot_sequence=2)
        self.assertEqual(watchdog.state, WatchdogState.ACTIVE)
        self.assertEqual(watchdog.metrics.recoveries, 1)

    def test_stale_vr_zeroes_velocity_once_then_emits_monotonic_hold(self) -> None:
        watchdog = TeleopWatchdog(TeleopConfig(), initially_active=True)
        active = watchdog.evaluate(vr(5, 900), robot(5, 900), now_ns=1_000)
        sent = watchdog.guard(action(5, RobotCommandType.JOINT_VELOCITY), active)
        stale = watchdog.evaluate(
            vr(5, 900),
            robot(5, 900),
            now_ns=300_000_901,
        )
        zero = watchdog.guard(action(5, RobotCommandType.JOINT_VELOCITY), stale)
        still_stale = watchdog.evaluate(
            vr(5, 900),
            robot(5, 900),
            now_ns=300_000_902,
        )
        hold = watchdog.guard(
            action(5, RobotCommandType.JOINT_VELOCITY),
            still_stale,
        )
        self.assertEqual(sent.command_type, RobotCommandType.JOINT_VELOCITY)
        self.assertTrue(stale.tripped)
        self.assertEqual(zero.command_type, RobotCommandType.JOINT_VELOCITY)
        self.assertEqual(zero.values, (0.0,) * 6)
        self.assertIsNone(zero.gripper_width_m)
        self.assertEqual(hold.command_type, RobotCommandType.HOLD)
        self.assertEqual((sent.sequence, zero.sequence, hold.sequence), (5, 6, 7))
        self.assertEqual(watchdog.metrics.zero_velocity_actions, 1)
        self.assertEqual(watchdog.metrics.hold_actions, 1)

    def test_stale_robot_position_command_transitions_directly_to_hold(self) -> None:
        watchdog = TeleopWatchdog(TeleopConfig(), initially_active=True)
        decision = watchdog.evaluate(
            vr(1, 100_000_000),
            robot(1, 0),
            now_ns=300_000_000,
        )
        guarded = watchdog.guard(action(1), decision)
        self.assertEqual(decision.reason, "robot_state_stale")
        self.assertEqual(guarded.command_type, RobotCommandType.HOLD)
        self.assertTrue(decision.reference_required)

    def test_outdated_active_decision_cannot_bypass_a_later_trip(self) -> None:
        watchdog = TeleopWatchdog(TeleopConfig(), initially_active=True)
        active = watchdog.evaluate(vr(1, 100), robot(1, 100), now_ns=101)
        watchdog.evaluate(None, robot(1, 100), now_ns=102)
        with self.assertRaises(LifecycleError):
            watchdog.guard(action(1), active)

    def test_incomparable_clock_and_future_timestamp_are_not_fresh(self) -> None:
        watchdog = TeleopWatchdog(TeleopConfig(), initially_active=True)
        incomparable = watchdog.evaluate(
            vr(1, 100, receive=False),
            robot(1, 100),
            now_ns=100,
        )
        self.assertEqual(incomparable.reason, "vr_input_clock_incomparable")
        future = watchdog.evaluate(
            vr(2, 20_000_000),
            robot(2, 20_000_000),
            now_ns=1,
        )
        self.assertEqual(future.reason, "vr_input_timestamp_in_future")

    def test_force_hold_and_stop_passthrough(self) -> None:
        watchdog = TeleopWatchdog(TeleopConfig(), initially_active=True)
        watchdog.force_hold("operator safe hold")
        decision = watchdog.evaluate(None, None, now_ns=1)
        stop = action(1, RobotCommandType.STOP)
        guarded = watchdog.guard(stop, decision)
        self.assertEqual(guarded.command_type, RobotCommandType.STOP)
        self.assertEqual(watchdog.metrics.trips, 1)


if __name__ == "__main__":
    unittest.main()
