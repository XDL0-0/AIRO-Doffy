"""Tests for scripted VR trajectories and deterministic fault injection."""

from __future__ import annotations

import unittest

from airo_doffy.config import NetworkConfig, VRConfig, VRSourceFactory
from airo_doffy.core import (
    ClockDomain,
    ControllerButton,
    ControllerState,
    HandSide,
    LifecycleError,
    ModelValidationError,
    VRInputMode,
    VRInputState,
)
from airo_doffy.devices.vr import MockVRInputSource, VRInputSource


class _Clock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def now_ns(self) -> int:
        return self.value


def controller_state(sequence: int, x: float = 0.0) -> VRInputState:
    controllers = tuple(
        ControllerState(
            sequence=sequence,
            source_timestamp_ns=sequence,
            clock_domain=ClockDomain.DEVICE,
            side=side,
            position_m=(x, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            joystick_xy=(0.0, 0.0),
            index_trigger=0.5,
            grip_trigger=0.25,
            buttons=frozenset({ControllerButton.PRIMARY}),
        )
        for side in (HandSide.LEFT, HandSide.RIGHT)
    )
    return VRInputState(
        sequence=sequence,
        source_timestamp_ns=sequence,
        clock_domain=ClockDomain.DEVICE,
        mode=VRInputMode.CONTROLLERS,
        controllers=controllers,
    )


class MockVRInputSourceTest(unittest.TestCase):
    def test_default_controller_lifecycle_and_packet_loss(self) -> None:
        source = MockVRInputSource(VRConfig(), drop_every=2)
        self.assertIsInstance(source, VRInputSource)
        with self.assertRaises(LifecycleError):
            source.read_latest()
        source.start()
        first = source.read_latest()
        self.assertEqual(first.sequence, 0)
        self.assertEqual(len(first.controllers), 2)
        self.assertIsNone(source.read_latest())
        self.assertEqual(source.read_latest().sequence, 2)
        source.close()
        source.close()
        with self.assertRaises(LifecycleError):
            source.read_latest()

    def test_script_preserves_trajectory_buttons_and_pair_reordering(self) -> None:
        states = tuple(controller_state(index, float(index)) for index in range(3))
        source = MockVRInputSource(
            VRConfig(),
            script=states,
            reorder_pairs=True,
        )
        source.start()
        outputs = [source.read_latest() for _ in range(4)]
        self.assertEqual(
            [state.sequence if state is not None else None for state in outputs],
            [1, 0, 2, None],
        )
        self.assertEqual(outputs[0].controllers[0].position_m[0], 1.0)
        self.assertIn(
            ControllerButton.PRIMARY,
            outputs[0].controllers[0].buttons,
        )
        source.close()

    def test_stale_timeout_force_stale_and_delay(self) -> None:
        clock = _Clock(100)
        sleeps: list[float] = []
        source = MockVRInputSource(
            VRConfig(),
            stale_after_s=0.5,
            artificial_delay_s=0.25,
            clock=clock,
            sleep=sleeps.append,
        )
        source.start()
        self.assertIsNotNone(source.read_latest())
        source.set_stale(True)
        self.assertIsNone(source.read_latest())
        source.set_stale(False)
        clock.value += 500_000_000
        self.assertIsNone(source.read_latest())
        self.assertEqual(sleeps, [0.25, 0.25, 0.25])
        source.close()

    def test_factory_creates_hand_source(self) -> None:
        factory = VRSourceFactory(
            target="airo_doffy.devices.vr.mock:create_mock_vr"
        )
        source = factory.create(
            VRConfig(tracking_mode="hand"),
            NetworkConfig(),
        )
        self.assertIsInstance(source, MockVRInputSource)
        source.start()
        state = source.read_latest()
        self.assertEqual(state.mode, VRInputMode.HANDS)
        self.assertEqual(len(state.hands[0].joints_m), 26)
        source.close()

    def test_validation(self) -> None:
        with self.assertRaises(ModelValidationError):
            MockVRInputSource(VRConfig(), script=())
        with self.assertRaises(ModelValidationError):
            MockVRInputSource(VRConfig(), script=(object(),))
        with self.assertRaises(ModelValidationError):
            MockVRInputSource(VRConfig(), reorder_pairs=True)
        with self.assertRaises(ModelValidationError):
            MockVRInputSource(VRConfig(), drop_every=0)
        with self.assertRaises(ModelValidationError):
            MockVRInputSource(VRConfig(), stale_after_s=-1)
        with self.assertRaises(ModelValidationError):
            MockVRInputSource(
                VRConfig(tracking_mode="hand"),
                script=(controller_state(0),),
            )


if __name__ == "__main__":
    unittest.main()
