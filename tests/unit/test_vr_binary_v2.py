"""Golden round-trip and malformed-packet tests for VR binary v2."""

from __future__ import annotations

import unittest

from airo_doffy.core import (
    ControllerButton,
    ControllerState,
    HandSide,
    HandState,
    ModelValidationError,
    VRInputMode,
    VRInputState,
)
from airo_doffy.devices.vr import (
    BINARY_V2_HEADER,
    BINARY_V2_MAGIC,
    BINARY_V2_VERSION,
    decode_vr_binary_v2,
    decode_vr_message,
    encode_vr_binary_v2,
)


def controller(side: HandSide, buttons=()) -> ControllerState:
    return ControllerState(
        sequence=42,
        source_timestamp_ns=123_456,
        side=side,
        position_m=(1, 2, 3),
        orientation_xyzw=(0, 0, 0, 1),
        joystick_xy=(0.25, -0.5),
        index_trigger=0.75,
        grip_trigger=0.5,
        buttons=frozenset(buttons),
    )


def hand(side: HandSide, *, wrist: bool) -> HandState:
    return HandState(
        sequence=42,
        source_timestamp_ns=123_456,
        side=side,
        joints_m=tuple((index, index + 0.25, index + 0.5) for index in range(26)),
        wrist_position_m=(1, 2, 3) if wrist else None,
        wrist_orientation_xyzw=(0, 0, 0, 1) if wrist else None,
    )


class VrBinaryV2Test(unittest.TestCase):
    def test_controller_header_layout_and_round_trip(self) -> None:
        state = VRInputState(
            sequence=42,
            source_timestamp_ns=123_456,
            mode=VRInputMode.CONTROLLERS,
            controllers=(
                controller(
                    HandSide.LEFT,
                    (ControllerButton.PRIMARY, ControllerButton.JOYSTICK),
                ),
                controller(HandSide.RIGHT, (ControllerButton.SECONDARY,)),
            ),
        )
        encoded = encode_vr_binary_v2(state)
        header = BINARY_V2_HEADER.unpack_from(encoded)
        self.assertEqual(
            header,
            (
                BINARY_V2_MAGIC,
                BINARY_V2_VERSION,
                1,
                2,
                0,
                42,
                123_456,
                len(encoded) - BINARY_V2_HEADER.size,
            ),
        )
        decoded = decode_vr_binary_v2(encoded, receive_timestamp_ns=999)
        self.assertEqual(decoded.sequence, state.sequence)
        self.assertEqual(decoded.receive_timestamp_ns, 999)
        self.assertEqual(decoded.controllers[0].side, HandSide.LEFT)
        self.assertEqual(decoded.controllers[0].buttons, state.controllers[0].buttons)
        self.assertEqual(decoded.controllers[1].buttons, state.controllers[1].buttons)
        for actual, expected in zip(
            decoded.controllers[0].position_m,
            state.controllers[0].position_m,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

    def test_one_or_two_hands_with_optional_wrist_round_trip(self) -> None:
        state = VRInputState(
            sequence=9,
            source_timestamp_ns=8_000,
            mode=VRInputMode.HANDS,
            hands=(
                hand(HandSide.LEFT, wrist=True),
                hand(HandSide.RIGHT, wrist=False),
            ),
        )
        encoded = encode_vr_binary_v2(state)
        decoded = decode_vr_message(encoded, receive_timestamp_ns=9_000)
        self.assertEqual(decoded.mode, VRInputMode.HANDS)
        self.assertEqual(len(decoded.hands), 2)
        self.assertEqual(decoded.hands[0].wrist_position_m, (1.0, 2.0, 3.0))
        self.assertIsNone(decoded.hands[1].wrist_position_m)
        self.assertEqual(decoded.hands[1].joints_m[25], (25.0, 25.25, 25.5))

    def test_version_length_flags_buttons_and_truncation_are_strict(self) -> None:
        state = VRInputState(
            sequence=1,
            source_timestamp_ns=2,
            mode=VRInputMode.CONTROLLERS,
            controllers=(
                controller(HandSide.LEFT),
                controller(HandSide.RIGHT),
            ),
        )
        encoded = bytearray(encode_vr_binary_v2(state))
        invalid_version = encoded.copy()
        invalid_version[4] = 3
        self.assertIsNone(
            decode_vr_binary_v2(invalid_version, receive_timestamp_ns=3)
        )
        invalid_flags = encoded.copy()
        invalid_flags[7] = 1
        self.assertIsNone(
            decode_vr_binary_v2(invalid_flags, receive_timestamp_ns=3)
        )
        invalid_buttons = encoded.copy()
        invalid_buttons[BINARY_V2_HEADER.size + 1] = 0x80
        self.assertIsNone(
            decode_vr_binary_v2(invalid_buttons, receive_timestamp_ns=3)
        )
        self.assertIsNone(
            decode_vr_binary_v2(encoded[:-1], receive_timestamp_ns=3)
        )
        self.assertIsNone(decode_vr_message(b"not-v2", receive_timestamp_ns=3))

    def test_encode_rejects_sequence_larger_than_uint32(self) -> None:
        state = VRInputState(
            sequence=2**32,
            source_timestamp_ns=2,
            mode=VRInputMode.CONTROLLERS,
            controllers=(
                controller(HandSide.LEFT),
                controller(HandSide.RIGHT),
            ),
        )
        with self.assertRaises(ModelValidationError):
            encode_vr_binary_v2(state)


if __name__ == "__main__":
    unittest.main()
