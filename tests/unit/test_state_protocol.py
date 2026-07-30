"""Tests for the versioned realtime state binary protocol."""

from __future__ import annotations

import struct
import unittest

from airo_doffy.core import (
    ClockDomain,
    ControllerButton,
    ControllerState,
    HandSide,
    ModelValidationError,
    RobotState,
    VRInputMode,
    VRInputState,
)
from airo_doffy.streaming.state import (
    STATE_HEADER,
    STATE_HEADER_SIZE,
    STATE_MAGIC,
    STATE_VERSION,
    StateMessageType,
    decode_state_packet,
    encode_state_packet,
)


def robot_state(
    sequence: int = 1,
    *,
    dof: int = 6,
    timestamp_ns: int = 123,
) -> RobotState:
    return RobotState(
        sequence=sequence,
        source_timestamp_ns=timestamp_ns,
        clock_domain=ClockDomain.MONOTONIC,
        joints_rad=tuple(index / 10 for index in range(dof)),
        tcp_pose=(
            (1, 0, 0, 0.1),
            (0, 1, 0, 0.2),
            (0, 0, 1, 0.3),
            (0, 0, 0, 1),
        ),
        gripper_width_m=0.04,
        wrench=(1, 2, 3, 0.1, 0.2, 0.3),
    )


def vr_state(sequence: int = 1, timestamp_ns: int = 456) -> VRInputState:
    controllers = tuple(
        ControllerState(
            sequence=sequence,
            source_timestamp_ns=timestamp_ns,
            clock_domain=ClockDomain.DEVICE,
            side=side,
            position_m=(0.1, 0.2, 0.3),
            orientation_xyzw=(0, 0, 0, 1),
            joystick_xy=(0.25, -0.25),
            index_trigger=0.5,
            grip_trigger=0.75,
            buttons=frozenset({ControllerButton.PRIMARY}),
        )
        for side in (HandSide.LEFT, HandSide.RIGHT)
    )
    return VRInputState(
        sequence=sequence,
        source_timestamp_ns=timestamp_ns,
        clock_domain=ClockDomain.DEVICE,
        mode=VRInputMode.CONTROLLERS,
        controllers=controllers,
    )


class StateProtocolTest(unittest.TestCase):
    def assert_float_tuple(self, actual, expected) -> None:
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=5)

    def test_header_has_fixed_twenty_byte_little_endian_layout(self) -> None:
        state = robot_state(sequence=0x01020304, timestamp_ns=0x0102030405060708)
        packet = encode_state_packet(state)
        payload = packet[STATE_HEADER_SIZE:]
        expected = struct.pack(
            "<HBBIQHH",
            STATE_MAGIC,
            STATE_VERSION,
            StateMessageType.ROBOT_STATE,
            0x01020304,
            0x0102030405060708,
            len(payload),
            0,
        )
        self.assertEqual(STATE_HEADER_SIZE, 20)
        self.assertEqual(packet[:STATE_HEADER_SIZE], expected)

    def test_vr_controller_round_trip(self) -> None:
        original = vr_state()
        decoded = decode_state_packet(
            encode_state_packet(original),
            receive_timestamp_ns=999,
        )
        self.assertIsInstance(decoded, VRInputState)
        assert isinstance(decoded, VRInputState)
        self.assertEqual(decoded.sequence, original.sequence)
        self.assertEqual(decoded.receive_timestamp_ns, 999)
        self.assertEqual(decoded.clock_domain, ClockDomain.DEVICE)
        self.assertEqual(decoded.controllers[0].buttons, original.controllers[0].buttons)
        self.assert_float_tuple(
            decoded.controllers[0].position_m,
            original.controllers[0].position_m,
        )

    def test_six_and_seven_dof_robot_round_trip(self) -> None:
        for dof in (6, 7):
            with self.subTest(dof=dof):
                original = robot_state(dof=dof)
                decoded = decode_state_packet(
                    encode_state_packet(original),
                    receive_timestamp_ns=789,
                )
                self.assertIsInstance(decoded, RobotState)
                assert isinstance(decoded, RobotState)
                self.assertEqual(decoded.sequence, original.sequence)
                self.assertEqual(decoded.receive_timestamp_ns, 789)
                self.assertEqual(decoded.clock_domain, ClockDomain.MONOTONIC)
                self.assert_float_tuple(decoded.joints_rad, original.joints_rad)
                self.assert_float_tuple(decoded.wrench, original.wrench)
                self.assertAlmostEqual(
                    decoded.gripper_width_m,
                    original.gripper_width_m,
                    places=5,
                )

    def test_strict_header_and_payload_validation(self) -> None:
        packet = bytearray(encode_state_packet(robot_state()))
        mutations = []
        wrong_magic = packet.copy()
        wrong_magic[0] ^= 0xFF
        mutations.append(wrong_magic)
        wrong_version = packet.copy()
        wrong_version[2] = 99
        mutations.append(wrong_version)
        wrong_type = packet.copy()
        wrong_type[3] = 99
        mutations.append(wrong_type)
        wrong_length = packet.copy()
        wrong_length[16:18] = struct.pack("<H", 1)
        mutations.append(wrong_length)
        wrong_flags = packet.copy()
        wrong_flags[18:20] = struct.pack("<H", 0x8000)
        mutations.append(wrong_flags)
        mutations.extend((packet[:10], packet + b"\x00"))
        for malformed in mutations:
            with self.subTest(packet=bytes(malformed[:20])):
                with self.assertRaises(ModelValidationError):
                    decode_state_packet(malformed, receive_timestamp_ns=1)

    def test_vr_inner_and_outer_metadata_must_match(self) -> None:
        packet = bytearray(encode_state_packet(vr_state()))
        payload_offset = STATE_HEADER_SIZE
        inner_sequence_offset = payload_offset + 8
        packet[inner_sequence_offset : inner_sequence_offset + 4] = struct.pack("<I", 2)
        with self.assertRaisesRegex(ModelValidationError, "metadata"):
            decode_state_packet(packet, receive_timestamp_ns=1)

    def test_sequence_and_timestamp_must_fit_wire_width(self) -> None:
        with self.assertRaisesRegex(ModelValidationError, "uint32"):
            encode_state_packet(robot_state(sequence=1 << 32))
        with self.assertRaisesRegex(ModelValidationError, "uint64"):
            encode_state_packet(robot_state(timestamp_ns=1 << 64))


if __name__ == "__main__":
    unittest.main()
