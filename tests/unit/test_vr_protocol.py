"""Golden compatibility and typed-conversion tests for current VR packets."""

from __future__ import annotations

import base64
import struct
import unittest

from airo_doffy.core import (
    ControllerButton,
    HandSide,
    VRInputMode,
)
from airo_doffy.devices.vr import (
    decode_vr_input,
    detect_packet_type,
    parse_data,
    parse_hand_data,
)


def controller_values(offset: float) -> list[float]:
    return [
        offset + 1,
        offset + 2,
        offset + 3,
        0,
        0,
        0,
        1,
        0.25,
        -0.5,
        0.75,
        0.5,
        1,
        0,
        1,
    ]


def csv(values) -> str:
    return ",".join(str(value) for value in values)


def hand_binary(*, magic: int = 0x48, count: int = 26) -> str:
    header = bytes((magic, ord("L"), count & 0xFF, count >> 8))
    header += struct.pack("<I", 0x12345678)
    joints = b"".join(
        struct.pack("<fff", index + 0.25, index + 0.5, index + 0.75)
        for index in range(26)
    )
    return "HB," + base64.b64encode(header + joints).decode("ascii")


class VrProtocolTest(unittest.TestCase):
    def test_packet_detection(self) -> None:
        self.assertEqual(detect_packet_type("1,rest"), "controller")
        self.assertEqual(detect_packet_type("C,rest"), "controller")
        self.assertEqual(detect_packet_type("H,rest"), "hand_text")
        self.assertEqual(detect_packet_type("HB,rest"), "hand_binary")
        self.assertEqual(detect_packet_type("unknown"), "unknown")

    def test_old_and_new_controller_dictionary_contract(self) -> None:
        payload = controller_values(0) + controller_values(10)
        old = parse_data(csv([123, *payload]))
        new = parse_data(csv(["C", 7, 456_000_000, *payload]))
        self.assertEqual(old[0]["FrameId"], 0)
        self.assertEqual(old[0]["Timestamp"], 123)
        self.assertEqual(old[1]["ControllerType"], "RTouch")
        self.assertEqual(old[1]["Position"], (11.0, 12.0, 13.0))
        self.assertEqual(new[0]["FrameId"], 7)
        self.assertEqual(new[0]["Timestamp"], 456_000_000)
        self.assertIsNone(parse_data("C,too,short"))
        self.assertIsNone(parse_data(csv([123, *payload[:-1]])))

    def test_hand_text_contract(self) -> None:
        wrist = (1, 2, 3, 0, 0, 0, 1)
        joints = tuple(value for index in range(26) for value in (index, index + 1, index + 2))
        packet = csv(("H", "R", 9, 1_000_000_000, *wrist, *joints))
        hand = parse_hand_data(packet)
        self.assertEqual(hand["side"], "R")
        self.assertEqual(hand["frame_id"], 9)
        self.assertEqual(hand["wrist_pose"]["position"], (1.0, 2.0, 3.0))
        self.assertEqual(len(hand["bones"]), 26)
        self.assertEqual(hand["bones"][25], (25.0, 26.0, 27.0))
        self.assertIsNone(parse_hand_data(packet + ",0"))

    def test_hand_binary_little_endian_contract(self) -> None:
        hand = parse_hand_data(hand_binary(), wall_time_ms=321)
        self.assertEqual(hand["side"], "L")
        self.assertEqual(hand["frame_id"], 0x12345678)
        self.assertEqual(hand["timestamp"], 321)
        self.assertIsNone(hand["wrist_pose"])
        self.assertEqual(hand["bones"][0], (0.25, 0.5, 0.75))
        self.assertEqual(hand["bones"][25], (25.25, 25.5, 25.75))
        self.assertIsNone(parse_hand_data(hand_binary(magic=0)))
        self.assertIsNone(parse_hand_data(hand_binary(count=25)))

    def test_typed_controller_conversion_and_old_timestamp_units(self) -> None:
        payload = controller_values(0) + controller_values(10)
        old = decode_vr_input(
            csv([123, *payload]),
            receive_timestamp_ns=999,
        )
        new = decode_vr_input(
            csv(["C", 7, 456_000_000, *payload]),
            receive_timestamp_ns=999,
        )
        self.assertEqual(old.mode, VRInputMode.CONTROLLERS)
        self.assertEqual(old.sequence, 999)
        self.assertEqual(old.source_timestamp_ns, 123_000_000)
        self.assertEqual(new.sequence, 7)
        self.assertEqual(new.source_timestamp_ns, 456_000_000)
        self.assertEqual(new.controllers[0].side, HandSide.LEFT)
        self.assertIn(ControllerButton.PRIMARY, new.controllers[0].buttons)
        self.assertIn(ControllerButton.JOYSTICK, new.controllers[0].buttons)

    def test_typed_hand_text_and_binary_conversion(self) -> None:
        joints = tuple(value for index in range(26) for value in (index, index + 1, index + 2))
        text = csv(("H", "R", 9, 1_000_000_000, 1, 2, 3, 0, 0, 0, 1, *joints))
        typed_text = decode_vr_input(text, receive_timestamp_ns=2_000_000_000)
        typed_binary = decode_vr_input(
            hand_binary(),
            receive_timestamp_ns=3_000_000_000,
        )
        self.assertEqual(typed_text.mode, VRInputMode.HANDS)
        self.assertEqual(typed_text.hands[0].side, HandSide.RIGHT)
        self.assertEqual(typed_text.hands[0].wrist_position_m, (1.0, 2.0, 3.0))
        self.assertEqual(typed_binary.hands[0].side, HandSide.LEFT)
        self.assertIsNone(typed_binary.hands[0].wrist_position_m)
        self.assertEqual(typed_binary.source_timestamp_ns, 3_000_000_000)


if __name__ == "__main__":
    unittest.main()
