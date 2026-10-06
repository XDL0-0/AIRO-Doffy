from __future__ import annotations

import base64
import struct
import unittest

from doffy_teleop.protocol.control import LegacyControlState
from doffy_teleop.protocol.jpeg import JpegChunkAssembler, JpegDatagram
from doffy_teleop.protocol.signaling import SignalingMessage, parse_signaling_message
from doffy_teleop.protocol.vr import LegacyVRPacketDecoder


def _controller_packet() -> str:
    values = [str(index / 10) for index in range(28)]
    return "C,7,123456," + ",".join(values)


def _hand_text_packet() -> str:
    # H,side,frame,timestamp,wrist pose (7), and 26 xyz joints.
    values = [str(index / 100) for index in range(7 + 26 * 3)]
    return "H,L,9,987654," + ",".join(values)


def _hand_binary_packet() -> str:
    header = struct.pack("<BBHI", 0x48, ord("R"), 26, 12)
    joints = b"".join(struct.pack("<fff", index, index + 1, index + 2) for index in range(26))
    return "HB," + base64.b64encode(header + joints).decode("ascii")


class LegacyProtocolTests(unittest.TestCase):
    def test_controller_text_and_both_hand_formats_are_shared(self) -> None:
        decoder = LegacyVRPacketDecoder()
        packet_type, controllers = decoder.decode(_controller_packet())
        self.assertEqual(packet_type, "controller")
        self.assertEqual(len(controllers), 2)
        self.assertEqual(controllers[0]["FrameId"], 7)

        packet_type, hand = decoder.decode(_hand_text_packet())
        self.assertEqual(packet_type, "hand_text")
        self.assertEqual(hand["side"], "L")
        self.assertEqual(len(hand["bones"]), 26)

        packet_type, hand = decoder.decode(_hand_binary_packet())
        self.assertEqual(packet_type, "hand_binary")
        self.assertEqual(hand["side"], "R")
        self.assertEqual(hand["frame_id"], 12)
        self.assertEqual(len(hand["bones"]), 26)

    def test_control_state_preserves_record_and_zoom_semantics(self) -> None:
        state = LegacyControlState(camera_num=2, initial_port=8000)
        decoder = LegacyVRPacketDecoder()
        self.assertTrue(state.consume_vr(_controller_packet(), decoder, receive_timestamp_ns=42))
        self.assertIsNotNone(state.data)
        self.assertFalse(state.hand_data)
        self.assertEqual(state.vr_input_timestamp_ns, 42)

        state.apply_record_control("Start")
        self.assertTrue(state.data_collecting_state)
        self.assertFalse(state.data_export_state)
        state.apply_record_control("Stop")
        self.assertFalse(state.data_collecting_state)
        self.assertTrue(state.data_export_state)
        state.apply_record_control("Undo")
        self.assertTrue(state.data_rollback_state)

        self.assertEqual(state.update_zoom("2,x1.5", key_mode="port"), [1])
        self.assertEqual(state.camera_zoom, [1.0, 1.5])
        self.assertEqual(state.update_zoom("0,2.0", key_mode="camera"), [0])
        self.assertEqual(state.camera_zoom, [2.0, 1.5])


class JpegAssemblerTests(unittest.TestCase):
    def test_bad_duplicate_expiry_and_restart(self) -> None:
        assembler = JpegChunkAssembler(frame_timeout_s=0.1)
        first = JpegDatagram(4, 0, 2, 4, b"ab").pack()
        second = JpegDatagram(4, 1, 2, 4, b"cd").pack()
        self.assertIsNone(assembler.feed(b"bad", now=0.0))
        self.assertIsNone(assembler.feed(first, now=0.0))
        self.assertIsNone(assembler.feed(first, now=0.01))
        self.assertEqual(assembler.expire(now=1.0), 1)
        self.assertIsNone(assembler.feed(second, now=1.1))

        assembler.reset()
        self.assertIsNone(assembler.feed(second, now=2.0))
        self.assertEqual(assembler.feed(first, now=2.01), b"abcd")
        self.assertIsNone(assembler.feed(first, now=2.02))
        self.assertGreaterEqual(assembler.stats["duplicates"], 2)
        self.assertEqual(assembler.stats["malformed"], 1)
        self.assertEqual(assembler.stats["expired"], 1)
        self.assertEqual(assembler.stats["restarts"], 1)


class SignalingEnvelopeTests(unittest.TestCase):
    def test_roundtrip_and_reject_non_object_payload(self) -> None:
        message = SignalingMessage("offer", "session-1", {"sdp": "v=0"})
        decoded = parse_signaling_message(message.to_json())
        self.assertEqual(decoded, message)
        with self.assertRaises(ValueError):
            parse_signaling_message('{"type":"offer","payload":[]}' )


if __name__ == "__main__":
    unittest.main()
