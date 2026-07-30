"""Tests for the strict reliable command and acknowledgement protocol."""

from __future__ import annotations

import json
import unittest

from airo_doffy.core import (
    ClockDomain,
    ModelValidationError,
    RuntimeCommand,
    RuntimeCommandType,
)
from airo_doffy.streaming.commands import (
    COMMAND_PROTOCOL_VERSION,
    CommandAcknowledgement,
    CommandAckStatus,
    decode_acknowledgement,
    decode_command,
    decode_reliable_message,
    encode_acknowledgement,
    encode_command,
)


def command(
    *,
    command_id: str = "command-1",
    sequence: int = 7,
    kind: RuntimeCommandType = RuntimeCommandType.START_RECORDING,
    value: str | None = None,
) -> RuntimeCommand:
    return RuntimeCommand(
        kind=kind,
        sequence=sequence,
        source_timestamp_ns=123,
        value=value,
        origin="unity",
        command_id=command_id,
        clock_domain=ClockDomain.DEVICE,
    )


class CommandProtocolTest(unittest.TestCase):
    def test_command_round_trip_is_canonical(self) -> None:
        original = command()
        encoded = encode_command(original)
        self.assertEqual(encoded, encode_command(original))
        self.assertEqual(decode_command(encoded), original)
        self.assertEqual(decode_reliable_message(encoded), original)
        value = json.loads(encoded)
        self.assertEqual(value["version"], COMMAND_PROTOCOL_VERSION)
        self.assertEqual(value["message_type"], "command")

    def test_all_planned_command_kinds_have_explicit_enum_values(self) -> None:
        expected = {
            "start_recording",
            "stop_recording",
            "rollback_last_episode",
            "recalibrate_tactile",
            "reset_wrench_baseline",
            "set_teleop_mode",
            "set_camera_zoom",
            "set_camera_resolution",
            "safe_hold",
            "controlled_stop",
        }
        self.assertTrue(expected.issubset({kind.value for kind in RuntimeCommandType}))

    def test_acknowledgement_round_trip_for_every_status(self) -> None:
        for status in CommandAckStatus:
            with self.subTest(status=status):
                original = CommandAcknowledgement(
                    command_id="command-1",
                    command_sequence=7,
                    timestamp_ns=456,
                    status=status,
                    message="result",
                    duplicate=status is CommandAckStatus.REJECTED,
                )
                encoded = encode_acknowledgement(original)
                self.assertEqual(decode_acknowledgement(encoded), original)
                self.assertEqual(decode_reliable_message(encoded), original)

    def test_wrong_message_kind_is_rejected(self) -> None:
        ack = CommandAcknowledgement(
            command_id="command-1",
            command_sequence=7,
            timestamp_ns=456,
            status=CommandAckStatus.ACCEPTED,
        )
        with self.assertRaisesRegex(ModelValidationError, "not a command"):
            decode_command(encode_acknowledgement(ack))
        with self.assertRaisesRegex(ModelValidationError, "not an acknowledgement"):
            decode_acknowledgement(encode_command(command()))

    def test_unknown_missing_duplicate_and_invalid_fields_are_rejected(self) -> None:
        value = json.loads(encode_command(command()))
        mutations = []
        unknown_envelope = dict(value)
        unknown_envelope["extra"] = True
        mutations.append(unknown_envelope)
        wrong_version = dict(value)
        wrong_version["version"] = 99
        mutations.append(wrong_version)
        unknown_payload = json.loads(encode_command(command()))
        unknown_payload["payload"]["extra"] = True
        mutations.append(unknown_payload)
        missing_payload = json.loads(encode_command(command()))
        del missing_payload["payload"]["origin"]
        mutations.append(missing_payload)
        invalid_kind = json.loads(encode_command(command()))
        invalid_kind["payload"]["kind"] = "launch"
        mutations.append(invalid_kind)
        for malformed in mutations:
            with self.subTest(value=malformed):
                with self.assertRaises(ModelValidationError):
                    decode_command(json.dumps(malformed).encode())
        duplicate_json = (
            b'{"message_type":"command","message_type":"ack",'
            b'"payload":{},"version":1}'
        )
        with self.assertRaisesRegex(ModelValidationError, "duplicate JSON"):
            decode_reliable_message(duplicate_json)
        with self.assertRaises(ModelValidationError):
            decode_reliable_message(b"\xff")


if __name__ == "__main__":
    unittest.main()
