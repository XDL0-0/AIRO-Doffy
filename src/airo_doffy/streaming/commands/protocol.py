"""Strict versioned JSON protocol for low-rate reliable runtime commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ...core.errors import ModelValidationError
from ...core.events import RuntimeCommand
from ...core.types import ClockDomain

COMMAND_PROTOCOL_VERSION = 1
MAX_COMMAND_MESSAGE_BYTES = 65_536

_COMMAND_FIELDS = {
    "clock_domain",
    "command_id",
    "kind",
    "origin",
    "sequence",
    "source_timestamp_ns",
    "value",
}
_ACK_FIELDS = {
    "clock_domain",
    "command_id",
    "command_sequence",
    "duplicate",
    "message",
    "status",
    "timestamp_ns",
}
_ENVELOPE_FIELDS = {"message_type", "payload", "version"}


class CommandMessageType(str, Enum):
    """Reliable channel envelope kinds."""

    COMMAND = "command"
    ACK = "ack"


class CommandAckStatus(str, Enum):
    """Result reported by a reliable command receiver."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelValidationError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandAcknowledgement:
    """One explicit result for an idempotency-addressable command."""

    command_id: str
    command_sequence: int
    timestamp_ns: int
    status: CommandAckStatus
    message: str = ""
    duplicate: bool = False
    clock_domain: ClockDomain = ClockDomain.MONOTONIC

    def __post_init__(self) -> None:
        if not isinstance(self.command_id, str) or not self.command_id.strip():
            raise ModelValidationError("command_id must be a non-empty string")
        if not isinstance(self.message, str):
            raise ModelValidationError("acknowledgement message must be a string")
        if not isinstance(self.duplicate, bool):
            raise ModelValidationError("acknowledgement duplicate must be a boolean")
        try:
            status = CommandAckStatus(self.status)
            clock_domain = ClockDomain(self.clock_domain)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("invalid acknowledgement enum value") from exc
        object.__setattr__(
            self,
            "command_sequence",
            _non_negative_int(self.command_sequence, "command_sequence"),
        )
        object.__setattr__(
            self,
            "timestamp_ns",
            _non_negative_int(self.timestamp_ns, "timestamp_ns"),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "clock_domain", clock_domain)


ReliableMessage = RuntimeCommand | CommandAcknowledgement


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_message(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    try:
        raw = bytes(data)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("command message must support the bytes protocol") from exc
    if not raw or len(raw) > MAX_COMMAND_MESSAGE_BYTES:
        raise ModelValidationError(
            f"command message length must be within [1, {MAX_COMMAND_MESSAGE_BYTES}]"
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelValidationError("command message must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ModelValidationError("command message root must be an object")
    if set(value) != _ENVELOPE_FIELDS:
        raise ModelValidationError("command envelope fields do not match the protocol")
    if value["version"] != COMMAND_PROTOCOL_VERSION:
        raise ModelValidationError(
            f"unsupported command protocol version: {value['version']!r}"
        )
    if not isinstance(value["payload"], dict):
        raise ModelValidationError("command payload must be an object")
    return value


def _dump_message(message_type: CommandMessageType, payload: dict[str, Any]) -> bytes:
    data = json.dumps(
        {
            "message_type": message_type.value,
            "payload": payload,
            "version": COMMAND_PROTOCOL_VERSION,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(data) > MAX_COMMAND_MESSAGE_BYTES:
        raise ModelValidationError("encoded command message exceeds maximum size")
    return data


def encode_command(command: RuntimeCommand) -> bytes:
    """Serialize one typed runtime command into a canonical envelope."""

    if not isinstance(command, RuntimeCommand):
        raise ModelValidationError("command must be a RuntimeCommand")
    return _dump_message(
        CommandMessageType.COMMAND,
        {
            "clock_domain": command.clock_domain.value,
            "command_id": command.command_id,
            "kind": command.kind.value,
            "origin": command.origin,
            "sequence": command.sequence,
            "source_timestamp_ns": command.source_timestamp_ns,
            "value": command.value,
        },
    )


def decode_command(data: bytes | bytearray | memoryview) -> RuntimeCommand:
    """Strictly decode a runtime command envelope."""

    message = _load_message(data)
    if message["message_type"] != CommandMessageType.COMMAND.value:
        raise ModelValidationError("reliable message is not a command")
    payload = message["payload"]
    if set(payload) != _COMMAND_FIELDS:
        raise ModelValidationError("runtime command fields do not match the protocol")
    try:
        return RuntimeCommand(
            kind=payload["kind"],
            sequence=payload["sequence"],
            source_timestamp_ns=payload["source_timestamp_ns"],
            value=payload["value"],
            origin=payload["origin"],
            command_id=payload["command_id"],
            clock_domain=payload["clock_domain"],
        )
    except (TypeError, ValueError, ModelValidationError) as exc:
        raise ModelValidationError("runtime command payload is invalid") from exc


def encode_acknowledgement(acknowledgement: CommandAcknowledgement) -> bytes:
    """Serialize one command acknowledgement into a canonical envelope."""

    if not isinstance(acknowledgement, CommandAcknowledgement):
        raise ModelValidationError(
            "acknowledgement must be a CommandAcknowledgement"
        )
    return _dump_message(
        CommandMessageType.ACK,
        {
            "clock_domain": acknowledgement.clock_domain.value,
            "command_id": acknowledgement.command_id,
            "command_sequence": acknowledgement.command_sequence,
            "duplicate": acknowledgement.duplicate,
            "message": acknowledgement.message,
            "status": acknowledgement.status.value,
            "timestamp_ns": acknowledgement.timestamp_ns,
        },
    )


def decode_acknowledgement(
    data: bytes | bytearray | memoryview,
) -> CommandAcknowledgement:
    """Strictly decode a command acknowledgement envelope."""

    message = _load_message(data)
    if message["message_type"] != CommandMessageType.ACK.value:
        raise ModelValidationError("reliable message is not an acknowledgement")
    payload = message["payload"]
    if set(payload) != _ACK_FIELDS:
        raise ModelValidationError("acknowledgement fields do not match the protocol")
    try:
        return CommandAcknowledgement(
            command_id=payload["command_id"],
            command_sequence=payload["command_sequence"],
            timestamp_ns=payload["timestamp_ns"],
            status=payload["status"],
            message=payload["message"],
            duplicate=payload["duplicate"],
            clock_domain=payload["clock_domain"],
        )
    except (TypeError, ValueError, ModelValidationError) as exc:
        raise ModelValidationError("acknowledgement payload is invalid") from exc


def decode_reliable_message(
    data: bytes | bytearray | memoryview,
) -> ReliableMessage:
    """Decode either supported reliable message kind."""

    message = _load_message(data)
    message_type = message["message_type"]
    if message_type == CommandMessageType.COMMAND.value:
        return decode_command(data)
    if message_type == CommandMessageType.ACK.value:
        return decode_acknowledgement(data)
    raise ModelValidationError(f"unsupported reliable message type: {message_type!r}")
