"""Minimal JSON signaling envelope shared by WebRTC server and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class SignalingMessage:
    type: str
    session_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.type).strip():
            raise ValueError("signaling message type cannot be empty")
        if not isinstance(self.payload, dict):
            raise TypeError("signaling payload must be a JSON object")

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "payload": dict(self.payload),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SignalingMessage":
        message_type = value.get("type")
        if not isinstance(message_type, str):
            raise ValueError("signaling message requires a string type")
        session_id = value.get("session_id", "")
        if not isinstance(session_id, str):
            session_id = str(session_id)
        payload = value.get("payload", {})
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("signaling payload must be a JSON object")
        return cls(message_type, session_id, dict(payload))


def parse_signaling_message(value: str | bytes | Mapping[str, Any]) -> SignalingMessage:
    if isinstance(value, (str, bytes)):
        decoded = json.loads(value)
    else:
        decoded = value
    if not isinstance(decoded, Mapping):
        raise ValueError("signaling message must be a JSON object")
    return SignalingMessage.from_mapping(decoded)
