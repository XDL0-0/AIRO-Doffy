"""Compatibility dispatch for the controller and hand packet families.

The original packet grammar lives in :mod:`parse_vr` and is still the source
of truth for v1 compatibility.  This small object makes the parser injectable
so both media managers share exactly one receive path while tests can patch
the old root symbols.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import doffy_teleop.protocol.parse_vr as parse_vr


class LegacyVRPacketDecoder:
    """Decode classic ``C/H/HB`` packets through injectable v1 functions."""

    def __init__(
        self,
        detect: Callable[[str], str] | None = None,
        parse_controller: Callable[[str | None], list[dict] | None] | None = None,
        parse_hand: Callable[[str | None], dict | None] | None = None,
    ) -> None:
        self.detect = detect or parse_vr.detect_packet_type
        self.parse_controller = parse_controller or parse_vr.parse_data
        self.parse_hand = parse_hand or parse_vr.parse_hand_data

    @staticmethod
    def normalize(raw: bytes | bytearray | memoryview | str) -> str:
        if isinstance(raw, str):
            return raw
        return bytes(raw).decode("utf-8", errors="replace")

    def decode(self, raw: bytes | bytearray | memoryview | str) -> tuple[str, Any] | None:
        """Return ``(packet_type, parsed_dict)`` or ``None`` for bad packets."""

        text = self.normalize(raw)
        packet_type = self.detect(text)
        if packet_type == "controller":
            parsed = self.parse_controller(text)
            return (packet_type, parsed) if parsed else None
        if packet_type in {"hand_text", "hand_binary"}:
            parsed = self.parse_hand(text)
            return (packet_type, parsed) if parsed else None
        return None
