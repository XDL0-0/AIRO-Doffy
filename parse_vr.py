"""Compatibility exports for the pure v2 VR protocol decoders."""

from airo_doffy.devices.vr.protocol import (
    CONTROLLER_FIELDS_PER_HAND,
    HAND_BINARY_HEADER,
    HAND_BYTES_PER_BONE,
    HAND_JOINT_COUNT,
    HAND_TEXT_FIELDS,
    detect_packet_type,
    parse_data,
    parse_hand_data,
)

__all__ = [
    "CONTROLLER_FIELDS_PER_HAND",
    "HAND_BINARY_HEADER",
    "HAND_BYTES_PER_BONE",
    "HAND_JOINT_COUNT",
    "HAND_TEXT_FIELDS",
    "detect_packet_type",
    "parse_data",
    "parse_hand_data",
]
