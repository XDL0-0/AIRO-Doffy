"""VR input protocols, types, receivers, and test sources."""

from .base import VRInputSource
from .binary_v2 import (
    HEADER as BINARY_V2_HEADER,
    MAGIC as BINARY_V2_MAGIC,
    VERSION as BINARY_V2_VERSION,
    decode_vr_binary_v2,
    encode_vr_binary_v2,
)
from .protocol import (
    decode_vr_message,
    decode_vr_input,
    detect_packet_type,
    parse_data,
    parse_hand_data,
)

__all__ = [
    "VRInputSource",
    "BINARY_V2_HEADER",
    "BINARY_V2_MAGIC",
    "BINARY_V2_VERSION",
    "decode_vr_input",
    "decode_vr_binary_v2",
    "decode_vr_message",
    "detect_packet_type",
    "parse_data",
    "parse_hand_data",
    "encode_vr_binary_v2",
]
