"""Wire protocols with lazy compatibility exports.

Importing a standalone protocol does not load the optional dependencies used
by the legacy VR and JPEG protocols.
"""

from importlib import import_module

__all__ = [
    "HD_HEADER_FMT",
    "HD_HEADER_SIZE",
    "JpegChunkAssembler",
    "JpegChunkSender",
    "JpegDatagram",
    "LegacyControlState",
    "LegacyVRPacketDecoder",
    "RecordControl",
    "SignalingMessage",
    "parse_signaling_message",
    "JOINT_NAMES",
    "JOINT_IDS",
    "BodyJoint",
    "BodyFrame",
    "is_finger",
    "parse_body_packet",
]

_EXPORT_MODULES = {
    "HD_HEADER_FMT": ".jpeg",
    "HD_HEADER_SIZE": ".jpeg",
    "JpegChunkAssembler": ".jpeg",
    "JpegChunkSender": ".jpeg",
    "JpegDatagram": ".jpeg",
    "LegacyControlState": ".control",
    "LegacyVRPacketDecoder": ".vr",
    "RecordControl": ".control",
    "SignalingMessage": ".signaling",
    "parse_signaling_message": ".signaling",
    "JOINT_NAMES": ".body",
    "JOINT_IDS": ".body",
    "BodyJoint": ".body",
    "BodyFrame": ".body",
    "is_finger": ".body",
    "parse_body_packet": ".body",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
