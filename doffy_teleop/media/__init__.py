"""Composable media services with lazy compatibility exports."""

from importlib import import_module

__all__ = [
    "CameraCaptureService",
    "CameraFrameProvider",
    "CameraVideoTrack",
    "FrameStore",
    "UDPManagerCore",
    "WebRTCSession",
    "WebRTCSignalingServer",
    "WebRTCUDPManagerCore",
    "center_zoom",
    "prepare_rgb_frame",
    "BodyReceiver",
]

_EXPORT_MODULES = {
    "CameraCaptureService": ".camera",
    "CameraFrameProvider": ".camera",
    "CameraVideoTrack": ".camera",
    "FrameStore": ".camera",
    "UDPManagerCore": ".udp",
    "WebRTCSession": ".webrtc_peer",
    "WebRTCSignalingServer": ".webrtc_signaling",
    "WebRTCUDPManagerCore": ".webrtc",
    "center_zoom": ".frames",
    "prepare_rgb_frame": ".frames",
    "BodyReceiver": ".body",
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
