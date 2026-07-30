"""Camera source interfaces and adapters."""

from .base import CameraSource, DepthCameraSource
from .realsense import (
    RealSenseCameraSource,
    RealSenseDevice,
    create_realsense_camera,
    discover_realsense_devices,
)

__all__ = [
    "CameraSource",
    "DepthCameraSource",
    "RealSenseCameraSource",
    "RealSenseDevice",
    "create_realsense_camera",
    "discover_realsense_devices",
]
