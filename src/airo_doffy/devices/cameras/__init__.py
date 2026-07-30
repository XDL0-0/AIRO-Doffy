"""Camera source interfaces and adapters."""

from .base import CameraSource, DepthCameraSource
from .mock import CameraMockMode, MockCameraSource, create_mock_camera
from .realsense import (
    RealSenseCameraSource,
    RealSenseDevice,
    create_realsense_camera,
    discover_realsense_devices,
)

__all__ = [
    "CameraSource",
    "CameraMockMode",
    "DepthCameraSource",
    "MockCameraSource",
    "RealSenseCameraSource",
    "RealSenseDevice",
    "create_realsense_camera",
    "create_mock_camera",
    "discover_realsense_devices",
]
