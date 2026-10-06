"""Failure during construction releases already acquired camera/socket resources."""

import threading
from types import SimpleNamespace

import pytest

from doffy_teleop.config import Config
from doffy_teleop.media.udp import UDPManagerCore
from doffy_teleop.media.webrtc import WebRTCUDPManagerCore
from doffy_teleop.media.sockets import UDPSocketAllocator


class SocketFactory:
    def __init__(self, fail_at=None):
        self.calls = []
        self.created = []
        self.fail_at = fail_at

    def UdpComms(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_at:
            raise OSError("occupied UDP port")
        obj = SimpleNamespace(closed=False)
        obj.close = lambda: setattr(obj, "closed", True)
        self.created.append(obj)
        return obj


@pytest.mark.parametrize("camera_count,expected", [(0, 3), (2, 3), (4, 4), (7, 5)])
def test_allocator_preserves_control_ports_and_camera_limits(camera_count, expected):
    factory = SocketFactory()
    sockets, next_port = UDPSocketAllocator(factory).allocate(
        pc_ip="127.0.0.1", vr_ip="127.0.0.1", base_port=8000,
        camera_num=camera_count, tactile_enabled=False, tactile_port=8012,
    )
    assert len(sockets) == expected
    assert [entry["port_rx"] for entry in factory.calls] == list(range(8001, 8000 + expected * 2, 2))
    assert [entry["enable_rx"] for entry in factory.calls] == [True] * 3 + [False] * (expected - 3)
    assert next_port == 8000 + 2 * expected
    for sock in sockets.values():
        sock.close()


@pytest.mark.parametrize("manager_class", [UDPManagerCore, WebRTCUDPManagerCore])
def test_second_socket_bind_failure_closes_first_socket_and_camera(manager_class):
    camera = SimpleNamespace(
        _lock=threading.Lock(), camera_num=0, camera_list={}, camera_images={},
        camera_image_timestamps_ns={}, depth_images={}, depth_timestamps_ns={},
        depth_mode=False, realsense_resolution=(640, 480), realsense_fps=30,
        closed=False,
    )
    camera.close = lambda: setattr(camera, "closed", True)
    factory = SocketFactory(fail_at=2)
    with pytest.raises(OSError, match="occupied UDP port"):
        manager_class(Config(), camera_manager=camera, socket_module=factory)
    assert camera.closed
    assert len(factory.created) == 1
    assert factory.created[0].closed


def test_second_webrtc_camera_failure_releases_first_camera():
    first = SimpleNamespace(closed=False)
    first.close = lambda: setattr(first, "closed", True)
    factories = []

    def camera_factory(**kwargs):
        factories.append(kwargs)
        if len(factories) == 2:
            raise RuntimeError("camera unavailable")
        return first

    with pytest.raises(RuntimeError, match="camera unavailable"):
        WebRTCUDPManagerCore(
            Config(), camera_detector=lambda: ["cameraA", "cameraB"],
            camera_factory=camera_factory, socket_module=SocketFactory(),
        )
    assert first.closed
