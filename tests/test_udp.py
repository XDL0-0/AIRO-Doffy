from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

from config import Config
from udp import UDPManager


class FakeRawSocket:
    def __init__(self) -> None:
        self.packets = []

    def sendto(self, packet, target) -> None:
        self.packets.append((packet, target))


class FakeUdpComms:
    def __init__(self, **kwargs) -> None:
        self.send_ip = kwargs["send_ip"]
        self.udp_send_port = kwargs["port_tx"]
        self._sock = FakeRawSocket()
        self.received = []
        self.closed = False

    def read(self):
        return self.received.pop(0) if self.received else None

    def read_all(self):
        packets = list(self.received)
        self.received.clear()
        return packets

    def send(self, data) -> None:
        self._sock.sendto(data, (self.send_ip, self.udp_send_port))

    def close(self) -> None:
        self.closed = True


class FakeCameraManager:
    def __init__(self, camera_num=2) -> None:
        self._lock = threading.Lock()
        self.camera_num = camera_num
        self.camera_images = {
            f"camera_{idx}": np.full((8, 10, 3), idx, dtype=np.uint8)
            for idx in range(camera_num)
        }
        self.camera_image_timestamps_ns = {
            f"camera_{idx}": idx + 1 for idx in range(camera_num)
        }
        self.depth_mode = False
        self.depth_images = {}
        self.depth_timestamps_ns = {}
        self.realsense_resolution = (10, 8)
        self.realsense_fps = 30
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


class UDPManagerTests(unittest.TestCase):
    def make_manager(self, camera_num=2):
        cameras = FakeCameraManager(camera_num=camera_num)
        with patch("udp.U.UdpComms", side_effect=FakeUdpComms):
            manager = UDPManager(Config(), camera_manager=cameras)
        return manager, cameras

    def test_composes_camera_buffers_without_owning_camera_devices(self) -> None:
        manager, cameras = self.make_manager(camera_num=2)

        self.assertIs(manager.camera_images, cameras.camera_images)
        self.assertIs(
            manager.camera_image_timestamps_ns,
            cameras.camera_image_timestamps_ns,
        )
        self.assertIs(manager.camera_data, cameras.camera_images)
        self.assertEqual(manager.camera_num, 2)
        self.assertFalse(hasattr(manager, "camera_list"))
        self.assertEqual(set(manager.socket_list), {"socket_0", "socket_1", "socket_2"})
        manager.close()
        self.assertTrue(cameras.closed)
        self.assertTrue(all(sock.closed for sock in manager.socket_list.values()))

    def test_connection_cycle_starts_camera_and_streams_current_frames(self) -> None:
        manager, cameras = self.make_manager(camera_num=1)

        manager.send_and_receive_data()

        self.assertTrue(cameras.started)
        self.assertTrue(manager.socket_list["socket_0"]._sock.packets)
        manager.close()

    def test_more_than_five_cameras_remain_local(self) -> None:
        manager, cameras = self.make_manager(camera_num=7)

        self.assertEqual(manager.camera_num, 7)
        self.assertEqual(len(cameras.camera_images), 7)
        self.assertEqual(
            {name for name in manager.socket_list if name.startswith("socket_")},
            {f"socket_{idx}" for idx in range(5)},
        )
        manager.close()

    def test_socket_initialization_failure_releases_cameras(self) -> None:
        cameras = FakeCameraManager(camera_num=1)
        with (
            patch("udp.U.UdpComms", side_effect=OSError("bind failed")),
            self.assertRaisesRegex(OSError, "bind failed"),
        ):
            UDPManager(Config(), camera_manager=cameras)

        self.assertTrue(cameras.closed)


if __name__ == "__main__":
    unittest.main()
