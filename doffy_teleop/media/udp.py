"""Legacy UDP media manager composed from camera, protocol, and control parts."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any

import cv2
import numpy as np

import doffy_teleop.media.udp_comms as U
import doffy_teleop.utils as utils

from doffy_teleop.protocol.control import LegacyControlState
from doffy_teleop.protocol.jpeg import HD_HEADER_FMT, HD_HEADER_SIZE, JpegChunkSender
from doffy_teleop.protocol.vr import LegacyVRPacketDecoder
from .frames import center_zoom, prepare_rgb_frame
from .sockets import UDPSocketAllocator


MAX_STREAM_CAMERAS = 5
STREAM_FPS = 30
TACTILE_FPS = 100
VR_RECEIVE_HZ = 100
CONNECTION_TIMEOUT = 30.0


class UDPManagerCore:
    """Compatibility implementation for the old ``udp.UDPManager`` API.

    The class intentionally owns only socket/stream orchestration.  JPEG
    packetization, VR parsing, record controls, and frame preparation live in
    independent modules so WebRTC can reuse them.
    """

    def __init__(
        self,
        config: Any,
        *,
        camera_manager: Any | None = None,
        camera_factory: Callable[[Any], Any] | None = None,
        socket_module: Any = U,
        decoder: LegacyVRPacketDecoder | None = None,
        logger: Any = utils.logger,
    ) -> None:
        self._logger = logger
        cfg = config
        self.running = True
        self._closed = False
        self.pc_ip = cfg.PC_IP
        self.vr_ip = cfg.VR_IP
        self.ip_port = cfg.IP_PORT
        self.initial_port = cfg.IP_PORT
        self.tactile_transfer_status = cfg.TACTILE_TRANSFER
        self.tactile_port = cfg.TACTILE_PORT
        self.tracking_mode = cfg.TRACKING_MODE
        self.jpeg_quality = cfg.JPEG_QUALITY
        self.hd_chunk_size = cfg.HD_CHUNK_SIZE
        self._socket_module = socket_module
        self._decoder = decoder or LegacyVRPacketDecoder()
        self._jpeg_sender = JpegChunkSender(
            chunk_size=self.hd_chunk_size,
            quality=self.jpeg_quality,
        )
        # Keep the old private counter visible to callers that reset frame ids
        # at a transport restart.  The sender owns the same dictionary.
        self._frame_counters = self._jpeg_sender._counters

        if camera_manager is None:
            if camera_factory is None:
                from doffy_teleop.media.realsense_camera import RealSenseCameraManager

                camera_factory = lambda value: RealSenseCameraManager(config=value)
            camera_manager = camera_factory(cfg)
        self.camera_manager = camera_manager
        self._lock = getattr(camera_manager, "_lock", threading.RLock())
        self.camera_num = int(camera_manager.camera_num)
        self.camera_images = camera_manager.camera_images
        self.camera_image_timestamps_ns = camera_manager.camera_image_timestamps_ns
        self.camera_data = self.camera_images
        self.camera_data_timestamps_ns = self.camera_image_timestamps_ns
        self.depth_mode = camera_manager.depth_mode
        self.depth_images = camera_manager.depth_images
        self.depth_timestamps_ns = camera_manager.depth_timestamps_ns
        self.realsense_resolution = camera_manager.realsense_resolution
        self.realsense_fps = camera_manager.realsense_fps

        self.control_state = LegacyControlState(
            self.camera_num,
            self.initial_port,
            lock=self._lock,
        )
        self.camera_zoom = self.control_state.camera_zoom
        try:
            self.socket_list = self._create_sockets()
        except BaseException:
            self.camera_manager.close()
            raise
        self.threads: list[threading.Thread] = []

    # State attributes remain writable for the recording service and Classic runtime.
    def _state_property(name: str):
        def get(self):
            return getattr(self.control_state, name)

        def set_(self, value):
            with self._lock:
                setattr(self.control_state, name, value)

        return property(get, set_)

    data = _state_property("data")
    hand_data = _state_property("hand_data")
    data_collecting_state = _state_property("data_collecting_state")
    data_export_state = _state_property("data_export_state")
    data_rollback_state = _state_property("data_rollback_state")
    tactile_byte = _state_property("tactile_byte")
    tactile_data = _state_property("tactile_data")
    tactile_timestamp_ns = _state_property("tactile_timestamp_ns")
    vr_input_timestamp_ns = _state_property("vr_input_timestamp_ns")

    @property
    def camera_zoom(self) -> list[float]:
        return self.control_state.camera_zoom

    @camera_zoom.setter
    def camera_zoom(self, value: list[float]) -> None:
        with self._lock:
            self.control_state.camera_zoom = value

    def _create_sockets(self) -> dict[str, Any]:
        self._logger.info(
            f"PC IP: {self.pc_ip}, VR IP: {self.vr_ip}, base_port={self.ip_port}"
        )
        if self.camera_num > MAX_STREAM_CAMERAS:
            self._logger.warning(
                f"UDP streams only the first {MAX_STREAM_CAMERAS} cameras; "
                f"all {self.camera_num} remain available locally."
            )
        sockets, self.ip_port = UDPSocketAllocator(self._socket_module, self._logger).allocate(
            pc_ip=self.pc_ip, vr_ip=self.vr_ip, base_port=self.ip_port,
            camera_num=self.camera_num, tactile_enabled=self.tactile_transfer_status,
            tactile_port=self.tactile_port,
        )
        return sockets

    def send_hd_frame(
        self,
        sock: Any,
        frame_bgr: np.ndarray,
        cam_idx: int = 0,
        quality: int | None = None,
    ) -> int:
        """JPEG encode and send the frozen v1 chunk protocol."""

        try:
            _frame_id, packets = self._jpeg_sender.packets(
                frame_bgr,
                cam_idx=cam_idx,
                quality=quality,
            )
            target = (sock.send_ip, sock.udp_send_port)
            raw_socket = getattr(sock, "_sock", sock)
            for packet in packets:
                raw_socket.sendto(packet, target)
            return len(packets)
        except (ValueError, cv2.error) as exc:
            self._logger.warning(f"JPEG frame encode/send failed: {exc}")
            return 0

    @staticmethod
    def center_zoom(image: np.ndarray, scale: float = 1.5, interpolation: int = 1) -> np.ndarray:
        return center_zoom(image, scale, interpolation)

    def data_process(self, frame: np.ndarray, cam_idx: int) -> tuple[np.ndarray, np.ndarray]:
        zoom = self.camera_zoom[cam_idx] if cam_idx < len(self.camera_zoom) else 1.0
        return prepare_rgb_frame(frame, zoom=zoom)

    def _parse_resolution_control(self, value: str | bytes) -> None:
        self.control_state.update_zoom(
            value,
            key_mode="port",
            logger=self._logger.warning,
        )

    def is_movement_exist(self) -> bool:
        return self.control_state.movement_exists()

    def _consume_vr_packet(self, raw_data: bytes | str) -> None:
        self.control_state.consume_vr(raw_data, self._decoder)

    def _apply_record_control(self, value: bytes | str | None) -> None:
        self.control_state.apply_record_control(value)

    def _camera_send_thread(self, sock: Any, idx: int) -> None:
        self._logger.info(f"TX camera thread {idx} starts!")
        frames = 0
        chunks = 0
        started = time.time()
        while self.running:
            try:
                with self._lock:
                    frame = self.camera_images.get(f"camera_{idx}")
                if frame is not None:
                    frame_bgr, _ = self.data_process(frame, idx)
                    chunks += self.send_hd_frame(sock, frame_bgr, cam_idx=idx)
                    frames += 1
                    elapsed = time.time() - started
                    if elapsed >= 5.0:
                        self._logger.info(
                            f"TX camera_{idx}: {frames} frames in {elapsed:.1f}s "
                            f"({frames / elapsed:.1f} fps), {chunks} chunks sent"
                        )
                        started = time.time()
                        frames = chunks = 0
            except OSError as exc:
                self._logger.error(f"TX camera_{idx} OSError: {exc}")
                break
            time.sleep(1.0 / STREAM_FPS)

    def _tactile_send_thread(self, sock: Any) -> None:
        while self.running:
            tactile_bytes = self.tactile_byte
            if tactile_bytes is not None:
                sock.send(tactile_bytes)
            time.sleep(1.0 / TACTILE_FPS)

    def _vr_receive_thread(self) -> None:
        target_dt = 1.0 / VR_RECEIVE_HZ
        while self.running:
            started = time.time()
            try:
                for raw_data in self.socket_list["socket_0"].read_all():
                    self._consume_vr_packet(raw_data)
                resolution_control = self.socket_list["socket_2"].read()
                if resolution_control:
                    self._parse_resolution_control(resolution_control)
                for record_control in self.socket_list["socket_1"].read_all():
                    self._apply_record_control(record_control)
            except Exception as exc:
                self._logger.error(f"Error in VR receive thread: {exc}")
                time.sleep(0.1)
            elapsed = time.time() - started
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    def send_and_receive_data(self) -> None:
        self.camera_manager.start()
        with self._lock:
            frames = dict(self.camera_images)
        for idx in range(min(self.camera_num, MAX_STREAM_CAMERAS)):
            frame = frames.get(f"camera_{idx}")
            sock = self.socket_list.get(f"socket_{idx}")
            if frame is not None and sock is not None:
                frame_bgr, _ = self.data_process(frame, idx)
                self.send_hd_frame(sock, frame_bgr, cam_idx=idx)
        for raw_data in self.socket_list["socket_0"].read_all():
            self._consume_vr_packet(raw_data)

    def test_connection(self) -> list[dict]:
        started = time.time()
        printed = False
        while True:
            self.send_and_receive_data()
            with self._lock:
                data = self.data
                has_hand = bool(self.hand_data)
            if data is not None or has_hand:
                if data is not None:
                    return data
                return [_dummy_controller("LTouch"), _dummy_controller("RTouch")]
            if not printed:
                self._logger.info("Connecting VR...")
                printed = True
            if time.time() - started > CONNECTION_TIMEOUT:
                raise TimeoutError(f"VR did not respond within {CONNECTION_TIMEOUT}s")
            time.sleep(0.05)

    def start_comms_threads(self) -> None:
        if self._closed:
            raise RuntimeError("cannot restart a closed UDP manager")
        self.running = True
        self.camera_manager.start()
        self.threads = []
        for idx in range(min(self.camera_num, MAX_STREAM_CAMERAS)):
            thread = threading.Thread(
                target=self._camera_send_thread,
                args=(self.socket_list[f"socket_{idx}"], idx),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)
        receiver = threading.Thread(target=self._vr_receive_thread, daemon=True)
        receiver.start()
        self.threads.append(receiver)
        if self.tactile_transfer_status:
            thread = threading.Thread(
                target=self._tactile_send_thread,
                args=(self.socket_list["socket_tactile"],),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def close(self) -> None:
        if self._closed:
            return
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        try:
            self.camera_manager.close()
        finally:
            for name, sock in self.socket_list.items():
                try:
                    sock.close()
                except Exception as exc:
                    self._logger.warning(f"Error closing {name}: {exc}")
            self._closed = True


def _dummy_controller(name: str) -> dict:
    return {
        "ControllerType": name,
        "Timestamp": 0,
        "Position": (0.0, 0.0, 0.0),
        "Rotation": (0.0, 0.0, 0.0, 1.0),
        "Joystick": (0.0, 0.0),
        "IndexTrigger": 0.0,
        "GripTrigger": 0.0,
        "Button_AX": 0,
        "Button_BY": 0,
        "Joystick_Press": 0,
    }
