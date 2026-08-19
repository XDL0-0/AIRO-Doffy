"""UDP video/control transport composed with local RealSense capture."""

from __future__ import annotations

import struct
import threading
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

import udp_comms as U
import utils
from config import Config
from parse_vr import detect_packet_type, parse_data, parse_hand_data
from realsense_camera import RealSenseCameraManager


MAX_STREAM_CAMERAS = 5
STREAM_FPS = 30
TACTILE_FPS = 100
VR_RECEIVE_HZ = 100
CONNECTION_TIMEOUT = 30.0

# HD chunk protocol header: frame id, chunk index, total chunks, total bytes.
HD_HEADER_FMT = "!IHHI"
HD_HEADER_SIZE = 12


class UDPManager:
    """VR UDP transport that consumes frames from a camera manager."""

    def __init__(
        self,
        config: Config | None = None,
        camera_manager: RealSenseCameraManager | None = None,
    ) -> None:
        cfg = Config() if config is None else config
        self.running = True
        self.pc_ip = cfg.PC_IP
        self.vr_ip = cfg.VR_IP
        self.ip_port = cfg.IP_PORT
        self.initial_port = cfg.IP_PORT
        self.tactile_transfer_status = cfg.TACTILE_TRANSFER
        self.tactile_port = cfg.TACTILE_PORT
        self.tracking_mode = cfg.TRACKING_MODE
        self.jpeg_quality = cfg.JPEG_QUALITY
        self.hd_chunk_size = cfg.HD_CHUNK_SIZE

        self.camera_manager = (
            RealSenseCameraManager(config=cfg)
            if camera_manager is None
            else camera_manager
        )
        self._lock = self.camera_manager._lock
        self.camera_num = self.camera_manager.camera_num
        self.camera_images = self.camera_manager.camera_images
        self.camera_image_timestamps_ns = (
            self.camera_manager.camera_image_timestamps_ns
        )
        # Existing collection/visualizer callers use these names for raw capture.
        self.camera_data = self.camera_images
        self.camera_data_timestamps_ns = self.camera_image_timestamps_ns
        self.depth_mode = self.camera_manager.depth_mode
        self.depth_images = self.camera_manager.depth_images
        self.depth_timestamps_ns = self.camera_manager.depth_timestamps_ns
        self.realsense_resolution = self.camera_manager.realsense_resolution
        self.realsense_fps = self.camera_manager.realsense_fps

        self.data: list[dict] | None = None
        self.hand_data: Dict[str, dict] = {}
        self.data_collecting_state = False
        self.data_export_state = False
        self.data_rollback_state = False
        self.tactile_byte: bytes | None = None
        self.tactile_data: np.ndarray | None = None
        self.tactile_timestamp_ns = 0
        self.vr_input_timestamp_ns = 0

        self.camera_zoom: List[float] = [1.0] * self.camera_num
        self._frame_counters: Dict[int, int] = {}
        try:
            self.socket_list = self._create_sockets()
        except Exception:
            self.camera_manager.close()
            raise
        self.threads: List[threading.Thread] = []

    def _alloc_socket(
        self,
        idx: int,
        enable_rx: bool,
        socket_list: Dict[str, U.UdpComms],
    ) -> None:
        name = "socket_tactile" if idx == -1 else f"socket_{idx}"
        socket_list[name] = U.UdpComms(
            udp_ip=self.pc_ip,
            send_ip=self.vr_ip,
            port_tx=self.ip_port,
            port_rx=self.ip_port + 1,
            enable_rx=enable_rx,
            suppress_warnings=True,
        )
        utils.logger.info(
            f"{name}: TX={self.ip_port}, RX={self.ip_port + 1}, enableRX={enable_rx}"
        )
        self.ip_port += 2

    def _create_sockets(self) -> Dict[str, U.UdpComms]:
        socket_list: Dict[str, U.UdpComms] = {}
        utils.logger.info(
            f"PC IP: {self.pc_ip}, VR IP: {self.vr_ip}, base_port={self.ip_port}"
        )

        stream_count = min(self.camera_num, MAX_STREAM_CAMERAS)
        if self.camera_num > MAX_STREAM_CAMERAS:
            utils.logger.warning(
                f"UDP streams only the first {MAX_STREAM_CAMERAS} cameras; "
                f"all {self.camera_num} remain available locally."
            )
        for idx in range(stream_count):
            self._alloc_socket(idx, idx < 3, socket_list)

        minimum_sockets = 3
        for idx in range(stream_count, max(minimum_sockets, stream_count)):
            self._alloc_socket(idx, True, socket_list)

        if self.tactile_transfer_status:
            self.ip_port = self.tactile_port
            self._alloc_socket(-1, False, socket_list)

        utils.logger.info(f"{len(socket_list)} UDP sockets created.")
        return socket_list

    def send_hd_frame(
        self,
        sock: U.UdpComms,
        frame_bgr: np.ndarray,
        cam_idx: int = 0,
        quality: int | None = None,
    ) -> int:
        """JPEG-encode and send one frame with the chunked HD protocol."""
        if quality is None:
            quality = self.jpeg_quality
        ok, buffer = cv2.imencode(
            ".jpg",
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            return 0

        image_bytes = buffer.tobytes()
        total_bytes = len(image_bytes)
        num_chunks = (total_bytes + self.hd_chunk_size - 1) // self.hd_chunk_size
        counter = self._frame_counters.get(cam_idx, 0)
        frame_id = counter & 0xFFFFFFFF
        self._frame_counters[cam_idx] = counter + 1
        target = (sock.send_ip, sock.udp_send_port)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.hd_chunk_size
            end = min(start + self.hd_chunk_size, total_bytes)
            header = struct.pack(
                HD_HEADER_FMT,
                frame_id,
                chunk_idx,
                num_chunks,
                total_bytes & 0xFFFFFFFF,
            )
            sock._sock.sendto(header + image_bytes[start:end], target)
        return num_chunks

    @staticmethod
    def center_zoom(
        image: np.ndarray,
        scale: float = 1.5,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        new_width, new_height = int(width * scale), int(height * scale)
        if new_width <= width or new_height <= height:
            return cv2.resize(image, (width, height), interpolation=interpolation)
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        return resized[start_y : start_y + height, start_x : start_x + width]

    def data_process(
        self,
        frame: np.ndarray,
        cam_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare a captured RGB frame for UDP streaming."""
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        zoom = self.camera_zoom[cam_idx] if cam_idx < len(self.camera_zoom) else 1.0
        return self.center_zoom(frame_bgr, zoom), frame

    def _parse_resolution_control(self, value: str) -> None:
        value = value.strip(";")
        if not value:
            return
        for item in value.split(";"):
            if not item:
                continue
            key, raw_zoom = map(str.strip, item.split(",", 1))
            if not key.isdigit():
                continue
            cam_idx = (int(key) % self.initial_port) // 2
            if cam_idx >= self.camera_num:
                utils.logger.warning(
                    f"Invalid camera index {cam_idx}, total={self.camera_num}"
                )
                continue
            zoom = float(raw_zoom[1:])
            if not np.isclose(self.camera_zoom[cam_idx], zoom):
                utils.logger.info(
                    f"camera{cam_idx}: zoom {self.camera_zoom[cam_idx]:.2f} -> {zoom:.2f}"
                )
                self.camera_zoom[cam_idx] = zoom

    def is_movement_exist(self) -> bool:
        with self._lock:
            data = self.data
        if data is None:
            return False
        return bool(data[1]["GripTrigger"]) or abs(data[1]["Joystick"][1]) > 0.7

    def _consume_vr_packet(self, raw_data: bytes | str) -> None:
        packet_type = detect_packet_type(raw_data)
        if packet_type == "controller":
            parsed = parse_data(raw_data)
            if parsed:
                with self._lock:
                    self.data = parsed
                    self.hand_data.clear()
                    self.vr_input_timestamp_ns = time.monotonic_ns()
        elif packet_type in ("hand_text", "hand_binary"):
            hand = parse_hand_data(raw_data)
            if hand:
                with self._lock:
                    self.hand_data[hand["side"]] = hand
                    self.data = None
                    self.vr_input_timestamp_ns = time.monotonic_ns()

    def _camera_send_thread(self, sock: U.UdpComms, idx: int) -> None:
        utils.logger.info(f"TX camera thread {idx} starts!")
        stats_started = time.time()
        frame_count = 0
        chunk_count = 0
        while self.running:
            try:
                with self._lock:
                    frame = self.camera_images.get(f"camera_{idx}")
                if frame is not None:
                    frame_bgr, _ = self.data_process(frame, idx)
                    chunk_count += self.send_hd_frame(sock, frame_bgr, cam_idx=idx)
                    frame_count += 1
                    elapsed = time.time() - stats_started
                    if elapsed >= 5.0:
                        utils.logger.info(
                            f"TX camera_{idx}: {frame_count} frames in {elapsed:.1f}s "
                            f"({frame_count / elapsed:.1f} fps), {chunk_count} chunks sent"
                        )
                        stats_started = time.time()
                        frame_count = 0
                        chunk_count = 0
            except OSError as exc:
                utils.logger.error(f"TX camera_{idx} OSError: {exc}")
                break
            time.sleep(1 / STREAM_FPS)

    def _tactile_send_thread(self, sock: U.UdpComms) -> None:
        while self.running:
            with self._lock:
                tactile_bytes = self.tactile_byte
            if tactile_bytes is not None:
                sock.send(tactile_bytes)
            time.sleep(1 / TACTILE_FPS)

    def _vr_receive_thread(self) -> None:
        stale = self.socket_list["socket_1"].read()
        if stale:
            utils.logger.warning(f"Flushed stale record control on startup: '{stale}'")
        target_dt = 1.0 / VR_RECEIVE_HZ
        while self.running:
            started = time.time()
            try:
                for raw_data in self.socket_list["socket_0"].read_all():
                    self._consume_vr_packet(raw_data)

                record_control = self.socket_list["socket_1"].read()
                resolution_control = self.socket_list["socket_2"].read()
                if resolution_control:
                    self._parse_resolution_control(resolution_control)
                if record_control:
                    with self._lock:
                        if record_control == "Start":
                            self.data_collecting_state = True
                            self.data_export_state = False
                            self.data_rollback_state = False
                        elif record_control == "Stop":
                            self.data_collecting_state = False
                            self.data_export_state = True
                        elif record_control in ("Undo", "Rollback", "DeleteLast"):
                            self.data_collecting_state = False
                            self.data_export_state = False
                            self.data_rollback_state = True
            except Exception as exc:
                utils.logger.error(f"Error in VR receive thread: {exc}")
                time.sleep(0.1)

            elapsed = time.time() - started
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    def send_and_receive_data(self) -> None:
        """Exchange one initial-connect cycle using current camera frames."""
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
        """Block until VR sends initial data, or raise after the timeout."""
        printed = False
        started = time.time()
        while True:
            self.send_and_receive_data()
            with self._lock:
                data = self.data
                has_hand = bool(self.hand_data)
            if data is not None or has_hand:
                utils.logger.info("Data received! VR connected!")
                if data is not None:
                    return data
                return [_dummy_controller("LTouch"), _dummy_controller("RTouch")]
            if not printed:
                utils.logger.info("Connecting VR...")
                printed = True
            if time.time() - started > CONNECTION_TIMEOUT:
                raise TimeoutError(f"VR did not respond within {CONNECTION_TIMEOUT}s")
            time.sleep(0.05)

    def start_comms_threads(self) -> None:
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
            tactile = threading.Thread(
                target=self._tactile_send_thread,
                args=(self.socket_list["socket_tactile"],),
                daemon=True,
            )
            tactile.start()
            self.threads.append(tactile)

    def close(self) -> None:
        utils.logger.info("Stopping UDPManager...")
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.camera_manager.close()
        for name, sock in self.socket_list.items():
            try:
                sock.close()
            except Exception as exc:
                utils.logger.warning(f"Error closing {name}: {exc}")
        utils.logger.info("UDPManager resources released.")


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
