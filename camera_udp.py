"""Camera capture + UDP streaming to VR headset, and VR data reception.

Supports HD chunked image transfer (matching UdpSocketMultiHD.cs) and
both controller / hand-tracking data from the VR headset.
"""

from __future__ import annotations

import struct
import cv2
import time
import threading
import numpy as np
import pyrealsense2 as rs
from typing import Dict, List, Tuple

import utils
import udp_comms as U
from config import Config
from parse_vr import parse_data, parse_hand_data, detect_packet_type
from airo_camera_toolkit.cameras.realsense.realsense import Realsense

MAX_CAMERAS = 5
STREAM_FPS = 30
TACTILE_FPS = 100
VR_RECEIVE_HZ = 100
CONNECTION_TIMEOUT = 30.0

# HD chunk protocol header: !IHHI = uint32 + uint16 + uint16 + uint32 = 12 bytes
HD_HEADER_FMT = "!IHHI"
HD_HEADER_SIZE = 12


class CameraUDPManager:
    def __init__(self):
        cfg = Config()
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
        self.realsense_resolution = cfg.REALSENSE_RESOLUTION
        self.realsense_fps = cfg.REALSENSE_FPS

        self._lock = threading.Lock()
        self.data: list[dict] | None = None
        self.hand_data: Dict[str, dict] = {}   # {"L": {...}, "R": {...}}
        self.fine_mode: str | None = None
        self.data_collecting_state = False
        self.data_export_state = False
        self.data_rollback_state = False
        self.camera_images: Dict[str, np.ndarray] = {}
        self.camera_image_timestamps_ns: Dict[str, int] = {}
        self.camera_data: Dict[str, np.ndarray] = {}
        self.camera_data_timestamps_ns: Dict[str, int] = {}
        self.depth_mode = cfg.DEPTH_INFO_ENABLE
        self.depth_images: Dict[str, np.ndarray] = {}
        self.depth_timestamps_ns: Dict[str, int] = {}
        self.tactile_byte: bytes | None = None
        self.tactile_data: np.ndarray | None = None
        self.tactile_timestamp_ns: int = 0
        self.vr_input_timestamp_ns: int = 0

        self._frame_counters: Dict[int, int] = {}

        self.camera_num, self.camera_series_num = self._detect_cameras()
        self.camera_zoom: List[float] = [1.0] * self.camera_num
        self.socket_list, self.camera_list = self._create_udp_and_cameras()
        self.threads: List[threading.Thread] = []

    # ── Camera detection ──────────────────────────────────────────────────

    def _detect_cameras(self) -> Tuple[int, List[str]]:
        context = rs.context()
        devices = context.query_devices()
        serials = []

        if len(devices) == 0:
            utils.logger.warning("No Realsense connected.")
        else:
            for i, device in enumerate(devices):
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                utils.logger.info(f"Camera {i}: {name}, serial={serial}")
                serials.append(serial)

        return len(devices), serials

    # ── Socket / Camera creation ──────────────────────────────────────────

    def _alloc_socket(
        self, idx: int, enable_rx: bool, socket_list: Dict[str, U.UdpComms]
    ) -> None:
        name = "socket_tactile" if idx == -1 else f"socket_{idx}"
        sock = U.UdpComms(
            udp_ip=self.pc_ip,
            send_ip=self.vr_ip,
            port_tx=self.ip_port,
            port_rx=self.ip_port + 1,
            enable_rx=enable_rx,
            suppress_warnings=True,
        )
        socket_list[name] = sock
        utils.logger.info(
            f"{name}: TX={self.ip_port}, RX={self.ip_port + 1}, enableRX={enable_rx}"
        )
        self.ip_port += 2

    def _create_camera(self, idx: int, camera_list: Dict[str, Realsense]) -> None:
        name = f"camera_{idx}"
        try:
            serial = self.camera_series_num[idx]
        except IndexError:
            utils.logger.warning(f"{name}: No serial number at index {idx}, skip.")
            return

        cam = Realsense(
            fps=self.realsense_fps,
            resolution=self.realsense_resolution,
            enable_depth=self.depth_mode,
            enable_pointcloud=False,
            enable_hole_filling=self.depth_mode,
            serial_number=serial,
        )
        camera_list[name] = cam
        w, h = self.realsense_resolution
        utils.logger.info(
            f"{name}: serial={serial}, fps={self.realsense_fps}, res={w}x{h}"
        )

    def _create_udp_and_cameras(
        self,
    ) -> Tuple[Dict[str, U.UdpComms], Dict[str, Realsense]]:
        socket_list: Dict[str, U.UdpComms] = {}
        camera_list: Dict[str, Realsense] = {}

        utils.logger.info(
            f"PC IP: {self.pc_ip}, VR IP: {self.vr_ip}, base_port={self.ip_port}"
        )

        n = min(self.camera_num, MAX_CAMERAS)
        if self.camera_num > MAX_CAMERAS:
            utils.logger.warning(
                f"Only {MAX_CAMERAS} camera sockets supported, got {self.camera_num}"
            )

        for i in range(n):
            self._alloc_socket(i, (i < 3), socket_list)
            self._create_camera(i, camera_list)

        min_sockets = 3
        for i in range(n, max(min_sockets, n)):
            self._alloc_socket(i, True, socket_list)

        if self.tactile_transfer_status:
            utils.logger.info("Initializing tactile transfer socket...")
            self.ip_port = self.tactile_port
            self._alloc_socket(-1, False, socket_list)

        utils.logger.info(
            f"{len(camera_list)} cameras, {len(socket_list)} UDP sockets created."
        )
        return socket_list, camera_list

    # ── HD chunked image sending ─────────────────────────────────────────

    def send_hd_frame(
        self, sock: U.UdpComms, frame_bgr: np.ndarray,
        cam_idx: int = 0, quality: int | None = None,
    ) -> int:
        """Encode frame to JPEG and send via chunked HD protocol.

        Protocol matches UdpSocketMultiHD.cs / python_sender_hd.py:
            Header (12B big-endian): frameId(u32) chunkIdx(u16) totalChunks(u16) totalBytes(u32)
            Payload: JPEG data slice

        Each camera has its own independent frame counter so that VR-side
        reassembly sees a monotonically increasing sequence per socket.

        Returns the number of chunks sent.
        """
        if quality is None:
            quality = self.jpeg_quality

        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return 0

        image_bytes: bytes = buf.tobytes()
        total_bytes = len(image_bytes)
        chunk_size = self.hd_chunk_size
        num_chunks = (total_bytes + chunk_size - 1) // chunk_size

        # Per-camera frame counter — avoids interleaving IDs across cameras
        ctr = self._frame_counters.get(cam_idx, 0)
        frame_id = ctr & 0xFFFFFFFF
        self._frame_counters[cam_idx] = ctr + 1

        raw_sock = sock._sock
        target = (sock.send_ip, sock.udp_send_port)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_bytes)
            payload = image_bytes[start:end]

            header = struct.pack(
                HD_HEADER_FMT,
                frame_id,
                i,
                num_chunks,
                total_bytes & 0xFFFFFFFF,
            )
            raw_sock.sendto(header + payload, target)

        return num_chunks

    # ── Image processing ──────────────────────────────────────────────────

    @staticmethod
    def center_zoom(
        image: np.ndarray, scale: float = 1.5, interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w <= w or new_h <= h:
            return cv2.resize(image, (w, h), interpolation=interpolation)

        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        return resized[start_y : start_y + h, start_x : start_x + w]

    def data_process(
        self, frame: np.ndarray, cam_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare frame for VR streaming: returns (bgr_zoomed, original_rgb)."""
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        zoom_factor = (
            self.camera_zoom[cam_idx] if cam_idx < len(self.camera_zoom) else 1.0
        )
        frame_bgr_zoomed = self.center_zoom(frame_bgr, zoom_factor)
        return frame_bgr_zoomed, frame

    # ── VR resolution / zoom control parsing ──────────────────────────────

    def _parse_resolution_control(self, s: str) -> None:
        s = s.strip(";")
        if not s:
            return

        for item in s.split(";"):
            if not item:
                continue
            key, value = map(str.strip, item.split(",", 1))

            if key.isdigit():
                cam_idx = (int(key) % self.initial_port) // 2
                if cam_idx < self.camera_num:
                    zoom_val = float(value[1:])
                    if not np.isclose(self.camera_zoom[cam_idx], zoom_val):
                        utils.logger.info(
                            f"camera{cam_idx}: zoom "
                            f"{self.camera_zoom[cam_idx]:.2f} → {zoom_val:.2f}"
                        )
                        self.camera_zoom[cam_idx] = zoom_val
                else:
                    utils.logger.warning(
                        f"Invalid camera index {cam_idx}, total={self.camera_num}"
                    )
            else:
                with self._lock:
                    self.fine_mode = value

    # ── Movement detection ────────────────────────────────────────────────

    def is_movement_exist(self) -> bool:
        with self._lock:
            d = self.data
        if d is None:
            return False
        return bool(d[1]["GripTrigger"]) or abs(d[1]["Joystick"][1]) > 0.7

    # ── Thread functions ──────────────────────────────────────────────────

    def _camera_read_thread(self, camera: Realsense, idx: int) -> None:
        utils.logger.info(f"RX camera thread {idx} starts!")
        consecutive_errors = 0
        MAX_RETRIES = 10
        while self.running:
            try:
                img = camera.get_rgb_image()
                depth = None
                if self.depth_mode:
                    try:
                        depth = camera._retrieve_depth_map()
                    except (RuntimeError, AttributeError):
                        pass
                with self._lock:
                    capture_ts_ns = time.monotonic_ns()
                    self.camera_data[f"camera_{idx}"] = img
                    self.camera_data_timestamps_ns[f"camera_{idx}"] = capture_ts_ns
                    if depth is not None:
                        self.depth_images[f"camera_{idx}"] = depth
                        self.depth_timestamps_ns[f"camera_{idx}"] = capture_ts_ns
                consecutive_errors = 0  # reset on success
                time.sleep(1 / STREAM_FPS)
            except RuntimeError as e:
                consecutive_errors += 1
                if consecutive_errors >= MAX_RETRIES:
                    utils.logger.error(
                        f"RX camera {idx}: {MAX_RETRIES} consecutive errors, giving up: {e}"
                    )
                    break
                utils.logger.warning(
                    f"RX camera {idx}: RuntimeError ({consecutive_errors}/{MAX_RETRIES}), "
                    f"retrying in 1s: {e}"
                )
                time.sleep(1.0)

    def _camera_send_thread(self, socket: U.UdpComms, idx: int) -> None:
        utils.logger.info(f"TX camera thread {idx} starts!")
        stats_t0 = time.time()
        stats_frames = 0
        stats_chunks = 0
        stats_bytes = 0
        STATS_INTERVAL = 5.0  # seconds between stats log lines

        while self.running:
            try:
                with self._lock:
                    raw = self.camera_data.get(f"camera_{idx}")
                    frame_ts_ns = self.camera_data_timestamps_ns.get(
                        f"camera_{idx}", 0
                    )
                if raw is None:
                    time.sleep(1 / STREAM_FPS)
                    continue
                frame_bgr, frame_rgb = self.data_process(raw, idx)
                with self._lock:
                    self.camera_images[f"camera_{idx}"] = frame_rgb
                    self.camera_image_timestamps_ns[f"camera_{idx}"] = frame_ts_ns
                n_chunks = self.send_hd_frame(socket, frame_bgr, cam_idx=idx)
                stats_frames += 1
                stats_chunks += n_chunks

                # Periodic stats
                now = time.time()
                elapsed = now - stats_t0
                if elapsed >= STATS_INTERVAL:
                    fps = stats_frames / elapsed
                    utils.logger.info(
                        f"TX camera_{idx}: {stats_frames} frames in {elapsed:.1f}s "
                        f"({fps:.1f} fps), {stats_chunks} chunks sent"
                    )
                    stats_t0 = now
                    stats_frames = 0
                    stats_chunks = 0

            except OSError as e:
                utils.logger.error(f"TX camera_{idx} OSError: {e}")
                break
            time.sleep(1 / STREAM_FPS)

    def _tactile_send_thread(self, socket: U.UdpComms) -> None:
        utils.logger.info("TX tactile thread starts!")
        while self.running:
            with self._lock:
                tb = self.tactile_byte
            if tb is not None:
                socket.send(tb)
            time.sleep(1 / TACTILE_FPS)

    def _vr_receive_thread(self, socket_list: Dict[str, U.UdpComms]) -> None:
        stale = socket_list["socket_1"].read() if "socket_1" in socket_list else None
        if stale:
            utils.logger.warning(f"Flushed stale record control on startup: '{stale}'")
        utils.logger.info(f"RX VR thread starts! (target rate: {VR_RECEIVE_HZ}Hz)")

        target_dt = 1.0 / VR_RECEIVE_HZ

        while self.running:
            t0 = time.time()
            try:
                for raw_data in socket_list["socket_0"].read_all():
                    ptype = detect_packet_type(raw_data)

                    if ptype == "controller":
                        parsed = parse_data(raw_data)
                        if parsed:
                            receive_ts_ns = time.monotonic_ns()
                            with self._lock:
                                self.data = parsed
                                self.hand_data.clear()
                                self.vr_input_timestamp_ns = receive_ts_ns

                    elif ptype in ("hand_text", "hand_binary"):
                        hand = parse_hand_data(raw_data)
                        if hand:
                            receive_ts_ns = time.monotonic_ns()
                            with self._lock:
                                self.hand_data[hand["side"]] = hand
                                self.data = None
                                self.vr_input_timestamp_ns = receive_ts_ns

                record_control = socket_list["socket_1"].read() if "socket_1" in socket_list else None
                resolution_control = socket_list["socket_2"].read() if "socket_2" in socket_list else None

                if resolution_control:
                    self._parse_resolution_control(resolution_control)

                if record_control:
                    utils.logger.debug(f"Record control: {record_control}")
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

            except Exception as e:
                utils.logger.error(f"Error in VR receive thread: {e}")
                time.sleep(0.1)

            elapsed = time.time() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    # ── Initial VR connection ─────────────────────────────────────────────

    def send_and_receive_data(
        self,
        socket_list: Dict[str, U.UdpComms],
        camera_list: Dict[str, Realsense],
    ) -> None:
        for i in range(self.camera_num):
            image = camera_list[f"camera_{i}"].get_rgb_image()
            frame_bgr, _ = self.data_process(image, i)
            self.send_hd_frame(socket_list[f"socket_{i}"], frame_bgr, cam_idx=i)

        for raw in socket_list["socket_0"].read_all():
            ptype = detect_packet_type(raw)
            if ptype == "controller":
                parsed = parse_data(raw)
                if parsed:
                    receive_ts_ns = time.monotonic_ns()
                    with self._lock:
                        self.data = parsed
                        self.hand_data.clear()
                        self.vr_input_timestamp_ns = receive_ts_ns
            elif ptype in ("hand_text", "hand_binary"):
                hand = parse_hand_data(raw)
                if hand:
                    receive_ts_ns = time.monotonic_ns()
                    with self._lock:
                        self.hand_data[hand["side"]] = hand
                        self.data = None
                        self.vr_input_timestamp_ns = receive_ts_ns

    def test_connection(self) -> list[dict]:
        """Block until VR sends initial data. Raises TimeoutError after deadline."""
        printed = False
        t0 = time.time()
        while True:
            self.send_and_receive_data(self.socket_list, self.camera_list)

            with self._lock:
                d = self.data
                has_hand = bool(self.hand_data)
            if d is not None or has_hand:
                utils.logger.info("Data received! VR connected!")
                if d is not None:
                    return d
                # In hand mode, return a dummy controller structure so callers
                # that expect list[dict] don't crash during init.
                return [
                    _dummy_controller("LTouch"),
                    _dummy_controller("RTouch"),
                ]

            if not printed:
                utils.logger.info("Connecting VR...")
                printed = True

            if time.time() - t0 > CONNECTION_TIMEOUT:
                raise TimeoutError(
                    f"VR did not respond within {CONNECTION_TIMEOUT}s"
                )
            time.sleep(0.05)

    # ── Start / Stop ──────────────────────────────────────────────────────

    def start_comms_threads(self) -> None:
        self.threads = []

        for i in range(self.camera_num):
            t = threading.Thread(
                target=self._camera_read_thread,
                args=(self.camera_list[f"camera_{i}"], i),
                daemon=True,
            )
            t.start()
            self.threads.append(t)

        utils.logger.info("Waiting for cameras to warm up...")
        deadline = time.time() + 20.0
        while len(self.camera_data) < self.camera_num:
            if time.time() > deadline:
                utils.logger.error("Timeout waiting for cameras!")
                break
            time.sleep(0.1)
        utils.logger.info(f"Cameras ready: {list(self.camera_data.keys())}")

        for i in range(self.camera_num):
            t = threading.Thread(
                target=self._camera_send_thread,
                args=(self.socket_list[f"socket_{i}"], i),
                daemon=True,
            )
            t.start()
            self.threads.append(t)

        t = threading.Thread(
            target=self._vr_receive_thread, args=(self.socket_list,), daemon=True
        )
        t.start()
        self.threads.append(t)
        utils.logger.info("RX VR data thread started")

        if self.tactile_transfer_status:
            t = threading.Thread(
                target=self._tactile_send_thread,
                args=(self.socket_list["socket_tactile"],),
                daemon=True,
            )
            t.start()
            self.threads.append(t)
            utils.logger.info("TX tactile data thread started")

    def close(self) -> None:
        utils.logger.info("Stopping CameraUDPManager...")
        self.running = False

        for t in self.threads:
            if t.is_alive():
                t.join(timeout=1.0)
        utils.logger.info("All threads stopped.")

        for name, cam in self.camera_list.items():
            try:
                if hasattr(cam, "pipeline"):
                    cam.pipeline.stop()
                elif hasattr(cam, "close"):
                    cam.close()
            except Exception as e:
                utils.logger.warning(f"Error stopping {name}: {e}")

        for name, sock in self.socket_list.items():
            try:
                sock.close()
            except Exception as e:
                utils.logger.warning(f"Error closing {name}: {e}")

        utils.logger.info("CameraUDPManager resources released.")


def _dummy_controller(name: str) -> dict:
    """Return a zeroed-out controller dict for init when in hand-tracking mode."""
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
