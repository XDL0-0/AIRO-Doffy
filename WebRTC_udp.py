"""Camera capture + WebRTC streaming to VR headset, and VR data reception.

Replaces the chunked-JPEG-over-UDP camera streaming in camera_udp.py with
WebRTC (aiortc).  Each Realsense camera becomes a VideoStreamTrack in a
single RTCPeerConnection.  A DataChannel named "control" replaces the old
UDP socket_2 resolution/zoom commands.

All other UDP channels (VR pose data on socket_0, record control on
socket_1, tactile on socket_tactile) are kept unchanged.

Dependencies:
    pip install aiortc aiohttp av
"""

from __future__ import annotations

import asyncio
import json
import cv2
import time
import threading
import numpy as np
import pyrealsense2 as rs
from typing import Dict, List, Optional, Tuple

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from aiortc.mediastreams import VideoStreamTrack
from aiohttp import web

import utils
import udp_comms as U
from config import Config
from parse_vr import parse_data, parse_hand_data, detect_packet_type
from airo_camera_toolkit.cameras.realsense.realsense import Realsense

MAX_CAMERAS = 5
STREAM_FPS = 30
TACTILE_FPS = 100
VR_RECEIVE_HZ = 60
CONNECTION_TIMEOUT = 60.0


# ── WebRTC video track ───────────────────────────────────────────────────


class RealsenseCameraTrack(VideoStreamTrack):
    """A VideoStreamTrack that reads frames from a shared camera data dict.

    Each instance is bound to one camera index.  The camera read thread
    (running in the main manager) writes RGB images into *camera_data*; this
    track's ``recv()`` picks them up, applies zoom, converts to BGR, and wraps
    the result in an ``av.VideoFrame``.
    """

    kind = "video"

    def __init__(
        self,
        manager: "WebRTCUDPManager",
        cam_idx: int,
    ):
        super().__init__()
        self._manager = manager
        self._cam_idx = cam_idx
        self._frame_interval = 1.0 / STREAM_FPS

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        # Busy-wait until a frame is available (shouldn't be long).
        raw = None
        while raw is None:
            with self._manager._lock:
                raw = self._manager.camera_data.get(f"camera_{self._cam_idx}")
            if raw is None:
                await asyncio.sleep(0.005)

        frame_bgr, _ = self._manager.data_process(raw, self._cam_idx)

        video_frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


# ── Manager ───────────────────────────────────────────────────────────────


class WebRTCUDPManager:
    """WebRTC video streaming + UDP VR data reception.

    Drop-in replacement for ``CameraUDPManager`` — same public attributes
    (``data``, ``hand_data``, ``camera_images``, ``depth_images``, etc.)
    and methods (``test_connection``, ``start_comms_threads``, ``close``).
    """

    def __init__(self):
        cfg = Config()
        self.running = True
        self.pc_ip = cfg.PC_IP
        self.vr_ip = cfg.VR_IP
        self.ip_port = cfg.IP_PORT
        self.initial_port = cfg.IP_PORT
        self.signaling_port = cfg.SIGNALING_PORT
        self.tactile_transfer_status = cfg.TACTILE_TRANSFER
        self.tactile_port = cfg.TACTILE_PORT
        self.tracking_mode = cfg.TRACKING_MODE
        self.jpeg_quality = cfg.JPEG_QUALITY
        self.realsense_resolution = cfg.REALSENSE_RESOLUTION
        self.realsense_fps = cfg.REALSENSE_FPS

        self._lock = threading.Lock()
        self.data: list[dict] | None = None
        self.hand_data: Dict[str, dict] = {}
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

        self.camera_num, self.camera_series_num = self._detect_cameras()
        self.camera_zoom: List[float] = [1.0] * self.camera_num

        # UDP sockets: only for VR data RX + tactile TX  (no camera TX sockets)
        self.socket_list, self.camera_list = self._create_udp_and_cameras()
        self.threads: List[threading.Thread] = []

        # WebRTC state
        self._pc: Optional[RTCPeerConnection] = None
        self._control_channel = None
        self._video_tracks: List[RealsenseCameraTrack] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._signaling_runner: Optional[web.AppRunner] = None
        self._ws_connections: Dict[str, web.WebSocketResponse] = {}
        self._session_id: Optional[str] = None

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
        """Create UDP sockets for VR data and cameras.

        Unlike CameraUDPManager we do NOT allocate per-camera TX sockets.
        We only need RX sockets for VR pose (socket_0) and record control
        (socket_1), plus an optional tactile TX socket.
        """
        socket_list: Dict[str, U.UdpComms] = {}
        camera_list: Dict[str, Realsense] = {}

        utils.logger.info(
            f"PC IP: {self.pc_ip}, VR IP: {self.vr_ip}, base_port={self.ip_port}"
        )

        # ── Create cameras (no per-camera UDP sockets) ────────────────────
        n = min(self.camera_num, MAX_CAMERAS)
        if self.camera_num > MAX_CAMERAS:
            utils.logger.warning(
                f"Only {MAX_CAMERAS} cameras supported, got {self.camera_num}"
            )
        for i in range(n):
            self._create_camera(i, camera_list)

        # ── Allocate RX-only UDP sockets for VR data ──────────────────────
        # socket_0: VR pose data (controller / hand tracking)
        self._alloc_socket(0, True, socket_list)
        # socket_1: record control (Start / Stop)
        self._alloc_socket(1, True, socket_list)
        # socket_2: resolution control — now handled by WebRTC DataChannel,
        # but we still allocate the socket for backward compatibility so the
        # port numbering stays consistent if needed.
        self._alloc_socket(2, True, socket_list)

        if self.tactile_transfer_status:
            utils.logger.info("Initializing tactile transfer socket...")
            self.ip_port = self.tactile_port
            self._alloc_socket(-1, False, socket_list)

        utils.logger.info(
            f"{len(camera_list)} cameras, {len(socket_list)} UDP sockets created."
        )
        return socket_list, camera_list

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
        """Prepare frame for streaming: returns (bgr_zoomed, original_rgb)."""
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        zoom_factor = (
            self.camera_zoom[cam_idx] if cam_idx < len(self.camera_zoom) else 1.0
        )
        frame_bgr_zoomed = self.center_zoom(frame_bgr, zoom_factor)
        return frame_bgr_zoomed, frame

    # ── VR resolution / zoom control parsing (from DataChannel) ───────────

    def _parse_resolution_control(self, s: str) -> None:
        s = s.strip(";")
        if not s:
            return

        for item in s.split(";"):
            if not item:
                continue
            key, value = map(str.strip, item.split(",", 1))

            if key.isdigit():
                cam_idx = int(key)
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
    # ── Movement detection ────────────────────────────────────────────────

    def is_movement_exist(self) -> bool:
        with self._lock:
            d = self.data
        if d is None:
            return False
        return bool(d[1]["GripTrigger"]) or abs(d[1]["Joystick"][1]) > 0.7

    # ── WebRTC Signaling Server ───────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle one WebSocket connection for WebRTC signaling."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        utils.logger.info("Signaling: WebSocket client connected")

        sid: str = ""
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                msg_type = data.get("type", "")
                sid = data.get("session_id", sid)
                payload = data.get("payload", {})

                if msg_type == "hello":
                    self._session_id = sid
                    self._ws_connections[sid] = ws
                    # Acknowledge
                    await ws.send_json({
                        "type": "hello_ack",
                        "session_id": sid,
                        "payload": {},
                    })
                    utils.logger.info(f"Signaling: hello from session {sid}")

                elif msg_type == "start_video":
                    utils.logger.info("Signaling: start_video received")
                    # PeerConnection is created when we receive the offer

                elif msg_type == "offer":
                    sdp = payload.get("sdp", "")
                    utils.logger.info("Signaling: received offer from VR")
                    await self._handle_offer(sdp, ws, sid)

                elif msg_type == "ice_candidate":
                    await self._handle_ice_candidate(payload)

                elif msg_type == "stop_video":
                    utils.logger.info("Signaling: stop_video received")
                    await self._close_peer()

                else:
                    utils.logger.debug(f"Signaling: unknown type '{msg_type}'")

        except Exception as e:
            utils.logger.error(f"Signaling WS error: {e}")
        finally:
            self._ws_connections.pop(sid, None)
            utils.logger.info("Signaling: WebSocket client disconnected")

        return ws

    async def _handle_offer(
        self, sdp: str, ws: web.WebSocketResponse, sid: str
    ) -> None:
        """Process an SDP offer from the VR headset and send back an answer."""
        # Close any previous peer connection
        await self._close_peer()

        self._pc = RTCPeerConnection()

        # ── Add video tracks ──────────────────────────────────────────────
        self._video_tracks = []
        for i in range(self.camera_num):
            track = RealsenseCameraTrack(self, i)
            self._pc.addTrack(track)
            self._video_tracks.append(track)
            utils.logger.info(f"WebRTC: added video track for camera_{i}")

        # ── DataChannel for resolution/zoom control ───────────────────────
        self._control_channel = self._pc.createDataChannel("control")

        @self._control_channel.on("open")
        def on_open():
            utils.logger.info("WebRTC DataChannel 'control' opened")

        @self._control_channel.on("message")
        def on_message(message):
            # Resolution/zoom control from VR
            utils.logger.debug(f"DataChannel control msg: {message}")
            self._parse_resolution_control(message)

        # ── ICE candidate callback ────────────────────────────────────────
        @self._pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                return
            await ws.send_json({
                "type": "ice_candidate",
                "session_id": sid,
                "payload": {
                    "candidate": candidate.candidate,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                },
            })

        @self._pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = self._pc.connectionState
            utils.logger.info(f"WebRTC connection state: {state}")
            if state == "failed":
                await self._close_peer()

        # ── Set remote offer, create answer ───────────────────────────────
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self._pc.setRemoteDescription(offer)

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        await ws.send_json({
            "type": "answer",
            "session_id": sid,
            "payload": {"sdp": self._pc.localDescription.sdp},
        })
        utils.logger.info("WebRTC: answer sent to VR")

    async def _handle_ice_candidate(self, payload: dict) -> None:
        """Add a remote ICE candidate from the VR headset."""
        if self._pc is None:
            return
        candidate_str = payload.get("candidate", "")
        if not candidate_str:
            return
        sdp_mid = payload.get("sdpMid", "")
        sdp_mline_index = payload.get("sdpMLineIndex", 0)
        
        from aiortc.sdp import candidate_from_sdp
        try:
            # candidate_str usually starts with "candidate:"
            # candidate_from_sdp expects the line content
            if candidate_str.startswith("candidate:"):
                candidate = candidate_from_sdp(candidate_str.split(":", 1)[1])
            else:
                candidate = candidate_from_sdp(candidate_str)
            candidate.sdpMid = sdp_mid
            candidate.sdpMLineIndex = sdp_mline_index
            await self._pc.addIceCandidate(candidate)
        except Exception as e:
            utils.logger.error(f"WebRTC: Failed to parse ICE candidate: {e}")

    async def _close_peer(self) -> None:
        """Shut down the current PeerConnection."""
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
        self._video_tracks = []
        self._control_channel = None
        utils.logger.info("WebRTC PeerConnection closed.")

    # ── Async event loop management ───────────────────────────────────────

    def _start_async_loop(self) -> None:
        """Start the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_signaling_server())
        except RuntimeError as e:
            if "Event loop stopped before Future completed" not in str(e):
                raise
            utils.logger.warning("WebRTC event loop stopped during shutdown.")
        finally:
            self._loop.close()

    async def _run_signaling_server(self) -> None:
        """Run the WebSocket signaling server until the manager stops."""
        self._shutdown_event = asyncio.Event()
        app = web.Application()
        app.router.add_get("/", self._ws_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        self._signaling_runner = runner

        site = web.TCPSite(runner, self.pc_ip, self.signaling_port)
        await site.start()
        utils.logger.info(
            f"Signaling server started at ws://{self.pc_ip}:{self.signaling_port}"
        )

        try:
            while self.running:
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.5)
                    break
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._close_peer()
            await runner.cleanup()
            utils.logger.info("Signaling server stopped.")

    # ── Thread functions ──────────────────────────────────────────────────

    def _camera_read_thread(self, camera: Realsense, idx: int) -> None:
        utils.logger.info(f"RX camera thread {idx} starts!")
        while self.running:
            try:
                camera.grab_images()
                img = camera.retrieve_rgb_image()
                if img.dtype != np.uint8:
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                depth = None
                if self.depth_mode:
                    try:
                        depth = camera.retrieve_depth_map()
                    except (RuntimeError, AttributeError):
                        pass
                with self._lock:
                    capture_ts_ns = time.monotonic_ns()
                    self.camera_data[f"camera_{idx}"] = img
                    self.camera_data_timestamps_ns[f"camera_{idx}"] = capture_ts_ns
                    # Recording and visualization must not depend on whether a
                    # WebRTC peer is currently consuming the video track.
                    self.camera_images[f"camera_{idx}"] = img
                    self.camera_image_timestamps_ns[f"camera_{idx}"] = capture_ts_ns
                    if depth is not None:
                        self.depth_images[f"camera_{idx}"] = depth
                        self.depth_timestamps_ns[f"camera_{idx}"] = capture_ts_ns
                time.sleep(1 / STREAM_FPS)
            except RuntimeError:
                break

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

                # Resolution control can also come from UDP socket_2 for
                # backward compatibility, but primarily uses DataChannel now.
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
        """Poll VR data sockets (no image sending in WebRTC mode)."""
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
        """Block until VR sends initial data.  Raises TimeoutError after deadline."""
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

        # ── Camera read threads (same as original) ────────────────────────
        for i in range(self.camera_num):
            t = threading.Thread(
                target=self._camera_read_thread,
                args=(self.camera_list[f"camera_{i}"], i),
                daemon=True,
            )
            t.start()
            self.threads.append(t)

        utils.logger.info("Waiting for cameras to warm up...")
        deadline = time.time() + 10.0
        while len(self.camera_data) < self.camera_num:
            if time.time() > deadline:
                utils.logger.error("Timeout waiting for cameras!")
                break
            time.sleep(0.1)
        utils.logger.info(f"Cameras ready: {list(self.camera_data.keys())}")

        # ── WebRTC signaling + video (replaces camera send threads) ───────
        self._loop_thread = threading.Thread(
            target=self._start_async_loop,
            daemon=True,
        )
        self._loop_thread.start()
        utils.logger.info(
            f"WebRTC signaling thread started on port {self.signaling_port}"
        )

        # ── VR data receive thread ────────────────────────────────────────
        t = threading.Thread(
            target=self._vr_receive_thread, args=(self.socket_list,), daemon=True
        )
        t.start()
        self.threads.append(t)
        utils.logger.info("RX VR data thread started")

        # ── Tactile TX thread (optional) ──────────────────────────────────
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
        utils.logger.info("Stopping WebRTCUDPManager...")
        self.running = False

        if self._loop is not None and self._loop.is_running():
            if self._shutdown_event is not None:
                self._loop.call_soon_threadsafe(self._shutdown_event.set)
            else:
                self._loop.call_soon_threadsafe(lambda: None)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10.0)
            if (
                self._loop_thread.is_alive()
                and self._loop is not None
                and self._loop.is_running()
            ):
                utils.logger.warning(
                    "WebRTC async loop did not stop cleanly; forcing stop."
                )
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=2.0)

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

        utils.logger.info("WebRTCUDPManager resources released.")


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
