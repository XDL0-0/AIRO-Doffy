"""WebRTC manager composition for classic UDP, cameras, and signaling.

Peer connection mechanics and aiohttp signaling live in sibling modules:
``webrtc_peer.py`` owns one aiortc session, while
``webrtc_signaling.py`` owns the WebSocket lifecycle.  This module keeps the
v1-compatible manager surface and composes those focused services.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import threading
import time
from typing import Any

import numpy as np

import doffy_teleop.media.udp_comms as U
import doffy_teleop.utils as utils

from doffy_teleop.protocol.control import LegacyControlState
from doffy_teleop.protocol.vr import LegacyVRPacketDecoder
from .camera import CameraCaptureService, CameraFrameProvider, CameraVideoTrack, FrameStore
from .frames import center_zoom, prepare_rgb_frame
from .sockets import UDPSocketAllocator
from .webrtc_peer import (
    STREAM_FPS,
    RTCPeerConnection,
    RTCSessionDescription,
    candidate_from_sdp,
    WebRTCSession,
)
from .webrtc_signaling import WebRTCSignalingServer, web

MAX_CAMERAS = 5
TACTILE_FPS = 100
VR_RECEIVE_HZ = 60
CONNECTION_TIMEOUT = 60.0


class WebRTCUDPManagerCore:
    """Drop-in v1 ``WebRTC_udp.WebRTCUDPManager`` implementation."""

    def __init__(
        self,
        config: Any,
        *,
        camera_manager: Any | None = None,
        camera_detector: Callable[[], list[str]] | None = None,
        camera_factory: Callable[..., Any] | None = None,
        socket_module: Any = U,
        decoder: LegacyVRPacketDecoder | None = None,
        pc_factory: Callable[[], Any] | None = None,
        description_factory: Callable[..., Any] | None = None,
        web_module: Any | None = None,
        track_factory: Callable[..., Any] = CameraVideoTrack,
        logger: Any = utils.logger,
    ) -> None:
        cfg = config
        self._logger = logger
        self.running = True
        self._closed = False
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
        self.depth_mode = cfg.DEPTH_INFO_ENABLE
        self._socket_module = socket_module
        self._decoder = decoder or LegacyVRPacketDecoder()
        self._pc_factory = pc_factory or RTCPeerConnection
        self._description_factory = description_factory or RTCSessionDescription
        self._web_module = web if web_module is None else web_module
        self._track_factory = track_factory
        self._camera_detector = camera_detector or self._default_camera_detector
        self._camera_factory = camera_factory or self._default_camera_factory

        self._lock = getattr(camera_manager, "_lock", threading.RLock()) if camera_manager else threading.RLock()
        self.camera_images: dict[str, np.ndarray]
        self.camera_image_timestamps_ns: dict[str, int]
        self.depth_images: dict[str, np.ndarray]
        self.depth_timestamps_ns: dict[str, int]
        self.camera_list: dict[str, Any] = {}
        self.camera_manager = camera_manager
        self.capture_service: CameraCaptureService | None = None

        if camera_manager is not None:
            self.camera_num = int(camera_manager.camera_num)
            self.camera_images = camera_manager.camera_images
            self.camera_image_timestamps_ns = camera_manager.camera_image_timestamps_ns
            self.depth_images = camera_manager.depth_images
            self.depth_timestamps_ns = camera_manager.depth_timestamps_ns
            self.camera_list = dict(getattr(camera_manager, "camera_list", {}))
        else:
            serials = list(self._camera_detector())
            self.camera_num = len(serials)
            store = FrameStore(lock=self._lock)
            self.camera_images = store.camera_images
            self.camera_image_timestamps_ns = store.camera_image_timestamps_ns
            self.depth_images = store.depth_images
            self.depth_timestamps_ns = store.depth_timestamps_ns
            self._frame_store = store
            try:
                for idx, serial in enumerate(serials[:MAX_CAMERAS]):
                    self._create_camera(idx, serial)
            except BaseException:
                CameraCaptureService(self.camera_list, store).close()
                raise
            self.capture_service = CameraCaptureService(
                self.camera_list,
                store,
                fps=self.realsense_fps,
                depth_enabled=self.depth_mode,
            )
        self.camera_data = self.camera_images
        self.camera_data_timestamps_ns = self.camera_image_timestamps_ns
        initial_zoom = [1.0] * self.camera_num
        self.control_state = LegacyControlState(
            self.camera_num,
            self.initial_port,
            lock=self._lock,
        )
        self.control_state.camera_zoom = initial_zoom
        self._frame_provider = CameraFrameProvider(
            self.camera_data,
            self._lock,
            self.camera_zoom,
            running=lambda: self.running,
        )

        try:
            self.socket_list, self.camera_list = self._create_udp_and_cameras()
        except BaseException:
            if self.capture_service is not None:
                self.capture_service.close()
            elif self.camera_manager is not None:
                self.camera_manager.close()
            raise
        self.threads: list[threading.Thread] = []
        self._pc: Any | None = None
        self._control_channel: Any | None = None
        self._video_tracks: list[Any] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._signaling_server: WebRTCSignalingServer | None = None
        self._signaling_runner: Any | None = None
        self._ws_connections: dict[str, Any] = {}
        self._session_id: str | None = None

    @property
    def data(self) -> list[dict] | None:
        return self.control_state.data

    @data.setter
    def data(self, value: list[dict] | None) -> None:
        with self._lock:
            self.control_state.data = value

    @property
    def hand_data(self) -> dict[str, dict]:
        return self.control_state.hand_data

    @hand_data.setter
    def hand_data(self, value: dict[str, dict]) -> None:
        with self._lock:
            self.control_state.hand_data = value

    def _state_property(name: str):
        def get(self):
            return getattr(self.control_state, name)

        def set_(self, value):
            with self._lock:
                setattr(self.control_state, name, value)

        return property(get, set_)

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

    @staticmethod
    def _default_camera_detector() -> list[str]:
        try:
            import pyrealsense2 as rs
        except ImportError:
            return []
        serials: list[str] = []
        for device in rs.context().query_devices():
            serials.append(device.get_info(rs.camera_info.serial_number))
        return serials

    def _default_camera_factory(self, **kwargs: Any) -> Any:
        from airo_camera_toolkit.cameras.realsense.realsense import Realsense

        return Realsense(**kwargs)

    def _create_camera(self, idx: int, serial: str) -> None:
        name = f"camera_{idx}"
        self.camera_list[name] = self._camera_factory(
            fps=self.realsense_fps,
            resolution=self.realsense_resolution,
            enable_depth=self.depth_mode,
            enable_pointcloud=False,
            enable_hole_filling=self.depth_mode,
            serial_number=serial,
        )

    def _create_udp_and_cameras(self) -> tuple[dict[str, Any], dict[str, Any]]:
        # WebRTC sends video over its peer, while the three UDP control sockets
        # keep the same pose/record/zoom numbering as classic UDP.
        sockets, self.ip_port = UDPSocketAllocator(self._socket_module, self._logger).allocate(
            pc_ip=self.pc_ip, vr_ip=self.vr_ip, base_port=self.ip_port,
            camera_num=0, tactile_enabled=self.tactile_transfer_status,
            tactile_port=self.tactile_port,
        )
        return sockets, self.camera_list

    @staticmethod
    def center_zoom(image: np.ndarray, scale: float = 1.5, interpolation: int = 1) -> np.ndarray:
        return center_zoom(image, scale, interpolation)

    def data_process(self, frame: np.ndarray, cam_idx: int) -> tuple[np.ndarray, np.ndarray]:
        zoom = self.camera_zoom[cam_idx] if cam_idx < len(self.camera_zoom) else 1.0
        return prepare_rgb_frame(frame, zoom=zoom)

    def _parse_resolution_control(self, value: str | bytes) -> None:
        # WebRTC's control DataChannel uses a direct camera index.
        self.control_state.update_zoom(value, key_mode="camera", logger=self._logger.warning)

    def is_movement_exist(self) -> bool:
        return self.control_state.movement_exists()

    def _consume_vr_packet(self, raw: bytes | str) -> None:
        self.control_state.consume_vr(raw, self._decoder)

    def _apply_record_control(self, value: bytes | str | None) -> None:
        self.control_state.apply_record_control(value)

    def _make_session(self) -> WebRTCSession:
        session = WebRTCSession(
            self._frame_provider,
            len(self.camera_list),
            fps=self.realsense_fps,
            control_callback=self._parse_resolution_control,
            pc_factory=self._pc_factory,
            description_factory=self._description_factory,
            track_factory=self._track_factory,
            logger=self._logger,
        )
        return session

    async def _ws_handler(self, request: Any) -> Any:
        if self._signaling_server is None:
            raise RuntimeError("signaling server is not running")
        return await self._signaling_server.handle_websocket(request)

    async def _handle_offer(self, sdp: str, ws: Any, sid: str) -> Any:
        if self._signaling_server is None:
            self._signaling_server = WebRTCSignalingServer(
                self.pc_ip,
                self.signaling_port,
                self._make_session,
                session_callback=lambda _session: self._sync_webrtc_aliases(),
                web_module=self._web_module,
                logger=self._logger,
            )
        answer = await self._signaling_server._handle_offer(sdp, ws, sid)
        self._sync_webrtc_aliases()
        return answer

    async def _handle_ice_candidate(self, payload: dict[str, Any]) -> None:
        if self._signaling_server and self._signaling_server.session:
            await self._signaling_server.session.add_ice_candidate(payload)

    async def _close_peer(self) -> None:
        if self._signaling_server is not None:
            await self._signaling_server.close_session()
        self._pc = None
        self._control_channel = None
        self._video_tracks = []

    def _sync_webrtc_aliases(self) -> None:
        session = self._signaling_server.session if self._signaling_server else None
        if session is None:
            self._pc = None
            self._control_channel = None
            self._video_tracks = []
        else:
            self._pc = session.pc
            self._control_channel = session.control_channel
            self._video_tracks = session.video_tracks

    def _start_async_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_signaling_server())
        except RuntimeError as exc:
            if "Event loop stopped before Future completed" not in str(exc):
                raise
        finally:
            self._loop.close()

    async def _run_signaling_server(self) -> None:
        self._shutdown_event = asyncio.Event()
        self._signaling_server = WebRTCSignalingServer(
            self.pc_ip,
            self.signaling_port,
            self._make_session,
            session_callback=lambda _session: self._sync_webrtc_aliases(),
            web_module=self._web_module,
            logger=self._logger,
        )
        await self._signaling_server.start()
        self.signaling_port = self._signaling_server.port
        self._signaling_runner = self._signaling_server.runner
        try:
            while self.running:
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.5)
                    break
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._signaling_server.stop()
            self._signaling_server = None

    def _camera_read_thread(self, camera: Any, idx: int) -> None:
        # Kept as a named compatibility hook.  New code uses
        # CameraCaptureService, which owns the same loop and shutdown rules.
        if self.capture_service is None:
            return
        self.capture_service._read_thread(f"camera_{idx}", camera)

    def _tactile_send_thread(self, sock: Any) -> None:
        while self.running:
            if self.tactile_byte is not None:
                sock.send(self.tactile_byte)
            time.sleep(1.0 / TACTILE_FPS)

    def _vr_receive_thread(self, socket_list: dict[str, Any] | None = None) -> None:
        sockets = self.socket_list if socket_list is None else socket_list
        target_dt = 1.0 / VR_RECEIVE_HZ
        while self.running:
            started = time.time()
            try:
                for raw in sockets["socket_0"].read_all():
                    self._consume_vr_packet(raw)
                records = sockets["socket_1"].read_all() if sockets.get("socket_1") else ()
                resolution = sockets.get("socket_2").read() if sockets.get("socket_2") else None
                if resolution:
                    self._parse_resolution_control(resolution)
                for record in records:
                    self._apply_record_control(record)
            except Exception as exc:
                self._logger.error(f"Error in VR receive thread: {exc}")
                time.sleep(0.1)
            elapsed = time.time() - started
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    def send_and_receive_data(
        self,
        socket_list: dict[str, Any] | None = None,
        camera_list: dict[str, Any] | None = None,
    ) -> None:
        sockets = self.socket_list if socket_list is None else socket_list
        for raw in sockets["socket_0"].read_all():
            self._consume_vr_packet(raw)

    def test_connection(self) -> list[dict]:
        started = time.time()
        printed = False
        while True:
            self.send_and_receive_data(self.socket_list, self.camera_list)
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
            raise RuntimeError("cannot restart a closed WebRTC manager")
        self.running = True
        if self.capture_service is not None:
            self.capture_service.start()
        elif self.camera_manager is not None:
            self.camera_manager.start()
        self.threads = []
        # Keep the old warmup behavior, but do not wait for a nonexistent
        # camera.  Capture workers are otherwise independent of WebRTC peers.
        if self.camera_num:
            deadline = time.time() + 10.0
            while len(self.camera_data) < min(self.camera_num, MAX_CAMERAS) and time.time() < deadline:
                time.sleep(0.1)
        self._loop_thread = threading.Thread(target=self._start_async_loop, daemon=True)
        self._loop_thread.start()
        receiver = threading.Thread(target=self._vr_receive_thread, args=(self.socket_list,), daemon=True)
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
        if self._closed:
            return
        self.running = False
        if self._loop is not None and self._loop.is_running() and self._shutdown_event is not None:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10.0)
            if self._loop_thread.is_alive() and self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=2.0)
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        if self.capture_service is not None:
            self.capture_service.close()
        elif self.camera_manager is not None:
            self.camera_manager.close()
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
