"""One aiortc peer connection and its transport-neutral media tracks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
from typing import Any

import doffy_teleop.utils as utils

from .camera import CameraFrameProvider, CameraVideoTrack

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.sdp import candidate_from_sdp
except ImportError:  # pragma: no cover - optional dependency path.
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    candidate_from_sdp = None  # type: ignore[assignment]


STREAM_FPS = 30


class WebRTCSession:
    """Own one aiortc peer connection and its video/control tracks."""

    def __init__(
        self,
        frame_provider: CameraFrameProvider,
        camera_count: int,
        *,
        fps: float = STREAM_FPS,
        control_callback: Callable[[Any], Any] | None = None,
        ice_callback: Callable[[dict[str, Any]], Any] | None = None,
        pc_factory: Callable[[], Any] | None = None,
        description_factory: Callable[..., Any] | None = None,
        track_factory: Callable[..., Any] = CameraVideoTrack,
        logger: Any = utils.logger,
        create_control_channel: bool = True,
    ) -> None:
        self.frame_provider = frame_provider
        self.camera_count = max(0, int(camera_count))
        self.fps = float(fps)
        self.control_callback = control_callback
        self.ice_callback = ice_callback
        self.pc_factory = pc_factory or RTCPeerConnection
        self.description_factory = description_factory or RTCSessionDescription
        self.track_factory = track_factory
        self.logger = logger
        self.create_control_channel = bool(create_control_channel)
        self.pc: Any | None = None
        self.video_tracks: list[Any] = []
        self.control_channel: Any | None = None

    def _require_aiortc(self) -> None:
        if self.pc_factory is None or self.description_factory is None:
            raise RuntimeError("WebRTC requires the aiortc optional dependency")

    def _on_control_message(self, message: Any) -> None:
        callback = self.control_callback
        if callback is None:
            return
        result = callback(message)
        if inspect.isawaitable(result):
            try:
                asyncio.create_task(result)
            except RuntimeError:
                # A synchronous callback is the normal legacy path.  If an
                # async callback arrives without a running loop, keep the
                # aiortc event emitter alive.
                pass

    def _add_control_channel(self, channel: Any) -> None:
        if channel is None or getattr(channel, "label", "") != "control":
            return
        self.control_channel = channel

        @channel.on("open")
        def on_open() -> None:
            self.logger.info("WebRTC DataChannel 'control' opened")

        @channel.on("message")
        def on_message(message: Any) -> None:
            self._on_control_message(message)

    async def handle_offer(
        self,
        sdp: str,
        *,
        session_id: str = "",
        ice_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> Any:
        """Set a remote offer, add local tracks, and return an SDP answer."""

        self._require_aiortc()
        callback = self.ice_callback if ice_callback is None else ice_callback
        await self.close()
        self.ice_callback = callback
        pc = self.pc_factory()
        self.pc = pc
        self.video_tracks = []

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            self._add_control_channel(channel)

        @pc.on("icecandidate")
        async def on_icecandidate(candidate: Any) -> None:
            if candidate is None or self.ice_callback is None:
                return
            payload = {
                "candidate": candidate.candidate,
                "sdpMid": candidate.sdpMid,
                "sdpMLineIndex": candidate.sdpMLineIndex,
            }
            try:
                result = self.ice_callback(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self.logger.warning(f"WebRTC ICE candidate callback failed: {exc}")

        for idx in range(self.camera_count):
            track = self.track_factory(self.frame_provider, idx, fps=self.fps)
            pc.addTrack(track)
            self.video_tracks.append(track)

        if self.create_control_channel:
            self._add_control_channel(pc.createDataChannel("control"))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = pc.connectionState
            self.logger.info(f"WebRTC connection state: {state}")
            if state == "failed":
                await self.close()

        await pc.setRemoteDescription(self.description_factory(sdp=sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        self.logger.info("WebRTC answer created for session %s", session_id or "<anonymous>")
        return pc.localDescription

    async def add_ice_candidate(self, payload: dict[str, Any]) -> None:
        if self.pc is None or candidate_from_sdp is None:
            return
        candidate_text = payload.get("candidate", "")
        if not candidate_text:
            return
        try:
            candidate_text = str(candidate_text)
            candidate = candidate_from_sdp(
                candidate_text.split(":", 1)[1]
                if candidate_text.startswith("candidate:")
                else candidate_text
            )
            candidate.sdpMid = payload.get("sdpMid", "")
            candidate.sdpMLineIndex = int(payload.get("sdpMLineIndex", 0))
            await self.pc.addIceCandidate(candidate)
        except Exception as exc:
            self.logger.warning(f"WebRTC ICE candidate rejected: {exc}")

    async def close(self) -> None:
        pc = self.pc
        self.pc = None
        self.video_tracks = []
        self.control_channel = None
        self.ice_callback = None
        if pc is not None:
            try:
                await pc.close()
            except Exception as exc:
                self.logger.warning(f"WebRTC peer close failed: {exc}")


__all__ = ["STREAM_FPS", "WebRTCSession"]
