"""Camera-free WebRTC H.264 transport with legacy signaling envelopes."""

from __future__ import annotations

import asyncio
import json
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol

from ...config.models import NetworkConfig, VideoStreamingConfig
from ...core.buffers import is_newer_sequence
from ...core.errors import (
    LifecycleError,
    ModelValidationError,
    OptionalDependencyError,
    VideoEncodingError,
)
from ...core.types import EncodedFrame, VideoCodec


@dataclass(frozen=True, slots=True)
class WebRTCSignalingMessage:
    """Validated form of the existing ``type/session_id/payload`` envelope."""

    message_type: str
    session_id: str
    payload: dict[str, Any]


def parse_signaling_envelope(
    value: str | Mapping[str, Any],
) -> WebRTCSignalingMessage:
    """Parse one legacy-compatible signaling object without WebRTC imports."""

    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ModelValidationError("signaling message is not valid JSON") from exc
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ModelValidationError("signaling message must be JSON text or a mapping")
    if not isinstance(raw, Mapping):
        raise ModelValidationError("signaling message root must be an object")
    message_type = raw.get("type")
    session_id = raw.get("session_id", "")
    payload = raw.get("payload", {})
    if not isinstance(message_type, str) or not message_type:
        raise ModelValidationError("signaling type must be a non-empty string")
    if not isinstance(session_id, str):
        raise ModelValidationError("signaling session_id must be a string")
    if not isinstance(payload, Mapping):
        raise ModelValidationError("signaling payload must be an object")
    return WebRTCSignalingMessage(
        message_type=message_type,
        session_id=session_id,
        payload=dict(payload),
    )


def signaling_envelope(
    message_type: str,
    session_id: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact legacy signaling envelope shape."""

    return {
        "type": message_type,
        "session_id": session_id,
        "payload": dict(payload or {}),
    }


def _resolve_waiter(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class _PublishedFrameBuffer:
    """Thread-to-async latest-only bridge for one encoded stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: EncodedFrame | None = None
        self._generation = 0
        self._consumed_generation = 0
        self._last_sequence: int | None = None
        self._closed = False
        self._waiters: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]
        ] = []

    def publish(self, frame: EncodedFrame) -> tuple[bool, bool]:
        with self._lock:
            if self._closed:
                raise LifecycleError("WebRTC frame buffer is closed")
            if self._last_sequence is not None and not is_newer_sequence(
                frame.sequence,
                self._last_sequence,
            ):
                return False, False
            overwritten = self._generation > self._consumed_generation
            self._last_sequence = frame.sequence
            self._frame = frame
            self._generation += 1
            waiters = self._waiters
            self._waiters = []
        for loop, future in waiters:
            loop.call_soon_threadsafe(_resolve_waiter, future)
        return True, overwritten

    async def wait_after(
        self,
        generation: int,
    ) -> tuple[EncodedFrame, int] | None:
        while True:
            loop = asyncio.get_running_loop()
            with self._lock:
                if self._generation > generation and self._frame is not None:
                    return self._frame, self._generation
                if self._closed:
                    return None
                future: asyncio.Future[None] = loop.create_future()
                waiter = (loop, future)
                self._waiters.append(waiter)
            try:
                await future
            finally:
                with self._lock:
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)

    def mark_consumed(self, generation: int) -> None:
        with self._lock:
            self._consumed_generation = max(
                self._consumed_generation,
                generation,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            waiters = self._waiters
            self._waiters = []
        for loop, future in waiters:
            loop.call_soon_threadsafe(_resolve_waiter, future)


class WebRTCRuntime(Protocol):
    """Blocking signaling/peer runtime owned by one transport thread."""

    ready: threading.Event
    error: Exception | None

    def run(self) -> None:
        """Run until ``stop`` is requested."""

    def stop(self) -> None:
        """Request thread-safe shutdown."""


RuntimeFactory = Callable[
    ["WebRTCVideoTransport", str, int],
    WebRTCRuntime,
]


@dataclass(frozen=True, slots=True)
class WebRTCTransportMetrics:
    frames_submitted: int
    frames_delivered: int
    dropped_latest: int
    dropped_stale: int
    signaling_errors: int
    peer_connections: int


class _AiortcRuntime:
    """Optional-dependency adapter for aiohttp, aiortc, and PyAV."""

    def __init__(
        self,
        owner: WebRTCVideoTransport,
        host: str,
        port: int,
    ) -> None:
        self.ready = threading.Event()
        self.error: Exception | None = None
        self._owner = owner
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._peer = None
        self._deps: dict[str, Any] = {}

    def _load_dependencies(self) -> None:
        try:
            from aiohttp import web
            from aiortc import (
                MediaStreamTrack,
                RTCPeerConnection,
                RTCRtpSender,
                RTCSessionDescription,
            )
            from aiortc.mediastreams import MediaStreamError
            from aiortc.sdp import candidate_from_sdp
            from av import Packet
        except ImportError as exc:
            raise OptionalDependencyError(
                "WebRTC video requires the 'video-webrtc' optional dependency: "
                "pip install 'airo-doffy[video-webrtc]'"
            ) from exc
        self._deps = {
            "web": web,
            "MediaStreamTrack": MediaStreamTrack,
            "MediaStreamError": MediaStreamError,
            "Packet": Packet,
            "RTCPeerConnection": RTCPeerConnection,
            "RTCRtpSender": RTCRtpSender,
            "RTCSessionDescription": RTCSessionDescription,
            "candidate_from_sdp": candidate_from_sdp,
        }

    def _make_track(self, stream_id: str):
        runtime = self
        buffer = self._owner._buffers[stream_id]
        media_stream_track = self._deps["MediaStreamTrack"]
        media_stream_error = self._deps["MediaStreamError"]
        packet_type = self._deps["Packet"]

        class EncodedH264Track(media_stream_track):
            kind = "video"

            def __init__(self) -> None:
                super().__init__()
                self._generation = 0

            async def recv(self):
                value = await buffer.wait_after(self._generation)
                if value is None:
                    raise media_stream_error
                frame, generation = value
                self._generation = generation
                buffer.mark_consumed(generation)
                packet = packet_type(frame.data)
                packet.pts = (
                    frame.source_timestamp_ns * 90_000 // 1_000_000_000
                )
                packet.dts = packet.pts
                packet.time_base = Fraction(1, 90_000)
                runtime._owner._record_delivered()
                return packet

        return EncodedH264Track()

    def _prefer_h264(self, peer, sender) -> None:
        capabilities = self._deps["RTCRtpSender"].getCapabilities("video")
        codecs = [
            codec
            for codec in capabilities.codecs
            if codec.mimeType.lower() == "video/h264"
        ]
        if not codecs:
            raise VideoEncodingError("aiortc reports no H.264 capability")
        codecs.sort(
            key=lambda codec: codec.parameters.get("packetization-mode") == "1",
            reverse=True,
        )
        transceiver = next(
            (
                item
                for item in peer.getTransceivers()
                if item.sender is sender
            ),
            None,
        )
        if transceiver is None:
            raise VideoEncodingError("cannot find transceiver for H.264 track")
        transceiver.setCodecPreferences(codecs)

    async def _close_peer(self, expected_peer=None) -> None:
        peer = self._peer
        if peer is None or (
            expected_peer is not None and peer is not expected_peer
        ):
            return
        self._peer = None
        await peer.close()
        self._owner._set_connection_state("closed")

    async def _handle_offer(self, message: WebRTCSignalingMessage, ws) -> None:
        sdp = message.payload.get("sdp")
        if not isinstance(sdp, str) or not sdp:
            raise ModelValidationError("offer payload.sdp must be a non-empty string")
        await self._close_peer()
        peer = self._deps["RTCPeerConnection"]()
        self._peer = peer
        for stream_id in self._owner.stream_ids:
            sender = peer.addTrack(self._make_track(stream_id))
            self._prefer_h264(peer, sender)

        @peer.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                return
            await ws.send_json(
                signaling_envelope(
                    "ice_candidate",
                    message.session_id,
                    {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    },
                )
            )

        @peer.on("connectionstatechange")
        async def on_connectionstatechange():
            state = peer.connectionState
            self._owner._set_connection_state(state)
            if state in {"failed", "closed"}:
                await self._close_peer(peer)

        description = self._deps["RTCSessionDescription"](sdp=sdp, type="offer")
        await peer.setRemoteDescription(description)
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)
        await ws.send_json(
            signaling_envelope(
                "answer",
                message.session_id,
                {"sdp": peer.localDescription.sdp},
            )
        )
        self._owner._record_peer()

    async def _handle_candidate(self, message: WebRTCSignalingMessage) -> None:
        if self._peer is None:
            return
        candidate_text = message.payload.get("candidate", "")
        if not isinstance(candidate_text, str) or not candidate_text:
            return
        if candidate_text.startswith("candidate:"):
            candidate_text = candidate_text.split(":", 1)[1]
        candidate = self._deps["candidate_from_sdp"](candidate_text)
        candidate.sdpMid = message.payload.get("sdpMid", "")
        candidate.sdpMLineIndex = message.payload.get("sdpMLineIndex", 0)
        await self._peer.addIceCandidate(candidate)

    async def _dispatch(self, message: WebRTCSignalingMessage, ws) -> None:
        if message.message_type == "hello":
            await ws.send_json(
                signaling_envelope(
                    "hello_ack",
                    message.session_id,
                )
            )
        elif message.message_type == "offer":
            await self._handle_offer(message, ws)
        elif message.message_type == "ice_candidate":
            await self._handle_candidate(message)
        elif message.message_type == "stop_video":
            await self._close_peer()
        elif message.message_type == "start_video":
            return

    async def _ws_handler(self, request):
        web = self._deps["web"]
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for incoming in ws:
            if incoming.type != web.WSMsgType.TEXT:
                continue
            try:
                message = parse_signaling_envelope(incoming.data)
                await self._dispatch(message, ws)
            except Exception:
                self._owner._record_signaling_error()
        return ws

    async def _run_async(self) -> None:
        web = self._deps["web"]
        self._stop_event = asyncio.Event()
        app = web.Application()
        app.router.add_get("/", self._ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        self.ready.set()
        try:
            await self._stop_event.wait()
        finally:
            await self._close_peer()
            await runner.cleanup()

    def run(self) -> None:
        try:
            self._load_dependencies()
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_async())
        except Exception as exc:
            self.error = exc
            self.ready.set()
        finally:
            loop = self._loop
            self._loop = None
            if loop is not None:
                loop.close()

    def stop(self) -> None:
        loop = self._loop
        stop_event = self._stop_event
        if loop is not None and stop_event is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_event.set)


def _aiortc_runtime(
    owner: WebRTCVideoTransport,
    host: str,
    port: int,
) -> WebRTCRuntime:
    return _AiortcRuntime(owner, host, port)


class WebRTCVideoTransport:
    """Latest-only encoded H.264 tracks plus signaling and peer lifecycle."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        stream_ids: tuple[str, ...] = ("camera_0",),
        runtime_factory: RuntimeFactory = _aiortc_runtime,
        start_timeout_s: float = 5.0,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ModelValidationError("WebRTC host must be a non-empty string")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ModelValidationError("WebRTC port must be within [1, 65535]")
        if (
            not stream_ids
            or len(set(stream_ids)) != len(stream_ids)
            or any(not isinstance(item, str) or not item.strip() for item in stream_ids)
        ):
            raise ModelValidationError(
                "stream_ids must contain unique non-empty strings"
            )
        timeout = float(start_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ModelValidationError("start_timeout_s must be positive and finite")
        self._host = host
        self._port = port
        self._stream_ids = tuple(stream_ids)
        self._runtime_factory = runtime_factory
        self._start_timeout_s = timeout
        self._buffers = {
            stream_id: _PublishedFrameBuffer() for stream_id in self._stream_ids
        }
        self._lock = threading.RLock()
        self._runtime: WebRTCRuntime | None = None
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._connection_state = "new"
        self._frames_submitted = 0
        self._frames_delivered = 0
        self._dropped_latest = 0
        self._dropped_stale = 0
        self._signaling_errors = 0
        self._peer_connections = 0

    @property
    def stream_ids(self) -> tuple[str, ...]:
        return self._stream_ids

    @property
    def connection_state(self) -> str:
        with self._lock:
            return self._connection_state

    @property
    def health_error(self) -> Exception | None:
        with self._lock:
            return None if self._runtime is None else self._runtime.error

    @property
    def metrics(self) -> WebRTCTransportMetrics:
        with self._lock:
            return WebRTCTransportMetrics(
                frames_submitted=self._frames_submitted,
                frames_delivered=self._frames_delivered,
                dropped_latest=self._dropped_latest,
                dropped_stale=self._dropped_stale,
                signaling_errors=self._signaling_errors,
                peer_connections=self._peer_connections,
            )

    def _set_connection_state(self, state: str) -> None:
        with self._lock:
            self._connection_state = state

    def _record_delivered(self) -> None:
        with self._lock:
            self._frames_delivered += 1

    def _record_signaling_error(self) -> None:
        with self._lock:
            self._signaling_errors += 1

    def _record_peer(self) -> None:
        with self._lock:
            self._peer_connections += 1

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed WebRTC transport")
            if self._started:
                raise LifecycleError("WebRTC transport is already started")
            runtime = self._runtime_factory(self, self._host, self._port)
            thread = threading.Thread(
                target=runtime.run,
                name="airo-doffy-webrtc",
                daemon=True,
            )
            self._runtime = runtime
            self._thread = thread
            try:
                thread.start()
            except Exception:
                self._runtime = None
                self._thread = None
                raise
        if not runtime.ready.wait(self._start_timeout_s):
            runtime.stop()
            thread.join(timeout=1.0)
            raise LifecycleError("WebRTC signaling server did not become ready")
        if runtime.error is not None:
            thread.join(timeout=1.0)
            with self._lock:
                self._runtime = None
                self._thread = None
                self._closed = True
            for buffer in self._buffers.values():
                buffer.close()
            raise LifecycleError("WebRTC signaling server failed to start") from runtime.error
        with self._lock:
            self._started = True

    def send(self, frame: EncodedFrame) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("WebRTC transport is closed")
            if not self._started:
                raise LifecycleError("WebRTC transport has not been started")
        if not isinstance(frame, EncodedFrame):
            raise ModelValidationError("frame must be an EncodedFrame")
        if frame.codec is not VideoCodec.H264:
            raise ModelValidationError("WebRTC transport requires H.264 frames")
        try:
            buffer = self._buffers[frame.stream_id]
        except KeyError as exc:
            raise ModelValidationError(
                f"WebRTC transport has no track for {frame.stream_id!r}"
            ) from exc
        accepted, overwritten = buffer.publish(frame)
        with self._lock:
            if not accepted:
                self._dropped_stale += 1
                return
            self._frames_submitted += 1
            if overwritten:
                self._dropped_latest += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            runtime = self._runtime
            thread = self._thread
            if not self._started and runtime is None:
                self._closed = True
                for buffer in self._buffers.values():
                    buffer.close()
                return
        assert runtime is not None
        runtime.stop()
        if thread is not None:
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise LifecycleError(
                    "WebRTC runtime did not stop; transport was not invalidated"
                )
        for buffer in self._buffers.values():
            buffer.close()
        cleanup_error = runtime.error
        with self._lock:
            self._started = False
            self._closed = True
            self._runtime = None
            self._thread = None
            self._connection_state = "closed"
        if cleanup_error is not None:
            raise LifecycleError("WebRTC runtime failed") from cleanup_error


def create_webrtc_video(
    _config: VideoStreamingConfig,
    network: NetworkConfig,
) -> WebRTCVideoTransport:
    """Create one unstarted camera-0 WebRTC H.264 transport."""

    return WebRTCVideoTransport(
        network.pc_ip or "0.0.0.0",
        network.signaling_port,
    )
