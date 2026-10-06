"""aiohttp WebSocket signaling lifecycle for the WebRTC media session."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import doffy_teleop.utils as utils

from doffy_teleop.protocol.signaling import SignalingMessage, parse_signaling_message
from .webrtc_peer import WebRTCSession

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - optional dependency path.
    web = None  # type: ignore[assignment]


class WebRTCSignalingServer:
    """aiohttp WebSocket signaling server for one or more sequential sessions."""

    def __init__(
        self,
        host: str,
        port: int,
        session_factory: Callable[[], WebRTCSession],
        *,
        session_callback: Callable[[WebRTCSession | None], Any] | None = None,
        web_module: Any | None = None,
        logger: Any = utils.logger,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.session_factory = session_factory
        self.session_callback = session_callback
        self._web = web if web_module is None else web_module
        self.logger = logger
        self.session: WebRTCSession | None = None
        self.runner: Any | None = None
        self.site: Any | None = None
        self.connections: dict[str, Any] = {}

    def _require_aiohttp(self) -> None:
        if self._web is None:
            raise RuntimeError("WebRTC signaling requires aiohttp")

    async def start(self) -> None:
        self._require_aiohttp()
        app = self._web.Application()
        app.router.add_get("/", self.handle_websocket)
        self.runner = self._web.AppRunner(app)
        await self.runner.setup()
        self.site = self._web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        # aiohttp chooses an ephemeral port when port=0.
        sockets = getattr(self.site, "_server", None)
        sockets = getattr(sockets, "sockets", None) or []
        if sockets:
            self.port = int(sockets[0].getsockname()[1])
        self.logger.info(f"Signaling server started at ws://{self.host}:{self.port}")

    async def handle_websocket(self, request: Any) -> Any:
        self._require_aiohttp()
        ws = self._web.WebSocketResponse()
        await ws.prepare(request)
        sid = ""
        self.logger.info("Signaling: WebSocket client connected")
        try:
            async for raw_message in ws:
                if raw_message.type != self._web.WSMsgType.TEXT:
                    continue
                try:
                    message = parse_signaling_message(raw_message.data)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    self.logger.warning(f"Signaling JSON rejected: {exc}")
                    continue
                sid = message.session_id or sid
                if message.type == "hello":
                    self.connections[sid] = ws
                    await ws.send_json(
                        SignalingMessage("hello_ack", sid, {}).as_dict()
                    )
                elif message.type == "offer":
                    await self._handle_offer(message.payload.get("sdp", ""), ws, sid)
                elif message.type == "ice_candidate":
                    if self.session is not None:
                        await self.session.add_ice_candidate(message.payload)
                elif message.type == "stop_video":
                    await self.close_session()
                elif message.type == "start_video":
                    continue
        except Exception as exc:
            self.logger.warning(f"Signaling WebSocket error: {exc}")
        finally:
            if sid:
                self.connections.pop(sid, None)
            self.logger.info("Signaling: WebSocket client disconnected")
        return ws

    async def _handle_offer(self, sdp: str, ws: Any, sid: str) -> Any:
        if not sdp:
            raise ValueError("SDP offer is empty")
        await self.close_session()
        self.session = self.session_factory()

        async def send_ice(payload: dict[str, Any]) -> None:
            await ws.send_json(
                SignalingMessage("ice_candidate", sid, payload).as_dict()
            )

        answer = await self.session.handle_offer(
            sdp,
            session_id=sid,
            ice_callback=send_ice,
        )
        if self.session_callback is not None:
            self.session_callback(self.session)
        await ws.send_json(
            SignalingMessage(
                "answer",
                sid,
                {"sdp": answer.sdp, "type": answer.type},
            ).as_dict()
        )
        return answer

    async def close_session(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None
            if self.session_callback is not None:
                self.session_callback(None)

    async def stop(self) -> None:
        await self.close_session()
        if self.runner is not None:
            await self.runner.cleanup()
        self.runner = None
        self.site = None
        self.connections.clear()


__all__ = ["WebRTCSignalingServer"]
