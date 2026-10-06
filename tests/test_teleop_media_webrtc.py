from __future__ import annotations

import asyncio
import logging
import unittest

import numpy as np

from aiortc import RTCPeerConnection

from doffy_teleop.media.camera import CameraFrameProvider, FrameStore
from doffy_teleop.media.webrtc import WebRTCSession


class RealAiortcLoopbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_real_peer_connections_negotiate_and_decode_frame(self) -> None:
        logging.getLogger("aioice").setLevel(logging.WARNING)
        store = FrameStore()
        source = np.zeros((48, 64, 3), dtype=np.uint8)
        source[:, :32] = (10, 80, 220)
        source[:, 32:] = (220, 80, 10)
        store.publish("camera_0", source, 1)
        provider = CameraFrameProvider(store.camera_images, store.lock, [1.0])
        server = WebRTCSession(provider, 1, create_control_channel=True)
        client = RTCPeerConnection()
        received: asyncio.Queue = asyncio.Queue()

        @client.on("track")
        def on_track(track):
            async def consume():
                await received.put(await track.recv())

            asyncio.create_task(consume())

        client.addTransceiver("video", direction="recvonly")
        # Include an SCTP m-line so the server's compatibility "control"
        # channel is negotiated as well as the video m-line.
        client.createDataChannel("client-control")
        try:
            offer = await client.createOffer()
            await client.setLocalDescription(offer)
            answer = await server.handle_offer(client.localDescription.sdp, session_id="loopback")
            await client.setRemoteDescription(answer)
            frame = await asyncio.wait_for(received.get(), timeout=10.0)
            decoded = frame.to_ndarray(format="bgr24")
            self.assertEqual(decoded.shape, source.shape)
            self.assertGreater(float(decoded[:, :32, 0].mean()), float(decoded[:, 32:, 0].mean()))
            self.assertEqual(server.pc.connectionState, "connected")
        finally:
            await server.close()
            await client.close()


if __name__ == "__main__":
    unittest.main()
