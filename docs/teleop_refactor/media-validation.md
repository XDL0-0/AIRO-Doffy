# PC media and communication split validation

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded test results and version information are retained.

Date: 2026-09-16\
Scope: Only `udp.py`, `WebRTC_udp.py`, `doffy_teleop/media/`, `doffy_teleop/protocol/`, media tests, loopback scripts, and this report, which are owned by this task, were modified. `config.py`, `main.py`, `realman_teleop.py`, and other root modules were not changed.

## Implementation

The two root files are now thin compatibility entry points, with the core implementation split into modules with separate responsibilities:

- `doffy_teleop/protocol/vr.py`: Dispatches classic `C,`, legacy controller CSV, `H,`, and `HB,` packets; `LegacyVRPacketDecoder` accepts injected legacy root functions, sharing the handling logic while preserving test semantics such as `patch("udp.parse_data")`.
- `doffy_teleop/protocol/control.py`: Thread-safe VR state, `Start`/`Stop`/`Undo`/`Rollback`/`DeleteLast` state transitions, and both legacy port-key and WebRTC camera-key zoom control. Property names remain exposed as `data`, `hand_data`, `data_collecting_state`, `data_export_state`, `data_rollback_state`, `tactile_byte`, and so on.
- `doffy_teleop/protocol/jpeg.py`: The frozen `!IHHI` (12-byte) JPEG header, `JpegChunkSender`, `JpegChunkAssembler`, and a real socket receiver. The assembler supports out-of-order chunks, duplicates, malformed packets, declared size/index validation, an in-flight limit, expiration of incomplete frames, completed frame-id deduplication, and explicit `reset()`.
- `doffy_teleop/protocol/signaling.py`: A strict JSON signaling envelope that preserves the `hello`/`hello_ack`/`offer`/`answer`/`ice_candidate`/`stop_video` shapes.
- `doffy_teleop/media/frames.py`: Pure functions for RGB→BGR conversion, uint8 normalization, and center zoom, shared by UDP and WebRTC.
- `doffy_teleop/media/camera.py`: `FrameStore`, a bounded `CameraCaptureService` with clear responsibilities, `CameraFrameProvider`, and `CameraVideoTrack`. Capture is independent of transport, and frames are always read from the shared latest store.
- `doffy_teleop/media/udp.py`: `UDPManagerCore` and socket/send/receive workers; the classic port layout, up to five JPEG camera TX streams, VR pose, recording control, zoom, and tactile TX remain compatible.
- `doffy_teleop/media/webrtc_peer.py`: `WebRTCSession` owns only the aiortc peer, tracks, the `control` DataChannel, and trickle ICE callbacks.
- `doffy_teleop/media/webrtc_signaling.py`: `WebRTCSignalingServer` owns only aiohttp WebSocket signaling, the session lifecycle, and `hello`/offer/answer/ICE/stop messages.
- `doffy_teleop/media/webrtc.py`: `WebRTCUDPManagerCore` combines camera capture with the legacy UDP pose/record/tactile compatibility interface; root entry points still export the historical names from here.
- Root `udp.py`/`WebRTC_udp.py` bind the legacy modules' `Config`, `U.UdpComms`, VR parser, RealSense factory, and aiortc symbols so existing callers are not forced to migrate.

## Automated validation

Using the environment specified by the task, `/tmp/airo-teleop-qa/bin/python`:

```text
tests/test_udp.py                                          4 passed
tests/test_teleop_media_protocol.py                       4 passed
tests/test_teleop_media_udp_loopback.py                   2 passed
tests/test_teleop_media_webrtc.py                         1 passed
---------------------------------------------------------------
Related test set                                        11 passed
```

Actual command:

```bash
PYTHONWARNINGS=ignore /tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_teleop_media_*.py tests/test_udp.py
# 11 passed in 11.57s
```

`test_teleop_media_webrtc.py` does not mock aiortc: it creates real server/client `RTCPeerConnection` instances, completes offer/answer/ICE, receives a real `VideoStreamTrack`, decodes `VideoFrame` objects, and checks dimensions/color regions and connection state.

`test_teleop_media_udp_loopback.py` uses real `AF_INET/SOCK_DGRAM` sockets to send malformed packets, out-of-order chunks, premature duplicate chunks, and a frame that expires after its final chunk is omitted, then resets to accept a frame id after a restart; the decoded JPEG dimensions remain unchanged.

The full-repository legacy baseline command was also run:

```bash
/tmp/airo-teleop-qa/bin/python -m pytest -q
# 686 passed, 95 subtests passed, 1 failed
```

The only failure is the existing `tests/test_eval_policy_latency.py::PolicyCompatibilityTest.test_dp_contact_no_vision_registration_and_resolution`: `ACTION_STEPS` in `eval_config.py` is 4, while the test requires 8. That file is outside this task's ownership and the failure is unrelated to the media changes.

## Python→Unity C# interoperability evidence

The `/tmp/teleop-protocol-tests.exe` provided by the primary agent uses the same `JpegFrameAssembler`/DatagramReceiver as Unity. Command:

```bash
/tmp/airo-teleop-qa/bin/python scripts/teleop_refactor/media_loopback.py --csharp
```

Key results from the actual output:

```json
{
  "status": "passed",
  "transport": "python-to-unity-csharp-udp",
  "shape": [72, 96],
  "chunks": 37,
  "ready": "READY 34599"
}
```

The script checks that the image reassembled by C# matches the Python transmission result; the sender emits 37 datagrams out of order, and the C# harness returns code 0. Python's own real UDP results are 61 chunks, `malformed=1`, `duplicates=1`, `expired=1`, and `restarts=1`, with decoded dimensions `[96, 128]`.

## RealSense on-site check

Command:

```bash
/tmp/airo-teleop-qa/bin/python scripts/teleop_refactor/media_loopback.py --realsense --frames 3
```

This machine detected a RealSense D435 with serial `REALSENSE_SERIAL_REDACTED`. The script only opens the camera, captures three frames, and passes them through local Python UDP with concurrent draining; the actual output is `status=passed` and `captured_frames=3`, with all three frames decoding to `[480, 640]`. Both sender and receiver use `127.0.0.1`; no robot is imported or operated. The script records the failure location at each stage: camera initialization, frame capture, JPEG encoding, UDP bind, and UDP loopback. If the device is busy or on-site capture times out, it reports `status=unavailable` with the specific `stage`, without disguising a camera failure as a network pass.

## Remaining integration boundaries

- Local negotiation and video decoding between two aiortc endpoints in `WebRTCSession` have been validated; real Unity WebRTC signaling/ICE, H.264 codec selection, networking across devices, and reconnection still require acceptance in the primary agent's Unity project.
- The classic JPEG receiver has Python↔C# byte-level evidence; the complete C# UI/camera subscription lifecycle remains the primary agent's responsibility.
- The WebRTC manager's built-in composition still preserves the basic v1 camera/VR/tactile paths; robot control and recording flows in `main.py`/`realman_teleop.py` have not been reorganized in this task, to remain within ownership.
- Legacy `UdpComms` remains in root `udp_comms.py`. This split uses the same module object to preserve the `udp.U.UdpComms` and `WebRTC_udp.U.UdpComms` patch entry points; this task did not move it.
# Final cross-language camera acceptance addendum

Final integration used `scripts/teleop_refactor/media_loopback.py --csharp --harness /tmp/doffy-csharp-checks/ProtocolHarness.exe --realsense --frames 3` to capture three real D435 images at 640×480 and send the final frame through the production Python JPEG chunk sender to the production C# reassembly module. The JPEG after reassembling 38 chunks matched the original before transmission, and the decoded dimensions were 640×480. Machine-readable results are in [camera-interop-result.json](camera-interop-result.json). This test ran on local Mono and does not constitute acceptance of Quest texture rendering or the wireless link.
