# AIRO-Doffy v2 Communication

## Scope

AIRO-Doffy keeps video, high-frequency state, reliable runtime commands, and
legacy VR compatibility as separate communication concerns.

| Path | Primary transport | Delivery semantics | Encoding |
|---|---|---|---|
| video | WebRTC or RTP/UDP | low latency, old frames dropped | H.264 |
| legacy video | chunked UDP | compatibility, no retries | JPEG |
| realtime state | WebRTC DataChannel | unordered, unreliable | binary v1 |
| runtime commands | WebRTC DataChannel | ordered, reliable | strict JSON v1 |
| VR input | injected transport | transport-neutral | text legacy or binary v2 |

Packet parsing and serialization do not open sockets. Network lifecycle belongs
to transport adapters.

## Default endpoints

| Setting | Default | Purpose |
|---|---:|---|
| `network.legacy_base_port` | 8000 | legacy per-camera UDP base |
| `network.pose_port` | 8001 | legacy pose/VR compatibility |
| `network.control_port` | 8005 | legacy low-rate controls |
| `network.signaling_port` | 8765 | WebRTC WebSocket signaling |
| `network.video_rtp_port` | 5004 | experimental H.264 RTP/UDP |
| `network.state_diagnostic_port` | 5005 | opt-in state UDP diagnostics |

Committed configuration leaves host and headset addresses unset. Deployments
must provide reachable addresses, firewall rules, and unique ports.

## WebRTC signaling

`WebRTCVideoTransport` owns one signaling/event-loop thread and one peer
connection with a video track for each configured stream ID. Signaling keeps the
legacy-compatible JSON envelope:

```json
{
  "type": "offer",
  "session_id": "opaque-session-id",
  "payload": {}
}
```

The exact top-level fields are:

- `type`: non-empty message type such as `hello`, `offer`, `answer`, or
  `ice_candidate`;
- `session_id`: string;
- `payload`: object.

Unknown payload details remain transport-specific. Malformed JSON, non-object
roots, missing type, invalid session ID, and non-object payloads are rejected
before aiortc handling.

The signaling server binds `network.pc_ip` or `0.0.0.0` when no explicit host is
configured. Starting the transport waits for readiness with a bounded timeout.
Closing stops the event loop, joins its thread, closes frame buffers, and reports
runtime errors.

## WebRTC video tracks

Each stream ID has an independent latest-only encoded-frame bridge. A producer
submits complete H.264 access units:

- duplicate or stale frame sequence: reject;
- a newer frame arrives before consumption: overwrite and increment the latest
  drop metric;
- unknown stream ID or non-H.264 codec: reject;
- transport not started or already closed: lifecycle error.

One slow video track cannot create an unbounded frame queue. Delivery metrics
separate submitted, delivered, overwritten, stale, signaling-error, and peer
counts.

WebRTC provides DTLS/SRTP transport security. Raw UDP video and diagnostic state
adapters do not add encryption, authentication, acknowledgement, or retries.

## Low-latency H.264 settings

`LowLatencyH264Encoder` accepts processed RGB8, BGR8, or GRAY8 frames with even
width and height. Geometry and pixel format cannot change after the first frame.

| Setting | Default | Behavior |
|---|---:|---|
| `encoder_backend` | `auto` | try `h264_nvenc`, then `libx264` |
| `bitrate_bps` | 4,000,000 | codec target bitrate |
| `gop_frames` | 30 | fixed GOP/keyframe interval |
| `target_fps` | 30 | time base and framerate |
| `input_queue_capacity` | 1 | drop oldest input on overload |
| `output_queue_capacity` | 1 | keep newest encoded output |

Common encoder options set zero B-frames and zero rate-control lookahead.

NVENC:

```text
preset=p1
tune=ull
zerolatency=1
bf=0
rc-lookahead=0
```

libx264:

```text
preset=ultrafast
tune=zerolatency
bframes=0
rc-lookahead=0
scenecut=0
```

`auto` fallback is observable through the selected codec name. If all configured
codec candidates fail, encoding raises `VideoEncodingError`; it does not silently
change transport or pixel geometry.

## RTP/H.264 over UDP

The experimental UDP path uses RTP version 2 and RFC 6184 single-NAL or FU-A
packetization:

- default payload type: 96;
- default MTU: 1200;
- RTP clock: 90 kHz;
- 16-bit packet sequence, 32-bit timestamp, 32-bit SSRC;
- marker bit on the final packet of an access unit.

The packetizer accepts Annex B, four-byte AVCC, or one raw NAL access unit. A
bounded jitter buffer handles reassembly and exposes late/jitter drops. RTP/UDP
does not perform retransmission or congestion control and must be treated as an
explicit deployment choice.

## Legacy JPEG/UDP compatibility

The legacy Unity `UdpSocketMultiHD` packet format is frozen:

```text
network byte order, 12-byte header
uint32 frame_id
uint16 chunk_index
uint16 chunk_count
uint32 total_jpeg_bytes
payload
```

The default maximum JPEG payload per datagram is 60,000 bytes. Frame ID wraps to
`uint32`. The v2 adapter validates total size and chunk count but preserves the
wire bytes. It is a compatibility transport, not the preferred low-latency
path.

## Realtime state channel v1

The WebRTC DataChannel is created as:

```text
label = "realtime_state"
ordered = false
maxRetransmits = 0
```

One DataChannel message contains one complete little-endian envelope:

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 2 | `uint16` | magic `0xAD20` |
| 2 | 1 | `uint8` | version `1` |
| 3 | 1 | `uint8` | message type |
| 4 | 4 | `uint32` | sequence |
| 8 | 8 | `uint64` | source timestamp ns |
| 16 | 2 | `uint16` | payload length |
| 18 | 2 | `uint16` | flags |

Message type `1` contains one complete VR binary v2 payload. Message type `2`
contains robot state:

```text
uint8 joint_count (6 or 7)
uint8 options
uint8 reserved = 0
uint8 reserved = 0
joint_count * float32 joints_rad
16 * float32 row-major TCP transform
optional float32 gripper_width_m
optional 6 * float32 wrench
```

Options bit 0 enables gripper width and bit 1 enables wrench. Wrench order is
`Fx, Fy, Fz, Tx, Ty, Tz`. Reserved flags and bytes must be zero.

The sender holds at most one unread packet per state type. Sequence comparison
uses unsigned 32-bit modular ordering. Equal, older, and exactly half-range
ambiguous values are rejected.

UDP port 5005 is diagnostic only. It provides no fragmentation or retry, so
payloads should remain below the path MTU.

Full specification: [state_channel_v1.md](protocols/state_channel_v1.md).

## Reliable command channel v1

Runtime commands use a separate channel:

```text
label = "commands"
ordered = true
maxRetransmits = null
maxPacketLifeTime = null
```

Messages are canonical UTF-8 JSON, limited to 65,536 bytes. The decoder rejects
unknown, missing, or duplicate fields and unsupported versions or message
types.

Command example:

```json
{
  "message_type": "command",
  "payload": {
    "clock_domain": "device",
    "command_id": "unique-client-id",
    "kind": "start_recording",
    "origin": "unity",
    "sequence": 7,
    "source_timestamp_ns": 123456789,
    "value": null
  },
  "version": 1
}
```

Acknowledgement status is `accepted`, `rejected`, or `error` and includes the
original ID/sequence, a timestamp, message, and duplicate flag.

Receiver idempotency:

1. new ID: dispatch once and cache the ACK before sending;
2. same ID and same canonical payload: replay ACK with `duplicate=true`;
3. same ID and different payload: reject without executing;
4. dedupe capacity overflow: evict least-recently-used ID.

The default ACK deadline is one second and the default dedupe capacity is 1024.
Clients must use unique IDs and stop retrying after the deployment's retention
window.

Full specification:
[reliable_commands_v1.md](protocols/reliable_commands_v1.md).

## VR binary protocol v2

VR v2 bytes are independent of their transport. The 24-byte little-endian header
contains:

```text
magic "AVR2"
uint8 version = 2
uint8 mode: 1 controllers, 2 hands
uint8 entity_count
uint8 flags = 0
uint32 sequence
uint64 device timestamp ns
uint32 payload length
```

Controller mode requires exactly two 48-byte entities. Each has side, buttons,
position XYZ in metres, quaternion XYZW, joystick XY, index trigger, and grip
trigger.

Hand mode accepts one or two entities. Each has a side, optional wrist position
and quaternion, and exactly 26 OpenXR joint XYZ positions in metres.

Full specification and C# packing order:
[vr_binary_v2.md](protocols/vr_binary_v2.md).

## Timestamps and clock domains

Timestamps use integer nanoseconds. Every typed sample identifies its clock
domain:

- `monotonic`: process-local monotonic time;
- `unix`: wall-clock Unix epoch;
- `device`: producer/device clock;
- `unspecified`: compatibility data with no trustworthy domain.

Do not subtract timestamps from different clock domains without an explicit
clock-synchronization mapping. Receive timestamps can measure local transport
latency; source timestamps preserve acquisition ordering.

Wire `uint32` sequences may wrap. Use modular comparison:

```text
delta = (candidate - current) & 0xffffffff
accept when 0 < delta < 0x80000000
```

Never compare wrapped wire sequences with a simple signed greater-than test.

## Compatibility and migration

| Existing behavior | v2 status |
|---|---|
| controller CSV | accepted by compatibility parser |
| hand text | accepted by compatibility parser |
| `HB,<base64>` hand payload | accepted |
| JPEG chunk header | byte-compatible and frozen |
| WebRTC signaling envelope | shape preserved |
| legacy record-control strings | retained until application migration |
| HDF5/LeRobot schema | frozen and tested |

New state and command channels are opt-in until the Unity application and
production composition have been validated. They do not silently replace legacy
ports.

## Unity implementation notes

- Write fields individually; do not marshal native C# structs with implicit
  padding.
- Use little-endian for VR/state binary values and network byte order only for
  the frozen JPEG chunk header and standard RTP fields.
- Use IEEE-754 32-bit floats.
- Preserve quaternion order XYZW and matrix row-major order.
- Treat source timestamps as unsigned 64-bit nanoseconds.
- Keep state and reliable commands on different DataChannels.
- Generate a stable unique `command_id` and retry only the identical payload.
- Do not queue old video or state messages in the Unity receiver.
- Validate protocol versions before interpreting payload bytes.
- Log sequence gaps, stale drops, ACK timeouts, and negotiated video codec.

## Failure handling

- malformed or unsupported packets are rejected without partial mutation;
- stale state does not replace the latest value;
- lost unreliable state is not retransmitted;
- lost command ACK may be retried safely with the same ID and payload;
- full reliable-command queues reject explicitly;
- video overload drops old frames and increments metrics;
- worker or signaling failures become health errors and trigger lifecycle
  shutdown.
