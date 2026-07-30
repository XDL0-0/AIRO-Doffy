# AIRO-Doffy Realtime State Channel v1

Status: opt-in protocol. It does not replace legacy VR UDP until deployment
validation and composition migration are complete.

Byte order: little-endian.

Numeric payloads: IEEE-754 `float32`.

## Envelope

Every DataChannel message or diagnostic UDP datagram contains one complete
envelope. The header is exactly 20 bytes:

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 2 | `uint16` | magic `0xAD20` |
| 2 | 1 | `uint8` | version, exactly `1` |
| 3 | 1 | `uint8` | message type |
| 4 | 4 | `uint32` | sequence |
| 8 | 8 | `uint64` | source timestamp in nanoseconds |
| 16 | 2 | `uint16` | payload byte length |
| 18 | 2 | `uint16` | flags |

Message types:

- `1`: complete `VRInputState`;
- `2`: `RobotState`.

Flags bits 0-1 encode the timestamp clock domain:

- `0`: monotonic;
- `1`: Unix;
- `2`: device;
- `3`: unspecified.

Bits 2-15 are reserved and must be zero. Receivers reject unknown versions,
types, flags, truncated packets, trailing bytes, invalid entities, non-finite
values, and metadata mismatches.

## VR payload

The payload is one complete
[VR binary v2 message](./vr_binary_v2.md). Its sequence and source timestamp
must exactly match the outer envelope. This preserves the already-tested
controller and OpenXR hand layouts while keeping the state envelope independent
from WebRTC or UDP.

## Robot payload

The payload begins with four bytes:

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 1 | `uint8` | joint count, `6` or `7` |
| 1 | 1 | `uint8` | options |
| 2 | 1 | `uint8` | reserved, zero |
| 3 | 1 | `uint8` | reserved, zero |

Options:

- bit 0: one gripper-width `float32` is present;
- bit 1: six wrench `float32` values are present;
- bits 2-7: reserved and zero.

The remaining values are tightly packed in this order:

```text
joint_count × float32 joints_rad
16 × float32 row-major TCP 4×4 transform
optional 1 × float32 gripper_width_m
optional 6 × float32 wrench [Fx, Fy, Fz, Tx, Ty, Tz]
```

## Ordering and delivery

The production WebRTC DataChannel is created as:

```text
label = "realtime_state"
ordered = false
maxRetransmits = 0
```

Sequence comparison uses unsigned 32-bit modular ordering. Equal, older, and
exactly-half-range ambiguous values are rejected independently for VR and robot
messages. The sender holds at most one pending packet per message type and
overwrites an unread older packet.

UDP port `5005` is available only as an opt-in diagnostic adapter. UDP does not
add fragmentation, acknowledgement, ordering, or retries; state payloads should
remain below the path MTU where possible.
