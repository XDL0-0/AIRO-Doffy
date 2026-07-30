# AIRO-Doffy VR Binary State Protocol v2

Status: migration protocol; legacy controller CSV, hand text, and `HB,<base64>`
remain accepted.

Byte order: little-endian.

Numeric payload: IEEE-754 `float32`.

## Header

Fixed size: 24 bytes.

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | bytes | ASCII magic `AVR2` |
| 4 | 1 | `uint8` | version, exactly `2` |
| 5 | 1 | `uint8` | mode: `1=controllers`, `2=hands` |
| 6 | 1 | `uint8` | entity count |
| 7 | 1 | `uint8` | flags, currently `0` |
| 8 | 4 | `uint32` | sequence |
| 12 | 8 | `uint64` | device source timestamp in nanoseconds |
| 20 | 4 | `uint32` | payload byte length |

Receivers reject unknown version/flags, length mismatch, invalid entity counts,
unknown sides/buttons, non-finite values, invalid quaternion/trigger values, trailing
bytes, and truncated payloads.

## Controller mode

Exactly two entities are required. Each entity is 48 bytes.

| Entity offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 1 | `uint8` | side: `0=left`, `1=right` |
| 1 | 1 | `uint8` | buttons bitset |
| 2 | 2 | `uint16` | reserved, must be `0` |
| 4 | 12 | `3 × float32` | position XYZ, metres |
| 16 | 16 | `4 × float32` | orientation XYZW |
| 32 | 8 | `2 × float32` | joystick XY |
| 40 | 4 | `float32` | index trigger `[0,1]` |
| 44 | 4 | `float32` | grip trigger `[0,1]` |

Button bits:

- bit 0: primary / A-X
- bit 1: secondary / B-Y
- bit 2: joystick press
- bits 3–7: reserved and must be zero

## Hand mode

One or two entities are allowed. Each begins with:

| Entity offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 1 | `uint8` | side: `0=left`, `1=right` |
| 1 | 1 | `uint8` | wrist present: `0` or `1` |
| 2 | 2 | `uint16` | joint count, exactly `26` |

If wrist is present, append 28 bytes:

```text
position XYZ     3 × float32 metres
orientation XYZW 4 × float32
```

Then append exactly 26 joints in OpenXR order:

```text
joint position XYZ = 3 × float32 metres = 12 bytes
```

Entity size is 316 bytes without wrist or 344 bytes with wrist.

## Sequence and stale rejection

The sequence is an unsigned 32-bit counter and may wrap. A receiver accepts a
candidate only when:

```text
delta = (candidate - current) & 0xffffffff
0 < delta < 0x80000000
```

Equal values are duplicates. A delta of exactly `0x80000000` is ambiguous and
must be rejected. Stale rejection is applied independently per controller state
stream and per hand side during legacy migration.

## Unity/C# packing notes

Unity must write every field explicitly without native-struct padding.
`BinaryWriter` writes primitive integers and floats little-endian on supported
Unity/.NET targets; if another API is used, byte order must be set explicitly.

Controller example outline:

```csharp
writer.Write(new byte[] { 0x41, 0x56, 0x52, 0x32 }); // AVR2
writer.Write((byte)2);      // version
writer.Write((byte)1);      // controllers
writer.Write((byte)2);      // two entities
writer.Write((byte)0);      // flags
writer.Write(sequence);     // UInt32
writer.Write(timestampNs);  // UInt64
writer.Write((uint)96);     // 2 × 48-byte entities
// Write each field in table order. Do not marshal a C# struct.
```

## Transport boundary

This document specifies message bytes only. It does not select UDP, WebRTC
DataChannel, QUIC, file, or replay transport. A transport must deliver one complete
protocol message to the decoder; fragmentation/reassembly belongs to the transport.

Binary v2 does not change legacy UDP ports, WebRTC signaling, reliable commands,
record control, or video channels.
