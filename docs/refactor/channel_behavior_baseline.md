# Phase 8 State and Command Channel Baseline

日期：2026-07-30

## 1. Current high-frequency state path

- Quest controller/hand state arrives through UDP `socket_0` as legacy text or
  `HB,` Base64 text.
- `UdpComms` receives UTF-8 into `deque(maxlen=128)`.
- manager VR loop polls at a nominal 100 Hz and drains pose with `read_all()`.
- manager stores controller or per-side hand dictionaries under one lock.
- packets do not have transport-level acknowledgement or retransmission.
- legacy controller without frame id cannot provide sender-side stale rejection.
- the old WebRTC DataChannel is not used for high-frequency state.
- optional robot state is not sent to Quest through a typed binary channel.

Phase 6 added transport-independent VR parsing and binary v2, but the root manager
has not switched to it.

## 2. Current control/command path

Record control arrives on UDP `socket_1` as exact strings:

| Text | Existing effect |
|---|---|
| `Start` | collecting true; export false; rollback false |
| `Stop` | collecting false; export true |
| `Undo` / `Rollback` / `DeleteLast` | collecting/export false; rollback true |

Resolution/zoom/fine-mode arrives on UDP `socket_2` or the old WebRTC
`"control"` DataChannel. It uses semicolon-separated `key,value` text. The
WebRTC path treats a numeric key as camera index; the legacy UDP path derives
camera index from the port-like key.

Visualizer rollback uses an in-process bounded multiprocessing queue containing
typed `RuntimeCommand(ROLLBACK_LAST_EPISODE)`. The main loop drains it and writes
the old manager rollback flag.

No network command path currently provides:

- command id validation;
- duplicate detection;
- idempotent replay;
- acknowledgement;
- timeout;
- failure response;
- ordered reliable separation from realtime state.

## 3. Existing typed command model

`core.events.RuntimeCommand` already defines:

- explicit command enum;
- sequence;
- source timestamp and clock domain;
- origin;
- unique command id;
- optional value only for `SET_VIDEO_PROFILE`.

Current enum values:

- start/stop recording;
- rollback latest episode;
- recalibrate tactile;
- reset teleop reference;
- pause/resume;
- set video profile;
- shutdown.

The model is immutable and validation is dependency-free. It is the Phase 8
command protocol boundary; transports must not convert it back into manager flags.

## 4. Compatibility boundary

Phase 8 adds new opt-in channels. It must not silently remove:

- `Start`/`Stop`/`Undo|Rollback|DeleteLast`;
- legacy zoom/fine-mode parsing;
- visualizer rollback queue;
- old `"control"` DataChannel;
- current dataset export/rollback semantics.

Adapters from legacy text to typed commands can be added later at the composition
boundary. New serializers and routers do not directly mutate dataset, tactile,
camera, robot, teleop, or runtime objects.

## 5. Required new policies

### Realtime state

- binary and versioned;
- unordered;
- `maxRetransmits=0`;
- latest-only bounded storage;
- sequence-aware duplicate/stale rejection;
- serializer independent from DataChannel/UDP;
- optional UDP diagnostic adapter;
- state send errors and bytes/drop metrics.

### Reliable commands

- separate label/channel;
- ordered;
- reliable (`maxRetransmits=None`, no lifetime limit);
- typed parse;
- command-id deduplication with bounded memory;
- same-id/same-payload idempotent replay;
- same-id/different-payload rejection;
- explicit acknowledgement/rejection/error;
- sender timeout;
- router composed from injected handlers.

## 6. Tests required before wiring

1. Binary state header golden bytes, version/type/length/flags validation.
2. VR and 6/7-DOF robot state round trips.
3. uint32 sequence wrap and stale rejection per message type.
4. sender latest-only overwrite under blocked channel.
5. UDP diagnostic target, bytes, lifecycle and error metrics.
6. aiortc state channel label, `ordered=False`, `maxRetransmits=0`.
7. command serialization and unknown-field rejection.
8. reliable channel policy (`ordered=True`, no retransmission limit).
9. ACK success/reject/error/timeout.
10. duplicate command id with same and conflicting payload.
11. bounded dedupe eviction.
12. router handler selection and exception reporting.
13. package imports without aiortc or hardware SDKs.

## 7. Hardware and Unity validation

- Unity binary state byte layout and float/joint conventions.
- Unity-created or Python-created DataChannel negotiation.
- unordered/no-retransmit delivery, reordering and loss under Wi-Fi impairment.
- ordered reliable command delivery and ACK round trip.
- reconnect and sender restart sequence policy.
- command replay after ACK loss without repeated side effects.
- safe handling of shutdown, recording and rollback commands.
- state/command channel close without peer or event-loop leaks.
